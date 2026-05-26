"""
LLM Posterior Correction: Uses an LLM to resolve ambiguous edge orientations.

Based on the SOTA framework from:
    "Large Language Models for Causal Discovery" (IJCAI 2025 survey)
    
Three strategies:
    1. Prior Injection (soft L2 priors from LLM domain knowledge)
    2. Posterior Correction (LLM reviews uncertain edges from BootstrapDAG)
    3. Hybrid (both prior + posterior refinement)

Following Wu et al. (2025):
    - LLMs should NOT directly decide causal edges
    - LLMs should provide non-decisional auxiliary support
    - We use LLM to SUGGEST orientations, not DICTATE them
"""
import numpy as np
from typing import Optional, Literal


class LLMCausalAdvisor:
    """LLM-powered causal discovery advisor.
    
    Operates as a critical component of the BootstrapDAG pipeline:
    - Phase 1 (Prior): LLM suggests causal relationships from domain text
    - Phase 2 (Posterior): LLM reviews uncertain edges and helps orient them
    
    Strategy follows the SOTA taxonomy:
    - Prior Injection = LLM as input to SCD (before optimization)
    - Posterior Correction = LLM as refinement (after SCD)
    - Hybrid = both, with posterior only on edges still uncertain after prior
    
    Parameters
    ----------
    api_key : str
        API key for the LLM provider
    model : str
        Model to use
    api_base : str
        Base URL for API
    temperature : float
        LLM temperature (low = conservative for causal claims)
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "opencode-go/deepseek-v4-flash",
        api_base: str = "https://api.opencode.ai/v1",
        temperature: float = 0.1,
    ):
        self.api_key = api_key
        self.model = model
        self.api_base = api_base
        self.temperature = temperature
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base,
                )
            except ImportError:
                raise ImportError("openai package required")
        return self._client
    
    def _query_llm(self, prompt: str) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a causal discovery expert. "
                 "Provide responses as valid JSON only. "
                 "Be conservative — only suggest relationships you are confident about."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content
    
    def extract_prior_matrix(
        self,
        variables: list,
        domain_description: str,
        confidence: str = "medium",
    ) -> np.ndarray:
        """Extract prior edge probability matrix from domain text.
        
        Returns a (d, d) matrix where prior[i,j] ∈ [0,1] indicates
        the LLM's belief that variable i causes variable j.
        
        This is a SOFT prior — the data can override it.
        """
        # This is adapted from the existing LLMPriorExtractor
        from causbayes.llm_prior import LLMPriorExtractor
        extractor = LLMPriorExtractor(
            api_key=self.api_key,
            model=self.model,
            api_base=self.api_base,
        )
        return extractor.extract_edge_priors(
            variables, domain_description, confidence
        )
    
    def correct_uncertain_edges(
        self,
        variables: list,
        edge_probs: np.ndarray,
        edge_stds: np.ndarray,
        domain_description: str,
        uncertainty_threshold: float = 0.3,
    ) -> tuple:
        """Posterior Correction: LLM reviews uncertain edges.
        
        Following ILS-CSL (Ban et al., 2023a):
        - Identify edges with high uncertainty (P ≈ 0.5 or high variance)
        - Present them to LLM grouped by context
        - LLM suggests orientation based on domain knowledge
        
        Returns:
            (corrected_matrix, suggestions) where:
            - corrected_matrix: (d, d) where 1=directed, 0=skip, -1=opposite direction
            - suggestions: list of (i, j, direction, confidence, rationale)
        """
        d = len(variables)
        entropy = -(edge_probs * np.log(edge_probs + 1e-8) 
                    + (1 - edge_probs) * np.log(1 - edge_probs + 1e-8))
        
        # Find uncertain edges (entropy > threshold)
        uncertain = []
        for i in range(d):
            for j in range(d):
                if i != j and entropy[i, j] > uncertainty_threshold:
                    uncertain.append((
                        i, j, variables[i], variables[j],
                        edge_probs[i, j], edge_stds[i, j], entropy[i, j]
                    ))
        
        if not uncertain:
            return np.zeros((d, d)), []
        
        # Sort by uncertainty
        uncertain.sort(key=lambda x: x[-1], reverse=True)
        
        # Group into batches of 8 for LLM (avoid context overflow)
        batch_size = 8
        suggestions = []
        
        for batch_start in range(0, min(len(uncertain), 16), batch_size):
            batch = uncertain[batch_start:batch_start + batch_size]
            
            edge_descriptions = []
            for i, j, vi, vj, prob, std, ent in batch:
                edge_descriptions.append(
                    f"- {vi} → {vj}: P(cause)={prob:.2f}±{std:.2f} (entropy={ent:.2f})"
                )
            
            var_context = "\n".join(f"- {v}" for v in variables)
            edge_text = "\n".join(edge_descriptions)
            
            prompt = f"""I need help resolving uncertain causal relationships.

Domain: {domain_description[:300]}
Variables:
{var_context}

A causal discovery algorithm found these relationships but is uncertain about the direction or existence. For each edge, tell me:

1. Is there a DIRECT causal relationship between these variables in this domain?
2. If yes, what is the most likely DIRECTION?
3. How confident are you? (high/medium/low)

Edge descriptions:
{edge_text}

Return as JSON array:
[
  {{
    "cause": "VariableName",
    "effect": "VariableName",
    "direction_confidence": "high|medium|low",
    "rationale": "One-sentence explanation based on domain knowledge"
  }}
]

Be conservative — only suggest edges you're confident about. If uncertain, OMIT the edge."""
            
            response = self._query_llm(prompt)
            parsed = self._parse_edges_from_json(response, variables)
            suggestions.extend(parsed)
        
        # Build correction matrix
        correction = np.zeros((d, d), dtype=float)
        for cause, effect, conf in suggestions:
            if cause in variables and effect in variables:
                i = variables.index(cause)
                j = variables.index(effect)
                conf_weight = {"high": 0.8, "medium": 0.6, "low": 0.4}.get(conf, 0.4)
                correction[i, j] = conf_weight
        
        return correction, suggestions
    
    def hybrid_refine(
        self,
        variables: list,
        X: np.ndarray,
        domain_description: str,
        prior_strength: float = 0.3,
        n_bootstraps: int = 30,
    ) -> tuple:
        """Full hybrid pipeline: Prior + Bootstrap + Posterior correction.
        
        1. Extract LLM prior from domain text
        2. Run BootstrapDAG with prior
        3. Apply posterior correction on uncertain edges
        4. Return final edge probabilities
        
        Returns:
            (edge_probs, edge_stds, prior_corrections)
        """
        from causbayes import BootstrapDAG
        
        # Phase 1: Prior
        print("  Phase 1: Extracting LLM prior...")
        prior = self.extract_prior_matrix(variables, domain_description)
        
        # Phase 2: Bootstrap with prior
        print(f"  Phase 2: BootstrapDAG with prior (λ={prior_strength})...")
        model = BootstrapDAG(
            n_bootstraps=n_bootstraps,
            lambda_1=0.01,
            max_iter=5,
            w_threshold=0.05,
            prior_matrix=prior,
            lambda_prior=prior_strength,
            calibrate=True,
            verbose=False,
        )
        model.fit(X)
        
        # Phase 3: Posterior correction
        print("  Phase 3: LLM posterior correction...")
        correction, suggestions = self.correct_uncertain_edges(
            variables, model.edge_probs, model.edge_stds,
            domain_description,
        )
        
        # Apply corrections: boost/suppress edges based on LLM feedback
        probs = model.edge_probs.copy()
        stds = model.edge_stds.copy()
        
        for (i, j), weight in np.ndenumerate(correction):
            if weight > 0:
                # LLM suggests edge exists: boost probability
                probs[i, j] = max(probs[i, j], 0.5 + weight * 0.4)
                stds[i, j] = max(stds[i, j], 0.05)
        
        return probs, stds, correction
    
    def _parse_edges_from_json(self, response: str, variables: list) -> list:
        """Parse LLM JSON response into (cause, effect, confidence) tuples."""
        import json
        import re
        
        json_str = response.strip()
        for delimiter in ["```json", "```", "```JSON"]:
            if delimiter in json_str:
                parts = json_str.split(delimiter)
                for i in range(1, len(parts)):
                    candidate = parts[i].strip()
                    if candidate.endswith("```"):
                        candidate = candidate[:-3]
                    try:
                        data = json.loads(candidate)
                        break
                    except json.JSONDecodeError:
                        pass
                else:
                    continue
                break
        else:
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                match = re.search(r'\[.*?\]', json_str, re.DOTALL)
                if match:
                    try:
                        data = json.loads(match.group())
                    except (json.JSONDecodeError, ValueError):
                        return []
                else:
                    return []
        
        if isinstance(data, dict):
            data = [data]
        
        var_set = set(variables)
        edges = []
        for item in data:
            cause = item.get("cause", "")
            effect = item.get("effect", "")
            conf = item.get("direction_confidence", item.get("confidence", "low"))
            if cause in var_set and effect in var_set:
                edges.append((cause, effect, conf))
        
        return edges
