"""
LLM-based domain knowledge extraction for causal priors.

Uses LLMs to extract causal relationships from domain text,
papers, and expert descriptions — but critically, these are used
as *soft priors* (inform but don't decide), not hard constraints.

Following guidance from:
Wu et al. (2025) "LLM Cannot Discover Causality" - LLMs should
be restricted to non-decisional auxiliary support.
"""

import json
import re
from typing import Optional

import numpy as np


class LLMPriorExtractor:
    """Extract causal priors from domain text using an LLM.

    The LLM suggests potential causal relationships based on
    domain knowledge. These are treated as *soft priors* that
    bias the structure learning but can be overridden by data.

    Parameters
    ----------
    api_key : str
        API key for the LLM provider
    model : str
        Model to use. Default: "opencode-go/deepseek-v4-flash"
    api_base : str
        Base URL for the API. Default: "https://api.opencode.ai/v1"
    temperature : float
        LLM temperature for sampling. Default: 0.1 (low = conservative)
    max_tokens : int
        Max tokens per response. Default: 2000
    """

    def __init__(
        self,
        api_key: str,
        model: str = "opencode-go/deepseek-v4-flash",
        api_base: str = "https://api.opencode.ai/v1",
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ):
        self.api_key = api_key
        self.model = model
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        """Lazy import of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base,
                )
            except ImportError:
                raise ImportError(
                    "openai package required for LLM priors. "
                    "Install with: pip install causbayes[llm]"
                )
        return self._client

    def extract_edge_priors(
        self,
        variables: list,
        domain_description: str,
        confidence: str = "medium",
    ) -> np.ndarray:
        """Extract edge priors from domain knowledge using LLM.

        Args:
            variables: List of variable names
            domain_description: Free text description of the domain
            confidence: 'high' (only very confident edges),
                       'medium' (reasonable suggestions),
                       'low' (any plausible connection)

        Returns:
            Prior probability matrix of shape (d, d)
        """
        d = len(variables)
        var_list_str = "\n".join(f"- {v}" for v in variables)

        confidence_instructions = {
            "high": (
                "Only include causal relationships that you are VERY confident about "
                "based on established scientific knowledge. Be conservative - it's worse "
                "to suggest a false relationship than to miss a true one."
            ),
            "medium": (
                "Include causal relationships that are reasonably well-established "
                "or strongly suggested by the domain description. Include the direction "
                "and your confidence level."
            ),
            "low": (
                "Include any plausible causal relationships suggested by the domain description, "
                "even if uncertain. Flag uncertain ones with low confidence."
            ),
        }

        prompt = f"""You are an expert in {domain_description[:100]}...

## Task: Extract Causal Knowledge from Domain Text

Given the following domain description and list of variables, identify potential causal relationships BETWEEN these variables. This will be used as a SOFT PRIOR for causal structure learning — the data can override your suggestions.

## Variables
{var_list_str}

## Domain Description
{domain_description}

## Instructions
{confidence_instructions.get(confidence, confidence_instructions["medium"])}

Return the result as a JSON array of objects with:
- "cause": name of the cause variable
- "effect": name of the effect variable
- "confidence": one of "high", "medium", "low"
- "rationale": one-sentence explanation

Only include relationships where BOTH cause and effect are in the variable list.
Only include DIRECT causal relationships, not indirect ones.

## Output Format
```json
[
  {{
    "cause": "VariableA",
    "effect": "VariableB",
    "confidence": "high",
    "rationale": "VariableA is a known regulator of VariableB"
  }}
]
```"""

        response = self._query_llm(prompt)

        # Parse response
        edges = self._parse_edges(response, variables)

        # Build prior matrix
        from causbayes.bayesian.priors import prior_from_associations

        associations = {}
        for cause, effect, conf_level in edges:
            associations[(cause, effect)] = conf_level

        return prior_from_associations(d, variables, associations)

    def extract_variables(
        self,
        domain_description: str,
        suggested_vars: Optional[list] = None,
    ) -> list:
        """Extract relevant variables from domain description.

        Args:
            domain_description: Free text domain description
            suggested_vars: Optional list of already-known variables

        Returns:
            List of variable names with descriptions
        """
        prompt = f"""Given this domain description, identify the key variables/entities that could be part of a causal system.

Domain: {domain_description}

{'Already known variables: ' + ', '.join(suggested_vars) if suggested_vars else ''}

Return as JSON array: [{{"name": "...", "description": "...", "type": "continuous|categorical|binary"}}]"""

        response = self._query_llm(prompt)
        return self._parse_variables(response)

    def suggest_experiments(
        self,
        variables: list,
        edge_probs: np.ndarray,
        edge_stds: np.ndarray,
        domain_description: str,
    ) -> list:
        """Suggest experiments to resolve uncertain edges.

        Args:
            variables: Variable names
            edge_probs: Learned edge probability matrix
            edge_stds: Edge uncertainty matrix
            domain_description: Domain context

        Returns:
            List of suggested experiments
        """
        # Find most uncertain edges
        d = len(variables)
        uncertain_edges = []
        for i in range(d):
            for j in range(d):
                if i != j:
                    prob = edge_probs[i, j]
                    std = edge_stds[i, j]
                    # High uncertainty: near 0.5 or high std
                    uncertainty = -(prob * np.log(prob + 1e-8) + (1 - prob) * np.log(1 - prob + 1e-8))
                    if uncertainty > 0.5:  # Configurable threshold
                        uncertain_edges.append(
                            (i, j, variables[i], variables[j], prob, std, uncertainty)
                        )

        if not uncertain_edges:
            return []

        # Sort by uncertainty
        uncertain_edges.sort(key=lambda x: x[-1], reverse=True)
        top_uncertain = uncertain_edges[:5]

        edge_descriptions = "\n".join(
            f"- {cause} -> {effect}: P={prob:.2f} ± {std:.2f} (entropy={ent:.2f})"
            for _, _, cause, effect, prob, std, ent in top_uncertain
        )

        prompt = f"""Given this causal discovery result with uncertainty:

Domain: {domain_description[:200]}
Variables: {', '.join(variables)}

Most uncertain edges (P ≈ 0.5 or high variance):
{edge_descriptions}

Suggest 3 specific experiments (e.g., perturbations, interventions) that would best resolve 
the uncertainty about these causal relationships. For each, explain which edge it would 
disambiguate and why.

Return as JSON array: [{{"experiment": "...", "target_edge": "...", "rationale": "..."}}]"""

        response = self._query_llm(prompt)
        return self._parse_experiments(response)

    def _query_llm(self, prompt: str) -> str:
        """Query the LLM API."""
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a causal discovery expert assistant. "
                    "Provide responses as valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    def _parse_edges(self, response: str, variables: list) -> list:
        """Parse LLM response to extract edge list.

        Returns list of (cause, effect, confidence) tuples.
        """
        # Extract JSON from response (handle markdown wrapping)
        json_str = response.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1]
            if "```" in json_str:
                json_str = json_str.split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1]
            if "```" in json_str:
                json_str = json_str.split("```")[0]

        json_str = json_str.strip()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Try to extract JSON array with regex
            match = re.search(r'\[.*?\]', json_str, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except (json.JSONDecodeError, ValueError):
                    return []
            else:
                return []

        var_set = set(variables)
        edges = []
        for item in data:
            cause = item.get("cause", "")
            effect = item.get("effect", "")
            conf = item.get("confidence", "low")
            if cause in var_set and effect in var_set:
                edges.append((cause, effect, conf))

        return edges

    def _parse_variables(self, response: str) -> list:
        """Parse LLM response to extract variable list."""
        json_str = response.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1]
            if "```" in json_str:
                json_str = json_str.split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1]
            if "```" in json_str:
                json_str = json_str.split("```")[0]

        json_str = json_str.strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            match = re.search(r'\[.*?\]', json_str, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except (json.JSONDecodeError, ValueError):
                    return []
            return []

    def _parse_experiments(self, response: str) -> list:
        """Parse LLM response for experiment suggestions."""
        json_str = response.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1]
            if "```" in json_str:
                json_str = json_str.split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1]
            if "```" in json_str:
                json_str = json_str.split("```")[0]

        json_str = json_str.strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            match = re.search(r'\[.*?\]', json_str, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except (json.JSONDecodeError, ValueError):
                    return []
            return []
