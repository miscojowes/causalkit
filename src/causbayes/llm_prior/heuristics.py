"""
LLM-guided heuristic search for causal structure.

Following Wu et al. (2025), LLMs should NOT directly determine
causal edges. Instead, they can accelerate the search process
by suggesting promising graph neighborhoods to explore.

This module implements LLM-guided heuristic search that:
1. Uses LLM to suggest parent sets worth exploring
2. Evaluates these with the score function
3. Combines with gradient-based refinement
"""

import numpy as np
import torch
from typing import Optional, Callable


class LLMHeuristicSearch:
    """LLM-guided heuristic search for causal graphs.

    Uses LLM to propose promising graph structures, then
    evaluates them using the data likelihood score.

    Parameters
    ----------
    llm_extractor : LLMPriorExtractor
        The LLM prior extractor instance
    temperature : float
        Exploration temperature. Default: 1.0
    n_proposals : int
        Number of graph proposals per iteration. Default: 10
    """

    def __init__(
        self,
        llm_extractor: "LLMPriorExtractor",
        temperature: float = 1.0,
        n_proposals: int = 10,
    ):
        self.llm = llm_extractor
        self.temperature = temperature
        self.n_proposals = n_proposals

    def propose_parent_sets(
        self,
        variables: list,
        domain_description: str,
        current_graph: Optional[np.ndarray] = None,
    ) -> dict:
        """Propose promising parent sets using LLM.

        Args:
            variables: Variable names
            domain_description: Domain context
            current_graph: Current best graph (optional)

        Returns:
            Dict mapping variable index -> list of proposed parent sets
        """
        d = len(variables)
        var_list = "\n".join(f"- {i}: {v}" for i, v in enumerate(variables))

        current_info = ""
        if current_graph is not None:
            edges = []
            for i in range(d):
                for j in range(d):
                    if current_graph[i, j] > 0.5:
                        edges.append(f"{variables[i]} -> {variables[j]}")
            current_info = "\n".join(edges)

        prompt = f"""You are guiding a causal structure search algorithm.

Variables:
{var_list}

Domain:
{domain_description[:300]}

{'Current best graph edges:' + current_info if current_info else 'No current graph yet.'}

For each variable, suggest 2-3 plausible parent sets (combinations of variables 
that could directly cause it). Focus on biologically/physically plausible 
combinations.

Return as JSON:
{{
  "0": [["var1", "var2"], ["var3"]],
  "1": [["var2"]]
}}

Use index numbers, not variable names."""

        response = self.llm._query_llm(prompt)
        return self._parse_proposals(response, d)

    def _parse_proposals(self, response: str, d: int) -> dict:
        """Parse LLM response for parent set proposals."""
        json_str = response.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1]
            if "```" in json_str:
                json_str = json_str.split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1]
            if "```" in json_str:
                json_str = json_str.split("```")[0]

        import json
        import re
        json_str = json_str.strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            match = re.search(r'\{.*?\}', json_str, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except (json.JSONDecodeError, ValueError):
                    return {}
            return {}

    def search(
        self,
        X: np.ndarray,
        variables: list,
        domain_description: str,
        score_fn: Callable,
        max_iterations: int = 5,
    ) -> np.ndarray:
        """Run LLM-guided heuristic search.

        Args:
            X: Data matrix
            variables: Variable names
            domain_description: Domain description
            score_fn: Score function f(graph_adjacency) -> score (lower is better)
            max_iterations: Number of LLM proposal rounds

        Returns:
            Best adjacency matrix found
        """
        d = len(variables)
        best_graph = np.zeros((d, d))
        best_score = float("inf")

        for iteration in range(max_iterations):
            # Get LLM proposals
            proposals = self.propose_parent_sets(
                variables, domain_description,
                current_graph=best_graph if iteration > 0 else None,
            )

            # Evaluate proposals
            for var_idx, parent_sets in proposals.items():
                var_idx = int(var_idx)
                for parents in parent_sets:
                    # Build candidate graph
                    candidate = best_graph.copy()
                    # Reset parents for this variable
                    candidate[:, var_idx] = 0.0
                    for parent_name in parents:
                        try:
                            parent_idx = variables.index(parent_name)
                            candidate[parent_idx, var_idx] = 1.0
                        except ValueError:
                            pass

                    # Score
                    try:
                        score = score_fn(candidate)
                        if score < best_score:
                            best_score = score
                            best_graph = candidate
                    except Exception:
                        continue

        return best_graph
