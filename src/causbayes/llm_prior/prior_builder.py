"""
Build informed prior matrices from LLM output for structure learning.
"""

from typing import Optional
import numpy as np


def build_prior_from_llm_response(
    llm_edges: list,
    variables: list,
    base_prob: float = 0.5,
    confidence_map: Optional[dict] = None,
) -> np.ndarray:
    """Build a prior probability matrix from parsed LLM edge data.

    Args:
        llm_edges: List of (cause_name, effect_name, confidence_str) tuples
        variables: Ordered list of variable names
        base_prob: Default prior probability for unknown edges
        confidence_map: Dict mapping confidence strings to probabilities.
            Default: {"high": 0.9, "medium": 0.7, "low": 0.55}

    Returns:
        Prior probability matrix of shape (len(variables), len(variables))
    """
    d = len(variables)
    name_to_idx = {name: i for i, name in enumerate(variables)}
    prior = np.full((d, d), base_prob)
    np.fill_diagonal(prior, 0.0)

    conf_map = confidence_map or {"high": 0.9, "medium": 0.7, "low": 0.55}

    for cause, effect, confidence in llm_edges:
        if cause in name_to_idx and effect in name_to_idx:
            i, j = name_to_idx[cause], name_to_idx[effect]
            if isinstance(confidence, (int, float)):
                prior[i, j] = float(confidence)
            elif isinstance(confidence, str):
                prior[i, j] = conf_map.get(confidence.lower(), base_prob)

    return prior


def build_prior_from_association_matrix(
    association_matrix: np.ndarray,
    base_prob: float = 0.5,
    scale: float = 1.0,
) -> np.ndarray:
    """Build a prior from an association/correlation matrix.

    Converts correlation values to edge probabilities using
    a sigmoid-like mapping.

    Args:
        association_matrix: Matrix of pairwise associations (d, d)
        base_prob: Base probability for weak associations
        scale: Scale factor for mapping associations to probabilities

    Returns:
        Prior probability matrix
    """
    d = association_matrix.shape[0]
    abs_assoc = np.abs(association_matrix)

    # Normalize to [0, 1]
    max_val = abs_assoc.max() if abs_assoc.max() > 0 else 1.0
    normalized = abs_assoc / max_val

    # Scale and sigmoid
    scaled = scale * (normalized - 0.5)
    prior = 1 / (1 + np.exp(-scaled))

    # Ensure diagonal is zero
    np.fill_diagonal(prior, 0.0)

    return prior


def fuse_priors(
    priors: list,
    weights: Optional[list] = None,
) -> np.ndarray:
    """Fuse multiple prior sources into a single prior matrix.

    Args:
        priors: List of prior probability matrices
        weights: Optional weights for each prior (default: equal)

    Returns:
        Fused prior matrix
    """
    if not priors:
        raise ValueError("At least one prior matrix required")

    n_priors = len(priors)
    if weights is None:
        weights = [1.0 / n_priors] * n_priors

    fused = sum(w * p for w, p in zip(weights, priors))
    d = fused.shape[0]
    np.fill_diagonal(fused, 0.0)
    return np.clip(fused, 0.0, 1.0)
