"""
causalkit.prior — LLM-based prior extraction from domain text
==============================================================
Convert natural language domain descriptions into structured prior matrices,
ready to be used by CausalDiscoverer.
"""

import re
import numpy as np


def extract_prior_from_text(
    domain_text: str,
    feature_names: list[str],
    default_confidence: float = 0.7,
    random_state: int = 42,
) -> np.ndarray:
    """Extract a causal prior matrix from domain description text.

    Uses simple NLP heuristics to extract causal relationships
    from text. Looks for patterns like:
        - "X causes Y", "X leads to Y", "X drives Y"
        - "X influences Y", "X affects Y"
        - "Y depends on X"
        - "X → Y", "X -> Y"

    Parameters
    ----------
    domain_text : str
        Natural language description of known causal relationships.
    feature_names : list of str
        Names of the variables in the dataset.
    default_confidence : float, default=0.7
        Default prior probability for explicitly mentioned edges.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    prior_matrix : ndarray (d, d)
        Prior matrix where prior[i,j] = confidence that i -> j.
    """
    d = len(feature_names)
    prior = np.full((d, d), 0.5)
    np.fill_diagonal(prior, 0.0)

    names_lower = [n.lower() for n in feature_names]

    # Pattern 1: "X causes Y", "X leads to Y", "X drives Y", etc.
    cause_patterns = [
        r"(\w+(?:\s+\w+)*)\s+(?:causes|leads?\s+to|drives|produces|generates|increases|raises|triggers|induces|promotes)\s+(\w+(?:\s+\w+)*)",
        r"(\w+(?:\s+\w+)*)\s+(?:influences|affects|determines|controls|regulates|modulates|shapes|impacts)\s+(\w+(?:\s+\w+)*)",
        r"(\w+(?:\s+\w+)*)\s+(?:reduces|decreases|lowers|suppresses|inhibits|blocks|prevents)\s+(\w+(?:\s+\w+)*)",
        r"(\w+(?:\s+\w+)*)\s+(?:predicts|forecasts|explains)\s+(\w+(?:\s+\w+)*)",
        r"(\w+(?:\s+\w+)*)\s+→\s+(\w+(?:\s+\w+)*)",
        r"(\w+(?:\s+\w+)*)\s+->\s+(\w+(?:\s+\w+)*)",
        r"(\w+(?:\s+\w+)*)\s+is\s+(?:a\s+)?cause\s+of\s+(\w+(?:\s+\w+)*)",
    ]

    # Pattern 2: "Y depends on X", "Y is determined by X"
    dep_patterns = [
        r"(\w+(?:\s+\w+)*)\s+depends?\s+(?:on|upon)\s+(\w+(?:\s+\w+)*)",
        r"(\w+(?:\s+\w+)*)\s+is\s+(?:determined|driven|influenced|affected|controlled|regulated)\s+by\s+(\w+(?:\s+\w+)*)",
        r"(\w+(?:\s+\w+)*)\s+(?:is|are)\s+a\s+function\s+of\s+(\w+(?:\s+\w+)*)",
    ]

    edges_found = 0

    # Collect all causal statements
    for pattern in cause_patterns:
        for match in re.finditer(pattern, domain_text, re.IGNORECASE):
            cause_str = match.group(1).strip().lower()
            effect_str = match.group(2).strip().lower()
            cause_idx = _match_var(cause_str, feature_names, names_lower)
            effect_idx = _match_var(effect_str, feature_names, names_lower)
            if cause_idx is not None and effect_idx is not None and cause_idx != effect_idx:
                prior[cause_idx, effect_idx] = default_confidence
                edges_found += 1

    for pattern in dep_patterns:
        for match in re.finditer(pattern, domain_text, re.IGNORECASE):
            dep_str = match.group(1).strip().lower()  # the dependent variable
            cause_str = match.group(2).strip().lower()  # the variable it depends on
            cause_idx = _match_var(cause_str, feature_names, names_lower)
            effect_idx = _match_var(dep_str, feature_names, names_lower)
            if cause_idx is not None and effect_idx is not None and cause_idx != effect_idx:
                prior[cause_idx, effect_idx] = default_confidence
                edges_found += 1

    # Pattern 3: Listed edges like "X → Y, Z → W"
    list_pattern = r"(\w+(?:\s+\w+)*)\s*[→,]\s*(\w+(?:\s+\w+)*)"
    for match in re.finditer(list_pattern, domain_text, re.IGNORECASE):
        cause_str = match.group(1).strip().lower()
        effect_str = match.group(2).strip().lower()
        # Avoid matching simple lists (like "X, Y, Z")
        if cause_str in names_lower and effect_str in names_lower:
            cause_idx = names_lower.index(cause_str)
            effect_idx = names_lower.index(effect_str)
            if cause_idx != effect_idx and prior[cause_idx, effect_idx] == 0.5:
                prior[cause_idx, effect_idx] = default_confidence
                edges_found += 1

    return prior


def _match_var(name_str, feature_names, names_lower):
    """Match a name string to a feature name (exact or substring)."""
    # Exact match
    if name_str in names_lower:
        return names_lower.index(name_str)
    # Substring match
    for i, n in enumerate(names_lower):
        if name_str in n or n in name_str:
            return i
    # Word match: check if name_str is a substring of a feature name
    for i, n in enumerate(names_lower):
        if any(word in n.split() for word in name_str.split() if len(word) > 2):
            return i
    return None


def edge_list_to_prior(edge_list, feature_names, confidence=0.9):
    """Convert a list of (cause, effect) tuples to a prior matrix.

    Parameters
    ----------
    edge_list : list of (str, str)
        List of known causal edges.
    feature_names : list of str
        Variable names.
    confidence : float, default=0.9
        Prior confidence for listed edges.

    Returns
    -------
    prior_matrix : ndarray (d, d)
    """
    d = len(feature_names)
    prior = np.full((d, d), 0.5)
    np.fill_diagonal(prior, 0.0)

    names_lower = [n.lower() for n in feature_names]
    for cause, effect in edge_list:
        ci = _match_var(cause.lower(), feature_names, names_lower)
        ei = _match_var(effect.lower(), feature_names, names_lower)
        if ci is not None and ei is not None and ci != ei:
            prior[ci, ei] = confidence

    return prior
