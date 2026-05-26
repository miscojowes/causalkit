"""
causalkit.adaptive_trust — Per-edge adaptive prior trust
=========================================================
Simplified version of PRCD-MAP's per-edge trust mechanism.

PRCD-MAP (arXiv:2605.01669) learns per-edge trust τ_ij via empirical Bayes,
then uses τ to weight the prior influence for each edge independently.

Our approach:
    1. Bootstrap with uniform λ → estimate edge strengths + std
    2. Compute τ_ij = agreement between prior and bootstrap strength
    3. λ_ij = λ_base * (τ_ij + 0.5)
       → range: [0.5 * λ_base, 1.5 * λ_base]
       → Edges where prior matches data: λ boosted (more prior influence)
       → Edges where prior contradicts data: λ attenuated (let data speak)
"""

import numpy as np


def compute_per_edge_lambda(
    prior_matrix: np.ndarray,
    edge_strength: np.ndarray,
    edge_std: np.ndarray,
    lambda_base: float = 0.5,
    min_lambda_ratio: float = 0.3,
    max_lambda_ratio: float = 1.8,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute per-edge λ_ij based on prior-data agreement.

    Parameters
    ----------
    prior_matrix : ndarray (d, d)
        Prior edge probabilities ∈ [0, 1].
        prior_matrix[i,j] = 0.9 means "strongly expect edge i→j".
        prior_matrix[i,j] = 0.1 means "strongly expect NO edge i→j".
    edge_strength : ndarray (d, d)
        Mean bootstrapped |W| values.
    edge_std : ndarray (d, d)
        Bootstrap standard deviation of |W|.
    lambda_base : float
        Base prior strength.
    min_lambda_ratio : float, default=0.3
        Minimum ratio of λ_base to use (attenuate wrong priors).
    max_lambda_ratio : float, default=1.8
        Maximum ratio of λ_base to use (boost correct priors).
    eps : float, default=1e-8
        Small constant to avoid division by zero.

    Returns
    -------
    lambda_per_edge : ndarray (d, d)
        Per-edge λ values.

    Notes
    -----
    Agreement is computed as:
        For each edge, we compute a trust score τ ∈ [0, 1]
        where τ ≈ 1 means prior matches data, τ ≈ 0 means they disagree.

        The trust score considers:
        - Strength agreement: |prior - normalized_strength|
        - Sign agreement: does prior say "edge" AND data agrees?

        λ_ij = λ_base * (min_lambda_ratio + τ_ij * (max_lambda_ratio - min_lambda_ratio))
    """
    d = prior_matrix.shape[0]
    lambda_per_edge = np.full((d, d), lambda_base, dtype=float)
    np.fill_diagonal(lambda_per_edge, 0.0)

    # Skip if no meaningful entries
    if np.max(prior_matrix) - np.min(prior_matrix) < eps:
        return lambda_per_edge

    # ── Normalize edge_strength to [0, 1] ──
    max_s = np.max(edge_strength)
    if max_s > eps:
        strength_norm = edge_strength / max_s
    else:
        strength_norm = edge_strength.copy()

    # ── For each edge, compute agreement ──
    # Case 1: prior says "edge" (P > 0.5)
    #   Agreement = data also shows edge (strength > threshold)
    # Case 2: prior says "no edge" (P < 0.5)
    #   Agreement = data also shows no edge (strength < threshold)
    # Case 3: prior is uncertain (P ≈ 0.5)
    #   Agreement = 0.5 (neutral)

    # A signed agreement measure:
    # prior_signed = 2 * (prior - 0.5)  → [-1, 1] (negative = "no edge", positive = "edge")
    # data_signed = 2 * (strength_norm - 0.5)  → [-1, 1]

    prior_signed = 2.0 * (prior_matrix - 0.5)
    data_signed = 2.0 * (strength_norm - 0.5)

    # Agreement: product of signs (1 = agree, -1 = disagree)
    # Then map to [0, 1]: τ = (agreement + 1) / 2
    agreement = prior_signed * data_signed  # ∈ [-1, 1]
    trust = (agreement + 1.0) / 2.0  # ∈ [0, 1]; 1 = perfect agreement

    # Apply reliability penalty: high std → lower trust
    # If bootstrap std is very high relative to strength, we're less sure of the estimate
    max_std = np.max(edge_std)
    if max_std > eps:
        reliability = 1.0 - np.clip(edge_std / max_std, 0, 1)
        reliability_weight = 0.3  # how much reliability affects trust
        trust = (1 - reliability_weight) * trust + reliability_weight * reliability

    np.fill_diagonal(trust, 0.0)

    # ── Map trust to λ ──
    # trust = 0  →  λ = λ_base * min_lambda_ratio (attenuate fully)
    # trust = 0.5 → λ = λ_base (neutral)
    # trust = 1  →  λ = λ_base * max_lambda_ratio (boost fully)
    lambda_range = max_lambda_ratio - min_lambda_ratio
    lambda_per_edge = lambda_base * (min_lambda_ratio + trust * lambda_range)

    # Protect diagonal
    np.fill_diagonal(lambda_per_edge, 0.0)

    return lambda_per_edge
