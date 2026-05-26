"""
causalkit.effects — Causal effect estimation
=============================================
ATE estimation, what-if prediction, and root cause analysis
using the discovered causal graph.
"""

import numpy as np


def estimate_ate(
    X,
    treatment,
    outcome,
    causal_matrix,
    feature_names=None,
    method="linear",
):
    """Estimate ATE via linear structural equation model.

    Uses the discovered DAG to identify the adjustment set
    (parents of treatment), then estimates the treatment effect
    via OLS.

    Parameters
    ----------
    X : ndarray (n, d)
        Data.
    treatment : str or int
        Treatment variable.
    outcome : str or int
        Outcome variable.
    causal_matrix : ndarray (d, d)
        Weighted adjacency matrix.
    feature_names : list or None
        Variable names.
    method : str
        Currently only 'linear' supported.

    Returns
    -------
    ate : float
    """
    d = X.shape[1]

    # Resolve indices
    if isinstance(treatment, str):
        if feature_names is None:
            raise ValueError("feature_names required when using string variables")
        t_idx = feature_names.index(treatment)
    else:
        t_idx = int(treatment)

    if isinstance(outcome, str):
        if feature_names is None:
            raise ValueError("feature_names required when using string variables")
        o_idx = feature_names.index(outcome)
    else:
        o_idx = int(outcome)

    if t_idx == o_idx:
        raise ValueError("Treatment and outcome must be different")

    W = causal_matrix

    # Find adjustment set: parents of treatment (back-door criterion)
    treatment_parents = np.where(W[:, t_idx] > 0)[0]
    # Also include any confounders (common causes of treatment AND outcome)
    outcome_parents = np.where(W[:, o_idx] > 0)[0]
    confounders = np.intersect1d(treatment_parents, outcome_parents)
    # Union of parents and confounders for a valid adjustment set
    adjust = np.unique(np.concatenate([treatment_parents, confounders]))
    # Remove treatment itself if accidentally included
    adjust = adjust[adjust != t_idx]

    # OLS: outcome ~ treatment + adjust_set
    if len(adjust) == 0:
        # Simple bivariate regression
        X_t = X[:, t_idx].reshape(-1, 1)
        coef = np.linalg.lstsq(X_t, X[:, o_idx], rcond=None)[0]
        return float(coef[0])

    # Multiple regression
    X_design = np.column_stack([X[:, t_idx], X[:, adjust]])
    coefs = np.linalg.lstsq(X_design, X[:, o_idx], rcond=None)[0]
    return float(coefs[0])


def estimate_risk_diff(X, treatment, outcome, causal_matrix, feature_names=None):
    """Risk difference: P(outcome=1 | do(treatment=1)) - P(outcome=1 | do(treatment=0))

    For binary treatment and outcome. Uses linear probability model.
    """
    ate = estimate_ate(X, treatment, outcome, causal_matrix, feature_names, method="linear")
    return ate  # For binary OLS, ATE ≈ risk difference
