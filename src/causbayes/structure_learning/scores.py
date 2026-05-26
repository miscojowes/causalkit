"""
Information criteria scores for causal model selection.

Provides BIC (Bayesian Information Criterion) for comparing
DAG structures without NOTEARS optimization. Useful for:
- Model selection among candidate DAGs
- Post-processing NOTEARS output to pick the best threshold
- Comparing gCastle-compatible BIC scores
"""

import numpy as np
import warnings


def bic_score(X: np.ndarray, W: np.ndarray) -> float:
    """Bayesian Information Criterion for a linear Gaussian SEM.

    BIC(M) = -2 * log_likelihood(M) + k * log(n)

    For linear Gaussian SEM: X = X @ W + E
        BIC = n * log(RSS / n) + num_edges * log(n)

    Where:
        RSS = ||X - X @ W||_F^2  (sum of squared residuals)
        num_edges = number of non-zero entries in W (excluding diagonal)

    Lower BIC indicates a better model. BIC penalizes complexity
    (more edges) and rewards good fit (low MSE).

    Args:
        X: Data matrix of shape (n_samples, n_vars)
        W: Weight/adjacency matrix of shape (n_vars, n_vars)
           W[i,j] = coefficient from variable i to variable j

    Returns:
        BIC score (float). Lower is better.

    Example:
        >>> X = np.random.randn(100, 5)
        >>> W = np.eye(5) * 0.0  # empty graph
        >>> bic_score(X, W)  # doctest:+ELLIPSIS
        6...
    """
    n, d = X.shape

    # Center the data (NOTEARS convention)
    X_centered = X - X.mean(axis=0, keepdims=True)

    # Compute predicted values and residuals
    X_pred = X_centered @ W
    residuals = X_centered - X_pred

    # Residual sum of squares
    RSS = float(np.sum(residuals ** 2))

    # Number of edges (non-zero off-diagonal parameters)
    W_abs = np.abs(W)
    num_edges = int(np.sum(W_abs > 1e-8))
    diag_mask = np.eye(d, dtype=bool)
    num_edges_offdiag = num_edges - int(np.sum(W_abs[diag_mask] > 1e-8))

    # BIC formulation
    # BIC = n * log(RSS / n) + k * log(n)
    # where k = number of free parameters (off-diagonal edges)
    if RSS <= 0.0:
        # Perfect fit — push score very negative (extreme preference)
        RSS = 1e-16
        warnings.warn("RSS is zero or negative in BIC computation. Clamping to 1e-16.")

    bic = n * np.log(RSS / n) + num_edges_offdiag * np.log(n)

    return float(bic)


def bic_per_variable(X: np.ndarray, W: np.ndarray) -> float:
    """BIC score computed per-variable (more precise for SEM with different noise variances).

    For each variable j, fit X_j = sum_i W[i,j] * X_i + epsilon_j.
    Sum the per-variable BIC scores.

    This is more precise than the global BIC when noise variances differ
    across variables.

    Args:
        X: Data matrix of shape (n_samples, n_vars)
        W: Weight matrix where W[i,j] = coefficient from i to j

    Returns:
        Per-variable BIC score (float). Lower is better.
    """
    n, d = X.shape
    X_centered = X - X.mean(axis=0, keepdims=True)

    total_bic = 0.0

    for j in range(d):
        # Find parents of j
        parents = np.where(np.abs(W[:, j]) > 1e-8)[0]

        if len(parents) == 0:
            # No parents: predict with mean (zero after centering)
            residuals = X_centered[:, j]
            RSS_j = float(np.sum(residuals ** 2))
            k_j = 0
        else:
            # Predict X_j from its parents
            X_pred_j = X_centered[:, parents] @ W[parents, j]
            residuals = X_centered[:, j] - X_pred_j
            RSS_j = float(np.sum(residuals ** 2))
            k_j = len(parents)

        if RSS_j <= 0.0:
            RSS_j = 1e-16

        # Per-variable BIC
        bic_j = n * np.log(RSS_j / n) + k_j * np.log(n)
        total_bic += bic_j

    return float(total_bic)


def aic_score(X: np.ndarray, W: np.ndarray) -> float:
    """Akaike Information Criterion for linear Gaussian SEM.

    AIC = n * log(RSS / n) + 2 * num_edges

    Args:
        X: Data matrix of shape (n_samples, n_vars)
        W: Weight/adjacency matrix of shape (n_vars, n_vars)

    Returns:
        AIC score (float). Lower is better.
    """
    n, d = X.shape
    X_centered = X - X.mean(axis=0, keepdims=True)

    X_pred = X_centered @ W
    residuals = X_centered - X_pred
    RSS = float(np.sum(residuals ** 2))

    if RSS <= 0.0:
        RSS = 1e-16

    W_abs = np.abs(W)
    num_edges = int(np.sum(W_abs > 1e-8))
    diag_mask = np.eye(d, dtype=bool)
    num_edges_offdiag = num_edges - int(np.sum(W_abs[diag_mask] > 1e-8))

    aic = n * np.log(RSS / n) + 2.0 * num_edges_offdiag
    return float(aic)
