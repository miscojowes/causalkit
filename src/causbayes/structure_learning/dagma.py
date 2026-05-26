"""
DAGMA-style acyclicity constraint for differentiable DAG learning.

Implements the DAGMA acyclicity function from:
    Bello, K., Aragam, B., & Ravikumar, P. (2022).
    "DAGMA: Learning DAGs via M-matrices and a Log-Determinant
     Acyclicity Characterization."
    NeurIPS 2022. https://arxiv.org/abs/2209.08037

Key advantages over expm-based h(W):
    - No overflow risk from matrix exponential
    - Simpler gradient computation
    - More stable optimization landscape
    - Penalty increases smoothly with cycle presence

The DAGMA acyclicity characterization:
    h(W) = -log(det(sI - W⊙W)) + d * log(s)

Properties:
    - h(W) = 0  iff  W represents a DAG
    - h(W) > 0  iff  W contains cycles
    - Requires s > spectral_radius(W⊙W) for valid determinant
"""

import warnings
import torch
import numpy as np


def dagma_acyclicity(W: torch.Tensor, s: float = 1.0) -> torch.Tensor:
    """DAGMA acyclicity penalty h(W) = -log(det(sI - W⊙W)) + d*log(s).

    From Bello et al. (2022). Returns zero for DAGs, positive for cyclic graphs.
    More numerically stable than the expm-based penalty because:
    - No risk of exponential overflow
    - Gradient is well-conditioned
    - Penalty magnitude scales naturally with cycle strength

    The parameter s must satisfy s > spectral_radius(W⊙W) for the
    determinant to be well-defined (positive). If the determinant is
    non-positive, this function falls back to an adaptive strategy:
    it computes the spectral radius and uses s = max(1.1 * rho, 1.0).

    Args:
        W: Weight matrix of shape (d, d). Gradients flow through this.
        s: Scaling parameter. Must be > spectral_radius(W⊙W) for valid det.
           Default: 1.0. The function will adapt s if needed.

    Returns:
        Scalar penalty (non-negative, zero for DAGs).

    Example:
        >>> d = 4
        >>> W_dag = torch.triu(torch.randn(d, d), k=1)  # upper triangular = DAG
        >>> dagma_acyclicity(W_dag).item()  # near zero
        0.0

        >>> W_cycle = torch.tensor([[0.0, 1.0], [1.0, 0.0]])  # 2-cycle
        >>> dagma_acyclicity(W_cycle).item() > 0.0
        True
    """
    d = W.shape[0]
    W_sq = W * W  # element-wise square (Hadamard product)
    device = W.device
    dtype = W.dtype

    # Try to compute with given s
    I = torch.eye(d, device=device, dtype=dtype)
    M = s * I - W_sq

    # Use slogdet for numerical stability (handles sign of det)
    sign, logdet = torch.linalg.slogdet(M)

    # If sign <= 0, the determinant is non-positive — s is too small
    if sign <= 0:
        # Compute spectral radius of W⊙W for adaptive scaling
        with torch.no_grad():
            try:
                # For symmetric positive semi-definite W_sq, use eigvalsh
                eigenvalues = torch.linalg.eigvalsh(W_sq)
                spectral_radius = eigenvalues.max().abs()
            except Exception:
                # Fallback: use norm estimate
                spectral_radius = torch.linalg.matrix_norm(W_sq, ord=2)

        # Ensure s > spectral_radius with a safety margin
        s_adaptive = max(1.1 * float(spectral_radius), 1.0)

        if s_adaptive == float(spectral_radius):
            s_adaptive *= 1.1

        # Recompute with adaptive s
        M = s_adaptive * I - W_sq
        sign, logdet = torch.linalg.slogdet(M)

        # If still broken, resort to very conservative s
        if sign <= 0:
            s_adaptive = max(2.0 * float(spectral_radius), 1.0)
            M = s_adaptive * I - W_sq
            sign, logdet = torch.linalg.slogdet(M)

        s_used = torch.tensor(s_adaptive, device=device, dtype=dtype)
        h_val = -logdet + d * torch.log(s_used)
    else:
        h_val = -logdet + d * torch.log(torch.tensor(s, device=device, dtype=dtype))

    # Clamp to non-negative (should be zero for DAGs, but numerical noise may
    # produce tiny negative values)
    return torch.clamp(h_val, min=0.0)


def dagma_spectral_radius(W: torch.Tensor) -> float:
    """Compute spectral radius of W⊙W for DAGMA s-parameter setting.

    Helper to determine the minimum valid s for dagma_acyclicity(W, s).

    Args:
        W: Weight matrix of shape (d, d)

    Returns:
        Spectral radius (largest absolute eigenvalue of W⊙W)

    Example:
        >>> W = torch.tensor([[0.0, 2.0], [1.0, 0.0]])
        >>> rho = dagma_spectral_radius(W)
        >>> rho > 0.0
        True
    """
    W_sq = W * W

    with torch.no_grad():
        try:
            eigenvalues = torch.linalg.eigvalsh(W_sq)
            return float(eigenvalues.max().abs())
        except Exception:
            return float(torch.linalg.matrix_norm(W_sq, ord=2))


def dagma_acyclicity_gradient(W: torch.Tensor, s: float = 1.0) -> torch.Tensor:
    """Analytic gradient of the DAGMA acyclicity penalty.

    d(h)/dW = 2 * W ⊙ (sI - W⊙W)^{-T}

    where ⊙ is element-wise multiplication and (·)^{-T} is inverse transpose.

    This is typically used via autograd (W.requires_grad = True, then
    dagma_acyclicity(W).backward()), but this function is provided for
    explicit gradient computation when needed.

    Args:
        W: Weight matrix of shape (d, d)
        s: Scaling parameter (> spectral_radius(W⊙W))

    Returns:
        Gradient matrix of shape (d, d)
    """
    d = W.shape[0]
    device = W.device
    dtype = W.dtype

    W_sq = W * W
    I = torch.eye(d, device=device, dtype=dtype)
    M = s * I - W_sq

    # Compute M^{-T} — the transpose of the inverse
    M_inv = torch.linalg.inv(M)
    M_inv_T = M_inv.T

    # d(h)/dW = 2 * W ⊙ M^{-T}
    grad = 2.0 * W * M_inv_T

    return grad


def dagma_is_dag(W: torch.Tensor, s: float = 1.0, tol: float = 1e-6) -> bool:
    """Check if weighted adjacency matrix represents a DAG using DAGMA penalty.

    More reliable than expm-based check for large weights because
    the DAGMA formulation doesn't suffer from exponential overflow.

    Args:
        W: Weight matrix (torch.Tensor)
        s: Scaling parameter for DAGMA
        tol: Tolerance for zero penalty

    Returns:
        True if the matrix represents a DAG
    """
    if isinstance(W, np.ndarray):
        W = torch.from_numpy(W).float()
    h_val = dagma_acyclicity(W, s=s)
    return h_val.item() < tol
