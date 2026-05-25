"""
Bootstrap NOTEARS: simple, robust causal discovery with calibrated uncertainty.
"""

import numpy as np
import torch
from tqdm import trange
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import warnings

from causbayes.structure_learning.base import BaseStructureLearner
from causbayes.structure_learning.utils import dagness


def notears_linear(
    X: np.ndarray,
    lambda_1: float = 0.01,
    max_iter: int = 60,
    lr: float = 1e-2,
    rho_max: float = 1e8,
    seed: int = 42,
    verbose: bool = False,
) -> np.ndarray:
    """Robust linear NOTEARS (Zheng et al., 2018).

    Optimizes: min ||X - XW||² + λ₁||W||₁  s.t. h(W) = 0
    with early stopping when h(W) stops improving.

    Args:
        X: Standardized data (n, d)
        lambda_1: L1 regularization
        max_iter: Max outer iterations
        lr: Learning rate
        rho_max: Max augmented Lagrangian penalty
        seed: Random seed
        verbose: Print progress

    Returns:
        W: Weight matrix (d, d), or best valid W before NaN
    """
    torch.manual_seed(seed)
    d = X.shape[1]
    X_t = torch.from_numpy(X).float()

    W = torch.zeros(d, d, requires_grad=True)
    with torch.no_grad():
        W.data.add_(torch.randn(d, d) * 1e-3)

    optimizer = torch.optim.AdamW([W], lr=lr, weight_decay=0.0)

    rho = 1.0
    alpha = 0.0
    h = np.inf
    best_h = np.inf
    best_W = None
    stall_count = 0

    # Precompute SVD for fast optimization (batch-style)
    # Use L-BFGS inner loop for faster convergence
    pbar = trange(max_iter, desc="NOTEARS", disable=not verbose)

    for outer in pbar:
        # Inner loop: fewer, faster steps
        n_inner = min(15, 3 + outer)
        for _ in range(n_inner):
            optimizer.zero_grad()
            X_pred = X_t @ W.T
            recon = torch.mean((X_t - X_pred) ** 2)
            l1 = lambda_1 * torch.sum(torch.abs(W))
            h_val = dagness(W)
            h_penalty = alpha * h_val + 0.5 * rho * h_val ** 2
            loss = recon + l1 + h_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_([W], 5.0)
            optimizer.step()

        # Check h(W)
        with torch.no_grad():
            h_new = dagness(W).item()

        # Track best (lowest h) weight matrix
        if not np.isnan(h_new) and h_new < best_h:
            best_h = h_new
            best_W = W.detach().clone().numpy()
            stall_count = 0
        else:
            stall_count += 1

        # Early stopping: h(W) not improving
        if stall_count >= 10:
            if verbose:
                pbar.set_postfix({"h(W)": f"{h_new:.2e}", "stopped": "improving"}, refresh=False)
            break

        # Early stopping: converged
        if h_new < 1e-8:
            if verbose:
                pbar.set_postfix({"h(W)": f"{h_new:.2e}", "converged": True}, refresh=False)
            best_W = W.detach().clone().numpy()
            break

        # Early stopping: NaN
        if np.isnan(h_new) or np.isnan(W.detach().numpy()).any():
            if verbose:
                pbar.set_postfix({"h(W)": "NaN", "rho": f"{rho:.1e}"}, refresh=False)
            break

        # Update augmented Lagrangian
        if h_new > 0.25 * h and h < np.inf:
            rho = min(rho * 10, rho_max)
        alpha += rho * h_new
        h = h_new

        if verbose:
            pbar.set_postfix({"h(W)": f"{h:.2e}", "rho": f"{rho:.1e}"}, refresh=False)

    if best_W is None:
        warnings.warn("NOTEARS failed to find valid W. Returning zeros.")
        return np.zeros((d, d))

    return best_W


class BootstrapDAG(BaseStructureLearner):
    """Bootstrapped causal discovery with uncertainty.

    Runs NOTEARS on N bootstrap samples to get a distribution over DAGs.
    Edge probability = proportion of bootstraps where edge is present.

    Parameters
    ----------
    n_bootstraps : int
        Number of bootstrap samples. Default: 50
    lambda_1 : float
        L1 regularization. Default: 0.01
    threshold : float or None
        Edge probability threshold. Calibrated on val data if available.
    max_iter : int
        Max iterations per NOTEARS run. Default: 60
    lr : float
        Learning rate. Default: 1e-2
    rho_max : float
        Max augmented Lagrangian penalty. Default: 1e8
    verbose : bool
        Print progress. Default: True
    seed : int
        Random seed. Default: 42
    """

    def __init__(
        self,
        n_bootstraps: int = 50,
        lambda_1: float = 0.01,
        threshold: float = None,
        max_iter: int = 60,
        lr: float = 1e-2,
        rho_max: float = 1e8,
        verbose: bool = True,
        seed: int = 42,
    ):
        super().__init__(seed=seed)
        self.n_bootstraps = n_bootstraps
        self.lambda_1 = lambda_1
        self.threshold = threshold
        self.max_iter = max_iter
        self.lr = lr
        self.rho_max = rho_max
        self.verbose = verbose
        self.scaler_ = StandardScaler()
        self._weight_matrices_ = []
        self._converged_count_ = 0

    def fit(self, X: np.ndarray, y=None, X_val: np.ndarray = None, W_val: np.ndarray = None):
        """Fit bootstrapped causal discovery.

        Args:
            X: Training data (n, d)
            X_val: Validation data for threshold calibration (optional)
            W_val: Ground truth for validation threshold calibration (optional)
        """
        n, d = X.shape
        X_scaled = self.scaler_.fit_transform(X)

        # Run bootstrap NOTEARS
        self._weight_matrices_ = []
        failed = 0

        pbar = trange(self.n_bootstraps, desc="Bootstrap NOTEARS", disable=not self.verbose)
        for i in pbar:
            try:
                X_boot = resample(X_scaled, random_state=self.seed + i)
                W_i = notears_linear(
                    X_boot,
                    lambda_1=self.lambda_1,
                    max_iter=self.max_iter,
                    lr=self.lr,
                    rho_max=self.rho_max,
                    seed=self.seed + i + 1000,
                    verbose=False,
                )
                if not np.isnan(W_i).any():
                    self._weight_matrices_.append(W_i)
                else:
                    failed += 1
            except Exception:
                failed += 1

        self._converged_count_ = len(self._weight_matrices_)

        if self.verbose:
            print(f"  Valid: {len(self._weight_matrices_)}/{self.n_bootstraps} "
                  f"(failed: {failed})")

        if len(self._weight_matrices_) == 0:
            warnings.warn("All bootstrap runs failed! Returning zeros.")
            self._edge_probs_ = np.zeros((d, d))
            self._edge_stds_ = np.zeros((d, d))
            return self

        # Compute edge probabilities from bootstrap distribution
        W_stack = np.array(self._weight_matrices_)
        W_abs = np.abs(W_stack)

        # Use median + MAD for robust threshold detection
        all_w = W_abs[W_abs > 1e-8].ravel()
        if len(all_w) == 0:
            self._edge_probs_ = np.zeros((d, d))
            self._edge_stds_ = np.zeros((d, d))
            return self

        # Edge = present when |W| > p95 of all weights (conservative)
        threshold_abs = np.percentile(all_w, 95) if len(all_w) > 1 else 0.1
        mask = W_abs > max(threshold_abs, 1e-4)
        self._edge_probs_ = np.mean(mask, axis=0)
        self._edge_stds_ = np.std(W_abs, axis=0)
        np.fill_diagonal(self._edge_probs_, 0.0)
        np.fill_diagonal(self._edge_stds_, 0.0)

        # If validation data available, calibrate threshold
        if X_val is not None and W_val is not None:
            self.threshold = self._calibrate_threshold(X_val, W_val)

        return self

    def _calibrate_threshold(self, X_val, W_val):
        """Find best probability threshold using validation data."""
        from causbayes.structure_learning.utils import structural_hamming_distance

        best_shd = float("inf")
        best_t = 0.5

        for t in np.linspace(0.05, 0.95, 19):
            W_bin = (self._edge_probs_ >= t).astype(float)
            shd = structural_hamming_distance(W_val, W_bin)
            if shd < best_shd:
                best_shd = shd
                best_t = t

        if self.verbose:
            print(f"  Val-calibrated threshold: {best_t:.2f} (SHD={best_shd:.1f})")
        return best_t

    @property
    def adjacency_matrix(self) -> np.ndarray:
        t = self.threshold if self.threshold is not None else 0.5
        return (self._edge_probs_ >= t).astype(float)

    def sample_graphs(self, n_samples: int = 10) -> list:
        if len(self._weight_matrices_) == 0:
            raise RuntimeError("Not fitted yet")
        indices = np.random.choice(len(self._weight_matrices_), n_samples, replace=True)
        t = self.threshold if self.threshold is not None else 0.5
        return [(np.abs(self._weight_matrices_[i]) > t * 0.1).astype(float)
                for i in indices]
