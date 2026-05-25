"""
Bootstrap NOTEARS: fast, robust causal discovery with calibrated uncertainty.

Uses SciPy L-BFGS-B NOTEARS (0.7s/run for d=5) for speed.
Supports L2 prior injection, Platt scaling calibration, and validation-based thresholding.
"""

import numpy as np
import warnings
from tqdm import trange
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from causbayes.structure_learning.base import BaseStructureLearner
from causbayes.structure_learning.notears_fast import (
    notears_lbfgs,
    calibrate_bootstrap_proportions,
    expected_calibration_error,
)
from causbayes.structure_learning.utils import structural_hamming_distance


class BootstrapDAG(BaseStructureLearner):
    """Bootstrapped causal discovery with calibrated uncertainty.

    Runs NOTEARS (SciPy L-BFGS-B, fast) on N bootstrap samples to get
    a distribution over DAGs. Edge probability = proportion of bootstraps
    where edge is present. Supports Platt scaling calibration and L2 priors.

    Parameters
    ----------
    n_bootstraps : int
        Number of bootstrap samples. Default: 50
    lambda_1 : float
        L1 regularization. Default: 0.01
    threshold : float or None
        Edge probability threshold. Calibrated on val data if available.
    max_iter : int
        Max augmented Lagrangian iterations. Default: 10
    w_threshold : float
        Prune small weights. Default: 0.1
    prior_matrix : np.ndarray or None
        Prior knowledge matrix (d,d) with values in [0,1].
    lambda_prior : float
        L2 penalty strength for prior deviation. Default: 0.0
    calibrate : bool
        Apply Platt scaling calibration. Default: True
    verbose : bool
        Print progress. Default: True
    seed : int
        Random seed. Default: 42
    lbfgs_maxiter : int
        Max L-BFGS iterations per call. Default: 20
    """

    def __init__(
        self,
        n_bootstraps: int = 50,
        lambda_1: float = 0.01,
        threshold: float = None,
        max_iter: int = 10,
        w_threshold: float = 0.1,
        prior_matrix: np.ndarray = None,
        lambda_prior: float = 0.0,
        calibrate: bool = True,
        verbose: bool = True,
        seed: int = 42,
        lbfgs_maxiter: int = 20,
    ):
        super().__init__(seed=seed)
        self.n_bootstraps = n_bootstraps
        self.lambda_1 = lambda_1
        self.threshold = threshold
        self.max_iter = max_iter
        self.w_threshold = w_threshold
        self.prior_matrix = prior_matrix
        self.lambda_prior = lambda_prior
        self.calibrate = calibrate
        self.verbose = verbose
        self.lbfgs_maxiter = lbfgs_maxiter
        self.scaler_ = StandardScaler()
        self._weight_matrices_ = []
        self._converged_count_ = 0
        self._calibration_params_ = None

    def fit(self, X: np.ndarray, y=None, X_val: np.ndarray = None, W_val: np.ndarray = None):
        """Fit bootstrapped causal discovery.

        Args:
            X: Training data (n, d)
            X_val: Validation data for threshold calibration (optional)
            W_val: Ground truth for validation threshold calibration (optional)
        """
        n, d = X.shape
        X_scaled = self.scaler_.fit_transform(X)

        # Run bootstrap NOTEARS (fast SciPy version)
        self._weight_matrices_ = []
        failed = 0

        pbar = trange(self.n_bootstraps, desc="Bootstrap NOTEARS",
                      disable=not self.verbose)
        for i in pbar:
            try:
                X_boot = resample(X_scaled, random_state=self.seed + i)
                X_boot = X_boot - X_boot.mean(axis=0, keepdims=True)
                W_i = notears_lbfgs(
                    X_boot,
                    lambda_1=self.lambda_1,
                    max_iter=self.max_iter,
                    w_threshold=self.w_threshold,
                    lbfgs_maxiter=self.lbfgs_maxiter,
                    prior_matrix=self.prior_matrix,
                    lambda_prior=self.lambda_prior,
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

        # Compute coarse edge probabilities from bootstrap distribution
        W_stack = np.array(self._weight_matrices_)
        W_abs = np.abs(W_stack)

        # Edge = present when |W| > small threshold (captures any non-zero)
        mask = W_abs > 1e-4
        self._edge_probs_raw_ = np.mean(mask, axis=0)
        self._edge_stds_ = np.std(W_abs, axis=0)
        np.fill_diagonal(self._edge_probs_raw_, 0.0)
        np.fill_diagonal(self._edge_stds_, 0.0)

        # Platt scaling calibration
        self._calibration_params_ = None
        if self.calibrate and W_val is not None:
            P_cal, a, b = calibrate_bootstrap_proportions(
                self._edge_probs_raw_, W_val
            )
            self._edge_probs_ = P_cal
            self._calibration_params_ = (a, b)
            if self.verbose:
                ece_before = expected_calibration_error(
                    self._edge_probs_raw_, W_val
                )
                ece_after = expected_calibration_error(P_cal, W_val)
                print(f"  ECE: {ece_before:.4f} → {ece_after:.4f} "
                      f"(Platt a={a:.3f}, b={b:.3f})")
        else:
            self._edge_probs_ = self._edge_probs_raw_.copy()

        # Threshold calibration using validation data
        if X_val is not None and W_val is not None:
            self.threshold = self._calibrate_threshold(X_val, W_val)

        # Apply default threshold
        if self.threshold is None:
            self.threshold = 0.5

        return self

    def _calibrate_threshold(self, X_val, W_val):
        """Find best probability threshold using validation data."""
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

    @property
    def edge_probs_raw(self) -> np.ndarray:
        """Uncalibrated raw bootstrap proportions."""
        if not hasattr(self, '_edge_probs_raw_'):
            return self._edge_probs_
        return self._edge_probs_raw_

    def sample_graphs(self, n_samples: int = 10) -> list:
        if len(self._weight_matrices_) == 0:
            raise RuntimeError("Not fitted yet")
        indices = np.random.choice(len(self._weight_matrices_), n_samples, replace=True)
        t = self.threshold if self.threshold is not None else 0.5
        return [(np.abs(self._weight_matrices_[i]) > t * 0.1).astype(float)
                for i in indices]
