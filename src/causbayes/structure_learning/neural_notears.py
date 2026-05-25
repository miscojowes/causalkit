"""
Neural Bayesian DAG: Non-linear causal structure learning with uncertainty.

Implements Neural NOTEARS with:
1. Non-linear SEM via neural networks (MLP per variable)
2. Acyclicity constraint (h(W) = 0)
3. Bayesian uncertainty via MC Dropout or variational inference
4. Optional LLM-informed priors

Based on:
- Zheng et al. (2018) "DAGs with NO TEARS"
- Lachapelle et al. (2020) "Gradient-based Neural DAG Learning"
- Ng et al. (2020) "DAGs with No Curl"
"""

import warnings
from typing import Optional, Union, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from tqdm import trange

from causbayes.structure_learning.base import BaseStructureLearner
from causbayes.structure_learning.utils import dagness, is_dag, _matrix_exp_power_series
from causbayes.models.nonlinear_sem import NonlinearSEM
from causbayes.bayesian.mc_dropout import MCDropout
from causbayes.bayesian.variational import VariationalInference


class NeuralBayesianDAG(BaseStructureLearner):
    """Neural Bayesian DAG: Non-linear causal discovery with uncertainty.

    Learns a DAG structure from observational data using gradient-based
    optimization with a neural network structural equation model.
    Provides uncertainty quantification over edges.

    Parameters
    ----------
    hidden_layers : list of int
        Hidden layer sizes for each variable's MLP. Default: [64, 64]
    learning_rate : float
        Adam learning rate. Default: 1e-3
    lambda_1 : float
        L1 regularization coefficient. Default: 1e-2
    lambda_2 : float
        Acyclicity penalty coefficient. Default: 5.0
    lambda_prior : float
        Prior regularization strength. Default: 1.0
    rho_max : float
        Maximum penalty for augmented Lagrangian. Default: 1e16
    h_tol : float
        Tolerance for acyclicity constraint. Default: 1e-8
    max_iter : int
        Maximum outer iterations. Default: 100
    batch_size : int
        Batch size for training. Default: None (full batch)
    uncertainty : str or None
        Uncertainty method: 'mc_dropout', 'variational', or None. Default: 'mc_dropout'
    mc_samples : int
        Number of MC dropout samples. Default: 50
    prior_matrix : np.ndarray or None
        Prior edge probabilities of shape (d, d). Default: None (uniform)
    prior_strength : float
        How strongly to enforce prior (0 = ignore, 1 = full). Default: 0.3
    device : str
        Device for computation ('cpu' or 'cuda'). Default: 'cpu'
    seed : int
        Random seed. Default: 42
    verbose : bool
        Print progress. Default: True
    """

    def __init__(
        self,
        hidden_layers: list = None,
        learning_rate: float = 1e-3,
        lambda_1: float = 1e-2,
        lambda_2: float = 5.0,
        lambda_prior: float = 1.0,
        rho_max: float = 1e16,
        h_tol: float = 1e-8,
        max_iter: int = 100,
        batch_size: Optional[int] = None,
        uncertainty: Optional[Literal["mc_dropout", "variational"]] = "mc_dropout",
        mc_samples: int = 50,
        prior_matrix: Optional[np.ndarray] = None,
        prior_strength: float = 0.3,
        device: str = "cpu",
        seed: int = 42,
        verbose: bool = True,
    ):
        super().__init__(seed=seed)
        self.hidden_layers = hidden_layers or [64, 64]
        self.learning_rate = learning_rate
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_prior = lambda_prior
        self.rho_max = rho_max
        self.h_tol = h_tol
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.uncertainty = uncertainty
        self.mc_samples = mc_samples
        self.prior_matrix = prior_matrix
        self.prior_strength = prior_strength
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.verbose = verbose

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Internal state
        self.model_ = None
        self.W_est_ = None
        self.scaler_ = StandardScaler()
        self._training_losses_ = []

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> "NeuralBayesianDAG":
        """Learn causal structure from data.

        Args:
            X: Data matrix of shape (n_samples, n_vars)
            y: Not used (for API compatibility)

        Returns:
            Self for chaining
        """
        n, d = X.shape

        # Standardize
        X_scaled = self.scaler_.fit_transform(X)
        X_tensor = torch.from_numpy(X_scaled).float().to(self.device)

        # Initialize neural SEM model
        self.model_ = NonlinearSEM(
            n_vars=d,
            hidden_layers=self.hidden_layers,
        ).to(self.device)

        # Training
        self._train_neural_notears(X_tensor, d)

        # Extract weight matrix (first layer weights as edge strength)
        self.W_est_ = self.model_.compute_weight_matrix().detach().cpu().numpy()

        # Compute uncertainty
        if self.uncertainty == "mc_dropout":
            self._compute_mc_dropout_uncertainty(X_tensor)
        elif self.uncertainty == "variational":
            self._compute_variational_uncertainty(X_tensor)
        else:
            # No uncertainty: binary from threshold
            W_abs = np.abs(self.W_est_)
            threshold = np.percentile(W_abs[W_abs > 1e-8], 50) if np.sum(W_abs > 1e-8) > 0 else 0.1
            self._edge_probs_ = (W_abs > threshold).astype(float)
            self._edge_stds_ = np.zeros((d, d))

        return self

    def _train_neural_notears(self, X: torch.Tensor, d: int):
        """Train using augmented Lagrangian for acyclicity constraint.

        Follows the NOTEARS augmented Lagrangian approach:
        min L(W,theta) + lambda_1 * ||W||_1 + alpha * h(W) + 0.5 * rho * h(W)^2
        """
        rho = 1.0
        alpha = 0.0
        h = np.inf
        n = X.shape[0]
        self._training_losses_ = []

        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        batch_size = self.batch_size or n

        outer_loop = trange(self.max_iter, desc="Neural NOTEARS", disable=not self.verbose)

        for outer_iter in outer_loop:
            # Inner loop (L-BFGS style inner optimization)
            inner_losses = []
            n_inner = min(50, 10 + outer_iter * 2)

            for _ in range(n_inner):
                perm = torch.randperm(n)
                epoch_loss = 0.0
                n_batches = 0

                for start in range(0, n, batch_size):
                    idx = perm[start:start + batch_size]
                    batch_X = X[idx]

                    optimizer.zero_grad()

                    # Forward pass
                    X_pred = self.model_(batch_X)

                    # Reconstruction loss (MSE)
                    recon_loss = torch.mean((batch_X - X_pred) ** 2)

                    # Get edge weight matrix
                    W = self.model_.compute_weight_matrix()

                    # L1 sparsity regularization
                    l1_reg = self.lambda_1 * torch.sum(torch.abs(W))

                    # Prior-based regularization
                    prior_loss = self._compute_prior_loss(W)

                    # Acyclicity penalty
                    h_val = dagness(W)
                    h_penalty = alpha * h_val + 0.5 * rho * h_val ** 2

                    # Total loss
                    loss = recon_loss + l1_reg + h_penalty + prior_loss

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 10.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                inner_losses.append(epoch_loss / max(n_batches, 1))

            # Update augmented Lagrangian multipliers
            with torch.no_grad():
                W_current = self.model_.compute_weight_matrix()
                h_new = dagness(W_current).item()

            # Increase rho if h(W) is not decreasing fast enough
            if h_new > 0.25 * h and h < np.inf:
                rho *= 10.0

            # Update Lagrange multiplier
            alpha += rho * h_new
            h = h_new

            self._training_losses_.append(float(np.mean(inner_losses)))

            # Check convergence
            if h <= self.h_tol and outer_iter >= 1:
                if self.verbose:
                    outer_loop.set_postfix({"h(W)": f"{h:.2e}", "converged": True}, refresh=False)
                break

            if rho > self.rho_max:
                warnings.warn(f"rho exceeded max ({self.rho_max}). Stopping.")
                break

            if self.verbose:
                outer_loop.set_postfix({"h(W)": f"{h:.2e}", "rho": f"{rho:.1e}"}, refresh=False)

    def _compute_prior_loss(self, W: torch.Tensor) -> torch.Tensor:
        """Compute prior regularization loss.

        If a prior matrix is provided, penalizes edges that disagree
        with the prior. Uses a soft penalty based on prior_strength.
        """
        if self.prior_matrix is None:
            return torch.tensor(0.0, device=self.device)

        d = W.shape[0]
        prior_t = torch.from_numpy(self.prior_matrix).float().to(self.device)

        # Convert weight matrix to probability-like via sigmoid
        edge_probs = torch.sigmoid(W.abs())

        # KL-divergence-like penalty between edge distribution and prior
        # Penalize when sigmoid(W) disagrees with prior matrix
        prior_loss = torch.sum(
            prior_t * torch.log(prior_t / (edge_probs + 1e-8) + 1e-8)
            + (1 - prior_t) * torch.log((1 - prior_t) / (1 - edge_probs + 1e-8) + 1e-8)
        )

        return self.lambda_prior * self.prior_strength * prior_loss

    def _compute_mc_dropout_uncertainty(self, X: torch.Tensor):
        """Compute edge posterior probabilities using MC Dropout.

        Uses a distribution-aware approach:
        1. Mean weight per edge from MC samples
        2. Calibrated probabilities via soft rank-normalization:
           edges are ranked by mean weight and mapped to [0,1] using
           the overall distribution of weights across all edges.
        """
        d = X.shape[1]

        edge_samples = []
        for _ in trange(self.mc_samples, desc="MC Dropout", disable=not self.verbose):
            self.model_.train()
            with torch.no_grad():
                _ = self.model_(X)
                W_sample = self.model_.compute_weight_matrix()
            self.model_.eval()
            edge_samples.append(W_sample.cpu().detach().numpy())

        edge_samples = np.array(edge_samples)
        mean_W = np.mean(edge_samples, axis=0)
        std_W = np.std(edge_samples, axis=0)

        eps = 1e-8

        # Rank-based calibration using distribution of all mean weights.
        # Edge probability = how strong is this edge relative to the noise floor.
        flat_weights = mean_W.flatten()
        non_zero = flat_weights[flat_weights > eps]

        if len(non_zero) > 1:
            median = np.median(non_zero)
            q75, q25 = np.percentile(non_zero, [75, 25])
            iqr = max(q75 - q25, np.std(non_zero) * 0.5)

            # Logistic mapping centered at median, scaled by IQR
            # median weight -> P=0.5, median+iqr -> P~0.88, median-iqr -> P~0.12
            scale = iqr / 2.0
            if scale > eps:
                logits = (mean_W - median) / scale
                self._edge_probs_ = 1.0 / (1.0 + np.exp(-logits))
            else:
                self._edge_probs_ = (mean_W > 0).astype(float)
        else:
            self._edge_probs_ = np.zeros((d, d))

        np.fill_diagonal(self._edge_probs_, 0.0)

        # Normalized std: raw std is on same scale as normalized weights [0,1]
        # Add a small prior to avoid zero-std edges and reflect limited MC samples
        mc_prior = 0.01 / np.sqrt(max(self.mc_samples, 1))
        self._edge_stds_ = np.clip(std_W + mc_prior, 0.0, 1.0)

    def _compute_variational_uncertainty(self, X: torch.Tensor):
        """Compute edge posterior using variational inference.
        
        Placeholder for full VI over graph structure.
        Falls back to sigmoid calibration for now.
        """
        d = X.shape[1]

        vi = VariationalInference(self.model_, n_mc_samples=30)
        edge_samples = vi.sample_posterior(X, n_samples=self.mc_samples)

        # Edge probability = sigmoid-calibrated mean weight
        threshold = np.median(np.abs(edge_samples[edge_samples != 0])) if np.any(edge_samples != 0) else 0.01
        self._edge_probs_ = np.mean(np.abs(edge_samples) > threshold, axis=0)
        self._edge_stds_ = np.std(np.abs(edge_samples), axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict values using the learned SEM.

        Args:
            X: Input data of shape (n_samples, n_vars)

        Returns:
            Reconstructed data
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted yet.")

        X_scaled = self.scaler_.transform(X)
        X_tensor = torch.from_numpy(X_scaled).float().to(self.device)

        with torch.no_grad():
            X_pred = self.model_(X_tensor)

        return self.scaler_.inverse_transform(X_pred.cpu().numpy())

    def sample_graphs(self, n_samples: int = 100) -> list:
        """Sample DAGs from the posterior over graphs.

        Args:
            n_samples: Number of DAGs to sample

        Returns:
            List of binary adjacency matrices
        """
        graphs = []
        for _ in range(n_samples):
            W_sample = np.random.binomial(1, self._edge_probs_)
            # Ensure DAG by zeroing lower triangle of random permutation
            # (simplistic but fast)
            perm = np.random.permutation(W_sample.shape[0])
            W_sorted = W_sample[perm][:, perm]
            W_dag = np.triu(W_sorted, k=1)
            # Map back to original ordering
            inv_perm = np.argsort(perm)
            W_dag = W_dag[inv_perm][:, inv_perm]
            graphs.append(W_dag)
        return graphs

    def plot(
        self,
        threshold: float = 0.3,
        show_uncertainty: bool = True,
        figsize: tuple = (10, 8),
    ):
        """Visualize the learned DAG with uncertainty.

        Args:
            threshold: Minimum edge probability to display
            show_uncertainty: Show edge uncertainty as color/width
            figsize: Figure size
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            warnings.warn("matplotlib and networkx required for plotting")
            return

        from causbayes.visualization import plot_probabilistic_dag
        plot_probabilistic_dag(
            self._edge_probs_,
            self._edge_stds_,
            threshold=threshold,
            uncertainty=show_uncertainty,
            figsize=figsize,
        )

    def __repr__(self) -> str:
        return (
            f"NeuralBayesianDAG("
            f"uncertainty={self.uncertainty}, "
            f"hidden_layers={self.hidden_layers}"
            f")"
        )
