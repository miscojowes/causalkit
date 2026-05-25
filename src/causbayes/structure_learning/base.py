"""
Base class for causal structure learners.
"""

from abc import ABC, abstractmethod
import numpy as np
import torch


class BaseStructureLearner(ABC):
    """Abstract base class for causal structure learning algorithms."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._adjacency_matrix_ = None
        self._edge_probs_ = None
        self._edge_stds_ = None

    @abstractmethod
    def fit(self, X: np.ndarray, **kwargs) -> "BaseStructureLearner":
        """Learn causal structure from data.

        Args:
            X: Data matrix of shape (n_samples, n_vars)

        Returns:
            Self for chaining
        """
        ...

    def fit_transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Fit and return edge probability matrix.

        Args:
            X: Data matrix of shape (n_samples, n_vars)

        Returns:
            Edge probability matrix P[i,j] = probability of edge i -> j
        """
        self.fit(X, **kwargs)
        return self.edge_probs

    @property
    def adjacency_matrix(self) -> np.ndarray:
        """Binary adjacency matrix (thresholded at 0.5)."""
        if self._edge_probs_ is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        return (self._edge_probs_ >= 0.5).astype(float)

    @property
    def edge_probs(self) -> np.ndarray:
        """Edge probability matrix P[i,j]."""
        if self._edge_probs_ is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        return self._edge_probs_

    @property
    def edge_stds(self) -> np.ndarray:
        """Uncertainty (std dev) for each edge probability."""
        if self._edge_stds_ is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        return self._edge_stds_

    def get_top_edges(self, k: int = 10) -> list:
        """Return the top-k most probable edges with their probabilities.

        Args:
            k: Number of top edges to return

        Returns:
            List of ((i, j), prob, std) tuples
        """
        probs = self.edge_probs
        stds = self.edge_stds
        triu_indices = np.triu_indices_from(probs, k=1)
        edge_list = []
        for i, j in zip(*triu_indices):
            edge_list.append(((i, j), probs[i, j], stds[i, j]))
            edge_list.append(((j, i), probs[j, i], stds[j, i]))
        edge_list.sort(key=lambda x: x[1], reverse=True)
        return edge_list[:k]
