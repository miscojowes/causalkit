"""
MC Dropout for epistemic uncertainty estimation.

Approximates Bayesian inference over neural network weights
by applying dropout at inference time (Gal & Ghahramani, 2016).
"""

import torch
import torch.nn as nn
import numpy as np


class MCDropout:
    """MC Dropout wrapper for weight uncertainty.

    Runs multiple forward passes with dropout enabled to
    approximate posterior distribution over model parameters.

    Parameters
    ----------
    model : nn.Module
        The trained neural network model
    n_samples : int
        Number of MC samples. Default: 100
    """

    def __init__(self, model: nn.Module, n_samples: int = 100):
        self.model = model
        self.n_samples = n_samples

    def sample_weight_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """Get weight matrix sample by running forward pass with dropout.

        Args:
            X: Input data

        Returns:
            Sampled weight matrix
        """
        self.model.train()  # Enable dropout
        with torch.no_grad():
            _ = self.model(X)  # Forward pass to propagate dropout
            W_sample = self.model.compute_weight_matrix()
        self.model.eval()
        return W_sample

    def predict_with_uncertainty(self, X: torch.Tensor) -> tuple:
        """Predict with MC dropout uncertainty.

        Args:
            X: Input data of shape (batch, n_vars)

        Returns:
            Tuple of (mean_prediction, prediction_variance)
        """
        self.model.train()
        predictions = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self.model(X).cpu().numpy()
                predictions.append(pred)
        self.model.eval()

        predictions = np.array(predictions)
        return predictions.mean(axis=0), predictions.var(axis=0)
