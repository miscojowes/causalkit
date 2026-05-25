"""
Prior distributions for Bayesian causal discovery.

Implements spike-and-slab, horseshoe, and edge-specific priors
for structured sparsity in DAG learning.
"""

import torch
import torch.nn as nn
import numpy as np


class SpikeAndSlabPrior:
    """Spike-and-slab prior for edge weights.

    Each edge weight w ~ pi * N(0, sigma_slab^2) + (1-pi) * delta_0

    Encourages sparsity while allowing strong edges to escape shrinkage.

    Parameters
    ----------
    d : int
        Number of variables
    pi_spike : float
        Probability of being in the spike (near-zero). Default: 0.8
    sigma_slab : float
        Standard deviation of the slab component. Default: 1.0
    """

    def __init__(self, d: int, pi_spike: float = 0.8, sigma_slab: float = 1.0):
        self.d = d
        self.pi_spike = pi_spike
        self.sigma_slab = sigma_slab

    def log_prob(self, W: torch.Tensor) -> torch.Tensor:
        """Compute log prior probability of weight matrix.

        Args:
            W: Weight matrix of shape (d, d)

        Returns:
            Log prior probability scalar
        """
        # Spike: N(0, 1e-6) - very concentrated at zero
        # Slab: N(0, sigma_slab^2) - diffuse
        spike_var = 1e-6
        slab_var = self.sigma_slab ** 2

        spike_log_prob = -0.5 * torch.log(2 * torch.pi * torch.tensor(spike_var)) - W ** 2 / (2 * spike_var)
        slab_log_prob = -0.5 * torch.log(2 * torch.pi * torch.tensor(slab_var)) - W ** 2 / (2 * slab_var)

        log_prior = torch.logsumexp(
            torch.stack([
                torch.log(torch.tensor(1 - self.pi_spike)) + slab_log_prob,
                torch.log(torch.tensor(self.pi_spike)) + spike_log_prob,
            ]),
            dim=0,
        )

        # Sum over all edges (excluding diagonal)
        return torch.sum(log_prior * (1 - torch.eye(self.d, device=W.device)))


class HorseshoePrior:
    """Horseshoe prior for global-local shrinkage.

    Global shrinkage tau shrinks all weights toward zero,
    while local lambda_j allows strong signals to escape.

    Parameters
    ----------
    d : int
        Number of variables
    global_scale : float
        Global shrinkage parameter. Default: 0.1
    """

    def __init__(self, d: int, global_scale: float = 0.1):
        self.d = d
        self.global_scale = global_scale

    def log_prob(self, W: torch.Tensor) -> torch.Tensor:
        """Compute log prior probability (approximate).

        Uses closed-form marginal: w ~ Cauch(0, tau)
        Approximated with Student-t(1) = Cauchy.

        Args:
            W: Weight matrix

        Returns:
            Log prior probability scalar
        """
        tau = self.global_scale
        # Cauchy(0, tau) log density
        log_prob = -torch.log(torch.pi * tau) - torch.log(1 + (W / tau) ** 2)

        # Exclude diagonal
        return torch.sum(log_prob * (1 - torch.eye(self.d, device=W.device)))


def build_edge_prior_matrix(
    d: int,
    known_edges: list = None,
    known_non_edges: list = None,
    base_prob: float = 0.5,
    edge_prob: float = 0.9,
    non_edge_prob: float = 0.1,
) -> np.ndarray:
    """Build a prior probability matrix from domain knowledge.

    Args:
        d: Number of variables
        known_edges: List of (i, j) tuples for known causal edges
        known_non_edges: List of (i, j) tuples for known non-edges
        base_prob: Default edge probability. Default: 0.5
        edge_prob: Prior probability for known edges. Default: 0.9
        non_edge_prob: Prior probability for known non-edges. Default: 0.1

    Returns:
        Prior probability matrix of shape (d, d)
    """
    prior = np.full((d, d), base_prob)
    np.fill_diagonal(prior, 0.0)  # No self-loops

    if known_edges:
        for i, j in known_edges:
            prior[i, j] = edge_prob

    if known_non_edges:
        for i, j in known_non_edges:
            prior[i, j] = non_edge_prob

    return prior


def prior_from_associations(
    d: int,
    variable_names: list,
    associations: dict,
    base_prob: float = 0.5,
) -> np.ndarray:
    """Build prior from an association dictionary.

    Args:
        d: Number of variables
        variable_names: List of variable names
        associations: Dict mapping (parent_name, child_name) -> probability or "high"/"medium"/"low"
        base_prob: Default probability

    Returns:
        Prior probability matrix
    """
    name_to_idx = {name: i for i, name in enumerate(variable_names)}
    prior = np.full((d, d), base_prob)
    np.fill_diagonal(prior, 0.0)

    confidence_map = {
        "high": 0.9,
        "medium": 0.7,
        "low": 0.55,
    }

    for (parent, child), confidence in associations.items():
        if parent in name_to_idx and child in name_to_idx:
            i, j = name_to_idx[parent], name_to_idx[child]
            if isinstance(confidence, (int, float)):
                prior[i, j] = float(confidence)
            elif isinstance(confidence, str):
                prior[i, j] = confidence_map.get(confidence.lower(), base_prob)

    return prior
