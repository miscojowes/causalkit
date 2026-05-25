"""
DAG utilities: acyclicity constraints, graph operations, metrics.
"""

import torch
import numpy as np
import networkx as nx


def dagness(W: torch.Tensor) -> torch.Tensor:
    """Acyclicity penalty h(W) = trace(exp(W⊙W)) - d.

    From Zheng et al. (2018) NOTEARS: h(W) = 0 iff W represents a DAG.
    Normalizes W first to prevent numerical overflow.

    Args:
        W: Weight matrix of shape (d, d)

    Returns:
        Scalar penalty (non-negative, zero for DAG)
    """
    d = W.shape[0]
    
    # Normalize to prevent overflow in matrix_exp
    max_val = W.abs().max()
    if max_val > 0:
        W_norm = W / max_val
    else:
        W_norm = W
    
    W_sq = W_norm * W_norm  # element-wise square
    
    # Use power series for stability and speed
    M = _matrix_exp_power_series(W_sq, terms=8)
    h = torch.trace(M) - d
    
    # Ensure non-negative (can be slightly negative due to numerical error)
    return torch.clamp(h, min=0.0)


def _matrix_exp_power_series(A: torch.Tensor, terms: int = 20) -> torch.Tensor:
    """Approximate matrix exponential using power series."""
    d = A.shape[0]
    result = torch.eye(d, device=A.device, dtype=A.dtype)
    term = torch.eye(d, device=A.device, dtype=A.dtype)
    for k in range(1, terms + 1):
        term = term @ A / k
        result = result + term
    return result


def dagness_gradient(W: torch.Tensor) -> torch.Tensor:
    """Gradient of h(W) wrt W.

    d(h)/dW = exp(W⊙W) ⊙ 2W

    Args:
        W: Weight matrix of shape (d, d)

    Returns:
        Gradient matrix of shape (d, d)
    """
    d = W.shape[0]
    W_sq = W * W
    M = torch.matrix_exp(W_sq)
    return M * 2 * W


def is_dag(W: np.ndarray, tol: float = 1e-6) -> bool:
    """Check if weighted adjacency matrix represents a DAG.

    Args:
        W: Weight matrix
        tol: Tolerance for zero

    Returns:
        True if the matrix represents a DAG
    """
    return np.isclose(dagness(torch.from_numpy(W).float()).item(), 0.0, atol=tol)


def threshold_probs(P: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Threshold edge probabilities to binary adjacency.

    Args:
        P: Edge probability matrix
        threshold: Probability threshold

    Returns:
        Binary adjacency matrix
    """
    return (P >= threshold).astype(float)


def count_parameters(W: np.ndarray) -> int:
    """Count edges in graph.

    Args:
        W: Adjacency or weight matrix

    Returns:
        Number of non-zero entries
    """
    return int(np.sum(np.abs(W) > 1e-6)) if W.dtype in [np.float64, np.float32] else int(np.sum(W > 0))


def structural_hamming_distance(W_true: np.ndarray, W_est: np.ndarray) -> float:
    """Compute Structural Hamming Distance (SHD).

    Args:
        W_true: True adjacency matrix
        W_est: Estimated adjacency matrix

    Returns:
        SHD score (lower is better)
    """
    diff = np.abs(W_true - W_est)
    return float(np.sum(diff > 0.5)) / 2  # undirected edges counted once


def edge_posterior_precision(W_true: np.ndarray, P_est: np.ndarray, threshold: float = 0.5) -> float:
    """Compute expected precision of edge predictions using posterior probabilities.

    Args:
        W_true: True binary adjacency matrix
        P_est: Estimated edge probabilities
        threshold: Decision threshold

    Returns:
        Precision score
    """
    W_bin = (P_est >= threshold).astype(float)
    tp = np.sum((W_bin > 0) & (W_true > 0))
    fp = np.sum((W_bin > 0) & (W_true == 0))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def edge_posterior_recall(W_true: np.ndarray, P_est: np.ndarray, threshold: float = 0.5) -> float:
    """Compute expected recall of edge predictions using posterior probabilities.

    Args:
        W_true: True binary adjacency matrix
        P_est: Estimated edge probabilities
        threshold: Decision threshold

    Returns:
        Recall score
    """
    W_bin = (P_est >= threshold).astype(float)
    tp = np.sum((W_bin > 0) & (W_true > 0))
    fn = np.sum((W_bin == 0) & (W_true > 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def expected_shd(W_true: np.ndarray, P_est: np.ndarray) -> float:
    """Expected Structural Hamming Distance using edge probabilities.

    Instead of thresholding, compute expected SHD under the posterior.

    Args:
        W_true: True binary adjacency matrix
        P_est: Estimated edge probabilities

    Returns:
        Expected SHD (lower is better)
    """
    d = W_true.shape[0]
    expected = 0.0
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            p = P_est[i, j]
            expected += p * (1 - W_true[i, j]) + (1 - p) * W_true[i, j]
    return float(expected)


def to_networkx(W: np.ndarray, probabilities: np.ndarray = None, threshold: float = 0.0) -> nx.DiGraph:
    """Convert adjacency matrix to NetworkX DiGraph.

    Args:
        W: Weight/adjacency matrix
        probabilities: Edge probabilities (optional, added as edge attribute)
        threshold: Minimum weight/probability to include edge

    Returns:
        NetworkX directed graph
    """
    G = nx.DiGraph()
    d = W.shape[0]
    G.add_nodes_from(range(d))
    for i in range(d):
        for j in range(d):
            if abs(W[i, j]) > threshold:
                G.add_edge(i, j, weight=float(W[i, j]))
                if probabilities is not None:
                    G[i][j]["probability"] = float(probabilities[i, j])
    return G
