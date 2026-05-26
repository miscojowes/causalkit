"""
Evaluation metrics for causal discovery with uncertainty.

Provides both standard metrics (SHD, precision, recall) and
uncertainty-aware metrics (expected SHD, coverage, calibration).
"""

from typing import Optional

import numpy as np
from scipy.stats import entropy
from sklearn.metrics import precision_recall_curve, auc as sklearn_auc

from causbayes.structure_learning.utils import (
    structural_hamming_distance,
    expected_shd,
)


def precision_recall_auc(W_true: np.ndarray, P_est: np.ndarray) -> dict:
    """Compute precision-recall AUC using edge probabilities.

    Args:
        W_true: True binary adjacency matrix
        P_est: Estimated edge probabilities

    Returns:
        dict with 'auc_pr', 'precision', 'recall', 'thresholds'
    """
    d = W_true.shape[0]
    y_true = []
    y_score = []

    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            y_true.append(int(W_true[i, j] > 0.5))
            y_score.append(P_est[i, j])

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    auc_pr = sklearn_auc(recall, precision)

    return {
        "auc_pr": float(auc_pr),
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
    }


def edge_calibration(W_true: np.ndarray, P_est: np.ndarray, n_bins: int = 10) -> dict:
    """Compute calibration of edge probability estimates.

    Groups edges by predicted probability and measures
    actual frequency of edges in each bin.

    Args:
        W_true: True binary adjacency matrix
        P_est: Estimated edge probabilities
        n_bins: Number of probability bins

    Returns:
        dict with 'bins', 'accuracy', 'counts', 'ece'
    """
    d = W_true.shape[0]
    probs = []
    actual = []

    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            probs.append(P_est[i, j])
            actual.append(int(W_true[i, j] > 0.5))

    probs = np.array(probs)
    actual = np.array(actual)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    accuracies = []
    counts = []

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        count = mask.sum()
        counts.append(count)
        if count > 0:
            accuracies.append(actual[mask].mean())
        else:
            accuracies.append(0.0)

    # Expected Calibration Error
    ece = np.sum(
        np.array(counts) * np.abs(np.array(accuracies) - bin_centers)
    ) / max(np.sum(counts), 1)

    return {
        "bins": bin_centers.tolist(),
        "accuracy": accuracies,
        "counts": counts,
        "ece": float(ece),
    }


def edge_entropy(P: np.ndarray) -> np.ndarray:
    """Compute entropy of each edge probability.

    H(p) = -p log p - (1-p) log(1-p)
    High entropy (~0.69) means high uncertainty.
    Low entropy (~0) means high certainty.

    Args:
        P: Edge probability matrix of shape (d, d)

    Returns:
        Entropy matrix of shape (d, d)
    """
    eps = 1e-8
    return -(P * np.log(P + eps) + (1 - P) * np.log(1 - P + eps))


def uncertainty_coverage(W_true: np.ndarray, P_est: np.ndarray, P_std: np.ndarray, 
                          alpha: float = 0.1) -> float:
    """Coverage of uncertainty intervals.

    For each edge, compute whether the 1-alpha confidence interval
    contains the true value.

    Args:
        W_true: True binary adjacency matrix
        P_est: Predicted edge probabilities
        P_std: Predicted edge standard deviations
        alpha: Significance level (Default: 0.1 for 90% CI)

    Returns:
        Coverage proportion
    """
    z = 1.96 if alpha == 0.05 else 1.645  # 95% or 90% CI
    lower = np.clip(P_est - z * P_std, 0, 1)
    upper = np.clip(P_est + z * P_std, 0, 1)

    covered = (W_true >= lower) & (W_true <= upper)
    return float(covered.mean())


def comprehensive_evaluation(
    W_true: np.ndarray,
    P_est: np.ndarray,
    P_std: Optional[np.ndarray] = None,
) -> dict:
    """Comprehensive evaluation of causal discovery with uncertainty.

    Args:
        W_true: True binary adjacency matrix
        P_est: Estimated edge probabilities
        P_std: Estimated edge uncertainties (optional)

    Returns:
        dict with all metrics
    """
    results = {
        "shd": structural_hamming_distance(W_true, (P_est >= 0.5).astype(float)),
        "expected_shd": expected_shd(W_true, P_est),
        "auc_pr": precision_recall_auc(W_true, P_est)["auc_pr"],
    }

    # Precision/recall at threshold 0.5
    W_bin = (P_est >= 0.5).astype(float)
    tp = np.sum((W_bin > 0) & (W_true > 0))
    fp = np.sum((W_bin > 0) & (W_true == 0))
    fn = np.sum((W_bin == 0) & (W_true > 0))
    results["precision@0.5"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    results["recall@0.5"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    p = results["precision@0.5"]
    r = results["recall@0.5"]
    results["f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    # Edge entropy (average)
    H = edge_entropy(P_est)
    results["avg_edge_entropy"] = float(H.mean())

    # Calibration
    cal = edge_calibration(W_true, P_est)
    results["ece"] = cal["ece"]

    # Uncertainty coverage
    if P_std is not None:
        results["coverage@0.9"] = uncertainty_coverage(W_true, P_est, P_std, alpha=0.1)
        results["coverage@0.95"] = uncertainty_coverage(W_true, P_est, P_std, alpha=0.05)

    return results


def compare_with_baseline(
    W_true: np.ndarray,
    causbayes_P: np.ndarray,
    baselines: dict,
) -> dict:
    """Compare CausalBayes against other methods.

    Args:
        W_true: True binary adjacency matrix
        causbayes_P: Edge probabilities from CausalBayes
        baselines: Dict of name -> (adjacency_matrix, optional_edge_probs)

    Returns:
        dict with comparison results
    """
    results = {}

    for name, baseline in baselines.items():
        if len(baseline) == 2:
            W_est, P_est = baseline
        else:
            W_est = baseline
            P_est = baseline

        results[name] = {
            "shd": structural_hamming_distance(W_true, W_est),
            "precision@0.5": (
                tp / (tp + fp)
                if (tp := np.sum((W_est > 0) & (W_true > 0)))
                   and (fp := np.sum((W_est > 0) & (W_true == 0)))
                else 0.0
            ),
            "recall@0.5": (
                tp / (tp + fn)
                if (tp := np.sum((W_est > 0) & (W_true > 0)))
                   and (fn := np.sum((W_est == 0) & (W_true > 0)))
                else 0.0
            ),
        }

    # Add CausalBayes results
    results["CausalBayes"] = comprehensive_evaluation(W_true, causbayes_P)

    return results
