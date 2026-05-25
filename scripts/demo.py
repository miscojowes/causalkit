"""
Demo: Basic usage of CausalBayes on synthetic data.

This script demonstrates the core workflow:
1. Generate synthetic data from a known DAG
2. Learn the structure with uncertainty
3. Evaluate against ground truth
"""

import numpy as np
import matplotlib.pyplot as plt

# Register the causbayes package
import sys
sys.path.insert(0, "src")

from causbayes import NeuralBayesianDAG
from causbayes.structure_learning.utils import structural_hamming_distance
from causbayes.evaluation import comprehensive_evaluation


def generate_linear_dag(
    d: int = 10,
    n: int = 1000,
    edge_prob: float = 0.3,
    noise_scale: float = 0.1,
    seed: int = 42,
) -> tuple:
    """Generate data from a linear Gaussian DAG.

    Returns:
        (X, W_true) where X is data (n, d) and W_true is adjacency matrix
    """
    rng = np.random.RandomState(seed)

    # Generate random DAG
    W_true = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < edge_prob:
                W_true[i, j] = rng.uniform(0.5, 2.0) * rng.choice([-1, 1])

    # Generate data according to SEM
    X = np.zeros((n, d))
    for j in range(d):
        parents = np.where(W_true[:, j] != 0)[0]
        if len(parents) > 0:
            X[:, j] = X[:, parents] @ W_true[parents, j]
        X[:, j] += rng.randn(n) * noise_scale

    # Standardize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    return X, W_true


def generate_nonlinear_dag(
    d: int = 6,
    n: int = 2000,
    noise_scale: float = 0.1,
    seed: int = 42,
) -> tuple:
    """Generate data from a non-linear DAG (sine/cosine relationships).

    Returns:
        (X, W_true)
    """
    rng = np.random.RandomState(seed)

    W_true = np.zeros((d, d))
    # Create a chain structure
    for i in range(d - 1):
        W_true[i, i + 1] = 1.0

    X = np.zeros((n, d))
    X[:, 0] = rng.randn(n)

    for j in range(1, d):
        parents = np.where(W_true[:, j] != 0)[0]
        if len(parents) > 0:
            # Non-linear: sin/cosine of parents
            for p in parents:
                X[:, j] += np.sin(X[:, p]) + 0.5 * np.cos(X[:, p] * 2)
        else:
            X[:, j] = rng.randn(n)
        X[:, j] += rng.randn(n) * noise_scale

    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X, W_true


def demo_basic():
    """Run a basic demo of CausalBayes."""
    print("=" * 60)
    print("CausalBayes Demo: Bayesian Causal Discovery")
    print("=" * 60)

    # Generate data
    print("\n[1] Generating synthetic linear DAG data...")
    X, W_true = generate_linear_dag(d=8, n=1000, seed=42)
    print(f"    Data shape: {X.shape}")
    print(f"    True edges: {int(np.sum(W_true))}")

    # Learn structure
    print("\n[2] Learning causal structure with uncertainty...")
    model = NeuralBayesianDAG(
        hidden_layers=[64, 64],
        learning_rate=1e-3,
        lambda_1=1e-2,
        lambda_2=5.0,
        uncertainty="mc_dropout",
        mc_samples=30,
        max_iter=50,
        verbose=True,
        device="cpu",
    )

    model.fit(X)

    # Results
    print(f"\n[3] Results:")
    print(f"    Edge probs shape: {model.edge_probs.shape}")
    print(f"    Top-10 edges by probability:")
    for (i, j), prob, std in model.get_top_edges(10):
        print(f"      X{i} -> X{j}: P = {prob:.3f} ± {std:.3f}")

    # Evaluate
    print(f"\n[4] Evaluation against ground truth:")
    metrics = comprehensive_evaluation(W_true, model.edge_probs, model.edge_stds)
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"    {key}: {val:.4f}")

    # Visualize
    print("\n[5] Generating plot...")
    model.plot(threshold=0.3, show_uncertainty=True)
    print("    Done!")

    # Sample graphs
    print("\n[6] Sampling from posterior over graphs...")
    samples = model.sample_graphs(n_samples=10)
    print(f"    Sampled {len(samples)} graphs from posterior")

    return model, W_true


def demo_nonlinear():
    """Demo with non-linear data."""
    print("\n" + "=" * 60)
    print("Non-linear DAG Demo")
    print("=" * 60)

    X, W_true = generate_nonlinear_dag(d=6, n=2000)
    print(f"    Data shape: {X.shape}")
    print(f"    True edges: {int(np.sum(W_true))} (chain)")

    model = NeuralBayesianDAG(
        hidden_layers=[128, 128],
        learning_rate=5e-4,
        lambda_1=1e-2,
        lambda_2=10.0,
        uncertainty="mc_dropout",
        mc_samples=30,
        max_iter=80,
        verbose=True,
    )

    model.fit(X)

    print(f"\n    Top edges:")
    for (i, j), prob, std in model.get_top_edges(10):
        print(f"      X{i} -> X{j}: P = {prob:.3f} ± {std:.3f}")

    metrics = comprehensive_evaluation(W_true, model.edge_probs, model.edge_stds)
    print(f"\n    SHD: {metrics['shd']}, Expected SHD: {metrics['expected_shd']:.2f}")
    print(f"    AUC-PR: {metrics['auc_pr']:.3f}")
    print(f"    ECE: {metrics['ece']:.3f}")

    model.plot(threshold=0.3, show_uncertainty=True, title="Non-linear DAG Recovery")

    return model


if __name__ == "__main__":
    print("CausalBayes Demo")
    print("================")

    model, W_true = demo_basic()
    demo_nonlinear()

    print("\n✅ Demo complete!")
