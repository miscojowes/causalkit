#!/usr/bin/env python3
"""
Benchmark: CausalBayes on linear random DAGs of varying sizes.
Tests d=5, d=10, d=20 and records: SHD, AUC-PR, Expected SHD, ECE, Coverage, Time.
"""

import sys, os, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
warnings.filterwarnings("ignore")

np.random.seed(42)


def generate_linear_dag(d, n=1000, edge_prob=0.25, noise_scale=0.1, seed=None):
    rng = np.random.RandomState(seed if seed is not None else np.random.randint(0, 10000))
    W_true = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < edge_prob:
                W_true[i, j] = rng.uniform(0.5, 2.0) * rng.choice([-1, 1])
    X = np.zeros((n, d))
    for j in range(d):
        parents = np.where(W_true[:, j] != 0)[0]
        if len(parents) > 0:
            X[:, j] = X[:, parents] @ W_true[parents, j]
        X[:, j] += rng.randn(n) * noise_scale
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X, W_true


CONFIGS = {
    5:  {"hidden": [32, 32], "lr": 1e-2, "max_iter": 30, "mc": 20},
    10: {"hidden": [64, 64], "lr": 1e-3, "max_iter": 40, "mc": 20},
    20: {"hidden": [64, 64], "lr": 1e-3, "max_iter": 50, "mc": 15},
}


def run_benchmark(d, n_samples=1000):
    from causbayes import NeuralBayesianDAG
    from causbayes.evaluation import comprehensive_evaluation

    cfg = CONFIGS[d]
    X, W_true = generate_linear_dag(d, n=n_samples)

    t0 = time.time()
    model = NeuralBayesianDAG(
        hidden_layers=cfg["hidden"],
        learning_rate=cfg["lr"],
        lambda_1=1e-2,
        lambda_2=5.0,
        uncertainty="mc_dropout",
        mc_samples=cfg["mc"],
        max_iter=cfg["max_iter"],
        verbose=False,
    )
    model.fit(X)
    elapsed = time.time() - t0

    metrics = comprehensive_evaluation(W_true, model.edge_probs, model.edge_stds)
    metrics["time"] = round(elapsed, 1)

    return metrics


def print_table(results):
    header = f"{'d':<6} {'SHD':<8} {'Exp.SHD':<10} {'AUC-PR':<10} {'ECE':<8} {'Coverage':<10} {'Time(s)':<8}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for d in sorted(results.keys()):
        m = results[d]
        shd = m["shd"]
        eshd = m["expected_shd"]
        auc = m["auc_pr"]
        ece = m["ece"]
        cov = m.get("coverage@0.9", float("nan"))
        t = m["time"]
        print(f"{d:<6} {shd:<8.2f} {eshd:<10.2f} {auc:<10.3f} {ece:<8.4f} {cov:<10.3f} {t:<8.1f}")


if __name__ == "__main__":
    print("=" * 66)
    print("  CausalBayes Benchmark: Linear Random DAGs")
    print("=" * 66)

    results = {}
    for d in [5, 10, 20]:
        print(f"\n  d={d} ... ", end="", flush=True)
        try:
            m = run_benchmark(d)
            results[d] = m
            name = f"d={d}"
            print(f"SHD={m['shd']:.1f}, AUC-PR={m['auc_pr']:.3f}, "
                  f"ECE={m['ece']:.4f}, {m['time']}s")
        except Exception as e:
            print(f"FAILED: {e}")

    print("\n" + "=" * 66)
    print("  Results Summary")
    print("=" * 66)
    print_table(results)
    print("\n" + "=" * 66)
    print("  Done!")
