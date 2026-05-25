#!/usr/bin/env python3
"""
CausalBayes Systematic Benchmark

Tests CausalBayes against causal-learn baselines on synthetic data.
Reports SHD, Expected SHD, AUC-PR, Precision, Recall, ECE, Time.
Saves results to JSON for later analysis.
"""

import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "benchmark_results.json")


# ═══════════════════════════════════════════════════════════════════════
#  DATA GENERATORS
# ═══════════════════════════════════════════════════════════════════════

def generate_linear_dag(d, n, edge_prob=0.2, noise_scale=0.1, seed=None):
    """Linear Gaussian SEM with random DAG."""
    rng = np.random.RandomState(seed)
    W_true = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < edge_prob:
                W_true[i, j] = rng.uniform(0.5, 1.5) * rng.choice([-1, 1])
    X = np.zeros((n, d))
    for j in range(d):
        parents = np.where(W_true[:, j] != 0)[0]
        if len(parents) > 0:
            X[:, j] = X[:, parents] @ W_true[parents, j]
        X[:, j] += rng.randn(n) * noise_scale
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X, (np.abs(W_true) > 1e-6).astype(float)


def generate_nonlinear_dag(d, n, noise_scale=0.15, seed=None):
    """Non-linear chain DAG with sin/cos functions."""
    rng = np.random.RandomState(seed)
    W_true = np.zeros((d, d))
    for i in range(d - 1):
        W_true[i, i + 1] = 1.0
    X = np.zeros((n, d))
    X[:, 0] = rng.randn(n)
    for j in range(1, d):
        parents = np.where(W_true[:, j] != 0)[0]
        if len(parents) > 0:
            for p in parents:
                X[:, j] += np.sin(X[:, p]) + 0.3 * np.cos(2 * X[:, p])
        X[:, j] += rng.randn(n) * noise_scale
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X, W_true


def generate_scale_free_dag(d, n, noise_scale=0.1, seed=None):
    """Scale-free (power law) DAG using preferential attachment."""
    rng = np.random.RandomState(seed)
    W_true = np.zeros((d, d))
    # Start with 2 connected nodes
    in_degrees = np.zeros(d)
    for i in range(1, min(3, d)):
        W_true[0, i] = rng.uniform(0.5, 1.0)
        in_degrees[i] += 1
    # Preferential attachment for remaining nodes
    for j in range(3, d):
        # Probability proportional to in-degree
        probs = in_degrees[:j] / max(in_degrees[:j].sum(), 1)
        n_parents = max(1, rng.randint(1, 4))
        parents = rng.choice(j, size=min(n_parents, j), p=probs, replace=False)
        for p in parents:
            W_true[p, j] = rng.uniform(0.5, 1.0)
            in_degrees[j] += 1
    X = np.zeros((n, d))
    for j in range(d):
        parents = np.where(W_true[:, j] != 0)[0]
        if len(parents) > 0:
            X[:, j] = X[:, parents] @ W_true[parents, j]
        X[:, j] += rng.randn(n) * noise_scale
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X, W_true


# ═══════════════════════════════════════════════════════════════════════
#  METHODS
# ═══════════════════════════════════════════════════════════════════════

def run_causbayes(X, W_true, label="CausalBayes", **kwargs):
    """Run CausalBayes with gradient-based weight computation for better sparsity."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from causbayes import NeuralBayesianDAG
    from causbayes.models.nonlinear_sem import NonlinearSEM
    
    # Monkey-patch compute_weight_matrix to use gradient-based method
    original_cwm = NonlinearSEM.compute_weight_matrix
    def patched_cwm(self):
        return self.compute_weight_matrix_with_grad(X_global)
    
    d = X.shape[1]
    global X_global
    X_global = torch.from_numpy(X[:100]).float()  # Use subset for grad computation

    t0 = time.time()
    NonlinearSEM.compute_weight_matrix = patched_cwm
    
    model = NeuralBayesianDAG(
        hidden_layers=[32, 32],
        learning_rate=1e-2,
        lambda_1=5e-3,  # Stronger L1 for sparsity
        lambda_2=1.0,
        uncertainty="mc_dropout",
        mc_samples=15,
        max_iter=25,
        verbose=False,
        **kwargs
    )
    model.fit(X)
    NonlinearSEM.compute_weight_matrix = original_cwm
    elapsed = time.time() - t0

    # Adaptive threshold: use mean prob + 1 std as threshold
    P = model.edge_probs.copy()
    np.fill_diagonal(P, 0)
    # Find a natural threshold from the distribution of non-zero probs
    nonzero = P[P > 0.05]
    threshold = max(0.3, nonzero.mean() if len(nonzero) > 0 else 0.5)
    
    return {
        "name": label,
        "P": P,
        "std": model.edge_stds,
        "W_bin": (P >= threshold).astype(float),
        "time": elapsed,
        "converged": True,
    }


def run_causbayes_linear(X, W_true, label="CausalBayes-Linear"):
    """CausalBayes with no uncertainty, simpler setup for speed."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from causbayes import NeuralBayesianDAG
    d = X.shape[1]

    t0 = time.time()
    model = NeuralBayesianDAG(
        hidden_layers=[8],  # Smaller net for linear data
        learning_rate=1e-2,
        lambda_1=5e-3,
        lambda_2=2.0,
        uncertainty=None,
        max_iter=20,
        verbose=False,
    )
    model.fit(X)
    elapsed = time.time() - t0

    # Convert raw weights to pseudo-probabilities via soft-threshold
    W_abs = np.abs(model.W_est_)
    W_max = max(W_abs.max(), 1e-8)
    P = W_abs / W_max

    return {
        "name": label,
        "P": P,
        "std": np.zeros_like(P),
        "W_bin": (P >= 0.1).astype(float),
        "time": elapsed,
        "converged": model._training_losses_[-1] < 1.0 if model._training_losses_ else False,
    }


def run_pc(X, W_true, label="PC (causal-learn)"):
    """Run Peter-Clark algorithm from causal-learn."""
    from causallearn.search.ConstraintBased.PC import pc
    t0 = time.time()
    try:
        pc_graph = pc(X, 0.05, "fisherz")
        elapsed = time.time() - t0
        d = X.shape[1]
        W_est = np.zeros((d, d))
        # causal-learn 0.1.x uses graph as adjacency matrix
        graph_mat = pc_graph.G.graph  # (d, d) matrix
        for i in range(d):
            for j in range(d):
                if graph_mat[i, j] == 1 and graph_mat[j, i] == -1:
                    W_est[i, j] = 1.0
        return {"name": label, "P": W_est, "std": np.zeros((d, d)),
                "W_bin": W_est, "time": elapsed}
    except Exception as e:
        print(f"  {label}: ERROR {e}")
        return None


def run_ges(X, W_true, label="GES (causal-learn)"):
    """Greedy Equivalence Search from causal-learn."""
    from causallearn.search.ScoreBased.GES import ges
    t0 = time.time()
    try:
        ges_result = ges(X, score_func="local_score_BIC")
        elapsed = time.time() - t0
        d = X.shape[1]
        W_est = np.zeros((d, d))
        graph_mat = ges_result['G'].graph
        for i in range(d):
            for j in range(d):
                if graph_mat[i, j, 0] == 1 and graph_mat[j, i, 0] == -1:
                    W_est[i, j] = 1.0
        return {"name": label, "P": W_est, "std": np.zeros((d, d)),
                "W_bin": W_est, "time": elapsed}
    except Exception as e:
        print(f"  {label}: ERROR {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def evaluate(W_true, result):
    """Score a result against ground truth."""
    if result is None:
        return None

    from causbayes.evaluation import comprehensive_evaluation

    d = W_true.shape[0]
    P = result["P"]
    std = result.get("std")

    try:
        metrics = comprehensive_evaluation(W_true, P, std)
    except Exception:
        metrics = {}

    # Binary metrics
    W_bin = result["W_bin"]
    tp = np.sum((W_bin > 0) & (W_true > 0))
    fp = np.sum((W_bin > 0) & (W_true == 0))
    fn = np.sum((W_bin == 0) & (W_true > 0))
    tn = np.sum((W_bin == 0) & (W_true == 0))

    # Edge count sanity
    true_edges = int(np.sum(W_true > 0))

    return {
        "shd": metrics.get("shd", float("nan")),
        "expected_shd": metrics.get("expected_shd", float("nan")),
        "auc_pr": metrics.get("auc_pr", float("nan")),
        "precision": tp / (tp + fp) if (tp + fp) > 0 else float("nan"),
        "recall": tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
        "ece": metrics.get("ece", float("nan")),
        "true_edges": true_edges,
        "est_edges": int(np.sum(W_bin > 0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "time_s": result["time"],
    }


# ═══════════════════════════════════════════════════════════════════════
#  EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════

EXPERIMENTS = [
    # (name, generator, d, n, generator_kwargs)
    ("linear_small", generate_linear_dag, 5, 1000, {"edge_prob": 0.3, "noise_scale": 0.1}),
    ("linear_medium", generate_linear_dag, 10, 2000, {"edge_prob": 0.2, "noise_scale": 0.1}),
    ("linear_large", generate_linear_dag, 15, 3000, {"edge_prob": 0.15, "noise_scale": 0.15}),
    ("nonlinear_chain_small", generate_nonlinear_dag, 5, 1500, {}),
    ("nonlinear_chain_medium", generate_nonlinear_dag, 8, 2000, {}),
    ("scale_free_small", generate_scale_free_dag, 6, 1500, {}),
    ("scale_free_medium", generate_scale_free_dag, 10, 2500, {}),
]

METHODS = [
    run_causbayes,
    run_pc,
    run_ges,
]


def main():
    print("=" * 80)
    print("  CausalBayes Systematic Benchmark")
    print("=" * 80)
    print(f"\n  Experiments: {len(EXPERIMENTS)}")
    print(f"  Methods: {[m.__name__ for m in METHODS]}")
    print()

    all_results = {}

    for exp_name, generator, d, n, gen_kwargs in EXPERIMENTS:
        print(f"\n{'─' * 70}")
        print(f"  Experiment: {exp_name} (d={d}, n={n})")
        print(f"{'─' * 70}")

        # Generate data (3 seeds per experiment for robustness)
        all_seed_results = []
        for seed in [42, 123, 256]:
            X, W_true = generator(d, n, seed=seed, **gen_kwargs)
            true_edges = int(np.sum(W_true > 0))
            print(f"    Seed {seed}: {true_edges} true edges")

            row = {"d": d, "n": n, "seed": seed, "true_edges": true_edges}

            for method in METHODS:
                try:
                    result = method(X, W_true)
                    if result is not None:
                        metrics = evaluate(W_true, result)
                        row[result["name"]] = metrics
                        print(f"      {result['name']:<25} SHD={metrics['shd']:.1f} "
                              f"P={metrics['precision']:.2f} R={metrics['recall']:.2f} "
                              f"t={metrics['time_s']:.1f}s")
                except Exception as e:
                    print(f"      {method.__name__:<25} ERROR: {e}")

            all_seed_results.append(row)

        all_results[exp_name] = all_seed_results

    # ─── Summary ────────────────────────────────────────────────────
    print(f"\n\n{'=' * 80}")
    print("  SUMMARY")
    print(f"{'=' * 80}")
    print(f"\n  {'Experiment':<25} {'Method':<22} {'SHD':<6} {'P':<6} {'R':<6} {'AUCPR':<6} {'ECE':<6} {'Time':<6}")
    print(f"  {'─'*25} {'─'*22} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")

    for exp_name, seed_results in all_results.items():
        # Aggregate across seeds
        method_metrics = {}
        for row in seed_results:
            for k, v in row.items():
                if isinstance(v, dict) and "shd" in v:
                    if k not in method_metrics:
                        method_metrics[k] = []
                    method_metrics[k].append(v)

        for method_name, metrics_list in method_metrics.items():
            shd_avg = np.mean([m["shd"] for m in metrics_list])
            pr_avg = np.mean([m["precision"] for m in metrics_list])
            rec_avg = np.mean([m["recall"] for m in metrics_list])
            auc_avg = np.mean([m["auc_pr"] for m in metrics_list if not np.isnan(m["auc_pr"])])
            ece_avg = np.mean([m["ece"] for m in metrics_list if not np.isnan(m["ece"])])
            time_avg = np.mean([m["time_s"] for m in metrics_list])
            print(f"  {exp_name:<25} {method_name:<22} {shd_avg:<6.1f} {pr_avg:<6.2f} "
                  f"{rec_avg:<6.2f} {auc_avg:<6.2f} {ece_avg:<6.3f} {time_avg:<6.1f}")

    # Save all results
    serializable = {}
    for exp_name, seed_results in all_results.items():
        serializable[exp_name] = []
        for row in seed_results:
            clean = {}
            for k, v in row.items():
                if isinstance(v, dict):
                    clean[k] = {kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv)
                                for kk, vv in v.items() if not (isinstance(vv, float) and np.isnan(vv))}
                else:
                    clean[k] = int(v) if isinstance(v, (np.integer, int)) else v
            serializable[exp_name].append(clean)

    with open(RESULTS_FILE, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved to {RESULTS_FILE}")

    print(f"\n{'=' * 80}")
    print("  BENCHMARK COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
