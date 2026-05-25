#!/usr/bin/env python3
"""
Comprehensive multi-seed benchmark: BootstrapDAG vs Single NOTEARS vs Random.

Tests d=5 linear Gaussian data across 10 seeds.
Reports SHD, Precision, Recall, F1, AUC-PR, ECE, Brier Score.
"""

import sys, os, json, time, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from causbayes.evaluation import comprehensive_evaluation, edge_calibration, precision_recall_auc
from causbayes.structure_learning.utils import structural_hamming_distance, expected_shd
from causbayes.structure_learning.notears_fast import notears_lbfgs

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiment_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS = list(range(42, 52))  # 10 seeds: 42, 43, ..., 51


def generate_data(d=5, n=1000, edge_prob=0.2, noise_scale=0.1, seed=42):
    """Generate linear Gaussian SEM data from a random DAG."""
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
    return X, (np.abs(W_true) > 1e-6).astype(float)


def run_bootstrap(X_tr, X_val, W_val, n_bootstraps=50):
    """Run BootstrapDAG with validation-calibrated threshold."""
    from causbayes import BootstrapDAG
    t0 = time.time()
    model = BootstrapDAG(
        n_bootstraps=n_bootstraps,
        lambda_1=0.01,
        max_iter=60,
        lr=1e-2,
        verbose=False,
    )
    model.fit(X_tr, X_val=X_val, W_val=W_val)
    elapsed = time.time() - t0
    return {
        "P": model.edge_probs.copy(),
        "S": model.edge_stds.copy(),
        "W_bin": model.adjacency_matrix.copy(),
        "time_s": elapsed,
        "model": model,
    }


def run_single_notears(X, lambda_1=0.01, w_threshold=0.1):
    """Run single NOTEARS with L-BFGS-B (fast)."""
    from causbayes.structure_learning.notears_fast import notears_lbfgs
    t0 = time.time()
    X_c = X - X.mean(axis=0, keepdims=True)
    W_est = notears_lbfgs(
        X_c,
        lambda_1=lambda_1,
        max_iter=10,
        w_threshold=w_threshold,
        lbfgs_maxiter=20,
    )
    elapsed = time.time() - t0
    P = (np.abs(W_est) > 0).astype(float)
    return {
        "P": P,
        "S": np.zeros_like(W_est),
        "W_bin": P,
        "time_s": elapsed,
        "model": None,
    }


def run_random(X, rng_seed=42):
    """Random baseline: uniform random edge probabilities."""
    d = X.shape[1]
    t0 = time.time()
    rng = np.random.RandomState(rng_seed + 999)
    P = rng.uniform(0, 1, (d, d))
    np.fill_diagonal(P, 0.0)
    W_bin = (P >= 0.5).astype(float)
    elapsed = time.time() - t0
    return {
        "P": P,
        "S": np.zeros((d, d)),
        "W_bin": W_bin,
        "time_s": elapsed,
        "model": None,
    }


def compute_metrics(W_true, result):
    """Compute all evaluation metrics."""
    P = result["P"]
    S = result.get("S")
    W_bin = result["W_bin"]

    metrics = {}

    # Structural metrics
    metrics["shd"] = structural_hamming_distance(W_true, W_bin)
    metrics["expected_shd"] = expected_shd(W_true, P)

    # Classification metrics at threshold 0.5
    tp = np.sum((W_bin > 0) & (W_true > 0))
    fp = np.sum((W_bin > 0) & (W_true == 0))
    fn = np.sum((W_bin == 0) & (W_true > 0))
    tn = np.sum((W_bin == 0) & (W_true == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1"] = f1
    metrics["true_edges"] = int(np.sum(W_true > 0))
    metrics["est_edges"] = int(np.sum(W_bin > 0))

    # AUC-PR
    try:
        pr_metrics = precision_recall_auc(W_true, P)
        metrics["auc_pr"] = pr_metrics["auc_pr"]
    except Exception:
        metrics["auc_pr"] = float("nan")

    # ECE (Expected Calibration Error)
    try:
        cal = edge_calibration(W_true, P, n_bins=10)
        metrics["ece"] = cal["ece"]
    except Exception:
        metrics["ece"] = float("nan")

    # Brier Score
    d = W_true.shape[0]
    n_edges = d * (d - 1)
    brier = 0.0
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            brier += (P[i, j] - W_true[i, j]) ** 2
    metrics["brier_score"] = brier / n_edges

    # Time
    metrics["time_s"] = result["time_s"]

    return metrics


def main():
    print("=" * 80)
    print("  Comprehensive Benchmark: BootstrapDAG vs Single NOTEARS vs Random")
    print("  d=5, n=1000, Erdos-Renyi p=0.2, 10 seeds, train/val/test 60/20/20")
    print("=" * 80)

    all_results = []

    for seed in SEEDS:
        print(f"\n{'─' * 60}")
        print(f"  Seed {seed}")
        print(f"{'─' * 60}")

        # Generate data
        X_all, W_true = generate_data(d=5, n=1000, seed=seed)
        n = len(X_all)
        n_tr, n_va = int(n * 0.6), int(n * 0.2)
        n_te = n - n_tr - n_va

        # Use different seeds for splits to ensure independence
        X_tr, _ = generate_data(d=5, n=n_tr, seed=seed)
        X_va, _ = generate_data(d=5, n=n_va, seed=seed + 1000)
        X_te, _ = generate_data(d=5, n=n_te, seed=seed + 2000)

        # Standardize
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)
        X_te = scaler.transform(X_te)

        te = int(np.sum(W_true > 0))
        print(f"    True edges: {te}")

        row = {"seed": seed, "d": 5, "n_tr": n_tr, "n_va": n_va, "n_te": n_te,
               "true_edges": te}

        # ─── BootstrapDAG ────────────────────────────────────────
        print(f"\n    ⚡ Bootstrap(50)...", end=" ", flush=True)
        t0 = time.time()
        try:
            result = run_bootstrap(X_tr, X_va, W_true, n_bootstraps=50)
            metrics = compute_metrics(W_true, result)
            # Store weight matrices for potential calibration analysis
            if hasattr(result["model"], "_weight_matrices_") and result["model"]._weight_matrices_:
                row["_n_valid_bootstraps"] = len(result["model"]._weight_matrices_)
            row["Bootstrap(50)"] = metrics
            print(f"SHD={metrics['shd']:.1f} P={metrics['precision']:.2f} "
                  f"R={metrics['recall']:.2f} F1={metrics['f1']:.2f} "
                  f"AUC-PR={metrics['auc_pr']:.2f} ECE={metrics['ece']:.3f} "
                  f"t={metrics['time_s']:.1f}s")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback; traceback.print_exc()
            row["Bootstrap(50)"] = {"error": str(e)}

        # ─── Single NOTEARS ──────────────────────────────────────
        print(f"    ⚡ Single NOTEARS...", end=" ", flush=True)
        try:
            result_n = run_single_notears(X_tr)
            metrics = compute_metrics(W_true, result_n)
            row["Single NOTEARS"] = metrics
            print(f"SHD={metrics['shd']:.1f} P={metrics['precision']:.2f} "
                  f"R={metrics['recall']:.2f} F1={metrics['f1']:.2f} "
                  f"AUC-PR={metrics['auc_pr']:.2f} ECE={metrics['ece']:.3f} "
                  f"t={metrics['time_s']:.1f}s")
        except Exception as e:
            print(f"ERROR: {e}")
            row["Single NOTEARS"] = {"error": str(e)}

        # ─── Random Baseline ─────────────────────────────────────
        print(f"    ⚡ Random...", end=" ", flush=True)
        try:
            result_r = run_random(X_tr, rng_seed=seed)
            metrics = compute_metrics(W_true, result_r)
            row["Random"] = metrics
            print(f"SHD={metrics['shd']:.1f} P={metrics['precision']:.2f} "
                  f"R={metrics['recall']:.2f} F1={metrics['f1']:.2f} "
                  f"AUC-PR={metrics['auc_pr']:.2f} ECE={metrics['ece']:.3f} "
                  f"t={metrics['time_s']:.1f}s")
        except Exception as e:
            print(f"ERROR: {e}")
            row["Random"] = {"error": str(e)}

        all_results.append(row)

    # ═══════════════════════════════════════════════════════════════
    #  SUMMARY
    # ═══════════════════════════════════════════════════════════════

    print(f"\n\n{'=' * 90}")
    print("  SUMMARY (mean ± std across 10 seeds)")
    print(f"{'=' * 90}")

    methods = ["Bootstrap(50)", "Single NOTEARS", "Random"]
    metrics_names = ["shd", "precision", "recall", "f1", "auc_pr", "ece", "brier_score", "time_s"]

    header = f"  {'Method':<20} " + "".join(f"{m:<12}" for m in metrics_names)
    print(header)
    print(f"  {'─'*20} " + "".join("─"*12 for _ in metrics_names))

    summary = {}
    for method in methods:
        vals = {}
        for mname in metrics_names:
            mvals = []
            for row in all_results:
                r = row.get(method, {})
                if isinstance(r, dict) and mname in r and r[mname] is not None and not (isinstance(r[mname], float) and np.isnan(r[mname])):
                    mvals.append(r[mname])
            if mvals:
                mean_v = np.mean(mvals)
                std_v = np.std(mvals)
                vals[mname] = (mean_v, std_v)
            else:
                vals[mname] = (float("nan"), float("nan"))

        summary[method] = vals
        line = f"  {method:<20} "
        for mname in metrics_names:
            mean_v, std_v = vals[mname]
            if mname == "ece" or mname == "brier_score":
                line += f"{mean_v:.4f}±{std_v:.4f}  "
            elif mname == "time_s":
                line += f"{mean_v:.1f}±{std_v:.1f}  "
            else:
                line += f"{mean_v:.2f}±{std_v:.2f}  "
        print(line)

    # ═══════════════════════════════════════════════════════════════
    #  SAVE
    # ═══════════════════════════════════════════════════════════════

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output = {
        "metadata": {
            "description": "Comprehensive d=5 linear Gaussian benchmark",
            "n_seeds": len(SEEDS),
            "seeds": SEEDS,
            "d": 5,
            "n": 1000,
            "data_config": {"edge_prob": 0.2, "noise_scale": 0.1, "split": "60/20/20"},
            "methods": methods,
        },
        "by_seed": all_results,
        "summary": {},
    }

    for method in methods:
        output["summary"][method] = {}
        for mname in metrics_names:
            if mname in summary[method]:
                mean_v, std_v = summary[method][mname]
                output["summary"][method][f"{mname}_mean"] = convert(mean_v)
                output["summary"][method][f"{mname}_std"] = convert(std_v)

    outpath = os.path.join(RESULTS_DIR, "comprehensive_d5.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=convert)
    print(f"\n  Results saved to {outpath}")

    # ECE comparison
    print(f"\n{'=' * 60}")
    print("  ECE DETAIL (calibration)")
    print(f"{'=' * 60}")
    for method in methods:
        vals = summary[method]
        ece_mean, ece_std = vals.get("ece", (float("nan"), float("nan")))
        print(f"  {method:<20} ECE = {ece_mean:.4f} ± {ece_std:.4f}  "
              f"{'✅' if ece_mean < 0.1 else '❌'} target < 0.1")

    print(f"\n{'=' * 90}")
    print("  BENCHMARK COMPLETE")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
