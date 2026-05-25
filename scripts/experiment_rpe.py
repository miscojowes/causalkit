#!/usr/bin/env python3
"""
Novel Experiment: Regularization Perturbation Ensemble (RPE)

Hypothesis: Bootstrap measures SAMPLING uncertainty (different data → different DAG),
but the real issue is STRUCTURAL uncertainty (same data → different DAGs in the
equivalence class fit equally well). 

By perturbing the regularization strength λ₁ instead of the data, we get a
distribution that captures structural uncertainty directly.

Method:
1. For the SAME data, run NOTEARS with λ₁ ∈ {0.001, 0.005, 0.01, 0.02, 0.05, 0.1}
2. Each λ₁ produces a different DAG (more λ₁ = sparser)
3. The distribution across λ₁ captures which edges are robust vs regularization-dependent
4. Compare with bootstrap distribution — is RPE wider? More meaningful?

Expected result: RPE should show graded probabilities (not 0/1) because
different regularization levels actually change which edges are found.
This directly addresses the "edges cluster at 0 or 1" problem.
"""

import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from collections import defaultdict

SEEDS = [42, 43, 44]
RESULTS_FILE = "experiment_results/rpe_vs_bootstrap.json"
os.makedirs("experiment_results", exist_ok=True)


def generate_linear_dag(d, n, edge_prob=0.2, noise_scale=0.1, seed=42):
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


def run_rpe_notears(X, lambdas=None):
    """Regularization Perturbation Ensemble.

    Run NOTEARS with different λ₁ values on the SAME data.
    Returns edge probabilities = proportion of λ values where edge is found.
    """
    from causbayes.structure_learning.notears_fast import notears_lbfgs
    from sklearn.preprocessing import StandardScaler

    if lambdas is None:
        lambdas = [0.001, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.1]

    X_scaled = StandardScaler().fit_transform(X)
    X_scaled = X_scaled - X_scaled.mean(axis=0, keepdims=True)
    d = X_scaled.shape[1]

    t0 = time.time()
    weights = []
    for lam in lambdas:
        try:
            W = notears_lbfgs(
                X_scaled, lambda_1=lam, max_iter=10, w_threshold=0.05,
                lbfgs_maxiter=25,
            )
            if not np.isnan(W).any():
                weights.append(W)
        except Exception:
            pass
    t = time.time() - t0

    if len(weights) == 0:
        return np.zeros((d, d)), 0.0

    W_stack = np.array(weights)
    W_abs = np.abs(W_stack)
    P = np.mean(W_abs > 1e-4, axis=0)
    np.fill_diagonal(P, 0.0)

    return P, t


def run_bootstrap_notears(X):
    """Standard bootstrap for comparison."""
    from causbayes.structure_learning.notears_fast import bootstrap_notears
    from sklearn.preprocessing import StandardScaler

    X_scaled = StandardScaler().fit_transform(X)
    P, S, W_list, W_abs = bootstrap_notears(
        X_scaled, n_bootstraps=50, max_iter=10,
        w_threshold=0.1, method="lbfgs", seed=42,
    )
    return P, 0.0  # time tracked externally


def run_single_notears(X):
    """Single NOTEARS run for baseline."""
    from causbayes.structure_learning.notears_fast import notears_lbfgs
    from sklearn.preprocessing import StandardScaler

    X_scaled = StandardScaler().fit_transform(X)
    X_scaled = X_scaled - X_scaled.mean(axis=0, keepdims=True)
    W = notears_lbfgs(X_scaled, max_iter=10, w_threshold=0.1)
    P = (np.abs(W) > 1e-4).astype(float)
    return P, 0.0


def compute_metrics(W_true, P):
    """Compute key metrics for comparison."""
    from causbayes.structure_learning.notears_fast import (
        expected_calibration_error, brier_score
    )

    ece = expected_calibration_error(P, W_true)
    bs = brier_score(P, W_true)

    W_bin = (P >= 0.5).astype(float)
    tp = np.sum((W_bin > 0) & (W_true > 0))
    fp = np.sum((W_bin > 0) & (W_true == 0))
    fn = np.sum((W_bin == 0) & (W_true > 0))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Edge entropy: measure of uncertainty spread
    eps = 1e-8
    H = -(P * np.log(P + eps) + (1 - P) * np.log(1 - P + eps))
    avg_entropy = float(np.mean(H))

    # How many edges have "intermediate" probability (not 0 or 1)?
    d_metric = P.shape[0]
    intermediate = np.sum((P > 0.05) & (P < 0.95))
    total_edges = d_metric * (d_metric - 1)

    return {
        "shd": float(np.sum(np.abs(W_true - W_bin)) / 2),
        "f1": f1, "precision": prec, "recall": rec,
        "ece": ece, "brier": bs,
        "avg_entropy": avg_entropy,
        "intermediate_edges": int(intermediate),
        "total_possible_edges": total_edges,
    }


def main():
    print("=" * 80)
    print("  Novel Experiment: RPE vs Bootstrap vs Single NOTEARS")
    print("  Hypothesis: Regularization perturbation captures structural uncertainty")
    print("=" * 80)

    all_results = {}
    d = 5

    for seed in SEEDS:
        n = 1000
        n_tr, n_va = int(n * 0.6), int(n * 0.2)

        X_all, W_true = generate_linear_dag(d, n, seed=seed)
        X_tr = X_all[:n_tr]
        X_va = X_all[n_tr:n_tr + n_va]
        X_te = X_all[n_tr + n_va:]

        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)

        print(f"\n  Seed {seed}: {int(np.sum(W_true>0))} true edges")

        # RPE: regularization perturbation (same data, different λ₁)
        P_rpe, t_rpe = run_rpe_notears(X_tr)
        m_rpe = compute_metrics(W_true, P_rpe)

        # Bootstrap (same data, bootstrap samples)
        P_boot, _ = run_bootstrap_notears(X_tr)
        m_boot = compute_metrics(W_true, P_boot)

        # Single NOTEARS
        P_single, _ = run_single_notears(X_tr)
        m_single = compute_metrics(W_true, P_single)

        print(f"  {'Method':<25} {'SHD':>4} {'F1':>5} {'ECE':>6} {'Entropy':>8} {'Intermed':>8}")
        print(f"  {'─'*25} {'─'*4} {'─'*5} {'─'*6} {'─'*8} {'─'*8}")
        print(f"  {'RPE':<25} {m_rpe['shd']:4.1f} {m_rpe['f1']:5.3f} "
              f"{m_rpe['ece']:6.4f} {m_rpe['avg_entropy']:8.4f} {m_rpe['intermediate_edges']:8d}")
        print(f"  {'Bootstrap(50)':<25} {m_boot['shd']:4.1f} {m_boot['f1']:5.3f} "
              f"{m_boot['ece']:6.4f} {m_boot['avg_entropy']:8.4f} {m_boot['intermediate_edges']:8d}")
        print(f"  {'Single NOTEARS':<25} {m_single['shd']:4.1f} {m_single['f1']:5.3f} "
              f"{m_single['ece']:6.4f} {m_single['avg_entropy']:8.4f} {m_single['intermediate_edges']:8d}")

        # Show edge probability distributions
        print(f"\n  --- Edge probability distribution ---")
        for method_name, P in [("RPE", P_rpe), ("Bootstrap", P_boot), ("Single", P_single)]:
            flat = P.flatten()
            flat = flat[flat > 0]
            print(f"  {method_name:<10}: mean={np.mean(flat):.3f} "
                  f"std={np.std(flat):.3f} "
                  f"frac[0.05-0.95]={np.mean((P>0.05)&(P<0.95)):.3f}")

        for method_name, m, P in [("RPE", m_rpe, P_rpe), ("Bootstrap", m_boot, P_boot), ("Single", m_single, P_single)]:
            all_results[f"d5_s{seed}_{method_name}"] = {
                "experiment": "rpe_vs_bootstrap", "seed": seed,
                "method": method_name, **m,
            }
            # Save full P matrix
            all_results[f"d5_s{seed}_{method_name}"]["P"] = P.tolist()

    # Aggregate summary
    agg = defaultdict(list)
    for k, v in all_results.items():
        agg[v["method"]].append(v)

    print(f"\n\n{'='*80}")
    print("  CROSS-SEED SUMMARY (mean ± std)")
    print(f"{'='*80}")
    print(f"  {'Method':<25} {'SHD':>8} {'F1':>7} {'ECE':>8} {'Entropy':>9} {'Intermed':>9}")
    print(f"  {'─'*25} {'─'*8} {'─'*7} {'─'*8} {'─'*9} {'─'*9}")

    for method, ml in sorted(agg.items()):
        vals = {k: np.mean([m[k] for m in ml]) for k in ["shd", "f1", "ece", "avg_entropy", "intermediate_edges"]}
        print(f"  {method:<25} {vals['shd']:5.1f}±{np.std([m['shd'] for m in ml]):.1f} "
              f"{vals['f1']:5.3f}±{np.std([m['f1'] for m in ml]):.3f} "
              f"{vals['ece']:7.4f}±{np.std([m['ece'] for m in ml]):.4f} "
              f"{vals['avg_entropy']:7.4f}±{np.std([m['avg_entropy'] for m in ml]):.4f} "
              f"{vals['intermediate_edges']:5.0f}±{np.std([m['intermediate_edges'] for m in ml]):.0f}")

    # Save
    ser = {}
    for k, v in all_results.items():
        ser[k] = {kk: vv for kk, vv in v.items() if kk != "P"}
    with open(RESULTS_FILE, "w") as f:
        json.dump(ser, f, indent=2)
    print(f"\n  Saved {RESULTS_FILE}")

    # Print key conclusion
    print(f"\n\n  🔑 KEY FINDING:")
    rpe_entropy = np.mean([m["avg_entropy"] for m in agg["RPE"]])
    boot_entropy = np.mean([m["avg_entropy"] for m in agg["Bootstrap"]])
    print(f"     RPE avg edge entropy:      {rpe_entropy:.4f}")
    print(f"     Bootstrap avg edge entropy: {boot_entropy:.4f}")
    if rpe_entropy > boot_entropy:
        print(f"     ✅ RPE produces MORE spread (higher entropy) than bootstrap")
        print(f"        → Better captures structural uncertainty!")
    else:
        print(f"     Bootstrap still has more spread than RPE")
        print(f"        → Need different perturbation strategy")


if __name__ == "__main__":
    main()
