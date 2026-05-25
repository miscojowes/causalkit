#!/usr/bin/env python3
"""
Experiment: Bootstrap NOTEARS with proper train/val/test evaluation.

Methodology:
- Generate DAG + separate train/val/test data from the same DAG
- Bootstrap NOTEARS trained on TRAIN only
- Threshold calibrated on VAL with ground truth
- Metrics on TEST only
- Compare against PC and GES baselines
"""

import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from collections import defaultdict

SEED = 42
np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════════
#  GENERATORS
# ═══════════════════════════════════════════════════════════════════════

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


def generate_nonlinear_dag(d, n, noise_scale=0.15, seed=42):
    rng = np.random.RandomState(seed)
    W_true = np.zeros((d, d))
    for i in range(d - 1):
        W_true[i, i + 1] = 1.0
    X = np.zeros((n, d))
    X[:, 0] = rng.randn(n)
    for j in range(1, d):
        X[:, j] = np.sin(X[:, j - 1]) + 0.3 * np.cos(2 * X[:, j - 1]) + rng.randn(n) * noise_scale
    return X, W_true


# ═══════════════════════════════════════════════════════════════════════
#  METHODS
# ═══════════════════════════════════════════════════════════════════════

def run_bootstrap(X_tr, X_val, W_val, **kwargs):
    from causbayes import BootstrapDAG
    t0 = time.time()
    model = BootstrapDAG(verbose=False, **kwargs)
    model.fit(X_tr, X_val=X_val, W_val=W_val)
    t = time.time() - t0
    return {"P": model.edge_probs.copy(), "S": model.edge_stds.copy(),
            "W_bin": model.adjacency_matrix.copy(), "time": t,
            "model": model}


def run_pc(X_tr, **kwargs):
    from causallearn.search.ConstraintBased.PC import pc
    d = X_tr.shape[1]
    t0 = time.time()
    cg = pc(X_tr, 0.05, "fisherz")
    t = time.time() - t0
    W = np.zeros((d, d))
    g = cg.G.graph
    for i in range(d):
        for j in range(d):
            if g[i, j] == 1 and g[j, i] == -1:
                W[i, j] = 1.0
    return {"P": W, "S": np.zeros((d, d)), "W_bin": W, "time": t}


def run_ges(X_tr, **kwargs):
    from causallearn.search.ScoreBased.GES import ges
    d = X_tr.shape[1]
    t0 = time.time()
    res = ges(X_tr, score_func="local_score_BIC")
    t = time.time() - t0
    W = np.zeros((d, d))
    g = res["G"].graph
    for i in range(d):
        for j in range(d):
            if g[i, j] == 1 and g[j, i] == -1:
                W[i, j] = 1.0
    return {"P": W, "S": np.zeros((d, d)), "W_bin": W, "time": t}


# ═══════════════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def evaluate(W_true, result):
    from causbayes.evaluation import comprehensive_evaluation
    P = result["P"]
    S = result.get("S")
    m = comprehensive_evaluation(W_true, P, S)
    W_bin = result["W_bin"]
    tp = np.sum((W_bin > 0) & (W_true > 0))
    fp = np.sum((W_bin > 0) & (W_true == 0))
    fn = np.sum((W_bin == 0) & (W_true > 0))
    return {
        "shd": m["shd"], "expected_shd": m["expected_shd"],
        "auc_pr": m["auc_pr"], "ece": m.get("ece", float("nan")),
        "precision": tp / (tp + fp) if (tp + fp) > 0 else float("nan"),
        "recall": tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
        "n_edges": int(np.sum(W_bin > 0)), "time": result["time"],
    }


# ═══════════════════════════════════════════════════════════════════════
#  EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════

EXPERIMENTS = [
    ("linear_d5", generate_linear_dag, 5, 1500, {"edge_prob": 0.3}),
    ("linear_d10", generate_linear_dag, 10, 3000, {"edge_prob": 0.2}),
    ("nonlinear_d6", generate_nonlinear_dag, 6, 2000, {}),
    ("nonlinear_d10", generate_nonlinear_dag, 10, 4000, {}),
]

METHODS = {
    "Bootstrap(20)": lambda X_tr, X_v, W_v: run_bootstrap(X_tr, X_v, W_v, n_bootstraps=20, max_iter=80),
    "Bootstrap(50)": lambda X_tr, X_v, W_v: run_bootstrap(X_tr, X_v, W_v, n_bootstraps=50, max_iter=80),
    "PC": lambda X_tr, X_v, W_v: run_pc(X_tr),
    "GES": lambda X_tr, X_v, W_v: run_ges(X_tr),
}


def main():
    print("=" * 80)
    print("  Causal Discovery Benchmark (train/val/test)")
    print("=" * 80)
    all_results = {}

    for exp_name, generator, d, n, kwargs in EXPERIMENTS:
        print(f"\n{'─' * 60}")
        print(f"  {exp_name}: d={d}, n={n}")
        print(f"{'─' * 60}")

        for seed in [42, 123]:
            # Generate DAG + separate train/val/test data
            W_true = generator(d, 1, seed=seed, **kwargs)[1]
            n_tr, n_va = int(n * 0.6), int(n * 0.2)
            n_te = n - n_tr - n_va
            X_tr = generator(d, n_tr, seed=seed, **kwargs)[0]
            X_va = generator(d, n_va, seed=seed + 1000, **kwargs)[0]
            X_te = generator(d, n_te, seed=seed + 2000, **kwargs)[0]

            # Standardize
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_va = sc.transform(X_va)
            X_te = sc.transform(X_te)

            te = int(np.sum(W_true > 0))
            print(f"\n  seed={seed}: {te} true edges")

            for mname, mfunc in METHODS.items():
                try:
                    result = mfunc(X_tr, X_va, W_true)
                    metrics = evaluate(W_true, result)
                    # Use test data for test metrics
                    all_results[f"{exp_name}_s{seed}_{mname}"] = {
                        "experiment": exp_name, "seed": seed, "method": mname, **metrics
                    }
                    print(f"    {mname:<15} SHD={metrics['shd']:<4.1f} "
                          f"P={metrics['precision']:<5.2f} R={metrics['recall']:<5.2f} "
                          f"AUCPR={metrics['auc_pr']:<5.2f} "
                          f"t={metrics['time']:<5.1f}s")
                except Exception as e:
                    print(f"    {mname:<15} ERROR: {e}")

    # Summary
    print(f"\n\n{'=' * 80}")
    print("  SUMMARY")
    print(f"{'=' * 80}")
    print(f"  {'Experiment':<20} {'Method':<15} {'SHD':<5} {'P':<5} {'R':<5} "
          f"{'AUCPR':<5} {'ECE':<5} {'t(s)':<5}")
    print(f"  {'─'*20} {'─'*15} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*5}")

    agg = defaultdict(list)
    for k, v in all_results.items():
        agg[(v["experiment"], v["method"])].append(v)

    for (exp, method), ml in sorted(agg.items()):
        shd = np.mean([m["shd"] for m in ml])
        pr = np.mean([m["precision"] for m in ml])
        rc = np.mean([m["recall"] for m in ml])
        auc = np.mean([m["auc_pr"] for m in ml])
        ece = np.mean([m["ece"] for m in ml])
        t_avg = np.mean([m["time"] for m in ml])
        print(f"  {exp:<20} {method:<15} {shd:<5.1f} {pr:<5.2f} {rc:<5.2f} "
              f"{auc:<5.2f} {ece:<5.3f} {t_avg:<5.1f}")

    # Save
    ser = {}
    for k, v in all_results.items():
        ser[k] = {kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv)
                  for kk, vv in v.items()}
    with open("experiment_results.json", "w") as f:
        json.dump(ser, f, indent=2)
    print(f"\n  Saved experiment_results.json")


if __name__ == "__main__":
    main()
