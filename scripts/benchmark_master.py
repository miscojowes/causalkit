#!/usr/bin/env python3
"""
Master benchmark: CausalBayes vs gCastle baselines on linear data.

Tests: CausalBayes(Bootstrap+Platt), Single NOTEARS, gCastle Notears, PC, GES
Data: Linear Gaussian, d=5, n=1000, 5 seeds
Metrics: SHD, Precision, Recall, F1, AUC-PR, ECE, Time
"""

import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

SEEDS = [42, 43, 44, 45, 46]
RESULTS_FILE = "experiment_results/master_benchmark.json"
os.makedirs("experiment_results", exist_ok=True)


def generate_data(d, n, edge_prob=0.2, noise_scale=0.1, seed=42):
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


def evaluate(W_true, P, time_taken):
    from causbayes.structure_learning.notears_fast import (
        expected_calibration_error, brier_score
    )
    from causbayes.evaluation import comprehensive_evaluation

    metrics = comprehensive_evaluation(W_true, P)
    ece = expected_calibration_error(P, W_true)
    bs = brier_score(P, W_true)

    # F1 at threshold 0.5
    W_bin = (P >= 0.5).astype(float)
    tp = np.sum((W_bin > 0) & (W_true > 0))
    fp = np.sum((W_bin > 0) & (W_true == 0))
    fn = np.sum((W_bin == 0) & (W_true > 0))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "shd": metrics["shd"],
        "auc_pr": metrics["auc_pr"],
        "ece": ece,
        "brier": bs,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "n_edges": int(np.sum(W_bin)),
        "time": time_taken,
    }


def run_causbayes_bootstrap(X_tr, X_val, W_val, n_boot=50):
    from causbayes import BootstrapDAG
    t0 = time.time()
    model = BootstrapDAG(n_bootstraps=n_boot, max_iter=10, verbose=False,
                         calibrate=True)
    model.fit(X_tr, X_val=X_val, W_val=W_val)
    t = time.time() - t0
    return model.edge_probs.copy(), t


def run_single_notears(X, w_threshold=0.1):
    from causbayes.structure_learning.notears_fast import notears_lbfgs
    from sklearn.preprocessing import StandardScaler
    t0 = time.time()
    X_scaled = StandardScaler().fit_transform(X)
    W = notears_lbfgs(X_scaled, max_iter=10, w_threshold=w_threshold)
    t = time.time() - t0
    P = (np.abs(W) > 1e-4).astype(float)
    return P, t


def run_gcastle_notears(X):
    from castle.algorithms import Notears
    t0 = time.time()
    model = Notears()
    model.learn(X)
    t = time.time() - t0
    W = model.causal_matrix
    P = (np.abs(W) > 1e-4).astype(float)
    return P, t


def run_gcastle_pc(X):
    from castle.algorithms import PC
    t0 = time.time()
    model = PC()
    model.learn(X)
    t = time.time() - t0
    W = model.causal_matrix
    P = W.copy()
    return P, t


def run_gcastle_ges(X):
    from castle.algorithms import GES
    t0 = time.time()
    model = GES()
    model.learn(X)
    t = time.time() - t0
    W = model.causal_matrix
    P = W.copy()
    return P, t


METHODS = {
    "CausalBayes(Bootstrap50)": run_causbayes_bootstrap,
    "SingleNOTEARS": lambda X, Xv, Wv: run_single_notears(X),
    "gCastle-Notears": lambda X, Xv, Wv: run_gcastle_notears(X),
    "gCastle-PC": lambda X, Xv, Wv: run_gcastle_pc(X),
    "gCastle-GES": lambda X, Xv, Wv: run_gcastle_ges(X),
}


def main():
    print("=" * 80)
    print("  Master Benchmark: CausalBayes vs gCastle (Linear d=5)")
    print("=" * 80)

    all_results = {}

    for seed in SEEDS:
        print(f"\n  Seed {seed}")
        n = 1000
        d = 5
        n_tr, n_va = int(n * 0.6), int(n * 0.2)

        X_all, W_true = generate_data(d, n, seed=seed)
        X_tr = X_all[:n_tr]
        X_va = X_all[n_tr:n_tr + n_va]
        X_te = X_all[n_tr + n_va:]

        # Standardize
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_va = sc.transform(X_va)
        X_te = sc.transform(X_te)

        n_te = int(np.sum(W_true > 0))
        print(f"    True edges: {n_te}")

        for mname, mfunc in METHODS.items():
            try:
                if "Bootstrap" in mname:
                    P, t = mfunc(X_tr, X_va, W_true)
                else:
                    P, t = mfunc(X_tr, None, None)
                metrics = evaluate(W_true, P, t)
                all_results[f"d5_s{seed}_{mname}"] = {
                    "experiment": "linear_d5",
                    "seed": seed, "method": mname, **metrics
                }
                print(f"    {mname:<30} SHD={metrics['shd']:<3.0f} "
                      f"F1={metrics['f1']:<.3f} AUCPR={metrics['auc_pr']:<.3f} "
                      f"ECE={metrics['ece']:<.4f} t={metrics['time']:<5.1f}s")
            except Exception as e:
                print(f"    {mname:<30} ERROR: {e}")

    # Summary
    print(f"\n\n{'='*80}")
    print("  SUMMARY (mean ± std across seeds)")
    print(f"{'='*80}")
    print(f"  {'Method':<30} {'SHD':>6} {'F1':>6} {'AUCPR':>6} {'ECE':>6} {'Brier':>6} {'t(s)':>6}")
    print(f"  {'─'*30} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")

    from collections import defaultdict
    agg = defaultdict(list)
    for k, v in all_results.items():
        agg[v["method"]].append(v)

    for method, ml in sorted(agg.items()):
        vals = {k: np.mean([m[k] for m in ml]) for k in ["shd", "f1", "auc_pr", "ece", "brier", "time"]}
        stds = {k: np.std([m[k] for m in ml]) for k in ["shd", "f1", "auc_pr", "ece", "brier", "time"]}
        print(f"  {method:<30} {vals['shd']:5.1f}±{stds['shd']:.1f} "
              f"{vals['f1']:5.3f}±{stds['f1']:.3f} "
              f"{vals['auc_pr']:5.3f}±{stds['auc_pr']:.3f} "
              f"{vals['ece']:5.4f}±{stds['ece']:.4f} "
              f"{vals['brier']:5.4f}±{stds['brier']:.4f} "
              f"{vals['time']:5.1f}±{stds['time']:.1f}")

    # Save
    ser = {}
    for k, v in all_results.items():
        ser[k] = {kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv)
                  for kk, vv in v.items()}
    with open(RESULTS_FILE, "w") as f:
        json.dump(ser, f, indent=2)
    print(f"\n  Saved {RESULTS_FILE}")


if __name__ == "__main__":
    main()
