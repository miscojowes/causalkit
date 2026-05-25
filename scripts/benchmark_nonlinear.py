#!/usr/bin/env python3
"""
Non-linear benchmark: CausalBayes (Bootstrap) vs gCastle baselines.

Non-linear additive noise data where PC should fail.
Tests: CausalBayes(Bootstrap50+Platt), Single NOTEARS (linear),
       gCastle NotearsNonlinear, GraNDAG, PC, GES
"""

import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

SEEDS = [42, 43, 44, 45, 46]
RESULTS_FILE = "experiment_results/benchmark_nonlinear.json"
os.makedirs("experiment_results", exist_ok=True)


def generate_nonlinear_data(d, n, noise_scale=0.2, seed=42):
    """Generate non-linear additive noise data from a chain DAG.

    X_0 → X_1 → X_2 → ... → X_{d-1}
    Each edge is non-linear: X_j = f_j(X_{j-1}) + noise
    """
    rng = np.random.RandomState(seed)
    W_true = np.zeros((d, d))
    for i in range(d - 1):
        W_true[i, i + 1] = 1.0

    X = np.zeros((n, d))
    X[:, 0] = rng.randn(n)

    for j in range(1, d):
        # Non-linear functions: sin, cos, tanh, abs, sign
        f_idx = (j - 1) % 5
        if f_idx == 0:
            X[:, j] = np.sin(X[:, j - 1])
        elif f_idx == 1:
            X[:, j] = np.tanh(X[:, j - 1]) * 0.8
        elif f_idx == 2:
            X[:, j] = 0.5 * X[:, j - 1] + 0.5 * np.sin(2 * X[:, j - 1])
        elif f_idx == 3:
            X[:, j] = np.sign(X[:, j - 1]) * np.sqrt(np.abs(X[:, j - 1]))
        else:
            X[:, j] = np.cos(X[:, j - 1]) * 0.5 + np.sin(X[:, j - 1]) * 0.5
        X[:, j] += rng.randn(n) * noise_scale

    return X, W_true


def evaluate(W_true, P, time_taken):
    from causbayes.structure_learning.notears_fast import (
        expected_calibration_error, brier_score
    )
    from causbayes.evaluation import comprehensive_evaluation

    metrics = comprehensive_evaluation(W_true, P)
    ece = expected_calibration_error(P, W_true)
    bs = brier_score(P, W_true)

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


def run_causbayes_bootstrap(X_tr, X_val, W_val):
    from causbayes import BootstrapDAG
    t0 = time.time()
    model = BootstrapDAG(n_bootstraps=50, max_iter=10, verbose=False, calibrate=True)
    model.fit(X_tr, X_val=X_val, W_val=W_val)
    t = time.time() - t0
    return model.edge_probs.copy(), t


def run_single_notears_linear(X):
    from causbayes.structure_learning.notears_fast import notears_lbfgs
    from sklearn.preprocessing import StandardScaler
    t0 = time.time()
    X_scaled = StandardScaler().fit_transform(X)
    W = notears_lbfgs(X_scaled, max_iter=10, w_threshold=0.1)
    t = time.time() - t0
    P = (np.abs(W) > 1e-4).astype(float)
    return P, t


def run_gcastle_notears_nonlinear(X):
    from castle.algorithms import NotearsNonlinear
    t0 = time.time()
    model = NotearsNonlinear()
    model.learn(X)
    t = time.time() - t0
    W = model.causal_matrix
    P = (np.abs(W) > 1e-4).astype(float)
    return P, t


def run_gcastle_grandag(X):
    from castle.algorithms import GraNDAG
    t0 = time.time()
    model = GraNDAG(input_dim=X.shape[1])
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
    return model.causal_matrix.copy(), t


def run_gcastle_ges(X):
    from castle.algorithms import GES
    t0 = time.time()
    model = GES()
    model.learn(X)
    t = time.time() - t0
    return model.causal_matrix.copy(), t


METHODS = {
    "CausalBayes(Bootstrap50)": run_causbayes_bootstrap,
    "SingleNOTEARS-Linear": lambda X, Xv, Wv: run_single_notears_linear(X),
    "gCastle-NotearsNonlinear": lambda X, Xv, Wv: run_gcastle_notears_nonlinear(X),
    "gCastle-GraNDAG": lambda X, Xv, Wv: run_gcastle_grandag(X),
    "gCastle-PC": lambda X, Xv, Wv: run_gcastle_pc(X),
    "gCastle-GES": lambda X, Xv, Wv: run_gcastle_ges(X),
}


def main():
    print("=" * 80)
    print("  Non-linear Benchmark: Chain DAG (d=6), Additive Noise")
    print("  PC should fail here — CI tests assume linearity")
    print("=" * 80)

    all_results = {}

    for seed in SEEDS:
        print(f"\n  Seed {seed}")
        d, n = 6, 2000
        n_tr, n_va = int(n * 0.6), int(n * 0.2)

        X_all, W_true = generate_nonlinear_data(d, n, seed=seed)
        X_tr = X_all[:n_tr]
        X_va = X_all[n_tr:n_tr + n_va]
        X_te = X_all[n_tr + n_va:]

        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_va = sc.transform(X_va)
        X_te = sc.transform(X_te)

        print(f"    True edges: {int(np.sum(W_true>0))}")

        for mname, mfunc in METHODS.items():
            try:
                if "Bootstrap" in mname:
                    P, t = mfunc(X_tr, X_va, W_true)
                else:
                    P, t = mfunc(X_tr, None, None)
                metrics = evaluate(W_true, P, t)
                all_results[f"nl_d6_s{seed}_{mname}"] = {
                    "experiment": "nonlinear_d6", "seed": seed,
                    "method": mname, **metrics
                }
                print(f"    {mname:<35} SHD={metrics['shd']:<3.0f} "
                      f"F1={metrics['f1']:<.3f} P={metrics['precision']:<.3f} "
                      f"R={metrics['recall']:<.3f} t={metrics['time']:<5.1f}s")
            except Exception as e:
                print(f"    {mname:<35} ERROR: {e}")

    # Summary
    print(f"\n\n{'='*80}")
    print("  SUMMARY (mean ± std)")
    print(f"{'='*80}")
    print(f"  {'Method':<35} {'SHD':>8} {'F1':>6} {'AUCPR':>6} {'ECE':>6} {'t(s)':>6}")
    print(f"  {'─'*35} {'─'*8} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")

    from collections import defaultdict
    agg = defaultdict(list)
    for k, v in all_results.items():
        agg[v["method"]].append(v)

    for method, ml in sorted(agg.items()):
        vals = {k: np.mean([m[k] for m in ml]) for k in ["shd", "f1", "auc_pr", "ece", "time"]}
        stds = {k: np.std([m[k] for m in ml]) for k in ["shd", "f1", "auc_pr", "ece", "time"]}
        print(f"  {method:<35} {vals['shd']:5.1f}±{stds['shd']:.1f} "
              f"{vals['f1']:5.3f}±{stds['f1']:.3f} "
              f"{vals['auc_pr']:5.3f}±{stds['auc_pr']:.3f} "
              f"{vals['ece']:5.4f}±{stds['ece']:.4f} "
              f"{vals['time']:5.1f}±{stds['time']:.1f}")

    ser = {}
    for k, v in all_results.items():
        ser[k] = {kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv)
                  for kk, vv in v.items()}
    with open(RESULTS_FILE, "w") as f:
        json.dump(ser, f, indent=2)
    print(f"\n  Saved {RESULTS_FILE}")


if __name__ == "__main__":
    main()
