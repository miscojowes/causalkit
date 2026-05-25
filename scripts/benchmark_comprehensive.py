#!/usr/bin/env python3
"""
Comprehensive multi-seed benchmark: BootstrapDAG vs Single NOTEARS vs Random.

Tests d=5 linear Gaussian data across 5 seeds.
Uses current codebase (notears_lbfgs ~8s per call).
Reports SHD, Precision, Recall, F1, AUC-PR, ECE, Brier Score.
"""

import sys, os, json, time, warnings, gc
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from sklearn.preprocessing import StandardScaler
from causbayes.evaluation import edge_calibration, precision_recall_auc
from causbayes.structure_learning.utils import structural_hamming_distance, expected_shd
from causbayes.structure_learning.notears_fast import notears_lbfgs

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiment_results")
os.makedirs(RESULTS_DIR, exist_ok=True)
SEEDS = [42, 43, 44, 45, 46]


def gen_data(d=5, n=1000, edge_prob=0.2, noise_scale=0.1, seed=42):
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


def compute_metrics(W_true, P, W_bin, time_s):
    shd = structural_hamming_distance(W_true, W_bin)
    e_shd = expected_shd(W_true, P)
    tp = np.sum((W_bin > 0) & (W_true > 0))
    fp = np.sum((W_bin > 0) & (W_true == 0))
    fn = np.sum((W_bin == 0) & (W_true > 0))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
    try:
        auc_pr = precision_recall_auc(W_true, P)["auc_pr"]
    except Exception:
        auc_pr = float("nan")
    try:
        ece = edge_calibration(W_true, P, n_bins=10)["ece"]
    except Exception:
        ece = float("nan")
    d = W_true.shape[0]
    brier = sum((P[i,j]-W_true[i,j])**2 for i in range(d) for j in range(d) if i!=j) / (d*(d-1))
    return {"shd": shd, "expected_shd": e_shd, "precision": prec, "recall": rec,
            "f1": f1, "auc_pr": auc_pr, "ece": ece, "brier_score": brier,
            "true_edges": int(np.sum(W_true>0)), "est_edges": int(np.sum(W_bin>0)),
            "time_s": time_s}


def main():
    print("="*80)
    print("  Comprehensive Benchmark (current codebase)")
    print("  d=5, n=1000, 5 seeds, non-bootstrap: Single NOTEARS vs Random")
    print("="*80)
    all_results = []

    for seed in SEEDS:
        print(f"\n  Seed {seed}: ", end="", flush=True)
        X_all, W_true = gen_data(seed=seed)
        n_tr, n_va = 600, 200
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_all[:n_tr])
        te = int(np.sum(W_true > 0))
        row = {"seed": seed, "true_edges": te}

        # Single NOTEARS
        t0 = time.time()
        X_c = X_tr - X_tr.mean(axis=0, keepdims=True)
        W_est = notears_lbfgs(X_c, lambda_1=0.01, max_iter=10, w_threshold=0.1)
        t = time.time()-t0
        P = (np.abs(W_est) > 0).astype(float)
        row["Single NOTEARS"] = compute_metrics(W_true, P, P, t)

        # Random
        t0 = time.time()
        rng = np.random.RandomState(seed+9999)
        P_r = rng.uniform(0,1,(5,5)); np.fill_diagonal(P_r,0)
        W_r = (P_r >= 0.5).astype(float)
        row["Random"] = compute_metrics(W_true, P_r, W_r, time.time()-t0)

        # Bootstrap(10) - quick version
        print(f"Bootstrap(10)...", end=" ", flush=True)
        t0 = time.time()
        from sklearn.utils import resample
        W_list = []
        for i in range(10):
            X_b = resample(X_tr, random_state=seed+i)
            X_b = X_b - X_b.mean(axis=0)
            try:
                Wi = notears_lbfgs(X_b, lambda_1=0.01, max_iter=10, w_threshold=0.1)
                if not np.isnan(Wi).any():
                    W_list.append(Wi)
            except: pass
        t = time.time()-t0
        if W_list:
            W_a = np.array(W_list)
            P_b = np.mean(np.abs(W_a) > 0, axis=0)
            np.fill_diagonal(P_b, 0)
            Wb = (P_b >= 0.5).astype(float)
            row["Bootstrap(10)"] = compute_metrics(W_true, P_b, Wb, t)
            nv = len(W_list)
        else:
            row["Bootstrap(10)"] = {"error": "all failed"}

        all_results.append(row)
        m = row.get("Bootstrap(10)", {})
        if isinstance(m, dict) and "shd" in m:
            print(f"SHD={m['shd']:.1f} P={m['precision']:.2f} R={m['recall']:.2f} t={t:.0f}s")

    # Summary
    print(f"\n\n  SUMMARY (mean±std)")
    for method in ["Bootstrap(10)", "Single NOTEARS", "Random"]:
        vals = {k:[] for k in ["shd","precision","recall","f1","auc_pr","ece","brier_score","time_s"]}
        for row in all_results:
            r = row.get(method, {})
            if isinstance(r, dict):
                for k in vals:
                    if k in r and r[k] is not None and not (isinstance(r[k], float) and np.isnan(r[k])):
                        vals[k].append(r[k])
        if not vals["shd"]: continue
        print(f"  {method:<20}", end="")
        for k in ["shd","precision","recall","f1","auc_pr","ece","brier_score","time_s"]:
            if vals[k]:
                mv, sv = float(np.mean(vals[k])), float(np.std(vals[k]))
                print(f" {k}={mv:.3f}±{sv:.3f}", end="")
        print()

    # Save
    def cv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        return o
    with open(os.path.join(RESULTS_DIR,"comprehensive_d5.json"), "w") as f:
        json.dump({"by_seed": all_results, "seeds": SEEDS}, f, indent=2, default=cv)
    print(f"\n  Saved to experiment_results/comprehensive_d5.json")

if __name__ == "__main__":
    main()
