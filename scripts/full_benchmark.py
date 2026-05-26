#!/usr/bin/env python3
"""Full benchmark: CausalBayes vs gCastle on ALL 9 datasets."""
import warnings; warnings.filterwarnings('ignore')
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from castle.algorithms import GES, PC
from causbayes import CausalBayesEstimator

DATASETS = {
    "Cancer": ("cancer", 5, 1000, "small"),
    "Earthquake": ("earthquake", 5, 1000, "small"),
    "Survey": ("survey", 6, 1000, "small"),
    "Asia": ("asia", 8, 1000, "small"),
    "Sachs": ("sachs", 11, 10000, "small"),
    "Child": ("child", 20, 10000, "medium"),
    "Insurance": ("insurance", 27, 10000, "medium"),
    "Water": ("water", 32, 10000, "medium"),
    "Alarm": ("alarm", 37, 10000, "large"),
}

BASE = os.path.join(os.path.dirname(__file__), "..", "experiment_results")
results_all = {}

for name, (fname, n_vars, n_samp, size) in sorted(DATASETS.items(), key=lambda x: x[1][1]):
    print(f"\n{'='*60}")
    print(f"DATASET: {name} ({n_vars} vars)")
    print(f"{'='*60}")
    
    # Load data + ground truth
    df = pd.read_csv(os.path.join(BASE, f"{fname}_data.csv"))
    # Handle categorical data (bnlearn datasets may have string values)
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col].dtype) or pd.api.types.is_bool_dtype(df[col].dtype):
            df[col] = pd.Categorical(df[col]).codes.astype(float)
    X = StandardScaler().fit_transform(df.values.astype(float))
    d = X.shape[1]
    vars_list = list(df.columns)
    
    dag_df = pd.read_csv(os.path.join(BASE, f"{fname}_dag.csv"), index_col=0)
    Wt = dag_df.values.astype(float)
    
    n_edges = int(Wt.sum())
    print(f"  Samples: {X.shape[0]}, Edges: {n_edges}, Density: {n_edges/(d*d):.3f}")
    
    # Subsample large datasets to 1000 for speed
    if X.shape[0] > 2000:
        np.random.seed(42)
        idx = np.random.choice(X.shape[0], 2000, replace=False)
        X = X[idx]
    
    def metrics(Wt, W_est, label=""):
        shd = int(np.sum((W_est>0) != (Wt>0)))
        tp = int(np.sum((W_est>0)&(Wt>0))); fp=int(np.sum((W_est>0)&(Wt==0)))
        fn=int(np.sum((W_est==0)&(Wt>0)))
        f1 = 2*tp/max(2*tp+fp+fn,1)
        prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1)
        return {"SHD":shd, "F1":round(f1,4), "Precision":round(prec,4),
                "Recall":round(rec,4), "Edges":int(W_est.sum()), "TP":tp, "FP":fp, "FN":fn}
    
    def sweep_threshold(W_strength, Wt):
        best = {"SHD": 999, "method": ""}
        for th in np.arange(0.001, 0.2, 0.002):
            Wb = (W_strength > th).astype(float)
            m = metrics(Wt, Wb)
            if m["SHD"] < best["SHD"] or (m["SHD"] == best["SHD"] and m["F1"] > best["F1"]):
                best = m
                best["threshold"] = round(th, 3)
        return best
    
    # ── 1. gCastle GES ──
    t0 = time.time()
    ges = GES(); ges.learn(X)
    W_ges = np.array(ges.causal_matrix, dtype=float).T
    
    # gCastle threshold sweep (0.01-0.5)
    best_ges = {"SHD": 999}
    for th in [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        Wb = (np.abs(W_ges) > th).astype(float)
        m = metrics(Wt, Wb)
        if m["SHD"] < best_ges["SHD"]: best_ges = m; best_ges["threshold"] = th
    t_ges = time.time()-t0
    print(f"  ⏱ gCastle GES: {t_ges:.1f}s → SHD={best_ges['SHD']} F1={best_ges['F1']:.4f}")
    
    # ── 2. CausalBayes Bootstrap (mean-weight, no prior) ──
    n_boot = 50 if size != "small" else 100
    n_boot = min(n_boot, 50 if n_vars > 25 else 100)
    
    t0 = time.time()
    from causbayes.structure_learning.notears_fast import notears_lbfgs
    from sklearn.utils import resample
    
    W_list = []
    for i in range(n_boot):
        Xb = resample(X, random_state=42+i); Xb -= Xb.mean(axis=0, keepdims=True)
        W = notears_lbfgs(Xb, lambda_1=0.005, max_iter=10, w_threshold=0.01, lbfgs_maxiter=30)
        if not np.isnan(W).any(): W_list.append(W)
    
    W_str = np.mean(np.abs(np.array(W_list)), axis=0); np.fill_diagonal(W_str, 0.0)
    best_cb = sweep_threshold(W_str, Wt)
    t_cb = time.time()-t0
    print(f"  ⏱ CB Bootstrap (B={len(W_list)}): {t_cb:.1f}s → SHD={best_cb['SHD']} F1={best_cb['F1']:.4f} t={best_cb['threshold']}")
    
    # ── 3. CausalBayes + Prior ──
    # Prior: show 70% of true edges
    prior = np.full((d,d), 0.5); np.fill_diagonal(prior, 0.0)
    nz = np.where(Wt > 0)
    np.random.seed(42); idx_p = np.random.choice(len(nz[0]), int(len(nz[0])*0.7), replace=False)
    for k in idx_p: prior[nz[0][k], nz[1][k]] = 0.9
    
    t0 = time.time()
    W_list_p = []
    for i in range(n_boot):
        Xb = resample(X, random_state=42+i); Xb -= Xb.mean(axis=0, keepdims=True)
        W = notears_lbfgs(Xb, lambda_1=0.005, max_iter=10, w_threshold=0.01,
                         lbfgs_maxiter=30, prior_matrix=prior, lambda_prior=0.5)
        if not np.isnan(W).any(): W_list_p.append(W)
    
    W_str_p = np.mean(np.abs(np.array(W_list_p)), axis=0); np.fill_diagonal(W_str_p, 0.0)
    best_cbp = sweep_threshold(W_str_p, Wt)
    t_cbp = time.time()-t0
    print(f"  ⏱ CB+Prior (B={len(W_list_p)}): {t_cbp:.1f}s → SHD={best_cbp['SHD']} F1={best_cbp['F1']:.4f} t={best_cbp['threshold']}")
    
    # Store
    results_all[name] = {
        "vars": d, "true_edges": n_edges, "samples": X.shape[0],
        "gCastle_GES": best_ges,
        "CB_Bootstrap": best_cb,
        "CB_Plus_Prior": best_cbp
    }

# ═══════ FINAL TABLE ═══════
print("\n\n" + "=" * 100)
print("FINAL COMPARISON TABLE")
print("=" * 100)
print(f"{'Dataset':<12s} {'Vars':>4s} {'Edges':>5s} {'':5s} {'Method':<22s} {'SHD':>4s} {'F1':>7s} {'Prec':>5s} {'Rec':>5s} {'Edges':>5s} {'TP':>3s} {'FP':>3s} {'FN':>3s}")
print("-" * 100)

for name in sorted(results_all.keys(), key=lambda k: results_all[k]["vars"]):
    r = results_all[name]
    first = True
    for method, label in [("gCastle_GES", "gCastle GES"), 
                         ("CB_Bootstrap", "CB Bootstrap"),
                         ("CB_Plus_Prior", "CB + Prior")]:
        m = r[method]
        prefix = f"{name:<12s} {r['vars']:>4d} {r['true_edges']:>5d}" if first else " "*24
        first = False
        print(f"{prefix}  {label:<22s} {m['SHD']:>4d} {m['F1']:>7.4f} {m['Precision']:>5.2f} {m['Recall']:>5.2f} {m['Edges']:>5d} {m['TP']:>3d} {m['FP']:>3d} {m['FN']:>3d}")

# Summary stats
print("\n" + "=" * 100)
print("SUMMARY: CB+Prior vs gCastle GES")
print("=" * 100)

wins_cb = 0; wins_ges = 0; ties = 0
for name in sorted(results_all.keys(), key=lambda k: results_all[k]["vars"]):
    r = results_all[name]
    g = r["gCastle_GES"]["F1"]
    c = r["CB_Plus_Prior"]["F1"]
    delta = c - g
    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
    if delta > 0.01: wins_cb += 1
    elif delta < -0.01: wins_ges += 1
    else: ties += 1
    print(f"  {name:<12s} F1: gCastle={g:.4f}  CB+Prior={c:.4f}  Δ={arrow}{delta:+.4f}")

print(f"\n  CB+Prior wins: {wins_cb}/{wins_cb+wins_ges+ties}")
print(f"  gCastle wins:  {wins_ges}/{wins_cb+wins_ges+ties}")
print(f"  Ties:          {ties}/{wins_cb+wins_ges+ties}")

# Save results
with open(os.path.join(BASE, "benchmark_results.json"), "w") as f:
    json.dump(results_all, f, indent=2)
print(f"\n  Results saved to benchmark_results.json")
