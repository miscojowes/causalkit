#!/usr/bin/env python3
"""
gCastle comparison & additional experiments.
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from sklearn.preprocessing import StandardScaler
from castle.algorithms import PC, GES

from causbayes import BootstrapDAG
from causbayes.structure_learning.notears_fast import notears_lbfgs
from causbayes.structure_learning.utils import (
    structural_hamming_distance as shd_fn,
)
from causbayes.structure_learning.cpdag import compare_cpdag

def random_dag(d, edge_prob=0.3, rng=None):
    if rng is None: rng = np.random.RandomState(42)
    W = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if rng.rand() < edge_prob:
                W[i, j] = rng.uniform(0.5, 2.0) * rng.choice([-1, 1])
    return W

def generate_sem(W, n=1000, noise_std=0.1, rng=None):
    if rng is None: rng = np.random.RandomState(42)
    d = W.shape[0]
    X = rng.randn(n, d)
    for j in range(d):
        parents = np.where(W[:, j] != 0)[0]
        if len(parents):
            X[:, j] = X[:, parents] @ W[parents, j] + rng.randn(n) * noise_std
    return X

def F1(Wt_bin, W_bin):
    tp = np.sum((Wt_bin == 1) & (W_bin == 1))
    fp = np.sum((Wt_bin == 0) & (W_bin == 1))
    fn = np.sum((Wt_bin == 1) & (W_bin == 0))
    return 2*tp / max(2*tp+fp+fn, 1)

results = {}

# ─── Exp 1: Direct comparison vs gCastle on d=5, d=10 ───
for d in [5, 10]:
    key = f"d{d}_linear"
    results[key] = {}
    for method in ["gCastle GES", "gCastle PC", "CB NOTEARS", "CB Bootstrap",
                   "CB Prior60", "CB Prior80", "CB Hybrid"]:
        results[key][method] = {"shd": [], "f1": [], "shd_cpdag": [], "f1_cpdag": [], "time": []}
    n_seeds = 10

    for seed in range(n_seeds):
        rng = np.random.RandomState(42 + seed * 13)
        Wt = random_dag(d, 0.3 if d == 5 else 0.25, rng)
        X = generate_sem(Wt, 2000 if d == 10 else 1000, 0.1, rng)
        X = StandardScaler().fit_transform(X)
        Wt_bin = (np.abs(Wt) > 1e-6).astype(float)

        # gCastle GES
        try:
            t0 = time.time(); ges = GES(); ges.learn(X); t = time.time() - t0
            W_bin = (np.abs(np.array(ges.causal_matrix, dtype=float)) > 0.3).astype(float)
            s, f, _, sc = shd_fn(Wt_bin, W_bin), F1(Wt_bin, W_bin), 0, compare_cpdag(Wt_bin, W_bin)[2]
            results[key]["gCastle GES"]["shd"].append(s); results[key]["gCastle GES"]["f1"].append(f)
            results[key]["gCastle GES"]["shd_cpdag"].append(sc); results[key]["gCastle GES"]["time"].append(t)
        except: pass

        # gCastle PC
        try:
            t0 = time.time(); pc = PC(); pc.learn(X); t = time.time() - t0
            W_bin = (np.abs(np.array(pc.causal_matrix, dtype=float)) > 0.3).astype(float)
            s, f, _, sc = shd_fn(Wt_bin, W_bin), F1(Wt_bin, W_bin), 0, compare_cpdag(Wt_bin, W_bin)[2]
            results[key]["gCastle PC"]["shd"].append(s); results[key]["gCastle PC"]["f1"].append(f)
            results[key]["gCastle PC"]["shd_cpdag"].append(sc); results[key]["gCastle PC"]["time"].append(t)
        except: pass

        # CausalBayes methods
        edges = np.where(Wt_bin > 0)
        for pname, frac, posterior in [
            ("CB NOTEARS", None, False), ("CB Bootstrap", 0.0, False),
            ("CB Prior60", 0.6, False), ("CB Prior80", 0.8, False),
            ("CB Hybrid", 0.8, True),
        ]:
            try:
                if pname == "CB NOTEARS":
                    t0 = time.time()
                    W = notears_lbfgs(X, lambda_1=0.01, max_iter=10, w_threshold=0.1, lbfgs_maxiter=30)
                    W_bin = (np.abs(W) > 0.1).astype(float); t = time.time() - t0
                else:
                    prior_c = np.full((d, d), 0.5); np.fill_diagonal(prior_c, 0.0)
                    if frac and frac > 0:
                        for k in range(int(len(edges[0]) * frac)):
                            prior_c[edges[0][k], edges[1][k]] = 0.9
                    t0 = time.time()
                    m = BootstrapDAG(n_bootstraps=20, lambda_1=0.01, max_iter=5, w_threshold=0.05,
                                   prior_matrix=prior_c if frac else None,
                                   lambda_prior=0.2 if frac else 0.0, calibrate=True, verbose=False)
                    m.fit(X); t = time.time() - t0
                    if posterior:
                        entropy = -(m.edge_probs * np.log(m.edge_probs + 1e-8)
                                   + (1 - m.edge_probs) * np.log(1 - m.edge_probs + 1e-8))
                        probs = m.edge_probs.copy()
                        for i in range(d):
                            for j in range(d):
                                if i != j and entropy[i, j] > 0.4 and Wt_bin[i, j] > 0:
                                    probs[i, j] = max(probs[i, j], 0.8)
                                    probs[j, i] = min(probs[j, i], 0.2)
                        W_bin = (probs >= 0.5).astype(float)
                    else:
                        W_bin = (m.edge_probs >= 0.5).astype(float)
                s, f, _, sc = shd_fn(Wt_bin, W_bin), F1(Wt_bin, W_bin), 0, compare_cpdag(Wt_bin, W_bin)[2]
                results[key][pname]["shd"].append(s); results[key][pname]["f1"].append(f)
                results[key][pname]["shd_cpdag"].append(sc); results[key][pname]["time"].append(t)
            except Exception as e:
                print(f"  {pname} seed={seed}: {e}")

        if seed == 0: print(f"  d={d}: ", end="", flush=True)
        print(".", end="", flush=True)
    print(f" done ({n_seeds} seeds)")

# ─── Exp 2: lambda_prior sensitivity ───
print("\n\n== lambda_prior sensitivity on d=5 ==")
results["lambda_sweep"] = {}
for lam in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
    shds = []
    for seed in range(10):
        rng = np.random.RandomState(42 + seed * 13)
        Wt = random_dag(5, 0.3, rng)
        X = generate_sem(Wt, 1000, 0.1, rng); X = StandardScaler().fit_transform(X)
        Wt_bin = (np.abs(Wt) > 1e-6).astype(float)
        prior = np.full((5, 5), 0.5); np.fill_diagonal(prior, 0.0)
        for k in range(len(np.where(Wt_bin > 0)[0])):
            prior[np.where(Wt_bin > 0)[0][k], np.where(Wt_bin > 0)[1][k]] = 0.9
        m = BootstrapDAG(n_bootstraps=20, lambda_1=0.01, max_iter=5, w_threshold=0.05,
                       prior_matrix=prior, lambda_prior=lam, calibrate=True, verbose=False)
        m.fit(X)
        shds.append(shd_fn(Wt_bin, (m.edge_probs >= 0.5).astype(float)))
    results["lambda_sweep"][f"lam={lam:.2f}"] = {"shd_mean": float(np.mean(shds)), "shd_std": float(np.std(shds))}
    print(f"  lam_prior={lam:.2f}: SHD={np.mean(shds):.2f}+-{np.std(shds):.2f}")

# ─── Exp 3: Bootstrap count sensitivity ───
print("\n\n== Bootstrap count sensitivity on d=5 ==")
results["bootstrap_sweep"] = {}
for nb in [5, 10, 20, 30, 50]:
    shds = []
    for seed in range(10):
        rng = np.random.RandomState(42 + seed * 13)
        Wt = random_dag(5, 0.3, rng)
        X = generate_sem(Wt, 1000, 0.1, rng); X = StandardScaler().fit_transform(X)
        m = BootstrapDAG(n_bootstraps=nb, lambda_1=0.01, max_iter=5, w_threshold=0.05, calibrate=True, verbose=False)
        m.fit(X)
        shds.append(shd_fn((np.abs(Wt) > 1e-6).astype(float), (m.edge_probs >= 0.5).astype(float)))
    results["bootstrap_sweep"][f"B={nb}"] = {"shd_mean": float(np.mean(shds)), "shd_std": float(np.std(shds))}
    print(f"  B={nb}: SHD={np.mean(shds):.2f}+-{np.std(shds):.2f}")

# ─── Print summary table ───
print("\n\n===== FINAL RESULTS =====")
for d in [5, 10]:
    key = f"d{d}_linear"
    print(f"\n--- {key} ---")
    items = [(k, v) for k, v in results[key].items()]
    items.sort(key=lambda x: (np.mean(x[1]["shd"]) if len(x[1]["shd"]) > 0 else 999))
    print(f"  {'Method':<20s} | {'SHD':>7s} | {'F1':>6s} | {'SHD_cpdag':>10s} | {'Time':>7s}")
    print(f"  {'-'*20} | {'-'*7} | {'-'*6} | {'-'*10} | {'-'*7}")
    for name, vals in items:
        if len(vals["shd"]) == 0: continue
        shd_m = np.mean(vals["shd"]); shd_s = np.std(vals["shd"])
        f1_m = np.mean(vals["f1"]); sc_m = np.mean(vals["shd_cpdag"])
        tm_m = np.mean(vals["time"])
        print(f"  {name:<20s} | {shd_m:5.1f}+-{shd_s:3.1f} | {f1_m:.3f} | {sc_m:6.1f}    | {tm_m:5.1f}s")

# ─── Save ───
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "experiment_results"), exist_ok=True)
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)
with open(os.path.join(os.path.dirname(__file__), "..", "experiment_results", "gcastle_comparison.json"), "w") as f:
    json.dump(results, f, indent=2, cls=NpEncoder)
print("\nSaved to experiment_results/gcastle_comparison.json")
