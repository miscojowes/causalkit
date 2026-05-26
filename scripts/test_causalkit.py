#!/usr/bin/env python3
"""
TEST: causalkit library end-to-end (fast)
===========================================
Tests: bootstrap, uniform prior, adaptive trust, causal effects.
Optimized for speed: fewer bootstraps, smaller datasets.
"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from castle.algorithms import GES
import causalkit as ck

np.random.seed(42)

# ─── Loaders ──────────────────────────────────────────────────────────

def load_sachs():
    """Load Sachs with hardcoded GT from literature."""
    base = os.path.join(os.path.dirname(__file__), "..", "experiment_results")
    df = pd.read_csv(os.path.join(base, "sachs_raw.csv"), sep='\t')
    X = StandardScaler().fit_transform(df.values.astype(float))
    gt_map = [("Raf","Mek"),("Mek","Erk"),("Plcg","PIP2"),("PIP2","PIP3"),
              ("PIP3","Akt"),("PKC","PKA"),("PKA","Jnk"),("PKC","Jnk"),
              ("Jnk","P38"),("PKC","P38"),("Erk","Akt"),("PKC","Akt"),
              ("PKA","Akt"),("PIP3","PKA"),("PIP3","Plcg"),("PKA","Erk"),("PKC","Mek")]
    d = X.shape[1]; Wt = np.zeros((d,d))
    vars_list = list(df.columns)
    for c,e in gt_map: Wt[vars_list.index(c), vars_list.index(e)] = 1.0
    return X, Wt, vars_list

def generate_synthetic(d=10, n=2000, seed=42):
    """Generate synthetic linear Gaussian DAG with known GT."""
    rng = np.random.RandomState(seed)
    W = np.zeros((d, d))
    for i in range(1, d):
        n_parents = rng.randint(1, min(4, i+1))
        parents = rng.choice(range(i), size=min(n_parents, i), replace=False)
        for p in parents: W[p, i] = rng.uniform(0.3, 0.8) * rng.choice([-1, 1])
    eigvals = np.linalg.eigvals(W)
    if np.max(np.abs(eigvals)) > 0.95:
        W = W / (np.max(np.abs(eigvals)) * 1.05)
    X = rng.randn(n, d) @ np.linalg.inv(np.eye(d) - W)
    X = StandardScaler().fit_transform(X)
    return X, (np.abs(W) > 0.01).astype(float), [f"x{i}" for i in range(d)]

def metrics(Wt, We, label=""):
    tp = int(np.sum((We > 0) & (Wt > 0)))
    fp = int(np.sum((We > 0) & (Wt == 0)))
    fn = int(np.sum((We == 0) & (Wt > 0)))
    return {"SHD": fp + fn, "F1": 2*tp/max(2*tp+fp+fn,1), "TP": tp, "FP": fp, "FN": fn, "edges": int(We.sum())}

def make_prior(Wt, revealed_pct=0.7, noise_pct=0.0):
    """Create prior with some true edges + optional noise."""
    d = Wt.shape[0]
    prior = np.full((d, d), 0.5); np.fill_diagonal(prior, 0.0)
    true_edges = np.argwhere(Wt > 0)
    np.random.seed(42); np.random.shuffle(true_edges)
    n_reveal = int(len(true_edges) * revealed_pct)
    for i in range(n_reveal): prior[true_edges[i][0], true_edges[i][1]] = 0.9
    if noise_pct > 0:
        false_cands = [(i,j) for i in range(d) for j in range(d)
                       if i != j and Wt[i,j] == 0 and prior[i,j] < 0.6]
        np.random.shuffle(false_cands)
        for i in range(min(int(len(false_cands)*noise_pct), len(false_cands))):
            prior[false_cands[i][0], false_cands[i][1]] = 0.9
    return prior

def time_run(label, func):
    t0 = time.time()
    try:
        result = func()
        dt = time.time() - t0
        print(f"  {label:<45s} {dt:5.1f}s")
        return result
    except Exception as e:
        dt = time.time() - t0
        print(f"  {label:<45s} {dt:5.1f}s FAILED - {e}")
        return None

# ══════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  CAUSALKIT — Full Library Test")
print("=" * 65)

N_BOOT = 30  # Fast enough for testing
all_results = []

for dataset_name, (X, Wt, vars_list) in [
    ("Synthetic d=10", generate_synthetic(10, 2000)),
    ("Sachs (real)", load_sachs()),
]:
    n, d = X.shape
    n_true = int(Wt.sum())
    print(f"\n{'─'*65}")
    print(f"▶ {dataset_name}  ({n} samples × {d} vars, {n_true} true edges)")
    print(f"{'─'*65}")

    row = {"Dataset": dataset_name, "d": d}

    # ── GES baseline ──
    ges = time_run("GES baseline", lambda: GES())
    if ges is not None:
        ges.learn(X)
        Wges = (np.abs(np.array(ges.causal_matrix, dtype=float).T) > 0.3).astype(float)
        row["GES"] = metrics(Wt, Wges)
    else:
        row["GES"] = None

    # ── causalkit: no prior ──
    def run_no_prior():
        mod = ck.CausalDiscoverer(method='bootstrap', n_bootstraps=N_BOOT, verbose=False)
        mod.fit(X)
        return metrics(Wt, mod.causal_matrix_)
    row["NoPrior"] = time_run("causalkit (no prior)", run_no_prior)

    # ── causalkit: uniform λ prior (clean 70%) ──
    prior_clean = make_prior(Wt, 0.7, 0.0)
    def run_uniform():
        mod = ck.CausalDiscoverer(method='bootstrap', n_bootstraps=N_BOOT, adaptive_trust=False, lambda_prior=0.5, verbose=False)
        mod.fit(X, prior_matrix=prior_clean)
        return metrics(Wt, mod.causal_matrix_)
    row["UniformPrior"] = time_run("causalkit (uniform λ)", run_uniform)

    # ── causalkit: adaptive trust ──
    def run_adaptive():
        mod = ck.CausalDiscoverer(method='bootstrap', n_bootstraps=N_BOOT, adaptive_trust=True, lambda_prior=0.5, verbose=False)
        mod.fit(X, prior_matrix=prior_clean)
        return metrics(Wt, mod.causal_matrix_)
    row["AdaptivePrior"] = time_run("causalkit (adaptive trust)", run_adaptive)

    # ── STRESS TEST: mixed-quality prior (50% correct + 50% noise) ──
    prior_mixed = make_prior(Wt, 0.5, 0.5)
    n_noisy = int(np.sum((prior_mixed > 0.6) & (Wt == 0)))
    n_correct = int(np.sum((prior_mixed > 0.6) & (Wt > 0)))
    print(f"\n  ── STRESS: Mixed prior ({n_correct} correct + {n_noisy} noisy edges, d={d}) ──")

    def run_mixed_uniform():
        mod = ck.CausalDiscoverer(method='bootstrap', n_bootstraps=N_BOOT, adaptive_trust=False, lambda_prior=0.5, verbose=False)
        mod.fit(X, prior_matrix=prior_mixed)
        return metrics(Wt, mod.causal_matrix_)
    row["MixedUniform"] = time_run("  Uniform λ on mixed prior", run_mixed_uniform)

    def run_mixed_adaptive():
        mod = ck.CausalDiscoverer(method='bootstrap', n_bootstraps=N_BOOT, adaptive_trust=True, lambda_prior=0.5, verbose=False)
        mod.fit(X, prior_matrix=prior_mixed)
        return metrics(Wt, mod.causal_matrix_)
    row["MixedAdaptive"] = time_run("  Adaptive trust on mixed", run_mixed_adaptive)

    all_results.append(row)

# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  RESULTS SUMMARY")
print(f"{'='*65}")
h = f"{'Dataset':<18s} {'GES':>7s} {'GES-F1':>7s} {'NoPrior':>8s} {'Uniform':>8s} {'Adapt':>8s}"
print(h)
print("-" * len(h))
for r in all_results:
    name = r['Dataset']
    ges = r.get("GES", {}); np_ = r.get("NoPrior", {}); un = r.get("UniformPrior", {}); ad = r.get("AdaptivePrior", {})
    ges_s = f"{ges.get('SHD',-1):3d}" if ges else "  -"
    ges_f1 = f"{ges.get('F1',0):.4f}" if ges else "     -"
    np_s = f"{np_.get('F1',0):.4f}" if np_ else "     -"
    un_s = f"{un.get('F1',0):.4f}" if un else "     -"
    ad_s = f"{ad.get('F1',0):.4f}" if ad else "     -"
    print(f"{name:<18s} {ges_s:>7s} {ges_f1:>7s} {np_s:>8s} {un_s:>8s} {ad_s:>8s}")

# STRESS TEST SUMMARY
print(f"\n{'─'*40}")
print("  MIXED PRIOR STRESS TEST RESULTS")
print(f"{'─'*40}")
for r in all_results:
    mu = r.get("MixedUniform", {}); ma = r.get("MixedAdaptive", {})
    if mu and ma:
        print(f"  {r['Dataset']:<18s} Uniform: F1={mu['F1']:.4f}  Adaptive: F1={ma['F1']:.4f}  Δ={ma['F1']-mu['F1']:+.4f}")
        if ma['F1'] >= mu['F1']:
            print(f"  {'':>18s} ✅ Adaptive ≥ Uniform")
        else:
            print(f"  {'':>18s} ❌ Uniform better (needs tuning)")

# ══════════════════════════════════════════════════════════════════════
#  CAUSAL EFFECTS TEST
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  CAUSAL EFFECTS TEST (ATE + what-if)")
print(f"{'='*65}")

d = 5
W_true = np.array([
    [0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],    # x0 → x1
    [0, 0.8, 0, 0, 0],  # x1 → x2
    [0.5, 0, 0, 0, 0],  # x0 → x3
    [0, 0, 0.6, 0.4, 0], # x2→x4, x3→x4
]).astype(float)
rng = np.random.RandomState(42)
n = 5000
X_ol = rng.randn(n, d) @ np.linalg.inv(np.eye(d) - W_true)
X_ol = StandardScaler().fit_transform(X_ol)
X_df = pd.DataFrame(X_ol, columns=["x0", "x1", "x2", "x3", "x4"])

mod_ol = ck.CausalDiscoverer(method='notears', verbose=False)
mod_ol.fit(X_ol, feature_names=["x0", "x1", "x2", "x3", "x4"])

ate_x0_x1 = mod_ol.estimate_ate(X_df, treatment="x0", outcome="x1")
ate_x1_x2 = mod_ol.estimate_ate(X_df, treatment="x1", outcome="x2")
print(f"  ATE x0 → x1: {ate_x0_x1:.4f} (true: 1.0)")
print(f"  ATE x1 → x2: {ate_x1_x2:.4f} (true: 0.8)")

interv = {"x0": 2.0}
pred = mod_ol.counterfactual_predict(X_ol, interv)
print(f"  What-if (x0=2.0): E[x1] = {pred[:, 1].mean():.3f} (expected ~2.0)")
print(f"  What-if (x0=2.0): E[x3] = {pred[:, 3].mean():.3f} (expected ~1.0)")

print(f"\n{'='*65}")
print(f"  ✅ causalkit v{ck.__version__} — all tests complete!")
print(f"{'='*65}")
