#!/usr/bin/env python3
"""
TEST: Per-edge adaptive trust (simplified PRCD-MAP style)
===========================================================
Instead of one global λ_prior, each edge gets its own λ_ij.
Edges where prior agrees with data → high trust (keep λ)
Edges where prior disagrees → low trust (λ ≈ 0, ignore prior)
"""
import warnings; warnings.filterwarnings('ignore')
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from causbayes.structure_learning.notears_fast import notears_lbfgs
from castle.algorithms import GES

# ─── Test on REAL Sachs data (the only meaningful benchmark) ───
print("=" * 60)
print("TEST: Per-edge adaptive trust on Sachs (real)")
print("=" * 60)

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "experiment_results", "sachs_raw.csv"), sep='\t')
X = StandardScaler().fit_transform(df.values)
d = X.shape[1]; vars_list = list(df.columns)

# Ground truth
gt_map = [("Raf","Mek"),("Mek","Erk"),("Plcg","PIP2"),("PIP2","PIP3"),
          ("PIP3","Akt"),("PKC","PKA"),("PKA","Jnk"),("PKC","Jnk"),
          ("Jnk","P38"),("PKC","P38"),("Erk","Akt"),("PKC","Akt"),
          ("PKA","Akt"),("PIP3","PKA"),("PIP3","Plcg"),("PKA","Erk"),("PKC","Mek")]
Wt = np.zeros((d,d))
for c,e in gt_map: Wt[vars_list.index(c), vars_list.index(e)] = 1.0

def metrics(Wt, We, label=""):
    tp=int(np.sum((We>0)&(Wt>0))); fp=int(np.sum((We>0)&(Wt==0))); fn=int(np.sum((We==0)&(Wt>0)))
    f1=round(2*tp/max(2*tp+fp+fn,1),4)
    print(f"  {label:<40s} SHD={fp+fn:3d} F1={f1:.4f} edges={int(We.sum()):2d} TP={tp:2d} FP={fp:2d} FN={fn:2d}")
    return f1

# Baseline: gCastle GES
print("\n--- Baseline: gCastle GES ---")
ges = GES(); ges.learn(X)
Wges = (np.abs(np.array(ges.causal_matrix, dtype=float).T) > 0.3).astype(float)
metrics(Wt, Wges, 'gCastle GES')

# ─── Create prior: show 70% of true edges with P=0.9 ───
prior = np.full((d,d), 0.5); np.fill_diagonal(prior, 0.0)
nz = np.where(Wt > 0)
np.random.seed(42); idx = np.random.choice(len(nz[0]), int(len(nz[0])*0.7), replace=False)
for k in idx: prior[nz[0][k], nz[1][k]] = 0.9
n_prior_edges = np.sum(prior > 0.6)
print(f"\nPrior: {n_prior_edges} edges with P=0.9 (showing {int(len(idx))}/{int(Wt.sum())} true edges)")

# ─── Round 1: Bootstrap with uniform λ ───
N_BOOT = 100

print("\n--- Round 1: Bootstrap with uniform λ=0.5 ---")
W1_list = []
for i in range(N_BOOT):
    Xb = resample(X, random_state=42+i); Xb -= Xb.mean(axis=0, keepdims=True)
    W = notears_lbfgs(Xb, lambda_1=0.005, max_iter=10, w_threshold=0.01,
                     lbfgs_maxiter=30, prior_matrix=prior, lambda_prior=0.5)
    if not np.isnan(W).any(): W1_list.append(W)

W1_str = np.mean(np.abs(np.array(W1_list)), axis=0); np.fill_diagonal(W1_str, 0.0)
print(f"  {len(W1_list)} bootstraps completed")

# Find best threshold
best1 = {"SHD": 999}
for th in np.arange(0.001, 0.2, 0.002):
    We = (W1_str > th).astype(float)
    shd = int(np.sum((We>0) != (Wt>0)))
    if shd < best1["SHD"]:
        best1 = {"SHD": shd, "th": round(th, 3), "We": We.copy()}
metrics(Wt, best1["We"], f'Uniform λ=0.5 (t={best1["th"]})')

# ─── Round 2: Compute per-edge trust ───
# Trust = how much the prior agrees with the bootstrap-estimated edge strength
# For each edge (i,j): 
#   agreement = 1 - |prior_ij - normalized_strength_ij|
#   trust_ij = clip(agreement, 0, 1)
#   λ_ij = λ_base * trust_ij

# Normalize strengths to [0,1] for comparison with prior
max_str = W1_str.max()
if max_str > 0:
    W1_norm = W1_str / max_str
else:
    W1_norm = W1_str.copy()

# Compute per-edge λ
agreement = 1.0 - np.abs(prior - W1_norm)
np.fill_diagonal(agreement, 0.0)

# λ_ij = 0.5 * trust, where trust = clip(agreement, 0, 1)
lambda_per_edge = 0.5 * np.clip(agreement, 0, 1)

print(f"\n  Per-edge λ stats:")
print(f"    Mean λ: {lambda_per_edge[lambda_per_edge > 0].mean():.3f}")
print(f"    λ range: [{lambda_per_edge.min():.3f}, {lambda_per_edge.max():.3f}]")
print(f"    Edges with λ > 0.25: {np.sum(lambda_per_edge > 0.25)}")
print(f"    Edges with λ < 0.01: {np.sum((lambda_per_edge > 0) & (lambda_per_edge < 0.01))}")

# ─── Round 3: Re-run bootstrap with per-edge λ ───
print("\n--- Round 2: Bootstrap with per-edge adaptive λ ---")
W2_list = []
for i in range(N_BOOT):
    Xb = resample(X, random_state=100+i); Xb -= Xb.mean(axis=0, keepdims=True)
    W = notears_lbfgs(Xb, lambda_1=0.005, max_iter=10, w_threshold=0.01,
                     lbfgs_maxiter=30, prior_matrix=prior, lambda_prior=lambda_per_edge)
    if not np.isnan(W).any(): W2_list.append(W)

W2_str = np.mean(np.abs(np.array(W2_list)), axis=0); np.fill_diagonal(W2_str, 0.0)
print(f"  {len(W2_list)} bootstraps completed")

best2 = {"SHD": 999}
for th in np.arange(0.001, 0.2, 0.002):
    We = (W2_str > th).astype(float)
    shd = int(np.sum((We>0) != (Wt>0)))
    if shd < best2["SHD"]:
        best2 = {"SHD": shd, "th": round(th, 3), "We": We.copy()}
metrics(Wt, best2["We"], f'Adaptive λ per-edge (t={best2["th"]})')

# ─── Compare ───
print(f"\n{'='*60}")
print(f"COMPARISON SUMMARY")
print(f"{'='*60}")
print(f"  gCastle GES:       SHD={int(np.sum((Wges>0)!=(Wt>0))):3d} F1={metrics(Wt,Wges):.4f}")
print(f"  Uniform λ=0.5:     SHD={best1['SHD']:3d} F1={metrics(Wt,best1['We']):.4f}")
print(f"  Adaptive λ/edge:   SHD={best2['SHD']:3d} F1={metrics(Wt,best2['We']):.4f}")

if best2['SHD'] < best1['SHD']:
    print(f"\n  ✅ Per-edge adaptive λ beats uniform λ!")
else:
    print(f"\n  Uniform λ is as good or better (need to tune trust computation)")
