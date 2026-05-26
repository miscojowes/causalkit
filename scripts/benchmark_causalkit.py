#!/usr/bin/env python3
"""BENCHMARK: causalkit vs gCastle GES — quick version"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from castle.algorithms import GES
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
import causalkit as ck

np.random.seed(42)

def load(name):
    if name == "sachs":
        df = pd.read_csv("experiment_results/sachs_raw.csv", sep="\t")
        X = StandardScaler().fit_transform(df.values.astype(float))
        gt = [("Raf","Mek"),("Mek","Erk"),("Plcg","PIP2"),("PIP2","PIP3"),("PIP3","Akt"),
              ("PKC","PKA"),("PKA","Jnk"),("PKC","Jnk"),("Jnk","P38"),("PKC","P38"),
              ("Erk","Akt"),("PKC","Akt"),("PKA","Akt"),("PIP3","PKA"),("PIP3","Plcg"),
              ("PKA","Erk"),("PKC","Mek")]
        Wt = np.zeros((X.shape[1], X.shape[1]))
        for c,e in gt: Wt[list(df.columns).index(c), list(df.columns).index(e)] = 1.0
        return X, Wt
    dag_file = f"experiment_results/{name}_dag.csv"
    dat_file = f"experiment_results/{name}_data.csv"
    if not os.path.exists(dag_file): return None, None
    dag_df = pd.read_csv(dag_file, index_col=0)
    Wt = dag_df.values.astype(float)
    dat_df = pd.read_csv(dat_file)
    Xl = []
    for col in dat_df.columns:
        try:
            Xl.append(pd.to_numeric(dat_df[col], errors='raise').values.astype(float))
        except (ValueError, TypeError):
            Xl.append(LabelEncoder().fit_transform(dat_df[col].astype(str)).astype(float))
    X = StandardScaler().fit_transform(np.column_stack(Xl))
    return X, Wt

def m(Wt, We):
    tp=int(np.sum((We>0)&(Wt>0))); fp=int(np.sum((We>0)&(Wt==0))); fn=int(np.sum((We==0)&(Wt>0)))
    return (round(2*tp/max(2*tp+fp+fn,1),4), fp+fn, int(We.sum()), tp, fp)

def prior(Wt, pct=0.7):
    prior = np.full(Wt.shape, 0.5); np.fill_diagonal(prior, 0.0)
    te = np.argwhere(Wt > 0); np.random.shuffle(te)
    for i in range(int(len(te)*pct)): prior[te[i][0], te[i][1]] = 0.9
    return prior

datasets = [
    ("sachs (real)", load("sachs")),
    ("cancer", load("cancer")),
    ("earthquake", load("earthquake")),
    ("survey", load("survey")),
    ("asia", load("asia")),
]

# Use very few bootstraps on discrete data
N_BOOT = 10  # fast

print("=" * 65)
print("  causalkit vs GES — Benchmark")
print("=" * 65)
print(f"{'Dataset':<20s} {'d':>3s} {'E':>3s} {'GES':>8s} {'CK':>8s} {'CK+P':>8s} {'Δ':>8s}")
print("-" * 60)

results = []
for name, (X, Wt) in datasets:
    if X is None: continue
    n, d = X.shape; ne = int(Wt.sum())
    best_nb = N_BOOT

    # GES
    t0 = time.time()
    ges = GES(); ges.learn(X)
    Wges = (np.abs(np.array(ges.causal_matrix, dtype=float).T) > 0.3).astype(float)
    ges_f1, _, _, _, _ = m(Wt, Wges)

    # CK no prior
    ck1 = ck.CausalDiscoverer(method="bootstrap", n_bootstraps=best_nb, verbose=False)
    ck1.fit(X)
    ck_f1, _, _, _, _ = m(Wt, ck1.causal_matrix_)

    # CK + prior
    pr = prior(Wt, 0.7)
    ck2 = ck.CausalDiscoverer(method="bootstrap", n_bootstraps=best_nb, lambda_prior=0.5, verbose=False)
    ck2.fit(X, prior_matrix=pr)
    ckp_f1, _, _, _, _ = m(Wt, ck2.causal_matrix_)

    delta = ckp_f1 - ges_f1
    mark = "✅" if delta > 0.01 else "≈" if abs(delta) <= 0.01 else "❌"
    print(f"{mark} {name:<18s} {d:3d} {ne:3d} {ges_f1:>8.4f} {ck_f1:>8.4f} {ckp_f1:>8.4f} {delta:>+8.4f}")
    results.append((name, d, ne, ges_f1, ck_f1, ckp_f1, delta))

wins = sum(1 for r in results if r[6] > 0.01)
ties = sum(1 for r in results if abs(r[6]) <= 0.01)
losses = sum(1 for r in results if r[6] < -0.01)
print(f"\n  CK+P vs GES: {wins} wins, {ties} ties, {losses} losses (out of {len(results)})")
print(f"\n--- causalkit v{ck.__version__} benchmark complete!")
