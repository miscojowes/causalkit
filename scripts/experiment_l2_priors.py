#!/usr/bin/env python3
"""
L2 Prior Experiment: Test whether L2 priors improve NOTEARS accuracy.

Tests 3 conditions:
  a) No prior (lambda_prior=0)
  b) Correct prior (known true edges have prior=0.9)
  c) Misleading prior (known non-edges have prior=0.9)
  
Each bootstrap uses notears_lbfgs with built-in L2 prior support.
Reports SHD, Precision, Recall for each condition.
"""

import sys, os, json, time, gc, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from causbayes.structure_learning.notears_fast import notears_lbfgs
from causbayes.structure_learning.utils import structural_hamming_distance

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiment_results")
os.makedirs(RESULTS_DIR, exist_ok=True)
SEEDS = [42, 43, 44]
LAMBDA_PRIORS = [0.0, 0.05, 0.1, 0.5]


def gen(d=5, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    W = np.zeros((d,d))
    for i in range(d):
        for j in range(i+1,d):
            if rng.random() < 0.2:
                W[i,j] = rng.uniform(0.5,1.5)*rng.choice([-1,1])
    X = np.zeros((n,d))
    for j in range(d):
        p = np.where(W[:,j]!=0)[0]
        if len(p)>0: X[:,j] = X[:,p]@W[p,j]
        X[:,j] += rng.randn(n)*0.1
    return X, (np.abs(W)>1e-6).astype(float)


def bootstrap_with_prior(X, prior_matrix, lambda_prior, n_boot=10, seed=42):
    d = X.shape[1]
    W_list = []
    for i in range(n_boot):
        X_b = resample(X, random_state=seed+i)
        X_b = X_b - X_b.mean(axis=0)
        try:
            Wi = notears_lbfgs(X_b, lambda_1=0.01, max_iter=10, w_threshold=0.1,
                              prior_matrix=prior_matrix, lambda_prior=lambda_prior)
            if not np.isnan(Wi).any():
                W_list.append(Wi)
        except: pass
    if not W_list:
        return np.zeros((d,d)), np.zeros((d,d))
    Wa = np.array(W_list)
    P = np.mean(np.abs(Wa) > 0, axis=0)
    np.fill_diagonal(P, 0)
    S = np.std(np.abs(Wa), axis=0)
    np.fill_diagonal(S, 0)
    return P, S


def eval_metrics(W_true, P):
    Wb = (P >= 0.5).astype(float)
    shd = structural_hamming_distance(W_true, Wb)
    tp = np.sum((Wb>0)&(W_true>0))
    fp = np.sum((Wb>0)&(W_true==0))
    fn = np.sum((Wb==0)&(W_true>0))
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return {"shd":shd,"precision":prec,"recall":rec,"f1":f1,
            "true_edges":int(np.sum(W_true>0)),"est_edges":int(np.sum(Wb>0))}


def main():
    print("="*60)
    print("  L2 Prior Experiment (n_boot=10)")
    print("="*60)
    all_r = []

    for seed in SEEDS:
        print(f"\n  Seed {seed}:")
        X, Wt = gen(seed=seed)
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        d = Xs.shape[1]
        te = int(np.sum(Wt>0))
        print(f"    True edges: {te}")
        row = {"seed":seed,"true_edges":te}

        # Condition A: No prior
        print(f"    ── No Prior ──")
        prior_none = np.zeros((d,d))
        for lp in LAMBDA_PRIORS:
            t0 = time.time()
            P, S = bootstrap_with_prior(Xs, prior_none if lp>0 else None, lp, n_boot=10, seed=seed)
            t = time.time()-t0
            m = eval_metrics(Wt, P)
            row[f"no_prior_lp{lp}"] = {**m, "time_s":round(t,1)}
            print(f"      λ={lp:.2f} SHD={m['shd']:.1f} P={m['precision']:.2f} R={m['recall']:.2f} t={t:.0f}s")

        # Condition B: Correct prior
        print(f"    ── Correct Prior ──")
        prior_c = np.zeros((d,d))
        for i in range(d):
            for j in range(d):
                if Wt[i,j]>0: prior_c[i,j]=0.9
        for lp in [0.05, 0.1, 0.5]:
            t0 = time.time()
            P, S = bootstrap_with_prior(Xs, prior_c, lp, n_boot=10, seed=seed)
            t = time.time()-t0
            m = eval_metrics(Wt, P)
            row[f"correct_prior_lp{lp}"] = {**m, "time_s":round(t,1)}
            print(f"      λ={lp:.2f} SHD={m['shd']:.1f} P={m['precision']:.2f} R={m['recall']:.2f} t={t:.0f}s")

        # Condition C: Misleading prior
        print(f"    ── Misleading Prior ──")
        prior_m = np.zeros((d,d))
        for i in range(d):
            for j in range(d):
                if Wt[i,j]==0 and i!=j: prior_m[i,j]=0.9
        for lp in [0.05, 0.1, 0.5]:
            t0 = time.time()
            P, S = bootstrap_with_prior(Xs, prior_m, lp, n_boot=10, seed=seed)
            t = time.time()-t0
            m = eval_metrics(Wt, P)
            row[f"mislead_prior_lp{lp}"] = {**m, "time_s":round(t,1)}
            print(f"      λ={lp:.2f} SHD={m['shd']:.1f} P={m['precision']:.2f} R={m['recall']:.2f} t={t:.0f}s")

        all_r.append(row)
        gc.collect()

    # Summary
    print(f"\n  SUMMARY:")
    for label, prefix in [("No Prior","no_prior_lp"),("Correct Prior","correct_prior_lp"),("Misleading","mislead_prior_lp")]:
        print(f"\n  {label}:")
        print(f"  {'λ':<6} {'SHD':<8} {'Prec':<8} {'Recall':<8} {'F1':<8}")
        for lp in LAMBDA_PRIORS:
            if prefix == "no_prior_lp":
                k = f"{prefix}{lp}"
            else:
                if lp == 0: continue
                k = f"{prefix}{lp}"
            shds, precs, recs, f1s = [], [], [], []
            for row in all_r:
                r = row.get(k, {})
                if isinstance(r, dict) and "shd" in r:
                    shds.append(r["shd"]); precs.append(r["precision"])
                    recs.append(r["recall"]); f1s.append(r["f1"])
            if shds:
                print(f"  {lp:<6.2f} {np.mean(shds):<8.1f} {np.mean(precs):<8.2f} {np.mean(recs):<8.2f} {np.mean(f1s):<8.2f}")

    # Save
    def cv(o):
        if isinstance(o,(np.integer,)): return int(o)
        if isinstance(o,(np.floating,)): return float(o)
        return o
    with open(os.path.join(RESULTS_DIR,"l2_priors_results.json"),"w") as f:
        json.dump({"by_seed":all_r,"lambda_values":LAMBDA_PRIORS}, f, indent=2, default=cv)
    print(f"\n  Saved to experiment_results/l2_priors_results.json")
    print(f"\n  Done!")

if __name__ == "__main__":
    main()
