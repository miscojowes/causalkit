#!/usr/bin/env python3
"""
Calibration Experiment: Platt scaling (built-in) vs Isotonic Regression.
Uses n_bootstraps=10, 3 seeds for speed.
"""

import sys, os, json, time, gc, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from causbayes import BootstrapDAG
from causbayes.evaluation import edge_calibration

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiment_results")
os.makedirs(RESULTS_DIR, exist_ok=True)
SEEDS = [42, 43, 44]


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


def run():
    print("="*60)
    print("  Calibration: Platt vs Isotonic (n_bootstraps=10)")
    print("="*60)
    all_r = []
    last = None

    for seed in SEEDS:
        print(f"\n  Seed {seed}:", end=" ", flush=True)
        X_all, Wt = gen(seed=seed)
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_all[:600])
        X_va = sc.transform(X_all[600:800])
        sc2 = StandardScaler()
        sc2.fit(X_all[:600])
        X_te = sc2.transform(X_all[800:])

        # Bootstrap(10) with Platt scaling
        t0 = time.time()
        m = BootstrapDAG(n_bootstraps=10, lambda_1=0.01, max_iter=10, calibrate=True, verbose=False)
        m.fit(X_tr, X_val=X_va, W_val=Wt)
        t = time.time()-t0
        P_raw = m.edge_probs_raw.copy()
        P_platt = m.edge_probs.copy()
        print(f"boot={len(m._weight_matrices_)}/{m.n_bootstraps} t={t:.0f}s")

        ece_raw = edge_calibration(Wt, P_raw, 10)["ece"]
        ece_platt = edge_calibration(Wt, P_platt, 10)["ece"]

        # Isotonic
        d = 5
        yv, pv = [], []
        for i in range(d):
            for j in range(d):
                if i!=j:
                    yv.append(1 if Wt[i,j]>0.5 else 0)
                    pv.append(P_raw[i,j])
        try:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(np.array(pv), np.array(yv))
            P_iso = P_raw.copy()
            idx = 0
            for i in range(d):
                for j in range(d):
                    if i!=j:
                        P_iso[i,j] = float(iso.predict([P_raw[i,j]])[0])
                        idx+=1
            np.fill_diagonal(P_iso, 0)
            ece_iso = edge_calibration(Wt, P_iso, 10)["ece"]
        except Exception as e:
            P_iso = P_raw.copy(); ece_iso = float("nan")

        print(f"    ECE: raw={ece_raw:.4f} platt={ece_platt:.4f} iso={ece_iso:.4f}")
        all_r.append({"seed":seed,"ece_raw":ece_raw,"ece_platt":ece_platt,"ece_iso":ece_iso,"time_s":t})
        last = (Wt, P_raw, P_platt, P_iso, seed)
        gc.collect()

    # Summary
    print(f"\n  SUMMARY:")
    for name, key in [("Raw","ece_raw"),("Platt","ece_platt"),("Isotonic","ece_iso")]:
        v = [r[key] for r in all_r if not np.isnan(r[key])]
        if v: print(f"  {name:<10} ECE = {np.mean(v):.4f} ± {np.std(v):.4f}")

    # Save
    def cv(o):
        if isinstance(o,(np.integer,)): return int(o)
        if isinstance(o,(np.floating,)): return float(o)
        return o
    out = {"by_seed":all_r,"summary":{k:{"mean":float(np.mean([r[k] for r in all_r])),"std":float(np.std([r[k] for r in all_r]))} for k in ["ece_raw","ece_platt","ece_iso"]}}
    with open(os.path.join(RESULTS_DIR,"calibration_results.json"),"w") as f:
        json.dump(out,f,indent=2,default=cv)

    # Plot
    if last:
        Wt, Pr, Pp, Pi, s = last
        fig, axs = plt.subplots(1,3,figsize=(15,5))
        for idx,(nm,P,c) in enumerate([("Raw",Pr,"C0"),("Platt",Pp,"C1"),("Isotonic",Pi,"C2")]):
            cal = edge_calibration(Wt,P,10)
            ax = axs[idx]
            ax.plot([0,1],[0,1],"k--",alpha=0.5)
            ax.plot(cal["bins"],cal["accuracy"],"o-",color=c,label=f"ECE={cal['ece']:.4f}")
            ax.fill_between(cal["bins"],0,cal["accuracy"],alpha=0.1,color=c)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Observed")
            ax.set_title(nm); ax.set_xlim(0,1); ax.set_ylim(0,1)
            ax.legend(loc="lower right"); ax.grid(True,alpha=0.3)
        plt.suptitle(f"Calibration Curves (d=5, Bootstrap=10)",fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR,"calibration_curves.png"),dpi=150,bbox_inches="tight")
        plt.close()
        print(f"  Plot saved")
    print(f"\n  Done!")

if __name__ == "__main__":
    run()
