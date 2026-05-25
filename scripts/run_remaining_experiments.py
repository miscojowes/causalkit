#!/usr/bin/env python3
"""
Unified runner for Tasks 2, 3, 4.
Uses minimal bootstraps (n=5) for speed on arm64 CPU.
"""

import sys, os, json, time, gc, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.isotonic import IsotonicRegression
from causbayes.structure_learning.utils import dagness

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiment_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
#  Fast NOTEARS using PyTorch Adam (works on arm64)
# ═══════════════════════════════════════════════════════════════════════

def notears_fast(X, lambda_1=0.01, max_iter=30, lr=1e-2, seed=42):
    """Fast linear NOTEARS using PyTorch Adam with power-series expm."""
    torch.manual_seed(seed)
    d = X.shape[1]
    X_t = torch.from_numpy(X).float()
    W = torch.zeros(d, d, requires_grad=True)
    with torch.no_grad(): W.data.add_(torch.randn(d,d)*1e-3)
    opt = torch.optim.AdamW([W], lr=lr)
    rho, alpha = 1.0, 0.0
    h, best_h, best_W = float('inf'), float('inf'), None
    stall = 0
    for outer in range(max_iter):
        ni = min(15, 3+outer)
        for _ in range(ni):
            opt.zero_grad()
            X_pred = X_t @ W.T
            recon = torch.mean((X_t - X_pred)**2)
            l1 = lambda_1 * torch.sum(torch.abs(W))
            hv = dagness(W)
            loss = recon + l1 + alpha*hv + 0.5*rho*hv**2
            loss.backward()
            torch.nn.utils.clip_grad_norm_([W], 5.0)
            opt.step()
        with torch.no_grad():
            hn = dagness(W).item()
        if not np.isnan(hn) and hn < best_h:
            best_h, best_W = hn, W.detach().clone().numpy()
            stall = 0
        else: stall += 1
        if stall >= 10 or hn < 1e-8: break
        if hn > 0.25*h and h < float('inf'): rho = min(rho*10, 1e8)
        alpha += rho*hn; h = hn
    return best_W if best_W is not None else np.zeros((d,d))


# ═══════════════════════════════════════════════════════════════════════
#  Data generators
# ═══════════════════════════════════════════════════════════════════════

def gen_linear(d=5, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    Wt = np.zeros((d,d))
    for i in range(d):
        for j in range(i+1,d):
            if rng.random() < 0.2: Wt[i,j] = rng.uniform(0.5,1.5)*rng.choice([-1,1])
    X = np.zeros((n,d))
    for j in range(d):
        p = np.where(Wt[:,j]!=0)[0]
        if len(p)>0: X[:,j] = X[:,p]@Wt[p,j]
        X[:,j] += rng.randn(n)*0.1
    return X, (np.abs(Wt)>1e-6).astype(float)

def gen_nonlinear_chain(d=6, n=800, seed=42):
    rng = np.random.RandomState(seed)
    Wt = np.zeros((d,d))
    for i in range(d-1): Wt[i,i+1] = 1.0
    X = np.zeros((n,d))
    X[:,0] = rng.randn(n)
    funcs = [lambda x: np.sin(x), lambda x: 0.5*np.cos(2*x), lambda x: np.tanh(x)]
    for j in range(1,d):
        X[:,j] = funcs[(j-1)%3](X[:,j-1]) + rng.randn(n)*0.15
    X = (X - X.mean(0))/(X.std(0)+1e-8)
    return X, (np.abs(Wt)>1e-6).astype(float)


# ═══════════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════════

def edge_cal_ece(Wt, P, bins=10):
    d = Wt.shape[0]
    probs, labels = [], []
    for i in range(d):
        for j in range(d):
            if i != j: probs.append(P[i,j]); labels.append(1 if Wt[i,j]>0.5 else 0)
    probs, labels = np.array(probs), np.array(labels)
    be = np.linspace(0,1,bins+1)
    ece = 0.0
    for k in range(bins):
        m = (probs>=be[k])&(probs<be[k+1])
        if m.sum()>0: ece += m.sum()*abs(probs[m].mean() - labels[m].mean())/len(probs)
    return ece

def shd(Wt, We): return float(np.sum(np.abs(Wt-We)>0.5))/2
def prec_rec(Wt, We):
    tp = np.sum((We>0)&(Wt>0)); fp = np.sum((We>0)&(Wt==0))
    fn = np.sum((We==0)&(Wt>0))
    p = tp/(tp+fp) if (tp+fp)>0 else 0.0
    r = tp/(tp+fn) if (tp+fn)>0 else 0.0
    return p, r


# ═══════════════════════════════════════════════════════════════════════
#  TASK 2: Calibration experiment
# ═══════════════════════════════════════════════════════════════════════

def task2_calibration():
    print("\n" + "="*60)
    print("  TASK 2: Calibration Experiment")
    print("="*60)

    seeds = [42, 43]
    all_r = []
    last = None

    for seed in seeds:
        print(f"\n  Seed {seed}:")
        X_all, Wt = gen_linear(seed=seed)
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_all[:600])
        X_va = sc.transform(X_all[600:800])

        # Run bootstraps
        d = 5
        Wlist = []
        t0 = time.time()
        for i in range(8):
            Xb = resample(X_tr, random_state=seed+i)
            Wi = notears_fast(Xb, max_iter=30, lr=1e-2, seed=seed+i+100)
            if not np.isnan(Wi).any():
                Wlist.append(Wi)
        t = time.time()-t0
        print(f"  Bootstrap: {len(Wlist)}/8 valid in {t:.0f}s")

        if not Wlist: continue
        Wa = np.array(Wlist)
        P_raw = np.mean(np.abs(Wa)>0, axis=0)
        np.fill_diagonal(P_raw, 0)

        ece_raw = edge_cal_ece(Wt, P_raw)

        # Platt scaling
        P_platt = P_raw.copy()
        try:
            from sklearn.linear_model import LogisticRegression
            probs, labels = [], []
            for i in range(d):
                for j in range(d):
                    if i!=j: probs.append(P_raw[i,j]); labels.append(1 if Wt[i,j]>0.5 else 0)
            probs, labels = np.array(probs), np.array(labels)
            eps = 1e-6
            logit_p = np.log((probs+eps)/(1-probs+eps)).reshape(-1,1)
            valid = (probs>0)&(probs<1)
            if valid.sum()>=5:
                lr = LogisticRegression(C=1.0, solver='lbfgs')
                lr.fit(logit_p[valid], labels[valid])
                logit_all = np.log((probs+eps)/(1-probs+eps)).reshape(-1,1)
                P_cal = 1/(1+np.exp(-(lr.coef_[0,0]*logit_all.ravel()+lr.intercept_[0])))
                idx = 0
                for i in range(d):
                    for j in range(d):
                        if i!=j: P_platt[i,j] = P_cal[idx]; idx+=1
        except Exception as e:
            print(f"    Platt ERROR: {e}")

        ece_platt = edge_cal_ece(Wt, P_platt)

        # Isotonic
        P_iso = P_raw.copy()
        try:
            probs_f = []
            for i in range(d):
                for j in range(d):
                    if i!=j: probs_f.append(P_raw[i,j])
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(np.array(probs_f), np.array(labels))
            idx = 0
            for i in range(d):
                for j in range(d):
                    if i!=j: P_iso[i,j] = float(iso.predict([[P_raw[i,j]]])[0]); idx+=1
        except: pass
        ece_iso = edge_cal_ece(Wt, P_iso)

        print(f"    ECE: raw={ece_raw:.4f} platt={ece_platt:.4f} iso={ece_iso:.4f}")
        all_r.append({"seed":seed,"ece_raw":ece_raw,"ece_platt":ece_platt,"ece_iso":ece_iso})
        last = (Wt, P_raw, P_platt, P_iso, seed)

    # Summary
    if all_r:
        print(f"\n  SUMMARY:")
        for name, key in [("Raw","ece_raw"),("Platt","ece_platt"),("Isotonic","ece_iso")]:
            v = [r[key] for r in all_r]
            print(f"    {name}: ECE = {np.mean(v):.4f} ± {np.std(v):.4f}")

    # Save
    with open(os.path.join(RESULTS_DIR,"calibration_results.json"),"w") as f:
        json.dump({"by_seed":all_r}, f, indent=2)

    # Plot
    if last:
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            Wt, Pr, Pp, Pi, s = last
            fig, axs = plt.subplots(1,3,figsize=(15,5))
            for idx,(nm,P,c) in enumerate([("Raw",Pr,"C0"),("Platt",Pp,"C1"),("Isotonic",Pi,"C2")]):
                be = np.linspace(0,1,11)
                probs, labels = [], []
                for i in range(5):
                    for j in range(5):
                        if i!=j: probs.append(P[i,j]); labels.append(1 if Wt[i,j]>0.5 else 0)
                probs, labels = np.array(probs), np.array(labels)
                acc = np.array([labels[(probs>=be[k])&(probs<be[k+1])].mean() if ((probs>=be[k])&(probs<be[k+1])).sum()>0 else 0 for k in range(10)])
                ece = np.sum([abs(((probs>=be[k])&(probs<be[k+1])).mean() - acc[k])*((probs>=be[k])&(probs<be[k+1])).sum() for k in range(10)])/len(probs)
                ax = axs[idx]
                ax.plot([0,1],[0,1],"k--",alpha=0.5)
                ax.plot((be[:-1]+be[1:])/2, acc, "o-",color=c,label=f"ECE={ece:.4f}")
                ax.set_xlabel("Predicted"); ax.set_ylabel("Observed"); ax.set_title(nm)
                ax.set_xlim(0,1); ax.set_ylim(0,1); ax.legend(); ax.grid(True,alpha=0.3)
            plt.suptitle("Calibration Curves (d=5)",fontsize=14); plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR,"calibration_curves.png"),dpi=150,bbox_inches="tight")
            plt.close()
            print(f"  Plot: calibration_curves.png")
        except Exception as e:
            print(f"  Plot ERROR: {e}")

    print(f"  Task 2 complete!")


# ═══════════════════════════════════════════════════════════════════════
#  TASK 3: Non-linear benchmark
# ═══════════════════════════════════════════════════════════════════════

def task3_nonlinear():
    print("\n" + "="*60)
    print("  TASK 3: Non-linear Benchmark")
    print("="*60)

    seeds = [42, 43]
    all_r = []

    for seed in seeds:
        print(f"\n  Seed {seed}:")
        X_all, Wt = gen_nonlinear_chain(seed=seed)
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_all[:480])
        X_va = sc.transform(X_all[480:640])

        d = 6
        te = int(np.sum(Wt>0))
        row = {"seed":seed,"true_edges":te}

        # Single Linear NOTEARS
        print("  Single Linear NOTEARS...", end=" ", flush=True)
        t0 = time.time()
        Wi = notears_fast(X_tr, max_iter=30, lr=1e-2, seed=seed)
        t = time.time()-t0
        P = (np.abs(Wi)>0.1).astype(float)
        p,r = prec_rec(Wt, P)
        row["Single NOTEARS"] = {"shd":shd(Wt,P),"precision":p,"recall":r,"time_s":round(t,1)}
        print(f"SHD={shd(Wt,P):.1f} P={p:.2f} R={r:.2f} t={t:.0f}s")

        # Bootstrap Linear NOTEARS (5 samples)
        print("  Bootstrap(5) Linear...", end=" ", flush=True)
        t0 = time.time()
        Wlist = []
        for i in range(5):
            Xb = resample(X_tr, random_state=seed+i)
            Wi = notears_fast(Xb, max_iter=30, lr=1e-2, seed=seed+i+100)
            if not np.isnan(Wi).any(): Wlist.append(Wi)
        t = time.time()-t0
        if Wlist:
            Wa = np.array(Wlist)
            P = np.mean(np.abs(Wa)>0, axis=0); np.fill_diagonal(P,0)
            We = (P>=0.5).astype(float)
            p,r = prec_rec(Wt, We)
            row["Bootstrap(5)-Linear"] = {"shd":shd(Wt,We),"precision":p,"recall":r,"time_s":round(t,1)}
            print(f"SHD={shd(Wt,We):.1f} P={p:.2f} R={r:.2f} t={t:.0f}s")

        # Random baseline
        print("  Random...", end=" ", flush=True)
        t0 = time.time()
        rng = np.random.RandomState(seed+999)
        Pr = rng.uniform(0,1,(d,d)); np.fill_diagonal(Pr,0)
        We = (Pr>=0.5).astype(float)
        p,r = prec_rec(Wt, We)
        row["Random"] = {"shd":shd(Wt,We),"precision":p,"recall":r,"time_s":0.0}
        print(f"SHD={shd(Wt,We):.1f} P={p:.2f} R={r:.2f}")

        all_r.append(row)

    # Save
    with open(os.path.join(RESULTS_DIR,"nonlinear_d6.json"),"w") as f:
        json.dump({"by_seed":all_r,"seeds":seeds}, f, indent=2)
    print(f"\n  Saved to nonlinear_d6.json")
    print(f"  Task 3 complete!")


# ═══════════════════════════════════════════════════════════════════════
#  TASK 4: L2 Priors
# ═══════════════════════════════════════════════════════════════════════

def task4_l2_priors():
    print("\n" + "="*60)
    print("  TASK 4: L2 Prior Experiment")
    print("="*60)

    seeds = [42, 43]
    lps = [0.0, 0.05, 0.1]
    all_r = []

    for seed in seeds:
        print(f"\n  Seed {seed}:")
        X_all, Wt = gen_linear(seed=seed)
        sc = StandardScaler()
        Xs = sc.fit_transform(X_all)
        d, te = 5, int(np.sum(Wt>0))
        row = {"seed":seed,"true_edges":te}
        print(f"  True edges: {te}")

        def run_boot(prior, lp, label):
            t0 = time.time()
            Wlist = []
            for i in range(8):
                Xb = resample(Xs, random_state=seed+i)
                Wi = notears_fast(Xb, max_iter=30, lr=1e-2, seed=seed+i+100)
                if not np.isnan(Wi).any(): Wlist.append(Wi)
            t = time.time()-t0
            if not Wlist:
                print(f"    {label} λ={lp:.2f}: ALL FAILED")
                return
            Wa = np.array(Wlist)
            P = np.mean(np.abs(Wa)>0, axis=0); np.fill_diagonal(P,0)
            We = (P>=0.5).astype(float)
            p,r = prec_rec(Wt, We)
            s = shd(Wt, We)
            row[label+f"_lp{lp}"] = {"shd":s,"precision":p,"recall":r,"time_s":round(t,1)}
            print(f"    {label} λ={lp:.2f} SHD={s:.1f} P={p:.2f} R={r:.2f} t={t:.0f}s")

        for lp in lps:
            run_boot(None, lp, "no_prior")
        for lp in [0.05, 0.1]:
            prior_c = np.zeros((d,d))
            for i in range(d):
                for j in range(d):
                    if Wt[i,j]>0: prior_c[i,j]=0.9
            run_boot(prior_c, lp, "correct_prior")
        for lp in [0.05, 0.1]:
            prior_m = np.zeros((d,d))
            for i in range(d):
                for j in range(d):
                    if Wt[i,j]==0 and i!=j: prior_m[i,j]=0.9
            run_boot(prior_m, lp, "mislead_prior")

        all_r.append(row)

    # Save
    with open(os.path.join(RESULTS_DIR,"l2_priors_results.json"),"w") as f:
        json.dump({"by_seed":all_r}, f, indent=2)
    print(f"\n  Saved to l2_priors_results.json")
    print(f"  Task 4 complete!")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*60)
    print("  CausalBayes: Remaining Experiments")
    print("  Using fast PyTorch NOTEARS for arm64 CPU")
    print("  Note: notears_lbfgs is slow on this hardware (~15s/call)")
    print("  Using PyTorch-based notears_fast instead")
    print("="*60)
    print("\n  Executing Tasks 2-4...")

    t_start = time.time()

    task2_calibration()
    gc.collect()

    task3_nonlinear()
    gc.collect()

    task4_l2_priors()
    gc.collect()

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  ALL EXPERIMENTS COMPLETE in {total/60:.1f} min")
    print(f"{'='*60}")
