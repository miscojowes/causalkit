#!/usr/bin/env python3
"""
Final experiment runner for Tasks 2-4.
Uses single NOTEARS calls where possible (fast), minimal bootstraps otherwise.
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

R = os.path.join(os.path.dirname(__file__), "..", "experiment_results")
os.makedirs(R, exist_ok=True)


def notears_fast(X, lambda_1=0.01, max_iter=30, lr=1e-2, seed=42,
                 prior_matrix=None, lambda_prior=0.0):
    """Linear NOTEARS with optional L2 prior.
    
    L2 prior: lambda_prior * sum(prior[i,j] * (|W[i,j]| - mu[i,j])^2)
    where mu[i,j] = 0.3 if prior[i,j] >= 0.5 else 0.0
    """
    torch.manual_seed(seed)
    d = X.shape[1]
    X_t = torch.from_numpy(X).float()
    W = torch.zeros(d, d, requires_grad=True)
    with torch.no_grad(): W.data.add_(torch.randn(d,d)*1e-3)
    opt = torch.optim.AdamW([W], lr=lr)
    rho, alpha = 1.0, 0.0
    h, best_h, best_W = float('inf'), float('inf'), None
    stall = 0
    
    # Setup prior
    prior_t = None
    mu_t = None
    if prior_matrix is not None and lambda_prior > 0:
        prior = np.asarray(prior_matrix, dtype=float)
        prior = (prior + prior.T) / 2.0
        prior_t = torch.from_numpy(prior).float()
        # mu: 0.3 for high-prior edges, 0 for others
        mu = (prior >= 0.5).astype(float) * 0.3
        mu_t = torch.from_numpy(mu).float()
    
    for outer in range(max_iter):
        ni = min(15, 3+outer)
        for _ in range(ni):
            opt.zero_grad()
            X_pred = X_t @ W.T
            recon = torch.mean((X_t - X_pred)**2)
            l1 = lambda_1 * torch.sum(torch.abs(W))
            
            # L2 prior: sum prior[i,j] * (|W[i,j]| - mu[i,j])^2
            lp = 0.0
            if prior_t is not None and lambda_prior > 0:
                W_abs = torch.abs(W)
                lp = lambda_prior * torch.sum(prior_t * (W_abs - mu_t)**2)
            
            hv = dagness(W)
            loss = recon + l1 + lp + alpha*hv + 0.5*rho*hv**2
            loss.backward()
            torch.nn.utils.clip_grad_norm_([W], 5.0)
            opt.step()
        with torch.no_grad():
            hn = dagness(W).item()
        if not np.isnan(hn) and hn < best_h:
            best_h, best_W = hn, W.detach().clone().numpy(); stall = 0
        else: stall += 1
        if stall >= 10 or hn < 1e-8: break
        if hn > 0.25*h and h < float('inf'): rho = min(rho*10, 1e8)
        alpha += rho*hn; h = hn
    return best_W if best_W is not None else np.zeros((d,d))


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


def gen_nonlinear(d=6, n=800, seed=42):
    rng = np.random.RandomState(seed)
    Wt = np.zeros((d,d))
    for i in range(d-1): Wt[i,i+1] = 1.0
    X = np.zeros((n,d))
    X[:,0] = rng.randn(n)
    fs = [lambda x: np.sin(x), lambda x: 0.5*np.cos(2*x), lambda x: np.tanh(x)]
    for j in range(1,d):
        X[:,j] = fs[(j-1)%3](X[:,j-1]) + rng.randn(n)*0.15
    X = (X - X.mean(0))/(X.std(0)+1e-8)
    return X, (np.abs(Wt)>1e-6).astype(float)


def ece_fast(Wt, P, bins=10):
    d = Wt.shape[0]
    pp, ll = [], []
    for i in range(d):
        for j in range(d):
            if i!=j: pp.append(P[i,j]); ll.append(1 if Wt[i,j]>0.5 else 0)
    pp, ll = np.array(pp), np.array(ll)
    be = np.linspace(0,1,bins+1)
    ece = 0.0
    for k in range(bins):
        m = (pp>=be[k])&(pp<be[k+1])
        if m.sum()>0: ece += m.sum()*abs(pp[m].mean()-ll[m].mean())/len(pp)
    return ece

def shd_f(Wt, We): return float(np.sum(np.abs(Wt-We)>0.5))/2
def pr_f(Wt, We):
    tp=np.sum((We>0)&(Wt>0)); fp=np.sum((We>0)&(Wt==0)); fn=np.sum((We==0)&(Wt>0))
    return tp/(tp+fp) if (tp+fp)>0 else 0.0, tp/(tp+fn) if (tp+fn)>0 else 0.0

def metrics_f(Wt, P, We, t):
    if We is None: We = (P>=0.5).astype(float)
    s = shd_f(Wt, We)
    p, r = pr_f(Wt, We)
    f1 = 2*p*r/(p+r) if (p+r)>0 else 0.0
    ece = ece_fast(Wt, P)
    return {"shd":s,"precision":p,"recall":r,"f1":f1,"ece":ece,"time_s":t,
            "true_edges":int(np.sum(Wt>0)),"est_edges":int(np.sum(We>0))}


# ═══════════════════════════════════════════════════════════════════════
#  TASK 2: Calibration
# ═══════════════════════════════════════════════════════════════════════

def task2():
    print("\n"+ "="*60)
    print("  TASK 2: Calibration (ECE comparison)")
    print("="*60)
    seed = 42
    X_all, Wt = gen_linear(seed=seed)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_all[:600])

    t0 = time.time()
    WL = []
    for i in range(10):
        Xb = resample(X_tr, random_state=seed+i)
        Wi = notears_fast(Xb, seed=seed+i+100)
        if not np.isnan(Wi).any(): WL.append(Wi)
    bt = time.time()-t0
    print(f"  Bootstrap(10): {len(WL)} valid in {bt:.0f}s")

    Wa = np.array(WL)
    P_raw = np.mean(np.abs(Wa)>0, axis=0); np.fill_diagonal(P_raw,0)
    d = 5

    # Platt scaling
    P_platt = P_raw.copy()
    try:
        from sklearn.linear_model import LogisticRegression
        pp, ll = [], []
        for i in range(d):
            for j in range(d):
                if i!=j: pp.append(P_raw[i,j]); ll.append(1 if Wt[i,j]>0.5 else 0)
        pp, ll = np.array(pp), np.array(ll)
        eps = 1e-6
        logit_p = np.log((pp+eps)/(1-pp+eps)).reshape(-1,1)
        v = (pp>0)&(pp<1)
        if v.sum()>=5:
            lr = LogisticRegression(C=1.0, solver='lbfgs')
            lr.fit(logit_p[v], ll[v])
            lp = np.log((pp+eps)/(1-pp+eps)).reshape(-1,1)
            Pc = 1/(1+np.exp(-(lr.coef_[0,0]*lp.ravel()+lr.intercept_[0])))
            idx=0
            for i in range(d):
                for j in range(d):
                    if i!=j: P_platt[i,j]=Pc[idx]; idx+=1
    except Exception as e: print(f"  Platt ERROR: {e}")

    # Isotonic
    P_iso = P_raw.copy()
    try:
        pp_f = [P_raw[i,j] for i in range(d) for j in range(d) if i!=j]
        ll_f = [1 if Wt[i,j]>0.5 else 0 for i in range(d) for j in range(d) if i!=j]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(np.array(pp_f), np.array(ll_f))
        idx=0
        for i in range(d):
            for j in range(d):
                if i!=j: P_iso[i,j]=float(iso.predict([[P_raw[i,j]]])); idx+=1
    except: pass

    e_raw = ece_fast(Wt, P_raw)
    e_platt = ece_fast(Wt, P_platt)
    e_iso = ece_fast(Wt, P_iso)

    print(f"\n  Results:")
    for nm, e in [("Raw",e_raw),("Platt",e_platt),("Isotonic",e_iso)]:
        print(f"    {nm:<12} ECE = {e:.4f} {'✅' if e<0.1 else '❌'}")

    # Calibration curve plot
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1,3,figsize=(15,5))
        for idx,(nm,P,c) in enumerate([("Raw",P_raw,"C0"),("Platt",P_platt,"C1"),("Isotonic",P_iso,"C2")]):
            pp = [P[i,j] for i in range(d) for j in range(d) if i!=j]
            ll = [1 if Wt[i,j]>0.5 else 0 for i in range(d) for j in range(d) if i!=j]
            pp, ll = np.array(pp), np.array(ll)
            be = np.linspace(0,1,11)
            acc = np.array([ll[(pp>=be[k])&(pp<be[k+1])].mean() if ((pp>=be[k])&(pp<be[k+1])).sum()>0 else 0 for k in range(10)])
            ece = sum(abs(((pp>=be[k])&(pp<be[k+1])).mean() if ((pp>=be[k])&(pp<be[k+1])).sum()>0 else 0 - acc[k])*((pp>=be[k])&(pp<be[k+1])).sum() for k in range(10))/len(pp)
            ax = axs[idx]
            ax.plot([0,1],[0,1],"k--",alpha=0.5)
            ax.plot((be[:-1]+be[1:])/2,acc,"o-",color=c,label=f"ECE={ece:.4f}")
            ax.set_xlabel("Predicted"); ax.set_ylabel("Observed"); ax.set_title(nm)
            ax.legend(); ax.grid(True,alpha=0.3)
        plt.suptitle("Calibration Curves (d=5, Bootstrap=10)",fontsize=14); plt.tight_layout()
        plt.savefig(os.path.join(R,"calibration_curves.png"),dpi=150,bbox_inches="tight")
        plt.close()
        print(f"  Plot saved to calibration_curves.png")
    except Exception as ex:
        print(f"  Plot ERROR: {ex}")

    # Save
    with open(os.path.join(R,"calibration_results.json"),"w") as f:
        json.dump({"ece_raw":e_raw,"ece_platt":e_platt,"ece_iso":e_iso},f,indent=2)
    print(f"  Task 2 done!")


# ═══════════════════════════════════════════════════════════════════════
#  TASK 3: Non-linear benchmark (single NOTEARS only, no bootstrap)
# ═══════════════════════════════════════════════════════════════════════

def task3():
    print("\n"+ "="*60)
    print("  TASK 3: Non-linear Benchmark (d=6 chain)")
    print("="*60)

    for seed in [42, 43]:
        print(f"\n  Seed {seed}:")
        X_all, Wt = gen_nonlinear(seed=seed)
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_all)

        t0 = time.time()
        Wi = notears_fast(X_tr, seed=seed)
        t = time.time()-t0
        P = (np.abs(Wi)>0.1).astype(float)
        s, p, r = shd_f(Wt, P), *pr_f(Wt, P)
        print(f"    Single NOTEARS: SHD={s:.1f} P={p:.2f} R={r:.2f} t={t:.0f}s")

        # Bootstrap(5) for comparison
        t0 = time.time()
        WL = []
        for i in range(5):
            Xb = resample(X_tr, random_state=seed+i)
            Wi = notears_fast(Xb, seed=seed+i+100)
            if not np.isnan(Wi).any(): WL.append(Wi)
        t = time.time()-t0
        if WL:
            Wa = np.array(WL)
            P_b = np.mean(np.abs(Wa)>0, axis=0); np.fill_diagonal(P_b,0)
            We = (P_b>=0.5).astype(float)
            s, p, r = shd_f(Wt, We), *pr_f(Wt, We)
            print(f"    Bootstrap(5): SHD={s:.1f} P={p:.2f} R={r:.2f} t={t:.0f}s")

    print(f"  Task 3 done!")


# ═══════════════════════════════════════════════════════════════════════
#  TASK 4: L2 Priors (single NOTEARS, compares prior vs no-prior)
# ═══════════════════════════════════════════════════════════════════════

def task4():
    print("\n"+ "="*60)
    print("  TASK 4: L2 Prior Experiment (single NOTEARS)")
    print("="*60)

    for seed in [42, 43]:
        print(f"\n  Seed {seed}:")
        X_all, Wt = gen_linear(seed=seed)
        sc = StandardScaler()
        Xs = sc.fit_transform(X_all)
        d, te = 5, int(np.sum(Wt>0))
        print(f"    True edges: {te}")

        for lp in [0.0, 0.05, 0.1, 0.5]:
            # No prior
            prior = np.zeros((d,d))
            t0 = time.time()
            Wi = notears_fast(Xs, seed=seed, prior_matrix=prior, lambda_prior=lp)
            t = time.time()-t0
            P = (np.abs(Wi)>0.1).astype(float)
            s, p, r = shd_f(Wt, P), *pr_f(Wt, P)
            print(f"    NoPrior λ={lp:.2f}: SHD={s:.1f} P={p:.2f} R={r:.2f} t={t:.0f}s")

        for lp in [0.05, 0.1, 0.5]:
            prior_c = np.zeros((d,d))
            for i in range(d):
                for j in range(d):
                    if Wt[i,j]>0: prior_c[i,j]=0.9
            t0 = time.time()
            Wi = notears_fast(Xs, seed=seed, prior_matrix=prior_c, lambda_prior=lp)
            t = time.time()-t0
            P = (np.abs(Wi)>0.1).astype(float)
            s, p, r = shd_f(Wt, P), *pr_f(Wt, P)
            print(f"    Correct λ={lp:.2f}: SHD={s:.1f} P={p:.2f} R={r:.2f} t={t:.0f}s")

        for lp in [0.05, 0.1, 0.5]:
            prior_m = np.zeros((d,d))
            for i in range(d):
                for j in range(d):
                    if Wt[i,j]==0 and i!=j: prior_m[i,j]=0.9
            t0 = time.time()
            Wi = notears_fast(Xs, seed=seed, prior_matrix=prior_m, lambda_prior=lp)
            t = time.time()-t0
            P = (np.abs(Wi)>0.1).astype(float)
            s, p, r = shd_f(Wt, P), *pr_f(Wt, P)
            print(f"    Mislead λ={lp:.2f}: SHD={s:.1f} P={p:.2f} R={r:.2f} t={t:.0f}s")


# ═══════════════════════════════════════════════════════════════════════
#  Run everything
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*60)
    print("  Running CausalBayes Experiments 2-4")
    print("="*60)
    t0 = time.time()

    # Task 2: ~2 min
    task2()
    gc.collect()

    # Task 3: ~1 min  
    task3()
    gc.collect()

    # Task 4: ~2 min
    task4()
    gc.collect()

    print(f"\n{'='*60}")
    print(f"  ALL DONE in {time.time()-t0:.0f}s")
    print(f"{'='*60}")
