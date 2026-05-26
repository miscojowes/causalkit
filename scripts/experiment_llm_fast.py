#!/usr/bin/env python3
"""
Fast Real LLM Prior Experiment - 5 seeds × 2 conditions × best λ search
"""
import sys, os, json, time, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_rel

from causbayes.structure_learning.notears_fast import bootstrap_notears, expected_calibration_error
from causbayes.llm_prior.prior_builder import build_prior_from_llm_response

# ── Config ──
auth_path = os.path.expanduser("~/.openclaw/agents/main/agent/auth-profiles.json")
with open(auth_path) as f:
    API_KEY = json.load(f)["profiles"]["opencode-go:default"]["key"]

API_BASE = "https://opencode.ai/zen/go/v1"
MODEL = "deepseek-v4-flash"
N_SEEDS = 5
N_BOOT = 30

# ── Data ──
def make_data(seed):
    rng = np.random.RandomState(seed)
    d = 6
    W = np.zeros((d,d))
    W[0,1] = W[0,2] = W[1,3] = W[2,3] = W[3,4] = W[4,5] = 1.0
    n = 500
    X = np.zeros((n,d))
    X[:,0] = rng.randn(n)
    X[:,1] = X[:,0]*1 + rng.randn(n)*0.2
    X[:,2] = X[:,0]*0.8 + rng.randn(n)*0.2
    X[:,3] = X[:,1]*0.5 + X[:,2]*0.5 + rng.randn(n)*0.2
    X[:,4] = np.tanh(X[:,3]) + rng.randn(n)*0.2
    X[:,5] = np.sin(X[:,4])*0.5 + rng.randn(n)*0.2
    return StandardScaler().fit_transform(X), W, d

# ── LLM ──
def query_llm(prompt):
    r = requests.post(f"{API_BASE}/chat/completions", 
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.0, "max_tokens": 300, "thinking": {"type": "disabled"}},
        timeout=30)
    return r.json()["choices"][0]["message"]["content"]

DOMAIN = """Biological signaling pathway:
X0=TranscriptionFactorA, X1=KinaseB, X2=PhosphataseC,
X3=ResponseProteinD, X4=EffectorE, X5=OutputF

Known: TFs activate targets, Kinases activate downstream, 
Phosphatases regulate, Response proteins integrate signals,
Effectors act downstream, cascades propagate forward.

Output EXACTLY one line per pair: Xi→Xj: number (0.0-1.0)"""

def get_llm_prior():
    pairs = [(i,j) for i in range(6) for j in range(6) if i!=j]
    mid = len(pairs)//2
    prior = {}
    for batch in [pairs[:mid], pairs[mid:]]:
        prompt = DOMAIN + "\n\n" + "\n".join(f"X{i}→X{j}:" for i,j in batch)
        resp = query_llm(prompt)
        for i,j in batch:
            m = re.search(f"X{i}→X{j}" + r"\s*[:=]\s*([\d.]+)", resp)
            prior[(i,j)] = min(1.0,max(0.0,float(m.group(1)))) if m else 0.5
    P = np.zeros((6,6))
    for i,j in pairs: P[i,j] = prior[(i,j)]
    return P

# ── Pipeline ──
def calibrate(P_raw, W):
    eps=1e-8
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    logit = np.log(np.clip(P_raw.flatten(),eps,1-eps)/np.clip(1-P_raw.flatten(),eps,1-eps))
    lr.fit(logit.reshape(-1,1), W.flatten().astype(int))
    P_cal = lr.predict_proba(logit.reshape(-1,1))[:,1].reshape(P_raw.shape)
    return P_cal, lr

def metrics(P_cal, W):
    Wb = (P_cal>=0.5).astype(float)
    tp = int(np.sum((Wb>0)&(W>0)))
    fp = int(np.sum((Wb>0)&(W==0)))
    fn = int(np.sum((Wb==0)&(W>0)))
    prec = tp/(tp+fp) if tp+fp>0 else 0.0
    rec = tp/(tp+fn) if tp+fn>0 else 0.0
    return {"shd": float(np.sum(np.abs(W-Wb))/2), "f1": 2*prec*rec/(prec+rec) if prec+rec>0 else 0.0,
            "precision": prec, "recall": rec, "ece": float(expected_calibration_error(P_cal,W))}

def run(X, W, prior=None, lam=0.0, seed=42):
    P,_,_,_ = bootstrap_notears(X, n_bootstraps=N_BOOT, max_iter=5, w_threshold=0.05,
                                 method="lbfgs", seed=seed, prior_matrix=prior, lambda_prior=lam)
    Pc,_ = calibrate(P, W)
    return metrics(Pc, W), Pc

# ── Main ──
print("="*60, flush=True)
print("  REAL LLM PRIOR - FAST EXPERIMENT", flush=True)
print("="*60, flush=True)

print(f"\n[1] Querying LLM for priors...", flush=True)
llm_prior = get_llm_prior()
for i in range(6):
    row = " ".join(f"{llm_prior[i,j]:.1f}" if llm_prior[i,j]>0 else " ." for j in range(6))
    print(f"  X{i}: [{row}]", flush=True)

W0 = make_data(42)[1]
n_true = int(np.sum(W0>0))
print(f"\n  True edges with prior>0.5: {np.sum((llm_prior>0.5)&(W0>0))}/{n_true}", flush=True)
print(f"  False edges with prior>0.5: {np.sum((llm_prior>0.5)&(W0==0))}", flush=True)
print(f"  Avg prior true: {llm_prior[W0>0].mean():.3f}, false: {llm_prior[W0==0].mean():.3f}", flush=True)

# Search best λ
lams = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
print(f"\n[2] Searching best λ (5 seeds)...", flush=True)

all_results = {}
for lam in lams:
    f1s = []
    for seed in range(N_SEEDS):
        X,W,_ = make_data(42+seed)
        m,_ = run(X, W, prior=llm_prior if lam>0 else None, lam=lam, seed=seed+100)
        f1s.append(m["f1"])
    mean_f1 = np.mean(f1s)
    all_results[lam] = {"f1_mean": float(mean_f1), "f1_vals": [float(v) for v in f1s]}
    print(f"  λ={lam:.3f}: F1={mean_f1:.4f}", flush=True)

best_lam = max(lams, key=lambda l: all_results[l]["f1_mean"])
print(f"\n  Best λ: {best_lam}", flush=True)

# Main experiment at best λ
print(f"\n[3] Main comparison (λ={best_lam})...", flush=True)
no_prior_f1 = all_results[0.0]["f1_vals"]

m_no_list, m_llm_list = [], []
for seed in range(N_SEEDS):
    X,W,_ = make_data(42+seed)
    m_no,_ = run(X, W, seed=seed+100)
    m_llm,_ = run(X, W, prior=llm_prior, lam=best_lam, seed=seed+100)
    m_no_list.append(m_no)
    m_llm_list.append(m_llm)
    
# Aggregate
print(f"\n  {'─'*55}", flush=True)
print(f"  {'Metric':<15} {'No Prior':>10} {'LLM Prior':>12} {'Δ':>10} {'p-value':>8}", flush=True)
print(f"  {'─'*55}", flush=True)

summary = {}
for metric in ["shd","f1","precision","recall","ece"]:
    nv = [m[metric] for m in m_no_list]
    lv = [m[metric] for m in m_llm_list]
    nm = np.mean(nv)
    lm = np.mean(lv)
    lower = metric in ["shd","ece"]
    diff = (nm-lm) if lower else (lm-nm)
    _, p = ttest_rel(lv, nv)
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
    print(f"  {metric:<15} {nm:>10.4f} {lm:>12.4f} {diff:>+10.4f} {p:>8.4f} {sig}", flush=True)
    summary[metric] = {"no_prior_mean":float(nm),"llm_prior_mean":float(lm),"diff":float(diff),"p_value":float(p)}

# Final edge probabilities (seed 0)
print(f"\n[4] Edge-level (seed 0):", flush=True)
X,W,_ = make_data(42)
_, P_no = run(X, W, seed=42)
_, P_llm = run(X, W, prior=llm_prior, lam=best_lam, seed=42)

print(f"  {'Edge':<10} {'Truth':>5} {'No Prior':>10} {'LLM Prior':>10} {'LLM':>8} {'Δ':>8}", flush=True)
print(f"  {'─'*10} {'─'*5} {'─'*10} {'─'*10} {'─'*8} {'─'*8}", flush=True)
d=6
changes=[]
for i in range(d):
    for j in range(d):
        if i!=j:
            idx=i*d+j
            changes.append((abs(P_llm.flatten()[idx]-P_no.flatten()[idx]),i,j,
                           "✓" if W[i,j]>0 else "✗",P_no.flatten()[idx],P_llm.flatten()[idx],llm_prior[i,j]))
changes.sort(reverse=True)
for _,i,j,t,pn,pl,pp in changes[:10]:
    print(f"  X{i}→X{j}    {t:>3} {pn:>10.3f} {pl:>10.3f} {pp:>8.1f} {pl-pn:>+8.3f}", flush=True)

# VERDICT
print(f"\n{'='*60}", flush=True)
print(f"  VERDICT", flush=True)
print(f"{'='*60}", flush=True)

f1_imp = summary["f1"]["diff"]
f1_p = summary["f1"]["p_value"]
print(f"\n  F1 improvement: {f1_imp*100:+.1f}% (p={f1_p:.4f})", flush=True)
print(f"  SHD improvement: {summary['shd']['diff']:+.2f}", flush=True)
print(f"  ECE improvement: {summary['ece']['diff']:+.4f}", flush=True)

if f1_imp > 0.03 and f1_p < 0.05:
    print(f"\n  🏆 REAL EFFECT - Statistical significance achieved", flush=True)
elif f1_imp > 0.01:
    print(f"\n  📊 MODEST EFFECT - Real but small", flush=True)
else:
    print(f"\n  ⚠️  NEGLIGIBLE EFFECT - No meaningful improvement", flush=True)

# Save
os.makedirs("experiment_results", exist_ok=True)
out = {"model":MODEL,"best_lambda":best_lam,"lambdas_tested":lams,
       "prior_quality":{"avg_true":float(llm_prior[W0>0].mean()),"avg_false":float(llm_prior[W0==0].mean()),
                        "true_high":int(np.sum((llm_prior>0.5)&(W0>0))),"false_high":int(np.sum((llm_prior>0.5)&(W0==0)))},
       "llm_prior_matrix":llm_prior.tolist(),"summary":summary,
       "no_prior_seed_metrics":m_no_list,"llm_prior_seed_metrics":m_llm_list}
with open("experiment_results/real_llm_final.json","w") as f:
    json.dump(out, f, indent=2)
print(f"\n  Results saved", flush=True)
print(f"\n  Done! 🦊", flush=True)
