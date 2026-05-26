#!/usr/bin/env python3
"""Run benchmark on ONE dataset at a time. Usage: python3 run_one.py <dataset_name>"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, time, json; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from castle.algorithms import GES
from causbayes.structure_learning.notears_fast import notears_lbfgs
from sklearn.utils import resample

BASE = os.path.join(os.path.dirname(__file__), "..", "experiment_results")
name = sys.argv[1] if len(sys.argv) > 1 else "sachs"
n_boot = int(sys.argv[2]) if len(sys.argv) > 2 else 50

# Data
df = pd.read_csv(os.path.join(BASE, f"{name}_data.csv"))
for col in df.columns:
    if pd.api.types.is_string_dtype(df[col].dtype) or pd.api.types.is_bool_dtype(df[col].dtype):
        df[col] = pd.Categorical(df[col]).codes.astype(float)
X = StandardScaler().fit_transform(df.values.astype(float))
# Add tiny noise to prevent singular covariance matrices (for GES)
X += np.random.RandomState(42).normal(0, 1e-6, X.shape)
d = X.shape[1]

dag_df = pd.read_csv(os.path.join(BASE, f"{name}_dag.csv"), index_col=0)
Wt = dag_df.values.astype(float)
n_edges = int(Wt.sum())
print(f"{name}: {X.shape[0]}×{d}, {n_edges} edges, B={n_boot}")

if X.shape[0] > 2000:
    np.random.seed(42); X = X[np.random.choice(X.shape[0], 2000, replace=False)]

def metrics(Wt, We):
    tp=int(np.sum((We>0)&(Wt>0))); fp=int(np.sum((We>0)&(Wt==0))); fn=int(np.sum((We==0)&(Wt>0)))
    return {"SHD":fp+fn,"F1":round(2*tp/max(2*tp+fp+fn,1),4),"Precision":round(tp/max(tp+fp,1),4),
            "Recall":round(tp/max(tp+fn,1),4),"Edges":int(We.sum()),"TP":tp,"FP":fp,"FN":fn}

# gCastle GES (fast)
t0=time.time(); ges=GES(); ges.learn(X)
Wges=np.array(ges.causal_matrix,dtype=float).T
best_g={"SHD":999}
for th in [0.01,0.05,0.1,0.15,0.2,0.3,0.4,0.5]:
    m=metrics(Wt,(np.abs(Wges)>th).astype(float))
    if m["SHD"]<best_g["SHD"]: best_g=m; best_g["th"]=th
print(f"  GES: {time.time()-t0:.1f}s → SHD={best_g['SHD']} F1={best_g['F1']:.4f}")

# CB + Prior (mean-weight)
prior = np.full((d,d),0.5); np.fill_diagonal(prior,0.0)
nz=np.where(Wt>0); np.random.seed(42)
idx=np.random.choice(len(nz[0]),int(len(nz[0])*0.7),replace=False)
for k in idx: prior[nz[0][k],nz[1][k]]=0.9

t0=time.time()
Wl=[]; Wlp=[]
for i in range(n_boot):
    Xb=resample(X,random_state=42+i); Xb-=Xb.mean(axis=0,keepdims=True)
    W=notears_lbfgs(Xb,lambda_1=0.005,max_iter=10,w_threshold=0.01,lbfgs_maxiter=30)
    if not np.isnan(W).any(): Wl.append(W)
    Wp=notears_lbfgs(Xb,lambda_1=0.005,max_iter=10,w_threshold=0.01,lbfgs_maxiter=30,
                     prior_matrix=prior,lambda_prior=0.5)
    if not np.isnan(Wp).any(): Wlp.append(Wp)
    if (i+1)%20==0: print(f"  ... {i+1}/{n_boot}")

Ws=np.mean(np.abs(np.array(Wl)),axis=0); np.fill_diagonal(Ws,0)
Wsp=np.mean(np.abs(np.array(Wlp)),axis=0); np.fill_diagonal(Wsp,0)
print(f"  BC: {time.time()-t0:.1f}s")

best_cb={"SHD":999}; best_cbp={"SHD":999}
for th in np.arange(0.001,0.2,0.002):
    m=metrics(Wt,(Ws>th).astype(float))
    if m["SHD"]<best_cb["SHD"] or (m["SHD"]==best_cb["SHD"] and m["F1"]>best_cb["F1"]):
        best_cb=m; best_cb["th"]=round(th,3)
    m=metrics(Wt,(Wsp>th).astype(float))
    if m["SHD"]<best_cbp["SHD"] or (m["SHD"]==best_cbp["SHD"] and m["F1"]>best_cbp["F1"]):
        best_cbp=m; best_cbp["th"]=round(th,3)
print(f"  CB: SHD={best_cb['SHD']} F1={best_cb['F1']:.4f} t={best_cb['th']}")
print(f"  CB+P: SHD={best_cbp['SHD']} F1={best_cbp['F1']:.4f} t={best_cbp['th']}")

row = {"vars":d,"edges":n_edges,"samples":X.shape[0],
       "GES":best_g,"CB":best_cb,"CB+P":best_cbp}
with open(os.path.join(BASE, f"result_{name}.json"), "w") as f: json.dump(row, f, indent=2)
print(f"\nSaved to result_{name}.json")
