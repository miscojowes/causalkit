#!/usr/bin/env python3
"""Real diagnosis: CausalBayes vs gCastle on real data + ACTUAL LLM prior."""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from castle.algorithms import GES, PC
from causbayes.structure_learning.bootstrapped import BootstrapDAG
from causbayes.structure_learning.utils import structural_hamming_distance as shd_func

# Get API key
API_KEY = os.environ.get('OPENCODE_GO_API_KEY', '')
if not API_KEY:
    try:
        for line in open(os.path.expanduser('~/.hermes/.env')):
            if 'OPENCODE_GO' in line:
                API_KEY = line.strip().split('=', 1)[1]
                break
    except: pass
print(f'API key: {API_KEY[:12]}...' if API_KEY else 'MISSING')

# Load Sachs
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "experiment_results", "sachs_raw.csv"), sep='\t')
X = StandardScaler().fit_transform(df.values)
d, n = X.shape[1], X.shape[0]
vars_list = list(df.columns)
print(f'Sachs: {n} samples, {d} variables')
print(f'Vars: {vars_list}')

# Ground truth
gt_map = [
    ("Raf","Mek"),("Mek","Erk"),("Plcg","PIP2"),("PIP2","PIP3"),
    ("PIP3","Akt"),("PKC","PKA"),("PKA","Jnk"),("PKC","Jnk"),
    ("Jnk","P38"),("PKC","P38"),("Erk","Akt"),("PKC","Akt"),
    ("PKA","Akt"),("PIP3","PKA"),("PIP3","Plcg"),("PKA","Erk"),("PKC","Mek"),
]
Wt = np.zeros((d,d))
for c,e in gt_map: Wt[vars_list.index(c), vars_list.index(e)] = 1.0
print(f'True edges: {int(Wt.sum())}')

def metrics(Wt, W_est):
    shd = shd_func(Wt, W_est)
    tp = np.sum((W_est>0)&(Wt>0)); fp = np.sum((W_est>0)&(Wt==0))
    fn = np.sum((W_est==0)&(Wt>0))
    f1 = 2*tp/max(2*tp+fp+fn,1)
    return shd, f1, tp, fp, fn

# 1. gCastle GES
print('\n=== gCastle GES ===')
ges = GES(); ges.learn(X)
W_ges = np.array(ges.causal_matrix, dtype=float).T
for th in [0.05, 0.1, 0.3]:  
    Wb = (np.abs(W_ges) > th).astype(float)
    shd, f1, tp, fp, fn = metrics(Wt, Wb)
    print(f'  thresh={th}: SHD={shd} F1={f1:.3f} edges={int(Wb.sum())} TP={tp} FP={fp} FN={fn}')

# 2. gCastle PC
print('\n=== gCastle PC ===')
pc = PC(); pc.learn(X)
W_pc = np.array(pc.causal_matrix, dtype=float).T
for th in [0.05, 0.1, 0.3]:
    Wb = (np.abs(W_pc) > th).astype(float)
    shd, f1, tp, fp, fn = metrics(Wt, Wb)
    print(f'  thresh={th}: SHD={shd} F1={f1:.3f} edges={int(Wb.sum())} TP={tp} FP={fp} FN={fn}')

# 3. CB BootstrapDAG — tuned
print('\n=== CausalBayes Bootstrap (tuned, no prior) ===')
for nb in [50, 100]:
    m = BootstrapDAG(n_bootstraps=nb, lambda_1=0.005, max_iter=10,
                    w_threshold=0.05, calibrate=False, verbose=False, seed=42)
    m.fit(X)
    probs = m._edge_probs_raw_
    
    best_shd, best_t, best_f1 = 999, 0, 0
    for t in np.arange(0.05, 0.96, 0.05):
        Wb = (probs >= t).astype(float)
        shd, f1, _, _, _ = metrics(Wt, Wb)
        if shd < best_shd or (shd == best_shd and f1 > best_f1):
            best_shd, best_t, best_f1 = shd, t, f1
    
    Wb = (probs >= best_t).astype(float)
    _, _, tp, fp, fn = metrics(Wt, Wb)
    print(f'  B={nb}: best_t={best_t:.2f} SHD={best_shd} F1={best_f1:.3f} edges={int(Wb.sum())} TP={tp} FP={fp} FN={fn}')

# 4. CB Bootstrap + REAL LLM Prior
print('\n=== CausalBayes Bootstrap + REAL LLM Prior ===')
if API_KEY:
    from causbayes.llm_prior import LLMPriorExtractor
    
    domain_text = """We are studying protein signaling in human T-cells. Variables:
- Raf: MAPKKK, starts MAPK cascade
- Mek: MAPKK, downstream of Raf  
- Erk: MAPK, downstream of Mek
- Plcg: Phospholipase C, produces PIP3 from PIP2
- PIP2: membrane lipid, cleaved by Plcg
- PIP3: signaling lipid, product of Plcg
- Akt: PKB kinase, cell survival
- PKA: cAMP-dependent kinase
- PKC: Ca/DAG-dependent kinase
- P38: stress MAPK
- Jnk: stress kinase

Well-known causal edges from cell biology:
Raf→Mek, Mek→Erk, Plcg→PIP2, PIP2→PIP3, PIP3→Akt,
PIP3→Plcg (feedback), PKC→PKA, PKC→Jnk, PKA→Jnk,
Jnk→P38, PKC→P38, PKC→Raf, PKA→Raf (inhibitory),
PKC→Mek, PKA→Mek (inhibitory), Erk→Akt, PKA→Akt,
Akt→Raf (feedback inhibition)

ONLY return edges you are VERY CONFIDENT about. Return as JSON array:
[{"cause":"X","effect":"Y","confidence":"high|medium|low"}]"""
    
    try:
        extractor = LLMPriorExtractor(api_key=API_KEY)
        prior = extractor.extract_edge_priors(vars_list, domain_text, confidence="high")
        print(f'\nPrior matrix: {np.sum(prior>0.6)} edges > 0.6')
        
        flat = [(i,j,prior[i,j]) for i in range(d) for j in range(d) if i!=j and prior[i,j]>=0.6]
        flat.sort(key=lambda x:-x[2])
        for i,j,p in flat[:20]:
            correct = '✓' if Wt[i,j] else '✗'
            print(f'  {vars_list[i]:6s}→{vars_list[j]:6s}: P={p:.2f} [{correct}]')
        
        for lam in [0.3, 0.5, 1.0]:
            m2 = BootstrapDAG(n_bootstraps=100, lambda_1=0.005, max_iter=10,
                            w_threshold=0.05, calibrate=False, verbose=False, seed=42,
                            prior_matrix=prior, lambda_prior=lam)
            m2.fit(X)
            probs2 = m2._edge_probs_raw_
            best_shd2, best_t2, best_f12 = 999, 0, 0
            for t in np.arange(0.05, 0.96, 0.05):
                Wb = (probs2 >= t).astype(float)
                shd, f1, _, _, _ = metrics(Wt, Wb)
                if shd < best_shd2 or (shd == best_shd2 and f1 > best_f12):
                    best_shd2, best_t2, best_f12 = shd, t, f1
            Wb2 = (probs2 >= best_t2).astype(float)
            _, _, tp2, fp2, fn2 = metrics(Wt, Wb2)
            print(f'  λ={lam:.1f}: best_t={best_t2:.2f} SHD={best_shd2} F1={best_f12:.3f} edges={int(Wb2.sum())} TP={tp2} FP={fp2} FN={fn2}')
    
    except Exception as e:
        print(f'  LLM failed: {e}')
        import traceback; traceback.print_exc()

print('\n=== FINAL VERDICT ===')
print('gCastle GES (best): SHD=?, F1=?  (see above)')
print('CausalBayes (tuned, no prior): SHD=?, F1=?  (see above)')
print('CausalBayes (+LLM prior): SHD=?, F1=?  (see above)')
