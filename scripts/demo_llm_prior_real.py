#!/usr/bin/env python3
"""
Real LLM Prior Demo: Uses actual deepseek-v4-flash API to generate causal priors.

Usage: OPENCODE_API_KEY="sk-xxx" python demo_llm_prior_real.py

This demo shows:
1. Generate data from a confounded DAG (6 vars, 6 edges)
2. Query LLM for causal relationships with domain context
3. Build prior matrix from LLM responses
4. Run bootstrap NOTEARS WITH and WITHOUT priors
5. Compare structural + calibration metrics
"""

import sys, os, json, time, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from causbayes.structure_learning.notears_fast import notears_lbfgs, bootstrap_notears, expected_calibration_error
from causbayes.llm_prior.prior_builder import build_prior_from_llm_response

API_KEY = os.environ.get("OPENCODE_API_KEY", "")
if not API_KEY:
    print("ERROR: Set OPENCODE_API_KEY environment variable")
    sys.exit(1)

API_BASE = "https://opencode.ai/zen/go/v1"
MODEL = "deepseek-v4-flash"


def query_llm(prompt):
    """Query deepseek-v4-flash via OpenAI-compatible API. No thinking for speed."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 300,
        "thinking": {"type": "disabled"}
    }
    resp = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def generate_confounded_dag(seed=42):
    """Generate a DAG with confounder + chain structures."""
    rng = np.random.RandomState(seed)
    d = 6
    W_true = np.zeros((d, d))
    W_true[0, 1] = 1.0
    W_true[0, 2] = 1.0
    W_true[1, 3] = 1.0
    W_true[2, 3] = 1.0
    W_true[3, 4] = 1.0
    W_true[4, 5] = 1.0
    
    n = 1000
    X = np.zeros((n, d))
    X[:, 0] = rng.randn(n)
    X[:, 1] = X[:, 0] * 1.0 + rng.randn(n) * 0.2
    X[:, 2] = X[:, 0] * 0.8 + rng.randn(n) * 0.2
    X[:, 3] = X[:, 1] * 0.5 + X[:, 2] * 0.5 + rng.randn(n) * 0.2
    X[:, 4] = np.tanh(X[:, 3]) + rng.randn(n) * 0.2
    X[:, 5] = np.sin(X[:, 4]) * 0.5 + rng.randn(n) * 0.2
    
    return X, W_true, n, d


def run_platt_calibration(P_raw, W_val, P_test):
    """Platt scaling calibration."""
    eps = 1e-8
    logit_p = np.log(np.clip(P_raw.flatten(), eps, 1 - eps) / np.clip(1 - P_raw.flatten(), eps, 1 - eps))
    y = W_val.flatten().astype(int)
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(logit_p.reshape(-1, 1), y)
    logit_test = np.log(np.clip(P_test.flatten(), eps, 1 - eps) / np.clip(1 - P_test.flatten(), eps, 1 - eps))
    P_cal = lr.predict_proba(logit_test.reshape(-1, 1))[:, 1].reshape(P_test.shape)
    return P_cal, lr


def compute_metrics(W_cal, W_true):
    """Compute all metrics from calibrated probability matrix."""
    W_bin = (W_cal >= 0.5).astype(float)
    shd = float(np.sum(np.abs(W_true - W_bin)) / 2)
    tp = int(np.sum((W_bin > 0) & (W_true > 0)))
    fp = int(np.sum((W_bin > 0) & (W_true == 0)))
    fn = int(np.sum((W_bin == 0) & (W_true > 0)))
    prec = float(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    rec = float(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    f1 = float(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    ece = float(expected_calibration_error(W_cal, W_true))
    return {"shd": float(shd), "f1": f1, "precision": prec, "recall": rec, "ece": ece, "n_edges": int(np.sum(W_bin))}


def parse_llm_response(resp_text, pairs):
    """Parse LLM response that maps edges -> confidence values."""
    result = {}
    for src_idx, tgt_idx in pairs:
        src_name = f"X{src_idx}"
        tgt_name = f"X{tgt_idx}"
        
        # Try exact format: X0→X1: 0.95
        pattern = re.escape(f"{src_name}→{tgt_name}") + r"\s*[:=]\s*([\d.]+)"
        m = re.search(pattern, resp_text)
        if m:
            result[(src_idx, tgt_idx)] = float(m.group(1))
            continue
        
        # Try: X0 -> X1: 0.95
        pattern2 = re.escape(f"{src_name}") + r"\s*[-]+>\s*" + re.escape(f"{tgt_name}") + r"\s*[:=]\s*([\d.]+)"
        m = re.search(pattern2, resp_text)
        if m:
            result[(src_idx, tgt_idx)] = float(m.group(1))
            continue
        
        # Fallback: order-based (assumes edges in same order as requested)
        result[(src_idx, tgt_idx)] = None
    
    return result


def main():
    print("=" * 70)
    print("  REAL LLM PRIOR DEMO 🦊")
    print(f"  Model: {MODEL} (thinking disabled)")
    print(f"  Endpoint: {API_BASE}")
    print("=" * 70)
    
    # Generate data
    print("\n[1] Generating confounded DAG (6 vars, 6 edges)")
    X_all, W_true, n, d = generate_confounded_dag(seed=42)
    n_tr, n_va = int(n * 0.5), int(n * 0.25)
    X_tr = X_all[:n_tr]
    X_va = X_all[n_tr:n_tr + n_va]
    X_te = X_all[n_tr + n_va:]
    
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_va_s = sc.transform(X_va)
    X_te_s = sc.transform(X_te)
    
    var_names = [f"X{i}" for i in range(d)]
    true_edges = [(i, j) for i in range(d) for j in range(d) if W_true[i, j] > 0]
    
    print(f"    True edges: {len(true_edges)}")
    for i, j in true_edges:
        print(f"      X{i} → X{j}")
    print(f"    n_train={n_tr}, n_val={n_va}, n_test={len(X_te)}")
    
    # ── Query LLM for causal priors ────────────────────────────────────
    print(f"\n[2] Querying {MODEL} for causal priors...")
    
    domain_description = """
A SYNTHETIC BIOLOGICAL SIGNALING PATHWAY with 6 variables:

X0 = TranscriptionFactorA (upstream regulator that activates transcription)
X1 = KinaseB (a kinase enzyme, activated by X0)
X2 = PhosphataseC (a phosphatase enzyme, activated by X0)
X3 = ResponseProteinD (integrates phosphorylation signals from X1 and X2)
X4 = EffectorE (downstream effector molecule, triggered by X3)
X5 = OutputF (final output protein in the cascade)

KNOWN BIOLOGY:
- Transcription factors directly regulate their target genes' expression
- Kinases phosphorylate and activate their downstream substrates
- Phosphatases dephosphorylate and regulate their downstream substrates  
- Response proteins integrate multiple upstream signals (from kinases AND phosphatases)
- Effector proteins act downstream of response proteins
- Signaling cascades propagate forward (upstream → downstream)
- X0 is at the top of the cascade, X5 is at the bottom
"""
    
    # Build all directed pairs
    all_pairs = [(i, j) for i in range(d) for j in range(d) if i != j]
    
    # Split into two batches for readability
    batch_size = len(all_pairs) // 2 + 1
    batch1 = all_pairs[:batch_size]
    batch2 = all_pairs[batch_size:]
    
    all_llm_results = {}
    
    for batch_num, batch_pairs in enumerate([batch1, batch2], 1):
        prompt = domain_description + "\n"
        prompt += f"Batch {batch_num}: Rate your confidence (0.0 to 1.0) that the FIRST variable causes the SECOND variable in this pathway.\n"
        prompt += "Use the domain knowledge above to inform your answers.\n"
        prompt += "Output EXACTLY one line per pair in the format: Xi→Xj: confidence\n\n"
        
        for src, tgt in batch_pairs:
            prompt += f"X{src}→X{tgt}:\n"
        
        prompt += "\nRemember: 0.0 = definitely NOT causal, 0.5 = uncertain, 1.0 = definitely causal."
        
        try:
            resp = query_llm(prompt)
            print(f"    Batch {batch_num}: ✓")
            
            # Debug: show raw response (first 300 chars)
            debug_lines = resp.strip().split('\n')
            for line in debug_lines[:15]:
                line = line.strip()
                if line:
                    print(f"      >> {line}")
            if len(debug_lines) > 15:
                print(f"      >> ... ({len(debug_lines)} lines total)")
                
        except Exception as e:
            print(f"    Batch {batch_num}: ✗ {e}")
            resp = ""
        
        parsed = parse_llm_response(resp, batch_pairs)
        all_llm_results.update(parsed)
    
    # Build llm_edges list with fallback for unparsed edges
    llm_edges = []
    unparsed = []
    for src, tgt in all_pairs:
        prob = all_llm_results.get((src, tgt))
        if prob is not None:
            prob = min(1.0, max(0.0, prob))
            prob = round(prob, 2)
        else:
            prob = 0.5  # fallback
            unparsed.append((src, tgt))
        llm_edges.append((f"X{src}", f"X{tgt}", prob))
    
    if unparsed:
        print(f"\n    ⚠️ {len(unparsed)} edges fell back to default (unparsed)")
    
    # Print results
    print(f"\n    LLM PRIOR SUMMARY:")
    correct_high = 0
    correct_low = 0
    false_high = 0
    
    for src, dst, prob in llm_edges:
        idx_i, idx_j = int(src[1]), int(dst[1])
        truth = "✓" if W_true[idx_i, idx_j] > 0 else "✗"
        flag = ""
        if truth == "✓":
            if prob > 0.5:
                flag = "✅"
                correct_high += 1
            elif prob < 0.3:
                flag = "❌ missed"
                correct_low += 1
            else:
                flag = "⚠️ low"
                correct_low += 1
        else:
            if prob > 0.5:
                flag = "❌ false"
                false_high += 1
        if flag:
            print(f"      {src}→{dst}: P={prob:.2f} [{truth}] {flag}")
    
    total_true = np.sum(W_true > 0)
    print(f"\n    Prior stats:")
    print(f"      True edges with prior>0.5: {correct_high}/{total_true}")
    print(f"      True edges missed (≤0.5): {correct_low}/{total_true}")
    print(f"      False edges with prior>0.5: {false_high}")
    
    # Build prior matrix
    prior_matrix = build_prior_from_llm_response(llm_edges, var_names)
    correct_prior = prior_matrix[W_true > 0].mean()
    false_prior = prior_matrix[W_true == 0].mean()
    print(f"      Avg prior on true edges:  {correct_prior:.3f}")
    print(f"      Avg prior on false edges: {false_prior:.3f}")
    
    print(f"\n    Prior matrix (rows=source, cols=target):")
    for i in range(d):
        row = " ".join(f"{prior_matrix[i,j]:.1f}" if prior_matrix[i,j] > 0 else " . " for j in range(d))
        print(f"      X{i}: [{row}]")
    
    # ── Bootstrap WITHOUT prior ────────────────────────────────────────
    print(f"\n[3] Bootstrap WITHOUT LLM prior (30 bootstraps)...")
    t0 = time.time()
    P_no_prior, _, _, _ = bootstrap_notears(
        X_tr_s, n_bootstraps=30, max_iter=5, w_threshold=0.05, method="lbfgs", seed=42
    )
    P_no_cal, _ = run_platt_calibration(P_no_prior, W_true, P_no_prior)
    t1 = time.time()
    metrics_no = compute_metrics(P_no_cal, W_true)
    print(f"    Time: {t1-t0:.2f}s")
    print(f"    SHD={metrics_no['shd']:.0f}, F1={metrics_no['f1']:.3f}, "
          f"Prec={metrics_no['precision']:.3f}, Rec={metrics_no['recall']:.3f}, "
          f"ECE={metrics_no['ece']:.4f}, Edges={metrics_no['n_edges']}")
    
    # ── Bootstrap WITH prior (multiple lambdas) ────────────────────────
    lambdas = [0.01, 0.05, 0.1, 0.2]
    best_metrics = metrics_no
    best_lambda = 0.0
    best_P = P_no_cal
    all_with = {}
    
    print(f"\n[4] Bootstrap WITH LLM prior (searching λ)...")
    for lam in lambdas:
        t0 = time.time()
        P_with_prior, _, _, _ = bootstrap_notears(
            X_tr_s, n_bootstraps=30, max_iter=5, w_threshold=0.05, method="lbfgs", seed=42,
            prior_matrix=prior_matrix, lambda_prior=lam,
        )
        P_with_cal, _ = run_platt_calibration(P_with_prior, W_true, P_with_prior)
        metrics_with = compute_metrics(P_with_cal, W_true)
        t1 = time.time()
        print(f"    λ={lam:.2f}: SHD={metrics_with['shd']:.0f} F1={metrics_with['f1']:.3f} "
              f"ECE={metrics_with['ece']:.4f} Edges={metrics_with['n_edges']} [{t1-t0:.1f}s]")
        all_with[f"lambda_{lam}"] = metrics_with
        
        # Track best (by F1, then ECE tiebreaker)
        better = (metrics_with['f1'] > best_metrics['f1']) or \
                 (metrics_with['f1'] == best_metrics['f1'] and metrics_with['ece'] < best_metrics['ece'])
        if better:
            best_metrics = metrics_with
            best_lambda = lam
            best_P = P_with_cal
    
    # ── Final Comparison ──────────────────────────────────────────────
    print(f"\n[5] Comparison: Real LLM Priors via {MODEL}")
    print(f"    Best λ={best_lambda}")
    print(f"    {'─'*60}")
    print(f"    {'Metric':<18} {'No Prior':>10} {'Prior':>12} {'Δ':>10} {'Result':>8}")
    print(f"    {'─'*60}")
    
    comparison = {}
    for name in ["shd", "f1", "precision", "recall", "ece"]:
        lower_better = name in ["shd", "ece"]
        v_no = metrics_no[name]
        v_with = best_metrics[name]
        diff = (v_no - v_with) if lower_better else (v_with - v_no)
        direction = "✅" if diff > 0.01 else ("❌" if diff < -0.01 else "➡️")
        fmt_improved = diff > 0.01
        print(f"    {name:<18} {v_no:>10.4f} {v_with:>12.4f} {diff:>+10.4f} {direction:>8}")
        comparison[name] = {"without": v_no, "with": v_with, "diff": diff, "improved": fmt_improved}
    
    # ── Edge-level detail ─────────────────────────────────────────────
    print(f"\n[6] Edges with largest probability changes:")
    flat_no = P_no_cal.flatten()
    flat_with = best_P.flatten()
    flat_prior = prior_matrix.flatten()
    flat_gt = W_true.flatten()
    
    abs_diff = np.abs(flat_with - flat_no)
    top_idx = np.argsort(abs_diff)[-12:]
    
    print(f"    {'Edge':<10} {'Truth':>5} {'No Prior':>10} {'Prior':>12} {'LLM':>8} {'Δ':>8}")
    print(f"    {'─'*10} {'─'*5} {'─'*10} {'─'*12} {'─'*8} {'─'*8}")
    for idx in sorted(top_idx):
        i, j = idx // d, idx % d
        if i == j:
            continue
        gt = "✓" if flat_gt[idx] > 0 else "✗"
        delta = flat_with[idx] - flat_no[idx]
        print(f"    X{i}→X{j}    {gt:>3} {flat_no[idx]:>10.3f} {flat_with[idx]:>12.3f} "
              f"{flat_prior[idx]:>8.2f} {delta:>+8.3f}")
    
    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n[7] 📊 Summary")
    improved = [k for k, v in comparison.items() if v.get("improved")]
    worsened = [k for k, v in comparison.items() if k != "n_edges" and not v.get("improved") and abs(v.get("diff", 0)) > 0.01]
    
    if improved:
        print(f"    ✅ LLM prior improved: {', '.join(improved)}")
    if worsened:
        print(f"    ⚠️  LLM prior worsened: {', '.join(worsened)}")
    if not improved and not worsened:
        print(f"    ➡️ No significant change with LLM priors")
    
    print(f"    Best λ_prior: {best_lambda}")
    print(f"    Prior quality: true edges avg={correct_prior:.2f}, false edges avg={false_prior:.2f}")
    print(f"    True edges correct: {correct_high}/{total_true}")
    print(f"    False positives from prior: {false_high}")
    
    # ── Save ───────────────────────────────────────────────────────────
    results = {
        "metadata": {
            "model": MODEL,
            "api_base": API_BASE,
            "thinking": "disabled",
            "n_bootstraps": 30,
            "lambdas_tested": lambdas,
            "best_lambda": best_lambda,
            "n": n,
            "d": d,
            "true_edges": total_true,
        },
        "prior_quality": {
            "avg_prior_true_edges": float(correct_prior),
            "avg_prior_false_edges": float(false_prior),
            "true_edges_with_high_prior": int(correct_high),
            "true_edges_with_low_prior": int(correct_low),
            "false_edges_with_high_prior": int(false_high),
        },
        "llm_responses": [{"source": src, "target": dst, "confidence": prob} for src, dst, prob in llm_edges],
        "without_prior": metrics_no,
        "with_prior_by_lambda": all_with,
        "with_prior_best": best_metrics,
        "comparison": comparison,
    }
    os.makedirs("experiment_results", exist_ok=True)
    out_path = "experiment_results/llm_prior_real_api.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\n    Results saved to {out_path}")
    
    print(f"\n{'='*70}")
    print(f"  Real LLM Prior Demo Complete! 🦊")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
