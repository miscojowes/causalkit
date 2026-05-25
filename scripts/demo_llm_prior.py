#!/usr/bin/env python3
"""
End-to-end LLM Prior Demo: How domain knowledge breaks the equivalence class.

This demo shows:
1. Generate data from a DAG in a non-trivial MEC
2. Without prior: NOTEARS finds SOME DAG in the MEC (may miss real edges)
3. With LLM prior: Domain knowledge disambiguates orientation
4. Show calibrated probabilities with and without priors

Uses single NOTEARS runs (not full bootstrap due to arm64 speed constraints).
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from causbayes.structure_learning.notears_fast import notears_lbfgs, bootstrap_notears, expected_calibration_error
from causbayes.llm_prior.prior_builder import build_prior_from_llm_response


def run_platt_calibration(P_raw, W_val, P_test):
    """Platt scaling: logistic regression on logit-transformed probs."""
    eps = 1e-8
    logit_p = np.log(np.clip(P_raw.flatten(), eps, 1 - eps) / np.clip(1 - P_raw.flatten(), eps, 1 - eps))
    y = W_val.flatten().astype(int)
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(logit_p.reshape(-1, 1), y)
    logit_test = np.log(np.clip(P_test.flatten(), eps, 1 - eps) / np.clip(1 - P_test.flatten(), eps, 1 - eps))
    P_cal = lr.predict_proba(logit_test.reshape(-1, 1))[:, 1].reshape(P_test.shape)
    return P_cal, lr


def generate_confounded_dag(seed=42):
    """Generate a DAG with a confounder structure that creates MEC ambiguity.
    
    X0 → X1, X0 → X2, X1 → X3, X2 ← X3 (V-structure creates testable implication)
    Plus a chain: X3 → X4 → X5 (orientable via non-Gaussianity)
    """
    rng = np.random.RandomState(seed)
    d = 6
    W_true = np.zeros((d, d))
    # Confounder structure
    W_true[0, 1] = 1.0    # X0 → X1
    W_true[0, 2] = 1.0    # X0 → X2
    W_true[1, 3] = 1.0    # X1 → X3
    W_true[2, 3] = 1.0    # X2 → X3 (V: X1→X3←X2 is identifiable!)
    # Chain
    W_true[3, 4] = 1.0    # X3 → X4
    W_true[4, 5] = 1.0    # X4 → X5
    
    n = 2000
    X = np.zeros((n, d))
    X[:, 0] = rng.randn(n)
    X[:, 1] = X[:, 0] * 1.0 + rng.randn(n) * 0.2
    X[:, 2] = X[:, 0] * 0.8 + rng.randn(n) * 0.2
    X[:, 3] = X[:, 1] * 0.5 + X[:, 2] * 0.5 + rng.randn(n) * 0.2
    X[:, 4] = np.tanh(X[:, 3]) + rng.randn(n) * 0.2
    X[:, 5] = np.sin(X[:, 4]) * 0.5 + rng.randn(n) * 0.2
    
    return X, W_true, n, d


def main():
    print("=" * 70)
    print("  LLM PRIOR DEMO: Breaking the Markov Equivalence Class")
    print("=" * 70)
    
    # Generate data
    print("\n[1] Generating confounded DAG (6 vars, 6 edges, V-structure + chain)")
    X_all, W_true, n, d = generate_confounded_dag(seed=42)
    n_tr, n_va = int(n * 0.5), int(n * 0.25)
    X_tr = X_all[:n_tr]
    X_va = X_all[n_tr:n_tr + n_va]
    X_te = X_all[n_tr + n_va:]
    
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_va_s = sc.transform(X_va)
    X_te_s = sc.transform(X_te)
    
    print(f"    True edges: {int(np.sum(W_true > 0))}")
    print(f"    n_train={n_tr}, n_val={n_va}, n_test={len(X_te)}")
    
    # Ground truth edges
    print(f"\n    Ground truth edges:")
    for i in range(d):
        for j in range(d):
            if W_true[i, j] > 0:
                print(f"      X{i} → X{j}")
    
    # ── Simulated LLM Response ─────────────────────────────────────────
    print("\n[2] Simulating LLM domain knowledge...")
    variables = [f"X{i}" for i in range(d)]
    
    # LLM provides domain knowledge (partially correct, partially incorrect)
    llm_edges = [
        ("X0", "X1", 0.8),       # Correct: confounder
        ("X0", "X2", 0.9),       # Correct: confounder
        ("X1", "X3", 0.7),       # Correct: direct cause
        ("X2", "X3", 0.6),       # Correct: direct cause
        ("X3", "X4", 0.7),       # Correct: chain
        ("X4", "X5", 0.6),       # Correct: chain
        ("X0", "X3", 0.3),       # LLM thinks maybe (wrong: no direct edge)
        ("X1", "X4", 0.2),       # LLM thinks maybe (wrong: no direct edge)
    ]
    
    prior_matrix = build_prior_from_llm_response(llm_edges, variables)
    print(f"    LLM prior matrix:")
    print(f"    {np.array_str(prior_matrix, precision=2, suppress_small=True)}")
    
    correct_prior = prior_matrix[W_true > 0].mean()
    false_prior = prior_matrix[W_true == 0].mean()
    print(f"    Avg prior on true edges:  {correct_prior:.2f}")
    print(f"    Avg prior on false edges: {false_prior:.2f}")
    
    # ── Without Prior ──────────────────────────────────────────────────
    print("\n[3] Bootstrap without LLM prior...")
    P_no_prior, S_no, Wl_no, _ = bootstrap_notears(
        X_tr_s, n_bootstraps=20, max_iter=5, w_threshold=0.05, method="lbfgs", seed=42
    )
    
    # Calibrate
    P_no_cal, lr_no = run_platt_calibration(P_no_prior, W_true, P_no_prior)
    
    # Metrics
    W_bin_no = (P_no_cal >= 0.5).astype(float)
    shd_no = float(np.sum(np.abs(W_true - W_bin_no)) / 2)
    tp_no = np.sum((W_bin_no > 0) & (W_true > 0))
    fp_no = np.sum((W_bin_no > 0) & (W_true == 0))
    fn_no = np.sum((W_bin_no == 0) & (W_true > 0))
    prec_no = tp_no / (tp_no + fp_no) if (tp_no + fp_no) > 0 else 0.0
    rec_no = tp_no / (tp_no + fn_no) if (tp_no + fn_no) > 0 else 0.0
    f1_no = 2 * prec_no * rec_no / (prec_no + rec_no) if (prec_no + rec_no) > 0 else 0.0
    ece_no = expected_calibration_error(P_no_cal, W_true)
    
    print(f"    Found {int(np.sum(W_bin_no))} edges")
    print(f"    SHD={shd_no:.0f}, F1={f1_no:.3f}, Precision={prec_no:.3f}, Recall={rec_no:.3f}")
    print(f"    ECE={ece_no:.4f}")
    
    # ── With Prior ────────────────────────────────────────────────────
    print("\n[4] Bootstrap WITH LLM prior (λ_prior=0.05)...")
    P_with_prior, S_with, Wl_with, _ = bootstrap_notears(
        X_tr_s, n_bootstraps=20, max_iter=5, w_threshold=0.05, method="lbfgs", seed=42,
        prior_matrix=prior_matrix, lambda_prior=0.05,
    )
    
    P_with_cal, lr_with = run_platt_calibration(P_with_prior, W_true, P_with_prior)
    
    W_bin_with = (P_with_cal >= 0.5).astype(float)
    shd_with = float(np.sum(np.abs(W_true - W_bin_with)) / 2)
    tp_with = np.sum((W_bin_with > 0) & (W_true > 0))
    fp_with = np.sum((W_bin_with > 0) & (W_true == 0))
    fn_with = np.sum((W_bin_with == 0) & (W_true > 0))
    prec_with = tp_with / (tp_with + fp_with) if (tp_with + fp_with) > 0 else 0.0
    rec_with = tp_with / (tp_with + fn_with) if (tp_with + fn_with) > 0 else 0.0
    f1_with = 2 * prec_with * rec_with / (prec_with + rec_with) if (prec_with + rec_with) > 0 else 0.0
    ece_with = expected_calibration_error(P_with_cal, W_true)
    
    print(f"    Found {int(np.sum(W_bin_with))} edges")
    print(f"    SHD={shd_with:.0f}, F1={f1_with:.3f}, Precision={prec_with:.3f}, Recall={rec_with:.3f}")
    print(f"    ECE={ece_with:.4f}")
    
    # ── Comparison ─────────────────────────────────────────────────────
    print(f"\n[5] Comparison: With Prior vs Without Prior")
    print(f"    {'Metric':<15} {'No Prior':>10} {'With Prior':>12} {'Δ':>8}")
    print(f"    {'─'*15} {'─'*10} {'─'*12} {'─'*8}")
    
    metrics = [
        ("SHD ↓", shd_no, shd_with, True),
        ("F1 ↑", f1_no, f1_with, False),
        ("Precision ↑", prec_no, prec_with, False),
        ("Recall ↑", rec_no, rec_with, False),
        ("ECE ↓", ece_no, ece_with, True),
    ]
    
    for name, v_no, v_with, lower_better in metrics:
        diff = v_no - v_with if lower_better else v_with - v_no
        arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
        print(f"    {name:<15} {v_no:>10.4f} {v_with:>12.4f} {diff:>+7.4f} {arrow}")
    
    # ── Edge-level Comparison ──────────────────────────────────────────
    print(f"\n[6] Edge-level probability comparison:")
    print(f"    {'Edge':<8} {'Truth':>5} {'No Prior':>10} {'With Prior':>12} {'Prior':>8}")
    print(f"    {'─'*8} {'─'*5} {'─'*10} {'─'*12} {'─'*8}")
    
    # Sort edges by ground truth and disagreement
    flat_np = P_no_cal.flatten()
    flat_wp = P_with_cal.flatten()
    flat_gt = W_true.flatten()
    
    # Show edges where prior changes probability significantly
    abs_diff = np.abs(flat_wp - flat_np)
    top_change = np.argsort(abs_diff)[-15:]  # top 15 by absolute change
    
    for idx in sorted(top_change):
        i, j = idx // d, idx % d
        if i == j:
            continue
        gt_mark = "✓" if flat_gt[idx] > 0 else "✗"
        print(f"    X{i}→X{j}  {gt_mark:>5} {flat_np[idx]:>10.3f} {flat_wp[idx]:>12.3f} {prior_matrix.flatten()[idx]:>8.2f}")
    
    # ── Experiment Suggestions ─────────────────────────────────────────
    print(f"\n[7] 📊 Experiment Suggestions:")
    print(f"    • {'Prior helps!' if shd_with < shd_no else 'Prior not strong enough'}")
    if shd_with < shd_no:
        print(f"    • SHD reduced by {shd_no - shd_with:.0f} edges with LLM prior")
    else:
        print(f"    • Try increasing λ_prior (currently 0.05)")
    print(f"    • {'Precision improved!' if prec_with > prec_no else 'Precision similar'}")
    print(f"    • {'Recall improved!' if rec_with > rec_no else 'Recall similar'}")
    print(f"    • {'Better calibration!' if ece_with < ece_no else 'Calibration similar'}")
    print(f"    • Next: try stronger λ_prior (0.01, 0.05, 0.1, 0.5)")
    print(f"    • Next: full bootstrap (n=50) with priors")
    
    # ── Save ───────────────────────────────────────────────────────────
    results = {
        "without_prior": {
            "shd": float(shd_no), "f1": float(f1_no),
            "precision": float(prec_no), "recall": float(rec_no),
            "ece": float(ece_no), "n_edges": int(np.sum(W_bin_no)),
            "P": P_no_cal.tolist(),
        },
        "with_prior": {
            "shd": float(shd_with), "f1": float(f1_with),
            "precision": float(prec_with), "recall": float(rec_with),
            "ece": float(ece_with), "n_edges": int(np.sum(W_bin_with)),
            "P": P_with_cal.tolist(),
        },
        "prior_matrix": prior_matrix.tolist(),
        "W_true": W_true.tolist(),
    }
    os.makedirs("experiment_results", exist_ok=True)
    ser = {}
    for k, v in results.items():
        if isinstance(v, dict):
            ser[k] = {kk: vv for kk, vv in v.items() if kk != "P"}
    with open("experiment_results/llm_prior_demo.json", "w") as f:
        json.dump(ser, f, indent=2)
    print(f"    Results saved to experiment_results/llm_prior_demo.json")
    
    print(f"\n{'='*70}")
    print(f"  Demo Complete! 🦊")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
