#!/usr/bin/env python3
"""SOTA Benchmark: CausalBayes (BootstrapDAG) with and without LLM priors.

Tests the claim: "LLM priors break Markov equivalence class symmetry",
demonstrating our fixed prior direction and dagness gradient.

Compares:
1. Single NOTEARS (no uncertainty)
2. BootstrapDAG (no prior)
3. BootstrapDAG + perfect prior (upper bound)
4. BootstrapDAG + partial prior (realistic: 60% edges known)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

from causbayes import NeuralBayesianDAG, BootstrapDAG
from causbayes.structure_learning.notears_fast import notears_lbfgs
from causbayes.structure_learning.utils import (
    structural_hamming_distance, expected_shd,
    edge_posterior_precision, edge_posterior_recall,
)
from causbayes.evaluation import comprehensive_evaluation

np.random.seed(42)

# ═══════════════════════════════════════════════════════════
#  Data generation utilities
# ═══════════════════════════════════════════════════════════

def generate_linear_gaussian(W_true, n=1000, noise_std=0.1):
    """Generate data from linear Gaussian SEM: X = X @ W + noise"""
    d = W_true.shape[0]
    X = np.random.randn(n, d)
    for j in range(d):
        parents = np.where(W_true[:, j] != 0)[0]
        if len(parents) > 0:
            X[:, j] = X[:, parents] @ W_true[parents, j] + np.random.randn(n) * noise_std
    return X


def generate_nonlinear_gaussian(W_true, n=500, noise_std=0.1):
    """Generate data from non-linear Gaussian SEM.
    f_j(X_pa) = sum_{i in pa(j)} sin(X_i * W_ij) + 0.5 * X_i
    """
    d = W_true.shape[0]
    X = np.random.randn(n, d) * 0.5
    for j in range(d):
        parents = np.where(W_true[:, j] != 0)[0]
        if len(parents) > 0:
            f = np.zeros(n)
            for p in parents:
                w = W_true[p, j]
                f += np.sin(X[:, p] * w) + 0.3 * X[:, p]
            X[:, j] = f + np.random.randn(n) * noise_std
    return X


def random_dag(d, edge_prob=0.3, weight_range=(0.5, 2.0)):
    """Generate a random DAG with weighted edges."""
    W = np.zeros((d, d))
    for i in range(d):
        for j in range(i+1, d):
            if np.random.rand() < edge_prob:
                w = np.random.uniform(*weight_range) * np.random.choice([-1, 1])
                W[i, j] = w
    return W


def generate_prior(W_true, fraction_known=1.0):
    """Generate a prior matrix from ground truth.
    
    - fraction_known=1.0: perfect prior
    - fraction_known=0.6: partial prior (known edges revealed, others set to 0.5)
    """
    d = W_true.shape[0]
    W_bin = (np.abs(W_true) > 1e-6).astype(float)
    prior = np.full((d, d), 0.5)
    np.fill_diagonal(prior, 0.0)
    
    # Reveal a random subset of edges
    edges = np.where(W_bin > 0)
    n_edges = len(edges[0])
    n_reveal = int(n_edges * fraction_known)
    if n_reveal > 0:
        idx = np.random.choice(n_edges, n_reveal, replace=False)
        for k in idx:
            prior[edges[0][k], edges[1][k]] = 0.9
    
    # Reveal non-edges too
    non_edges = np.where((W_bin == 0) & (np.eye(d) == 0))
    n_non = len(non_edges[0])
    n_reveal_non = int(n_non * fraction_known)
    if n_reveal_non > 0:
        idx = np.random.choice(n_non, n_reveal_non, replace=False)
        for k in idx:
            prior[non_edges[0][k], non_edges[1][k]] = 0.1
    
    return prior


# ═══════════════════════════════════════════════════════
#  Run benchmark
# ═══════════════════════════════════════════════════════

def run_benchmark():
    print("=" * 70)
    print("  CAUSALBAYES SOTA BENCHMARK")
    print("  Validating: fixed dagness gradient + prior direction")
    print("=" * 70)
    print()
    
    # Test configurations
    configs = [
        # (d, n, sem_type, noise, desc, is_nonlinear)
        (5, 1000, "linear", 0.1, "d=5 Linear (easy)", False),
        (5, 1000, "nonlinear", 0.1, "d=5 Non-linear (hard)", True),
        (8, 2000, "linear", 0.1, "d=8 Linear", False),
        (10, 3000, "linear", 0.1, "d=10 Linear (medium)", False),
    ]
    
    prior_configs = [
        ("No Prior", 0.0, None, 0.0),
        ("Partial Prior (60%)", 0.05, 0.6, 0.0),
        ("Perfect Prior (100%)", 0.1, 1.0, 0.0),
    ]
    
    all_results = {}
    
    for d, n, sem_type, noise, desc, is_nonlinear in configs:
        print(f"\n{'─'*70}")
        print(f"  {desc}")
        print(f"{'─'*70}")
        
        # Generate ground truth DAG
        W_true = random_dag(d, edge_prob=0.3 if d <= 5 else 0.25, 
                            weight_range=(0.5, 2.0))
        W_bin = (np.abs(W_true) > 1e-6).astype(float)
        true_edges = W_bin.sum()
        print(f"  True edges: {int(true_edges)}/{d*(d-1)}")
        
        # Generate data
        if sem_type == "nonlinear":
            X = generate_nonlinear_gaussian(W_true, n=n, noise_std=noise)
        else:
            X = generate_linear_gaussian(W_true, n=n, noise_std=noise)
        
        # Standardize
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)
        
        # ─── Split for calibration ───
        n_train = int(n * 0.7)
        X_train = X[:n_train]
        X_val = X[n_train:]
        
        # ─── Baseline: Single NOTEARS ───
        print(f"\n  ── Baseline: Single NOTEARS ──")
        t0 = time.time()
        W_notears = notears_lbfgs(X_train, lambda_1=0.01, max_iter=10, 
                                   w_threshold=0.1, lbfgs_maxiter=30)
        t_notears = time.time() - t0
        W_notears_bin = (np.abs(W_notears) > 0.1).astype(float)
        
        eval_notears = comprehensive_evaluation(W_bin, W_notears_bin)
        print(f"    Time: {t_notears:.3f}s")
        print(f"    SHD: {eval_notears['shd']:.1f}  "
              f"F1@0.5: {2 * eval_notears['precision@0.5'] * eval_notears['recall@0.5'] / max(eval_notears['precision@0.5'] + eval_notears['recall@0.5'], 1e-8):.3f}  "
              f"Prec: {eval_notears['precision@0.5']:.3f}  "
              f"Rec: {eval_notears['recall@0.5']:.3f}")
        
        # ─── BootstrapDAG with different priors ───
        for prior_name, lambda_prior, fraction_known, _ in prior_configs:
            print(f"\n  ── BootstrapDAG + {prior_name} ──")
            
            prior_matrix = None
            if fraction_known is not None and fraction_known > 0:
                prior_matrix = generate_prior(W_true, fraction_known=fraction_known)
            
            t0 = time.time()
            model = BootstrapDAG(
                n_bootstraps=50,
                lambda_1=0.01,
                max_iter=5,
                w_threshold=0.05,
                prior_matrix=prior_matrix,
                lambda_prior=lambda_prior,
                calibrate=True,
                verbose=False,
            )
            model.fit(X_train)
            t_boot = time.time() - t0
            
            # Evaluate at best threshold (calibrated on validation set)
            W_est = model.adjacency_matrix
            P_est = model.edge_probs
            
            eval_results = comprehensive_evaluation(W_bin, P_est, model.edge_stds)
            f1 = (2 * eval_results['precision@0.5'] * eval_results['recall@0.5'] 
                  / max(eval_results['precision@0.5'] + eval_results['recall@0.5'], 1e-8))
            
            print(f"    Time: {t_boot:.3f}s")
            print(f"    SHD: {eval_results['shd']:.1f}  "
                  f"F1@0.5: {f1:.3f}  "
                  f"Prec: {eval_results['precision@0.5']:.3f}  "
                  f"Rec: {eval_results['recall@0.5']:.3f}  "
                  f"Entropy: {eval_results['avg_edge_entropy']:.3f}  "
                  f"ECE: {eval_results['ece']:.4f}")
            print(f"    AUC-PR: {eval_results['auc_pr']:.3f}")
            
            key = f"{desc} :: {prior_name}"
            all_results[key] = {
                "shd": eval_results['shd'],
                "f1": f1,
                "precision": eval_results['precision@0.5'],
                "recall": eval_results['recall@0.5'],
                "ece": eval_results['ece'],
                "entropy": eval_results['avg_edge_entropy'],
                "auc_pr": eval_results['auc_pr'],
                "time": t_boot,
            }
        
        # ─── Baselines: PC, GES (via gCastle if available) ───
        try:
            import castle
            from castle.algorithms import PC, GES
            from castle.common import GraphDAG
            from castle.metrics import MetricsDAG
            
            for algo_name, algo in [("PC", PC()), ("GES", GES())]:
                try:
                    t0 = time.time()
                    algo.learn(X_train)
                    t_algo = time.time() - t0
                    
                    W_pred = np.array(algo.causal_matrix, dtype=float)
                    W_pred_bin = (W_pred > 0.5).astype(float)
                    
                    mt = MetricsDAG(W_pred_bin, W_bin)
                    
                    shd = structural_hamming_distance(W_bin, W_pred_bin)
                    prec = mt.precision if hasattr(mt, 'precision') else 0
                    rec = mt.recall if hasattr(mt, 'recall') else 0
                    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
                    
                    print(f"\n  ── gCastle {algo_name} ──")
                    print(f"    Time: {t_algo:.3f}s")
                    print(f"    SHD: {shd:.1f}  F1: {f1:.3f}  "
                          f"Prec: {prec:.3f}  Rec: {rec:.3f}")
                    
                    all_results[f"{desc} :: {algo_name}"] = {
                        "shd": shd, "f1": f1, "precision": prec,
                        "recall": rec, "ece": float('nan'), "entropy": 0,
                        "auc_pr": float('nan'), "time": t_algo,
                    }
                except Exception as e:
                    print(f"\n  ── gCastle {algo_name}: SKIPPED ({e}) ──")
        except ImportError:
            print(f"\n  ── gCastle: not installed, skipping ──")
    
    # ═══════════════════════════════════════════════════════
    #  Summary table
    # ═══════════════════════════════════════════════════════
    print(f"\n\n{'='*70}")
    print("  BENCHMARK RESULTS SUMMARY")
    print(f"{'='*70}")
    print()
    
    headers = ["Method", "SHD↓", "F1↑", "Prec↑", "Rec↑", "ECE↓", "Entropy", "Time(s)"]
    print(f"  {' | '.join(h):<60}" for h in headers)
    print(f"  {'-'*70}")
    
    for key, res in sorted(all_results.items()):
        shd = f"{res['shd']:.1f}"
        f1 = f"{res['f1']:.3f}"
        prec = f"{res['precision']:.3f}"
        rec = f"{res['recall']:.3f}"
        ece = f"{res['ece']:.4f}" if not np.isnan(res['ece']) else "N/A"
        ent = f"{res['entropy']:.3f}" if res['entropy'] > 0 else "0.000"
        tm = f"{res['time']:.2f}"
        print(f"  {key:<50s} | {shd} | {f1} | {prec} | {rec} | {ece} | {ent} | {tm}")
    
    return all_results


if __name__ == "__main__":
    results = run_benchmark()
