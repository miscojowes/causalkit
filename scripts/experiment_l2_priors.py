#!/usr/bin/env python3
"""
L2 Prior Experiment: Test whether L2 priors improve NOTEARS accuracy.

Tests 3 conditions:
  a) No prior (current baseline behavior)
  b) Correct prior (known true edges have prior=0.9)
  c) Misleading prior (known non-edges misleadingly have prior=0.9)

For each condition, we run bootstrap NOTEARS with an L2 prior penalty:
  L2_prior = lambda_prior * sum_{i,j} prior[i,j] * (abs(W[i,j]) - mu[i,j])^2
  where mu[i,j] = 0.5 if prior[i,j] >= 0.5 else 0.0

Reports SHD, Precision, Recall for each condition.
"""

import sys, os, json, time, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from tqdm import trange
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from causbayes.structure_learning.utils import structural_hamming_distance, dagness
from causbayes.structure_learning.notears_fast import notears_lbfgs

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiment_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS = list(range(42, 52))  # 10 seeds


def generate_data(d=5, n=1000, edge_prob=0.2, noise_scale=0.1, seed=42):
    rng = np.random.RandomState(seed)
    W_true = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < edge_prob:
                W_true[i, j] = rng.uniform(0.5, 1.5) * rng.choice([-1, 1])
    X = np.zeros((n, d))
    for j in range(d):
        parents = np.where(W_true[:, j] != 0)[0]
        if len(parents) > 0:
            X[:, j] = X[:, parents] @ W_true[parents, j]
        X[:, j] += rng.randn(n) * noise_scale
    return X, (np.abs(W_true) > 1e-6).astype(float)


def notears_with_prior(
    X: np.ndarray,
    prior_matrix: np.ndarray = None,
    lambda_1: float = 0.01,
    lambda_prior: float = 0.1,
    max_iter: int = 60,
    lr: float = 1e-2,
    rho_max: float = 1e8,
    seed: int = 42,
    verbose: bool = False,
) -> np.ndarray:
    """Linear NOTEARS with L2 prior penalty.

    Modified loss:
      L = ||X - XW||² + λ₁||W||₁ + λ_prior * sum prior[i,j] * (|W[i,j]| - mu[i,j])²
      + acyclicity penalty

    where mu[i,j] = 0.5 if prior[i,j] >= 0.5 else 0.0
    """
    torch.manual_seed(seed)
    d = X.shape[1]
    X_t = torch.from_numpy(X).float()

    W = torch.zeros(d, d, requires_grad=True)
    with torch.no_grad():
        W.data.add_(torch.randn(d, d) * 1e-3)

    optimizer = torch.optim.AdamW([W], lr=lr, weight_decay=0.0)

    # Setup prior
    if prior_matrix is None:
        prior_matrix = np.zeros((d, d))

    prior_t = torch.from_numpy(prior_matrix).float()
    # mu[i,j] = 0.5 if prior[i,j] >= 0.5 else 0.0
    mu = (prior_t >= 0.5).float() * 0.5

    rho = 1.0
    alpha = 0.0
    h = np.inf
    best_h = np.inf
    best_W = None
    stall_count = 0

    pbar = trange(max_iter, desc="NOTEARS+L2Prior", disable=not verbose)

    for outer in pbar:
        n_inner = min(15, 3 + outer)
        for _ in range(n_inner):
            optimizer.zero_grad()
            X_pred = X_t @ W.T
            recon = torch.mean((X_t - X_pred) ** 2)
            l1 = lambda_1 * torch.sum(torch.abs(W))

            # L2 prior penalty
            W_abs = torch.abs(W)
            prior_penalty = prior_t * (W_abs - mu) ** 2
            l2_prior = lambda_prior * torch.sum(prior_penalty)

            h_val = dagness(W)
            h_penalty = alpha * h_val + 0.5 * rho * h_val ** 2
            loss = recon + l1 + l2_prior + h_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_([W], 5.0)
            optimizer.step()

        with torch.no_grad():
            h_new = dagness(W).item()

        if not np.isnan(h_new) and h_new < best_h:
            best_h = h_new
            best_W = W.detach().clone().numpy()
            stall_count = 0
        else:
            stall_count += 1

        if stall_count >= 10:
            if verbose:
                pbar.set_postfix({"h(W)": f"{h_new:.2e}", "stopped": "improving"}, refresh=False)
            break

        if h_new < 1e-8:
            if verbose:
                pbar.set_postfix({"h(W)": f"{h_new:.2e}", "converged": True}, refresh=False)
            best_W = W.detach().clone().numpy()
            break

        if np.isnan(h_new) or np.isnan(W.detach().numpy()).any():
            if verbose:
                pbar.set_postfix({"h(W)": "NaN"}, refresh=False)
            break

        if h_new > 0.25 * h and h < np.inf:
            rho = min(rho * 10, rho_max)
        alpha += rho * h_new
        h = h_new

    if best_W is None:
        return np.zeros((d, d))

    return best_W


def bootstrap_with_prior(
    X: np.ndarray,
    prior_matrix: np.ndarray = None,
    n_bootstraps: int = 50,
    lambda_1: float = 0.01,
    lambda_prior: float = 0.1,
    seed: int = 42,
) -> tuple:
    """Bootstrap NOTEARS with L2 prior."""
    d = X.shape[1]
    W_list = []
    n_failed = 0

    for i in range(n_bootstraps):
        try:
            X_boot = resample(X, random_state=seed + i)
            W_i = notears_with_prior(
                X_boot,
                prior_matrix=prior_matrix,
                lambda_1=lambda_1,
                lambda_prior=lambda_prior,
                seed=seed + i + 1000,
                verbose=False,
            )
            if not np.isnan(W_i).any():
                W_list.append(W_i)
            else:
                n_failed += 1
        except Exception:
            n_failed += 1

    if len(W_list) == 0:
        return np.zeros((d, d)), np.zeros((d, d))

    W_stack = np.array(W_list)
    W_abs = np.abs(W_stack)
    all_w = W_abs[W_abs > 1e-8].ravel()

    if len(all_w) == 0:
        return np.zeros((d, d)), np.zeros((d, d))

    threshold = np.percentile(all_w, 95) if len(all_w) > 1 else 0.1
    P = np.mean(W_abs > max(threshold, 1e-4), axis=0)
    S = np.std(W_abs, axis=0)
    np.fill_diagonal(P, 0.0)
    np.fill_diagonal(S, 0.0)

    return P, S


def evaluate(W_true, P):
    """Compute metrics from edge probabilities."""
    SHD = structural_hamming_distance(W_true, (P >= 0.5).astype(float))
    tp = np.sum((P >= 0.5) & (W_true > 0))
    fp = np.sum((P >= 0.5) & (W_true == 0))
    fn = np.sum((P < 0.5) & (W_true > 0))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "shd": float(SHD),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "true_edges": int(np.sum(W_true > 0)),
        "est_edges": int(np.sum(P >= 0.5)),
    }


def main():
    print("=" * 80)
    print("  L2 Prior Experiment")
    print("  Tests: No prior, Correct prior, Misleading prior")
    print("=" * 80)

    lambda_prior_values = [0.0, 0.05, 0.1, 0.5, 1.0]
    all_results = []

    for seed in SEEDS:
        print(f"\n{'─' * 60}")
        print(f"  Seed {seed}")
        print(f"{'─' * 60}")

        X, W_true = generate_data(d=5, n=1000, seed=seed)

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        d = X_scaled.shape[1]
        te = int(np.sum(W_true > 0))
        print(f"    True edges: {te}")

        row = {"seed": seed, "true_edges": te}

        # ─── Condition a: No prior (lambda_prior=0) ───────────
        print(f"\n    ┌─ Condition A: No Prior")
        prior_none = np.zeros((d, d))
        for lp in lambda_prior_values:
            t0 = time.time()
            P, S = bootstrap_with_prior(
                X_scaled, prior_matrix=prior_none if lp > 0 else None,
                n_bootstraps=50, lambda_prior=lp, seed=seed
            )
            elapsed = time.time() - t0
            metrics = evaluate(W_true, P)
            key = f"no_prior_lp{lp}"
            row[key] = {**metrics, "time_s": round(elapsed, 1)}
            print(f"    │  λ_prior={lp:.2f} SHD={metrics['shd']:.1f} "
                  f"P={metrics['precision']:.2f} R={metrics['recall']:.2f} "
                  f"t={elapsed:.1f}s")

        # ─── Condition b: Correct prior ────────────────────────
        print(f"\n    ┌─ Condition B: Correct Prior")
        prior_correct = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                if W_true[i, j] > 0:
                    prior_correct[i, j] = 0.9  # Known edges
        for lp in lambda_prior_values:
            if lp == 0.0:
                continue  # Same as no prior
            t0 = time.time()
            P, S = bootstrap_with_prior(
                X_scaled, prior_matrix=prior_correct,
                n_bootstraps=50, lambda_prior=lp, seed=seed
            )
            elapsed = time.time() - t0
            metrics = evaluate(W_true, P)
            key = f"correct_prior_lp{lp}"
            row[key] = {**metrics, "time_s": round(elapsed, 1)}
            print(f"    │  λ_prior={lp:.2f} SHD={metrics['shd']:.1f} "
                  f"P={metrics['precision']:.2f} R={metrics['recall']:.2f} "
                  f"t={elapsed:.1f}s")

        # ─── Condition c: Misleading prior ────────────────────
        print(f"\n    ┌─ Condition C: Misleading Prior")
        prior_mislead = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                if W_true[i, j] == 0 and i != j:
                    prior_mislead[i, j] = 0.9  # Wrong edges
        for lp in lambda_prior_values:
            if lp == 0.0:
                continue
            t0 = time.time()
            P, S = bootstrap_with_prior(
                X_scaled, prior_matrix=prior_mislead,
                n_bootstraps=50, lambda_prior=lp, seed=seed
            )
            elapsed = time.time() - t0
            metrics = evaluate(W_true, P)
            key = f"mislead_prior_lp{lp}"
            row[key] = {**metrics, "time_s": round(elapsed, 1)}
            print(f"    │  λ_prior={lp:.2f} SHD={metrics['shd']:.1f} "
                  f"P={metrics['precision']:.2f} R={metrics['recall']:.2f} "
                  f"t={elapsed:.1f}s")

        all_results.append(row)

    # ═══════════════════════════════════════════════════════════════
    #  SUMMARY (focus on best λ_prior for each condition)
    # ═══════════════════════════════════════════════════════════════

    print(f"\n\n{'=' * 80}")
    print("  SUMMARY: Best λ_prior per condition (mean ± std, 10 seeds)")
    print(f"{'=' * 80}")

    for condition_label, prefix in [
        ("No Prior", "no_prior_lp"),
        ("Correct Prior", "correct_prior_lp"),
        ("Misleading Prior", "mislead_prior_lp"),
    ]:
        print(f"\n  ── {condition_label} ──")
        print(f"  {'λ_prior':<10} {'SHD':<8} {'Precision':<12} {'Recall':<10} {'F1':<8} {'Time':<8}")
        print(f"  {'─'*8} {'─'*8} {'─'*10} {'─'*10} {'─'*8} {'─'*8}")

        for lp in lambda_prior_values:
            if condition_label == "No Prior":
                key = f"{prefix}{lp}"
            else:
                if lp == 0:
                    continue
                key = f"{prefix}{lp}"

            shds = []
            precs = []
            recs = []
            f1s = []
            times = []

            for row in all_results:
                r = row.get(key, {})
                if isinstance(r, dict):
                    if "shd" in r and not (isinstance(r["shd"], float) and np.isnan(r["shd"])):
                        shds.append(r["shd"])
                        precs.append(r.get("precision", float("nan")))
                        recs.append(r.get("recall", float("nan")))
                        f1s.append(r.get("f1", float("nan")))
                        times.append(r.get("time_s", 0))

            if shds:
                print(f"  {lp:<8.2f} {np.mean(shds):<8.1f} {np.mean(precs):<10.2f}±{np.std(precs):.2f} "
                      f"{np.mean(recs):<8.2f}±{np.std(recs):.2f} {np.mean(f1s):<8.2f} {np.mean(times):<8.1f}")

    # ═══════════════════════════════════════════════════════════════
    #  SAVE
    # ═══════════════════════════════════════════════════════════════

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    output = {
        "metadata": {
            "description": "L2 Prior experiment: No prior vs Correct prior vs Misleading prior",
            "n_seeds": len(SEEDS),
            "seeds": SEEDS,
            "d": 5,
            "n": 1000,
            "lambda_prior_values": lambda_prior_values,
        },
        "by_seed": all_results,
    }

    outpath = os.path.join(RESULTS_DIR, "l2_priors_results.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=convert)
    print(f"\n  Results saved to {outpath}")

    print(f"\n{'=' * 80}")
    print("  L2 PRIOR EXPERIMENT COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
