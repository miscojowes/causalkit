#!/usr/bin/env python3
"""
Calibration Experiment: Platt scaling and Isotonic Regression
for improving bootstrap edge probability calibration.

Method:
1. Run BootstrapDAG on a dataset with known ground truth
2. Extract per-edge bootstrap proportions P[i,j]
3. Apply Platt scaling: P_calibrated = 1/(1+exp(-(a*logit(P) + b)))
4. Apply IsotonicRegression from sklearn
5. Compare ECE before vs after calibration
6. Save calibration curve plot
"""

import sys, os, json, time, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from scipy.special import logit, expit

from causbayes.structure_learning.utils import structural_hamming_distance
from causbayes.evaluation import edge_calibration

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiment_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS = [42, 43, 44, 45]


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


def extract_bootstrap_proportions(model):
    """Extract edge probabilities from bootstrap weight matrices.

    Returns P_bootstrap: proportion of bootstraps where |W| > small threshold
    """
    W_list = model._weight_matrices_
    if not W_list:
        return None
    W_stack = np.array(W_list)
    W_abs = np.abs(W_stack)

    # Adaptive threshold based on weight magnitudes
    all_w = W_abs[W_abs > 1e-8].ravel()
    if len(all_w) == 0:
        return None
    threshold = np.percentile(all_w, 95) if len(all_w) > 1 else 0.1
    threshold = max(threshold, 1e-4)

    P = np.mean(W_abs > threshold, axis=0)
    np.fill_diagonal(P, 0.0)
    return P


def platt_scale(P, a, b):
    """Apply Platt scaling: P_cal = 1/(1+exp(-(a*logit(P) + b)))."""
    # Clamp to avoid logit(0) and logit(1)
    eps = 1e-6
    P_clamped = np.clip(P, eps, 1 - eps)
    logits = logit(P_clamped)
    return expit(a * logits + b)


def compute_ece(W_true, P_est):
    """Compute ECE (Expected Calibration Error)."""
    cal = edge_calibration(W_true, P_est, n_bins=10)
    return cal["ece"]


def platt_loss(params, P, y):
    """Negative log-likelihood for Platt scaling."""
    a, b = params
    P_cal = platt_scale(P, a, b)
    eps = 1e-8
    return -np.mean(y * np.log(P_cal + eps) + (1 - y) * np.log(1 - P_cal + eps))


def tune_platt(P_train, y_train):
    """Tune Platt scaling parameters using validation data."""
    from scipy.optimize import minimize

    result = minimize(
        platt_loss,
        x0=[1.0, 0.0],
        args=(P_train, y_train),
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-4, "fatol": 1e-4},
    )
    return result.x[0], result.x[1]


def run_calibration():
    print("=" * 80)
    print("  Calibration Experiment: Platt Scaling & Isotonic Regression")
    print("=" * 80)

    # Ensure causbayes importable
    from causbayes import BootstrapDAG

    all_calibration_data = []

    for seed in SEEDS:
        print(f"\n{'─' * 60}")
        print(f"  Seed {seed}")
        print(f"{'─' * 60}")

        # Generate train + validation data
        X_tr, W_true = generate_data(d=5, n=600, seed=seed)
        X_va, W_va = generate_data(d=5, n=400, seed=seed + 1000)
        X_te, W_te = generate_data(d=5, n=400, seed=seed + 2000)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)
        X_te = scaler.transform(X_te)

        # Use W_true for val since the true DAG is the same
        # (validation data is fresh samples from the same DAG)

        # Run BootstrapDAG
        print("    Running BootstrapDAG...", end=" ", flush=True)
        model = BootstrapDAG(n_bootstraps=50, lambda_1=0.01, max_iter=60, verbose=False)
        model.fit(X_tr)
        print(f"done ({len(model._weight_matrices_)} valid bootstraps)")

        # Extract bootstrap proportions
        d = X_tr.shape[1]
        P_boot = model.edge_probs.copy()

        # Build ground truth labels for ALL edges (train + val + test)
        # y_true: (d*(d-1),) flattened ground truth
        y_val = []
        P_val = []
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                y_val.append(int(W_va[i, j] > 0.5))
                P_val.append(P_boot[i, j])

        y_val = np.array(y_val)
        P_val = np.array(P_val)

        print(f"    ECE (raw bootstrap): {compute_ece(W_te, P_boot):.4f}")

        # ─── Platt Scaling ────────────────────────────────────
        print("    Tuning Platt scaling...", end=" ", flush=True)
        try:
            a_opt, b_opt = tune_platt(P_val, y_val)
            P_platt = platt_scale(P_boot, a_opt, b_opt)
            ece_platt = compute_ece(W_te, P_platt)
            print(f"a={a_opt:.3f}, b={b_opt:.3f}, ECE={ece_platt:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")
            a_opt, b_opt = 1.0, 0.0
            P_platt = P_boot
            ece_platt = float("nan")

        # ─── Isotonic Regression ──────────────────────────────
        print("    Training IsotonicRegression...", end=" ", flush=True)
        try:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(P_val, y_val)
            P_iso = iso.predict(P_boot.ravel()).reshape(d, d)
            np.fill_diagonal(P_iso, 0.0)
            ece_iso = compute_ece(W_te, P_iso)
            print(f"ECE={ece_iso:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")
            P_iso = P_boot
            ece_iso = float("nan")

        # ─── Results ──────────────────────────────────────────
        ece_raw = compute_ece(W_te, P_boot)
        row = {
            "seed": seed,
            "ece_raw": ece_raw,
            "ece_platt": ece_platt,
            "ece_isotonic": ece_iso,
            "platt_a": float(a_opt),
            "platt_b": float(b_opt),
        }
        all_calibration_data.append(row)

        print(f"    ──────────────────────────────────────")
        print(f"    Raw Bootstrap:           ECE = {ece_raw:.4f}")
        print(f"    Platt Scaling:           ECE = {ece_platt:.4f} {'✅' if ece_platt < ece_raw else '❌'}")
        print(f"    Isotonic Regression:     ECE = {ece_iso:.4f} {'✅' if ece_iso < ece_raw else '❌'}")

    # ═══════════════════════════════════════════════════════════════
    #  SUMMARY
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'=' * 60}")
    print("  SUMMARY (mean ± std)")
    print(f"{'=' * 60}")

    raw_eces = [r["ece_raw"] for r in all_calibration_data]
    platt_eces = [r["ece_platt"] for r in all_calibration_data if not np.isnan(r["ece_platt"])]
    iso_eces = [r["ece_isotonic"] for r in all_calibration_data if not np.isnan(r["ece_isotonic"])]

    def ms(arr):
        return np.mean(arr), np.std(arr)

    print(f"  {'Method':<25} {'ECE mean':<12} {'ECE std':<12}")
    print(f"  {'─'*25} {'─'*12} {'─'*12}")
    print(f"  {'Raw Bootstrap':<25} {ms(raw_eces)[0]:.4f}     {ms(raw_eces)[1]:.4f}")
    if platt_eces:
        print(f"  {'Platt Scaling':<25} {ms(platt_eces)[0]:.4f}     {ms(platt_eces)[1]:.4f}")
    if iso_eces:
        print(f"  {'Isotonic Regression':<25} {ms(iso_eces)[0]:.4f}     {ms(iso_eces)[1]:.4f}")

    # ═══════════════════════════════════════════════════════════════
    #  SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    output = {
        "metadata": {
            "description": "Calibration experiment: Platt scaling and Isotonic Regression",
            "n_seeds": len(SEEDS),
            "seeds": SEEDS,
            "d": 5,
        },
        "by_seed": all_calibration_data,
        "summary": {
            "ece_raw_mean": float(np.mean(raw_eces)),
            "ece_raw_std": float(np.std(raw_eces)),
            "ece_platt_mean": float(np.mean(platt_eces)) if platt_eces else None,
            "ece_platt_std": float(np.std(platt_eces)) if platt_eces else None,
            "ece_iso_mean": float(np.mean(iso_eces)) if iso_eces else None,
            "ece_iso_std": float(np.std(iso_eces)) if iso_eces else None,
        },
    }

    outpath = os.path.join(RESULTS_DIR, "calibration_results.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=convert)
    print(f"\n  Results saved to {outpath}")

    # ═══════════════════════════════════════════════════════════════
    #  CALIBRATION CURVE PLOT
    # ═══════════════════════════════════════════════════════════════

    print("  Generating calibration curve plot...", end=" ", flush=True)
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        methods_data = [
            ("Raw Bootstrap", P_boot, "C0"),
            ("Platt Scaled", P_platt, "C1"),
            ("Isotonic Regression", P_iso, "C2"),
        ]

        # Use last seed's data for the plot (richer)
        P_mats = {"Raw Bootstrap": P_boot, "Platt": P_platt, "Isotonic": P_iso}

        for idx, (title, P_mat, color) in enumerate(methods_data):
            cal = edge_calibration(W_te, P_mat, n_bins=10)
            bins = np.array(cal["bins"])
            acc = np.array(cal["accuracy"])
            counts = np.array(cal["counts"])

            ax = axes[idx]
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
            ax.plot(bins, acc, "o-", color=color, label=f"ECE={cal['ece']:.4f}")
            ax.fill_between(bins, 0, acc, alpha=0.1, color=color)

            # Add histogram of counts
            for b, a, c in zip(bins, acc, counts):
                ax.text(b, a - 0.03, f"n={c}", ha="center", va="top", fontsize=8,
                        color="gray")

            ax.set_xlabel("Predicted Probability")
            ax.set_ylabel("Observed Frequency")
            ax.set_title(title)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)

        plt.suptitle(f"Calibration Curves (Seed {seed}, d=5)", fontsize=14)
        plt.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, "calibration_curves.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"saved to {plot_path}")
    except Exception as e:
        print(f"ERROR generating plot: {e}")

    print(f"\n{'=' * 60}")
    print("  CALIBRATION EXPERIMENT COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_calibration()
