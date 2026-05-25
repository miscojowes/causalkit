"""
Systematic benchmark: CausalBayes vs baselines on synthetic data.
"""

import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "benchmark_results.json")


# ═══════════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════════

def generate_dag(d, n, graph_type="random", edge_prob=0.2, noise_scale=0.1, seed=42):
    """Generate data from a DAG with linear Gaussian SEM."""
    rng = np.random.RandomState(seed)
    W = np.zeros((d, d))

    if graph_type == "random":
        for i in range(d):
            for j in range(i + 1, d):
                if rng.random() < edge_prob:
                    W[i, j] = rng.uniform(0.5, 1.5) * rng.choice([-1, 1])

    elif graph_type == "chain":
        for i in range(d - 1):
            W[i, i + 1] = 1.0

    elif graph_type == "scale_free":
        in_deg = np.zeros(d)
        for i in range(1, min(3, d)):
            W[0, i] = rng.uniform(0.5, 1.0)
            in_deg[i] += 1
        for j in range(3, d):
            probs = in_deg[:j] / max(in_deg[:j].sum(), 1)
            n_parents = max(1, rng.randint(1, min(4, j + 1)))
            parents = rng.choice(j, size=n_parents, p=probs, replace=False)
            for p in parents:
                W[p, j] = rng.uniform(0.5, 1.0)
                in_deg[j] += 1

    # Generate data
    X = np.zeros((n, d))
    for j in range(d):
        parents = np.where(W[:, j] != 0)[0]
        if len(parents) > 0:
            X[:, j] = X[:, parents] @ W[parents, j]
        X[:, j] += rng.randn(n) * noise_scale

    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    W_bin = (np.abs(W) > 1e-6).astype(float)
    return X, W_bin


def generate_nonlinear_chain(d, n, noise_scale=0.15, seed=42):
    """Non-linear chain: X_{j} = sin(X_{j-1}) + cos(2*X_{j-1}) / 2 + noise."""
    rng = np.random.RandomState(seed)
    W = np.zeros((d, d))
    for i in range(d - 1):
        W[i, i + 1] = 1.0

    X = np.zeros((n, d))
    X[:, 0] = rng.randn(n)
    for j in range(1, d):
        X[:, j] = np.sin(X[:, j - 1]) + 0.3 * np.cos(2 * X[:, j - 1]) + rng.randn(n) * noise_scale

    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X, W


# ═══════════════════════════════════════════════════════════════════════
#  METHODS
# ═══════════════════════════════════════════════════════════════════════

def run_causbayes(X, **kwargs):
    from causbayes import NeuralBayesianDAG
    t0 = time.time()
    model = NeuralBayesianDAG(
        hidden_layers=[16, 16],
        learning_rate=1e-2,
        lambda_1=5e-3,
        lambda_2=2.0,
        uncertainty="mc_dropout",
        mc_samples=10,
        max_iter=25,
        verbose=False,
    )
    model.fit(X)
    elapsed = time.time() - t0
    return {"P": model.edge_probs, "std": model.edge_stds, "time": elapsed}


def run_pc(X, alpha=0.05):
    from causallearn.search.ConstraintBased.PC import pc
    t0 = time.time()
    try:
        cg = pc(X, alpha, "fisherz")
        elapsed = time.time() - t0
        d = X.shape[1]
        W = np.zeros((d, d))
        g = cg.G.graph
        for i in range(d):
            for j in range(d):
                if g[i, j] == 1 and g[j, i] == -1:
                    W[i, j] = 1.0
        return {"P": W, "std": np.zeros((d, d)), "time": elapsed}
    except Exception as e:
        print(f"    PC ERROR: {e}")
        return None


def run_ges(X):
    from causallearn.search.ScoreBased.GES import ges
    t0 = time.time()
    try:
        res = ges(X, score_func="local_score_BIC")
        elapsed = time.time() - t0
        d = X.shape[1]
        W = np.zeros((d, d))
        g = res['G'].graph
        for i in range(d):
            for j in range(d):
                if g[i, j] == 1 and g[j, i] == -1:
                    W[i, j] = 1.0
        return {"P": W, "std": np.zeros((d, d)), "time": elapsed}
    except Exception as e:
        print(f"    GES ERROR: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def evaluate(W_true, result):
    if result is None:
        return None
    from causbayes.evaluation import comprehensive_evaluation
    P = result["P"]
    std = result.get("std")
    try:
        m = comprehensive_evaluation(W_true, P, std)
    except Exception:
        m = {}

    W_bin = (P >= 0.5).astype(float)
    tp = np.sum((W_bin > 0) & (W_true > 0))
    fp = np.sum((W_bin > 0) & (W_true == 0))
    fn = np.sum((W_bin == 0) & (W_true > 0))
    tn = np.sum((W_bin == 0) & (W_true == 0))

    # Also compute best achievable SHD via threshold search
    thresh_range = np.linspace(0.05, 0.95, 19)
    best_shd = float('inf')
    best_thresh = 0.5
    for t in thresh_range:
        W_t = (P >= t).astype(float)
        shd_t = 0.5 * np.sum(np.abs(W_t - W_true))
        if shd_t < best_shd:
            best_shd = shd_t
            best_thresh = t

    return {
        "shd_at_0.5": m.get("shd", float("nan")),
        "best_shd": best_shd,
        "best_thresh": float(best_thresh),
        "expected_shd": m.get("expected_shd", float("nan")),
        "auc_pr": m.get("auc_pr", float("nan")),
        "ece": m.get("ece", float("nan")),
        "coverage": m.get("coverage@0.9", float("nan")),
        "precision": tp / (tp + fp) if (tp + fp) > 0 else float("nan"),
        "recall": tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
        "true_edges": int(np.sum(W_true > 0)),
        "est_edges": int(np.sum(W_bin > 0)),
        "time_s": result["time"],
    }


# ═══════════════════════════════════════════════════════════════════════
#  EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════

EXPERIMENTS = [
    ("random_d5_n500", 5, 500, "random", {"edge_prob": 0.3}),
    ("random_d10_n1000", 10, 1000, "random", {"edge_prob": 0.2}),
    ("random_d15_n2000", 15, 2000, "random", {"edge_prob": 0.15}),
    ("chain_d5_n500", 5, 500, "chain", {}),
    ("chain_d10_n1000", 10, 1000, "chain", {}),
    ("scale_free_d6_n600", 6, 600, "scale_free", {}),
    ("nonlinear_chain_d6_n1000", 6, 1000, "_nonlinear", {}),
]

METHODS = {
    "CausalBayes": run_causbayes,
    "PC": run_pc,
    "GES": run_ges,
}


def main():
    print("=" * 80)
    print("  CausalBayes Systematic Benchmark")
    print("=" * 80)

    all_results = {}

    for exp_name, d, n, gtype, gkwargs in EXPERIMENTS:
        print(f"\n{'─' * 70}")
        print(f"  {exp_name}: d={d}, n={n}, type={gtype}")
        print(f"{'─' * 70}")

        for seed in [42, 123]:
            if gtype == "_nonlinear":
                X, W_true = generate_nonlinear_chain(d, n, seed=seed)
            else:
                X, W_true = generate_dag(d, n, graph_type=gtype, seed=seed, **gkwargs)

            te = int(np.sum(W_true > 0))
            print(f"    seed={seed}: true_edges={te}")

            row = {"experiment": exp_name, "d": d, "n": n, "seed": seed, "true_edges": te}

            for mname, mfunc in METHODS.items():
                try:
                    result = mfunc(X)
                    if result is not None:
                        metrics = evaluate(W_true, result)
                        row[mname] = metrics
                        print(f"      {mname:<15} SHD@0.5={metrics['shd_at_0.5']:.1f} "
                              f"best_SHD={metrics['best_shd']:.1f} "
                              f"P={metrics['precision']:.2f} R={metrics['recall']:.2f} "
                              f"AUC-PR={metrics['auc_pr']:.2f} "
                              f"t={metrics['time_s']:.1f}s")
                except Exception as e:
                    print(f"      {mname:<15} ERROR: {e}")
                    import traceback; traceback.print_exc()

            key = f"{exp_name}_seed{seed}"
            all_results[key] = row

    # ─── Summary Table ──────────────────────────────────────────
    print(f"\n\n{'=' * 90}")
    print("  SUMMARY")
    print(f"{'=' * 90}")
    print(f"  {'Experiment':<25} {'Method':<15} {'SHD@0.5':<8} {'BestSHD':<8} "
          f"{'P':<6} {'R':<6} {'AUCPR':<6} {'ECE':<6}")
    print(f"  {'─'*25} {'─'*15} {'─'*8} {'─'*8} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")

    for exp_name, _, _, _, _ in EXPERIMENTS:
        for seed in [42, 123]:
            key = f"{exp_name}_seed{seed}"
            row = all_results.get(key, {})
            for mname in METHODS:
                m = row.get(mname)
                if m:
                    print(f"  {exp_name:<25} {mname:<15} {m['shd_at_0.5']:<8.1f} "
                          f"{m['best_shd']:<8.1f} {m['precision']:<6.2f} "
                          f"{m['recall']:<6.2f} {m['auc_pr']:<6.2f} "
                          f"{m['ece']:<6.3f}")

    # Save
    serializable = {}
    for k, v in all_results.items():
        clean = {}
        for kk, vv in v.items():
            if isinstance(vv, dict):
                clean[kk] = {kkk: (float(vvv) if isinstance(vvv, (np.floating, float, np.integer)) else vvv)
                             for kkk, vvv in vv.items()}
            else:
                clean[kk] = vv
        serializable[k] = clean

    with open(RESULTS_FILE, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
