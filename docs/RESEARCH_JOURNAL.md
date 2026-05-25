# CausalBayes — Research Journal

## Methodology & Decisions

Track of every design decision, its research basis, empirical validation, and whether it worked.

---

## 2026-05-25: Initial Architecture

### Decision: Neural NOTEARS with Per-Variable MLPs

**Basis:** Lachapelle et al. (2020) "Gradient-based Neural DAG Learning" shows per-variable MLPs outperform shared architectures for non-linear causal discovery. Each variable gets its own function f_j(X, θⱼ), enabling variable-specific non-linearities.

**Alternative considered:** Shared encoder + per-variable decoders (GraN-DAG's simpler variant). Rejected because shared encoder forces all variables to use the same representation, which can miss variable-specific patterns.

**Empirical check needed:** Does the per-variable MLP cost justify the accuracy gain? For d < 20 the cost is manageable. For d > 50 this may need rethinking.

**Status:** Implemented. Tests pass on d=5, d=10 synthetic data.

### Decision: Acyclicity via Augmented Lagrangian (NOTEARS)

**Basis:** Zheng et al. (2018) proved that h(W) = tr(exp(W⊙W)) - d = 0 iff W is a DAG. Augmented Lagrangian optimization is the standard method for this constraint.

**Implementation notes:** torch.matrix_exp is unstable on arm64 CPU. Switched to power series approximation (20 terms) with normalization to prevent overflow.

**Status:** Working. Converges to h(W) ≈ 0 in ~10-15 outer iterations.

### Decision: MC Dropout for Uncertainty

**Basis:** Gal & Ghahramani (2016) shows dropout at inference approximates Bayesian inference over NN weights. Applied per-variable MLPs, each MC sample produces a different weight matrix, giving a distribution over DAGs.

**Initial approach:** Sigmoid of normalized mean weights → poor calibration.
**Current approach:** SNR-based (mean/std) → rank normalization → better spread.

**Alternative to test:** Variational Inference via Bayes-by-Backprop (Blundell et al., 2015). Would give proper posterior but is computationally expensive for per-variable MLPs.

**Hypothesis to test:** MC Dropout should give reasonable uncertainty for edge detection because dropout at inference changes which features each MLP uses, directly affecting the weight matrix. If an edge is consistently activated across MC samples, it's a robust discovery.

**Status:** Implemented, needs calibration validation on known ground truth.

### Decision: LLM as Soft Prior Only (Not Edge Decider)

**Basis:** Wu et al. (2025) "LLM Cannot Discover Causality" proves that using LLM outputs as hard constraints degrades performance. LLMs should only assist search (non-decisional role).

**Implementation:** LLM extracts causal suggestions → converted to soft prior matrix (probabilities [0,1]) → used as KL-divergence regularization in loss. Data can override LLM priors.

**Status:** Module built, needs end-to-end testing with real API calls.

### Decision: Spike-and-Slab + Horseshoe Priors

**Basis:** Carvalho et al. (2010) shows horseshoe prior has optimal shrinkage properties. Spike-and-slab is the classic Bayesian variable selection prior. Both encourage sparsity while allowing strong signals.

**Implementation:** Used as regularization in loss (not full Bayesian yet). True Bayesian treatment would require sampling (MCMC) which is too slow for structure learning.

**Status:** Implemented as prior regularization terms.

---

## Empirical Findings (2026-05-25 Benchmarks)

### Finding 1: NOTEARS produces dense graphs (expected)
**Benchmark:** Linear data, d=5, 4 true edges. Linear NOTEARS converges to SHD=11 (all weights ~equal magnitude).
**Root cause:** Gaussian observational data only identifies the Markov equivalence class, not the exact DAG. Many DAGs fit equally well. NOTEARS finds ONE DAG (any DAG), not the true one.
**Implication:** The single-DAG output of NOTEARS is MISLEADING when multiple DAGs fit. This is the central justification for CausalBayes — we should output uncertainty, not false precision.

### Finding 2: PC algorithm is more robust on linear Gaussian data
**Benchmark:** PC (causal-learn) on same d=5 data → SHD=1.0, AUC-PR=0.55. Better than NOTEARS.
**Why:** PC uses conditional independence tests (Fisher-Z) which correctly identify the CPDAG skeleton. It doesn't try to orient edges it's uncertain about.
**Implication:** For purely observational Gaussian data, PC is a strong baseline. CausalBayes should beat PC on NON-LINEAR data (where PC's CI tests fail).

### Finding 3: MC Dropout calibration is poor
**Observation:** Edge probabilities cluster at 0.5 or 1.0, no middle ground. ECE=0.339 (high).
**Root cause:** The weight matrix normalization + sigmoid mapping doesn't produce graded probabilities. The per-variable MLPs learn similar weights for all inputs because all inputs have similar predictive power.
**Fix needed:** Use bootstrap over NOTEARS runs instead of MC dropout on neural weights. Bootstrap naturally produces varied weight matrices, giving meaningful proportions of edge presence.

### Finding 4: KL prior loss too weak
**Observation:** Adding a prior matrix (known edges, known non-edges) changes probability by <0.01.
**Root cause:** The KL divergence between the prior and the sigmoid-transformed weights is tiny compared to the reconstruction loss.
**Fix:** Use L2 penalty on deviation from prior: `lambda_prior * sum(prior * W^2 + (1-prior) * clip(1-W^2, 0))`. Much stronger signal.

### Finding 5: Per-variable MLPs vs linear NOTEARS
**Observation:** On LINEAR data, per-variable MLPs (CausalBayes) and linear NOTEARS have SIMILAR SHD (~3-5 vs ~11). Neither finds a clean DAG.
**Implication:** The MLP complexity doesn't help for linear data. But for NON-LINEAR data, MLPs are necessary. The answer: default to linear NOTEARS for linear-looking data, neural for clearly non-linear.

## Revised Research Direction

**Core problem:** Causal discovery from observational data is fundamentally underdetermined. Outputting a single DAG is misleading.
**CausalBayes value:** Output a DISTRIBUTION over edges (not a DAG) with calibrated uncertainty.
**How to get calibrated uncertainty:** Bootstrap over NOTEARS runs (simple, proven) > MC Dropout on neural weights (complex, uncalibrated).

### Updated Architecture Plan

1. **Core**: Linear NOTEARS with bootstrapped ensembles (not per-variable MLPs)
2. **Uncertainty**: Bootstrap proportion → edge probability. Std of bootstrap weights → edge uncertainty.
3. **Priors**: L2 penalty on weight deviation from prior (not KL). Stronger, simpler.
4. **Non-linear**: Add neural version as optional (default to linear for speed/robustness)
5. **LLM priors**: Same prior injection, simpler loss.

## 2026-05-25 (Late Night Session): Core Algorithm Improvements

### Decision: Fast SciPy L-BFGS-B NOTEARS as Default

**Basis:** Official NOTEARS implementation uses SciPy L-BFGS-B with doubled variables for L1. This is 10-100x faster than PyTorch Adam on arm64 CPU (0.7s vs 17s per run for d=5). Makes bootstrapping practical even with 50-100 samples.

**Implementation:** Refactored `BootstrapDAG` to use `notears_lbfgs()` from `notears_fast.py` instead of the slow PyTorch-based `notears_linear()`. Each bootstrap run now completes in ~0.7-1.5s.

### Decision: L2 Prior Penalty (replacing KL)

**Hypothesis:** KL divergence between prior and sigmoid-transformed weights is too weak compared to reconstruction loss. L2 penalty on weight deviation from prior target should give stronger prior signal.

**Implementation:**
- Both `notears_lbfgs` and `notears_adam` accept `prior_matrix` (d,d) and `lambda_prior` (float)
- Loss = recon + L1 + L2_prior + acyclicity, where L2_prior = lambda_prior * sum(prior * W²)
- For edges with prior=0 (unlikely), this penalizes |W|² directly
- For edges with prior=0.9 (likely), the penalty is weaker

**Status:** Implemented. Needs empirical validation on known priors.

### Decision: Platt Scaling for Probability Calibration

**Hypothesis:** Bootstrap proportions are systematically overconfident (edges at 0 or 1). Platt scaling (logistic regression on logit-transformed proportions) can map raw proportions to better-calibrated probabilities.

**Implementation:**
- `calibrate_bootstrap_proportions(P_raw, W_binary_val)` fits a logistic regression on val data
- Uses logit(P) = ln(P/(1-P)) as input feature
- Returns calibrated probabilities P_cal along with Platt parameters (a,b)

**Preliminary test:** On random data, ECE went from ~0.30 to ~0.03 after Platt calibration.

### Decision: gCastle Baseline Integration

**Status:** gCastle v1.0.4 installed (works on arm64). Available baselines:
- PC, GES, Notears (linear), NotearsNonlinear, GraNDAG, DirectLiNGAM
- Will use these for proper comparisons in benchmarks

### Running Benchmarks (Autonomous Session — 2026-05-25, 22:55-23:15 UTC)

### Hardware Note
Arm64 CPU (Radxa Dragon Q6A, Qualcomm QCM6490).
- `notears_lbfgs` (SciPy L-BFGS-B): ~8-19s per call (slow due to unoptimized BLAS on arm64)
- `notears_fast` (PyTorch Adam with power-series expm): ~3-5s per call
- Cached `.pyc` from earlier version gave artificially fast timings; results rerun with actual code

### Task 1: Comprehensive Multi-Seed Benchmark (d=5 linear)

**Script:** `scripts/benchmark_comprehensive.py`

**Configuration:**
- d=5, n=1000, Erdos-Renyi p=0.2, linear Gaussian SEM, noise_scale=0.1
- 5 seeds: 42, 43, 44, 45, 46
- 60/20/20 train/val/test split
- Methods: Bootstrap(30), Single NOTEARS, Random
- Threshold calibrated on validation data

**Results (mean ± std, 5 seeds):**

| Method | SHD | Precision | Recall | F1 | AUC-PR | ECE | Brier | Time (s) |
|--------|-----|-----------|--------|-----|--------|-----|-------|----------|
| **Bootstrap(30)** | 1.80±1.63 | 0.51±0.42 | 0.47±0.34 | 0.46±0.35 | 0.60±0.33 | 0.1455±0.0947 | 0.176±0.155 | 6.0±1.9 |
| **Single NOTEARS** | 2.20±1.81 | 0.44±0.31 | 0.49±0.28 | 0.46±0.30 | 0.51±0.26 | 0.0944±0.0969 | 0.220±0.181 | 0.2±0.1 |
| **Random** | 4.70±1.12 | 0.16±0.05 | 0.59±0.27 | 0.25±0.09 | 0.21±0.15 | 0.4130±0.0698 | 0.334±0.068 | 0.0±0.0 |

**Hypothesis tested:** Bootstrap uncertainty should improve calibration over single NOTEARS.

**Conclusions:**
- ✅ Bootstrap(30) outperforms Single NOTEARS on SHD (1.80 vs 2.20) and AUC-PR (0.60 vs 0.51)
- ❌ Bootstrap ECE (0.1455) exceeds the target 0.1 and is worse than Single NOTEARS (0.0944)
- High variability across seeds (large std) indicates performance depends heavily on DAG structure
- Random baseline confirms methods learn meaningful structure (SHD 1.80 vs 4.70)
- **Key insight:** Bootstrap improves structure recovery but degrades calibration compared to binary outputs (which are by definition already at extreme prob values)

### Task 2: Calibration Experiment

**Configuration:**
- d=5 linear, seed=42, Bootstrap(10) using `notears_fast` (PyTorch Adam)
- Methods: Raw bootstrap proportions, Platt scaling, Isotonic Regression

**Results:** All methods showed ECE≈0.0000 — measurement artifact due to insufficient probability granularity (10 bootstraps produce only 0.1 increments).

**Conclusion:**
- 10 bootstraps with low noise produce nearly identical weight matrices → all probabilities at 0 or 1
- Need 30+ bootstraps with subsampling or regularization perturbation for diverse matrices
- Codebase has built-in `calibrate_bootstrap_proportions()` in `notears_fast.py` — approach is correct but needs diverse samples

### Task 3: Non-linear Benchmark

**Configuration:**
- d=6 chain DAG, sin/cos/tanh additive noise, n=800
- Methods: Single Linear NOTEARS, Bootstrap(5) Linear NOTEARS (2 seeds)

**Results:**
| Seed | Method | SHD | Precision | Recall |
|------|--------|-----|-----------|--------|
| 42 | Single NOTEARS | 3.0 | 0.00 | 0.00 |
| 42 | Bootstrap(5) | 12.5 | 0.17 | 1.00 |
| 43 | Single NOTEARS | 3.5 | 0.33 | 0.40 |
| 43 | Bootstrap(5) | 12.5 | 0.17 | 1.00 |

**Hypothesis:** Linear NOTEARS should fail on non-linear data.

**Conclusions:**
- ✅ Single linear NOTEARS performs poorly — confirms linear model fails for non-linear SEMs
- ❌ Bootstrap(5) performs WORSE (SHD=12.5) — bootstrapping amplifies instability
- Future: Need Neural NOTEARS (per-variable MLPs) for non-linear data

### Task 4: L2 Prior Experiment

**Configuration:**
- d=5 linear, single NOTEARS, 2 seeds (2 and 5 true edges)
- L2 prior: λ * sum(prior[i,j] * (|W| - mu)²), mu=0.3 for high-prior edges
- Conditions: No prior, Correct prior (P=0.9 on true edges), Misleading prior (P=0.9 on non-edges)

**Results (SHD):**
| Condition | λ=0.05 | λ=0.10 | λ=0.50 |
|-----------|--------|--------|--------|
| No Prior | SHD=1.0/4.0 | SHD=1.0/4.0 | SHD=1.0/4.0 |
| Correct Prior | SHD=1.0/4.5 | SHD=1.0/4.5 | SHD=1.5/4.5 |
| Misleading | **SHD=1.0/5.0** | **SHD=2.0/5.5** | **SHD=5.0/5.0** |

**Hypothesis:** L2 priors with correct domain knowledge should improve accuracy.

**Conclusions:**
- ❌ Correct priors do NOT improve SHD at any λ tested
- ✅ Misleading priors consistently hurt accuracy (as expected)
- Data signal (n=1000, noise=0.1) dominates the prior; prior needs stronger influence
- Future: Use larger λ or bootstrapped ensemble for L2 prior injection

### Summary

| Finding | Status |
|---------|--------|
| Bootstrap(30) outperforms Single NOTEARS (SHD 1.80 vs 2.20) | ✅ |
| ECE target < 0.1 not met (Bootstrap: 0.1455) | ❌ |
| 10 bootstraps insufficient for calibration granularity | ❌ |
| Linear NOTEARS fails on non-linear data (as expected) | ✅ |
| Correct L2 priors don't improve accuracy at low λ | ❌ |
| Misleading L2 priors consistently hurt accuracy | ✅ |

---

## Next Steps

1. Increase bootstrap count to 50+ with bootstrapping variance for diverse weight matrices
2. Implement proper Neural NOTEARS (per-variable MLPs) for non-linear data
3. Test L2 priors with larger λ and bootstrapped ensemble context
4. LLM prior integration: use LLM output as prior matrix
5. Paper-ready benchmark with PC, GES, and gCastle baselines
