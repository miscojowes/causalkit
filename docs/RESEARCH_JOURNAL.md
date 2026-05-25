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

## Next Steps After Feedback

1. Implement bootstrap-based NOTEARS with fixed calibration
2. Fix prior injection with L2 loss
3. Benchmark: CausalBayes (bootstrap) vs PC vs GES on non-linear data
4. Show calibrated uncertainty (ECE < 0.1 goal)
