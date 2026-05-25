# CausalBayes Architecture & Design

## Overview

**CausalBayes** is a Python library for **uncertainty-aware Bayesian causal discovery**. It combines:

1. **Neural structure learning** via gradient-based DAG optimization (NOTEARS)
2. **Bayesian uncertainty quantification** over the learned DAG structure
3. **LLM-informed domain knowledge** as soft priors (not hard constraints)

## Core Innovation

Existing causal discovery tools output a single DAG. CausalBayes outputs a **distribution over DAGs** — each edge has a posterior inclusion probability, confidence interval, and uncertainty estimate.

### Why This Matters

| Application | Why Uncertainty Matters |
|-------------|------------------------|
| Healthcare | A doctor needs to know "this edge has 85% probability" not "edge exists" |
| Policy | "Causal effect uncertain" is actionable; "causal effect estimated" is misleading |
| Science | High-uncertainty edges tell researchers where to experiment next |
| Industry | Downstream decisions need calibrated confidence, not binary suggestions |

## Architecture

```
                        ┌──────────────────────┐
                        │   User Data (X, d)    │
                        └──────────┬───────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │  1. Neural NOTEARS        │
                    │  ┌────────────────────┐   │
                    │  │  Per-variable MLPs │   │
                    │  │  + Acyclicity      │   │
                    │  │  Augmented Lagr.   │   │
                    │  └────────┬───────────┘   │
                    │           │               │
                    │  ┌────────▼───────────┐   │
                    │  │  Weight Matrix W   │   │
                    │  │  W[i,j] = ∥∂fᵢ/∂xj∥│   │
                    │  └────────────────────┘   │
                    └──────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
                    ▼              ▼              ▼
            ┌────────────┐ ┌────────────┐ ┌──────────────┐
            │MC Dropout  │ │Variational │ │ 3. LLM Prior │
            │Uncertainty │ │ Inference  │ │  Injection   │
            │(epistemic) │ │(structural)│ │  (soft only) │
            └──────┬─────┘ └──────┬─────┘ └──────┬───────┘
                   │              │              │
                   └──────────────┴──────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │  Posterior over DAGs      │
                    │  P(W|X) ≈ probabilistic   │
                    │  adjacency matrix         │
                    └──────────────────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │  4. Outputs              │
                    │  • Edge probabilities    │
                    │  • Uncertainty intervals │
                    │  • DAG samples           │
                    │  • Experiment suggestions│
                    └──────────────────────────┘
```

## Module Design

### 1. Structure Learning (`structure_learning/`)

**`neural_notears.py`** — Core algorithm
- `NeuralBayesianDAG`: Main entry point
- Implements augmented Lagrangian optimization for h(W)=0
- Gradient clipping, progressive inner iterations, batch training

**`utils.py`** — DAG utilities
- `dagness(W)`: Acyclicity penalty with numerical stability
- `is_dag(W)`: DAG detection
- `structural_hamming_distance`, `expected_shd`: Metrics
- Graph conversion to NetworkX

### 2. Neural Models (`models/`)

**`nonlinear_sem.py`** — Structural Equation Models
- `NonlinearSEM`: Per-variable MLPs with separate first-layer encoders
- `compute_weight_matrix()`: L2 norms of first-layer weights → W[i,j]
- Small weight initialization for optimization stability

### 3. Bayesian Inference (`bayesian/`)

**`mc_dropout.py`** — MC Dropout uncertainty
- Stochastically enables dropout at inference time
- Generates distribution over weight matrices
- Edge probability from coefficient of variation

**`variational.py`** — Variational inference over graph
- Mean-field approximation
- KL regularization for posterior compression

**`priors.py`** — Prior distributions
- Spike-and-slab: pi * N(0, σ²) + (1-pi) * delta(0)
- Horseshoe: Cauchy(0, τ) heavy-tailed shrinkage
- `build_edge_prior_matrix()`: From expert knowledge
- `prior_from_associations()`: From association data

### 4. LLM Prior Module (`llm_prior/`)

**`__init__.py`** — `LLMPriorExtractor` class
- Extracts causal relationships from domain text via LLM API
- **Key constraint**: Outputs soft priors only, not hard edge decisions
- Follows Wu et al. (2025): LLMs should NOT determine causal edges
- `suggest_experiments()`: Proposes interventions to resolve uncertainty

**`prior_builder.py`** — Prior construction utilities
- `build_prior_from_llm_response()`: LLM → prior matrix
- `fuse_priors()`: Combine multiple prior sources

**`heuristics.py`** — LLM-guided search
- `LLMHeuristicSearch`: Uses LLM to propose parent sets for score evaluation
- Non-decisional: LLM suggests *what to evaluate*, not *what is true*

### 5. Evaluation (`evaluation/`)

- `comprehensive_evaluation()`: All metrics in one call
- `edge_calibration()`: Predicted prob vs actual frequency
- `uncertainty_coverage()`: CI coverage check
- `compare_with_baseline()`: Benchmark vs gCastle etc.

### 6. Visualization (`visualization/`)

- `plot_probabilistic_dag()`: DAG with uncertainty (color = certainty, width = probability)
- `plot_uncertainty_calibration()`: Calibration curve

## How It Differs From Existing Work

### vs gCastle (Huawei)
- gCastle has 10+ algorithms but no uncertainty
- CausalBayes: Same gradient-based foundation + uncertainty layer
- CausalBayes: Outputs distributions, not point estimates

### vs CausalNex (McKinsey)
- CausalNex only handles Bayesian networks, semi-abandoned
- CausalBayes handles non-linear SEMs, actively maintained

### vs IBCD (Han et al., 2025)
- IBCD requires interventional data (CRISPR perturbations)
- CausalBayes works with observational data
- CausalBayes uses neural networks (not spike-and-slab linear)

### vs LLM+Causal Methods
- Most LLM methods naively ask "what causes what"
- CausalBayes: LLM informs priors, doesn't decide edges
- CausalBayes: Follows critical findings (Wu et al., 2025)

## Limitations & Future Work

### Current Limitations
1. **Scalability**: Neural NOTEARS is O(d²) per iteration; ~50 vars max on CPU
2. **Non-convexity**: Neural optimization may find local optima
3. **Prior calibration**: LLM prior strength parameter needs tuning
4. **MC Dropout approximation**: Not fully Bayesian on weights

### Future Directions
1. **Scaling**: Sparse attention, sub-sampling for large d >> 100
2. **Full VI**: Bayes by Backprop over all weights
3. **Normalizing flows**: Score-based generative models over DAGs
4. **Online learning**: Streaming causal discovery
5. **Multi-modal priors**: Images, time series, natural language
6. **Interventional data integration**: Perturb-seq style experiments

## References

1. Zheng et al. (2018). "DAGs with NO TEARS: Continuous Optimization for Structure Learning." NeurIPS.
2. Lachapelle et al. (2020). "Gradient-based Neural DAG Learning." ICLR.
3. Gal & Ghahramani (2016). "Dropout as a Bayesian Approximation." ICML.
4. Wu et al. (2025). "LLM Cannot Discover Causality." arXiv:2506.00844.
5. Han et al. (2025). "Large-Scale Bayesian Causal Discovery with Interventional Data." arXiv:2510.01562.
6. Bello et al. (2025). "LLM-Driven Causal Discovery via Harmonized Prior." IEEE TKDE.
7. Ma et al. (2026). "Foundation Models for Causal Inference via Prior-Data Fitted Networks." ICLR.
