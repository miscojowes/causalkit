# CausalBayes

**Bayesian Causal Discovery with Neural Structure Learning & LLM-Informed Priors**

A Python library for uncertainty-aware causal structure learning that outputs *distributions over DAGs*, not single point estimates. Combines gradient-based neural structure learning with Bayesian uncertainty quantification and LLM-extracted domain knowledge.

## Motivation

Existing tools make a critical assumption — that causal structure is a single, known graph:

| Tool | Limit |
|------|-------|
| **gCastle** (Huawei) | Gradient-based, no uncertainty |
| **CausalNex** (McKinsey) | Semi-abandoned, only BNs |
| **IBCD** (2025) | Bayesian, but requires interventional data |
| **LLM+Causal** (2025) | LLM used as decision-maker (flawed) |

**CausalBayes** fills the gap: neural structure learning with principled uncertainty + LLM as prior informant (not decision-maker).

## Key Innovation

```
Input Data ──► Neural NOTEARS ──► Bayesian Edge Distribution ──► Probabilistic DAG
                    ▲                        ▲
                    │                        │
         LLM Domain Knowledge          MC Dropout / VI
         (soft priors only)
```

- **Neural NOTEARS**: Non-linear DAG learning via neural networks with acyclicity constraint
- **Bayesian Edge Uncertainty**: Each edge gets a posterior inclusion probability, not binary
- **LLM Priors**: LLM extracts domain knowledge as *soft priors* (not hard constraints), following the principle that LLMs should inform, not decide (Wu et al., 2025)
- **Output**: Full distribution over graphs with edge-level uncertainty quantification

## Installation

```bash
pip install causbayes
```

Requires Python 3.10+, PyTorch.

## Quick Start

### Basic Causal Discovery with Uncertainty

```python
from causbayes import NeuralBayesianDAG
import numpy as np

# Generate synthetic data (n_samples, n_vars)
X = np.random.randn(500, 10)

# Learn with uncertainty
model = NeuralBayesianDAG(
    hidden_layers=[64, 64],
    learning_rate=1e-3,
    lambda_1=0.01,       # L1 regularization
    lambda_2=5.0,        # Acyclicity penalty
    uncertainty="mc_dropout",  # or "variational"
)

# Returns edge probability matrix P[i,j] = P(X_i -> X_j)
edge_probs, edge_stds = model.fit_transform(X)

# Visualize with uncertainty
model.plot(threshold=0.5, show_uncertainty=True)
```

### With LLM Domain Knowledge

```python
from causbayes import LLMInformedPrior

# Describe your domain
domain_text = """
In genomics, transcription factors regulate gene expression.
TP53 is a tumor suppressor that activates DNA repair genes.
"""

# Extract soft priors (not hard constraints)
prior = LLMInformedPrior(api_key="...")
prior_matrix = prior.extract_edge_priors(
    variables=["TP53", "MDM2", "CDKN1A", "BAX"],
    domain_description=domain_text,
    confidence="low"  # conservative: only confident edges
)

# Learn with informed priors
model = NeuralBayesianDAG(prior_matrix=prior_matrix)
edge_probs = model.fit_transform(X)
```

## Architecture

```
causbayes/
├── structure_learning/       # Gradient-based DAG learning
│   ├── neural_notears.py     # Neural NOTEARS with MLP
│   ├── base.py               # Abstract structure learner
│   └── utils.py              # DAG constraints, metrics
├── bayesian/                 # Uncertainty quantification
│   ├── variational.py        # VI over graph structure
│   ├── mc_dropout.py         # MC Dropout for epistemic uncertainty
│   └── priors.py             # Edge prior distributions
├── llm_prior/                # LLM-informed domain knowledge
│   ├── extractor.py          # LLM-based knowledge extraction
│   ├── prior_builder.py      # Convert to structured priors
│   └── heuristics.py         # LLM-guided search acceleration
├── models/                   # Neural architectures
│   ├── nonlinear_sem.py      # Non-linear SEM
│   └── bayesian_nn.py        # Bayesian NN modules
├── evaluation/               # Metrics and benchmarking
│   └── metrics.py            # Uncertainty-aware metrics
└── visualization/            # Plotting
    └── plot.py               # Probabilistic DAG plotting
```

## How It Differs

| Feature | gCastle | CausalNex | IBCD | **CausalBayes** |
|---------|---------|-----------|------|-----------------|
| Uncertainty | ❌ | ❌ | ✅ (edges) | ✅ (full graph) |
| Neural Structure | ✅ (basic) | ❌ | ❌ | ✅ (deep) |
| Domain Knowledge | ❌ | ✅ (expert) | ❌ | ✅ (LLM soft priors) |
| Observational Data | ✅ | ✅ | ❌ | ✅ |
| Output | Single DAG | Single DAG | Edge PIPs | Distribution over DAGs |
| LLM Integration | ❌ | ❌ | ❌ | ✅ (non-decisional) |

## Citation

If you use CausalBayes, please cite:

```bibtex
@software{causbayes2026,
  title = {CausalBayes: Bayesian Causal Discovery with Neural Structure Learning and LLM-Informed Priors},
  year = {2026}
}
```

## License

MIT
