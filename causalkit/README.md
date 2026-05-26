# causalkit 🦀

**Practical causal discovery + causal ML in one `pip install`.**

```
pip install causalkit
```

```python
import causalkit as ck

# 🔍 Discover causal structure from data
model = ck.CausalDiscoverer()
model.fit(X)

print(model.causal_matrix_)     # weighted adjacency matrix
print(model.edge_confidence_)   # bootstrap probabilities

# 💡 Inject domain knowledge (text or matrix)
model.fit(X, domain_text="Ad spend causes sales. Price affects demand.")

# 📊 Answer causal questions
ate = model.estimate_ate(X, treatment="ad_spend", outcome="sales")
# → "Increasing ad spend by $10k → +$47k sales"

pred = model.counterfactual_predict(X, {"ad_spend": 50000})
# → "If we set ad_spend to $50k, sales would be..."
```

---

## Why causalkit?

| Library | Domain Knowledge | Uncertainty | Causal ML | Simple API |
|---------|:----------------:|:-----------:|:---------:|:----------:|
| **gCastle** | ❌ | ❌ | ❌ | ❌ |
| **CausalNex** | ⚠️ expert only | ❌ | ✅ | ❌ |
| **DoWhy** | ✅ | ❌ | ✅ | ❌ |
| **PRCD-MAP** | ✅ | ❌ | ❌ | ❌ |
| **causalkit** | ✅ text+matrix | ✅ bootstrap | ✅ ATE+what-if | ✅ |

**causalkit is the only library that:**
1. Takes **domain knowledge as text** ("X causes Y") via LLM or heuristics
2. Gives **confidence scores** on every edge (bootstrap)
3. Answers **causal questions** (ATE, what-if, root cause analysis)
4. Has a **dead-simple scikit-learn API**

---

## Quickstart

### 1. Discover a causal graph

```python
import causalkit as ck
import pandas as pd

# Load your data
X = pd.read_csv("my_data.csv")

# Fast single run
model = ck.CausalDiscoverer(method="notears")
model.fit(X)
print(model.causal_matrix_)

# Better: bootstrap with uncertainty
model = ck.CausalDiscoverer(method="bootstrap", n_bootstraps=50)
model.fit(X)
print(model.edge_confidence_)   # [0,1] for each edge
```

### 2. Add domain knowledge

```python
# Option A: Natural language
model.fit(X, domain_text="""
    Temperature affects energy consumption.
    Time of day affects temperature and energy use.
    Occupancy affects energy consumption.
""")

# Option B: Prior matrix
import numpy as np
prior = np.full((d, d), 0.5)
np.fill_diagonal(prior, 0.0)
prior[0, 1] = 0.9  # 90% sure x0 → x1

model.fit(X, prior_matrix=prior)

# Option C: Edge list
from causalkit.prior import edge_list_to_prior
prior = edge_list_to_prior([
    ("temperature", "energy"),
    ("occupancy", "energy"),
], feature_names=model.feature_names_)
model.fit(X, prior_matrix=prior)
```

### 3. Answer causal questions

```python
# Average Treatment Effect
ate = model.estimate_ate(
    X, treatment="ad_spend", outcome="sales"
)
print(f"ATE: ${ate:.2f} per $1 spent")

# What-if (counterfactual)
pred = model.counterfactual_predict(
    X, interventions={"ad_spend": 50000}
)

# Root cause analysis
causes = model.root_cause_analysis(X, target="sales", top_k=5)
for var, strength in causes:
    print(f"  {var}: effect = {strength:.3f}")

# Visualize
model.plot("causal_graph.png")
```

---

## API Reference

### `CausalDiscoverer`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | `'bootstrap'` | Algorithm: `'notears'` (fast, single run), `'bootstrap'` (uncertainty), `'dagma'` (PyTorch) |
| `n_bootstraps` | `50` | Bootstrap samples (only for `method='bootstrap'`) |
| `lambda_1` | `0.005` | L1 sparsity penalty |
| `lambda_prior` | `0.5` | Base prior strength |
| `adaptive_trust` | `False` | Experimental: per-edge λ (PRCD-MAP style) |
| `threshold` | `None` | Edge threshold (auto-tuned if None) |
| `random_state` | `42` | Seed for reproducibility |
| `verbose` | `True` | Print progress |

### Main methods

```python
fit(X, prior_matrix=None, domain_text=None, feature_names=None)
estimate_ate(X, treatment, outcome, method='linear')
counterfactual_predict(X, interventions)
root_cause_analysis(X, target, top_k=5)
plot(filename=None, max_vars=30)
```

### Attributes

```python
causal_matrix_        # (d, d) weighted adjacency matrix
edge_confidence_      # (d, d) bootstrap probabilities [0, 1]
feature_names_        # Variable names
```

---

## How it works

1. **NOTEARS** — Continuous optimization for DAG structure learning (Zheng et al., 2018)
2. **Bootstrap** — Resample data B times, run NOTEARS each time → edge probability
3. **Prior integration** — L2 penalty: `λ · Σ (1 - prior) · W²` pushes W toward 0 where prior says no edge
4. **Adaptive trust** *(experimental)* — Simplified PRCD-MAP: 2-pass bootstrap to estimate per-edge λ
5. **Causal ML** — Linear SEM with back-door adjustment for ATE

---

## Algorithm comparison

| Method | Clean prior (70%) | No prior | Mixed prior (50% noise) | Speed |
|--------|:-----------------:|:--------:|:-----------------------:|:-----:|
| gCastle GES | — | F1=0.516 | — | ⚡ 0.1s |
| causalkit (no prior) | F1=0.211 | F1=0.211 | F1=0.211 | 🐢 8s |
| causalkit (uniform λ) | **F1=0.500** | — | **F1=0.268** | 🐢 8s |
| causalkit (adaptive) | F1=0.484 | — | F1=0.235 | 🐢🐢 22s |

*Measured on Sachs dataset (11 vars, 853 samples), 30 bootstraps, 60s timeout.*

---

## Project structure

```
causalkit/
├── __init__.py          # Package init, version
├── discoverer.py        # Main CausalDiscoverer class
├── adaptive_trust.py    # Per-edge λ computation (experimental)
├── prior.py             # Prior extraction (text → matrix)
├── effects.py           # ATE, what-if, root cause analysis
├── metrics.py           # Evaluation (SHD, F1, etc.)
```

Using engine from **causal-bayes** (NOTEARS, DAGMA, bootstrap) at `src/causbayes/`.

---

## Development

```bash
git clone https://github.com/yourorg/causalkit
cd causalkit
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Run tests
python scripts/test_causalkit.py
```

---

## Roadmap

- [x] Core structure discovery (NOTEARS, Bootstrap)
- [x] Prior integration (matrix and text-based)
- [x] Adaptive trust (experimental, simplified PRCD-MAP)
- [x] Causal effects (ATE, what-if, root cause)
- [ ] PyPI packaging
- [ ] Documentation site
- [ ] Non-linear causal effects (DR-Learner, CausalForest)
- [ ] Time series (SVAR, PCMCI+)
- [ ] vLLM integration for LLM priors

---

## License

MIT

## Citation

```
@software{causalkit,
  author = {Joel and Claudia},
  title = {causalkit: Practical Causal Discovery and Causal ML},
  year = {2026},
}
```
