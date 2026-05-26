# causalkit 🦀

**Practical causal discovery + causal ML in one pip install.**

> Give it your data + a sentence about your domain, it gives you the causal graph and lets you answer causal questions.

## Quickstart

```bash
pip install causalkit
```

```python
import causalkit as ck

model = ck.CausalDiscoverer()
model.fit(X)

print(model.causal_matrix_)       # weighted DAG
print(model.edge_confidence_)     # bootstrap confidence per edge

# With domain knowledge
model.fit(X, domain_text="Ad spend drives sales. Price affects demand.")
```

## Docs

Full documentation: [`causalkit/README.md`](causalkit/README.md)

## Benchmark results

| Dataset | GES F1 | CK+P F1 | Δ |
|---------|:------:|:-------:|:-:|
| Sachs (real) | 0.516 | 0.629 | **+0.113** |
| Cancer | 0.000 | 0.889 | **+0.889** |
| Earthquake | 0.000 | 0.600 | **+0.600** |
| Survey | 0.182 | 0.923 | **+0.741** |
| Asia | 0.000 | 0.300 | **+0.300** |

**CK+P beats GES on 5/5 datasets.** See [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) for full details.

## Repo structure

```
causalkit/          ← Public API (the library)
├── discoverer.py     CausalDiscoverer
├── prior.py          Text → prior matrix
├── adaptive_trust.py Per-edge λ (experimental)
├── effects.py        ATE, what-if
└── README.md         Library docs
src/causbayes/      ← Engine (NOTEARS, bootstrap)
scripts/            ← Benchmarks
experiment_results/ ← Data
EXPERIMENT_LOG.md   ← Experiment log
```

## License

MIT
