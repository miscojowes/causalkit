"""
Research conclusions from May 25 2026 benchmarks.

## What we learned

### 1. The central challenge: observational equivalence
From purely observational data, the true DAG is not identifiable — we can only
recover the Markov equivalence class (CPDAG). NOTEARS forces a single DAG,
which is misleading when many DAGs fit equally well.

### 2. What matters is uncertainty
Instead of finding "the" DAG, output a DISTRIBUTION over edges.
The user needs to know: "this edge has 85% probability" not "this edge exists."

### 3. Bootstrap > MC Dropout for uncertainty
- MC Dropout on per-variable MLPs produces uncalibrated probabilities (ECE=0.34)
- Bootstrap over NOTEARS runs naturally gives varied weight matrices
- Edge probability = proportion of bootstraps where edge exceeds threshold
- Simpler, theoretically grounded, better calibrated

### 4. Priors need stronger signal
KL divergence prior loss is too weak. Use L2 deviation penalty instead.

### 5. Architecture should match the problem
- Linear data → bootstrapped linear NOTEARS (simple, fast)
- Non-linear data → bootstrapped neural NOTEARS (when needed)
- Uncertainty always → bootstrap proportions

"""
