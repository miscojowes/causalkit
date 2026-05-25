#!/usr/bin/env python3
"""
Structural Uncertainty via CPDAG Analysis.

Key insight: Not all edges are equally uncertain. In a Markov equivalence class:
- Compelled edges MUST be oriented this way in every DAG in the MEC
- Reversible edges CAN be oriented either way — THIS is structural uncertainty

If we can compute the CPDAG from a NOTEARS DAG, we get:
- Which edges are structurally uncertain (reversible)
- Which edges are structurally certain (compelled)
- No bootstrapping needed!

Implementation: Use gCastle's CPDAG computation from a DAG.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from sklearn.preprocessing import StandardScaler


def generate_dag(d, n, seed=42):
    rng = np.random.RandomState(seed)
    W_true = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < 0.2:
                W_true[i, j] = rng.uniform(0.5, 1.5) * rng.choice([-1, 1])
    X = np.zeros((n, d))
    for j in range(d):
        parents = np.where(W_true[:, j] != 0)[0]
        if len(parents) > 0:
            X[:, j] = X[:, parents] @ W_true[parents, j]
        X[:, j] += rng.randn(n) * 0.1
    return X, (np.abs(W_true) > 1e-6).astype(float)


def dag_to_cpdag(W_dag):
    """
    Convert a DAG to its CPDAG (completed partially directed acyclic graph).
    
    Uses gCastle's DAG to CPDAG conversion.
    Returns: CPDAG matrix where 1 = directed edge, -1 = undirected (reversible)
    """
    try:
        from castle.common import GraphDAG
        from castle.common.pri_knowledge import PriKnowledge
        
        # Convert to gCastle format
        dag_castle = W_dag.copy()
        cpdag = GraphDAG.dag2cpdag(dag_castle)
        
        # CPDAG: 1 = directed, 2 = undirected (reversible)
        # Convert to: 1 = directed, 0 = undirected (reversible), 0 = no edge
        # Actually, let me just work with what we get
        return cpdag
    except Exception as e:
        print(f"  gCastle CPDAG failed: {e}")
        return None


def analyze_cpdag(W_dag, W_true=None):
    """Compute CPDAG and analyze structural uncertainty."""
    cpdag = dag_to_cpdag(W_dag)
    if cpdag is None:
        return None
    
    d = W_dag.shape[0]
    
    # gCastle CPDAG encoding: let me check what it returns
    # From castle docs: dag2cpdag returns a matrix where
    # 1 = directed edge (i→j), -1 = directed edge (i←j)
    # or maybe 1 = compelled, -1 = reversible?
    
    print(f"  CPDAG matrix:")
    print(f"  {np.array_str(cpdag, precision=0, suppress_small=True)}")
    
    # Let's try a different approach: use PC to get the CPDAG directly
    return cpdag


def get_cpdag_via_pc(X, alpha=0.05):
    """Get CPDAG directly from PC algorithm (it outputs CPDAG by default)."""
    try:
        from castle.algorithms import PC
        pc = PC(alpha=alpha, ci_test='fisherz')
        pc.learn(X)
        W_pc = pc.causal_matrix
        
        # PC returns: 0 = no edge, 1 = directed, 2 = undirected (reversible)
        print(f"  PC CPDAG matrix:")
        print(f"  {np.array_str(W_pc, precision=0, suppress_small=True)}")
        
        # Count reversible edges (value 2)
        n_compelled = np.sum(W_pc == 1)
        n_reversible = np.sum(np.abs(W_pc) == 2)
        n_total = n_compelled + n_reversible
        
        print(f"  Compelled edges: {n_compelled}")
        print(f"  Reversible edges (structurally uncertain): {n_reversible}")
        print(f"  Structural uncertainty ratio: {n_reversible / max(n_total, 1):.2%}")
        
        return W_pc
    except Exception as e:
        print(f"  PC failed: {e}")
        return None


def main():
    print("=" * 70)
    print("  STRUCTURAL UNCERTAINTY via CPDAG Analysis")
    print("=" * 70)
    
    print("\n  Hypothesis: Reversible edges in the CPDAG are the TRUE")
    print("  source of structural uncertainty. Bootstrap should assign")
    print("  high probability to compelled edges and low to reversible.")
    
    for seed in [42, 43, 44]:
        print(f"\n{'─'*70}")
        print(f"  Seed {seed}")
        d, n = 5, 1000
        X, W_true = generate_dag(d, n, seed=seed)
        X_scaled = StandardScaler().fit_transform(X)
        
        ne = int(np.sum(W_true > 0))
        print(f"  True edges: {ne}")
        
        # 1. Get CPDAG via PC
        print(f"\n  [A] CPDAG from PC algorithm:")
        cpdag = get_cpdag_via_pc(X_scaled)
        
        if cpdag is not None:
            # 2. Run NOTEARS to get a DAG
            from causbayes.structure_learning.notears_fast import notears_lbfgs
            
            print(f"\n  [B] NOTEARS DAG:")
            W_nt = notears_lbfgs(X_scaled, max_iter=5, w_threshold=0.05, lbfgs_maxiter=10)
            W_bin = (np.abs(W_nt) > 1e-4).astype(float)
            shd_nt = float(np.sum(np.abs(W_true - W_bin)) / 2)
            print(f"  NOTEARS edges: {int(np.sum(W_bin))}, SHD={shd_nt:.0f}")
            
            # 3. Compare NOTEARS with CPDAG
            # Edges where NOTEARS makes a decision but CPDAG says reversible
            for i in range(d):
                for j in range(d):
                    if W_bin[i, j] > 0:
                        cpdag_val = cpdag[i, j]
                        note = ""
                        if cpdag_val == 2 or cpdag_val == -2:
                            note = "⚠️ REVERSIBLE (structural uncertainty)"
                        elif cpdag_val == 1 or cpdag_val == -1:
                            note = "✓ compelled edge"
                        is_true = "✓" if W_true[i, j] > 0 else "✗"
                        print(f"    X{i}→X{j} (true={is_true}) CPDAG={cpdag_val} {note}")
    
    # On the data seed 42, also do bootstrap and compare
    print(f"\n{'═'*70}")
    print(f"  KEY ANALYSIS: Bootstrap probabilities vs CPDAG reversibility")
    print(f"{'═'*70}")
    
    print(f"\n  Hypothesis: Bootstrap gives different probabilities to")
    print(f"  compelled vs reversible edges in the CPDAG.")
    
    seed = 42
    d, n = 5, 1000
    X, W_true = generate_dag(d, n, seed=seed)
    X_scaled = StandardScaler().fit_transform(X)
    
    # Bootstrap
    from causbayes.structure_learning.notears_fast import bootstrap_notears
    P, S, Wl, Wa = bootstrap_notears(X_scaled, n_bootstraps=30, max_iter=5, 
                                      w_threshold=0.05, method='lbfgs', seed=seed)
    
    # CPDAG
    cpdag = get_cpdag_via_pc(X_scaled)
    
    if cpdag is not None:
        print(f"\n  {'Edge':<8} {'Bootstrap P':>12} {'True?':>5} {'CPDAG':>6}")
        print(f"  {'─'*8} {'─'*12} {'─'*5} {'─'*6}")
        
        # For each edge, show bootstrap prob, truth, and CPDAG status
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                p = P[i, j]
                if p > 0.1:  # show non-trivial edges
                    is_true = "✓" if W_true[i, j] > 0 else "✗"
                    cp = cpdag[i, j]
                    cp_label = "rev" if (cp == 2 or cp == -2) else ("dir" if (cp == 1 or cp == -1) else "none")
                    print(f"  X{i}→X{j}  {p:>12.3f} {is_true:>5} {cp_label:>6}")
        
        # Summary statistics
        print(f"\n  Summary:")
        compelled_probs = []
        reversible_probs = []
        for i in range(d):
            for j in range(d):
                if i == j: continue
                cp = cpdag[i, j]
                if cp == 1 or cp == -1:
                    compelled_probs.append(P[i, j])
                elif cp == 2 or cp == -2:
                    reversible_probs.append(P[i, j])
        
        if compelled_probs:
            print(f"  Compelled edges: mean P={np.mean(compelled_probs):.3f} ± {np.std(compelled_probs):.3f}")
        if reversible_probs:
            print(f"  Reversible edges: mean P={np.mean(reversible_probs):.3f} ± {np.std(reversible_probs):.3f}")


if __name__ == "__main__":
    main()
