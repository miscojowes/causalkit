"""
CPDAG (Completed Partially Directed Acyclic Graph) conversion.

Given an estimated DAG adjacency matrix, compute its Markov equivalence
class — the CPDAG (also called the essential graph). Two DAGs are
Markov equivalent iff they have the same skeleton and same v-structures.

The CPDAG represents the entire equivalence class:
    - Directed edges (→) are compelled — same direction in every
      DAG in the class
    - Undirected edges (—) are reversible — direction differs
      among DAGs in the class

Algorithm:
    1. Build the skeleton (undirected graph) from the DAG
    2. Identify v-structures (unshielded colliders i→j←k)
    3. Orient compelled edges from v-structures
    4. Apply Meek's rules (R1-R4) to propagate compelled orientations

Reference:
    Chickering, D. M. (2002). "Learning equivalence classes of
    Bayesian-network structures." JMLR, 2, 445-498.

    Meek, C. (1995). "Causal inference and causal explanation with
    background knowledge." UAI, 403-410.
"""

import numpy as np
import networkx as nx
from typing import Tuple


def dag_to_cpdag(W: np.ndarray) -> np.ndarray:
    """Convert a DAG adjacency matrix to its CPDAG (Markov equivalence class).

    The output encodes three edge types:
        0 = no edge
        1 = directed edge i → j
        2 = undirected edge i — j (stored symmetrically in result[i,j]
            and result[j,i])

    Args:
        W: DAG adjacency matrix of shape (d, d).
           W[i,j] != 0 means edge i → j exists.
           Can be binary or weighted (weights are thresholded at 1e-8).

    Returns:
        CPDAG matrix of shape (d, d) with values {0, 1, 2}.

    Example:
        >>> # Simple 3-variable chain: 0 → 1 → 2
        >>> W = np.array([[0, 1, 0],
        ...               [0, 0, 1],
        ...               [0, 0, 0]])
        >>> cpdag = dag_to_cpdag(W)
        >>> # In a chain, all edges are compelled (directed)
        >>> cpdag[0, 1]
        1
        >>> cpdag[1, 2]
        1

        >>> # V-structure: 0 → 1 ← 2
        >>> W = np.array([[0, 1, 0],
        ...               [0, 0, 0],
        ...               [0, 1, 0]])
        >>> cpdag = dag_to_cpdag(W)
        >>> cpdag[0, 1], cpdag[2, 1]  # both directed
        (1, 1)
    """
    d = W.shape[0]

    # Binarize: any non-zero (or above threshold) entry is an edge
    W_bin = (np.abs(W) > 1e-8).astype(np.int8)

    # --- Step 1: Build skeleton (undirected adjacency ignoring direction) ---
    skeleton = ((W_bin + W_bin.T) > 0).astype(np.int8)

    # --- Step 2: Initialize edge types ---
    # We track orientation using two boolean matrices:
    #   directed[i,j] = True  means i → j is compelled/directed
    #   undirected[i,j] = True means i — j is undirected (reversible)
    # These are mutually exclusive for any pair (i,j).
    directed = np.zeros((d, d), dtype=bool)
    undirected = np.zeros((d, d), dtype=bool)

    # Start with all skeleton edges as undirected
    for i in range(d):
        for j in range(d):
            if skeleton[i, j] and i != j:
                undirected[i, j] = True

    # --- Step 3: Identify and orient v-structures ---
    # A v-structure is i → j ← k where i and k are NOT adjacent.
    # These are compelled — we can orient them from the skeleton alone.
    for j in range(d):
        # Find all children of j (j is the child/parent based on direction)
        # Actually: find all i such that i → j (parents of j in the DAG)
        parents_of_j = np.where(W_bin[:, j] > 0)[0]

        for idx_i, i in enumerate(parents_of_j):
            for k in parents_of_j[idx_i + 1:]:
                # Check if i and k are NOT adjacent in skeleton
                if skeleton[i, k] == 0 and skeleton[k, i] == 0:
                    # V-structure! Orient i → j ← k
                    directed[i, j] = True
                    undirected[i, j] = False
                    undirected[j, i] = False

                    directed[k, j] = True
                    undirected[k, j] = False
                    undirected[j, k] = False

    # --- Step 4: Apply Meek's rules ---
    # Repeatedly apply orientation propagation rules until convergence
    _apply_meeks_rules(directed, undirected, skeleton, d)

    # --- Step 5: Build output matrix ---
    result = np.zeros((d, d), dtype=np.int8)
    for i in range(d):
        for j in range(d):
            if directed[i, j]:
                result[i, j] = 1
            elif undirected[i, j]:
                result[i, j] = 2

    return result


def _apply_meeks_rules(
    directed: np.ndarray,
    undirected: np.ndarray,
    skeleton: np.ndarray,
    d: int,
) -> None:
    """Apply Meek's four orientation rules until no more edges can be oriented.

    Modifies ``directed`` and ``undirected`` in-place.

    Reference:
        Meek (1995); Chickering (2002); Spirtes, Glymour, Scheines (2000).
    """
    changed = True
    iteration = 0
    max_iterations = d * d * 4  # Safety limit

    while changed and iteration < max_iterations:
        changed = False
        iteration += 1

        # --- R1: i → j, j — k, i not adj to k  ⇒  j → k ---
        for i in range(d):
            for j in range(d):
                if not directed[i, j]:
                    continue
                # i → j exists
                for k in range(d):
                    if k == i or k == j:
                        continue
                    if not undirected[j, k]:
                        continue
                    # j — k exists, check i not adj to k
                    if skeleton[i, k] == 0:
                        # Orient j → k
                        directed[j, k] = True
                        undirected[j, k] = False
                        undirected[k, j] = False
                        changed = True

        # --- R2: i → j, i → k, j — k  ⇒  orient j → k ---
        # Prevents creating a new v-structure i → j ← k:
        # since i → k exists (i and k are adjacent), the v-structure
        # would be shielded and not identifiable from the equivalence
        # class. We must orient j → k to avoid this.
        for i in range(d):
            for j in range(d):
                if not directed[i, j]:
                    continue
                for k in range(d):
                    if k == i or k == j:
                        continue
                    if not directed[i, k]:
                        continue
                    if not undirected[j, k]:
                        continue
                    # i → j, i → k, j — k  ⇒  orient j → k
                    directed[j, k] = True
                    undirected[j, k] = False
                    undirected[k, j] = False
                    changed = True

        # --- R3: i — j, i → k, j → k  ⇒  i → j ---
        for i in range(d):
            for k in range(d):
                if not directed[i, k]:
                    continue
                for j in range(d):
                    if j == i or j == k:
                        continue
                    if not directed[j, k]:
                        continue
                    if not undirected[i, j]:
                        continue
                    # i → k, j → k, i — j  ⇒  orient i → j
                    directed[i, j] = True
                    undirected[i, j] = False
                    undirected[j, i] = False
                    changed = True

        # --- R4: i → k, j → k, i — j, directed path i → ... → j  ⇒  i → j ---
        # (This prevents cycles by resolving the direction of i-j when
        #  there's already a directed path one way.)
        for i in range(d):
            for k in range(d):
                if not directed[i, k]:
                    continue
                for j in range(d):
                    if j == i or j == k:
                        continue
                    if not directed[j, k]:
                        continue
                    if not undirected[i, j]:
                        continue

                    # Check for directed path i → ... → j (excluding via k)
                    if _has_directed_path(i, j, directed, exclude=k):
                        directed[i, j] = True
                        undirected[i, j] = False
                        undirected[j, i] = False
                        changed = True
                    elif _has_directed_path(j, i, directed, exclude=k):
                        directed[j, i] = True
                        undirected[i, j] = False
                        undirected[j, i] = False
                        changed = True


def _has_directed_path(
    source: int,
    target: int,
    directed: np.ndarray,
    exclude: int = -1,
) -> bool:
    """Check if there is a directed path from source to target.

    Uses DFS on the directed edge graph, excluding ``exclude`` node
    (typically the common child in R4).

    Args:
        source: Start node
        target: End node
        directed: (d, d) boolean matrix, directed[i,j] = i → j
        exclude: Node to exclude from path (or -1 for no exclusion)

    Returns:
        True if a directed path exists from source to target
    """
    d = directed.shape[0]
    visited = np.zeros(d, dtype=bool)

    def _dfs(current: int) -> bool:
        if current == target:
            return True
        if visited[current]:
            return False
        visited[current] = True

        for nxt in range(d):
            if nxt == exclude:
                continue
            if directed[current, nxt]:
                if _dfs(nxt):
                    return True
        return False

    return _dfs(source)


def cpdag_to_nx(cpdag: np.ndarray) -> nx.Graph:
    """Convert a CPDAG matrix to a NetworkX mixed graph.

    Directed edges (value=1) become directed graph edges.
    Undirected edges (value=2) become undirected graph edges.

    Args:
        cpdag: (d, d) matrix with values {0, 1, 2}

    Returns:
        networkx.Graph with directed and undirected edges

    Example:
        >>> W = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        >>> cpdag = dag_to_cpdag(W)
        >>> G = cpdag_to_nx(cpdag)
        >>> len(G.edges())
        2
    """
    d = cpdag.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(d))

    for i in range(d):
        for j in range(d):
            if cpdag[i, j] == 1:  # directed i → j
                if not G.has_edge(i, j):
                    G.add_edge(i, j, directed=True, weight=1.0)
            elif cpdag[i, j] == 2 and i < j:  # undirected (count once)
                G.add_edge(i, j, directed=False, weight=1.0)

    return G


def compare_cpdag(
    W_true: np.ndarray,
    W_est: np.ndarray,
) -> Tuple[float, float, float]:
    """Compare estimated DAG against true DAG in CPDAG space.

    Computes precision, recall, and SHD between the CPDAGs of the
    true and estimated DAGs. This is important because two DAGs
    from the same Markov equivalence class should be considered
    equivalent.

    Args:
        W_true: True DAG adjacency matrix (d, d)
        W_est: Estimated DAG adjacency matrix (d, d)

    Returns:
        (precision, recall, shd) tuple

    Example:
        >>> W_true = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        >>> W_est = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])
        >>> p, r, s = compare_cpdag(W_true, W_est)
        >>> 0 <= p <= 1 and 0 <= r <= 1
        True
    """
    cpdag_true = dag_to_cpdag(W_true)
    cpdag_est = dag_to_cpdag(W_est)

    d = W_true.shape[0]

    # Extract directed and undirected edge sets
    true_directed = set()
    true_undirected = set()
    est_directed = set()
    est_undirected = set()

    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            if cpdag_true[i, j] == 1:
                true_directed.add((i, j))
            elif cpdag_true[i, j] == 2:
                true_undirected.add(tuple(sorted([i, j])))

            if cpdag_est[i, j] == 1:
                est_directed.add((i, j))
            elif cpdag_est[i, j] == 2:
                est_undirected.add(tuple(sorted([i, j])))

    # For SHD, count differences in edge presence and orientation
    all_edges_true = set()
    all_edges_est = set()

    for i in range(d):
        for j in range(d):
            if cpdag_true[i, j] in (1, 2):
                all_edges_true.add((i, j))
            if cpdag_est[i, j] in (1, 2):
                all_edges_est.add((i, j))

    true_edges_set = set()
    for i in range(d):
        for j in range(d):
            if cpdag_true[i, j] == 1:
                true_edges_set.add((i, j))  # directed
            elif cpdag_true[i, j] == 2:
                if i < j:
                    true_edges_set.add((i, j))  # undirected once

    est_edges_set = set()
    for i in range(d):
        for j in range(d):
            if cpdag_est[i, j] == 1:
                est_edges_set.add((i, j))
            elif cpdag_est[i, j] == 2:
                if i < j:
                    est_edges_set.add((i, j))

    # SHD: symmetric difference of edge sets
    shd = float(len(true_edges_set.symmetric_difference(est_edges_set)))

    # Precision and recall on CPDAG edges
    true_pos = len(true_edges_set & est_edges_set)
    est_total = len(est_edges_set)
    true_total = len(true_edges_set)

    precision = true_pos / est_total if est_total > 0 else 0.0
    recall = true_pos / true_total if true_total > 0 else 1.0

    return precision, recall, shd
