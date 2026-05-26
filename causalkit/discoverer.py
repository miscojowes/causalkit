"""
causalkit.discoverer — CausalDiscoverer (main public API)
==========================================================

Usage:
    >>> import causalkit as ck
    >>> model = ck.CausalDiscoverer(adaptive_trust=True)
    >>> model.fit(X)
    >>> print(model.causal_matrix_)
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from causbayes.structure_learning.notears_fast import notears_lbfgs
from causalkit.adaptive_trust import compute_per_edge_lambda
from causalkit.effects import estimate_ate, estimate_risk_diff


class CausalDiscoverer:
    """Causal structure discovery with domain knowledge integration.

    Parameters
    ----------
    method : str, default='bootstrap'
        Algorithm: 'notears' (single run, fast), 'bootstrap' (with uncertainty,
        recommended), 'dagma' (pytorch-based).
    n_bootstraps : int, default=50
        Number of bootstrap samples. Only for method='bootstrap'.
    lambda_1 : float, default=0.005
        L1 sparsity penalty (NOTEARS parameter).
    lambda_prior : float, default=0.5
        Base strength of prior penalty.
    adaptive_trust : bool, default=True
        Whether to use per-edge adaptive trust (simplified PRCD-MAP).
        When True, the prior influence per edge is automatically adjusted:
        boosted where prior agrees with data, attenuated where it disagrees.
    threshold : float or None, default=None
        Edge threshold. If None, auto-determined by F1-optimization
        against the prior (or default 0.01 when no prior).
    random_state : int, default=42
        Random seed for reproducibility.
    verbose : bool, default=True
        Print progress info.

    Attributes
    ----------
    causal_matrix_ : ndarray of shape (d, d)
        Weighted adjacency matrix of the discovered DAG.
        causal_matrix_[i,j] > 0 means i -> j.
    edge_confidence_ : ndarray of shape (d, d)
        Bootstrap edge probabilities [0, 1].
        Only available with method='bootstrap'.
    feature_names_ : list of str
        Variable names (from DataFrame columns or generated).
    """

    def __init__(
        self,
        method="bootstrap",
        n_bootstraps=50,
        lambda_1=0.005,
        lambda_prior=0.5,
        adaptive_trust=False,
        threshold=None,
        random_state=42,
        verbose=True,
    ):
        self.method = method
        self.n_bootstraps = n_bootstraps
        self.lambda_1 = lambda_1
        self.lambda_prior = lambda_prior
        self.adaptive_trust = adaptive_trust
        self.threshold = threshold
        self.random_state = random_state
        self.verbose = verbose

        self.causal_matrix_ = None
        self.edge_confidence_ = None
        self.feature_names_ = None

    def _log(self, msg):
        if self.verbose:
            print(f"[causalkit] {msg}")

    def fit(self, X, prior_matrix=None, domain_text=None, feature_names=None):
        """Discover the causal graph from data.

        Parameters
        ----------
        X : ndarray or DataFrame of shape (n_samples, n_features)
            The observational data.
        prior_matrix : ndarray or None, default=None
            Prior adjacency matrix. prior_matrix[i,j] ∈ [0, 1] indicates
            confidence that edge i -> j exists.
        domain_text : str or None, default=None
            Natural language description of domain knowledge.
            If provided and prior_matrix is None, this is auto-extracted into
            a prior matrix via LLM (requires LLM access).
        feature_names : list of str or None, default=None
            Names for variables. If X is a DataFrame, column names are used.

        Returns
        -------
        self : CausalDiscoverer
        """
        # ── Input handling ────────────────────────────────────────────
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X = X.values
        elif feature_names is not None:
            self.feature_names_ = list(feature_names)
        else:
            self.feature_names_ = [f"x{i}" for i in range(X.shape[1])]

        n, d = X.shape
        self._log(f"Fitting on {n} samples × {d} features, method='{self.method}'")

        # ── Standardize ───────────────────────────────────────────────
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        # ── Prior resolution ──────────────────────────────────────────
        if prior_matrix is None and domain_text is not None:
            prior_matrix = self._extract_prior_from_text(domain_text)

        if prior_matrix is not None:
            prior_matrix = np.asarray(prior_matrix, dtype=float)
            if prior_matrix.shape != (d, d):
                raise ValueError(
                    f"prior_matrix shape {prior_matrix.shape} != ({d}, {d})"
                )
            np.fill_diagonal(prior_matrix, 0.0)
        else:
            if self.adaptive_trust:
                self._log("No prior given — adaptive_trust has no effect without one")

        # ── Run discovery ─────────────────────────────────────────────
        if self.method == "notears":
            self._fit_notears(X_std, prior_matrix)
        elif self.method == "bootstrap":
            self._fit_bootstrap(X_std, prior_matrix)
        elif self.method == "dagma":
            self._fit_dagma(X_std, prior_matrix)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # ── Threshold ─────────────────────────────────────────────────
        if self.threshold is not None:
            t = self.threshold
            self.causal_matrix_ = (self._raw_strength_ > t).astype(float)
        else:
            # Auto-threshold mean-strength
            # Default: 0.01 for mean-strength (empirically validated)
            DEFAULT_TH = 0.01
            strengths = self._raw_strength_.flatten()
            strengths = strengths[strengths > 1e-6]
            if len(strengths) > 0:
                # Use mean as adaptive threshold (robust to scale)
                adaptive_th = max(DEFAULT_TH, strengths.mean() * 0.5)
            else:
                adaptive_th = DEFAULT_TH
            self.causal_matrix_ = (self._raw_strength_ > adaptive_th).astype(float)
            if self.verbose:
                n_edges = int(self.causal_matrix_.sum())
                self._log(f"  Auto-threshold: {adaptive_th:.4f}, {n_edges} edges")

        return self

    def _fit_notears(self, X, prior_matrix):
        """Single NOTEARS run (fast, no uncertainty)."""
        d = X.shape[1]

        if prior_matrix is not None and self.adaptive_trust:
            # 2-pass: uniform → adaptive → final
            prior_lambda = self._compute_adaptive_lambda(X, prior_matrix, self.lambda_prior)
        else:
            prior_lambda = self.lambda_prior if prior_matrix is not None else 0.0

        W = notears_lbfgs(
            X,
            lambda_1=self.lambda_1,
            prior_matrix=prior_matrix,
            lambda_prior=prior_lambda,
            w_threshold=0.01 if self.threshold is None else self.threshold,
        )
        self._raw_strength_ = np.abs(W)
        np.fill_diagonal(self._raw_strength_, 0.0)

    def _fit_bootstrap(self, X, prior_matrix):
        """Bootstrap with uncertainty estimates."""
        from sklearn.utils import resample

        n, d = X.shape
        self._log(f"Running {self.n_bootstraps} bootstraps...")

        def _run_bootstrap(prior_matrix, lambda_prior, seed_offset=0):
            """Run B bootstraps, return mean strength + std + count."""
            W_list = []
            for i in range(self.n_bootstraps):
                Xb = resample(X, random_state=self.random_state + seed_offset + i)
                Xb -= Xb.mean(axis=0, keepdims=True)
                kwargs = dict(
                    lambda_1=self.lambda_1,
                    max_iter=10, w_threshold=0.001,
                    lbfgs_maxiter=30,
                )
                if prior_matrix is not None:
                    kwargs['prior_matrix'] = prior_matrix
                    kwargs['lambda_prior'] = lambda_prior
                W = notears_lbfgs(Xb, **kwargs)
                if not np.isnan(W).any():
                    W_list.append(W)
            if not W_list:
                raise RuntimeError("All bootstrap runs returned NaN")
            W_arr = np.abs(np.array(W_list))
            W_mean = np.mean(W_arr, axis=0)
            W_std = np.std(W_arr, axis=0)
            np.fill_diagonal(W_mean, 0.0)
            np.fill_diagonal(W_std, 0.0)
            return W_mean, W_std, len(W_list)

        if prior_matrix is not None and self.adaptive_trust:
            # ── Round 1: uniform λ → estimate strengths ──
            self._log("  Round 1: uniform λ prior...")
            W1_str, W1_std, n1 = _run_bootstrap(prior_matrix, self.lambda_prior, 0)
            self._log(f"  Round 1: {n1}/{self.n_bootstraps} completed")

            # ── Compute per-edge adaptive λ ──
            lambda_per_edge = compute_per_edge_lambda(
                prior_matrix, W1_str, W1_std, self.lambda_prior
            )
            n_boosted = int(np.sum(lambda_per_edge > self.lambda_prior * 1.1))
            n_atten = int(np.sum(lambda_per_edge < self.lambda_prior * 0.9))
            self._log(f"  Adaptive λ: {n_boosted} boosted, {n_atten} attenuated")

            # ── Round 2: bootstrap with per-edge λ ──
            self._log("  Round 2: per-edge adaptive λ...")
            W_str, W_std, n2 = _run_bootstrap(prior_matrix, lambda_per_edge, 100)
            self._log(f"  Round 2: {n2}/{self.n_bootstraps} completed")

        elif prior_matrix is not None:
            # Bootstrap with uniform λ
            W_str, W_std, n = _run_bootstrap(prior_matrix, self.lambda_prior, 0)
            self._log(f"  {n}/{self.n_bootstraps} completed")

        else:
            # Bootstrap without prior
            W_str, W_std, n = _run_bootstrap(None, 0.0, 0)
            self._log(f"  {n}/{self.n_bootstraps} completed")

        self._raw_strength_ = W_str
        # edge_confidence_ = mean strength normalized to [0,1] for interpretability
        max_s = np.max(W_str)
        if max_s > 0:
            self.edge_confidence_ = W_str / max_s
        else:
            self.edge_confidence_ = W_str

    def _fit_dagma(self, X, prior_matrix):
        """DAGMA-based discovery (PyTorch)."""
        try:
            from causbayes.structure_learning.notears_fast import dagma_linear
        except ImportError:
            raise ImportError("DAGMA requires torch. Install with: pip install torch")
        d = X.shape[1]

        if prior_matrix is not None and self.adaptive_trust:
            prior_lambda = self._compute_adaptive_lambda(X, prior_matrix, self.lambda_prior)
        else:
            prior_lambda = self.lambda_prior if prior_matrix is not None else 0.0

        W = dagma_linear(
            X,
            lambda_1=self.lambda_1,
            prior_matrix=prior_matrix,
            lambda_prior=prior_lambda,
        )
        self._raw_strength_ = np.abs(W)
        np.fill_diagonal(self._raw_strength_, 0.0)

    def _compute_adaptive_lambda(self, X, prior_matrix, lambda_base):
        """Two-pass: uniform λ → edge strengths → per-edge λ."""
        from sklearn.utils import resample

        n_boot = min(self.n_bootstraps, 50)
        self._log(f"Computing adaptive λ with {n_boot} pilot bootstraps...")

        W_list = []
        for i in range(n_boot):
            Xb = resample(X, random_state=self.random_state + i)
            Xb -= Xb.mean(axis=0, keepdims=True)
            W = notears_lbfgs(
                Xb,
                lambda_1=self.lambda_1,
                max_iter=10,
                w_threshold=0.001,
                lbfgs_maxiter=30,
                prior_matrix=prior_matrix,
                lambda_prior=lambda_base,
            )
            if not np.isnan(W).any():
                W_list.append(W)

        if not W_list:
            self._log("Warning: pilot bootstrap failed, using uniform λ")
            return lambda_base

        W_str = np.mean(np.abs(np.array(W_list)), axis=0)
        W_std = np.std(np.abs(np.array(W_list)), axis=0)
        np.fill_diagonal(W_str, 0.0)
        np.fill_diagonal(W_std, 0.0)

        return compute_per_edge_lambda(prior_matrix, W_str, W_std, lambda_base)

    def _extract_prior_from_text(self, domain_text):
        """Extract prior matrix from domain description using LLM."""
        self._log("Extracting prior from domain text...")
        try:
            from causalkit.prior import extract_prior_from_text

            prior = extract_prior_from_text(
                domain_text, self.feature_names_, random_state=self.random_state
            )
            return prior
        except Exception as e:
            self._log(f"Warning: prior extraction failed ({e}), no prior used")
            return None

    # ── Causal ML methods ─────────────────────────────────────────────

    def estimate_ate(self, X, treatment, outcome, method="linear"):
        """Estimate Average Treatment Effect.

        Parameters
        ----------
        X : ndarray or DataFrame
            Original data.
        treatment : str or int
            Treatment variable name/index.
        outcome : str or int
            Outcome variable name/index.
        method : str, default='linear'
            Estimator: 'linear' (OLS on adjustment set from DAG).

        Returns
        -------
        ate : float
            Average Treatment Effect estimate.
        """
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = X

        return estimate_ate(
            X_arr, treatment, outcome,
            self.causal_matrix_, self.feature_names_,
            method=method,
        )

    def counterfactual_predict(self, X, interventions):
        """What-if: set variable(s) to new values, propagate through DAG.

        Parameters
        ----------
        X : ndarray or DataFrame of shape (n_samples, n_features)
        interventions : dict
            {variable_name_or_index: new_value}
            E.g., {'ad_spend': 50000} or {0: 1.5}

        Returns
        -------
        pred : ndarray of shape (n_samples, n_features)
            Predicted values for all variables under intervention.
        """
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = X

        return self._linear_whatif(X_arr, interventions)

    def _linear_whatif(self, X, interventions):
        """Linear SEM what-if: W @ X = ε → X = inv(I - W) @ ε"""
        d = X.shape[1]
        W = self.causal_matrix_

        # Resolve intervention indices
        idx_map = {}
        for k, v in interventions.items():
            if isinstance(k, str):
                if self.feature_names_ is not None and k in self.feature_names_:
                    idx_map[self.feature_names_.index(k)] = v
                else:
                    raise ValueError(f"Variable '{k}' not found")
            else:
                idx_map[int(k)] = v

        # Linear SEM: X_j = sum_i W[i,j] * X_i + ε_j
        # Under intervention: fix intervened variables to their new value
        # For non-intervened variables, propagate through the DAG ordering
        X_pred = X.copy()

        # Get topological order
        # Simple heuristic: sort by column sum (causal flow direction)
        in_degree = np.sum(W > 0, axis=0)
        topo_order = np.argsort(in_degree)

        for j in topo_order:
            if j in idx_map:
                X_pred[:, j] = idx_map[j]
            else:
                parents = np.where(W[:, j] > 0)[0]
                if len(parents) > 0:
                    # Linear prediction: X_j = sum_i coef_i * X_i
                    coefs = W[parents, j]
                    X_pred[:, j] = X_pred[:, parents] @ coefs
                # else: root node, keep as-is

        return X_pred

    def root_cause_analysis(self, X, target, top_k=5):
        """Rank root causes of a target variable by causal effect strength.

        Parameters
        ----------
        X : ndarray or DataFrame
        target : str or int
            Target variable.
        top_k : int, default=5
            Number of top root causes to return.

        Returns
        -------
        causes : list of (var_name, effect_strength)
        """
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = X

        if isinstance(target, str):
            if self.feature_names_ is not None and target in self.feature_names_:
                target_idx = self.feature_names_.index(target)
            else:
                raise ValueError(f"Variable '{target}' not found")
        else:
            target_idx = int(target)

        # Direct parents have causal effect
        W = self.causal_matrix_
        parents = np.where(W[:, target_idx] > 0)[0]
        causes = []
        for p in parents:
            name = self.feature_names_[p] if self.feature_names_ else str(p)
            strength = W[p, target_idx]
            causes.append((name, float(strength)))

        # Sort by strength
        causes.sort(key=lambda x: -abs(x[1]))
        return causes[:top_k]

    def plot(self, filename=None, max_vars=30, **kwargs):
        """Visualize the discovered DAG with edge confidence.

        Requires matplotlib and networkx.
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        W = self.causal_matrix_
        d = W.shape[0]
        if d > max_vars:
            print(f"Graph too large ({d} vars > {max_vars}), skipping plot")
            return

        G = nx.DiGraph()
        labels = {}
        for i in range(d):
            name = self.feature_names_[i] if self.feature_names_ else str(i)
            labels[i] = name
            G.add_node(i)

        for i in range(d):
            for j in range(d):
                if W[i, j] > 0 and i != j:
                    conf = self.edge_confidence_[i, j] if self.edge_confidence_ is not None else 0.5
                    G.add_edge(i, j, weight=conf)

        pos = nx.spring_layout(G, seed=self.random_state)
        plt.figure(figsize=(10, 8))
        nx.draw(G, pos, labels=labels, with_labels=True,
                node_color="lightblue", node_size=800,
                font_size=8, arrowsize=15,
                edge_color=[G[u][v]["weight"] for u, v in G.edges],
                edge_cmap=plt.cm.Blues, width=2)
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches="tight")
            print(f"Saved to {filename}")
        else:
            plt.show()

    def __repr__(self):
        return (
            f"CausalDiscoverer(method='{self.method}', n_bootstraps={self.n_bootstraps}, "
            f"adaptive_trust={self.adaptive_trust})"
        )
