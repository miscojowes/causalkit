"""
CausalBayesEstimator: Unified scikit-learn style Estimator API for CausalBayes.

Single class for causal structure discovery with:
- NOTEARS (L-BFGS, DAGMA, Adam variants)
- BootstrapDAG with Platt-scaled uncertainty calibration
- NeuralBayesianDAG (nonlinear SEM)
- LLM-informed soft priors
- CPDAG output (Markov equivalence class)
- Pandas DataFrame support with named variables
- Automatic preprocessing (centering, scaling)
- Evaluation metrics and basic visualization

Usage:

    >>> import numpy as np
    >>> from causbayes import CausalBayesEstimator
    >>>
    >>> X = np.random.randn(200, 5)
    >>>
    >>> # Fast single NOTEARS run
    >>> model = CausalBayesEstimator(method='notears')
    >>> model.fit(X)
    >>> print(model.causal_matrix_)
    >>>
    >>> # Bootstrap with uncertainty (recommended)
    >>> model = CausalBayesEstimator(method='bootstrap', n_bootstraps=30)
    >>> model.fit(X)
    >>> print(model.edge_probs_)
    >>>
    >>> # With LLM prior from domain description
    >>> # model.fit(X, domain_description="Gene regulatory network in yeast")
    >>>
    >>> # Get CPDAG representation
    >>> cpdag = model.predict_cpdag()
    >>>
    >>> # Evaluate against ground truth
    >>> metrics = model.score(W_true=W_true)
"""

from typing import Optional, Union, Literal, List, Dict
import warnings
import copy

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════
#  Main Estimator Class
# ═══════════════════════════════════════════════════════════════════════


class CausalBayesEstimator:
    """Unified causal discovery estimator with scikit-learn style API.

    Supports:
    - **NOTEARS** (L-BFGS, DAGMA, Adam variants) — fast, no uncertainty
    - **BootstrapDAG** with Platt-scaled calibration (recommended)
    - **NeuralBayesianDAG** — nonlinear SEM with MC Dropout / VI
    - **LLM prior integration** — soft priors from domain description
    - **CPDAG output** — Markov equivalence class representation
    - **pandas DataFrames** with named variables
    - **Automatic preprocessing** — centering / scaling

    Parameters
    ----------
    method : str, default='bootstrap'
        One of:
        - ``'bootstrap'``: BootstrapDAG with calibrated uncertainty (recommended)
        - ``'notears'``: Single NOTEARS L-BFGS run (fast, no uncertainty)
        - ``'notears_lbfgs'``: Same as 'notears'
        - ``'notears_dagma'``: NOTEARS with DAGMA log-det acyclicity (stable)
        - ``'notears_adam'``: NOTEARS with Adam optimizer
        - ``'neural'``: NeuralBayesianDAG (nonlinear, slower)
    prior_source : str or None, default=None
        How to obtain the prior:
        - ``None``: No prior
        - ``'llm'``: Extract prior from LLM using ``domain_description``
        - ``'matrix'``: Provide explicit ``prior_matrix`` to ``fit()``
    lambda_prior : float, default=1.0
        Prior regularization strength. Higher = stronger bias toward prior.
        Used as L2 penalty: ``lambda_prior * sum((1 - prior) * W^2)``
    n_bootstraps : int, default=30
        Number of bootstrap samples (bootstrap method only).
    threshold : float or None, default=0.5
        Edge probability threshold for binary graph classification.
        If ``None`` and validation data is provided, auto-calibrated.
    calibrate : bool, default=True
        Apply Platt scaling calibration to bootstrap proportions
        (bootstrap method only). Requires ``W_val`` in ``fit()``.
    lambda_1 : float, default=0.01
        L1 regularization coefficient for NOTEARS.
    max_iter : int, default=10
        Maximum augmented Lagrangian iterations.
    w_threshold : float, default=0.1
        Prune edge weights below this magnitude.
    learning_rate : float, default=1e-3
        Learning rate for Adam optimiser (notears_adam, neural).
    hidden_layers : list or None, default=None
        Hidden layer sizes for neural method. Default: ``[64, 64]``.
    uncertainty : str or None, default='mc_dropout'
        Uncertainty method for neural: ``'mc_dropout'``, ``'variational'``,
        or ``None``.
    mc_samples : int, default=50
        MC Dropout or VI samples for neural method.
    llm_api_key : str or None, default=None
        API key for LLM prior extraction. If None and prior_source='llm',
        tries environment variable ``OPENAI_API_KEY``.
    llm_model : str, default='opencode-go/deepseek-v4-flash'
        LLM model for prior extraction.
    llm_api_base : str, default='https://api.opencode.ai/v1'
        API base URL for LLM.
    llm_confidence : str, default='medium'
        Confidence level for LLM prior: ``'high'``, ``'medium'``, ``'low'``.
    verbose : bool, default=False
        Print progress information.
    random_state : int, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    causal_matrix_ : np.ndarray
        Binary adjacency matrix of shape (d, d) after threshold.
        ``causal_matrix_[i, j] = 1`` means i -> j is predicted present.
    weight_matrix_ : np.ndarray
        Weighted adjacency matrix of shape (d, d) from the optimiser.
        ``weight_matrix_[i, j]`` is the estimated coefficient for i -> j.
    edge_probs_ : np.ndarray
        Edge existence probabilities of shape (d, d).
        ``edge_probs_[i, j]`` = probability that edge i -> j exists.
    edge_probs_raw_ : np.ndarray
        Uncalibrated raw bootstrap proportions (bootstrap method only).
    edge_stds_ : np.ndarray
        Standard deviation of edge weight estimates (d, d).
    cpdag_matrix_ : np.ndarray
        CPDAG representation of shape (d, d) with values:
        - ``0``: no edge
        - ``1``: directed edge i -> j (compelled)
        - ``2``: undirected edge i — j (reversible)
    variables_ : list of str
        Variable names (inferred from DataFrame columns or auto-named).
    n_features_in_ : int
        Number of variables.
    n_samples_in_ : int
        Number of training samples.
    fitted_ : bool
        Whether the model has been fitted.
    training_losses_ : list
        Training loss history (neural method only).
    """

    def __init__(
        self,
        method: str = "bootstrap",
        prior_source: Optional[str] = None,
        lambda_prior: float = 1.0,
        n_bootstraps: int = 30,
        threshold: Optional[float] = None,
        calibrate: bool = True,
        lambda_1: float = 0.01,
        max_iter: int = 10,
        w_threshold: float = 0.1,
        learning_rate: float = 1e-3,
        hidden_layers: Optional[list] = None,
        uncertainty: Optional[str] = "mc_dropout",
        mc_samples: int = 50,
        llm_api_key: Optional[str] = None,
        llm_model: str = "opencode-go/deepseek-v4-flash",
        llm_api_base: str = "https://api.opencode.ai/v1",
        llm_confidence: str = "medium",
        verbose: bool = False,
        random_state: int = 42,
    ):
        # Method selection
        valid_methods = {
            "bootstrap", "notears", "notears_lbfgs", "notears_dagma",
            "notears_adam", "neural",
        }
        if method not in valid_methods:
            raise ValueError(
                f"Unknown method '{method}'. Choose from: {sorted(valid_methods)}"
            )
        self.method = method
        self.prior_source = prior_source
        self.lambda_prior = lambda_prior
        self.n_bootstraps = n_bootstraps
        self.threshold = threshold
        self.calibrate = calibrate
        self.lambda_1 = lambda_1
        self.max_iter = max_iter
        self.w_threshold = w_threshold
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        self.uncertainty = uncertainty
        self.mc_samples = mc_samples
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.llm_api_base = llm_api_base
        self.llm_confidence = llm_confidence
        self.verbose = verbose
        self.random_state = random_state

        # Internal state
        self.fitted_ = False
        self._inner_model_ = None
        self._prior_matrix_ = None
        self._llm_extractor_ = None
        self._scaler = None

    # ─────────────────────────────────────────────────────────────────
    #  Fit
    # ─────────────────────────────────────────────────────────────────

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[np.ndarray] = None,
        domain_description: Optional[str] = None,
        prior_matrix: Optional[np.ndarray] = None,
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        W_val: Optional[np.ndarray] = None,
    ) -> "CausalBayesEstimator":
        """Fit the causal discovery model to data.

        Args:
            X: Training data. Shape (n_samples, n_features).
                Accepts ``np.ndarray`` or ``pd.DataFrame``.
            y: Ignored. Present for scikit-learn pipeline compatibility.
            domain_description: Free-text description of the domain.
                Required when ``prior_source='llm'``.
            prior_matrix: Explicit prior probability matrix of shape (d, d).
                Values in [0, 1]. Required when ``prior_source='matrix'``.
            X_val: Validation data for threshold calibration (optional).
                Shape (n_val, d).
            W_val: Ground truth adjacency for validation calibration
                (optional). Shape (d, d).

        Returns:
            Self for method chaining.
        """
        np.random.seed(self.random_state)

        # ── Extract / store variable names ──────────────────────
        if isinstance(X, pd.DataFrame):
            self.variables_ = list(X.columns)
            X = X.values.astype(float)
        else:
            X = np.asarray(X, dtype=float)
            d = X.shape[1]
            self.variables_ = [f"X{i}" for i in range(d)]

        n, d = X.shape
        self.n_features_in_ = d
        self.n_samples_in_ = n

        # Validate shape
        if n < 2:
            raise ValueError(
                f"Need at least 2 samples, got {n}"
            )

        # ── Handle validation data ──────────────────────────────
        if X_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values.astype(float)

        # ── Resolve prior ───────────────────────────────────────
        self._prior_matrix_ = self._resolve_prior(
            d, domain_description, prior_matrix
        )

        # ── Pick backend ────────────────────────────────────────
        method = self.method

        if method in ("notears", "notears_lbfgs"):
            self._fit_notears_lbfgs(X, d)

        elif method == "notears_dagma":
            self._fit_notears_dagma(X, d)

        elif method == "notears_adam":
            self._fit_notears_adam(X, d)

        elif method == "bootstrap":
            self._fit_bootstrap(X, d, X_val, W_val)

        elif method == "neural":
            self._fit_neural(X, d)

        else:
            raise ValueError(f"Unknown method: {method}")

        # ── Compute CPDAG ───────────────────────────────────────
        self._compute_cpdag()

        self.fitted_ = True
        return self

    # ─────────────────────────────────────────────────────────────────
    #  Prediction / Inference
    # ─────────────────────────────────────────────────────────────────

    def predict(self) -> np.ndarray:
        """Get the binary causal adjacency matrix (thresholded).

        Returns
        -------
        causal_matrix : np.ndarray of shape (d, d)
            Binary matrix where ``result[i, j] = 1`` means edge i -> j.
        """
        self._check_fitted()
        return self.causal_matrix_

    def predict_proba(self) -> np.ndarray:
        """Get edge existence probabilities.

        Returns
        -------
        edge_probs : np.ndarray of shape (d, d)
            ``result[i, j]`` = probability that edge i -> j exists.
        """
        self._check_fitted()
        return self.edge_probs_

    def predict_cpdag(self) -> np.ndarray:
        """Get the CPDAG (Markov equivalence class) representation.

        Encoding:
        - ``0``: no edge
        - ``1``: directed edge i -> j (compelled)
        - ``2``: undirected edge i — j (reversible)

        Returns
        -------
        cpdag : np.ndarray of shape (d, d) with values {0, 1, 2}.
        """
        self._check_fitted()
        return self.cpdag_matrix_

    def sample_graphs(self, n: int = 10) -> List[np.ndarray]:
        """Sample DAGs from the posterior over graphs.

        Uses Bernoulli sampling from edge probabilities with
        cycle rejection. Only available for methods that provide
        uncertainty (bootstrap, neural).

        Args:
            n: Number of DAGs to sample.

        Returns:
            List of binary adjacency matrices, each shape (d, d).
        """
        self._check_fitted()

        # Delegate to inner model if available
        if self._inner_model_ is not None and hasattr(
            self._inner_model_, "sample_graphs"
        ):
            return self._inner_model_.sample_graphs(n_samples=n)

        # Generic fallback: Bernoulli sampling with cycle rejection
        graphs = []
        d = self.n_features_in_
        max_attempts = n * 50
        attempts = 0

        while len(graphs) < n and attempts < max_attempts:
            attempts += 1
            W_sample = np.random.binomial(1, self.edge_probs_)
            np.fill_diagonal(W_sample, 0.0)
            if _is_dag_numpy(W_sample):
                graphs.append(W_sample)

        if len(graphs) < n:
            warnings.warn(
                f"Only sampled {len(graphs)}/{n} DAGs "
                f"({attempts} attempts) — high cycle probability"
            )
        return graphs

    # ─────────────────────────────────────────────────────────────────
    #  Score (Evaluation)
    # ─────────────────────────────────────────────────────────────────

    def score(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        W_true: Optional[np.ndarray] = None,
        metric: str = "shd",
    ) -> Union[float, Dict]:
        """Evaluate the learned graph against ground truth or data.

        Args:
            X: Data for computing BIC/AIC (if no ground truth available).
            W_true: True binary adjacency matrix of shape (d, d).
            metric: One of ``'shd'``, ``'precision'``, ``'recall'``,
                ``'f1'``, ``'auc_pr'``, ``'bic'``, ``'aic'``,
                ``'comprehensive'`` (returns all metrics).

        Returns:
            Float score or dict of scores for ``'comprehensive'``.
            For SHD / BIC / AIC: lower is better.
            For precision / recall / f1 / AUC-PR: higher is better.
        """
        self._check_fitted()

        if metric == "comprehensive":
            return self._comprehensive_evaluation(W_true)

        if metric == "bic":
            if X is None:
                raise ValueError("X is required for BIC computation")
            return self._bic_score(X)

        if metric == "aic":
            if X is None:
                raise ValueError("X is required for AIC computation")
            return self._aic_score(X)

        if W_true is None:
            raise ValueError(f"W_true is required for metric '{metric}'")

        W_bin = self.causal_matrix_

        if metric == "shd":
            from causbayes.structure_learning.utils import (
                structural_hamming_distance,
            )
            return structural_hamming_distance(W_true, W_bin)

        elif metric in ("precision", "recall", "f1"):
            tp = np.sum((W_bin > 0) & (W_true > 0))
            fp = np.sum((W_bin > 0) & (W_true == 0))
            fn = np.sum((W_bin == 0) & (W_true > 0))

            if metric == "precision":
                return tp / (tp + fp) if (tp + fp) > 0 else 0.0
            elif metric == "recall":
                return tp / (tp + fn) if (tp + fn) > 0 else 0.0
            else:  # f1
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        elif metric == "auc_pr":
            from causbayes.evaluation import precision_recall_auc
            result = precision_recall_auc(W_true, self.edge_probs_)
            return result["auc_pr"]

        elif metric == "cpdag":
            from causbayes.structure_learning.cpdag import compare_cpdag
            p, r, shd = compare_cpdag(W_true, W_bin)
            return {"precision": p, "recall": r, "shd": shd}

        else:
            raise ValueError(
                f"Unknown metric '{metric}'. Choose from: "
                "shd, precision, recall, f1, auc_pr, bic, aic, "
                "cpdag, comprehensive"
            )

    # ─────────────────────────────────────────────────────────────────
    #  LLM Prior Interface
    # ─────────────────────────────────────────────────────────────────

    def get_prior(
        self,
        variables: list,
        domain_description: str,
        confidence: Optional[str] = None,
    ) -> np.ndarray:
        """Extract a causal prior matrix from domain text using an LLM.

        This is a standalone utility that does NOT require calling
        ``fit()`` first. Returns a prior probability matrix that can
        be passed to ``fit(prior_matrix=...)``.

        Args:
            variables: List of variable names.
            domain_description: Free-text domain description.
            confidence: Confidence level. Defaults to ``self.llm_confidence``.

        Returns:
            Prior probability matrix of shape (d, d) with values in [0, 1].
        """
        from causbayes.llm_prior import LLMPriorExtractor

        api_key = self.llm_api_key
        if api_key is None:
            import os
            api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "LLM prior extraction requires an API key. "
                "Set llm_api_key or OPENAI_API_KEY environment variable."
            )

        conf = confidence or self.llm_confidence

        if self.verbose:
            print(
                f"  Extracting LLM prior: {len(variables)} vars, "
                f"confidence={conf}"
            )

        extractor = LLMPriorExtractor(
            api_key=api_key,
            model=self.llm_model,
            api_base=self.llm_api_base,
        )
        prior = extractor.extract_edge_priors(
            variables=variables,
            domain_description=domain_description,
            confidence=conf,
        )
        return prior

    # ─────────────────────────────────────────────────────────────────
    #  Visualization
    # ─────────────────────────────────────────────────────────────────

    def plot(
        self,
        threshold: float = 0.3,
        show_uncertainty: bool = True,
        figsize: tuple = (10, 8),
        title: Optional[str] = None,
        show_edge_labels: bool = False,
    ):
        """Visualize the learned causal graph with uncertainty.

        Args:
            threshold: Minimum edge probability to display.
            show_uncertainty: Color edges by uncertainty if True.
            figsize: Figure dimensions (width, height) in inches.
            title: Plot title. Defaults to auto-generated title.
            show_edge_labels: Show probability/std on edges.
        """
        self._check_fitted()

        from causbayes.visualization import plot_probabilistic_dag

        plot_probabilistic_dag(
            edge_probs=self.edge_probs_,
            edge_stds=self.edge_stds_,
            threshold=threshold,
            uncertainty=show_uncertainty,
            variable_names=self.variables_,
            figsize=figsize,
            title=title or f"CausalBayes — {self.method}",
            show_edge_labels=show_edge_labels,
        )

    # ─────────────────────────────────────────────────────────────────
    #  scikit-learn compatibility
    # ─────────────────────────────────────────────────────────────────

    def get_params(self, deep: bool = True) -> dict:
        """Get estimator parameters (scikit-learn compatibility)."""
        return {
            "method": self.method,
            "prior_source": self.prior_source,
            "lambda_prior": self.lambda_prior,
            "n_bootstraps": self.n_bootstraps,
            "threshold": self.threshold,
            "calibrate": self.calibrate,
            "lambda_1": self.lambda_1,
            "max_iter": self.max_iter,
            "w_threshold": self.w_threshold,
            "learning_rate": self.learning_rate,
            "hidden_layers": self.hidden_layers,
            "uncertainty": self.uncertainty,
            "mc_samples": self.mc_samples,
            "llm_api_key": self.llm_api_key,
            "llm_model": self.llm_model,
            "llm_api_base": self.llm_api_base,
            "llm_confidence": self.llm_confidence,
            "verbose": self.verbose,
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> "CausalBayesEstimator":
        """Set estimator parameters (scikit-learn compatibility)."""
        for key, val in params.items():
            if hasattr(self, key):
                setattr(self, key, val)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        return self

    # ─────────────────────────────────────────────────────────────────
    #  String representation
    # ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"CausalBayesEstimator(method={self.method!r}, "
            f"prior_source={self.prior_source!r}, "
            f"lambda_prior={self.lambda_prior})"
        )

    # ─────────────────────────────────────────────────────────────────
    #  Internal: Prior resolution
    # ─────────────────────────────────────────────────────────────────

    def _resolve_prior(
        self,
        d: int,
        domain_description: Optional[str],
        prior_matrix: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Resolve the prior matrix from the available sources."""
        source = self.prior_source

        if source is None:
            # If a prior_matrix was passed directly to fit(), use it
            if prior_matrix is not None:
                return prior_matrix
            return None

        if source == "matrix":
            if prior_matrix is None:
                raise ValueError(
                    "prior_source='matrix' requires passing "
                    "prior_matrix to fit()"
                )
            prior_matrix = np.asarray(prior_matrix, dtype=float)
            if prior_matrix.shape != (d, d):
                raise ValueError(
                    f"prior_matrix shape {prior_matrix.shape} "
                    f"does not match data shape ({d}, {d})"
                )
            return prior_matrix

        if source == "llm":
            if domain_description is None:
                raise ValueError(
                    "prior_source='llm' requires passing "
                    "domain_description to fit()"
                )
            prior = self.get_prior(
                variables=self.variables_,
                domain_description=domain_description,
            )
            return prior

        raise ValueError(
            f"Unknown prior_source '{source}'. "
            "Choose from: None, 'llm', 'matrix'"
        )

    # ─────────────────────────────────────────────────────────────────
    #  Internal: Backend fitting methods
    # ─────────────────────────────────────────────────────────────────

    def _fit_notears_lbfgs(self, X: np.ndarray, d: int):
        """Fit single NOTEARS L-BFGS run."""
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self._scaler = scaler

        from causbayes.structure_learning.notears_fast import (
            notears_lbfgs,
        )

        W = notears_lbfgs(
            X_scaled,
            lambda_1=self.lambda_1,
            max_iter=self.max_iter,
            w_threshold=self.w_threshold,
            prior_matrix=self._prior_matrix_,
            lambda_prior=self.lambda_prior,
            lbfgs_maxiter=20,
        )

        self.weight_matrix_ = W
        W_abs = np.abs(W)
        # Edge probability = 1 if non-zero (no uncertainty for single run)
        self.edge_probs_ = (W_abs > self.w_threshold).astype(float)
        np.fill_diagonal(self.edge_probs_, 0.0)
        self.edge_stds_ = np.zeros((d, d))
        self.edge_probs_raw_ = self.edge_probs_.copy()
        self.causal_matrix_ = self.edge_probs_.copy()
        self._inner_model_ = None

    def _fit_notears_dagma(self, X: np.ndarray, d: int):
        """Fit single NOTEARS run with DAGMA log-det acyclicity."""
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self._scaler = scaler

        from causbayes.structure_learning.notears_fast import (
            notears_lbfgs_dagma,
        )

        W = notears_lbfgs_dagma(
            X_scaled,
            lambda_1=self.lambda_1,
            max_iter=self.max_iter,
            w_threshold=self.w_threshold,
            prior_matrix=self._prior_matrix_,
            lambda_prior=self.lambda_prior,
            lbfgs_maxiter=20,
        )

        self.weight_matrix_ = W
        W_abs = np.abs(W)
        self.edge_probs_ = (W_abs > self.w_threshold).astype(float)
        np.fill_diagonal(self.edge_probs_, 0.0)
        self.edge_stds_ = np.zeros((d, d))
        self.edge_probs_raw_ = self.edge_probs_.copy()
        self.causal_matrix_ = self.edge_probs_.copy()
        self._inner_model_ = None

    def _fit_notears_adam(self, X: np.ndarray, d: int):
        """Fit single NOTEARS run with Adam optimiser."""
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self._scaler = scaler

        from causbayes.structure_learning.notears_fast import (
            notears_adam,
        )

        W = notears_adam(
            X_scaled,
            lambda_1=self.lambda_1,
            max_iter=self.max_iter,
            lr=self.learning_rate,
            w_threshold=self.w_threshold,
            prior_matrix=self._prior_matrix_,
            lambda_prior=self.lambda_prior,
        )

        self.weight_matrix_ = W
        W_abs = np.abs(W)
        self.edge_probs_ = (W_abs > self.w_threshold).astype(float)
        np.fill_diagonal(self.edge_probs_, 0.0)
        self.edge_stds_ = np.zeros((d, d))
        self.edge_probs_raw_ = self.edge_probs_.copy()
        self.causal_matrix_ = self.edge_probs_.copy()
        self._inner_model_ = None

    def _fit_bootstrap(
        self,
        X: np.ndarray,
        d: int,
        X_val: Optional[np.ndarray] = None,
        W_val: Optional[np.ndarray] = None,
    ):
        """Fit bootstrap DAG with uncertainty calibration."""
        from causbayes.structure_learning.bootstrapped import BootstrapDAG

        model = BootstrapDAG(
            n_bootstraps=self.n_bootstraps,
            lambda_1=self.lambda_1,
            threshold=self.threshold,
            max_iter=self.max_iter,
            w_threshold=self.w_threshold,
            prior_matrix=self._prior_matrix_,
            lambda_prior=self.lambda_prior,
            calibrate=self.calibrate,
            verbose=self.verbose,
            seed=self.random_state,
            lbfgs_maxiter=20,
        )
        model.fit(X, X_val=X_val, W_val=W_val)

        self._inner_model_ = model
        self._scaler = model.scaler_
        self.weight_matrix_ = (
            np.mean(model._weight_matrices_, axis=0)
            if len(model._weight_matrices_) > 0
            else np.zeros((d, d))
        )
        self.edge_probs_ = model._edge_probs_.copy()
        self.edge_probs_raw_ = model._edge_probs_raw_.copy()
        self.edge_stds_ = model._edge_stds_.copy()
        self.edge_strength_ = model._edge_strength_.copy() if hasattr(model, '_edge_strength_') else None
        
        # Use mean edge strength for thresholding (more robust on real data)
        if self.edge_strength_ is not None:
            t = self.threshold if self.threshold is not None else 0.03
            self.causal_matrix_ = (self.edge_strength_ >= t).astype(float)
            self.threshold = t
        else:
            t = self.threshold if self.threshold is not None else 0.5
            self.causal_matrix_ = (self.edge_probs_ >= t).astype(float)
            self.threshold = t

    def _fit_neural(self, X: np.ndarray, d: int):
        """Fit neural Bayesian DAG (nonlinear SEM)."""
        from causbayes.structure_learning.neural_notears import (
            NeuralBayesianDAG,
        )

        model = NeuralBayesianDAG(
            hidden_layers=self.hidden_layers or [64, 64],
            learning_rate=self.learning_rate,
            lambda_1=self.lambda_1,
            lambda_prior=self.lambda_prior,
            max_iter=self.max_iter,
            uncertainty=self.uncertainty,
            mc_samples=self.mc_samples,
            prior_matrix=self._prior_matrix_,
            prior_strength=self.lambda_prior,
            device="cpu",
            seed=self.random_state,
            verbose=self.verbose,
        )
        model.fit(X)

        self._inner_model_ = model
        self._scaler = model.scaler_
        self.weight_matrix_ = model.W_est_.copy()
        self.edge_probs_ = model._edge_probs_.copy()
        self.edge_stds_ = model._edge_stds_.copy()
        self.edge_probs_raw_ = self.edge_probs_.copy()
        self.training_losses_ = model._training_losses_.copy()
        self.threshold = self.threshold if self.threshold is not None else 0.5
        self.causal_matrix_ = (self.edge_probs_ >= self.threshold).astype(float)

    # ─────────────────────────────────────────────────────────────────
    #  Internal: CPDAG computation
    # ─────────────────────────────────────────────────────────────────

    def _compute_cpdag(self):
        """Compute CPDAG from the binary adjacency matrix."""
        from causbayes.structure_learning.cpdag import dag_to_cpdag

        if self.causal_matrix_ is not None and np.any(self.causal_matrix_):
            self.cpdag_matrix_ = dag_to_cpdag(self.causal_matrix_)
        else:
            d = self.n_features_in_
            self.cpdag_matrix_ = np.zeros((d, d), dtype=np.int8)

    # ─────────────────────────────────────────────────────────────────
    #  Internal: Evaluation helpers
    # ─────────────────────────────────────────────────────────────────

    def _comprehensive_evaluation(
        self, W_true: Optional[np.ndarray]
    ) -> dict:
        """Run comprehensive evaluation with all available metrics."""
        from causbayes.evaluation import comprehensive_evaluation

        if W_true is not None:
            # Full evaluation from evaluation module
            result = comprehensive_evaluation(
                W_true=W_true,
                P_est=self.edge_probs_,
                P_std=self.edge_stds_,
            )
            # Add CPDAG metrics
            from causbayes.structure_learning.cpdag import compare_cpdag
            p, r, shd = compare_cpdag(W_true, self.causal_matrix_)
            result["cpdag_precision"] = p
            result["cpdag_recall"] = r
            result["cpdag_shd"] = shd
            return result

        return {"error": "W_true not provided"}

    def _bic_score(self, X: Union[np.ndarray, pd.DataFrame]) -> float:
        """Compute BIC score on data."""
        from causbayes.structure_learning.scores import bic_score

        if isinstance(X, pd.DataFrame):
            X = X.values.astype(float)
        return bic_score(X, self.causal_matrix_)

    def _aic_score(self, X: Union[np.ndarray, pd.DataFrame]) -> float:
        """Compute AIC score on data."""
        from causbayes.structure_learning.scores import aic_score

        if isinstance(X, pd.DataFrame):
            X = X.values.astype(float)
        return aic_score(X, self.causal_matrix_)

    # ─────────────────────────────────────────────────────────────────
    #  Internal: Guard
    # ─────────────────────────────────────────────────────────────────

    def _check_fitted(self):
        """Raise RuntimeError if model is not fitted."""
        if not self.fitted_:
            raise RuntimeError(
                "CausalBayesEstimator not fitted yet. "
                "Call fit() before using this method."
            )


# ═══════════════════════════════════════════════════════════════════════
#  Utility functions
# ═══════════════════════════════════════════════════════════════════════


def _is_dag_numpy(W: np.ndarray) -> bool:
    """Check if a binary adjacency matrix represents a DAG.

    Uses topological ordering: a DAG has a permutation that makes the
    matrix strictly upper triangular. We use a simple heuristic with
    repeated source removal (Kahn's algorithm).
    """
    d = W.shape[0]
    adj = W.copy().astype(float)
    visited = np.zeros(d, dtype=bool)
    # Large sentinel to mark visited nodes as ineligible
    LARGE = 1e9

    for _ in range(d):
        # Find a node with no incoming edges (among unvisited)
        indegrees = adj.sum(axis=0)
        indegrees[visited] = LARGE
        candidates = np.where(indegrees < 0.5)[0]
        if len(candidates) == 0:
            return False  # Cycle found
        node = candidates[0]
        visited[node] = True
        # Remove outgoing edges from this node
        adj[node, :] = 0.0

    return True
