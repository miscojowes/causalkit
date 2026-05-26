"""
CausalBayes: Bayesian Causal Discovery with Neural Structure Learning & LLM-Informed Priors.

Main public API
---------------
>>> from causbayes import CausalBayesEstimator
>>> model = CausalBayesEstimator(method='bootstrap')
>>> model.fit(X)
>>> print(model.causal_matrix_)

Sub-modules
-----------
- structure_learning: Low-level algorithms (NOTEARS, BootstrapDAG, NeuralBayesianDAG)
- bayesian: Prior distributions, MC Dropout, variational inference
- llm_prior: LLM-based prior extraction (LLMPriorExtractor, LLMCausalAdvisor)
- evaluation: Metrics (SHD, AUC-PR, calibration, coverage)
- visualization: Plotting utilities for probabilistic DAGs
- models: Neural network architectures (NonlinearSEM, BayesianNN)
"""

__version__ = "0.1.0"

# ── Main public API ────────────────────────────────────────────────
from causbayes.estimator import CausalBayesEstimator

# ── Lower-level classes (still available for advanced use) ─────────
from causbayes.structure_learning.neural_notears import NeuralBayesianDAG
from causbayes.structure_learning.bootstrapped import BootstrapDAG
from causbayes.structure_learning.base import BaseStructureLearner

# ── LLM prior tools ────────────────────────────────────────────────
from causbayes.llm_prior import LLMPriorExtractor
from causbayes.llm_prior.advisor import LLMCausalAdvisor

# ── Evaluation ─────────────────────────────────────────────────────
from causbayes.evaluation import comprehensive_evaluation

# ── CPDAG utilities ────────────────────────────────────────────────
from causbayes.structure_learning.cpdag import (
    dag_to_cpdag,
    compare_cpdag,
    cpdag_to_nx,
)

# ── Metrics ────────────────────────────────────────────────────────
from causbayes.structure_learning.utils import (
    structural_hamming_distance,
    expected_shd,
)

__all__ = [
    # Main API
    "CausalBayesEstimator",
    # Lower-level structure learning
    "NeuralBayesianDAG",
    "BootstrapDAG",
    "BaseStructureLearner",
    # LLM priors
    "LLMPriorExtractor",
    "LLMCausalAdvisor",
    # Evaluation
    "comprehensive_evaluation",
    # CPDAG
    "dag_to_cpdag",
    "compare_cpdag",
    "cpdag_to_nx",
    # Metrics
    "structural_hamming_distance",
    "expected_shd",
]
