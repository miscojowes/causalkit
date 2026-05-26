"""
causalkit — Practical Causal Discovery & Causal ML
===================================================
The easiest way to discover causal structures from data + domain knowledge,
then answer causal questions.

Main API:
    >>> import causalkit as ck
    >>> model = ck.CausalDiscoverer()
    >>> model.fit(X, domain_text="ad spend drives sales")
    >>> print(model.causal_matrix_)        # weighted DAG
    >>> print(model.edge_confidence_)      # bootstrap confidence
    >>> ate = model.estimate_ate(X, treatment="ads", outcome="sales")
"""

__version__ = "0.2.0"

from causalkit.discoverer import CausalDiscoverer

__all__ = ["CausalDiscoverer"]
