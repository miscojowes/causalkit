"""
Visualization tools for probabilistic DAGs.

Plots DAGs with edge uncertainty represented through
color intensity, line width, and optional annotations.
"""

import numpy as np
import warnings


def plot_probabilistic_dag(
    edge_probs: np.ndarray,
    edge_stds: np.ndarray = None,
    threshold: float = 0.3,
    uncertainty: bool = True,
    variable_names: list = None,
    figsize: tuple = (10, 8),
    title: str = "Probabilistic Causal DAG",
    show_edge_labels: bool = False,
    ax=None,
):
    """Plot a DAG with uncertainty visualization.

    Args:
        edge_probs: Edge probability matrix P[i,j]
        edge_stds: Edge standard deviation matrix
        threshold: Minimum probability to show edge
        uncertainty: Color edges by uncertainty if True
        variable_names: Labels for nodes
        figsize: Figure size
        title: Plot title
        show_edge_labels: Show probability/std on edges
        ax: Matplotlib axis (optional)
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        warnings.warn("matplotlib and networkx required for plotting")
        return

    d = edge_probs.shape[0]
    if variable_names is None:
        variable_names = [f"X{i}" for i in range(d)]

    G = nx.DiGraph()
    for i in range(d):
        G.add_node(i, label=variable_names[i])

    edges = []
    for i in range(d):
        for j in range(d):
            if i != j and edge_probs[i, j] >= threshold:
                std = edge_stds[i, j] if edge_stds is not None else 0.0
                edges.append((i, j, edge_probs[i, j], std))

    if not edges:
        warnings.warn("No edges above threshold to plot")
        return

    # Normalize probabilities for edge attributes
    max_prob = max(e[2] for e in edges) if edges else 1.0
    min_prob = min(e[2] for e in edges) if edges else 0.0
    prob_range = max(max_prob - min_prob, 0.01)

    for i, j, prob, std in edges:
        G.add_edge(i, j,
                   weight=prob,
                   probability=prob,
                   uncertainty=std,
                   normalized_prob=(prob - min_prob) / prob_range)

    # Layout
    pos = nx.spring_layout(G, k=3 / np.sqrt(d), seed=42)

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    # Edge colors and widths
    edge_colors = []
    edge_widths = []
    edge_labels = {}

    cmap = plt.cm.RdYlGn  # Red (low prob) to Green (high prob)

    for i, j, data in G.edges(data=True):
        prob = data.get("probability", 0.5)
        unc = data.get("uncertainty", 0.0)

        if uncertainty and unc > 0:
            # Color by uncertainty: more blue = more uncertain
            norm_unc = min(unc / 0.5, 1.0)  # Normalize uncertainty
            color = (0.2 + 0.8 * (1 - norm_unc), 0.2 + 0.8 * norm_unc, 1.0)
        else:
            # Color by probability
            color = cmap(prob)

        edge_colors.append(color)
        edge_widths.append(2 + 4 * (prob - 0.3) / 0.7)  # Width by probability

        if show_edge_labels:
            if uncertainty and std is not None:
                edge_labels[(i, j)] = f"P={prob:.2f}±{unc:.2f}"
            else:
                edge_labels[(i, j)] = f"{prob:.2f}"

    # Draw
    nx.draw_networkx_nodes(
        G, pos,
        node_color='lightblue',
        node_size=800,
        edgecolors='black',
        linewidths=1.5,
        ax=ax,
    )

    nx.draw_networkx_labels(
        G, pos,
        labels={i: f"{variable_names[i]}" for i in range(d)},
        font_size=10,
        ax=ax,
    )

    # Draw edges in two passes: low prob first, high prob on top
    sorted_edges = sorted(G.edges(data=True), key=lambda e: e[2].get("probability", 0))
    for i, j, data in sorted_edges:
        prob = data.get("probability", 0.5)
        unc = data.get("uncertainty", 0.0)
        color = edge_colors[list(G.edges).index((i, j))]

        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(i, j)],
            width=edge_widths[list(G.edges).index((i, j))],
            edge_color=[color],
            arrows=True,
            arrowsize=15,
            arrowstyle='-|>',
            alpha=min(0.3 + 0.7 * prob, 1.0),
            ax=ax,
            connectionstyle="arc3,rad=0.1",
        )

    if show_edge_labels and edge_labels:
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=8,
            ax=ax,
        )

    ax.set_title(title, fontsize=14)
    ax.axis('off')

    # Add color bar
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(0, 1),
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label("Edge Probability")

    if uncertainty and edge_stds is not None:
        # Add uncertainty note
        ax.text(
            0.5, -0.05,
            "Edge width ∝ probability | Color intensity ∝ certainty",
            transform=ax.transAxes,
            ha='center',
            fontsize=9,
            alpha=0.7,
        )

    plt.tight_layout()
    plt.show()


def plot_uncertainty_calibration(
    edge_probs: np.ndarray,
    true_edges: np.ndarray,
    ax=None,
    figsize: tuple = (8, 6),
):
    """Plot calibration curve for edge probabilities.

    Args:
        edge_probs: Predicted edge probabilities
        true_edges: True binary adjacency
        ax: Matplotlib axis (optional)
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    from causbayes.evaluation import edge_calibration

    cal = edge_calibration(true_edges, edge_probs)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax.plot(cal["bins"], cal["accuracy"], 'o-', linewidth=2, markersize=8)
    ax.fill_between(cal["bins"], 0, cal["accuracy"], alpha=0.1)

    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title(f"Edge Calibration (ECE = {cal['ece']:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
