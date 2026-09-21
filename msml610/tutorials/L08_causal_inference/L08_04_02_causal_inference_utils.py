"""
Utility functions for L08_04_02 causal inference tutorial.

Import as:

import L08_04_02_causal_inference_utils as mtl0cinut
"""

import logging
from typing import Any, Iterable, Optional

import ipywidgets
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import networkx as nx

import helpers.hnotebook as hnotebo

_LOG = logging.getLogger(__name__)


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utils logger.

    :param notebook_log: logger owned by the notebook
    """
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# Graph visualization
# #############################################################################


def _find_path_edges(
    graph: nx.Graph,
    node1: str,
    node2: str,
) -> set:
    """
    Find all edges that lie on any simple path between node1 and node2.

    Paths are searched on the underlying undirected graph so that both causal
    directions are considered.

    :param graph: directed graph
    :param node1: source node
    :param node2: target node
    :return: set of (u, v) edge tuples (directed, as they appear in graph)
    """
    undirected = graph.to_undirected()
    path_edges: set = set()
    if not (node1 in undirected and node2 in undirected):
        return path_edges
    for path in nx.all_simple_paths(undirected, node1, node2):
        for u, v in zip(path[:-1], path[1:]):
            # Store edge in directed form (either direction may exist).
            if graph.has_edge(u, v):
                path_edges.add((u, v))
            if graph.has_edge(v, u):
                path_edges.add((v, u))
    return path_edges


def plot_graph_highlight(
    graph: nx.Graph,
    *,
    node1: Optional[str] = None,
    node2: Optional[str] = None,
    conditioning_node_set: Optional[Iterable[str]] = None,
    layout: str = "shell",
    figsize: tuple = (6, 4),
) -> None:
    """
    Plot a graph with highlighted nodes and paths.

    node1 is green, node2 is blue, conditioning nodes are red.  When both
    node1 and node2 are provided, all edges on any simple path between them
    are drawn in orange.

    :param graph: directed graph to plot
    :param node1: node colored green
    :param node2: node colored blue
    :param conditioning_node_set: nodes colored red
    :param layout: layout algorithm ("shell", "spring", "kamada_kawai")
    :param figsize: figure size
    """
    conditioning_node_set = (
        set(conditioning_node_set) if conditioning_node_set else set()
    )
    plt.figure(figsize=figsize)
    # Choose layout.
    if layout == "spring":
        pos = nx.spring_layout(graph)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(graph)
    else:
        pos = nx.shell_layout(graph)
    # Assign node colors.
    node_colors = []
    for node in graph.nodes():
        if node == node1:
            node_colors.append("green")
        elif node == node2:
            node_colors.append("blue")
        elif node in conditioning_node_set:
            node_colors.append("red")
        else:
            node_colors.append("lightblue")
    # Highlight edges on all simple paths between node1 and node2.
    path_edges: set = set()
    if node1 and node2:
        path_edges = _find_path_edges(graph, node1, node2)
    edge_colors = [
        "orange" if (u, v) in path_edges else "gray" for u, v in graph.edges()
    ]
    edge_widths = [
        3.0 if (u, v) in path_edges else 1.0 for u, v in graph.edges()
    ]
    # Draw.
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=2000,
        font_size=12,
        font_weight="bold",
        edge_color=edge_colors,
        width=edge_widths,
        edgecolors="black",
        linewidths=2,
    )
    plt.show()


# #############################################################################
# Graph utilities
# #############################################################################


def reachable_subgraph(graph: nx.DiGraph, nodes: Iterable[str]) -> nx.Graph:
    """
    Return the subgraph containing all nodes reachable from the given nodes.

    :param graph: directed graph
    :param nodes: source nodes to start from (included in result)
    :return: subgraph of reachable nodes
    """
    reachable = set(nodes)
    for node in nodes:
        reachable |= nx.descendants(graph, node)
    return graph.subgraph(reachable).copy()


# #############################################################################
# Cell 1: Causal Roles Explorer
# #############################################################################


# Predefined graphs for causal role demonstration.
_PREDEFINED_GRAPHS: dict = {
    "Confounder": {
        "edges": [("Z", "X"), ("Z", "Y")],
        "default_treatment": "X",
        "default_outcome": "Y",
    },
    "Mediator": {
        "edges": [("X", "M"), ("M", "Y"), ("X", "Y")],
        "default_treatment": "X",
        "default_outcome": "Y",
    },
    "Collider": {
        "edges": [("X", "C"), ("Y", "C"), ("A", "X")],
        "default_treatment": "X",
        "default_outcome": "Y",
    },
    "Confounder + Mediator": {
        "edges": [("Z", "X"), ("Z", "Y"), ("X", "M"), ("M", "Y")],
        "default_treatment": "X",
        "default_outcome": "Y",
    },
    "Fork + Collider": {
        "edges": [("A", "X"), ("A", "Y"), ("X", "Z"), ("Y", "Z")],
        "default_treatment": "X",
        "default_outcome": "Y",
    },
    "Complex (original)": {
        "edges": [
            ("C", "A"),
            ("C", "B"),
            ("D", "A"),
            ("B", "E"),
            ("F", "E"),
            ("A", "G"),
        ],
        "default_treatment": "D",
        "default_outcome": "G",
    },
}


def _classify_causal_roles(
    graph: nx.DiGraph,
    treatment: str,
    outcome: str,
) -> tuple:
    """
    Classify graph nodes relative to a treatment-outcome pair.

    A node is classified as:
    - Confounder: common ancestor of both treatment and outcome
    - Mediator: on a directed path from treatment to outcome (excluding endpoints)
    - Collider: receives arrows from both path-adjacent nodes on some undirected
      path between treatment and outcome

    :param graph: directed graph
    :param treatment: treatment node name
    :param outcome: outcome node name
    :return: (confounders, mediators, colliders) as sets of node names
    """
    confounders: set = set()
    mediators: set = set()
    colliders: set = set()
    if treatment not in graph or outcome not in graph:
        return confounders, mediators, colliders
    # Confounders: common ancestors of both treatment and outcome.
    treatment_ancestors = nx.ancestors(graph, treatment)
    outcome_ancestors = nx.ancestors(graph, outcome)
    confounders = treatment_ancestors & outcome_ancestors
    # Mediators: intermediate nodes on directed paths from treatment to outcome.
    for path in nx.all_simple_paths(graph, treatment, outcome):
        for node in path[1:-1]:
            mediators.add(node)
    # Colliders: nodes where both path-neighbors point into the node.
    undirected = graph.to_undirected()
    if treatment in undirected and outcome in undirected:
        for path in nx.all_simple_paths(undirected, treatment, outcome):
            for i in range(1, len(path) - 1):
                prev_node = path[i - 1]
                curr_node = path[i]
                next_node = path[i + 1]
                if graph.has_edge(prev_node, curr_node) and graph.has_edge(
                    next_node, curr_node
                ):
                    colliders.add(curr_node)
    return confounders, mediators, colliders


def _plot_causal_roles(
    graph: nx.DiGraph,
    treatment: str,
    outcome: str,
    *,
    layout: str = "shell",
    figsize: tuple = (8, 5),
) -> None:
    """
    Plot a graph with nodes colored by their causal role.

    Color coding:
    - Treatment: green
    - Outcome: blue
    - Confounder: orange
    - Mediator: purple
    - Collider: red
    - Other: lightblue

    :param graph: directed graph to plot
    :param treatment: treatment node name
    :param outcome: outcome node name
    :param layout: layout algorithm ("shell", "spring", "kamada_kawai")
    :param figsize: figure size
    """
    confounders, mediators, colliders = _classify_causal_roles(
        graph, treatment, outcome
    )
    # Assign colors based on causal role (priority: treatment > outcome >
    # collider > mediator > confounder > default).
    node_colors: list = []
    for node in graph.nodes():
        if node == treatment:
            node_colors.append("green")
        elif node == outcome:
            node_colors.append("blue")
        elif node in colliders:
            node_colors.append("red")
        elif node in mediators:
            node_colors.append("purple")
        elif node in confounders:
            node_colors.append("orange")
        else:
            node_colors.append("lightblue")
    # Choose layout.
    if layout == "spring":
        pos = nx.spring_layout(graph, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(graph)
    else:
        pos = nx.shell_layout(graph)
    _, ax = plt.subplots(figsize=figsize)
    nx.draw(
        graph,
        pos,
        ax=ax,
        with_labels=True,
        node_color=node_colors,
        node_size=2000,
        font_size=12,
        font_weight="bold",
        edge_color="gray",
        width=1.5,
        edgecolors="black",
        linewidths=2,
    )
    # Build legend with role summaries.
    legend_items = [
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="green",
            markersize=12,
            label=f"Treatment: {treatment}",
        ),
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markersize=12,
            label=f"Outcome: {outcome}",
        ),
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="orange",
            markersize=12,
            label=f"Confounders: {sorted(confounders) if confounders else 'none'}",
        ),
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="purple",
            markersize=12,
            label=f"Mediators: {sorted(mediators) if mediators else 'none'}",
        ),
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markersize=12,
            label=f"Colliders: {sorted(colliders) if colliders else 'none'}",
        ),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=9)
    ax.set_title(
        f"Causal roles: Treatment={treatment}, Outcome={outcome}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


def cell1_causal_roles_explorer() -> ipywidgets.VBox:
    """
    Build an interactive widget for exploring causal roles in a DAG.

    The widget lets the user select a predefined graph, a treatment node, and an
    outcome node, then highlights which nodes are confounders, mediators, or
    colliders relative to that treatment-outcome pair.

    :return: VBox widget ready to display in a notebook
    """
    graph_names = list(_PREDEFINED_GRAPHS.keys())
    first_graph_info = _PREDEFINED_GRAPHS[graph_names[0]]
    first_graph = nx.DiGraph(first_graph_info["edges"])
    first_nodes = sorted(first_graph.nodes())
    # Graph selector dropdown.
    graph_selector = ipywidgets.Dropdown(
        options=graph_names,
        value=graph_names[0],
        description="Graph:",
        style={"description_width": "70px"},
        layout={"width": "280px"},
    )
    # Treatment and outcome node selectors.
    treatment_selector = ipywidgets.Dropdown(
        options=first_nodes,
        value=first_graph_info["default_treatment"],
        description="Treatment:",
        style={"description_width": "80px"},
        layout={"width": "200px"},
    )
    outcome_selector = ipywidgets.Dropdown(
        options=first_nodes,
        value=first_graph_info["default_outcome"],
        description="Outcome:",
        style={"description_width": "80px"},
        layout={"width": "200px"},
    )
    show_button = ipywidgets.Button(
        description="Show",
        button_style="primary",
        layout={"width": "100px"},
    )
    output = ipywidgets.Output()

    def _update_node_selectors(change: dict) -> None:
        """
        Repopulate treatment/outcome dropdowns when the graph changes.
        """
        graph_info = _PREDEFINED_GRAPHS[graph_selector.value]
        graph = nx.DiGraph(graph_info["edges"])
        nodes = sorted(graph.nodes())
        treatment_selector.options = nodes
        treatment_selector.value = graph_info["default_treatment"]
        outcome_selector.options = nodes
        outcome_selector.value = graph_info["default_outcome"]

    graph_selector.observe(_update_node_selectors, names="value")

    def _on_show_clicked(
        _button: Optional[ipywidgets.Button],
    ) -> None:
        """
        Render the causal roles plot when the Show button is clicked.
        """
        graph_info = _PREDEFINED_GRAPHS[graph_selector.value]
        graph = nx.DiGraph(graph_info["edges"])
        treatment = treatment_selector.value
        outcome = outcome_selector.value
        output.clear_output(wait=True)
        with output:
            _plot_causal_roles(graph, treatment, outcome)

    show_button.on_click(_on_show_clicked)
    controls = ipywidgets.HBox(
        [
            graph_selector,
            treatment_selector,
            outcome_selector,
            show_button,
        ]
    )
    # Render the plot automatically on first display.
    _on_show_clicked(None)
    return ipywidgets.VBox([controls, output])


# #############################################################################
# Cell 3: Interactive D-Separation Explorer
# #############################################################################


def cell3_d_separation_explorer(
    model: nx.DiGraph,
    dag: Any,
    *,
    default_node1: Optional[str] = None,
    default_node2: Optional[str] = None,
    default_conditioning: Optional[Iterable[str]] = None,
) -> ipywidgets.VBox:
    """
    Build an interactive widget for exploring d-separation in a DAG.

    Given two nodes and a conditioning set, the widget plots the reachable
    subgraph with highlighted nodes, highlights all paths between the two nodes,
    and reports whether they are d-connected given the conditioning set.

    :param model: directed graph representing the causal model
    :param dag: pgmpy DAG used for d-separation queries
    :param default_node1: initial value for node 1 dropdown
    :param default_node2: initial value for node 2 dropdown
    :param default_conditioning: initial selection for conditioning nodes
    :return: VBox widget ready to display in a notebook
    """
    if not default_node1:
        default_node1 = "D"
    if not default_node2:
        default_node2 = "C"
    if not default_conditioning:
        default_conditioning = ["A"]
    all_nodes = sorted(model.nodes())
    # Set defaults if not provided.
    node1_default = default_node1 if default_node1 in all_nodes else all_nodes[0]
    node2_default = default_node2 if default_node2 in all_nodes else all_nodes[1]
    cond_default = [n for n in (default_conditioning or []) if n in all_nodes]
    # Build selection widgets.
    node1_widget = ipywidgets.Dropdown(
        options=all_nodes,
        value=node1_default,
        description="Node 1:",
        style={"description_width": "80px"},
        layout={"width": "250px"},
    )
    node2_widget = ipywidgets.Dropdown(
        options=all_nodes,
        value=node2_default,
        description="Node 2:",
        style={"description_width": "80px"},
        layout={"width": "250px"},
    )
    cond_widget = ipywidgets.SelectMultiple(
        options=all_nodes,
        value=cond_default,
        description="Conditioning:",
        style={"description_width": "90px"},
        layout={"width": "250px", "height": "160px"},
    )
    run_button = ipywidgets.Button(
        description="Run",
        button_style="primary",
        layout={"width": "100px"},
    )
    output = ipywidgets.Output()

    def _on_run_clicked(
        _button: Optional[ipywidgets.Button],
    ) -> None:
        """
        Update plot and d-separation result when the Run button is clicked.
        """
        node1 = node1_widget.value
        node2 = node2_widget.value
        conditioning_node_set = list(cond_widget.value)
        output.clear_output(wait=True)
        with output:
            # Compute subgraph including selected nodes and their descendants.
            nodes_of_interest = [node1, node2] + conditioning_node_set
            subgraph = reachable_subgraph(model, nodes_of_interest)
            plot_graph_highlight(
                subgraph,
                node1=node1,
                node2=node2,
                conditioning_node_set=conditioning_node_set,
            )
            # Report d-separation result.
            observed = (
                set(conditioning_node_set) if conditioning_node_set else set()
            )
            is_dependent = dag.is_dconnected(node1, node2, observed=observed)
            cond_str = (
                "given {" + ", ".join(sorted(observed)) + "}"
                if observed
                else "unconditionally"
            )
            print(
                f"Are {node1} and {node2} dependent {cond_str}? {is_dependent}"
            )

    run_button.on_click(_on_run_clicked)
    # Assemble controls and output.
    controls = ipywidgets.VBox(
        [
            ipywidgets.HBox([node1_widget, node2_widget]),
            cond_widget,
            run_button,
        ]
    )
    # Render the plot automatically on first display.
    _on_run_clicked(None)
    return ipywidgets.VBox([controls, output])
