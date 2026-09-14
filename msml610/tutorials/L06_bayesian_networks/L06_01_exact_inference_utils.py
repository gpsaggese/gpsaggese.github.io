"""
Utility functions for the exact inference notebook.

Implements the interactive widgets, visualizations, and exact-inference
computations for the canonical AIMA burglary-alarm Bayesian network.

Import as:

import msml610.tutorials.L06_bayesian_networks.L06_01_exact_inference_utils as mtlbnl0eiu
"""

import itertools
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import ipywidgets
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import clear_output, display

import pgmpy.factors.discrete as pgfactors
import pgmpy.inference as pginference
import pgmpy.models as pgmodels

import helpers.hdbg as hdbg
import helpers.hgraphviz as hgraphv
import helpers.hnotebook as hnotebo
import helpers.htutorial as htutori

_LOG = logging.getLogger(__name__)

# Silence noisy library warnings so the notebook output stays clean.
warnings.filterwarnings("ignore", category=FutureWarning)


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger to the utils logger.

    :param notebook_log: Logger created in the notebook
    """
    global _LOG
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# Burglary-alarm network definition (shared by all cells).
# #############################################################################

# Standard AIMA burglary-alarm probabilities.
# State convention everywhere: 0 = False, 1 = True.
_P_B = 0.001
_P_E = 0.002
# P(Alarm=True | Burglary, Earthquake) for the four parent combinations.
_P_A_GIVEN_BE = {
    (0, 0): 0.001,
    (0, 1): 0.29,
    (1, 0): 0.94,
    (1, 1): 0.95,
}
# P(JohnCalls=True | Alarm).
_P_J_GIVEN_A = {0: 0.05, 1: 0.90}
# P(MaryCalls=True | Alarm).
_P_M_GIVEN_A = {0: 0.01, 1: 0.70}

# Canonical node names and structure of the network.
_NODES = ["Burglary", "Earthquake", "Alarm", "JohnCalls", "MaryCalls"]
_EDGES = [
    ("Burglary", "Alarm"),
    ("Earthquake", "Alarm"),
    ("Alarm", "JohnCalls"),
    ("Alarm", "MaryCalls"),
]
# Short math symbols used in equations and compact table headers.
_SHORT = {
    "Burglary": "B",
    "Earthquake": "E",
    "Alarm": "A",
    "JohnCalls": "J",
    "MaryCalls": "M",
}
# Color scheme by role in the causal story: root causes, the alarm, the calls.
_COLOR_CAUSE = "#A6C8F4"
_COLOR_ALARM = "#FFD27F"
_COLOR_CALL = "#D8B4F0"
_ROLE_COLORS = {
    "Burglary": _COLOR_CAUSE,
    "Earthquake": _COLOR_CAUSE,
    "Alarm": _COLOR_ALARM,
    "JohnCalls": _COLOR_CALL,
    "MaryCalls": _COLOR_CALL,
}


def _build_burglary_network() -> "pgmodels.DiscreteBayesianNetwork":
    """
    Build the AIMA burglary-alarm network as a `pgmpy` model.

    :return: Bayesian network with all five CPTs attached
    """
    model = pgmodels.DiscreteBayesianNetwork(_EDGES)
    # Priors on the two root causes.
    cpd_b = pgfactors.TabularCPD("Burglary", 2, [[1 - _P_B], [_P_B]])
    cpd_e = pgfactors.TabularCPD("Earthquake", 2, [[1 - _P_E], [_P_E]])
    # P(Alarm | Burglary, Earthquake): pgmpy orders columns with the last
    # evidence variable changing fastest, i.e. (B,E) in (0,0),(0,1),(1,0),(1,1).
    a_true = [_P_A_GIVEN_BE[(b, e)] for b in (0, 1) for e in (0, 1)]
    cpd_a = pgfactors.TabularCPD(
        "Alarm",
        2,
        [[1 - p for p in a_true], list(a_true)],
        evidence=["Burglary", "Earthquake"],
        evidence_card=[2, 2],
    )
    # The two call nodes depend only on Alarm.
    j_true = [_P_J_GIVEN_A[a] for a in (0, 1)]
    cpd_j = pgfactors.TabularCPD(
        "JohnCalls",
        2,
        [[1 - p for p in j_true], list(j_true)],
        evidence=["Alarm"],
        evidence_card=[2],
    )
    m_true = [_P_M_GIVEN_A[a] for a in (0, 1)]
    cpd_m = pgfactors.TabularCPD(
        "MaryCalls",
        2,
        [[1 - p for p in m_true], list(m_true)],
        evidence=["Alarm"],
        evidence_card=[2],
    )
    model.add_cpds(cpd_b, cpd_e, cpd_a, cpd_j, cpd_m)
    hdbg.dassert(model.check_model(), "Burglary network is not a valid model")
    return model


def _joint_product(assignment: Dict[str, int]) -> float:
    """
    Evaluate the factored joint probability for a full assignment.

    Computes the product of the five CPTs:
    P(B) P(E) P(A | B,E) P(J | A) P(M | A).

    :param assignment: Map from each node name to its value (0 or 1)
    :return: Joint probability of the assignment
    """
    b = assignment["Burglary"]
    e = assignment["Earthquake"]
    a = assignment["Alarm"]
    j = assignment["JohnCalls"]
    m = assignment["MaryCalls"]
    # Look up each CPT entry, flipping to the False branch when value is 0.
    p_b = _P_B if b == 1 else 1 - _P_B
    p_e = _P_E if e == 1 else 1 - _P_E
    pa = _P_A_GIVEN_BE[(b, e)]
    p_a = pa if a == 1 else 1 - pa
    pj = _P_J_GIVEN_A[a]
    p_j = pj if j == 1 else 1 - pj
    pm = _P_M_GIVEN_A[a]
    p_m = pm if m == 1 else 1 - pm
    return p_b * p_e * p_a * p_j * p_m


def _enumerate_posterior(
    query_var: str, evidence: Dict[str, int]
) -> Tuple[Dict[int, float], Dict[int, float], List[str], Dict[int, List[Any]]]:
    """
    Compute a posterior by brute-force enumeration over hidden variables.

    For each value of the query variable, sums the joint over every hidden
    assignment, then normalizes across query values.

    :param query_var: Name of the query variable X
    :param evidence: Map from observed node names to their values
    :return: Tuple of
        - posterior: map from query value to P(X=value | evidence)
        - unnorm: map from query value to the unnormalized P(X=value, evidence)
        - hidden: list of hidden variable names that were summed out
        - rows_by_qval: per query value, list of (hidden combo, product) rows
    """
    # Hidden variables are everything that is neither query nor evidence.
    hidden = [n for n in _NODES if n != query_var and n not in evidence]
    unnorm = {}
    rows_by_qval = {}
    for qval in (0, 1):
        total = 0.0
        rows = []
        # Enumerate every joint assignment of the hidden variables.
        for combo in itertools.product([0, 1], repeat=len(hidden)):
            assignment = dict(evidence)
            assignment[query_var] = qval
            for h, v in zip(hidden, combo):
                assignment[h] = v
            p = _joint_product(assignment)
            rows.append((combo, p))
            total += p
        unnorm[qval] = total
        rows_by_qval[qval] = rows
    # Normalize so the posterior over the query variable sums to 1.
    z = unnorm[0] + unnorm[1]
    posterior = {0: unnorm[0] / z, 1: unnorm[1] / z}
    return posterior, unnorm, hidden, rows_by_qval


def _fmt_evidence(evidence: Dict[str, int]) -> str:
    """
    Format an evidence dict as a compact human-readable string.

    :param evidence: Map from observed node names to their values
    :return: String like "J=T, M=T" using short symbols
    """
    if not evidence:
        return "(none)"
    parts = []
    for n in _NODES:
        if n in evidence:
            tag = "T" if evidence[n] == 1 else "F"
            parts.append(f"{_SHORT[n]}={tag}")
    return ", ".join(parts)


def _build_query_controls(
    *,
    default_query: str = "Burglary",
    default_evidence: Optional[Dict[str, int]] = None,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any], List[Any]]:
    """
    Build the shared query-and-evidence control widgets.

    Creates a dropdown to pick the query variable and, for every node, a checkbox
    marking it as observed plus a True/False value selector.

    :param default_query: Query variable selected initially
    :param default_evidence: Initial evidence assignment
    :return: Tuple of (query dropdown, checkboxes, value dropdowns, layout rows)
    """
    if default_evidence is None:
        default_evidence = {"JohnCalls": 1, "MaryCalls": 1}
    query_dd = ipywidgets.Dropdown(
        options=_NODES,
        value=default_query,
        description="Query X:",
        style={"description_width": "initial"},
    )
    checks = {}
    valdds = {}
    rows = []
    # One observed-checkbox plus a True/False value picker per node.
    for n in _NODES:
        cb = ipywidgets.Checkbox(
            value=(n in default_evidence),
            description=f"observe {n}",
            indent=False,
            layout={"width": "180px"},
        )
        vd = ipywidgets.Dropdown(
            options=[("True", 1), ("False", 0)],
            value=default_evidence.get(n, 1),
            description="",
            layout={"width": "90px"},
        )
        checks[n] = cb
        valdds[n] = vd
        rows.append(ipywidgets.HBox([cb, vd]))
    return query_dd, checks, valdds, rows


def _read_query(
    query_dd: Any, checks: Dict[str, Any], valdds: Dict[str, Any]
) -> Tuple[str, Dict[str, int]]:
    """
    Read the current query variable and evidence from the control widgets.

    The query variable can never also be evidence, so it is skipped.

    :param query_dd: Query variable dropdown
    :param checks: Per-node observed checkboxes
    :param valdds: Per-node value dropdowns
    :return: Tuple of (query variable, evidence map)
    """
    query_var = query_dd.value
    evidence = {}
    for n in _NODES:
        if n == query_var:
            continue
        if checks[n].value:
            evidence[n] = valdds[n].value
    return query_var, evidence


def _factor_to_df(factor: "pgfactors.DiscreteFactor") -> pd.DataFrame:
    """
    Convert a `pgmpy` discrete factor into a tidy DataFrame.

    Each row is one assignment of the factor's variables together with the
    factor value `phi`.

    :param factor: Discrete factor to render
    :return: DataFrame with one column per variable plus a `phi` column
    """
    # Build every assignment in the variable order used by `factor.values`.
    ranges = [range(c) for c in factor.cardinality]
    data = []
    for assignment in itertools.product(*ranges):
        row = {}
        for var, val in zip(factor.variables, assignment):
            row[_SHORT.get(var, var)] = "T" if val == 1 else "F"
        row["phi"] = float(factor.values[assignment])
        data.append(row)
    return pd.DataFrame(data)


# #############################################################################
# Cell 1.1: The Burglary Alarm Network and Its CPTs
# #############################################################################


def cell1_1_show_network_and_cpts(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Display the burglary-alarm DAG and its five conditional probability tables.

    Reference cell that anchors every later inference step in a concrete
    structure and set of numbers.

    :param figsize: Optional figure size for the DAG plot
    """
    if figsize is None:
        figsize = (7, 5)
    # Draw the causal DAG, colored by role in the story.
    graph = nx.DiGraph(_EDGES)
    hgraphv.plot_causal_dag(
        graph,
        "Burglary-alarm network",
        node_colors=_ROLE_COLORS,
        figsize=figsize,
    )
    plt.show()
    # Show the two priors as a single small table.
    priors_df = pd.DataFrame(
        {
            "Variable": ["Burglary", "Earthquake"],
            "P(True)": [_P_B, _P_E],
            "P(False)": [1 - _P_B, 1 - _P_E],
        }
    )
    print("Priors on the root causes:")
    display(priors_df)
    # Show P(Alarm | Burglary, Earthquake) over the four parent combinations.
    alarm_rows = []
    for b in (0, 1):
        for e in (0, 1):
            p = _P_A_GIVEN_BE[(b, e)]
            alarm_rows.append(
                {
                    "Burglary": "T" if b else "F",
                    "Earthquake": "T" if e else "F",
                    "P(Alarm=T)": p,
                    "P(Alarm=F)": 1 - p,
                }
            )
    print("P(Alarm | Burglary, Earthquake):")
    display(pd.DataFrame(alarm_rows))
    # Show the two call CPTs, which depend only on Alarm.
    calls_df = pd.DataFrame(
        {
            "Alarm": ["T", "F"],
            "P(JohnCalls=T)": [_P_J_GIVEN_A[1], _P_J_GIVEN_A[0]],
            "P(MaryCalls=T)": [_P_M_GIVEN_A[1], _P_M_GIVEN_A[0]],
        }
    )
    print("P(JohnCalls | Alarm) and P(MaryCalls | Alarm):")
    display(calls_df)


# #############################################################################
# Cell 1.2: Query, Evidence, and Hidden Variables
# #############################################################################


def cell1_2_query_roles_widget(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Interactively assign each node a role: query, evidence, or hidden.

    Recolors the DAG by role for the current query and restates the query in
    math so students see how the evidence choice changes what is hidden.

    :param figsize: Optional figure size for the DAG plot
    """
    if figsize is None:
        figsize = (8, 5)
    query_dd, checks, valdds, rows = _build_query_controls()
    graph = nx.DiGraph(_EDGES)
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw the role-colored DAG when any control changes.
        """
        _ = change
        with output:
            clear_output(wait=True)
            query_var, evidence = _read_query(query_dd, checks, valdds)
            hidden = [n for n in _NODES if n != query_var and n not in evidence]
            # Color each node by its role in the current query.
            node_colors = {}
            for n in _NODES:
                if n == query_var:
                    node_colors[n] = "#2E86C1"
                elif n in evidence:
                    node_colors[n] = "#F1948A"
                else:
                    node_colors[n] = "#D5D8DC"
            _, (ax1, ax2) = plt.subplots(
                1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.4, 1]}
            )
            hgraphv.plot_causal_dag(
                graph,
                "Roles for the current query",
                node_colors=node_colors,
                ax=ax1,
            )
            # Summarize the partition as text in the second panel.
            ax2.axis("off")
            ax2.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
            query_str = f"P({_SHORT[query_var]} | {_fmt_evidence(evidence)})"
            hidden_str = (
                ", ".join(_SHORT[h] for h in hidden) if hidden else "(none)"
            )
            text_content = (
                f"Query X:\n  {query_var}\n\n"
                f"Evidence e:\n  {_fmt_evidence(evidence)}\n\n"
                f"Hidden Y:\n  {hidden_str}\n\n"
                f"We want:\n  {query_str}\n\n"
                f"Hidden variables are summed\n"
                f"out: 2^{len(hidden)} = "
                f"{2 ** len(hidden)} terms."
            )
            htutori.add_fitted_text_box(
                ax2, text_content, max_fontsize=13, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    # Observe every control so the plot stays in sync.
    query_dd.observe(update_plot, names="value")
    for n in _NODES:
        checks[n].observe(update_plot, names="value")
        valdds[n].observe(update_plot, names="value")
    update_plot()
    display(
        ipywidgets.VBox(
            [
                ipywidgets.Label(
                    "Choose the query variable and which nodes are observed:"
                ),
                query_dd,
                *rows,
                output,
            ]
        )
    )


# #############################################################################
# Cell 2.1: From Conditional to Joint via Normalization
# #############################################################################


def cell2_1_normalization_widget(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Show why a conditional query reduces to summing the joint and normalizing.

    Reveals the derivation as three equation lines and lets students toggle
    between the unnormalized joint values and the normalized posterior.

    :param figsize: Optional figure size
    """
    if figsize is None:
        figsize = (20, 5)
    # Use the canonical query P(Burglary | JohnCalls=T, MaryCalls=T).
    query_var = "Burglary"
    evidence = {"JohnCalls": 1, "MaryCalls": 1}
    posterior, unnorm, _, _ = _enumerate_posterior(query_var, evidence)
    normalize_toggle = ipywidgets.ToggleButton(
        value=False,
        description="show normalization",
        tooltip="Switch between unnormalized joint and normalized posterior",
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw equation panels and the value bars on toggle.
        """
        _ = change
        with output:
            clear_output(wait=True)
            show_norm = normalize_toggle.value
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            # Panel 1: the three derivation equations as math text.
            ax1.axis("off")
            ax1.set_title("Derivation", fontsize=14, fontweight="bold", pad=20)
            equations = [
                r"$P(X \mid e) = \alpha\, P(X, e)$",
                r"$P(X, e) = \sum_{y} P(X, e, y)$",
                r"$P(b \mid j,m) = \alpha \sum_{e}\sum_{a}$"
                "\n"
                r"$P(b)P(e)P(a|b,e)P(j|a)P(m|a)$",
            ]
            y_pos = [0.82, 0.58, 0.30]
            for eq, y in zip(equations, y_pos):
                ax1.text(
                    0.05,
                    y,
                    eq,
                    transform=ax1.transAxes,
                    fontsize=15,
                    va="center",
                    ha="left",
                )
            # Panel 2: the value bars, unnormalized or normalized.
            if show_norm:
                values = [posterior[1], posterior[0]]
                title = "Normalized posterior (sums to 1)"
                ylabel = "P(B | j,m)"
            else:
                values = [unnorm[1], unnorm[0]]
                title = "Unnormalized joint P(B, j,m)"
                ylabel = "P(B, j,m)"
            bar_df = pd.DataFrame({"B": ["B=T", "B=F"], "value": values})
            sns.barplot(
                data=bar_df,
                x="B",
                y="value",
                hue="B",
                palette=["#2E86C1", "#AED6F1"],
                legend=False,
                ax=ax2,
                edgecolor="black",
            )
            ax2.set_title(title, fontsize=14, fontweight="bold")
            ax2.set_ylabel(ylabel, fontsize=12)
            ax2.set_xlabel("")
            for i, v in enumerate(values):
                ax2.text(i, v, f"{v:.4g}", ha="center", va="bottom", fontsize=11)
            # Panel 3: comments explaining alpha.
            ax3.axis("off")
            ax3.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
            z = unnorm[0] + unnorm[1]
            text_content = (
                f"Unnormalized values:\n"
                f"  P(b, j,m)  = {unnorm[1]:.3e}\n"
                f"  P(~b, j,m) = {unnorm[0]:.3e}\n\n"
                f"Normalization:\n"
                f"  sum = {z:.3e}\n"
                f"  alpha = 1 / sum\n"
                f"        = {1 / z:.2f}\n\n"
                f"Posterior:\n"
                f"  P(b | j,m)  = {posterior[1]:.4f}\n"
                f"  P(~b | j,m) = {posterior[0]:.4f}\n\n"
                f"alpha removes the need to\n"
                f"compute P(e) directly."
            )
            htutori.add_fitted_text_box(
                ax3, text_content, max_fontsize=13, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    normalize_toggle.observe(update_plot, names="value")
    update_plot()
    display(
        ipywidgets.VBox(
            [
                ipywidgets.Label(
                    "Toggle to rescale the joint values into a posterior:"
                ),
                normalize_toggle,
                output,
            ]
        )
    )


# #############################################################################
# Cell 2.2: Computing the Posterior by Enumeration
# #############################################################################


def cell2_2_enumeration_widget(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Compute the posterior by enumeration and check it against `pgmpy`.

    Shows the summation tables for both query values, the resulting posterior
    bar chart with the `pgmpy` reference overlaid, and a comments panel.

    :param figsize: Optional figure size
    """
    if figsize is None:
        figsize = (22, 5)
    model = _build_burglary_network()
    infer = pginference.VariableElimination(model)
    query_dd, checks, valdds, rows = _build_query_controls()
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Recompute the enumeration tables and posterior on any change.
        """
        _ = change
        with output:
            clear_output(wait=True)
            query_var, evidence = _read_query(query_dd, checks, valdds)
            posterior, unnorm, hidden, rows_by_qval = _enumerate_posterior(
                query_var, evidence
            )
            # The pgmpy reference posterior for validation.
            ref = infer.query(
                [query_var], evidence=evidence, show_progress=False
            )
            sym = _SHORT[query_var]
            _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)

            # Helper to render a summation table for a query value.
            def _fill_table(ax: Any, qval: int) -> None:
                tag = "T" if qval == 1 else "F"
                ax.axis("off")
                ax.set_title(
                    f"Sum for {sym}={tag}",
                    fontsize=13,
                    fontweight="bold",
                    pad=10,
                )
                # Build a small table of hidden assignments and products.
                table_rows = []
                for combo, p in rows_by_qval[qval]:
                    label = ", ".join(
                        f"{_SHORT[h]}={'T' if v else 'F'}"
                        for h, v in zip(hidden, combo)
                    )
                    if not label:
                        label = "(no hidden)"
                    table_rows.append([label, f"{p:.3e}"])
                table_rows.append(["sum", f"{unnorm[qval]:.3e}"])
                tbl = ax.table(
                    cellText=table_rows,
                    colLabels=["hidden assignment", "product"],
                    loc="center",
                    cellLoc="center",
                )
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(9)
                tbl.scale(1, 1.3)

            _fill_table(ax1, 1)
            _fill_table(ax2, 0)
            # Panel 3: posterior bars with pgmpy reference overlaid.
            computed = [posterior[1], posterior[0]]
            reference = [ref.values[1], ref.values[0]]
            x = np.arange(2)
            width = 0.55
            ax3.bar(
                x,
                computed,
                width,
                color=["#2E86C1", "#AED6F1"],
                edgecolor="black",
                label="enumeration",
            )
            ax3.bar(
                x,
                reference,
                width * 0.5,
                color="none",
                edgecolor="red",
                linestyle=":",
                linewidth=2,
                label="pgmpy reference",
            )
            ax3.set_xticks(x)
            ax3.set_xticklabels([f"{sym}=T", f"{sym}=F"])
            ax3.set_ylim([0, 1.05])
            ax3.set_ylabel(f"P({sym} | e)", fontsize=12)
            ax3.set_title("Posterior", fontsize=13, fontweight="bold")
            ax3.legend(fontsize=10)
            for i, v in enumerate(computed):
                ax3.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
            # Panel 4: comments.
            ax4.axis("off")
            ax4.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
            text_content = (
                f"Query: P({sym} | {_fmt_evidence(evidence)})\n\n"
                f"Hidden variables: {len(hidden)}\n"
                f"Rows summed per value:\n"
                f"  2^{len(hidden)} = {2 ** len(hidden)}\n\n"
                f"Posterior:\n"
                f"  P({sym}=T | e) = {posterior[1]:.4f}\n"
                f"  P({sym}=F | e) = {posterior[0]:.4f}\n\n"
                f"Matches pgmpy:\n"
                f"  {np.allclose(computed, reference)}\n\n"
                f"Row count doubles with each\n"
                f"extra hidden variable."
            )
            htutori.add_fitted_text_box(
                ax4, text_content, max_fontsize=12, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    query_dd.observe(update_plot, names="value")
    for n in _NODES:
        checks[n].observe(update_plot, names="value")
        valdds[n].observe(update_plot, names="value")
    update_plot()
    display(
        ipywidgets.VBox(
            [
                ipywidgets.Label(
                    "Set the query and evidence; watch the summed rows:"
                ),
                query_dd,
                *rows,
                output,
            ]
        )
    )


# #############################################################################
# Cell 2.3: Visualizing the Enumeration Tree
# #############################################################################


def cell2_3_enumeration_tree_widget(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Draw the enumeration computation as a tree to expose repeated work.

    Fixes the canonical query P(Burglary | j,m) with hidden Earthquake and Alarm,
    branches over their values in a chosen order, and highlights the repeated
    leaf subexpressions that motivate variable elimination.

    :param figsize: Optional figure size
    """
    if figsize is None:
        figsize = (20, 6)
    order_dd = ipywidgets.Dropdown(
        options=[
            ("Earthquake first, then Alarm", ("Earthquake", "Alarm")),
            ("Alarm first, then Earthquake", ("Alarm", "Earthquake")),
        ],
        value=("Earthquake", "Alarm"),
        description="Sum order:",
        style={"description_width": "initial"},
    )
    highlight_toggle = ipywidgets.ToggleButton(
        value=True,
        description="highlight repeated work",
    )
    output = ipywidgets.Output()
    # Fix the running example: query Burglary=True, observe both calls.
    query_val = 1
    evidence = {"JohnCalls": 1, "MaryCalls": 1}

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Rebuild the enumeration tree for the chosen summation order.
        """
        _ = change
        with output:
            clear_output(wait=True)
            order = order_dd.value
            highlight = highlight_toggle.value
            _, (ax1, ax2) = plt.subplots(
                1, 2, figsize=figsize, gridspec_kw={"width_ratios": [2, 1]}
            )
            # Build the tree positions level by level over the two hidden vars.
            ax1.axis("off")
            ax1.set_xlim(0, 10)
            ax1.set_ylim(0, 10)
            ax1.set_title(
                "Enumeration evaluation tree",
                fontsize=14,
                fontweight="bold",
            )
            # Root node at the top.
            ax1.text(
                5,
                9.3,
                f"sum for B={'T' if query_val else 'F'}",
                ha="center",
                va="center",
                fontsize=11,
                bbox=dict(boxstyle="round", facecolor="#FDEBD0"),
            )
            # The leaf factor that recurs (calls depend only on Alarm).
            leaf_colors = {0: "#AED6F1", 1: "#F5B7B1"}
            first_var, second_var = order
            x_leaf = np.linspace(1.2, 8.8, 4)
            idx = 0
            for v1 in (0, 1):
                # First-level branch over the first summed variable.
                x1 = 2.5 + 5 * v1
                ax1.plot([5, x1], [9.0, 6.7], color="gray", linewidth=1)
                ax1.text(
                    x1,
                    6.5,
                    f"{_SHORT[first_var]}={'T' if v1 else 'F'}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="#D6EAF8"),
                )
                for v2 in (0, 1):
                    # Second-level branch over the second summed variable.
                    x2 = x_leaf[idx]
                    ax1.plot([x1, x2], [6.3, 3.8], color="gray", linewidth=1)
                    ax1.text(
                        x2,
                        3.6,
                        f"{_SHORT[second_var]}={'T' if v2 else 'F'}",
                        ha="center",
                        va="center",
                        fontsize=10,
                        bbox=dict(boxstyle="round", facecolor="#D6EAF8"),
                    )
                    # The repeated call factor depends on the Alarm value.
                    a_val = v1 if first_var == "Alarm" else v2
                    face = leaf_colors[a_val] if highlight else "#EAECEE"
                    leaf_txt = "P(j|a)P(m|a)"
                    ax1.text(
                        x2,
                        1.6,
                        leaf_txt,
                        ha="center",
                        va="center",
                        fontsize=8,
                        bbox=dict(boxstyle="round", facecolor=face),
                    )
                    ax1.plot([x2, x2], [3.3, 2.1], color="gray", linewidth=1)
                    idx += 1
            # Count operations: products and additions over the leaves.
            n_leaves = 4
            n_factors = 5
            multiplications = n_leaves * (n_factors - 1)
            additions = n_leaves - 1
            # Panel 2: operation counts and explanation.
            ax2.axis("off")
            ax2.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
            text_content = (
                f"Hidden variables: E, A\n"
                f"Leaves: 2^2 = {n_leaves}\n\n"
                f"Operation count:\n"
                f"  multiplications: {multiplications}\n"
                f"  additions: {additions}\n\n"
                f"Same-colored leaves share\n"
                f"the factor P(j|a)P(m|a):\n"
                f"enumeration recomputes it\n"
                f"under every value of the\n"
                f"other variable.\n\n"
                f"Caching that shared factor\n"
                f"is the one idea behind\n"
                f"variable elimination."
            )
            htutori.add_fitted_text_box(
                ax2, text_content, max_fontsize=13, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    order_dd.observe(update_plot, names="value")
    highlight_toggle.observe(update_plot, names="value")
    update_plot()
    display(
        ipywidgets.VBox(
            [
                ipywidgets.Label(
                    "Pick a summation order and highlight the repeated work:"
                ),
                order_dd,
                highlight_toggle,
                output,
            ]
        )
    )


# #############################################################################
# Cell 3.1: Factors as the Unit of Computation
# #############################################################################


def cell3_1_factor_operations_widget(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Illustrate the two factor operations: looking at a factor and summing out.

    Each CPT is shown as a labeled factor; a dropdown picks a variable to sum
    out, and the before/after tables show how the factor shrinks.

    :param figsize: Optional figure size
    """
    if figsize is None:
        figsize = (20, 5)
    model = _build_burglary_network()
    # Map a friendly factor label to its `pgmpy` factor.
    factor_map = {}
    for cpd in model.get_cpds():
        var = cpd.variable
        scope = ", ".join(_SHORT[v] for v in cpd.to_factor().variables)
        factor_map[f"f_{_SHORT[var]}  ({scope})"] = cpd.to_factor()
    factor_dd = ipywidgets.Dropdown(
        options=list(factor_map.keys()),
        value=list(factor_map.keys())[2],
        description="Factor:",
        style={"description_width": "initial"},
    )
    sumout_dd = ipywidgets.Dropdown(
        options=[],
        description="Sum out:",
        style={"description_width": "initial"},
    )
    output = ipywidgets.Output()

    def _refresh_sumout_options() -> None:
        """
        Repopulate the sum-out dropdown with the selected factor's scope.
        """
        factor = factor_map[factor_dd.value]
        sumout_dd.options = list(factor.variables)
        sumout_dd.value = factor.variables[0]

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Show the chosen factor before and after summing out a variable.
        """
        _ = change
        with output:
            clear_output(wait=True)
            factor = factor_map[factor_dd.value]
            sum_var = sumout_dd.value
            if sum_var not in factor.variables:
                sum_var = factor.variables[0]
            before_df = _factor_to_df(factor)
            after = factor.marginalize([sum_var], inplace=False)
            after_df = _factor_to_df(after)
            _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            # Panel 1: factor before summing out.
            ax1.axis("off")
            ax1.set_title(
                f"Before: scope {{{', '.join(_SHORT[v] for v in factor.variables)}}}",
                fontsize=12,
                fontweight="bold",
            )
            tbl1 = ax1.table(
                cellText=before_df.round(4).values,
                colLabels=list(before_df.columns),
                loc="center",
                cellLoc="center",
            )
            tbl1.auto_set_font_size(False)
            tbl1.set_fontsize(9)
            tbl1.scale(1, 1.3)
            # Panel 2: factor after summing out the chosen variable.
            ax2.axis("off")
            scope_after = (
                ", ".join(_SHORT[v] for v in after.variables)
                if after.variables
                else "(scalar)"
            )
            ax2.set_title(
                f"After summing out {_SHORT[sum_var]}: scope {{{scope_after}}}",
                fontsize=12,
                fontweight="bold",
            )
            tbl2 = ax2.table(
                cellText=after_df.round(4).values,
                colLabels=list(after_df.columns),
                loc="center",
                cellLoc="center",
            )
            tbl2.auto_set_font_size(False)
            tbl2.set_fontsize(9)
            tbl2.scale(1, 1.3)
            # Panel 3: comments on the size change.
            ax3.axis("off")
            ax3.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
            text_content = (
                f"A factor is a table over a\n"
                f"subset of variables.\n\n"
                f"Rows before: {len(before_df)}\n"
                f"Rows after:  {len(after_df)}\n\n"
                f"Summing out {_SHORT[sum_var]} collapses\n"
                f"the table along that\n"
                f"dimension.\n\n"
                f"Two operations suffice for\n"
                f"inference:\n"
                f"  - pointwise product\n"
                f"  - summing out a variable"
            )
            htutori.add_fitted_text_box(
                ax3, text_content, max_fontsize=13, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    def _on_factor_change(change: Optional[Any] = None) -> None:
        """
        Refresh sum-out options, then redraw when the factor changes.
        """
        _ = change
        _refresh_sumout_options()
        update_plot()

    _refresh_sumout_options()
    factor_dd.observe(_on_factor_change, names="value")
    sumout_dd.observe(update_plot, names="value")
    update_plot()
    display(
        ipywidgets.VBox(
            [
                ipywidgets.Label("Pick a factor and a variable to sum out:"),
                factor_dd,
                sumout_dd,
                output,
            ]
        )
    )


# #############################################################################
# Cell 3.2: Variable Elimination Step by Step
# #############################################################################


def _variable_elimination_steps(
    model: "pgmodels.DiscreteBayesianNetwork",
    query_var: str,
    evidence: Dict[str, int],
    order: List[str],
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Run variable elimination one variable at a time, recording each step.

    :param model: The Bayesian network
    :param query_var: Query variable to keep
    :param evidence: Observed assignment
    :param order: Order in which to eliminate the hidden variables
    :return: Tuple of
        - steps: per step, the active factor scopes, the new factor, and the
          cumulative multiplication and addition counts
        - enum_ops: enumeration multiplication count for comparison
    """
    # Start from every CPT reduced by the evidence.
    factors = []
    for cpd in model.get_cpds():
        f = cpd.to_factor()
        ev_in = [(v, val) for v, val in evidence.items() if v in f.variables]
        if ev_in:
            f = f.reduce(ev_in, inplace=False)
        factors.append(f)
    steps = []
    mults = 0
    adds = 0
    # Record the initial state before any elimination.
    steps.append(
        {
            "scopes": [list(f.variables) for f in factors],
            "new_factor": None,
            "mults": mults,
            "adds": adds,
            "eliminated": None,
        }
    )
    # Eliminate each hidden variable in turn.
    for var in order:
        relevant = [f for f in factors if var in f.variables]
        others = [f for f in factors if var not in f.variables]
        # Pointwise product of all factors mentioning the variable.
        prod = relevant[0]
        for f in relevant[1:]:
            prod = prod.product(f, inplace=False)
            mults += int(np.prod(prod.cardinality))
        # Sum the variable out of the product.
        new_factor = prod.marginalize([var], inplace=False)
        adds += int(np.prod(prod.cardinality))
        factors = others + [new_factor]
        steps.append(
            {
                "scopes": [list(f.variables) for f in factors],
                "new_factor": new_factor,
                "mults": mults,
                "adds": adds,
                "eliminated": var,
            }
        )
    # Enumeration cost for the same query, for the comparison bar.
    n_hidden = len(order)
    enum_ops = (2**n_hidden) * (len(model.get_cpds()) - 1)
    return steps, enum_ops


def cell3_2_variable_elimination_widget(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Walk through variable elimination on the alarm query step by step.

    Advances the elimination one variable at a time, showing the shrinking factor
    list, the factor created at each step, and how the operation count compares
    with enumeration.

    :param figsize: Optional figure size
    """
    if figsize is None:
        figsize = (22, 5)
    model = _build_burglary_network()
    infer = pginference.VariableElimination(model)
    # Fix the canonical query so the stepper has a stable hidden-variable set.
    query_var = "Burglary"
    evidence = {"JohnCalls": 1, "MaryCalls": 1}
    order_dd = ipywidgets.Dropdown(
        options=[
            ("Earthquake, then Alarm", ["Earthquake", "Alarm"]),
            ("Alarm, then Earthquake", ["Alarm", "Earthquake"]),
        ],
        value=["Earthquake", "Alarm"],
        description="Order:",
        style={"description_width": "initial"},
    )
    step_slider, step_box = htutori.build_widget_control(
        name="step",
        description="elimination step",
        min_val=0,
        max_val=2,
        step=1,
        initial_value=2,
        is_float=False,
    )
    output = ipywidgets.Output()
    # The final posterior is the same regardless of order, compute it once.
    ref = infer.query([query_var], evidence=evidence, show_progress=False)

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Render the elimination state at the current step.
        """
        _ = change
        with output:
            clear_output(wait=True)
            order = order_dd.value
            steps, enum_ops = _variable_elimination_steps(
                model, query_var, evidence, order
            )
            step = min(int(step_slider.value), len(steps) - 1)
            state = steps[step]
            _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
            # Panel 1: active factor scopes after `step` eliminations.
            ax1.axis("off")
            ax1.set_title(
                f"Active factors (step {step})",
                fontsize=13,
                fontweight="bold",
            )
            scope_lines = []
            for sc in state["scopes"]:
                if sc:
                    scope_lines.append(
                        "{" + ", ".join(_SHORT[v] for v in sc) + "}"
                    )
                else:
                    scope_lines.append("{scalar}")
            ax1.text(
                0.1,
                0.9,
                "\n".join(scope_lines),
                transform=ax1.transAxes,
                fontsize=12,
                va="top",
                family="monospace",
                bbox=dict(boxstyle="round", facecolor="#EBF5FB"),
            )
            # Panel 2: the factor created at this step.
            ax2.axis("off")
            if state["new_factor"] is not None:
                new_df = _factor_to_df(state["new_factor"])
                ax2.set_title(
                    f"New factor (sum out {_SHORT[state['eliminated']]})",
                    fontsize=12,
                    fontweight="bold",
                )
                tbl = ax2.table(
                    cellText=new_df.round(5).values,
                    colLabels=list(new_df.columns),
                    loc="center",
                    cellLoc="center",
                )
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(9)
                tbl.scale(1, 1.4)
            else:
                ax2.set_title("New factor", fontsize=12, fontweight="bold")
                ax2.text(
                    0.5,
                    0.5,
                    "no elimination yet\n(initial factors)",
                    ha="center",
                    va="center",
                    fontsize=11,
                )
            # Panel 3: posterior plus operation-count comparison.
            posterior = [ref.values[1], ref.values[0]]
            ax3.bar(
                [0, 1],
                posterior,
                0.5,
                color=["#2E86C1", "#AED6F1"],
                edgecolor="black",
            )
            ax3.set_xticks([0, 1])
            ax3.set_xticklabels(["B=T", "B=F"])
            ax3.set_ylim([0, 1.05])
            ax3.set_ylabel("P(B | j,m)", fontsize=11)
            ax3_twin = ax3.twinx()
            ax3_twin.bar(
                [3, 4],
                [enum_ops, state["mults"] + state["adds"]],
                0.5,
                color=["#CD6155", "#52BE80"],
                edgecolor="black",
            )
            ax3_twin.set_xticks([0, 1, 3, 4])
            ax3_twin.set_xticklabels(["B=T", "B=F", "enum", "VE"], fontsize=9)
            ax3_twin.set_ylabel("operations", fontsize=11)
            ax3.set_title(
                "Posterior and op count", fontsize=12, fontweight="bold"
            )
            # Panel 4: comments.
            ax4.axis("off")
            ax4.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
            ve_ops = state["mults"] + state["adds"]
            text_content = (
                f"Order: {', '.join(_SHORT[v] for v in order)}\n\n"
                f"Step {step} of {len(steps) - 1}\n"
                f"Active factors: {len(state['scopes'])}\n\n"
                f"VE ops so far:\n"
                f"  mult: {state['mults']}\n"
                f"  add:  {state['adds']}\n"
                f"  total: {ve_ops}\n\n"
                f"Enumeration ops: {enum_ops}\n\n"
                f"Same answer as enumeration,\n"
                f"fewer operations. Order sets\n"
                f"how big the factors get."
            )
            htutori.add_fitted_text_box(
                ax4, text_content, max_fontsize=12, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    order_dd.observe(update_plot, names="value")
    step_slider.observe(update_plot, names="value")
    update_plot()
    display(
        ipywidgets.VBox(
            [
                ipywidgets.Label(
                    "Advance the elimination step and change the order:"
                ),
                order_dd,
                step_box,
                output,
            ]
        )
    )


# #############################################################################
# Cell 3.3: Pruning Irrelevant Variables
# #############################################################################


def _build_extended_network() -> "pgmodels.DiscreteBayesianNetwork":
    """
    Build the burglary network plus an extra Neighbor child of Alarm.

    The leaf `Neighbor` is used to show that an unobserved, non-query effect
    node drops out of the computation.

    :return: Extended Bayesian network with six nodes
    """
    edges = _EDGES + [("Alarm", "Neighbor")]
    model = pgmodels.DiscreteBayesianNetwork(edges)
    cpd_b = pgfactors.TabularCPD("Burglary", 2, [[1 - _P_B], [_P_B]])
    cpd_e = pgfactors.TabularCPD("Earthquake", 2, [[1 - _P_E], [_P_E]])
    a_true = [_P_A_GIVEN_BE[(b, e)] for b in (0, 1) for e in (0, 1)]
    cpd_a = pgfactors.TabularCPD(
        "Alarm",
        2,
        [[1 - p for p in a_true], list(a_true)],
        evidence=["Burglary", "Earthquake"],
        evidence_card=[2, 2],
    )
    j_true = [_P_J_GIVEN_A[a] for a in (0, 1)]
    cpd_j = pgfactors.TabularCPD(
        "JohnCalls",
        2,
        [[1 - p for p in j_true], list(j_true)],
        evidence=["Alarm"],
        evidence_card=[2],
    )
    m_true = [_P_M_GIVEN_A[a] for a in (0, 1)]
    cpd_m = pgfactors.TabularCPD(
        "MaryCalls",
        2,
        [[1 - p for p in m_true], list(m_true)],
        evidence=["Alarm"],
        evidence_card=[2],
    )
    # A second neighbor who sometimes calls when the alarm sounds.
    cpd_n = pgfactors.TabularCPD(
        "Neighbor",
        2,
        [[0.8, 0.3], [0.2, 0.7]],
        evidence=["Alarm"],
        evidence_card=[2],
    )
    model.add_cpds(cpd_b, cpd_e, cpd_a, cpd_j, cpd_m, cpd_n)
    hdbg.dassert(model.check_model(), "Extended network is not a valid model")
    return model


def cell3_3_pruning_widget(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Show that non-ancestors of the query or evidence can be pruned for free.

    Shades the DAG by relevance and compares the posterior on the full network
    with the posterior on the pruned network.

    :param figsize: Optional figure size
    """
    if figsize is None:
        figsize = (20, 5)
    model = _build_extended_network()
    infer = pginference.VariableElimination(model)
    nodes_ext = list(model.nodes())
    graph = nx.DiGraph(list(model.edges()))
    # Build query controls over the extended node set.
    query_dd = ipywidgets.Dropdown(
        options=nodes_ext,
        value="Burglary",
        description="Query X:",
        style={"description_width": "initial"},
    )
    checks = {}
    valdds = {}
    rows = []
    default_evidence = {"JohnCalls": 1, "MaryCalls": 1}
    for n in nodes_ext:
        cb = ipywidgets.Checkbox(
            value=(n in default_evidence),
            description=f"observe {n}",
            indent=False,
            layout={"width": "180px"},
        )
        vd = ipywidgets.Dropdown(
            options=[("True", 1), ("False", 0)],
            value=default_evidence.get(n, 1),
            description="",
            layout={"width": "90px"},
        )
        checks[n] = cb
        valdds[n] = vd
        rows.append(ipywidgets.HBox([cb, vd]))
    prune_toggle = ipywidgets.ToggleButton(
        value=True,
        description="prune irrelevant variables",
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Recolor by relevance and compare full vs pruned posteriors.
        """
        _ = change
        with output:
            clear_output(wait=True)
            query_var = query_dd.value
            evidence = {
                n: valdds[n].value
                for n in nodes_ext
                if n != query_var and checks[n].value
            }
            prune = prune_toggle.value
            # Relevant set: query, evidence, and their ancestors.
            relevant = set([query_var]) | set(evidence.keys())
            for node in [query_var] + list(evidence.keys()):
                relevant |= nx.ancestors(graph, node)
            short = {n: n[0] if n != "Neighbor" else "N" for n in nodes_ext}
            node_colors = {
                n: ("#82E0AA" if n in relevant else "#D5D8DC") for n in nodes_ext
            }
            _, (ax1, ax2, ax3) = plt.subplots(
                1, 3, figsize=figsize, gridspec_kw={"width_ratios": [1.4, 1, 1]}
            )
            hgraphv.plot_causal_dag(
                graph,
                "Relevant (green) vs irrelevant (grey)",
                node_colors=node_colors,
                ax=ax1,
            )
            # Posterior on the full network.
            full = infer.query(
                [query_var], evidence=evidence, show_progress=False
            )
            # Posterior on the pruned network (irrelevant nodes removed).
            if prune:
                sub_nodes = relevant
                # `subgraph()` keeps only nodes/edges, not CPDs, so rebuild
                # the model from the induced edges and the matching CPDs.
                sub_edges = [
                    (u, v)
                    for u, v in model.edges()
                    if u in sub_nodes and v in sub_nodes
                ]
                sub_model = pgmodels.DiscreteBayesianNetwork(sub_edges)
                sub_model.add_nodes_from(sub_nodes)
                sub_model.add_cpds(
                    *[
                        cpd
                        for cpd in model.get_cpds()
                        if cpd.variable in sub_nodes
                    ]
                )
                sub_infer = pginference.VariableElimination(sub_model)
                pruned = sub_infer.query(
                    [query_var], evidence=evidence, show_progress=False
                )
                pruned_vals = [pruned.values[1], pruned.values[0]]
            else:
                pruned_vals = [full.values[1], full.values[0]]
            full_vals = [full.values[1], full.values[0]]
            x = np.arange(2)
            width = 0.35
            ax2.bar(
                x - width / 2,
                full_vals,
                width,
                color="#5DADE2",
                edgecolor="black",
                label="full",
            )
            ax2.bar(
                x + width / 2,
                pruned_vals,
                width,
                color="#82E0AA",
                edgecolor="black",
                label="pruned",
            )
            ax2.set_xticks(x)
            ax2.set_xticklabels(
                [f"{short[query_var]}=T", f"{short[query_var]}=F"]
            )
            ax2.set_ylim([0, 1.05])
            ax2.set_ylabel(f"P({short[query_var]} | e)", fontsize=11)
            ax2.set_title(
                "Full vs pruned posterior", fontsize=12, fontweight="bold"
            )
            ax2.legend(fontsize=10)
            # Panel 3: comments.
            ax3.axis("off")
            ax3.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
            irrelevant = [n for n in nodes_ext if n not in relevant]
            irr_str = (
                ", ".join(short[n] for n in irrelevant)
                if irrelevant
                else "(none)"
            )
            text_content = (
                f"Query: {short[query_var]}\n"
                f"Evidence: {_fmt_evidence({k: v for k, v in evidence.items() if k in _SHORT}) if all(k in _SHORT for k in evidence) else 'mixed'}\n\n"
                f"Relevant nodes: {len(relevant)}\n"
                f"Irrelevant: {irr_str}\n\n"
                f"Posteriors identical:\n"
                f"  {np.allclose(full_vals, pruned_vals)}\n\n"
                f"A node that is not asked\n"
                f"about, observed, nor an\n"
                f"ancestor of either cannot\n"
                f"change the answer. Delete\n"
                f"it before computing."
            )
            htutori.add_fitted_text_box(
                ax3, text_content, max_fontsize=12, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    query_dd.observe(update_plot, names="value")
    for n in nodes_ext:
        checks[n].observe(update_plot, names="value")
        valdds[n].observe(update_plot, names="value")
    prune_toggle.observe(update_plot, names="value")
    update_plot()
    display(
        ipywidgets.VBox(
            [
                ipywidgets.Label("Set the query and evidence; toggle pruning:"),
                query_dd,
                *rows,
                prune_toggle,
                output,
            ]
        )
    )


# #############################################################################
# Cell 4.1: Complexity: Polytrees vs General Graphs
# #############################################################################


def cell4_1_complexity_widget(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Contrast exact-inference cost on polytrees versus dense networks.

    Shows a polytree and a dense network of the same size, plus operation-count
    curves that grow linearly for polytrees and exponentially for dense graphs.

    :param figsize: Optional figure size
    """
    if figsize is None:
        figsize = (22, 5)
    n_slider, n_box = htutori.build_widget_control(
        name="n",
        description="number of nodes",
        min_val=3,
        max_val=10,
        step=1,
        initial_value=6,
        is_float=False,
    )
    structure_dd = ipywidgets.Dropdown(
        options=["polytree", "fully connected"],
        value="polytree",
        description="Structure:",
        style={"description_width": "initial"},
    )
    quality_dd = ipywidgets.Dropdown(
        options=["good", "bad"],
        value="good",
        description="Order quality:",
        style={"description_width": "initial"},
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw the two example graphs and the cost curves.
        """
        _ = change
        with output:
            clear_output(wait=True)
            n = int(n_slider.value)
            structure = structure_dd.value
            quality = quality_dd.value
            # A good order keeps the constant small; a bad one inflates it.
            const = 4 if quality == "good" else 12
            _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
            # Panel 1: a polytree (simple chain) of n nodes.
            chain = nx.path_graph(n, create_using=nx.DiGraph)
            pos1 = {i: (i, 0) for i in range(n)}
            nx.draw(
                chain,
                pos1,
                ax=ax1,
                node_color="#A6C8F4",
                node_size=400,
                with_labels=True,
                arrows=True,
            )
            ax1.set_title(f"Polytree (n={n})", fontsize=12, fontweight="bold")
            # Panel 2: a densely connected DAG on n nodes.
            dense = nx.DiGraph()
            dense.add_nodes_from(range(n))
            for i in range(n):
                for j in range(i + 1, n):
                    dense.add_edge(i, j)
            pos2 = nx.circular_layout(dense)
            nx.draw(
                dense,
                pos2,
                ax=ax2,
                node_color="#F5B7B1",
                node_size=400,
                with_labels=True,
                arrows=True,
            )
            ax2.set_title(
                f"Dense network (n={n})", fontsize=12, fontweight="bold"
            )
            # Panel 3: cost curves over node count, with the current n marked.
            ns = np.arange(3, 11)
            poly_cost = const * ns
            dense_cost = const * (2.0**ns)
            ax3.plot(
                ns,
                poly_cost,
                "-",
                color="#2E86C1",
                linewidth=2.5,
                label="polytree O(n)",
            )
            ax3.plot(
                ns,
                dense_cost,
                "-",
                color="#CD6155",
                linewidth=2.5,
                label="dense O(2^n)",
            )
            current_cost = (
                const * n if structure == "polytree" else const * (2.0**n)
            )
            ax3.scatter([n], [current_cost], color="black", s=80, zorder=5)
            ax3.set_yscale("log")
            ax3.set_xlabel("number of nodes n", fontsize=11)
            ax3.set_ylabel("operations (log scale)", fontsize=11)
            ax3.set_title("Cost vs network size", fontsize=12, fontweight="bold")
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)
            # Panel 4: comments.
            ax4.axis("off")
            ax4.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
            text_content = (
                f"Nodes: n = {n}\n"
                f"Structure: {structure}\n"
                f"Order quality: {quality}\n\n"
                f"Operating point cost:\n"
                f"  {current_cost:,.0f}\n\n"
                f"Polytree: O(n), linear.\n"
                f"Dense: O(2^n), exponential.\n\n"
                f"A good elimination order\n"
                f"keeps factors small. A bad\n"
                f"one inflates the constant\n"
                f"and can change growth on\n"
                f"dense graphs."
            )
            htutori.add_fitted_text_box(
                ax4, text_content, max_fontsize=12, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    n_slider.observe(update_plot, names="value")
    structure_dd.observe(update_plot, names="value")
    quality_dd.observe(update_plot, names="value")
    update_plot()
    display(
        ipywidgets.VBox(
            [
                ipywidgets.Label(
                    "Vary the node count, structure, and order quality:"
                ),
                n_box,
                structure_dd,
                quality_dd,
                output,
            ]
        )
    )


# #############################################################################
# Cell 4.2: When Exact Inference Breaks Down
# #############################################################################


def cell4_2_breakdown_widget(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Summarize the regimes where exact inference works or breaks down.

    A dropdown selects one of three regimes and updates a recommendation banner;
    a sketch reminds students that continuous variables turn the summation into
    an intractable integral.

    :param figsize: Optional figure size
    """
    if figsize is None:
        figsize = (20, 5)
    scenario_dd = ipywidgets.Dropdown(
        options=[
            "Small discrete polytree",
            "Large dense discrete network",
            "Continuous variables",
        ],
        value="Small discrete polytree",
        description="Scenario:",
        style={"description_width": "initial"},
    )
    output = ipywidgets.Output()
    # Recommendation and explanation for each regime.
    regime_info = {
        "Small discrete polytree": (
            "EXACT",
            "#82E0AA",
            "Few discrete nodes, tree-like.\n"
            "Enumeration and variable\n"
            "elimination are fast and\n"
            "recommended.",
        ),
        "Large dense discrete network": (
            "APPROXIMATE",
            "#F5B7B1",
            "Intermediate factors explode.\n"
            "Exact cost is exponential, so\n"
            "use sampling / MCMC instead.",
        ),
        "Continuous variables": (
            "APPROXIMATE",
            "#F5B7B1",
            "The sum over hidden values\n"
            "becomes an integral with no\n"
            "closed form. Use sampling or\n"
            "specialized methods.",
        ),
    }

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Show the regime comparison table, a sketch, and the recommendation.
        """
        _ = change
        with output:
            clear_output(wait=True)
            scenario = scenario_dd.value
            recommendation, color, detail = regime_info[scenario]
            _, (ax1, ax2, ax3) = plt.subplots(
                1, 3, figsize=figsize, gridspec_kw={"width_ratios": [1.3, 1, 1]}
            )
            # Panel 1: the three-regime comparison table.
            ax1.axis("off")
            ax1.set_title(
                "Regimes for exact inference",
                fontsize=13,
                fontweight="bold",
            )
            table_rows = [
                ["small discrete\npolytree", "fast", "exact"],
                ["large dense\ndiscrete", "exponential", "approximate"],
                ["continuous\nvariables", "integral", "approximate"],
            ]
            tbl = ax1.table(
                cellText=table_rows,
                colLabels=["regime", "cost", "recommended"],
                loc="center",
                cellLoc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.scale(1, 2.0)
            # Highlight the selected scenario's row.
            scenario_to_row = {
                "Small discrete polytree": 1,
                "Large dense discrete network": 2,
                "Continuous variables": 3,
            }
            for col in range(3):
                tbl[scenario_to_row[scenario], col].set_facecolor(color)
            # Panel 2: a small continuous-variable sketch.
            ax2.axis("off")
            ax2.set_title("Continuous case", fontsize=13, fontweight="bold")
            sketch = nx.DiGraph([("X", "Y"), ("Y", "Z")])
            pos = {"X": (0, 1), "Y": (1, 1), "Z": (2, 1)}
            nx.draw(
                sketch,
                pos,
                ax=ax2,
                node_color="#F9E79F",
                node_size=900,
                with_labels=True,
                arrows=True,
            )
            ax2.text(
                0.5,
                0.15,
                r"$\sum_{y}\ \rightarrow\ \int dy$",
                transform=ax2.transAxes,
                ha="center",
                fontsize=15,
            )
            ax2.set_xlim(-0.5, 2.5)
            # Panel 3: recommendation banner and detail.
            ax3.axis("off")
            ax3.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
            ax3.text(
                0.5,
                0.92,
                f"Use: {recommendation}",
                transform=ax3.transAxes,
                ha="center",
                va="top",
                fontsize=14,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=color),
            )
            htutori.add_fitted_text_box(
                ax3,
                detail + "\n\nExact methods stay the\n"
                "gold-standard reference for\n"
                "validating approximate ones\n"
                "on small networks.",
                box_xy=(0.05, 0.72),
                box_height=0.66,
                max_fontsize=12,
                min_fontsize=9,
            )
            plt.tight_layout()
            plt.show()

    scenario_dd.observe(update_plot, names="value")
    update_plot()
    display(
        ipywidgets.VBox(
            [
                ipywidgets.Label("Pick a scenario to see the recommendation:"),
                scenario_dd,
                output,
            ]
        )
    )
