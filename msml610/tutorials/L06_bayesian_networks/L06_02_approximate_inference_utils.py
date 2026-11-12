"""
Utility functions for the approximate inference notebook.

Implements the interactive widgets, samplers, and visualizations that teach
approximate posterior inference by sampling on the canonical AIMA sprinkler
Bayesian network.

Import as:

import msml610.tutorials.L06_bayesian_networks.L06_02_approximate_inference_utils as mtlbnl0aiu
"""

import itertools
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import ipywidgets
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
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
warnings.filterwarnings("ignore", category=UserWarning)


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger to the utils logger.

    :param notebook_log: Logger created in the notebook
    """
    global _LOG
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# Sprinkler network definition (shared by all cells).
# #############################################################################

# Standard AIMA sprinkler-network probabilities.
# State convention everywhere: 0 = False, 1 = True.
_P_C = 0.5
# P(Sprinkler=True | Cloudy) for each Cloudy value.
_P_S_GIVEN_C = {0: 0.5, 1: 0.1}
# P(Rain=True | Cloudy) for each Cloudy value.
_P_R_GIVEN_C = {0: 0.2, 1: 0.8}
# P(WetGrass=True | Sprinkler, Rain) for the four parent combinations.
_P_W_GIVEN_SR = {
    (0, 0): 0.0,
    (0, 1): 0.9,
    (1, 0): 0.9,
    (1, 1): 0.99,
}

# Canonical node names in topological order and the network structure.
_NODES = ["Cloudy", "Sprinkler", "Rain", "WetGrass"]
_EDGES = [
    ("Cloudy", "Sprinkler"),
    ("Cloudy", "Rain"),
    ("Sprinkler", "WetGrass"),
    ("Rain", "WetGrass"),
]
# Short math symbols used in equations and compact table headers.
_SHORT = {
    "Cloudy": "C",
    "Sprinkler": "S",
    "Rain": "R",
    "WetGrass": "W",
}
# Color scheme by depth in the topological order (root to leaf).
_TOPO_COLORS = {
    "Cloudy": "#A6C8F4",
    "Sprinkler": "#9FE3C5",
    "Rain": "#9FE3C5",
    "WetGrass": "#F5B7B1",
}
# Visual style constants shared across cells.
_EMPIRICAL_COLOR = "#2E86C1"
_REFERENCE_COLOR = "#CD6155"


def _build_sprinkler_network() -> "pgmodels.DiscreteBayesianNetwork":
    """
    Build the AIMA sprinkler network as a `pgmpy` model.

    The model provides the exact ground-truth posteriors that every sampling
    estimate in the notebook is compared against.

    :return: Bayesian network with all four CPTs attached
    """
    model = pgmodels.DiscreteBayesianNetwork(_EDGES)
    # Prior on the single root cause.
    cpd_c = pgfactors.TabularCPD("Cloudy", 2, [[1 - _P_C], [_P_C]])
    # P(Sprinkler | Cloudy): columns are Cloudy in (0, 1).
    s_true = [_P_S_GIVEN_C[c] for c in (0, 1)]
    cpd_s = pgfactors.TabularCPD(
        "Sprinkler",
        2,
        [[1 - p for p in s_true], list(s_true)],
        evidence=["Cloudy"],
        evidence_card=[2],
    )
    # P(Rain | Cloudy): columns are Cloudy in (0, 1).
    r_true = [_P_R_GIVEN_C[c] for c in (0, 1)]
    cpd_r = pgfactors.TabularCPD(
        "Rain",
        2,
        [[1 - p for p in r_true], list(r_true)],
        evidence=["Cloudy"],
        evidence_card=[2],
    )
    # P(WetGrass | Sprinkler, Rain): pgmpy orders columns with the last evidence
    # variable changing fastest, i.e. (S,R) in (0,0),(0,1),(1,0),(1,1).
    w_true = [_P_W_GIVEN_SR[(s, r)] for s in (0, 1) for r in (0, 1)]
    cpd_w = pgfactors.TabularCPD(
        "WetGrass",
        2,
        [[1 - p for p in w_true], list(w_true)],
        evidence=["Sprinkler", "Rain"],
        evidence_card=[2, 2],
    )
    model.add_cpds(cpd_c, cpd_s, cpd_r, cpd_w)
    hdbg.dassert(model.check_model(), "Sprinkler network is not a valid model")
    return model


def _joint_product(assignment: Dict[str, int]) -> float:
    """
    Evaluate the factored joint probability for a full assignment.

    Computes the product of the four CPTs:
    P(C) P(S | C) P(R | C) P(W | S,R).

    :param assignment: Map from each node name to its value (0 or 1)
    :return: Joint probability of the assignment
    """
    c = assignment["Cloudy"]
    s = assignment["Sprinkler"]
    r = assignment["Rain"]
    w = assignment["WetGrass"]
    # Look up each CPT entry, flipping to the False branch when value is 0.
    p_c = _P_C if c == 1 else 1 - _P_C
    ps = _P_S_GIVEN_C[c]
    p_s = ps if s == 1 else 1 - ps
    pr = _P_R_GIVEN_C[c]
    p_r = pr if r == 1 else 1 - pr
    pw = _P_W_GIVEN_SR[(s, r)]
    p_w = pw if w == 1 else 1 - pw
    return p_c * p_s * p_r * p_w


def _exact_marginal(
    model: "pgmodels.DiscreteBayesianNetwork",
    query_var: str,
    evidence: Dict[str, int],
) -> np.ndarray:
    """
    Compute the exact posterior over a query variable with `pgmpy`.

    :param model: The sprinkler Bayesian network
    :param query_var: Name of the query variable X
    :param evidence: Map from observed node names to their values
    :return: Array `[P(X=False | e), P(X=True | e)]`
    """
    infer = pginference.VariableElimination(model)
    result = infer.query(
        [query_var], evidence=evidence or None, show_progress=False
    )
    return np.asarray(result.values, dtype=float)


def _pw_array(s: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Vectorized lookup of P(WetGrass=True | Sprinkler, Rain).

    :param s: Array of Sprinkler values (0 or 1)
    :param r: Array of Rain values (0 or 1)
    :return: Array of P(WetGrass=True) for each (Sprinkler, Rain) pair
    """
    # Pick the CPT entry by the (Sprinkler, Rain) combination per element.
    out = np.empty(len(s), dtype=float)
    for (sv, rv), p in _P_W_GIVEN_SR.items():
        out[(s == sv) & (r == rv)] = p
    return out


def _prior_sample_array(
    rng: np.random.Generator, n: int
) -> Dict[str, np.ndarray]:
    """
    Draw `n` prior samples from the network by topological inverse transform.

    Each variable is sampled only after its parents, so every draw uses a fully
    specified CPT row. Sampling is vectorized for speed.

    :param rng: NumPy random generator
    :param n: Number of full events to generate
    :return: Map from node name to an array of `n` sampled values
    """
    # Sample the root cause from its prior.
    c = (rng.random(n) < _P_C).astype(int)
    # Sample Sprinkler from P(Sprinkler | Cloudy).
    ps = np.where(c == 1, _P_S_GIVEN_C[1], _P_S_GIVEN_C[0])
    s = (rng.random(n) < ps).astype(int)
    # Sample Rain from P(Rain | Cloudy).
    pr = np.where(c == 1, _P_R_GIVEN_C[1], _P_R_GIVEN_C[0])
    r = (rng.random(n) < pr).astype(int)
    # Sample WetGrass from P(WetGrass | Sprinkler, Rain).
    pw = _pw_array(s, r)
    w = (rng.random(n) < pw).astype(int)
    return {"Cloudy": c, "Sprinkler": s, "Rain": r, "WetGrass": w}


def _likelihood_weight_array(
    rng: np.random.Generator, n: int, evidence: Dict[str, int]
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Draw `n` likelihood-weighted samples conditioned on the evidence.

    Evidence nodes are clamped to their observed values and contribute their CPT
    probability to the importance weight; non-evidence nodes are sampled from
    their CPT given the already-sampled parents.

    :param rng: NumPy random generator
    :param n: Number of weighted samples to generate
    :param evidence: Map from observed node names to their values
    :return: Tuple of (samples map, importance weights array)
    """
    w = np.ones(n, dtype=float)
    # Cloudy has no parents.
    if "Cloudy" in evidence:
        c = np.full(n, evidence["Cloudy"], dtype=int)
        w *= np.where(c == 1, _P_C, 1 - _P_C)
    else:
        c = (rng.random(n) < _P_C).astype(int)
    # Sprinkler given Cloudy.
    ps = np.where(c == 1, _P_S_GIVEN_C[1], _P_S_GIVEN_C[0])
    if "Sprinkler" in evidence:
        s = np.full(n, evidence["Sprinkler"], dtype=int)
        w *= np.where(s == 1, ps, 1 - ps)
    else:
        s = (rng.random(n) < ps).astype(int)
    # Rain given Cloudy.
    pr = np.where(c == 1, _P_R_GIVEN_C[1], _P_R_GIVEN_C[0])
    if "Rain" in evidence:
        r = np.full(n, evidence["Rain"], dtype=int)
        w *= np.where(r == 1, pr, 1 - pr)
    else:
        r = (rng.random(n) < pr).astype(int)
    # WetGrass given Sprinkler and Rain.
    pw = _pw_array(s, r)
    if "WetGrass" in evidence:
        wg = np.full(n, evidence["WetGrass"], dtype=int)
        w *= np.where(wg == 1, pw, 1 - pw)
    else:
        wg = (rng.random(n) < pw).astype(int)
    samples = {"Cloudy": c, "Sprinkler": s, "Rain": r, "WetGrass": wg}
    return samples, w


def _fmt_evidence(evidence: Dict[str, int]) -> str:
    """
    Format an evidence dict as a compact human-readable string.

    :param evidence: Map from observed node names to their values
    :return: String like "S=T" using short symbols
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
    default_query: str = "Rain",
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
        default_evidence = {"Sprinkler": 1}
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


def _add_seed_control() -> Tuple[Any, Any]:
    """
    Build the standard seed slider, placed first in every widget.

    :return: Tuple of (seed slider, seed box)
    """
    return htutori.build_widget_control(
        name="seed",
        description="random seed",
        min_val=0,
        max_val=200,
        step=1,
        initial_value=42,
        is_float=False,
    )


# #############################################################################
# Cell 1.1: Turning Uniform Randomness into a Discrete Distribution
# #############################################################################


def _plot_inverse_transform(
    dist: str,
    lam: float,
    n: int,
    seed: int,
    figsize: Tuple[float, float],
) -> None:
    """
    Draw the target, the CDF with one inverse-transform sample, and a sample
    histogram vs the target, shared by the discrete and continuous inverse-
    transform cells.

    :param dist: "biased die (discrete)" or "exponential (continuous)"
    :param lam: exponential rate, used only when `dist` is exponential
    :param n: number of samples to draw
    :param seed: random seed
    :param figsize: figure size for the 1x4 panel layout
    """
    rng = np.random.default_rng(seed)
    # Every sampler starts from a stream of uniform numbers.
    u = rng.random(n)
    _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
    if dist == "biased die (discrete)":
        # A six-faced die with probabilities proportional to face value.
        faces = np.arange(1, 7)
        probs = faces / faces.sum()
        cdf = np.cumsum(probs)
        # Inverse transform: smallest face whose CDF exceeds r.
        samples = np.searchsorted(cdf, u, side="right") + 1
        # Panel 1: the target probability mass function.
        ax1.bar(
            faces,
            probs,
            color=_EMPIRICAL_COLOR,
            edgecolor="black",
            alpha=0.85,
        )
        ax1.set_title("Target: biased die", fontsize=13, fontweight="bold")
        ax1.set_xlabel("face")
        ax1.set_ylabel("P(face)")
        # Panel 2: the staircase CDF with one r mapped to its x.
        ax2.step(
            np.concatenate([[0.5], faces, [6.5]]),
            np.concatenate([[0.0], cdf, [1.0]]),
            where="post",
            color="black",
            linewidth=2,
            label="CDF F(x)",
        )
        r0 = float(u[0])
        x0 = int(samples[0])
        # Horizontal line at the sampled r and vertical drop to its x.
        ax2.axhline(r0, color=_REFERENCE_COLOR, linestyle=":", linewidth=2)
        ax2.plot(
            [x0, x0], [0, r0], color=_REFERENCE_COLOR, linestyle=":", linewidth=2
        )
        ax2.scatter([x0], [r0], color=_REFERENCE_COLOR, zorder=5)
        ax2.text(0.55, r0 + 0.02, f"r={r0:.2f}", color=_REFERENCE_COLOR)
        ax2.text(x0 + 0.05, 0.02, f"x={x0}", color=_REFERENCE_COLOR)
        ax2.set_title("CDF and inverse map", fontsize=13, fontweight="bold")
        ax2.set_xlabel("face")
        ax2.set_ylabel("F(x) = P(X <= x)")
        ax2.legend(fontsize=9, loc="lower right")
        # Panel 3: sample histogram (solid) vs target pmf (dotted).
        counts = np.bincount(samples, minlength=7)[1:] / n
        ax3.bar(
            faces,
            counts,
            color=_EMPIRICAL_COLOR,
            edgecolor="black",
            alpha=0.85,
            label=f"samples (N={n})",
        )
        ax3.bar(
            faces,
            probs,
            color="none",
            edgecolor=_REFERENCE_COLOR,
            linestyle=":",
            linewidth=2,
            label="target",
        )
        ax3.set_title("Sample histogram", fontsize=13, fontweight="bold")
        ax3.set_xlabel("face")
        ax3.set_ylabel("frequency")
        ax3.legend(fontsize=9)
        max_err = float(np.max(np.abs(counts - probs)))
        detail = (
            f"Target: biased die\n"
            f"P(face) ~ face value\n\n"
            f"Inverse transform:\n"
            f"  find smallest x with\n"
            f"  F(x) > r\n\n"
            f"r = {r0:.3f} -> x = {x0}\n\n"
            f"N = {n}\n"
            f"max |freq - P| = {max_err:.3f}"
        )
    else:
        # Exponential target with closed-form inverse CDF.
        samples = -np.log(1 - u) / lam
        grid = np.linspace(0, np.max(samples) + 1e-9, 400)
        pdf = lam * np.exp(-lam * grid)
        cdf = 1 - np.exp(-lam * grid)
        # Panel 1: the target density.
        ax1.plot(grid, pdf, color=_EMPIRICAL_COLOR, linewidth=2.5)
        ax1.fill_between(grid, pdf, alpha=0.2, color=_EMPIRICAL_COLOR)
        ax1.set_title("Target: exponential", fontsize=13, fontweight="bold")
        ax1.set_xlabel("x")
        ax1.set_ylabel("density f(x)")
        # Panel 2: the smooth CDF with one r mapped to its x.
        ax2.plot(grid, cdf, color="black", linewidth=2, label="CDF F(x)")
        r0 = float(u[0])
        x0 = -np.log(1 - r0) / lam
        ax2.axhline(r0, color=_REFERENCE_COLOR, linestyle=":", linewidth=2)
        ax2.plot(
            [x0, x0], [0, r0], color=_REFERENCE_COLOR, linestyle=":", linewidth=2
        )
        ax2.scatter([x0], [r0], color=_REFERENCE_COLOR, zorder=5)
        ax2.text(grid[1], r0 + 0.02, f"r={r0:.2f}", color=_REFERENCE_COLOR)
        ax2.text(x0, 0.03, f"x={x0:.2f}", color=_REFERENCE_COLOR)
        ax2.set_title("CDF and inverse map", fontsize=13, fontweight="bold")
        ax2.set_xlabel("x")
        ax2.set_ylabel("F(x) = P(X <= x)")
        ax2.legend(fontsize=9, loc="lower right")
        # Panel 3: sample histogram (solid) vs target density (dotted).
        sns.histplot(
            samples,
            bins=40,
            stat="density",
            color=_EMPIRICAL_COLOR,
            alpha=0.6,
            ax=ax3,
            label=f"samples (N={n})",
        )
        ax3.plot(
            grid,
            pdf,
            color=_REFERENCE_COLOR,
            linestyle=":",
            linewidth=2.5,
            label="target",
        )
        ax3.set_title("Sample histogram", fontsize=13, fontweight="bold")
        ax3.set_xlabel("x")
        ax3.set_ylabel("density")
        ax3.legend(fontsize=9)
        mean_emp = float(np.mean(samples))
        detail = (
            f"Target: exponential\n"
            f"rate lambda = {lam:.2f}\n\n"
            f"Inverse CDF (closed form):\n"
            f"  x = -ln(1 - r) / lambda\n\n"
            f"r = {r0:.3f} -> x = {x0:.3f}\n\n"
            f"N = {n}\n"
            f"sample mean = {mean_emp:.3f}\n"
            f"theory mean = {1 / lam:.3f}"
        )
    # Panel 4: comments tying the construction together.
    ax4.axis("off")
    ax4.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
    htutori.add_fitted_text_box(ax4, detail, max_fontsize=12, min_fontsize=9)
    plt.tight_layout()
    plt.show()


def cell1_1_inverse_transform_discrete_widget(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Show how a uniform stream is reshaped into a discrete distribution.

    Draws a biased die, its CDF with a sampled `r` mapped through the
    inverse CDF to a face `x`, and a histogram of generated samples that
    fills in the target as the sample count grows.

    :param figsize: Optional figure size for the 1x4 panel layout
    """
    if figsize is None:
        figsize = (22, 5)
    seed_slider, seed_box = _add_seed_control()
    # N spans orders of magnitude, so it uses a logarithmic slider.
    n_exp_slider, n_box = htutori.build_log_widget_control(
        name="log(N)",
        description="N (samples)",
        min_exp=1,
        max_exp=14,
        initial_exp=8,
        base=2,
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Regenerate die samples by inverse transform and redraw all panels.
        """
        _ = change
        with output:
            clear_output(wait=True)
            n = 2**n_exp_slider.value
            _plot_inverse_transform(
                "biased die (discrete)", 1.0, n, seed_slider.value, figsize
            )

    # Keep the plot in sync with every control.
    seed_slider.observe(update_plot, names="value")
    n_exp_slider.observe(update_plot, names="value")
    update_plot()
    display(ipywidgets.VBox([seed_box, n_box, output]))


# #############################################################################
# Cell 1.2: Turning Uniform Randomness into a Continuous Distribution
# #############################################################################


def cell1_2_inverse_transform_continuous_widget(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Show how a uniform stream is reshaped into a continuous distribution.

    Draws an exponential density, its CDF with a sampled `r` mapped through
    the closed-form inverse CDF to an `x`, and a histogram of generated
    samples that fills in the target as the sample count grows.

    :param figsize: Optional figure size for the 1x4 panel layout
    """
    if figsize is None:
        figsize = (22, 5)
    seed_slider, seed_box = _add_seed_control()
    lam_slider, lam_box = htutori.build_widget_control(
        name="lambda",
        description="exponential rate",
        min_val=0.2,
        max_val=3.0,
        step=0.1,
        initial_value=1.0,
        is_float=True,
    )
    # N spans orders of magnitude, so it uses a logarithmic slider.
    n_exp_slider, n_box = htutori.build_log_widget_control(
        name="log(N)",
        description="N (samples)",
        min_exp=1,
        max_exp=14,
        initial_exp=8,
        base=2,
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Regenerate exponential samples by inverse transform and redraw.
        """
        _ = change
        with output:
            clear_output(wait=True)
            n = 2**n_exp_slider.value
            _plot_inverse_transform(
                "exponential (continuous)",
                lam_slider.value,
                n,
                seed_slider.value,
                figsize,
            )

    # Keep the plot in sync with every control.
    seed_slider.observe(update_plot, names="value")
    lam_slider.observe(update_plot, names="value")
    n_exp_slider.observe(update_plot, names="value")
    update_plot()
    display(ipywidgets.VBox([seed_box, lam_box, n_box, output]))


# #############################################################################
# Cell 1.3: Prior Sampling from the Sprinkler Network
# #############################################################################


def cell1_3_prior_sampling_widget(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Scale inverse-transform sampling up to a whole Bayesian network.

    Generates `N` full events in topological order and compares the estimated
    marginal of a chosen variable and the estimated joint frequencies with the
    exact `pgmpy` values.

    :param figsize: Optional figure size for the 1x4 panel layout
    """
    if figsize is None:
        figsize = (22, 5)
    model = _build_sprinkler_network()
    graph = nx.DiGraph(_EDGES)
    seed_slider, seed_box = _add_seed_control()
    n_exp_slider, n_box = htutori.build_log_widget_control(
        name="log(N)",
        description="N (full events)",
        min_exp=2,
        max_exp=14,
        initial_exp=9,
        base=2,
    )
    # Dropdown to track one variable's marginal estimate against its exact value.
    var_dd = ipywidgets.Dropdown(
        options=_NODES,
        value="Rain",
        description="Track marginal:",
        style={"description_width": "initial"},
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Generate prior samples and compare estimates with the exact values.
        """
        _ = change
        with output:
            clear_output(wait=True)
            n = 2**n_exp_slider.value
            track = var_dd.value
            rng = np.random.default_rng(seed_slider.value)
            samples = _prior_sample_array(rng, n)
            _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
            # Panel 1: the DAG colored by topological depth.
            hgraphv.plot_causal_dag(
                graph,
                "Sample parents before children",
                node_colors=_TOPO_COLORS,
                ax=ax1,
            )
            # Panel 2: estimated marginal of the tracked variable vs exact.
            est_true = float(np.mean(samples[track]))
            estimate = [1 - est_true, est_true]
            exact = _exact_marginal(model, track, {})
            x = np.arange(2)
            ax2.bar(
                x,
                estimate,
                0.55,
                color=_EMPIRICAL_COLOR,
                edgecolor="black",
                label="estimate",
            )
            ax2.bar(
                x,
                exact,
                0.28,
                color="none",
                edgecolor=_REFERENCE_COLOR,
                linestyle=":",
                linewidth=2,
                label="exact (pgmpy)",
            )
            ax2.set_xticks(x)
            ax2.set_xticklabels([f"{_SHORT[track]}=F", f"{_SHORT[track]}=T"])
            ax2.set_ylim([0, 1.05])
            ax2.set_ylabel(f"P({_SHORT[track]})")
            ax2.set_title(f"Marginal of {track}", fontsize=13, fontweight="bold")
            ax2.legend(fontsize=9)
            # Panel 3: estimated joint frequencies vs the exact joint.
            combos = list(itertools.product([0, 1], repeat=4))
            labels = [
                "".join(
                    f"{_SHORT[node]}{'T' if val else 'F'}"
                    for node, val in zip(_NODES, combo)
                )
                for combo in combos
            ]
            # Empirical joint frequency from the generated events.
            keys = (
                samples["Cloudy"] * 8
                + samples["Sprinkler"] * 4
                + samples["Rain"] * 2
                + samples["WetGrass"]
            )
            emp = np.bincount(keys, minlength=16) / n
            # Exact joint probability of every configuration.
            exact_joint = np.array(
                [_joint_product(dict(zip(_NODES, combo))) for combo in combos]
            )
            pos = np.arange(16)
            ax3.bar(
                pos,
                emp,
                color=_EMPIRICAL_COLOR,
                edgecolor="black",
                alpha=0.85,
                label="estimate",
            )
            ax3.plot(
                pos,
                exact_joint,
                "o",
                color=_REFERENCE_COLOR,
                markersize=4,
                label="exact joint",
            )
            ax3.set_xticks(pos)
            ax3.set_xticklabels(labels, rotation=90, fontsize=6)
            ax3.set_ylabel("P(C,S,R,W)")
            ax3.set_title("Joint frequencies", fontsize=13, fontweight="bold")
            ax3.legend(fontsize=9)
            # Panel 4: comments.
            ax4.axis("off")
            ax4.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
            joint_err = float(np.max(np.abs(emp - exact_joint)))
            text_content = (
                f"N = {n} full events\n\n"
                f"Topological order:\n"
                f"  C -> S, R -> W\n\n"
                f"Marginal P({_SHORT[track]}=T):\n"
                f"  estimate = {est_true:.4f}\n"
                f"  exact    = {exact[1]:.4f}\n\n"
                f"Max joint error:\n"
                f"  {joint_err:.4f}\n\n"
                f"Relative frequency of an\n"
                f"event approximates its\n"
                f"joint probability."
            )
            htutori.add_fitted_text_box(
                ax4, text_content, max_fontsize=12, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    n_exp_slider.observe(update_plot, names="value")
    var_dd.observe(update_plot, names="value")
    seed_slider.observe(update_plot, names="value")
    update_plot()
    display(ipywidgets.VBox([seed_box, n_box, var_dd, output]))


# #############################################################################
# Cell 1.4: Consistency and the 1/sqrt(N) Convergence Rate
# #############################################################################


def cell1_4_convergence_widget(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Make Monte Carlo convergence and its 1/sqrt(N) rate tangible.

    Shows one estimate settling toward the exact value, a fan of independent
    chains narrowing as N grows, and the absolute error on log-log axes with a
    reference slope of -1/2.

    :param figsize: Optional figure size for the 1x4 panel layout
    """
    if figsize is None:
        figsize = (22, 5)
    model = _build_sprinkler_network()
    seed_slider, seed_box = _add_seed_control()
    n_exp_slider, n_box = htutori.build_log_widget_control(
        name="log(N)",
        description="max N",
        min_exp=4,
        max_exp=15,
        initial_exp=12,
        base=2,
    )
    reps_slider, reps_box = htutori.build_widget_control(
        name="reps",
        description="independent chains",
        min_val=2,
        max_val=30,
        step=1,
        initial_value=12,
        is_float=False,
    )
    # Dropdown picks which marginal event is being estimated.
    event_dd = ipywidgets.Dropdown(
        options=[
            ("P(Rain=T)", ("Rain", 1)),
            ("P(Cloudy=T)", ("Cloudy", 1)),
            ("P(Sprinkler=T)", ("Sprinkler", 1)),
            ("P(WetGrass=T)", ("WetGrass", 1)),
        ],
        value=("Rain", 1),
        description="Estimate:",
        style={"description_width": "initial"},
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Recompute the running estimates and the error curve.
        """
        _ = change
        with output:
            clear_output(wait=True)
            max_n = 2**n_exp_slider.value
            reps = int(reps_slider.value)
            var, val = event_dd.value
            exact = float(_exact_marginal(model, var, {})[val])
            base_seed = int(seed_slider.value)
            # Build one running-estimate chain per independent seed.
            chains = []
            for k in range(reps):
                rng = np.random.default_rng(base_seed + k)
                indicator = (_prior_sample_array(rng, max_n)[var] == val).astype(
                    float
                )
                running = np.cumsum(indicator) / np.arange(1, max_n + 1)
                chains.append(running)
            chains = np.array(chains)
            steps = np.arange(1, max_n + 1)
            _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
            # Panel 1: a single estimate's trajectory vs the exact value.
            ax1.plot(
                steps,
                chains[0],
                color=_EMPIRICAL_COLOR,
                linewidth=1.5,
                label="running estimate",
            )
            ax1.axhline(
                exact,
                color=_REFERENCE_COLOR,
                linestyle=":",
                linewidth=2,
                label="exact",
            )
            ax1.set_xscale("log")
            ax1.set_xlabel("N (log scale)")
            ax1.set_ylabel(f"P({_SHORT[var]}={'T' if val else 'F'})")
            ax1.set_title("One estimate", fontsize=13, fontweight="bold")
            ax1.legend(fontsize=9)
            # Panel 2: the fan of all chains narrowing toward the exact value.
            for chain in chains:
                ax2.plot(
                    steps,
                    chain,
                    color=_EMPIRICAL_COLOR,
                    alpha=0.3,
                    linewidth=0.8,
                )
            ax2.axhline(
                exact,
                color=_REFERENCE_COLOR,
                linestyle=":",
                linewidth=2,
                label="exact",
            )
            ax2.set_xscale("log")
            ax2.set_xlabel("N (log scale)")
            ax2.set_ylabel("estimate")
            ax2.set_title(
                f"{reps} independent chains", fontsize=13, fontweight="bold"
            )
            ax2.legend(fontsize=9)
            # Panel 3: RMS error vs N on log-log axes with a -1/2 reference.
            rms_err = np.sqrt(np.mean((chains - exact) ** 2, axis=0))
            ax3.loglog(
                steps,
                rms_err,
                color=_EMPIRICAL_COLOR,
                linewidth=1.5,
                label="RMS error",
            )
            # Reference line proportional to 1/sqrt(N), anchored at the start.
            anchor = rms_err[0] if rms_err[0] > 0 else 1.0
            ref = anchor / np.sqrt(steps)
            ax3.loglog(
                steps,
                ref,
                color=_REFERENCE_COLOR,
                linestyle=":",
                linewidth=2,
                label="slope -1/2",
            )
            ax3.set_xlabel("N (log scale)")
            ax3.set_ylabel("error (log scale)")
            ax3.set_title("Error vs N", fontsize=13, fontweight="bold")
            ax3.legend(fontsize=9)
            # Panel 4: comments.
            ax4.axis("off")
            ax4.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
            final_err = float(rms_err[-1])
            text_content = (
                f"Event: P({_SHORT[var]}=T)\n"
                f"Exact value: {exact:.4f}\n\n"
                f"Chains: {reps}\n"
                f"Max N: {max_n}\n\n"
                f"RMS error at max N:\n"
                f"  {final_err:.4f}\n\n"
                f"Error shrinks like\n"
                f"  1 / sqrt(N)\n\n"
                f"10x accuracy costs\n"
                f"100x samples."
            )
            htutori.add_fitted_text_box(
                ax4, text_content, max_fontsize=12, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    n_exp_slider.observe(update_plot, names="value")
    reps_slider.observe(update_plot, names="value")
    event_dd.observe(update_plot, names="value")
    seed_slider.observe(update_plot, names="value")
    update_plot()
    display(ipywidgets.VBox([seed_box, n_box, reps_box, event_dd, output]))


# #############################################################################
# Cell 2.1: Rejection Sampling
# #############################################################################


def cell2_1_rejection_sampling_widget(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Condition on evidence by discarding prior samples that disagree with it.

    Shows the kept-vs-rejected sample stream, the retained-fraction funnel, and
    the estimated posterior against the exact `pgmpy` reference, exposing how
    rare evidence wastes most of the work.

    :param figsize: Optional figure size for the 1x4 panel layout
    """
    if figsize is None:
        figsize = (22, 5)
    model = _build_sprinkler_network()
    seed_slider, seed_box = _add_seed_control()
    n_exp_slider, n_box = htutori.build_log_widget_control(
        name="log(N)",
        description="N (prior samples)",
        min_exp=4,
        max_exp=14,
        initial_exp=9,
        base=2,
    )
    query_dd, checks, valdds, rows = _build_query_controls()
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """Generate prior samples, filter by evidence, and estimate the
        posterior.
        """
        _ = change
        with output:
            clear_output(wait=True)
            n = 2**n_exp_slider.value
            query_var, evidence = _read_query(query_dd, checks, valdds)
            rng = np.random.default_rng(seed_slider.value)
            samples = _prior_sample_array(rng, n)
            # A sample is kept only if it matches every observed value.
            keep_mask = np.ones(n, dtype=bool)
            for var, val in evidence.items():
                keep_mask &= samples[var] == val
            n_kept = int(keep_mask.sum())
            n_rej = n - n_kept
            _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
            # Panel 1: a stream of dots colored by kept vs rejected.
            show = min(n, 400)
            xs = np.arange(show) % 25
            ys = np.arange(show) // 25
            kept_show = keep_mask[:show]
            ax1.scatter(
                xs[~kept_show],
                ys[~kept_show],
                color="#D5D8DC",
                s=20,
                label="rejected",
            )
            ax1.scatter(
                xs[kept_show],
                ys[kept_show],
                color=_EMPIRICAL_COLOR,
                s=24,
                label="kept",
            )
            ax1.set_title(
                f"Sample stream (first {show})", fontsize=13, fontweight="bold"
            )
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.legend(fontsize=9, loc="upper right")
            ax1.invert_yaxis()
            # Panel 2: the retained-fraction funnel as a bar count.
            ax2.bar(
                ["generated", "rejected", "kept"],
                [n, n_rej, n_kept],
                color=["#85929E", "#D5D8DC", _EMPIRICAL_COLOR],
                edgecolor="black",
            )
            ax2.set_title("Retained fraction", fontsize=13, fontweight="bold")
            ax2.set_ylabel("count")
            for i, v in enumerate([n, n_rej, n_kept]):
                ax2.text(i, v, f"{v}", ha="center", va="bottom", fontsize=9)
            # Panel 3: estimated posterior vs the exact reference.
            exact = _exact_marginal(model, query_var, evidence)
            if n_kept > 0:
                q_kept = samples[query_var][keep_mask]
                est_true = float(np.mean(q_kept == 1))
            else:
                est_true = 0.0
            estimate = [1 - est_true, est_true]
            x = np.arange(2)
            ax3.bar(
                x,
                estimate,
                0.55,
                color=_EMPIRICAL_COLOR,
                edgecolor="black",
                label="estimate",
            )
            ax3.bar(
                x,
                exact,
                0.28,
                color="none",
                edgecolor=_REFERENCE_COLOR,
                linestyle=":",
                linewidth=2,
                label="exact (pgmpy)",
            )
            ax3.set_xticks(x)
            ax3.set_xticklabels(
                [f"{_SHORT[query_var]}=F", f"{_SHORT[query_var]}=T"]
            )
            ax3.set_ylim([0, 1.05])
            ax3.set_ylabel(f"P({_SHORT[query_var]} | e)")
            ax3.set_title("Posterior estimate", fontsize=13, fontweight="bold")
            ax3.legend(fontsize=9)
            # Panel 4: comments.
            ax4.axis("off")
            ax4.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
            frac = n_kept / n if n > 0 else 0.0
            text_content = (
                f"Query: P({_SHORT[query_var]} | {_fmt_evidence(evidence)})\n\n"
                f"Generated: {n}\n"
                f"Rejected:  {n_rej}\n"
                f"Kept:      {n_kept}\n\n"
                f"Retained fraction:\n"
                f"  {frac:.3f} ~= P(e)\n\n"
                f"Estimate P({_SHORT[query_var]}=T | e):\n"
                f"  {est_true:.4f}\n"
                f"Exact: {exact[1]:.4f}\n\n"
                f"Rarer evidence -> more\n"
                f"samples burned."
            )
            htutori.add_fitted_text_box(
                ax4, text_content, max_fontsize=12, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    n_exp_slider.observe(update_plot, names="value")
    query_dd.observe(update_plot, names="value")
    for n in _NODES:
        checks[n].observe(update_plot, names="value")
        valdds[n].observe(update_plot, names="value")
    seed_slider.observe(update_plot, names="value")
    update_plot()
    display(ipywidgets.VBox([seed_box, n_box, query_dd, *rows, output]))


# #############################################################################
# Cell 2.2: Importance Sampling and Likelihood Weighting
# #############################################################################


def cell2_2_likelihood_weighting_widget(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Keep every sample and correct the bias with importance weights.

    Shows likelihood-weighted samples sized by weight, the weight distribution
    that flags weight collapse, and the weighted posterior estimate against the
    exact reference and the rejection-sampling estimate at the same N.

    :param figsize: Optional figure size for the 1x4 panel layout
    """
    if figsize is None:
        figsize = (22, 5)
    model = _build_sprinkler_network()
    seed_slider, seed_box = _add_seed_control()
    n_exp_slider, n_box = htutori.build_log_widget_control(
        name="log(N)",
        description="N (weighted samples)",
        min_exp=4,
        max_exp=14,
        initial_exp=9,
        base=2,
    )
    query_dd, checks, valdds, rows = _build_query_controls()
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Draw likelihood-weighted samples and form the weighted estimate.
        """
        _ = change
        with output:
            clear_output(wait=True)
            n = 2**n_exp_slider.value
            query_var, evidence = _read_query(query_dd, checks, valdds)
            rng = np.random.default_rng(seed_slider.value)
            samples, weights = _likelihood_weight_array(rng, n, evidence)
            _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
            # Panel 1: samples laid out as dots sized by their weight.
            show = min(n, 400)
            xs = np.arange(show) % 25
            ys = np.arange(show) // 25
            q_show = samples[query_var][:show]
            w_show = weights[:show]
            # Scale dot area by weight relative to the largest shown weight.
            sizes = 10 + 120 * (w_show / (w_show.max() + 1e-12))
            colors = np.where(q_show == 1, _EMPIRICAL_COLOR, "#AED6F1")
            ax1.scatter(
                xs, ys, s=sizes, c=colors, edgecolor="black", linewidth=0.3
            )
            ax1.set_title(
                f"Weighted samples (first {show})",
                fontsize=13,
                fontweight="bold",
            )
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.invert_yaxis()
            ax1.text(
                0.02,
                -0.08,
                f"dot size ~ weight; dark = {_SHORT[query_var]}=T",
                transform=ax1.transAxes,
                fontsize=8,
            )
            # Panel 2: the weight distribution, flagging weight collapse.
            sns.histplot(weights, bins=40, color=_EMPIRICAL_COLOR, ax=ax2)
            ax2.set_title("Weight distribution", fontsize=13, fontweight="bold")
            ax2.set_xlabel("importance weight w")
            ax2.set_ylabel("count")
            # Effective sample size summarizes how balanced the weights are.
            ess = (weights.sum() ** 2) / (np.sum(weights**2) + 1e-12)
            ax2.text(
                0.95,
                0.95,
                f"ESS = {ess:.0f}\nof {n}",
                transform=ax2.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="#FDEBD0"),
            )
            # Panel 3: weighted estimate vs exact and vs rejection at same N.
            exact = _exact_marginal(model, query_var, evidence)
            w_true = weights[samples[query_var] == 1].sum()
            est_true = float(w_true / (weights.sum() + 1e-12))
            estimate = [1 - est_true, est_true]
            # Rejection estimate on the same prior draw for a fair comparison.
            prior = _prior_sample_array(
                np.random.default_rng(seed_slider.value), n
            )
            keep = np.ones(n, dtype=bool)
            for var, val in evidence.items():
                keep &= prior[var] == val
            if keep.sum() > 0:
                rej_true = float(np.mean(prior[query_var][keep] == 1))
            else:
                rej_true = 0.0
            x = np.arange(2)
            width = 0.28
            ax3.bar(
                x - width,
                estimate,
                width,
                color=_EMPIRICAL_COLOR,
                edgecolor="black",
                label="likelihood weighting",
            )
            ax3.bar(
                x,
                [1 - rej_true, rej_true],
                width,
                color="#85929E",
                edgecolor="black",
                label="rejection",
            )
            ax3.bar(
                x + width,
                exact,
                width,
                color="none",
                edgecolor=_REFERENCE_COLOR,
                linestyle=":",
                linewidth=2,
                label="exact",
            )
            ax3.set_xticks(x)
            ax3.set_xticklabels(
                [f"{_SHORT[query_var]}=F", f"{_SHORT[query_var]}=T"]
            )
            ax3.set_ylim([0, 1.05])
            ax3.set_ylabel(f"P({_SHORT[query_var]} | e)")
            ax3.set_title("Posterior estimate", fontsize=13, fontweight="bold")
            ax3.legend(fontsize=8)
            # Panel 4: comments.
            ax4.axis("off")
            ax4.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
            text_content = (
                f"Query: P({_SHORT[query_var]} | {_fmt_evidence(evidence)})\n\n"
                f"N = {n} (all kept)\n"
                f"Effective sample size:\n"
                f"  ESS = {ess:.0f}\n\n"
                f"Weighted estimate:\n"
                f"  {est_true:.4f}\n"
                f"Rejection estimate:\n"
                f"  {rej_true:.4f}\n"
                f"Exact: {exact[1]:.4f}\n\n"
                f"Every sample is kept and\n"
                f"reweighted; low ESS means\n"
                f"weight collapse."
            )
            htutori.add_fitted_text_box(
                ax4, text_content, max_fontsize=12, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    n_exp_slider.observe(update_plot, names="value")
    query_dd.observe(update_plot, names="value")
    for n in _NODES:
        checks[n].observe(update_plot, names="value")
        valdds[n].observe(update_plot, names="value")
    seed_slider.observe(update_plot, names="value")
    update_plot()
    display(ipywidgets.VBox([seed_box, n_box, query_dd, *rows, output]))


# #############################################################################
# Cell 3.1: Markov Chains and the Stationary Distribution
# #############################################################################

# A small ergodic transition matrix over four abstract states used to build
# intuition for convergence to a stationary distribution.
_MC_STATES = ["s0", "s1", "s2", "s3"]
_MC_TRANSITION = np.array(
    [
        [0.50, 0.30, 0.15, 0.05],
        [0.20, 0.45, 0.25, 0.10],
        [0.10, 0.25, 0.45, 0.20],
        [0.05, 0.15, 0.30, 0.50],
    ]
)


def _stationary_distribution(transition: np.ndarray) -> np.ndarray:
    """
    Compute the stationary distribution of a transition matrix.

    Solves for the left eigenvector with eigenvalue 1 and normalizes it.

    :param transition: Row-stochastic transition matrix
    :return: Stationary distribution as a probability vector
    """
    eigvals, eigvecs = np.linalg.eig(transition.T)
    # The stationary distribution is the eigenvector for eigenvalue 1.
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    vec = np.real(eigvecs[:, idx])
    return vec / vec.sum()


def cell3_1_markov_chain_widget(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Show a random walk over states converging to a stationary distribution.

    Steps a Markov chain forward, plots the evolving state distribution against
    its stationary target, and tracks the total-variation distance shrinking to
    zero regardless of the starting state.

    :param figsize: Optional figure size for the 1x4 panel layout
    """
    if figsize is None:
        figsize = (22, 5)
    stationary = _stationary_distribution(_MC_TRANSITION)
    seed_slider, seed_box = _add_seed_control()
    t_slider, t_box = htutori.build_widget_control(
        name="t",
        description="number of steps",
        min_val=0,
        max_val=40,
        step=1,
        initial_value=8,
        is_float=False,
    )
    # Dropdown chooses the starting state to show the limit is start-independent.
    init_dd = ipywidgets.Dropdown(
        options=_MC_STATES,
        value="s0",
        description="Initial state:",
        style={"description_width": "initial"},
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Step the chain to time t and compare with the stationary target.
        """
        _ = change
        with output:
            clear_output(wait=True)
            t = int(t_slider.value)
            init_idx = _MC_STATES.index(init_dd.value)
            # Start from a one-hot distribution on the chosen initial state.
            pi0 = np.zeros(len(_MC_STATES))
            pi0[init_idx] = 1.0
            # Distribution at time t is pi0 times the t-th matrix power.
            pi_t = pi0 @ np.linalg.matrix_power(_MC_TRANSITION, t)
            # Total-variation distance to the stationary target over time.
            tv = []
            cur = pi0.copy()
            for _ in range(41):
                tv.append(0.5 * np.sum(np.abs(cur - stationary)))
                cur = cur @ _MC_TRANSITION
            _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
            # Panel 1: the state-transition diagram with the current state lit.
            g = nx.DiGraph()
            for i, si in enumerate(_MC_STATES):
                for j, sj in enumerate(_MC_STATES):
                    if _MC_TRANSITION[i, j] >= 0.15:
                        g.add_edge(si, sj, weight=_MC_TRANSITION[i, j])
            pos = nx.circular_layout(g)
            # Mark the most probable current state as the walk's location.
            cur_state = int(np.argmax(pi_t))
            node_colors = [
                _EMPIRICAL_COLOR if k == cur_state else "#D5D8DC"
                for k in range(len(_MC_STATES))
            ]
            nx.draw(
                g,
                pos,
                ax=ax1,
                node_color=node_colors,
                node_size=900,
                with_labels=True,
                arrows=True,
                edge_color="#85929E",
            )
            ax1.set_title("Transition diagram", fontsize=13, fontweight="bold")
            # Panel 2: evolving distribution (solid) vs stationary (dotted).
            x = np.arange(len(_MC_STATES))
            ax2.bar(
                x,
                pi_t,
                0.55,
                color=_EMPIRICAL_COLOR,
                edgecolor="black",
                label=f"pi_t (t={t})",
            )
            ax2.bar(
                x,
                stationary,
                0.28,
                color="none",
                edgecolor=_REFERENCE_COLOR,
                linestyle=":",
                linewidth=2,
                label="stationary",
            )
            ax2.set_xticks(x)
            ax2.set_xticklabels(_MC_STATES)
            ax2.set_ylim([0, 1.0])
            ax2.set_ylabel("probability")
            ax2.set_title("State distribution", fontsize=13, fontweight="bold")
            ax2.legend(fontsize=9)
            # Panel 3: total-variation distance decaying toward zero.
            ax3.plot(
                np.arange(41),
                tv,
                color=_EMPIRICAL_COLOR,
                linewidth=2,
            )
            ax3.scatter([t], [tv[t]], color=_REFERENCE_COLOR, zorder=5, s=50)
            ax3.set_xlabel("step t")
            ax3.set_ylabel("TV distance to stationary")
            ax3.set_title("Convergence", fontsize=13, fontweight="bold")
            ax3.grid(True, alpha=0.3)
            # Panel 4: comments.
            ax4.axis("off")
            ax4.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
            text_content = (
                f"Steps: t = {t}\n"
                f"Start: {init_dd.value}\n\n"
                f"TV distance at t:\n"
                f"  {tv[t]:.4f}\n\n"
                f"Stationary distribution:\n"
                + "".join(
                    f"  {s}: {p:.3f}\n" for s, p in zip(_MC_STATES, stationary)
                )
                + "\nThe limit is the same for\n"
                "any starting state."
            )
            htutori.add_fitted_text_box(
                ax4, text_content, max_fontsize=12, min_fontsize=8
            )
            plt.tight_layout()
            plt.show()

    t_slider.observe(update_plot, names="value")
    init_dd.observe(update_plot, names="value")
    seed_slider.observe(update_plot, names="value")
    update_plot()
    display(
        ipywidgets.VBox(
            [
                seed_box,
                t_box,
                init_dd,
                output,
            ]
        )
    )


# #############################################################################
# Cell 3.2: Mixing and Burn-in
# #############################################################################


def _bimodal_density(x: np.ndarray) -> np.ndarray:
    """
    Evaluate the unnormalized bimodal target density used for mixing demos.

    The target is an equal mixture of two well-separated Gaussians.

    :param x: Points at which to evaluate the density
    :return: Density values at `x`
    """
    # Two modes at -2 and +2 with unit variance.
    left = np.exp(-0.5 * (x + 2.0) ** 2)
    right = np.exp(-0.5 * (x - 2.0) ** 2)
    return 0.5 * (left + right)


def _metropolis_1d(rng: np.random.Generator, n: int, step: float) -> np.ndarray:
    """
    Run a 1-D random-walk Metropolis sampler on the bimodal target.

    :param rng: NumPy random generator
    :param n: Number of iterations
    :param step: Proposal standard deviation controlling mixing
    :return: Array of sampled states, one per iteration
    """
    samples = np.empty(n)
    # Start deliberately inside the right mode to expose poor mixing.
    cur = 2.0
    cur_p = _bimodal_density(np.array([cur]))[0]
    for i in range(n):
        # Propose a Gaussian step and accept by the Metropolis ratio.
        prop = cur + rng.normal(0, step)
        prop_p = _bimodal_density(np.array([prop]))[0]
        if rng.random() < prop_p / (cur_p + 1e-300):
            cur, cur_p = prop, prop_p
        samples[i] = cur
    return samples


def _acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """
    Compute the normalized autocorrelation function up to `nlags`.

    :param x: Input series
    :param nlags: Maximum lag to compute
    :return: Autocorrelation values for lags 0 to `nlags`
    """
    x = x - np.mean(x)
    var = np.sum(x**2)
    acf = np.empty(nlags + 1)
    # Correlate the series with lagged copies of itself.
    for lag in range(nlags + 1):
        if lag == 0:
            acf[lag] = 1.0
        else:
            acf[lag] = np.sum(x[:-lag] * x[lag:]) / (var + 1e-300)
    return acf


def cell3_2_mixing_burnin_widget(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Show that a correct stationary distribution is not enough without mixing.

    Contrasts well-mixed and poorly-mixed Metropolis chains on a bimodal target
    via the trace, the collected histogram, and the autocorrelation, and marks
    the discarded burn-in region.

    :param figsize: Optional figure size for the 1x4 panel layout
    """
    if figsize is None:
        figsize = (22, 5)
    seed_slider, seed_box = _add_seed_control()
    step_slider, step_box = htutori.build_widget_control(
        name="step",
        description="proposal step size",
        min_val=0.1,
        max_val=4.0,
        step=0.1,
        initial_value=0.4,
        is_float=True,
    )
    burnin_slider, burnin_box = htutori.build_widget_control(
        name="burnin",
        description="burn-in length",
        min_val=0,
        max_val=2000,
        step=50,
        initial_value=200,
        is_float=False,
    )
    n_exp_slider, n_box = htutori.build_log_widget_control(
        name="log(N)",
        description="total iterations",
        min_exp=8,
        max_exp=14,
        initial_exp=12,
        base=2,
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Run the Metropolis chain and visualize its mixing behavior.
        """
        _ = change
        with output:
            clear_output(wait=True)
            step = step_slider.value
            n = 2**n_exp_slider.value
            burnin = min(int(burnin_slider.value), n - 1)
            rng = np.random.default_rng(seed_slider.value)
            chain = _metropolis_1d(rng, n, step)
            kept = chain[burnin:]
            _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
            # Panel 1: the trace plot with the burn-in region shaded.
            show = min(n, 2000)
            ax1.plot(
                np.arange(show),
                chain[:show],
                color=_EMPIRICAL_COLOR,
                linewidth=0.7,
            )
            if burnin > 0:
                ax1.axvspan(
                    0,
                    min(burnin, show),
                    color="#F5B7B1",
                    alpha=0.4,
                    label="burn-in",
                )
                ax1.legend(fontsize=9)
            ax1.set_title(
                f"Trace (first {show})", fontsize=13, fontweight="bold"
            )
            ax1.set_xlabel("iteration")
            ax1.set_ylabel("state")
            # Panel 2: histogram of kept samples vs the true bimodal density.
            sns.histplot(
                kept,
                bins=60,
                stat="density",
                color=_EMPIRICAL_COLOR,
                alpha=0.6,
                ax=ax2,
                label="samples",
            )
            grid = np.linspace(-6, 6, 400)
            dens = _bimodal_density(grid)
            # Normalize the target density for a fair overlay.
            dens = dens / (np.sum(dens) * (grid[1] - grid[0]))
            ax2.plot(
                grid,
                dens,
                color=_REFERENCE_COLOR,
                linestyle=":",
                linewidth=2.5,
                label="true posterior",
            )
            ax2.set_title("Collected samples", fontsize=13, fontweight="bold")
            ax2.set_xlabel("state")
            ax2.set_ylabel("density")
            ax2.legend(fontsize=9)
            # Panel 3: autocorrelation vs lag.
            nlags = 50
            acf = _acf(kept, nlags)
            ax3.bar(
                np.arange(nlags + 1),
                acf,
                color=_EMPIRICAL_COLOR,
                edgecolor="black",
            )
            ax3.axhline(0, color="black", linewidth=0.8)
            ax3.set_title("Autocorrelation", fontsize=13, fontweight="bold")
            ax3.set_xlabel("lag")
            ax3.set_ylabel("autocorrelation")
            # Panel 4: comments.
            ax4.axis("off")
            ax4.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
            # Fraction of samples landing in the left mode diagnoses mixing.
            left_frac = float(np.mean(kept < 0))
            quality = "good" if step >= 1.5 else "poor"
            text_content = (
                f"Step size: {step:.2f} ({quality})\n"
                f"Burn-in: {burnin}\n"
                f"Total iters: {n}\n\n"
                f"Left-mode fraction:\n"
                f"  {left_frac:.3f} (target 0.5)\n\n"
                f"Lag-1 autocorrelation:\n"
                f"  {acf[1]:.3f}\n\n"
                f"Small steps stay stuck in\n"
                f"one mode; large steps mix\n"
                f"and visit both."
            )
            htutori.add_fitted_text_box(
                ax4, text_content, max_fontsize=12, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    step_slider.observe(update_plot, names="value")
    burnin_slider.observe(update_plot, names="value")
    n_exp_slider.observe(update_plot, names="value")
    seed_slider.observe(update_plot, names="value")
    update_plot()
    display(
        ipywidgets.VBox(
            [
                seed_box,
                step_box,
                burnin_box,
                n_box,
                output,
            ]
        )
    )


# #############################################################################
# Cell 3.3: Gibbs Sampling and the Markov Blanket
# #############################################################################


def _markov_blanket(graph: nx.DiGraph, node: str) -> List[str]:
    """
    Compute the Markov blanket of a node: parents, children, and co-parents.

    :param graph: The network as a directed graph
    :param node: Node whose blanket is requested
    :return: Sorted list of blanket node names
    """
    parents = set(graph.predecessors(node))
    children = set(graph.successors(node))
    # Co-parents are the other parents of this node's children.
    coparents = set()
    for child in children:
        coparents |= set(graph.predecessors(child))
    blanket = (parents | children | coparents) - {node}
    return sorted(blanket)


def _gibbs_full_conditional(var: str, state: Dict[str, int]) -> np.ndarray:
    """
    Compute the full conditional P(var | everything else) for a binary node.

    Uses the joint product with the variable set to each value, then normalizes.

    :param var: Variable to resample
    :param state: Current assignment of all nodes
    :return: Array `[P(var=0 | rest), P(var=1 | rest)]`
    """
    probs = np.empty(2)
    for val in (0, 1):
        trial = dict(state)
        trial[var] = val
        probs[val] = _joint_product(trial)
    return probs / probs.sum()


def _gibbs_chain(
    rng: np.random.Generator,
    n_sweeps: int,
    evidence: Dict[str, int],
    order: List[str],
) -> Dict[str, np.ndarray]:
    """
    Run a Gibbs sampler that clamps evidence and resamples hidden variables.

    Each sweep resamples every non-evidence variable from its full conditional,
    recording the full state after each sweep.

    :param rng: NumPy random generator
    :param n_sweeps: Number of complete sweeps
    :param evidence: Map from observed node names to their clamped values
    :param order: Order in which to resample the non-evidence variables
    :return: Map from node name to the per-sweep array of values
    """
    # Initialize hidden variables randomly; clamp evidence to its values.
    state = {}
    for node in _NODES:
        if node in evidence:
            state[node] = evidence[node]
        else:
            state[node] = int(rng.random() < 0.5)
    history = {node: np.empty(n_sweeps, dtype=int) for node in _NODES}
    for sweep in range(n_sweeps):
        # Resample each non-evidence variable from its full conditional.
        for var in order:
            cond = _gibbs_full_conditional(var, state)
            state[var] = int(rng.random() < cond[1])
        for node in _NODES:
            history[node][sweep] = state[node]
    return history


def cell3_3_gibbs_sampling_widget(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Specialize MCMC to Bayesian networks via Gibbs sampling.

    Holds the evidence clamped, resamples each hidden variable from its Markov
    blanket, and tracks the running posterior estimate against the exact
    `pgmpy` reference.

    :param figsize: Optional figure size for the 1x4 panel layout
    """
    if figsize is None:
        figsize = (22, 5)
    model = _build_sprinkler_network()
    graph = nx.DiGraph(_EDGES)
    seed_slider, seed_box = _add_seed_control()
    sweeps_slider, sweeps_box = htutori.build_log_widget_control(
        name="log(sweeps)",
        description="Gibbs sweeps",
        min_exp=4,
        max_exp=14,
        initial_exp=10,
        base=2,
    )
    burnin_slider, burnin_box = htutori.build_widget_control(
        name="burnin",
        description="burn-in sweeps",
        min_val=0,
        max_val=500,
        step=10,
        initial_value=50,
        is_float=False,
    )
    # Default evidence clamps both Sprinkler and WetGrass to True.
    query_dd, checks, valdds, rows = _build_query_controls(
        default_query="Rain",
        default_evidence={"Sprinkler": 1, "WetGrass": 1},
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """Run the Gibbs sampler and compare its estimate with the exact
        value.
        """
        _ = change
        with output:
            clear_output(wait=True)
            n_sweeps = 2**sweeps_slider.value
            query_var, evidence = _read_query(query_dd, checks, valdds)
            # Hidden variables are everything not observed; they are resampled.
            hidden = [n for n in _NODES if n not in evidence]
            burnin = min(int(burnin_slider.value), n_sweeps - 1)
            rng = np.random.default_rng(seed_slider.value)
            history = _gibbs_chain(rng, n_sweeps, evidence, hidden)
            _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
            # Panel 1: the DAG with evidence frozen and a blanket highlighted.
            resampled = query_var if query_var in hidden else hidden[0]
            blanket = _markov_blanket(graph, resampled)
            node_colors = {}
            for n in _NODES:
                if n in evidence:
                    node_colors[n] = "#F5B7B1"
                elif n == resampled:
                    node_colors[n] = _EMPIRICAL_COLOR
                elif n in blanket:
                    node_colors[n] = "#F9E79F"
                else:
                    node_colors[n] = "#D5D8DC"
            hgraphv.plot_causal_dag(
                graph,
                f"Resample {resampled} from its blanket",
                node_colors=node_colors,
                ax=ax1,
            )
            # Panel 2: the full conditional being sampled at the last state.
            last_state = {n: int(history[n][-1]) for n in _NODES}
            cond = _gibbs_full_conditional(resampled, last_state)
            ax2.bar(
                [0, 1],
                cond,
                0.55,
                color=_EMPIRICAL_COLOR,
                edgecolor="black",
            )
            ax2.set_xticks([0, 1])
            ax2.set_xticklabels(
                [f"{_SHORT[resampled]}=F", f"{_SHORT[resampled]}=T"]
            )
            ax2.set_ylim([0, 1.05])
            ax2.set_ylabel("conditional probability")
            ax2.set_title(
                f"P({_SHORT[resampled]} | blanket)",
                fontsize=12,
                fontweight="bold",
            )
            ax2.text(
                0.5,
                0.92,
                "MB: " + ", ".join(_SHORT[b] for b in blanket),
                transform=ax2.transAxes,
                ha="center",
                fontsize=9,
            )
            # Panel 3: running posterior estimate vs the exact reference.
            q_series = history[query_var][burnin:]
            running = np.cumsum(q_series) / np.arange(1, len(q_series) + 1)
            exact = _exact_marginal(model, query_var, evidence)
            ax3.plot(
                np.arange(len(running)),
                running,
                color=_EMPIRICAL_COLOR,
                linewidth=1.2,
                label="estimate",
            )
            ax3.axhline(
                exact[1],
                color=_REFERENCE_COLOR,
                linestyle=":",
                linewidth=2,
                label="exact",
            )
            ax3.set_ylim([0, 1.05])
            ax3.set_xlabel("sweep (after burn-in)")
            ax3.set_ylabel(f"P({_SHORT[query_var]}=T | e)")
            ax3.set_title("Running estimate", fontsize=13, fontweight="bold")
            ax3.legend(fontsize=9)
            # Panel 4: comments.
            ax4.axis("off")
            ax4.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
            final_est = float(running[-1]) if len(running) else 0.0
            text_content = (
                f"Query: P({_SHORT[query_var]} | {_fmt_evidence(evidence)})\n\n"
                f"Sweeps: {n_sweeps}\n"
                f"Burn-in: {burnin}\n"
                f"Hidden: {', '.join(_SHORT[h] for h in hidden)}\n\n"
                f"Estimate: {final_est:.4f}\n"
                f"Exact:    {exact[1]:.4f}\n\n"
                f"Evidence stays clamped, so\n"
                f"every sample agrees with it.\n"
                f"Only the Markov blanket is\n"
                f"needed per update."
            )
            htutori.add_fitted_text_box(
                ax4, text_content, max_fontsize=12, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    sweeps_slider.observe(update_plot, names="value")
    burnin_slider.observe(update_plot, names="value")
    query_dd.observe(update_plot, names="value")
    for n in _NODES:
        checks[n].observe(update_plot, names="value")
        valdds[n].observe(update_plot, names="value")
    seed_slider.observe(update_plot, names="value")
    update_plot()
    display(
        ipywidgets.VBox(
            [
                seed_box,
                sweeps_box,
                burnin_box,
                query_dd,
                *rows,
                output,
            ]
        )
    )


# #############################################################################
# Cell 3.4: Metropolis-Hastings and Accept/Reject Moves
# #############################################################################


def _mh_chain(
    rng: np.random.Generator,
    n_iters: int,
    evidence: Dict[str, int],
    p_local: float,
) -> Tuple[Dict[str, np.ndarray], float, Dict[str, int], Dict[str, int], float]:
    """
    Run a Metropolis-Hastings sampler over the non-evidence variables.

    The proposal flips a single hidden variable (local move) with probability
    `p_local`, otherwise flips all hidden variables (a broad jump). Both moves
    are symmetric, so the Hastings ratio reduces to the posterior ratio.

    :param rng: NumPy random generator
    :param n_iters: Number of iterations
    :param evidence: Map from observed node names to their clamped values
    :param p_local: Probability of proposing a local single-variable move
    :return: Tuple of
        - history: per-iteration array of values for each node
        - accept_rate: fraction of proposals accepted
        - last_current: the final accepted state
        - last_proposed: the last proposed state
        - last_accept_prob: the acceptance probability of the last proposal
    """
    hidden = [n for n in _NODES if n not in evidence]
    # Initialize hidden variables; evidence stays clamped throughout.
    state = {}
    for node in _NODES:
        state[node] = (
            evidence[node] if node in evidence else int(rng.random() < 0.5)
        )
    history = {node: np.empty(n_iters, dtype=int) for node in _NODES}
    n_accept = 0
    last_proposed = dict(state)
    last_accept_prob = 1.0
    for i in range(n_iters):
        # Propose either a single flip or a flip of all hidden variables.
        proposal = dict(state)
        if rng.random() < p_local:
            flip = hidden[rng.integers(len(hidden))]
            proposal[flip] = 1 - proposal[flip]
        else:
            for node in hidden:
                proposal[node] = 1 - proposal[node]
        # The posterior over hidden vars is proportional to the joint.
        cur_p = _joint_product(state)
        prop_p = _joint_product(proposal)
        accept_prob = min(1.0, prop_p / (cur_p + 1e-300))
        last_proposed = proposal
        last_accept_prob = accept_prob
        # Accept the move with the Metropolis-Hastings acceptance probability.
        if rng.random() < accept_prob:
            state = proposal
            n_accept += 1
        for node in _NODES:
            history[node][i] = state[node]
    accept_rate = n_accept / n_iters
    return history, accept_rate, dict(state), last_proposed, last_accept_prob


def cell3_4_metropolis_hastings_widget(
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Generalize MCMC to Metropolis-Hastings with accept/reject moves.

    Shows the current and proposed states with the acceptance probability, the
    trace and running acceptance rate, and the running posterior estimate
    against the exact `pgmpy` reference.

    :param figsize: Optional figure size for the 1x4 panel layout
    """
    if figsize is None:
        figsize = (22, 5)
    model = _build_sprinkler_network()
    # Fix the canonical query P(Rain | Sprinkler=T).
    query_var = "Rain"
    evidence = {"Sprinkler": 1}
    seed_slider, seed_box = _add_seed_control()
    mix_slider, mix_box = htutori.build_widget_control(
        name="p_local",
        description="prob of local move",
        min_val=0.0,
        max_val=1.0,
        step=0.05,
        initial_value=0.7,
        is_float=True,
    )
    n_exp_slider, n_box = htutori.build_log_widget_control(
        name="log(iters)",
        description="iterations",
        min_exp=6,
        max_exp=14,
        initial_exp=10,
        base=2,
    )
    burnin_slider, burnin_box = htutori.build_widget_control(
        name="burnin",
        description="burn-in iterations",
        min_val=0,
        max_val=1000,
        step=50,
        initial_value=100,
        is_float=False,
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """Run Metropolis-Hastings and visualize proposals, trace, and
        estimate.
        """
        _ = change
        with output:
            clear_output(wait=True)
            p_local = mix_slider.value
            n_iters = 2**n_exp_slider.value
            burnin = min(int(burnin_slider.value), n_iters - 1)
            rng = np.random.default_rng(seed_slider.value)
            history, accept_rate, cur, prop, accept_prob = _mh_chain(
                rng, n_iters, evidence, p_local
            )
            _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
            # Panel 1: current vs proposed state with the acceptance prob.
            ax1.axis("off")
            ax1.set_title("Last proposed move", fontsize=13, fontweight="bold")
            cur_str = ", ".join(
                f"{_SHORT[n]}={'T' if cur[n] else 'F'}" for n in _NODES
            )
            prop_str = ", ".join(
                f"{_SHORT[n]}={'T' if prop[n] else 'F'}" for n in _NODES
            )
            ax1.text(
                0.05,
                0.75,
                "current:",
                fontsize=11,
                fontweight="bold",
                transform=ax1.transAxes,
            )
            ax1.text(
                0.05,
                0.66,
                cur_str,
                fontsize=11,
                transform=ax1.transAxes,
                family="monospace",
            )
            ax1.text(
                0.05,
                0.48,
                "proposed:",
                fontsize=11,
                fontweight="bold",
                transform=ax1.transAxes,
            )
            ax1.text(
                0.05,
                0.39,
                prop_str,
                fontsize=11,
                transform=ax1.transAxes,
                family="monospace",
            )
            ax1.text(
                0.05,
                0.18,
                f"A(x, x') = {accept_prob:.3f}",
                fontsize=13,
                transform=ax1.transAxes,
                bbox=dict(boxstyle="round", facecolor="#FDEBD0"),
            )
            # Panel 2: trace of the query variable and the acceptance rate.
            q_series = history[query_var]
            show = min(n_iters, 1500)
            # A lightly jittered trace makes the 0/1 path readable.
            jitter = history[query_var][:show] + rng.normal(0, 0.04, show)
            ax2.plot(
                np.arange(show),
                jitter,
                color=_EMPIRICAL_COLOR,
                linewidth=0.5,
            )
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(
                [f"{_SHORT[query_var]}=F", f"{_SHORT[query_var]}=T"]
            )
            ax2.set_title(
                f"Trace (accept rate {accept_rate:.2f})",
                fontsize=12,
                fontweight="bold",
            )
            ax2.set_xlabel("iteration")
            # Panel 3: running posterior estimate vs the exact reference.
            kept = q_series[burnin:]
            running = np.cumsum(kept) / np.arange(1, len(kept) + 1)
            exact = _exact_marginal(model, query_var, evidence)
            ax3.plot(
                np.arange(len(running)),
                running,
                color=_EMPIRICAL_COLOR,
                linewidth=1.2,
                label="estimate",
            )
            ax3.axhline(
                exact[1],
                color=_REFERENCE_COLOR,
                linestyle=":",
                linewidth=2,
                label="exact",
            )
            ax3.set_ylim([0, 1.05])
            ax3.set_xlabel("iteration (after burn-in)")
            ax3.set_ylabel(f"P({_SHORT[query_var]}=T | e)")
            ax3.set_title("Running estimate", fontsize=13, fontweight="bold")
            ax3.legend(fontsize=9)
            # Panel 4: comments.
            ax4.axis("off")
            ax4.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
            final_est = float(running[-1]) if len(running) else 0.0
            text_content = (
                f"Query: P({_SHORT[query_var]} | {_fmt_evidence(evidence)})\n\n"
                f"P(local move): {p_local:.2f}\n"
                f"Iterations: {n_iters}\n"
                f"Burn-in: {burnin}\n"
                f"Accept rate: {accept_rate:.3f}\n\n"
                f"Estimate: {final_est:.4f}\n"
                f"Exact:    {exact[1]:.4f}\n\n"
                f"Any proposal is valid once\n"
                f"corrected by A(x, x'). Gibbs\n"
                f"is the always-accept case."
            )
            htutori.add_fitted_text_box(
                ax4, text_content, max_fontsize=12, min_fontsize=9
            )
            plt.tight_layout()
            plt.show()

    mix_slider.observe(update_plot, names="value")
    n_exp_slider.observe(update_plot, names="value")
    burnin_slider.observe(update_plot, names="value")
    seed_slider.observe(update_plot, names="value")
    update_plot()
    display(
        ipywidgets.VBox(
            [
                seed_box,
                mix_box,
                n_box,
                burnin_box,
                output,
            ]
        )
    )
