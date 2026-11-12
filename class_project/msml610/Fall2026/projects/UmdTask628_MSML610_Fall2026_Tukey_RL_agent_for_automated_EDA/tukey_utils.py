"""
Notebook helpers for `tukey.API.ipynb` and `tukey.example.ipynb` (Milestones 1-2).

The notebooks call these functions instead of holding logic inline, so each
notebook cell stays one task with one visible outcome. Functions of the API
notebook are named `cell<part>_<id>_*`, those of the example notebook
`example_cell<part>_<id>_*`.

Import as:

import tukey_utils as utils
"""

import importlib.metadata
import json
import logging
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

import helpers.hgraphviz as hgraphv
import helpers.hnotebook as hnotebo
import tukey_baselines as tbasel
import tukey_env as tenv
import tukey_scoring as tscore
import tukey_tools as ttools

_LOG = logging.getLogger(__name__)

# Resolution of the notebook figures: small enough for the repository's file
# size limit (500 KB) with every figure embedded.
_DPI = 60


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utils logger.

    :param notebook_log: logger owned by the notebook
    """
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# Cell 1.1: Environment check
# #############################################################################


# Packages whose versions the notebooks report, in display order.
_REPORTED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "networkx",
    "causal-learn",
    "lingam",
    "CausalDisco",
    "pgmpy",
)


def cell1_1_get_package_versions(
    *, packages: Tuple[str, ...] = _REPORTED_PACKAGES
) -> pd.DataFrame:
    """
    Report the installed version of each package the project depends on.

    :param packages: distribution names to look up
        - Default: the project's pinned scientific stack
    :return: one row per package with columns `package` and `version`
    """
    rows = []
    for package in packages:
        # Read the version from the installed distribution metadata.
        version = importlib.metadata.version(package)
        rows.append({"package": package, "version": version})
    versions = pd.DataFrame(rows)
    return versions


# #############################################################################
# Cell 2.1: Configure and sample an instance
# #############################################################################


def cell2_1_describe_instance(inst: tenv.Instance) -> pd.DataFrame:
    """
    Summarize an instance's settings and identifiable target.

    :param inst: sampled instance
    :return: one row: size, family, noise, mechanism, target class and reason
    """
    row = {
        "n_nodes": inst.cfg.n_nodes,
        "n_edges": int(inst.truth.B.sum()),
        "family": inst.cfg.family,
        "noise": inst.noise_families[0],
        "mechanism": inst.mechanism,
        "target_class": inst.truth.target_class,
        "target_reason": inst.target_reason,
        "n_train": len(inst.d_train),
        "n_test": len(inst.d_test),
    }
    return pd.DataFrame([row])


# #############################################################################
# Cell 2.2: Plot the hidden graph
# #############################################################################


def cell2_2_plot_graph(
    B: np.ndarray, title: str, *, figsize: Tuple[int, int] = (4, 4)
) -> None:
    """
    Plot a DAG with the course's causal-DAG style.

    :param B: adjacency, `B[i, j] = 1` for x_i -> x_j
    :param title: plot title
    :param figsize: figure size in inches
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(f"x{k}" for k in range(B.shape[0]))
    graph.add_edges_from((f"x{i}", f"x{j}") for i, j in np.argwhere(B != 0))
    hgraphv.plot_causal_dag(graph, title, figsize=figsize)
    plt.show()


# #############################################################################
# Cell 2.3: Score candidate graphs
# #############################################################################


def _reverse_edge(B: np.ndarray, *, covered: bool) -> np.ndarray:
    """
    Reverse the first edge of the requested kind, keeping the graph acyclic.

    Edge i -> j is covered when the parents of j are the parents of i plus i;
    reversing a covered edge keeps the Markov equivalence class (Chickering
    1995), reversing any other edge changes it.

    :param B: DAG adjacency
    :param covered: reverse a covered edge (True) or an uncovered one (False)
    :return: DAG with one edge reversed (a copy of `B` if none qualifies)
    """
    for i, j in np.argwhere(B != 0):
        parents_i = set(np.flatnonzero(B[:, i]))
        parents_j = set(np.flatnonzero(B[:, j]))
        if (parents_j == parents_i | {i}) != covered:
            continue
        candidate = B.copy()
        candidate[i, j], candidate[j, i] = 0, 1
        if tscore.is_dag(candidate):
            return candidate
    return B.copy()


def cell2_3_score_candidates(inst: tenv.Instance) -> pd.DataFrame:
    """
    Score G*, the empty graph and a few edits of G* against the instance.

    :param inst: sampled instance
    :return: one row per candidate with the main score keys
    """
    B = inst.truth.B
    B_true_reversed = B.T.copy()
    candidates = {
        "G* (the answer)": B,
        "empty graph": np.zeros_like(B),
        "covered edge reversed": _reverse_edge(B, covered=True),
        "uncovered edge reversed": _reverse_edge(B, covered=False),
        "all edges reversed": B_true_reversed,
        "complete order": np.triu(np.ones_like(B), 1),
    }
    rows = []
    for name, B_hat in candidates.items():
        scores = tscore.score_graph(B_hat, inst.truth, legacy=False)
        rows.append(
            {
                "candidate": name,
                "score": round(scores["score"], 3) + 0.0,
                "shd_target": scores["shd_target"],
                "shd_dag": scores["shd_dag"],
                "shd_cpdag": scores["shd_cpdag"],
                "pred_frac": round(scores["pred_frac"], 3),
            }
        )
    return pd.DataFrame(rows)


# #############################################################################
# Cell 3.1: Call each tool once
# #############################################################################


def cell3_1_call_each_tool(
    evidence: ttools.Evidence, i: int, j: int, k: int
) -> pd.DataFrame:
    """
    Call every tool once on variables i, j (and k as conditioning variable).

    :param evidence: evidence of a reset environment, with budget for 5 calls
    :param i: first variable
    :param j: second variable
    :param k: conditioning variable of the CI test
    :return: one row per call with its result
    """
    actions = [
        ttools.ci_test_action(i, j, (k,)),
        ttools.regress_action(j, (i,)),
        ttools.Action(ttools.LINGAM_DIR, (min(i, j), max(i, j))),
        ttools.Action(ttools.NONGAUSS, (i,)),
        ttools.Action(ttools.VARIANCE, (i,)),
    ]
    rows = []
    for action in actions:
        result = evidence.record(action)
        summary = ", ".join(
            f"{key}={value:.3g}" for key, value in result.items()
        )
        rows.append(
            {"tool": action.kind, "args": action.args, "result": summary}
        )
    return pd.DataFrame(rows)


# #############################################################################
# Cell 3.3: Compare baselines on one instance
# #############################################################################


def cell3_3_compare_baselines(inst: tenv.Instance) -> pd.DataFrame:
    """
    Run every baseline on one instance and score it.

    :param inst: sampled instance
    :return: one row per baseline
    """
    rows = []
    # FCI outputs a PAG, which has no combined score; it is the reference for
    # hidden confounders (Milestone 4), not for this instance.
    for name in [name for name in tbasel.BASELINES if name != "fci"]:
        output = tbasel.run_baseline(name, inst.d_train.to_numpy(), seed=0)
        scores = tbasel.score_output(output, inst.truth)
        rows.append(
            {
                "baseline": name,
                "output": output.kind,
                "score": round(scores["score"], 3) + 0.0,
                "shd_target": scores["shd_target"],
                "seconds": round(output.seconds, 3),
            }
        )
    return pd.DataFrame(rows)



# #############################################################################
# Example notebook, Cell 1.2: Gallery of hidden graphs
# #############################################################################


def example_cell1_2_plot_gallery(*, seed: int = 0) -> None:
    """
    Plot one hidden graph per topology, with its identifiable target.

    :param seed: instance seed shared by the panels
    """
    configs = {
        "Erdos-Renyi": tenv.GraphConfig(n_nodes=8, n_edges=10),
        "scale-free": tenv.GraphConfig(
            n_nodes=8, n_edges=10, topology="scale_free"
        ),
        "star": tenv.GraphConfig(n_nodes=5, n_edges=4, topology="star"),
        "layered": tenv.GraphConfig(n_nodes=9, n_edges=10, topology="layered"),
    }
    _, axes = plt.subplots(
        # A lower resolution keeps the example notebook under the 500 KB
        # file limit.
        1,
        len(configs),
        figsize=(3.5 * len(configs), 3.5),
        dpi=52,
    )
    for ax, (name, cfg) in zip(axes, configs.items()):
        inst = tenv.sample_instance(cfg, seed)
        graph = nx.DiGraph()
        graph.add_nodes_from(f"x{k}" for k in range(cfg.n_nodes))
        graph.add_edges_from(
            (f"x{i}", f"x{j}") for i, j in np.argwhere(inst.truth.B != 0)
        )
        title = f"{name}\ntarget: {inst.truth.target_class}"
        hgraphv.plot_causal_dag(graph, title, ax=ax)
    plt.tight_layout()
    plt.show()


# #############################################################################
# Example notebook, Cell 1.3: Sortability per family
# #############################################################################


def example_cell1_3_load_sortability(
    path: str = "results/m1/sortability.json",
) -> pd.DataFrame:
    """
    Load the M1 sortability report as one row per family.

    :param path: committed result file
    :return: edge-definition sortabilities on raw and standardized data, and
        the var-sort-regress minus random-order SHD on standardized data
    """
    with open(path) as file:
        report = json.load(file)
    rows = []
    for family, values in report.items():
        test = values["var_minus_random_shd"]
        rows.append(
            {
                "family": family,
                "var_raw": values["raw_var_edges"]["mean"],
                "var_std": values["std_var_edges"]["mean"],
                "r2": values["raw_r2_edges"]["mean"],
                "rel_oracle": values["raw_rel_oracle_edges"]["mean"],
                "rel_empirical": values["raw_rel_emp_edges"]["mean"],
                "var_sort_minus_random_shd": test["mean"],
                "holm_p": test["holm_p"],
            }
        )
    return pd.DataFrame(rows).round(3)


def example_cell1_3_plot_sortability(table: pd.DataFrame) -> None:
    """
    Plot raw and standardized variance and relatives sortability per family.

    :param table: output of `example_cell1_3_load_sortability()`
    """
    columns = {
        "var_raw": "variance, raw",
        "var_std": "variance, standardized",
        "rel_oracle": "relatives (oracle)",
    }
    long = table.melt(
        id_vars="family", value_vars=list(columns), var_name="score"
    )
    long["score"] = long["score"].map(columns)
    _, ax = plt.subplots(figsize=(11, 3.5), dpi=_DPI)
    sns.barplot(data=long, x="family", y="value", hue="score", ax=ax)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("sortability (edges)")
    ax.set_xlabel("")
    ax.set_title("1 = the score reveals the causal order, 0.5 = random guess")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), title="score")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show()


def _load_json(path: str) -> Dict[str, Any]:
    """
    Read a JSON result file.

    :param path: file path
    :return: its content
    """
    with open(path) as file:
        content = json.load(file)
    return content


# #############################################################################
# Example notebook, Cell 2.1: The M2 baseline table
# #############################################################################


def _corrected_pc_calls(
    path: str, *, corrections: str = "results/post_audit/pc_ci_counts.json"
) -> Dict[str, float]:
    """
    Read the corrected PC test counts of one result file.

    The stored counts include 3 metadata entries of causal-learn's p-value
    cache (`results/post_audit/README.md`).

    :param path: result file as named in the corrections
    :param corrections: post-audit corrections of the PC counts
    :return: corrected count by JSON path (e.g., "/table/D/pc/calls")
    """
    values = _load_json(corrections)["values"]
    corrected = {v["path"]: v["new"] for v in values if v["file"] == path}
    return corrected


def example_cell2_1_load_m2_table(
    setting: str, path: str = "results/m2/table_summary.json"
) -> pd.DataFrame:
    """
    Load one setting of the M2 table, best mean score first.

    PC's `calls` are its distinct CI tests, corrected after the audit.

    :param setting: M2 setting (`A0`, ..., `H`)
    :param path: committed result file
    :return: one row per method
    """
    with open(path) as file:
        table = json.load(file)["table"][setting]
    corrected = _corrected_pc_calls(path)
    rows = []
    for method, summary in table.items():
        low, high = summary["score_ci"]
        calls_path = f"/table/{setting}/{method}/calls"
        rows.append(
            {
                "method": method,
                "score": summary["score"],
                "score_ci": f"[{low:.3f}, {high:.3f}]",
                "shd_target": summary["shd_target"],
                "exact_recovery": summary["exact_recovery"],
                "calls": corrected.get(calls_path, summary["calls"]),
            }
        )
    frame = pd.DataFrame(rows).sort_values("score", ascending=False)
    return frame.round(3).reset_index(drop=True)


def example_cell2_2_plot_m2_heatmap(
    path: str = "results/m2/table_summary.json",
) -> None:
    """
    Plot the mean combined score of every method in every setting.

    :param path: committed result file
    """
    with open(path) as file:
        table = json.load(file)["table"]
    scores = pd.DataFrame(
        {
            setting: {method: s["score"] for method, s in methods.items()}
            for setting, methods in table.items()
        }
    )
    scores = scores.loc[scores.mean(axis=1).sort_values(ascending=False).index]
    _, ax = plt.subplots(figsize=(9, 5), dpi=_DPI)
    sns.heatmap(
        scores, annot=True, fmt=".2f", cmap="viridis", vmin=-1, vmax=0, ax=ax
    )
    ax.set_title(
        "Mean combined score (0 = G*, -1 = empty graph), 200 graphs each"
    )
    ax.set_xlabel("setting")
    plt.tight_layout()
    plt.show()
