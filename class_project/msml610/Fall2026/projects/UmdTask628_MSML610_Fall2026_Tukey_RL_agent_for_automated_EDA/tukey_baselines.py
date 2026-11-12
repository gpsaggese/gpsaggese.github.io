"""
Non-RL causal discovery baselines on the training rows.

- `pc`: PC with Fisher-z tests (causal-learn); emits a PDAG. With the default
  `uc_priority=2` used here, causal-learn 0.1.4.8 orients an edge into a
  collider only if it is not already oriented the other way, so no
  bidirected conflict arises; the bidirected mark is still kept as a distinct
  mark for other settings
- `ges`: GES with the BIC score (causal-learn); emits a CPDAG
- `fci`: FCI with Fisher-z tests (causal-learn); emits a PAG, the reference
  for instances with a hidden common cause
- `direct_lingam`: DirectLiNGAM (lingam); emits a DAG
- `var_sort_regress`, `r2_sort_regress`, `rel_sort_regress`,
  `random_sort_regress`: sort the variables by variance, by R^2 on the others,
  by the estimated number of relatives (Reisach et al. 2026, Algorithm 1), or
  at random, then regress each on its predecessors with an adaptive lasso
  (CausalDisco); emit a DAG
- `empty`: no edge

Every baseline sees only D_train. Outputs are scored with the same marks as the
policy (`score_output()`).

Import as:

import tukey_baselines as tbasel
"""

import dataclasses
import logging
import time
import warnings
from typing import Any, Dict, Optional, Tuple

import causallearn.search.ConstraintBased.FCI as cscofci
import causallearn.search.ConstraintBased.PC as cscopc
import causallearn.search.ScoreBased.GES as cscoges
import CausalDisco.analytics as cdanaly
import CausalDisco.baselines as cdbasel
import lingam
import numpy as np

import helpers.hdbg as hdbg
import tukey_scoring as tscore

_LOG = logging.getLogger(__name__)

BASELINES = (
    "pc",
    "ges",
    "fci",
    "direct_lingam",
    "var_sort_regress",
    "r2_sort_regress",
    "rel_sort_regress",
    "random_sort_regress",
    "empty",
)


@dataclasses.dataclass(frozen=True)
class BaselineOutput:
    """
    A baseline's graph and cost.
    """

    name: str
    # "dag" (B[i, j] = 1 for i -> j), "pdag" (symmetric entries undirected) or
    # "pag" (endpoint marks, see `tukey_scoring.PAG_TAIL`).
    kind: str
    adjacency: np.ndarray
    # PDAG only: pairs PC marked as bidirected conflicts.
    bidirected: Optional[np.ndarray]
    seconds: float
    # PC only: number of distinct CI tests run.
    n_ci_tests: int = 0
    # PC only: the count stored by every run made with code up to `10f4d19`
    # (the audited commit), i.e., every entry of causal-learn's p-value cache,
    # including its 3 metadata entries; kept to reproduce those values.
    n_ci_tests_legacy: int = 0


# Entries of causal-learn's p-value cache that are not CI tests: 0.1.4.8's
# `CIT_Base.__init__` and `check_cache_method_consistent` (`utils/cit.py`)
# store them next to one entry per distinct test (e.g., "0;1|2").
_CACHE_METADATA_KEYS = ("data_hash", "method_name", "parameters_hash")


def count_ci_tests(pvalue_cache: Dict[str, Any]) -> int:
    """
    Count the distinct CI tests in a causal-learn p-value cache.

    :param pvalue_cache: `cg.test.pvalue_cache` after a PC run
    :return: number of test entries (metadata entries excluded)
    """
    n_tests = sum(key not in _CACHE_METADATA_KEYS for key in pvalue_cache)
    return n_tests


def causallearn_to_pdag(graph: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a causal-learn graph matrix into a PDAG and its bidirected pairs.

    causal-learn encodes i -> j as `graph[i, j] = -1, graph[j, i] = 1`, an
    undirected edge as both -1, and a bidirected edge as both 1.

    :param graph: causal-learn `GeneralGraph.graph`
    :return: PDAG adjacency (bidirected pairs entered as undirected) and a
        boolean matrix of the bidirected pairs
    """
    n_nodes = graph.shape[0]
    A = np.zeros((n_nodes, n_nodes), dtype=int)
    bidirected = np.zeros((n_nodes, n_nodes), dtype=bool)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            mark_ij, mark_ji = int(graph[i, j]), int(graph[j, i])
            if (mark_ij, mark_ji) == (0, 0):
                continue
            if (mark_ij, mark_ji) == (-1, 1):
                A[i, j] = 1
            elif (mark_ij, mark_ji) == (1, -1):
                A[j, i] = 1
            elif (mark_ij, mark_ji) in ((-1, -1), (1, 1)):
                A[i, j] = A[j, i] = 1
                bidirected[i, j] = bidirected[j, i] = mark_ij == 1
            else:
                raise ValueError(f"Unexpected marks {mark_ij, mark_ji}")
    return A, bidirected


def run_baseline(
    name: str,
    x_train: np.ndarray,
    *,
    seed: int = 0,
    params: Optional[Dict[str, Any]] = None,
) -> BaselineOutput:
    """
    Run one baseline on the training rows.

    :param name: one of `BASELINES`
    :param x_train: training rows, shape (n_train, n_nodes)
    :param seed: random seed (`random_sort_regress` only)
    :param params: `alpha` for `pc`, `fci` and `rel_sort_regress` (default
        0.05)
    :return: the baseline's graph and cost
    """
    hdbg.dassert_in(name, BASELINES)
    params = params or {}
    n_nodes = x_train.shape[1]
    bidirected = None
    n_ci_tests = n_ci_tests_legacy = 0
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name == "pc":
            cg = cscopc.pc(
                x_train,
                alpha=params.get("alpha", 0.05),
                indep_test="fisherz",
                show_progress=False,
            )
            adjacency, bidirected = causallearn_to_pdag(cg.G.graph)
            n_ci_tests = count_ci_tests(cg.test.pvalue_cache)
            n_ci_tests_legacy = len(cg.test.pvalue_cache)
            kind = "pdag"
        elif name == "fci":
            graph, _ = cscofci.fci(
                x_train,
                independence_test_method="fisherz",
                alpha=params.get("alpha", 0.05),
                verbose=False,
                show_progress=False,
            )
            # Nodes come in column order; the matrix is already in the PAG
            # endpoint-mark encoding.
            adjacency = graph.graph.astype(int)
            kind = "pag"
        elif name == "ges":
            record = cscoges.ges(x_train, score_func="local_score_BIC")
            adjacency, bidirected = causallearn_to_pdag(record["G"].graph)
            kind = "pdag"
        elif name == "direct_lingam":
            model = lingam.DirectLiNGAM()
            model.fit(x_train)
            # lingam's adjacency_matrix_[j, i] is the effect of i on j.
            adjacency = (model.adjacency_matrix_.T != 0).astype(int)
            kind = "dag"
        elif name == "empty":
            adjacency = np.zeros((n_nodes, n_nodes), dtype=int)
            kind = "dag"
        else:
            if name == "var_sort_regress":
                W = cdbasel.var_sort_regress(x_train)
            elif name == "r2_sort_regress":
                W = cdbasel.r2_sort_regress(x_train)
            elif name == "rel_sort_regress":
                scores = cdanaly.rel_count_emp_hard(
                    x_train, alpha=params.get("alpha", 0.05)
                )
                W = cdbasel.sort_regress(x_train, scores.astype(float))
            else:
                W = cdbasel.random_sort_regress(x_train, seed=seed)
            # CausalDisco's W[i, j] is the coefficient of i in j's regression.
            adjacency = (W != 0).astype(int)
            kind = "dag"
    output = BaselineOutput(
        name=name,
        kind=kind,
        adjacency=adjacency,
        bidirected=bidirected,
        seconds=time.perf_counter() - start,
        n_ci_tests=n_ci_tests,
        n_ci_tests_legacy=n_ci_tests_legacy,
    )
    return output


def fallback_score(
    shd_target: float, n_true_edges: int, *, informative: bool, lam: float
) -> float:
    """
    Score a graph that predicts nothing (`pred_frac = 0`).

    Uses the same branch as `tukey_scoring.score_graph()`: when the true
    graph's held-out gain is below `PRED_GAIN_MIN` the prediction term is
    uninformative and the score is structural only.

    :param shd_target: structural distance to the target
    :param n_true_edges: number of true edges
    :param informative: whether the instance's prediction term is informative
    :param lam: weight of the structural term
    :return: combined score
    """
    score = tscore.combined_score(
        shd_target,
        n_true_edges,
        pred_frac=0.0,
        informative=informative,
        lam=lam,
    )
    return score


def fallback_score_legacy(
    shd_target: float, n_true_edges: int, *, lam: float
) -> float:
    """
    Score a graph that predicts nothing as code up to `10f4d19` did.

    Kept only to reproduce stored values: it always applied the prediction
    penalty, also on instances whose prediction term is uninformative.

    :param shd_target: structural distance to the target
    :param n_true_edges: number of true edges
    :param lam: weight of the structural term
    :return: combined score as stored by those runs
    """
    shd_norm = shd_target / max(1, n_true_edges)
    score = -lam * shd_norm - (1 - lam)
    return float(score)


def score_output(
    output: BaselineOutput, truth: tscore.Truth, *, lam: float = 0.5
) -> Dict[str, Any]:
    """
    Score a baseline's graph with the scorer used for every agent.

    :param output: baseline output
    :param truth: instance truth
    :param lam: weight of the structural term
    :return: score dictionary plus `seconds`, `n_ci_tests` and
        `score_fallback` (True when a PDAG had no DAG extension and was scored
        as predicting nothing, by `fallback_score()`)
    """
    if output.kind == "dag":
        scores = tscore.score_graph(
            output.adjacency, truth, lam=lam, legacy=False
        )
    elif output.kind == "pdag":
        scores = tscore.score_pdag(
            output.adjacency, truth, lam=lam, bidirected=output.bidirected
        )
    else:
        hdbg.dassert_eq(output.kind, "pag")
        # Without a hidden cause the true PAG is the PAG of the DAG itself.
        pag = truth.pag
        if pag is None:
            pag = tscore.pag_from_dag(truth.B, [])
        scores = {"target_class": truth.target_class}
        scores.update(tscore.score_pag(output.adjacency, pag, truth.confounded))
        scores["shd_target"] = float("nan")
        scores["score"] = float("nan")
    scores["score_fallback"] = False
    no_score = truth.target_class == tscore.TARGET_PAG or output.kind == "pag"
    if not no_score and not np.isfinite(scores["score"]):
        # A PDAG with no consistent DAG extension has no prediction term; it is
        # scored as predicting nothing (pred_frac = 0) rather than dropped.
        informative = truth.ll_true - truth.ll_empty >= tscore.PRED_GAIN_MIN
        scores["prediction_informative"] = bool(informative)
        scores["score"] = fallback_score(
            scores["shd_target"],
            scores["n_true_edges"],
            informative=informative,
            lam=lam,
        )
        scores["score_fallback"] = True
    scores["seconds"] = output.seconds
    scores["n_ci_tests"] = output.n_ci_tests
    return scores
