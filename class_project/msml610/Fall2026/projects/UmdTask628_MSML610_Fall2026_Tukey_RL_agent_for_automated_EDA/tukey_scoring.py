"""
Score a discovered graph against the hidden graph: structure and prediction.

Conventions:
- A DAG is a binary adjacency `B` with `B[i, j] = 1` meaning i -> j (the
  prototype's convention)
- A PDAG is a binary adjacency `A` where `A[i, j] = A[j, i] = 1` means the
  undirected edge i - j
- Marks summarize each unordered pair i < j as one of: none, i -> j, j -> i,
  undirected, bidirected; SHD counts the pairs whose marks differ, so a
  reversed edge costs 1 (as in the prototype and causal-learn)

The combined score follows the spec and GP's draft paper:
`score = -lam * SHD_target / max(1, |E*|) - (1 - lam) * (1 - pred_frac)`,
where `SHD_target` compares against what the data can identify (the DAG, its
equivalence class, or the non-Gaussian pattern) and `pred_frac` is the held-out
log-likelihood gain of the discovered graph over the empty graph, relative to
the true graph's gain. The true graph scores 0 and the empty graph -1.

Instances with a hidden common cause have a PAG target (Zhang 2008), which a
DAG cannot express: they get PAG-compatible metrics (adjacencies, orientation
contradictions, the confounded pairs) and no combined score.

Import as:

import tukey_scoring as tscore
"""

import dataclasses
import functools
import itertools
import logging
import types
from typing import Any, Dict, List, Optional, Tuple

import causallearn.graph.Dag as cldag
import causallearn.graph.GraphNode as clgnode
import causallearn.utils.DAG2PAG as cldagpag
import numpy as np

import helpers.hdbg as hdbg
import research.agentic_data_science.RL_for_auto_EDA_prototype as rlproto

_LOG = logging.getLogger(__name__)

# Mark codes for an unordered pair i < j.
MARK_NONE = 0
MARK_FORWARD = 1
MARK_BACKWARD = 2
MARK_UNDIRECTED = 3
MARK_BIDIRECTED = 4

# Target classes: what the data can identify for an instance.
TARGET_DAG = "dag"
TARGET_CPDAG = "cpdag"
TARGET_NGDAG = "ngdag_pattern"
# Hidden common causes: the observed data identify a PAG.
TARGET_PAG = "pag"

# PAG endpoint marks in causal-learn's encoding: `P[i, j]` is the mark at node
# i on the edge between i and j; 0 means no edge.
PAG_TAIL = -1
PAG_ARROW = 1
PAG_CIRCLE = 2

# Minimum held-out log-likelihood gain of the true graph (nats per sample) for
# the prediction term to be informative.
PRED_GAIN_MIN = 0.02


# #############################################################################
# Graph conversions
# #############################################################################


def edges_to_adjacency(
    edges: List[Tuple[int, int]], n_nodes: int
) -> Tuple[np.ndarray, int]:
    """
    Convert an edge list into a binary adjacency, as the prototype reads it.

    Self-loops, out-of-range indices and duplicates are dropped, as in the
    prototype's `read_graph()`; so are indices that are not integers (e.g.,
    `1.0` from a JSON reply), on which the prototype crashes.

    :param edges: `[(parent, child), ...]`
    :param n_nodes: number of nodes
    :return: adjacency `B` and the number of dropped entries
    """
    B = np.zeros((n_nodes, n_nodes), dtype=int)
    n_dropped = 0
    for parent, child in edges:
        valid = all(
            isinstance(k, (int, np.integer))
            and not isinstance(k, bool)
            and 0 <= k < n_nodes
            for k in (parent, child)
        )
        if not valid or parent == child or B[parent, child]:
            n_dropped += 1
            continue
        B[parent, child] = 1
    return B, n_dropped


def is_dag(B: np.ndarray) -> bool:
    """
    Check whether a directed adjacency has no directed cycle.

    :param B: directed adjacency (`B[i, j] = 1` means i -> j)
    :return: True if acyclic
    """
    # Kahn's algorithm: repeatedly remove nodes with no incoming edge.
    in_degree = B.sum(axis=0).astype(int)
    stack = list(np.flatnonzero(in_degree == 0))
    n_removed = 0
    remaining = B.astype(int).copy()
    while stack:
        node = stack.pop()
        n_removed += 1
        for child in np.flatnonzero(remaining[node]):
            remaining[node, child] = 0
            in_degree[child] -= 1
            if in_degree[child] == 0:
                stack.append(child)
    acyclic = n_removed == B.shape[0]
    return acyclic


def marks_from_pdag(A: np.ndarray) -> np.ndarray:
    """
    Summarize a (P)DAG as one mark per unordered pair.

    :param A: adjacency; `A[i, j] = A[j, i] = 1` is read as undirected
    :return: integer matrix whose upper triangle holds the mark of each pair
        i < j; the rest is 0
    """
    forward = np.triu(A, 1)
    backward = np.triu(A.T, 1)
    marks = np.zeros(A.shape, dtype=int)
    marks[(forward == 1) & (backward == 0)] = MARK_FORWARD
    marks[(forward == 0) & (backward == 1)] = MARK_BACKWARD
    marks[(forward == 1) & (backward == 1)] = MARK_UNDIRECTED
    return marks


# Labels of an emitted edge in `edge_marks()`.
EDGE_DIRECTED = "->"
EDGE_UNDIRECTED = "--"
EDGE_BIDIRECTED = "<->"


def edge_marks(
    A: np.ndarray, bidirected: Optional[np.ndarray] = None
) -> List[List[Any]]:
    """
    List an output's edges with their marks, so that records keep them.

    An adjacency alone cannot tell an undirected edge from a bidirected
    conflict (both set both entries).

    :param A: (P)DAG adjacency
    :param bidirected: boolean matrix of the pairs marked bidirected
        - Default: None means no bidirected pair
    :return: `[i, j, "->"]` for i -> j, and `[i, j, "--"]` or `[i, j, "<->"]`
        with i < j for an undirected edge or a bidirected conflict
    """
    edges: List[List[Any]] = []
    for i, j in np.argwhere(A == 1):
        i, j = int(i), int(j)
        if A[j, i] != 1:
            edges.append([i, j, EDGE_DIRECTED])
        elif i < j:
            two_way = bidirected is not None and bool(bidirected[i, j])
            edges.append([i, j, EDGE_BIDIRECTED if two_way else EDGE_UNDIRECTED])
    return edges


def marks_from_edge_marks(edges: List[List[Any]], n_nodes: int) -> np.ndarray:
    """
    Rebuild the pair marks of `marks_from_pdag()` from `edge_marks()`.

    :param edges: output of `edge_marks()`
    :param n_nodes: number of variables
    :return: marks (bidirected pairs as `MARK_BIDIRECTED`)
    """
    marks = np.zeros((n_nodes, n_nodes), dtype=int)
    for i, j, mark in edges:
        lo, hi = min(i, j), max(i, j)
        if mark == EDGE_DIRECTED:
            marks[lo, hi] = MARK_FORWARD if i < j else MARK_BACKWARD
        elif mark == EDGE_UNDIRECTED:
            marks[lo, hi] = MARK_UNDIRECTED
        else:
            hdbg.dassert_eq(mark, EDGE_BIDIRECTED)
            marks[lo, hi] = MARK_BIDIRECTED
    return marks


def shd_marks(marks1: np.ndarray, marks2: np.ndarray) -> int:
    """
    Count the unordered pairs whose marks differ.

    :param marks1: marks of the first graph
    :param marks2: marks of the second graph
    :return: structural Hamming distance
    """
    shd = int(np.triu(marks1 != marks2, 1).sum())
    return shd


def _orient(A: np.ndarray, src: int, dst: int) -> None:
    """
    Turn the undirected edge src - dst into src -> dst, in place.

    :param A: PDAG adjacency
    :param src: tail of the oriented edge
    :param dst: head of the oriented edge
    """
    A[dst, src] = 0


def _apply_meek_rules(A: np.ndarray) -> np.ndarray:
    """
    Orient undirected edges with Meek's rules R1-R3 until nothing changes.

    R1: a -> u - v, a and v non-adjacent => u -> v.
    R2: u -> w -> v and u - v => u -> v.
    R3: u - c, u - d, c -> v, d -> v, c and d non-adjacent, u - v => u -> v.
    For a PDAG obtained from a DAG's skeleton and v-structures, R1-R3 give its
    equivalence class (Meek 1995); R4 is needed only with background knowledge.

    :param A: PDAG adjacency, modified in place
    :return: the same array
    """
    adjacent = (A + A.T) > 0
    changed = True
    while changed:
        changed = False
        directed = (A == 1) & (A.T == 0)
        undirected = (A == 1) & (A.T == 1)
        for u, v in zip(*np.nonzero(np.triu(undirected, 1))):
            for src, dst in ((u, v), (v, u)):
                # R1: some a -> src with a not adjacent to dst.
                parents = np.flatnonzero(directed[:, src])
                rule1 = any(not adjacent[a, dst] for a in parents if a != dst)
                # R2: src -> w -> dst for some w.
                rule2 = bool(np.any(directed[src] & directed[:, dst]))
                # R3: two non-adjacent c, d with src - c, src - d, c -> dst,
                # d -> dst.
                cands = np.flatnonzero(undirected[src] & directed[:, dst])
                rule3 = any(
                    not adjacent[c, d]
                    for c, d in itertools.combinations(cands, 2)
                )
                if rule1 or rule2 or rule3:
                    _orient(A, src, dst)
                    changed = True
                    break
            if changed:
                break
    return A


def cpdag_from_dag(B: np.ndarray) -> np.ndarray:
    """
    Compute the equivalence class (CPDAG) of a DAG.

    Keeps the skeleton, orients the v-structures, then applies Meek's rules.
    Cross-checked against causal-learn's `dag2cpdag()` and brute-force class
    enumeration in the unit tests.

    :param B: DAG adjacency
    :return: PDAG adjacency of the CPDAG
    """
    hdbg.dassert(is_dag(B), "cpdag_from_dag needs a DAG")
    skeleton = ((B + B.T) > 0).astype(int)
    A = skeleton.copy()
    # Orient every v-structure i -> k <- j with i, j non-adjacent.
    for k in range(B.shape[0]):
        parents = np.flatnonzero(B[:, k])
        for i, j in itertools.combinations(parents, 2):
            if not skeleton[i, j]:
                _orient(A, i, k)
                _orient(A, j, k)
    A = _apply_meek_rules(A)
    return A


def ngdag_pattern_from_dag(B: np.ndarray, nongauss: np.ndarray) -> np.ndarray:
    """
    Compute what is identifiable when some noise terms are non-Gaussian.

    Hoyer et al. 2008: start from the equivalence class and orient every edge
    touching a non-Gaussian node, then propagate. Implemented with their trick:
    add two parentless parents to each non-Gaussian node, take the CPDAG, drop
    the added nodes.

    :param B: DAG adjacency
    :param nongauss: boolean per node, True if its noise is non-Gaussian
    :return: PDAG adjacency of the pattern
    """
    d = B.shape[0]
    ng_nodes = np.flatnonzero(nongauss)
    n_total = d + 2 * len(ng_nodes)
    B_aug = np.zeros((n_total, n_total), dtype=int)
    B_aug[:d, :d] = B
    for idx, node in enumerate(ng_nodes):
        B_aug[d + 2 * idx, node] = 1
        B_aug[d + 2 * idx + 1, node] = 1
    pattern = cpdag_from_dag(B_aug)[:d, :d]
    return pattern


def dag_extension(A: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Find a DAG consistent with a PDAG (Dor and Tarsi 1992).

    Repeatedly picks a node with no outgoing directed edge whose undirected
    neighbours are adjacent to all its other neighbours, orients its undirected
    edges into it, and removes it.

    :param A: PDAG adjacency
    :return: a consistent DAG, and False if the PDAG admits no extension (the
        returned matrix is then the PDAG's directed part)
    """
    d = A.shape[0]
    work = A.copy()
    B = ((A == 1) & (A.T == 0)).astype(int)
    alive = np.ones(d, dtype=bool)
    for _ in range(d):
        found = False
        for x in np.flatnonzero(alive):
            directed_out = (work[x] == 1) & (work[:, x] == 0) & alive
            if directed_out.any():
                continue
            undirected = np.flatnonzero(
                (work[x] == 1) & (work[:, x] == 1) & alive
            )
            neighbours = np.flatnonzero(((work[x] + work[:, x]) > 0) & alive)
            ok = all(
                ((work[y, z] + work[z, y]) > 0) or y == z
                for y in undirected
                for z in neighbours
            )
            if not ok:
                continue
            # Orient every undirected edge y - x as y -> x, then remove x.
            for y in undirected:
                B[y, x] = 1
            alive[x] = False
            found = True
            break
        if not found:
            return B, False
    return B, True


def target_marks(
    B: np.ndarray, target_class: str, nongauss: np.ndarray
) -> np.ndarray:
    """
    Compute the marks of a DAG under an identifiability class.

    :param B: DAG adjacency
    :param target_class: `TARGET_DAG`, `TARGET_CPDAG` or `TARGET_NGDAG`
    :param nongauss: boolean per node (used by `TARGET_NGDAG`)
    :return: marks matrix
    """
    if target_class == TARGET_DAG:
        A = B
    elif target_class == TARGET_CPDAG:
        A = cpdag_from_dag(B)
    elif target_class == TARGET_NGDAG:
        A = ngdag_pattern_from_dag(B, nongauss)
    else:
        raise ValueError(f"Unknown target_class='{target_class}'")
    marks = marks_from_pdag(A)
    return marks


# #############################################################################
# PAGs (hidden common causes)
# #############################################################################


def pag_from_dag(B: np.ndarray, latent: List[int]) -> np.ndarray:
    """
    Compute the PAG of a DAG over its observed variables.

    causal-learn's `dag2pag` may reorder the nodes; the result is mapped back
    to the input order of the observed variables.

    :param B: DAG adjacency over observed and latent variables
    :param latent: indices of the latent variables
    :return: endpoint marks over the observed variables, in input order
    """
    names = [f"X{k}" for k in range(B.shape[0])]
    nodes = [clgnode.GraphNode(name) for name in names]
    dag = cldag.Dag(nodes)
    for i, j in np.argwhere(B != 0):
        dag.add_directed_edge(nodes[i], nodes[j])
    pag = cldagpag.dag2pag(dag, [nodes[k] for k in latent])
    observed = [k for k in range(B.shape[0]) if k not in latent]
    position = {f"X{k}": p for p, k in enumerate(observed)}
    order = [position[node.get_name()] for node in pag.get_nodes()]
    P = np.zeros((len(observed), len(observed)), dtype=int)
    P[np.ix_(order, order)] = pag.graph
    return P


def pag_marks_from_pdag(
    A: np.ndarray, *, bidirected: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Express a DAG or PDAG output as PAG endpoint marks.

    A directed edge i -> j claims a tail at i and an arrowhead at j; an
    undirected edge makes no orientation claim (circles); a bidirected
    conflict (PC) claims arrowheads at both ends.

    :param A: (P)DAG adjacency
    :param bidirected: boolean matrix of the pairs marked bidirected
        - Default: None means no bidirected pair
    :return: endpoint marks
    """
    P = np.zeros(A.shape, dtype=int)
    forward = (A == 1) & (A.T == 0)
    undirected = (A == 1) & (A.T == 1)
    P[forward] = PAG_TAIL
    P[forward.T] = PAG_ARROW
    P[undirected] = PAG_CIRCLE
    if bidirected is not None:
        mask = np.asarray(bidirected, dtype=bool)
        P[mask | mask.T] = PAG_ARROW
    return P


def _is_causal_claim(P: np.ndarray, i: int, j: int) -> bool:
    """
    Check whether endpoint marks orient a pair as a causal edge.

    :param P: endpoint marks
    :param i: first node
    :param j: second node
    :return: True for a tail at one end and an arrowhead at the other
    """
    is_claim = {int(P[i, j]), int(P[j, i])} == {PAG_TAIL, PAG_ARROW}
    return is_claim


def score_pag(
    P_hat: np.ndarray,
    P_true: np.ndarray,
    confounded: Tuple[Tuple[int, int], ...],
) -> Dict[str, Any]:
    """
    Compare endpoint marks with a true PAG.

    - Adjacency precision, recall and F1 against the PAG's skeleton
    - `pag_contradictions`: pairs adjacent in both whose output puts a tail
      where the PAG has an arrowhead, or the reverse (a circle is compatible
      with either mark)
    - `shd_pag`: pairs whose endpoint marks differ (for PAG outputs such as
      FCI's; a DAG cannot express a circle or `<->`)
    - Confounded pairs (hidden common cause): how many the output keeps
      adjacent (correct: no observed set separates them) and how many it
      orients as a causal edge (an overclaim)

    :param P_hat: endpoint marks of the output
    :param P_true: endpoint marks of the true PAG
    :param confounded: pairs with a hidden common cause
    :return: metrics
    """
    adj_hat = np.triu(P_hat != 0, 1)
    adj_true = np.triu(P_true != 0, 1)
    tp = int((adj_hat & adj_true).sum())
    n_hat = int(adj_hat.sum())
    n_true = int(adj_true.sum())
    clash = ((P_hat == PAG_TAIL) & (P_true == PAG_ARROW)) | (
        (P_hat == PAG_ARROW) & (P_true == PAG_TAIL)
    )
    clash_pairs = np.triu(clash | clash.T, 1) & adj_hat & adj_true
    differ = np.triu((P_hat != P_true) | (P_hat != P_true).T, 1)
    metrics = {
        "pag_adj_precision": tp / n_hat if n_hat else 1.0,
        "pag_adj_recall": tp / n_true if n_true else 1.0,
        "pag_adj_f1": 2 * tp / (n_hat + n_true) if n_hat + n_true else 1.0,
        "pag_contradictions": int(clash_pairs.sum()),
        "shd_pag": int(differ.sum()),
        "n_confounded": len(confounded),
        "confounded_adjacent": sum(int(P_hat[i, j] != 0) for i, j in confounded),
        "confounded_directed": sum(
            int(_is_causal_claim(P_hat, i, j)) for i, j in confounded
        ),
    }
    return metrics


# #############################################################################
# Held-out prediction
# #############################################################################


def get_moments(x: np.ndarray) -> np.ndarray:
    """
    Compute raw second moments of the data with an intercept column.

    :param x: data, shape (n_samples, n_nodes)
    :return: `Z.T @ Z / n` with `Z = [1, x]`, shape (n_nodes + 1, n_nodes + 1)
    """
    n = x.shape[0]
    z = np.hstack([np.ones((n, 1)), x])
    moments = z.T @ z / n
    return moments


def pred_ll(
    B: np.ndarray, moments_train: np.ndarray, moments_test: np.ndarray
) -> Tuple[float, bool]:
    """
    Compute the held-out Gaussian log-likelihood of a DAG, per sample.

    Each node is regressed on its parents (OLS with intercept) on the training
    rows; the residual variance is the MLE `RSS / n`. The mean log-density of
    the test rows is summed over nodes. Markov-equivalent DAGs fit the same
    Gaussian model, so they get the same value.

    :param B: DAG adjacency
    :param moments_train: `get_moments()` of the training rows
    :param moments_test: `get_moments()` of the test rows
    :return: log-likelihood in nats per sample, and whether a residual
        variance had to be floored (near-deterministic node)
    """
    total = 0.0
    degenerate = False
    for j in range(B.shape[0]):
        # Index 0 is the intercept; node k is at index k + 1.
        idx = [0] + [p + 1 for p in np.flatnonzero(B[:, j])]
        target = j + 1
        coef = np.linalg.solve(
            moments_train[np.ix_(idx, idx)], moments_train[idx, target]
        )
        sigma2 = (
            moments_train[target, target] - coef @ moments_train[idx, target]
        )
        floor = 1e-10 * max(1.0, moments_train[target, target])
        if sigma2 < floor:
            sigma2 = floor
            degenerate = True
        # Mean squared residual on the test rows, from test moments.
        test_mse = (
            moments_test[target, target]
            - 2 * coef @ moments_test[idx, target]
            + coef @ moments_test[np.ix_(idx, idx)] @ coef
        )
        total += -0.5 * np.log(2 * np.pi * sigma2) - test_mse / (2 * sigma2)
    return float(total), degenerate


# #############################################################################
# Score
# #############################################################################


@dataclasses.dataclass(frozen=True)
class Truth:
    """
    Everything the scorer needs about one instance; never shown to an agent.
    """

    # True DAG adjacency.
    B: np.ndarray
    # Identifiability class of the instance.
    target_class: str
    # Boolean per node: True if its noise is non-Gaussian.
    nongauss: np.ndarray
    # Raw moments of the training and test rows (see `get_moments()`).
    moments_train: np.ndarray
    moments_test: np.ndarray
    # Data, used only for the prototype's legacy `oos_r2_mean` key.
    x_train: np.ndarray
    x_test: np.ndarray
    # Held-out log-likelihoods of the empty and the true graph (cached).
    ll_empty: float
    ll_true: float
    # PAG targets only: endpoint marks of the true PAG over the observed
    # variables, and the pairs that share a hidden common cause.
    pag: Optional[np.ndarray] = None
    confounded: Tuple[Tuple[int, int], ...] = ()

    # The truth's endpoint marks, computed on first use and kept: training
    # scores many graphs against one truth. They are read-only, and not
    # fields, so equality, hashing and the manifests ignore them.

    @functools.cached_property
    def marks_dag(self) -> np.ndarray:
        """
        Endpoint marks of the true DAG.
        """
        return _read_only(marks_from_pdag(self.B))

    @functools.cached_property
    def marks_cpdag(self) -> np.ndarray:
        """
        Endpoint marks of the true graph's CPDAG.
        """
        return _read_only(marks_from_pdag(cpdag_from_dag(self.B)))

    @functools.cached_property
    def marks_target(self) -> np.ndarray:
        """
        Endpoint marks of the identifiable target (`target_marks()`).
        """
        return _read_only(target_marks(self.B, self.target_class, self.nongauss))


def _read_only(array: np.ndarray) -> np.ndarray:
    """
    Make an array read-only, so a cached value cannot be changed by a caller.

    :param array: array to protect
    :return: the same array
    """
    array.setflags(write=False)
    return array


def make_truth(
    B: np.ndarray,
    target_class: str,
    nongauss: np.ndarray,
    x_train: np.ndarray,
    x_test: np.ndarray,
    *,
    pag: Optional[np.ndarray] = None,
    confounded: Tuple[Tuple[int, int], ...] = (),
) -> Truth:
    """
    Build the scorer's view of an instance, caching the reference likelihoods.

    :param B: true DAG adjacency (with a hidden common cause: the edges among
        the observed variables)
    :param target_class: identifiability class of the instance
    :param nongauss: boolean per node, True if its noise is non-Gaussian
    :param x_train: training rows, shape (n_train, n_nodes)
    :param x_test: test rows, shape (n_test, n_nodes)
    :param pag: true PAG's endpoint marks (`TARGET_PAG` only)
    :param confounded: pairs with a hidden common cause (`TARGET_PAG` only)
    :return: `Truth` with moments and the empty/true log-likelihoods
    """
    hdbg.dassert_eq(target_class == TARGET_PAG, pag is not None)
    moments_train = get_moments(x_train)
    moments_test = get_moments(x_test)
    ll_empty, _ = pred_ll(np.zeros_like(B), moments_train, moments_test)
    ll_true, _ = pred_ll(B, moments_train, moments_test)
    truth = Truth(
        B=B,
        target_class=target_class,
        nongauss=nongauss,
        moments_train=moments_train,
        moments_test=moments_test,
        x_train=x_train,
        x_test=x_test,
        ll_empty=ll_empty,
        ll_true=ll_true,
        pag=pag,
        confounded=confounded,
    )
    return truth


def _get_prediction_terms(B_hat: np.ndarray, truth: Truth) -> Dict[str, Any]:
    """
    Compute the held-out log-likelihood terms of a DAG.

    :param B_hat: discovered DAG
    :param truth: instance truth
    :return: `pred_ll`, `pred_gain`, `pred_frac`, `prediction_informative`,
        `degenerate`
    """
    ll_hat, degenerate = pred_ll(B_hat, truth.moments_train, truth.moments_test)
    gain_hat = ll_hat - truth.ll_empty
    gain_true = truth.ll_true - truth.ll_empty
    informative = gain_true >= PRED_GAIN_MIN
    pred_frac = (
        float(np.clip(gain_hat / gain_true, -1.0, 1.5)) if informative else 0.0
    )
    terms = {
        "pred_ll": ll_hat,
        "pred_gain": gain_hat,
        "pred_frac": pred_frac,
        "prediction_informative": bool(informative),
        "degenerate": degenerate,
    }
    return terms


def pdag_agreement(marks_hat: np.ndarray, B: np.ndarray) -> Dict[str, Any]:
    """
    Compare the marks a method emitted with the true DAG.

    The skeleton counts every emitted adjacency, whatever its mark. Only
    directed marks are orientations: an undirected or bidirected mark leaves
    the direction unresolved (Hoyer et al. 2008, Section 3), so it is never
    scored as right or wrong; it lowers the coverage instead.

    :param marks_hat: emitted marks (`marks_from_pdag()`, bidirected pairs as
        `MARK_BIDIRECTED`)
    :param B: true DAG
    :return: `skeleton_precision`, `skeleton_recall`, `skeleton_f1_emitted`
        (1.0 when both graphs are empty), the shared-adjacency counts
        `n_shared`, `n_directed_shared`, `n_undirected_shared`,
        `n_bidirected_shared`, `orientation_accuracy_definite` (correct /
        `n_directed_shared`) and `orientation_coverage` (`n_directed_shared` /
        `n_shared`); a ratio with a zero denominator is NaN
    """
    hdbg.dassert_eq(marks_hat.shape, B.shape)
    upper = np.triu(np.ones(B.shape, dtype=bool), 1)
    marks_true = marks_from_pdag(B)
    adj_hat = upper & (marks_hat != MARK_NONE)
    adj_true = upper & (marks_true != MARK_NONE)
    shared = adj_hat & adj_true
    directed = shared & np.isin(marks_hat, (MARK_FORWARD, MARK_BACKWARD))
    n_hat, n_true = int(adj_hat.sum()), int(adj_true.sum())
    n_shared, n_directed = int(shared.sum()), int(directed.sum())
    n_correct = int((directed & (marks_hat == marks_true)).sum())
    n_undirected = int((shared & (marks_hat == MARK_UNDIRECTED)).sum())
    n_bidirected = int((shared & (marks_hat == MARK_BIDIRECTED)).sum())

    def _ratio(num: int, den: int) -> float:
        return num / den if den else float("nan")

    agreement = {
        "skeleton_precision": _ratio(n_shared, n_hat),
        "skeleton_recall": _ratio(n_shared, n_true),
        "skeleton_f1_emitted": (
            2 * n_shared / (n_hat + n_true) if n_hat + n_true else 1.0
        ),
        "n_shared": n_shared,
        "n_directed_shared": n_directed,
        "n_undirected_shared": n_undirected,
        "n_bidirected_shared": n_bidirected,
        "orientation_accuracy_definite": _ratio(n_correct, n_directed),
        "orientation_coverage": _ratio(n_directed, n_shared),
    }
    return agreement


def _get_skeleton_scores(B_hat: np.ndarray, B: np.ndarray) -> Dict[str, float]:
    """
    Compute skeleton F1 and orientation accuracy on correctly found edges.

    For a DAG these equal `pdag_agreement()`'s `skeleton_f1_emitted` and
    `orientation_accuracy_definite`.

    :param B_hat: discovered graph (directed adjacency)
    :param B: true DAG
    :return: `skeleton_f1`, `orientation_accuracy` (NaN if no edge is shared)
    """
    skel_hat = np.triu((B_hat + B_hat.T) > 0, 1)
    skel = np.triu((B + B.T) > 0, 1)
    tp = int((skel_hat & skel).sum())
    denom = int(skel_hat.sum() + skel.sum())
    skeleton_f1 = 2 * tp / denom if denom else 1.0
    shared = np.argwhere(skel_hat & skel)
    n_correct = sum(
        int(B_hat[i, j] == B[i, j] and B_hat[j, i] == B[j, i]) for i, j in shared
    )
    orientation = n_correct / len(shared) if len(shared) else float("nan")
    scores = {"skeleton_f1": skeleton_f1, "orientation_accuracy": orientation}
    return scores


def skeleton_scores_extension_legacy(
    A_hat: np.ndarray, B: np.ndarray
) -> Dict[str, float]:
    """
    Compute the skeleton and orientation keys as code before the audit did.

    Kept only to reproduce stored values: those runs scored a PDAG through one
    DAG extension (`dag_extension()`), which picks arbitrary directions for
    undirected edges and drops adjacencies when no extension exists.
    `pdag_agreement()` replaces it.

    :param A_hat: emitted PDAG adjacency
    :param B: true DAG
    :return: `skeleton_f1`, `orientation_accuracy` as stored by those runs
    """
    B_ext, _ = dag_extension(A_hat)
    scores = _get_skeleton_scores(B_ext, B)
    return scores


def _get_legacy_scores(B_hat: np.ndarray, truth: Truth) -> Dict[str, float]:
    """
    Compute the prototype's score keys with the prototype's own `score()`.

    :param B_hat: discovered graph (directed adjacency)
    :param truth: instance truth
    :return: `shd`, `tpr`, `fdr`, `f1`, `oos_r2_mean` as the prototype defines
    """
    env_like = types.SimpleNamespace(
        n_nodes=truth.B.shape[0],
        B_true=truth.B,
        X_train=truth.x_train,
        X_test=truth.x_test,
    )
    legacy = rlproto.score(B_hat, env_like)
    return legacy


def _score_against_pag(
    P_hat: np.ndarray,
    truth: Truth,
    B_dag: Optional[np.ndarray],
    lam: float,
) -> Dict[str, Any]:
    """
    Score endpoint marks against a PAG target.

    A DAG output cannot reach a PAG's marks, so there is no `shd_target` and
    no combined score (both NaN); the label-free prediction gain is kept.

    :param P_hat: endpoint marks of the output
    :param truth: instance truth with a PAG target
    :param B_dag: a DAG of the output for the prediction gain, or None
    :param lam: weight of the structural term (recorded only)
    :return: score dictionary
    """
    scores: Dict[str, Any] = {
        "target_class": TARGET_PAG,
        "n_true_edges": int(np.triu(truth.pag != 0, 1).sum()),
    }
    scores.update(score_pag(P_hat, truth.pag, truth.confounded))
    scores["shd_target"] = float("nan")
    scores["score"] = float("nan")
    if B_dag is not None:
        ll_hat, degenerate = pred_ll(
            B_dag, truth.moments_train, truth.moments_test
        )
        scores["pred_gain"] = ll_hat - truth.ll_empty
        scores["degenerate"] = degenerate
    scores["lam"] = lam
    return scores


def combined_score(
    shd_target: float,
    n_true_edges: int,
    *,
    pred_frac: float,
    informative: bool,
    lam: float,
) -> float:
    """
    Combine the structural distance and the prediction term into one score.

    When the instance's prediction term is uninformative the score is
    structural only.

    :param shd_target: structural distance to the identifiable target
    :param n_true_edges: number of true edges (normalizes the distance)
    :param pred_frac: share of the true graph's held-out gain recovered
    :param informative: whether the instance's prediction term is informative
    :param lam: weight of the structural term
    :return: combined score
    """
    shd_norm = shd_target / max(1, n_true_edges)
    if informative:
        score = -lam * shd_norm - (1 - lam) * (1 - pred_frac)
    else:
        score = -shd_norm
    # Adding 0.0 turns a negative zero (e.g., -0.5 * 0) into 0.0.
    return float(score) + 0.0


def score_pdag(
    A_hat: np.ndarray,
    truth: Truth,
    *,
    lam: float = 0.5,
    bidirected: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Score a PDAG output (e.g., PC or GES) against the instance truth.

    Marks are compared as emitted: an undirected edge matches only an
    undirected edge of the target, and a bidirected conflict (PC) matches no
    mark of a DAG or CPDAG target. The prediction term uses any DAG extension
    (all extensions of a valid PDAG are Markov equivalent, so they agree).

    :param A_hat: discovered PDAG adjacency (bidirected pairs as undirected)
    :param truth: instance truth
    :param lam: weight of the structural term
        - Default: 0.5
    :param bidirected: boolean matrix of the pairs marked bidirected
        - Default: None means no bidirected pair
    :return: score dictionary (the keys of `score_graph()` minus the legacy
        prototype keys and the directed-graph `skeleton_f1` and
        `orientation_accuracy`) plus `extendable` and `n_bidirected`;
        adjacency and orientation come from the emitted marks
        (`pdag_agreement()`), never from a DAG extension
    """
    if truth.target_class == TARGET_PAG:
        B_ext, extendable = dag_extension(A_hat)
        scores = {
            "is_dag": False,
            "extendable": extendable,
            "n_bidirected": (
                0 if bidirected is None else int(np.triu(bidirected, 1).sum())
            ),
        }
        scores.update(
            _score_against_pag(
                pag_marks_from_pdag(A_hat, bidirected=bidirected),
                truth,
                B_ext if extendable else None,
                lam,
            )
        )
        return scores
    n_true_edges = int(truth.B.sum())
    marks_hat = marks_from_pdag(A_hat)
    n_bidirected = 0
    if bidirected is not None:
        upper = np.triu(np.asarray(bidirected, dtype=bool), 1)
        marks_hat[upper] = MARK_BIDIRECTED
        n_bidirected = int(upper.sum())
    marks_true_target = truth.marks_target
    B_ext, extendable = dag_extension(A_hat)
    scores: Dict[str, Any] = {
        "is_dag": False,
        "extendable": extendable,
        "n_bidirected": n_bidirected,
        "target_class": truth.target_class,
        "n_true_edges": n_true_edges,
        "shd_dag": shd_marks(marks_hat, truth.marks_dag),
        "shd_cpdag": shd_marks(marks_hat, truth.marks_cpdag),
        "shd_target": shd_marks(marks_hat, marks_true_target),
    }
    scores.update(pdag_agreement(marks_hat, truth.B))
    if extendable:
        scores.update(_get_prediction_terms(B_ext, truth))
        scores["score"] = combined_score(
            scores["shd_target"],
            n_true_edges,
            pred_frac=scores["pred_frac"],
            informative=scores["prediction_informative"],
            lam=lam,
        )
    else:
        scores["score"] = float("nan")
    scores["lam"] = lam
    return scores


def score_graph(
    B_hat: np.ndarray,
    truth: Truth,
    *,
    lam: float = 0.5,
    legacy: bool = True,
) -> Dict[str, Any]:
    """
    Score a discovered directed graph against the instance truth.

    Cyclic graphs get structural scores only (`is_dag` False): the
    log-likelihood is not a valid likelihood for them and the combined score is
    left undefined (NaN).

    :param B_hat: discovered graph, directed adjacency
    :param truth: instance truth
    :param lam: weight of the structural term
        - Default: 0.5
    :param legacy: also compute the prototype's keys (slower; off in training)
        - Default: True
    :return: score dictionary
    """
    hdbg.dassert_eq(B_hat.shape, truth.B.shape)
    acyclic = is_dag(B_hat)
    if truth.target_class == TARGET_PAG:
        scores = {"is_dag": acyclic}
        scores.update(
            _score_against_pag(
                pag_marks_from_pdag(B_hat),
                truth,
                B_hat if acyclic else None,
                lam,
            )
        )
        return scores
    n_true_edges = int(truth.B.sum())
    marks_true_dag = truth.marks_dag
    marks_true_cpdag = truth.marks_cpdag
    marks_true_target = truth.marks_target
    marks_hat_dag = marks_from_pdag(B_hat)
    if acyclic:
        marks_hat_cpdag = marks_from_pdag(cpdag_from_dag(B_hat))
        marks_hat_target = target_marks(
            B_hat, truth.target_class, truth.nongauss
        )
    else:
        # No equivalence-class credit for a cyclic graph.
        marks_hat_cpdag = marks_hat_dag
        marks_hat_target = marks_hat_dag
    scores: Dict[str, Any] = {
        "is_dag": acyclic,
        "target_class": truth.target_class,
        "n_true_edges": n_true_edges,
        "shd_dag": shd_marks(marks_hat_dag, marks_true_dag),
        "shd_cpdag": shd_marks(marks_hat_cpdag, marks_true_cpdag),
        "shd_target": shd_marks(marks_hat_target, marks_true_target),
    }
    scores.update(_get_skeleton_scores(B_hat, truth.B))
    scores.update(pdag_agreement(marks_hat_dag, truth.B))
    if acyclic:
        scores.update(_get_prediction_terms(B_hat, truth))
        scores["score"] = combined_score(
            scores["shd_target"],
            n_true_edges,
            pred_frac=scores["pred_frac"],
            informative=scores["prediction_informative"],
            lam=lam,
        )
    else:
        scores["score"] = float("nan")
    scores["lam"] = lam
    if legacy:
        scores.update(_get_legacy_scores(B_hat, truth))
    return scores
