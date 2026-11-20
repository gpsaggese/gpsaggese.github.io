"""
Non-learned agents: the prototype's scripted rule and its tool-API version.

- `proto_faithful_edges()` runs the prototype's own `simulate_agent_output()`
  on the training rows; it is the reference implementation of the rule
- `ScriptedExpert` is the same rule as a state -> action function over
  `Evidence`: it plays through `TukeyEnv` with real tool calls, can run under a
  budget, and is the behaviour-cloning teacher
- `ScriptedPlus` is a stronger hand-written rule (two-variable CI tests,
  LiNGAM orientation of non-Gaussian pairs), a baseline for Milestone 2
- `project_to_dag()` removes cycles with a fixed rule (weakest `|r|` first)
- `run_episode()` plays an agent in an environment and scores its graph

The rule (prototype, `simulate_agent_output()`): for each pair i < j in index
order, keep the pair if `|r| >= 0.15` and no single other variable k gives
`|pcorr(i, j | k)| < 0.15` (tested in index order, stopping at the first that
removes it); orient a kept pair as i -> j if the residual of j on i depends
less on i than the residual of i on j depends on j.

Import as:

import tukey_scripted as tscrip
"""

import itertools
import json
import logging
import os
import tempfile
import types
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import helpers.hdbg as hdbg
import research.agentic_data_science.RL_for_auto_EDA_prototype as rlproto
import tukey_env as tenv
import tukey_scoring as tscore
import tukey_tools as ttools

_LOG = logging.getLogger(__name__)

# Threshold of the prototype for |r| and |pcorr|.
PROTO_THRESHOLD = 0.15


# #############################################################################
# Reference implementation
# #############################################################################


def proto_faithful_edges(x_train: np.ndarray) -> List[Tuple[int, int]]:
    """
    Run the prototype's scripted agent on training rows.

    :param x_train: training rows, shape (n_train, n_nodes)
    :return: edges `[(parent, child), ...]` exactly as the prototype emits them
    """
    env_like = types.SimpleNamespace(X_train=x_train, n_nodes=x_train.shape[1])
    with tempfile.TemporaryDirectory() as workdir:
        rlproto.simulate_agent_output(env_like, workdir)
        with open(os.path.join(workdir, "graph.json")) as file:
            edges = json.load(file)
    edges = [(int(parent), int(child)) for parent, child in edges]
    return edges


# #############################################################################
# Projection
# #############################################################################


def _find_cycle(B: np.ndarray) -> Optional[List[int]]:
    """
    Find one directed cycle.

    :param B: directed adjacency
    :return: nodes of a cycle in order, or None if `B` is acyclic
    """
    n_nodes = B.shape[0]
    # 0 = unvisited, 1 = on the current path, 2 = done.
    state = [0] * n_nodes
    path: List[int] = []

    def _visit(node: int) -> Optional[List[int]]:
        state[node] = 1
        path.append(node)
        for child in np.flatnonzero(B[node]):
            child = int(child)
            if state[child] == 1:
                return path[path.index(child) :]
            if state[child] == 0:
                cycle = _visit(child)
                if cycle is not None:
                    return cycle
        state[node] = 2
        path.pop()
        return None

    for start in range(n_nodes):
        if state[start] == 0:
            cycle = _visit(start)
            if cycle is not None:
                return cycle
    return None


def project_to_dag(
    edges: List[Tuple[int, int]], corr: np.ndarray
) -> Tuple[List[Tuple[int, int]], int]:
    """
    Remove cycles by deleting the weakest edge on each cycle found.

    The weakest edge has the smallest `|r|`; ties go to the smallest
    `(parent, child)`. Used only for agents whose rule can emit cycles.

    :param edges: `[(parent, child), ...]`
    :param corr: correlation matrix of the training rows
    :return: acyclic edge list and the number of deleted edges
    """
    n_nodes = corr.shape[0]
    B, _ = tscore.edges_to_adjacency(edges, n_nodes)
    n_deleted = 0
    cycle = _find_cycle(B)
    while cycle is not None:
        cycle_edges = [
            (cycle[k], cycle[(k + 1) % len(cycle)]) for k in range(len(cycle))
        ]
        weakest = min(cycle_edges, key=lambda e: (abs(corr[e[0], e[1]]), e))
        B[weakest] = 0
        n_deleted += 1
        cycle = _find_cycle(B)
    projected = [(int(i), int(j)) for i, j in np.argwhere(B == 1)]
    return projected, n_deleted


# #############################################################################
# ScriptedExpert
# #############################################################################


class ScriptedExpert:
    """
    The prototype's rule as a state -> action function over `Evidence`.

    `next_action()` returns the first call of the rule whose result is not yet
    in the evidence, so the agent can be interrupted by a budget at any point;
    `emit_graph()` turns whatever evidence exists into a DAG.
    """

    def __init__(
        self, *, threshold: float = PROTO_THRESHOLD, order: str = "index"
    ) -> None:
        """
        Build the agent.

        The processing order changes which calls a budget cuts, never the
        unbudgeted graph (each pair's decision uses only its own tests).

        :param threshold: cut-off for `|r|` and `|pcorr|`
            - Default: the prototype's 0.15
        :param order: processing order of pairs and conditioning variables
            - "index": the prototype's (pairs i < j in lexicographic order,
              conditioning variables in increasing index)
            - "strength": pairs by decreasing `|r|`, conditioning variables k
              by decreasing `max(|r_ik|, |r_jk|)`, and first the regression of
              the endpoint with the larger mean `|r|`, ties by index; the
              behaviour-cloning teacher, since a policy sees no column
              indices
        """
        hdbg.dassert_in(order, ("index", "strength"), "Unknown processing order")
        self.threshold = threshold
        self.order = order
        # Lists that depend only on the episode's evidence (see `_memo()`).
        self._memo_evidence: Optional[ttools.Evidence] = None
        self._memo_store: Dict[Any, Any] = {}

    def _memo(self, evidence: ttools.Evidence) -> Dict[Any, Any]:
        """
        Return the memo of the current episode, starting a new one if needed.

        Within an episode the pair order and each pair's call lists never
        change, and a finished pair stays finished, so they are built once and
        the scan resumes at the first unfinished pair. The memo keeps a
        reference to the evidence it belongs to, so a new episode's evidence
        can never be mistaken for the old one.

        :param evidence: evidence so far
        :return: memo dict of this episode
        """
        if self._memo_evidence is not evidence:
            self._memo_evidence = evidence
            self._memo_store = {}
        return self._memo_store

    def _pairs(self, evidence: ttools.Evidence) -> List[Tuple[int, int]]:
        """
        List the pairs `i < j` in processing order.

        :param evidence: evidence so far
        :return: pairs
        """
        memo = self._memo(evidence)
        if "pairs" not in memo:
            pairs = list(itertools.combinations(range(evidence.n_nodes), 2))
            if self.order == "strength":
                pairs.sort(key=lambda p: (-abs(evidence.corr[p[0], p[1]]), p))
            memo["pairs"] = pairs
        return memo["pairs"]

    def _ci_actions(
        self, evidence: ttools.Evidence, i: int, j: int
    ) -> List[ttools.Action]:
        """
        List the pair's CI tests in the rule's order.

        :param evidence: evidence so far
        :param i: first variable
        :param j: second variable
        :return: `ci_test` actions with each single other variable
        """
        memo = self._memo(evidence)
        key = ("ci", i, j)
        if key not in memo:
            others = [k for k in range(evidence.n_nodes) if k not in (i, j)]
            if self.order == "strength":
                corr = np.abs(evidence.corr)
                others.sort(key=lambda k: (-max(corr[i, k], corr[j, k]), k))
            memo[key] = [ttools.ci_test_action(i, j, (k,)) for k in others]
        return memo[key]

    def _regress_actions(
        self, evidence: ttools.Evidence, i: int, j: int
    ) -> List[ttools.Action]:
        """
        List the pair's two regressions in the rule's order.

        :param evidence: evidence so far
        :param i: lower-index variable
        :param j: higher-index variable
        :return: `regress` actions (j on i, i on j, in the prototype's order)
        """
        memo = self._memo(evidence)
        key = ("regress", i, j)
        if key not in memo:
            actions = [
                ttools.regress_action(j, (i,)),
                ttools.regress_action(i, (j,)),
            ]
            if self.order == "strength":
                # First regress the endpoint with the larger mean |r| to the
                # other variables (the node feature `mean_abs_r`; the pair's
                # own |r| is shared, so this compares the endpoints' links to
                # third variables).
                corr = np.abs(evidence.corr)
                mean_abs_r = (corr.sum(axis=1) - 1.0) / max(
                    1, evidence.n_nodes - 1
                )
                actions.sort(key=lambda a: (-mean_abs_r[a.args[0]], a.args[0]))
            memo[key] = actions
        return memo[key]

    def _next_pair_action(
        self, evidence: ttools.Evidence, i: int, j: int
    ) -> Optional[ttools.Action]:
        """
        Return the pair's next missing call, or None if the pair is finished.

        :param evidence: evidence so far
        :param i: first variable
        :param j: second variable
        :return: next action for the pair, or None
        """
        for action in self._ci_actions(evidence, i, j):
            result = evidence.results.get(action)
            if result is None:
                return action
            if abs(result["pcorr"]) < self.threshold:
                # Removed: no orientation needed.
                return None
        for action in self._regress_actions(evidence, i, j):
            if action not in evidence.results:
                return action
        return None

    def next_action(self, evidence: ttools.Evidence) -> ttools.Action:
        """
        Return the rule's next tool call, or STOP when done or out of budget.

        :param evidence: evidence so far
        :return: action
        """
        stop = ttools.Action(ttools.STOP)
        if evidence.budget_left <= 0:
            return stop
        pairs = self._pairs(evidence)
        memo = self._memo(evidence)
        # Every pair before the cursor is screened out or finished.
        for index in range(memo.get("cursor", 0), len(pairs)):
            i, j = pairs[index]
            if abs(evidence.corr[i, j]) < self.threshold:
                continue
            action = self._next_pair_action(evidence, i, j)
            if action is not None:
                memo["cursor"] = index
                return action
        memo["cursor"] = len(pairs)
        return stop

    def _is_removed(self, evidence: ttools.Evidence, i: int, j: int) -> bool:
        """
        Check whether an observed CI test removed the pair.

        :param evidence: evidence so far
        :param i: first variable
        :param j: second variable
        :return: True if some observed `|pcorr|` is below the threshold
        """
        for action in self._ci_actions(evidence, i, j):
            result = evidence.results.get(action)
            if result is not None and abs(result["pcorr"]) < self.threshold:
                return True
        return False

    def _is_kept(self, evidence: ttools.Evidence, i: int, j: int) -> bool:
        """
        Check whether the pair is in the emitted skeleton.

        :param evidence: evidence so far
        :param i: first variable
        :param j: second variable
        :return: True if the pair passes the screen and was not removed
        """
        passes_screen = abs(evidence.corr[i, j]) >= self.threshold
        kept = passes_screen and not self._is_removed(evidence, i, j)
        return kept

    def _orient(
        self, evidence: ttools.Evidence, i: int, j: int
    ) -> Tuple[int, int]:
        """
        Orient a kept pair from the evidence.

        :param evidence: evidence so far
        :param i: lower-index variable
        :param j: higher-index variable
        :return: `(parent, child)`
        """
        dep_ij = evidence.results.get(ttools.regress_action(j, (i,)))
        dep_ji = evidence.results.get(ttools.regress_action(i, (j,)))
        lingam = evidence.results.get(ttools.Action(ttools.LINGAM_DIR, (i, j)))
        if dep_ij is not None and dep_ji is not None:
            # As in the prototype: i -> j if dep(i, j) < dep(j, i).
            forward = dep_ij["resid_dep"] < dep_ji["resid_dep"]
        elif lingam is not None:
            forward = lingam["stat"] > 0
        else:
            forward = True
        edge = (i, j) if forward else (j, i)
        return edge

    def emit_raw_edges(self, evidence: ttools.Evidence) -> List[Tuple[int, int]]:
        """
        Emit the rule's edges from the evidence, possibly with cycles.

        A pair is kept unless an observed CI test removed it (untested pairs
        that pass the free `|r|` screen are kept); a kept pair is oriented by
        its two regressions if both were run, else by `lingam_dir` if run,
        else from the lower to the higher column index (columns are randomly
        permuted, so this is an uninformed guess).

        :param evidence: evidence so far
        :return: edges `[(parent, child), ...]` in pair order
        """
        edges = []
        for i, j in itertools.combinations(range(evidence.n_nodes), 2):
            if self._is_kept(evidence, i, j):
                edges.append(self._orient(evidence, i, j))
        return edges

    def emit_graph(self, evidence: ttools.Evidence) -> List[Tuple[int, int]]:
        """
        Emit a DAG: the rule's edges with cycles removed by `project_to_dag()`.

        :param evidence: evidence so far
        :return: acyclic edges `[(parent, child), ...]`
        """
        edges, _ = project_to_dag(self.emit_raw_edges(evidence), evidence.corr)
        return edges


# #############################################################################
# Episodes
# #############################################################################


class ScriptedPlus(ScriptedExpert):
    """
    A stronger hand-written rule: two-variable CI tests and LiNGAM orientation.

    Differences from the prototype's rule:
    - after the single-variable CI tests, a kept pair is also tested given every
      pair of other variables (until one removes it)
    - after the two regressions, if either residual rejects normality (Anderson-
      Darling statistic above `ad_threshold`), `lingam_dir` is called and
      orients the pair; otherwise the prototype's residual statistic does
    """

    def __init__(
        self,
        *,
        threshold: float = PROTO_THRESHOLD,
        ad_threshold: float = 0.787,
    ) -> None:
        """
        Build the agent.

        :param threshold: cut-off for `|r|` and `|pcorr|`
        :param ad_threshold: Anderson-Darling statistic above which a residual
            is called non-Gaussian
            - Default: 0.787, the 5% critical value for large samples
        """
        super().__init__(threshold=threshold, order="index")
        self.ad_threshold = ad_threshold

    def _ci_actions(
        self, evidence: ttools.Evidence, i: int, j: int
    ) -> List[ttools.Action]:
        """
        List single- then two-variable CI tests of a pair.

        :param evidence: evidence so far
        :param i: first variable
        :param j: second variable
        :return: `ci_test` actions
        """
        memo = self._memo(evidence)
        key = ("ci", i, j)
        if key not in memo:
            others = [k for k in range(evidence.n_nodes) if k not in (i, j)]
            actions = [ttools.ci_test_action(i, j, (k,)) for k in others]
            actions += [
                ttools.ci_test_action(i, j, cond)
                for cond in itertools.combinations(others, 2)
            ]
            memo[key] = actions
        return memo[key]

    def _non_gaussian(self, evidence: ttools.Evidence, i: int, j: int) -> bool:
        """
        Check whether either regression residual of the pair rejects normality.

        :param evidence: evidence so far
        :param i: first variable
        :param j: second variable
        :return: True if an observed residual's AD statistic is large
        """
        stats = [
            evidence.results[action]["resid_ad"]
            for action in self._regress_actions(evidence, i, j)
            if action in evidence.results
        ]
        return any(stat > self.ad_threshold for stat in stats)

    def _next_pair_action(
        self, evidence: ttools.Evidence, i: int, j: int
    ) -> Optional[ttools.Action]:
        """
        Return the pair's next missing call, adding `lingam_dir` when needed.

        :param evidence: evidence so far
        :param i: first variable
        :param j: second variable
        :return: next action for the pair, or None
        """
        action = super()._next_pair_action(evidence, i, j)
        if action is not None or self._is_removed(evidence, i, j):
            return action
        lingam = ttools.Action(ttools.LINGAM_DIR, (i, j))
        if self._non_gaussian(evidence, i, j) and lingam not in evidence.results:
            return lingam
        return None

    def _orient(
        self, evidence: ttools.Evidence, i: int, j: int
    ) -> Tuple[int, int]:
        """
        Orient a kept pair with `lingam_dir` when it was called.

        :param evidence: evidence so far
        :param i: lower-index variable
        :param j: higher-index variable
        :return: `(parent, child)`
        """
        lingam = evidence.results.get(ttools.Action(ttools.LINGAM_DIR, (i, j)))
        if lingam is not None:
            edge = (i, j) if lingam["stat"] > 0 else (j, i)
        else:
            edge = super()._orient(evidence, i, j)
        return edge


def run_episode(
    env: tenv.TukeyEnv, agent: Any, *, seed: int
) -> Tuple[Dict[str, Any], List[ttools.Action]]:
    """
    Play an agent for one episode and score its graph.

    :param env: environment
    :param agent: object with `next_action(evidence)` and `emit_graph(evidence)`
    :param seed: instance seed
    :return: score dictionary and the tool calls made
    """
    evidence, _ = env.reset(seed=seed)
    while True:
        action = agent.next_action(evidence)
        evidence, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    edges = agent.emit_graph(evidence)
    B_hat, _ = tscore.edges_to_adjacency(edges, env.cfg.n_nodes)
    hdbg.dassert(tscore.is_dag(B_hat), "Agents must emit a DAG")
    scores = env.submit_graph(edges)
    return scores, list(evidence.history)
