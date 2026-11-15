"""
Test the scripted agents: prototype parity, budgets and the DAG projection.

Import as:

import test.test_tukey_scripted as ttetuscr
"""

import logging
from typing import List, Tuple

import numpy as np

import helpers.hunit_test as hunitest
import tukey_env as tenv
import tukey_scoring as tscore
import tukey_scripted as tscrip
import tukey_tools as ttools

_LOG = logging.getLogger(__name__)


def _play(
    env: tenv.TukeyEnv, expert: tscrip.ScriptedExpert, seed: int
) -> Tuple[ttools.Evidence, List[ttools.Action]]:
    """
    Run the expert until it stops or the budget runs out.

    :param env: environment
    :param expert: scripted agent
    :param seed: instance seed
    :return: final evidence and the actions taken
    """
    evidence, _ = env.reset(seed=seed)
    actions = []
    while True:
        action = expert.next_action(evidence)
        if action.kind == ttools.STOP:
            break
        actions.append(action)
        evidence, _, _, truncated, _ = env.step(action)
        if truncated:
            break
    return evidence, actions


def _relabel(action: ttools.Action, new_index: List[int]) -> ttools.Action:
    """
    Rewrite an action for relabelled variables.

    :param action: action on the original labels
    :param new_index: `new_index[v]` is the new label of variable v
    :return: the same call on the new labels, canonicalized
    """
    args = [int(new_index[v]) for v in action.args]
    if action.kind == ttools.CI_TEST:
        return ttools.ci_test_action(args[0], args[1], tuple(args[2:]))
    if action.kind == ttools.REGRESS:
        return ttools.regress_action(args[0], tuple(args[1:]))
    if action.kind == ttools.LINGAM_DIR:
        return ttools.Action(ttools.LINGAM_DIR, tuple(sorted(args)))
    return ttools.Action(action.kind, tuple(args))


def _expert_calls(
    expert: tscrip.ScriptedExpert, x: np.ndarray, *, budget: int
) -> List[ttools.Action]:
    """
    Record the expert's calls on a dataset.

    :param expert: scripted expert
    :param x: training rows
    :param budget: tool budget
    :return: calls in order
    """
    evidence = ttools.Evidence(ttools.ToolCache(x), budget=budget)
    calls = []
    while True:
        action = expert.next_action(evidence)
        if action.kind == ttools.STOP:
            return calls
        evidence.record(action)
        calls.append(action)


# #############################################################################
# TestScriptedExpert
# #############################################################################


class TestScriptedExpert(hunitest.TestCase):
    """
    The tool-API rule matches the prototype and respects budgets.
    """

    def test1(self) -> None:
        """
        Through the tool API the rule emits the prototype's edges on seeds 0-19.
        """
        expert = tscrip.ScriptedExpert()
        n_equal = 0
        calls = []
        for seed in range(20):
            env = tenv.TukeyEnv(tenv.PROTO_CONFIG, budget=1000)
            evidence, _ = _play(env, expert, seed)
            reference = tscrip.proto_faithful_edges(env.d_train.to_numpy())
            n_equal += int(expert.emit_raw_edges(evidence) == reference)
            calls.append(evidence.calls_used)
        self.assertEqual(n_equal, 20)
        actual = f"calls {calls}"
        expected = (
            "calls [55, 53, 34, 51, 48, 46, 49, 54, 44, 38, 26, 36, 42, 40, "
            "42, 65, 44, 41, 40, 55]"
        )
        self.assert_equal(actual, expected)

    def test2(self) -> None:
        """
        Seed 0 reproduces the prototype's reported score (SHD 3, F1 0.667).
        """
        env = tenv.TukeyEnv(tenv.PROTO_CONFIG, budget=1000)
        evidence, _ = _play(env, tscrip.ScriptedExpert(), 0)
        edges = tscrip.ScriptedExpert().emit_raw_edges(evidence)
        B_hat, _ = tscore.edges_to_adjacency(edges, 6)
        scores = tscore.score_graph(B_hat, env._instance.truth)
        actual = str({key: scores[key] for key in ("shd", "f1", "tpr", "fdr")})
        expected = "{'shd': 3, 'f1': 0.667, 'tpr': 0.667, 'fdr': 0.333}"
        self.assert_equal(actual, expected)

    def test3(self) -> None:
        """
        Under small budgets the expert stays within budget and emits a DAG.
        """
        expert = tscrip.ScriptedExpert()
        cfg = tenv.GraphConfig(family="mixed", n_nodes=8, n_edges=12)
        for budget in (0, 1, 5, 10):
            env = tenv.TukeyEnv(cfg, budget=budget)
            for seed in range(5):
                scores, history = tscrip.run_episode(env, expert, seed=seed)
                self.assertLessEqual(len(history), budget)
                self.assertTrue(scores["is_dag"])

    def test4(self) -> None:
        """
        The next action depends only on the evidence (replays are identical).
        """
        expert = tscrip.ScriptedExpert()
        cfg = tenv.GraphConfig(family="linear_exp")
        _, actions1 = _play(tenv.TukeyEnv(cfg, budget=1000), expert, 3)
        _, actions2 = _play(tenv.TukeyEnv(cfg, budget=1000), expert, 3)
        self.assertEqual(actions1, actions2)
        self.assertEqual(len(set(actions1)), len(actions1))

    def test5(self) -> None:
        """
        Strength order: relabelling the columns relabels the whole trajectory.
        """
        cfg = tenv.GraphConfig(family="linear_exp")
        perm = [4, 2, 0, 5, 1, 3]
        new_index = list(np.argsort(perm))
        strength = tscrip.ScriptedExpert(order="strength")
        index = tscrip.ScriptedExpert(order="index")
        n_index_equal = 0
        for seed in range(5):
            x = tenv.sample_instance(cfg, seed).d_train.to_numpy()
            calls = _expert_calls(strength, x, budget=200)
            calls_perm = _expert_calls(strength, x[:, perm], budget=200)
            self.assertEqual([_relabel(a, new_index) for a in calls], calls_perm)
            calls = _expert_calls(index, x, budget=200)
            calls_perm = _expert_calls(index, x[:, perm], budget=200)
            relabelled = [_relabel(a, new_index) for a in calls]
            n_index_equal += int(relabelled == calls_perm)
        # The prototype's order depends on column indices.
        self.assertEqual(n_index_equal, 0)


# #############################################################################
# Test_project_to_dag
# #############################################################################


class Test_project_to_dag(hunitest.TestCase):
    """
    Cycles are removed by deleting the weakest edge.
    """

    def test1(self) -> None:
        """
        A 3-cycle loses its weakest edge; a DAG is unchanged.
        """
        corr = np.eye(3)
        corr[0, 1] = corr[1, 0] = 0.5
        corr[1, 2] = corr[2, 1] = 0.3
        corr[0, 2] = corr[2, 0] = -0.4
        cyclic = [(0, 1), (1, 2), (2, 0)]
        actual = str(tscrip.project_to_dag(cyclic, corr))
        actual += "\n" + str(tscrip.project_to_dag([(0, 1), (1, 2)], corr))
        expected = """
        ([(0, 1), (2, 0)], 1)
        ([(0, 1), (1, 2)], 0)
        """
        self.assert_equal(actual, expected, dedent=True)

    def test2(self) -> None:
        """
        Two cycles sharing an edge: deletions stop as soon as no cycle is left.
        """
        corr = np.full((4, 4), 0.5) + 0.5 * np.eye(4)
        corr[1, 2] = corr[2, 1] = 0.1
        # 0 -> 1 -> 2 -> 0 and 1 -> 2 -> 3 -> 1 share the weakest edge 1 -> 2.
        edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 1)]
        actual = str(tscrip.project_to_dag(edges, corr))
        expected = "([(0, 1), (2, 0), (2, 3), (3, 1)], 1)"
        self.assert_equal(actual, expected)


# #############################################################################
# TestScriptedPlus
# #############################################################################


class _PlusWithProtoOrientation(tscrip.ScriptedPlus):
    """
    `ScriptedPlus` that orients with the prototype's statistic, not LiNGAM's.

    It makes the same tool calls as `ScriptedPlus`, so the two differ only in
    the orientation step.
    """

    _orient = tscrip.ScriptedExpert._orient


class TestScriptedPlus(hunitest.TestCase):
    """
    The stronger rule orients symmetric-noise graphs better.
    """

    def test1(self) -> None:
        """
        Uniform noise: LiNGAM orientation beats the prototype's statistic.

        The prototype's statistic `|corr(x_c, |resid|)|` is zero in both
        directions under symmetric noise, so its orientations are guesses.
        Plus must beat the prototype, and also `plus_proto_orient`, which makes
        Plus's calls but orients as the prototype: the extra CI tests alone
        must not explain the gain.
        """
        # Prepare inputs.
        cfg = tenv.GraphConfig(family="linear_uniform")
        agents = {
            "proto": tscrip.ScriptedExpert,
            "plus": tscrip.ScriptedPlus,
            "plus_proto_orient": _PlusWithProtoOrientation,
        }
        shd = {name: 0 for name in agents}
        # Run test.
        for seed in range(15):
            for name, agent_class in agents.items():
                env = tenv.TukeyEnv(cfg, budget=10_000)
                scores, _ = tscrip.run_episode(env, agent_class(), seed=seed)
                shd[name] += scores["shd_dag"]
        _LOG.info("Total SHD over 15 graphs: %s", shd)
        # Check outputs.
        self.assertLess(shd["plus"], shd["proto"])
        self.assertLess(shd["plus"], shd["plus_proto_orient"])
