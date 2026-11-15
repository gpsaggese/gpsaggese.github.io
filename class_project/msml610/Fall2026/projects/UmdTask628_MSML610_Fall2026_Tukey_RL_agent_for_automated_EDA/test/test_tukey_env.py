"""
Test the Tukey environment: generator contracts, prototype parity, leaks.

This is integration gate G3 and the environment part of G8. The `proto` family
must reproduce the prototype bit for bit; the linear families must meet their
contracts (acyclic, exact edge counts, noise-share floor, edge-strength margin,
train-only standardization, per-instance target class); nothing an agent
receives may carry the hidden graph or the test rows.

Import as:

import test.test_tukey_env as ttetuenv
"""

import collections
import dataclasses
import logging
from typing import Dict, List

import numpy as np
import scipy.stats

import helpers.hunit_test as hunitest
import research.agentic_data_science.RL_for_auto_EDA_prototype as rlproto
import tukey_env as tenv
import tukey_scoring as tscore
import tukey_tools as ttools

_LOG = logging.getLogger(__name__)

_LINEAR_CONFIGS: Dict[str, tenv.GraphConfig] = {
    "linear_exp": tenv.GraphConfig(family="linear_exp"),
    "linear_uniform": tenv.GraphConfig(family="linear_uniform"),
    "linear_gauss": tenv.GraphConfig(family="linear_gauss"),
    "linear_gauss_eqvar_raw": tenv.EQVAR_RAW_CONFIG,
    "mixed": tenv.GraphConfig(family="mixed"),
}


# #############################################################################
# Fixtures
# #############################################################################


def _check_contracts(test: hunitest.TestCase, inst: tenv.Instance) -> None:
    """
    Assert the contracts every linear-family instance must meet.

    :param test: calling test case
    :param inst: sampled instance
    """
    cfg = inst.cfg
    B = inst.truth.B
    x_train = inst.d_train.to_numpy()
    x_test = inst.d_test.to_numpy()
    test.assertTrue(tscore.is_dag(B))
    test.assertEqual(int(B.sum()), cfg.n_edges)
    test.assertTrue(np.array_equal(B, (inst.W != 0).astype(int)))
    # `int(3000 * (1 - 0.33))` is 2009 in floating point, as in the prototype.
    test.assertEqual(x_train.shape, (2009, cfg.n_nodes))
    test.assertEqual(x_test.shape, (991, cfg.n_nodes))
    test.assertTrue(np.isfinite(x_train).all() and np.isfinite(x_test).all())
    if cfg.standardize == "post":
        test.assertLess(np.abs(x_train.mean(axis=0)).max(), 1e-12)
        test.assertLess(np.abs(x_train.std(axis=0) - 1).max(), 1e-12)
    noise_var = inst.noise_std**2
    cov = tenv.population_covariance(inst.W, noise_var)
    test.assertGreaterEqual((noise_var / np.diag(cov)).min(), 0.1 - 1e-9)
    test.assertGreaterEqual(tenv.edge_strengths(inst.W, noise_var).min(), 0.1)
    nongauss = np.array([f != "gaussian" for f in inst.noise_families])
    test.assertTrue(np.array_equal(inst.truth.nongauss, nongauss))


def _chain(weights: List[float]) -> np.ndarray:
    """
    Build the weighted adjacency of a chain 0 -> 1 -> ... with given weights.

    :param weights: weight of each consecutive edge
    :return: weighted adjacency
    """
    n_nodes = len(weights) + 1
    W = np.zeros((n_nodes, n_nodes))
    for k, weight in enumerate(weights):
        W[k, k + 1] = weight
    return W


# #############################################################################
# Test_sample_instance_proto
# #############################################################################


class Test_sample_instance_proto(hunitest.TestCase):
    """
    The `proto` family reproduces the prototype exactly.
    """

    def test1(self) -> None:
        """
        Seeds 0-19 give the prototype's weights, graph and split bit for bit.
        """
        for seed in range(20):
            inst = tenv.sample_instance(tenv.PROTO_CONFIG, seed)
            env = rlproto.Environment(seed=seed)
            self.assertTrue(np.array_equal(inst.W, env.W))
            self.assertTrue(np.array_equal(inst.truth.B, env.B_true))
            self.assertTrue(np.array_equal(inst.d_train.to_numpy(), env.X_train))
            self.assertTrue(np.array_equal(inst.d_test.to_numpy(), env.X_test))

    def test2(self) -> None:
        """
        The scorer's legacy keys equal the prototype's `score()` dictionaries.
        """
        for seed in range(5):
            inst = tenv.sample_instance(tenv.PROTO_CONFIG, seed)
            env = rlproto.Environment(seed=seed)
            for B_hat in (env.B_true, np.zeros_like(env.B_true), env.B_true.T):
                scores = tscore.score_graph(B_hat, inst.truth)
                expected = rlproto.score(B_hat, env)
                actual = {key: scores[key] for key in expected}
                self.assertEqual(actual, expected)

    def test3(self) -> None:
        """
        The prototype family refuses standardization.
        """
        cfg = dataclasses.replace(tenv.PROTO_CONFIG, standardize="post")
        with self.assertRaises(AssertionError):
            tenv.sample_instance(cfg, 0)


# #############################################################################
# Test_sample_instance_linear
# #############################################################################


class Test_sample_instance_linear(hunitest.TestCase):
    """
    The linear families meet their contracts.
    """

    def test1(self) -> None:
        """
        Every family meets the contracts on 50 seeds; targets follow the noise.
        """
        lines = []
        for name, cfg in _LINEAR_CONFIGS.items():
            targets: collections.Counter = collections.Counter()
            n_rejected = 0
            for seed in range(50):
                inst = tenv.sample_instance(cfg, seed)
                _check_contracts(self, inst)
                targets[inst.truth.target_class] += 1
                n_rejected += inst.n_rejected
            lines.append(f"{name}: {dict(sorted(targets.items()))} {n_rejected}")
        actual = "\n".join(lines)
        expected = """
        linear_exp: {'dag': 50} 0
        linear_uniform: {'dag': 50} 0
        linear_gauss: {'cpdag': 50} 0
        linear_gauss_eqvar_raw: {'dag': 50} 0
        mixed: {'cpdag': 15, 'dag': 35} 0
        """
        self.assert_equal(actual, expected, dedent=True)

    def test2(self) -> None:
        """
        The same seed gives the same instance; another seed a different one.
        """
        cfg = _LINEAR_CONFIGS["mixed"]
        inst1 = tenv.sample_instance(cfg, 7)
        inst2 = tenv.sample_instance(cfg, 7)
        inst3 = tenv.sample_instance(cfg, 8)
        self.assertTrue(inst1.d_train.equals(inst2.d_train))
        self.assertTrue(np.array_equal(inst1.W, inst2.W))
        self.assertFalse(inst1.d_train.equals(inst3.d_train))

    def test3(self) -> None:
        """
        Families share graph and weights per seed and differ only in noise.
        """
        for seed in range(10):
            inst_exp = tenv.sample_instance(_LINEAR_CONFIGS["linear_exp"], seed)
            inst_gauss = tenv.sample_instance(
                _LINEAR_CONFIGS["linear_gauss"], seed
            )
            self.assertTrue(np.array_equal(inst_exp.W, inst_gauss.W))
            self.assertFalse(inst_exp.d_train.equals(inst_gauss.d_train))

    def test4(self) -> None:
        """
        The sampled noise has the family's shape (raw scale, true weights).
        """
        lines = []
        for family in ("linear_exp", "linear_uniform", "linear_gauss"):
            cfg = tenv.GraphConfig(
                family=family, noise_scale="equal", standardize="none"
            )
            inst = tenv.sample_instance(cfg, 11)
            x = inst.d_train.to_numpy()
            noise = x - x @ inst.W
            skew = scipy.stats.skew(noise, axis=0)
            kurt = scipy.stats.kurtosis(noise, axis=0)
            lines.append(
                f"{family}: skew {skew.min():.1f}..{skew.max():.1f}, "
                f"kurt {kurt.min():.1f}..{kurt.max():.1f}"
            )
        actual = "\n".join(lines)
        expected = """
        linear_exp: skew 1.6..1.9, kurt 3.3..5.5
        linear_uniform: skew -0.0..0.0, kurt -1.2..-1.2
        linear_gauss: skew -0.1..0.0, kurt -0.1..0.3
        """
        self.assert_equal(actual, expected, dedent=True)

    def test5(self) -> None:
        """
        Column order does not reveal edge direction (500 graphs).
        """
        n_forward = 0
        n_edges = 0
        for seed in range(500):
            B = tenv.sample_instance(_LINEAR_CONFIGS["linear_exp"], seed).truth.B
            parents, children = np.nonzero(B)
            n_forward += int((parents < children).sum())
            n_edges += len(parents)
        frac = n_forward / n_edges
        _LOG.info("Edges from a lower to a higher column: %.3f", frac)
        self.assertLess(abs(frac - 0.5), 0.05)

    def test6(self) -> None:
        """
        The equal-variance raw family refuses other settings.
        """
        cfg = tenv.GraphConfig(family="linear_gauss_eqvar_raw")
        with self.assertRaises(AssertionError):
            tenv.sample_instance(cfg, 0)

    def test7(self) -> None:
        """
        Mixed noise families with equal raw noise variances target the DAG.

        Seed 10 has one edge, between its two Gaussian nodes: equal variances
        identify its direction, so the reversed edge costs 1.
        """
        # Prepare inputs.
        cfg = tenv.GraphConfig(
            family="mixed_node",
            n_nodes=3,
            n_edges=1,
            noise_scale="equal",
            standardize="none",
        )
        # Prepare outputs.
        expected = """
        families ('gaussian', 'gaussian', 'uniform'), edges [[1, 0]]
        target dag: linear, equal noise variances, raw scale
        true: shd_target 0, score 0.0
        reversed: shd_target 1, score -0.5
        """
        # Run test.
        inst = tenv.sample_instance(cfg, 10)
        B = inst.truth.B
        scores_true = tscore.score_graph(B, inst.truth, legacy=False)
        scores_reversed = tscore.score_graph(B.T, inst.truth, legacy=False)
        # Check outputs.
        actual = (
            f"families {inst.noise_families}, "
            f"edges {np.argwhere(B).tolist()}\n"
            f"target {inst.truth.target_class}: {inst.target_reason}\n"
            f"true: shd_target {scores_true['shd_target']}, "
            f"score {scores_true['score']}\n"
            f"reversed: shd_target {scores_reversed['shd_target']}, "
            f"score {scores_reversed['score']}"
        )
        self.assert_equal(actual, expected, dedent=True)

    def test8(self) -> None:
        """
        A single variable is refused: the tools and scores compare pairs.
        """
        # Prepare inputs.
        cfg = tenv.GraphConfig(n_nodes=1, n_edges=0, family="linear_exp")
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            tenv.sample_instance(cfg, 0)


# #############################################################################
# Test_generator_checks
# #############################################################################


class Test_generator_checks(hunitest.TestCase):
    """
    Check the signal cap, the edge margin and the difficulty covariate.
    """

    def test1(self) -> None:
        """
        The cap scales down deep weights of a strong chain to a 10% noise share.
        """
        W = _chain([2.0, 2.0, 2.0])
        noise_var = np.ones(4)
        W_capped = tenv.cap_signal_share(W, noise_var)
        cov = tenv.population_covariance(W_capped, noise_var)
        shares = noise_var / np.diag(cov)
        actual = (
            f"weights {np.round(np.diag(W_capped, k=1), 4).tolist()}\n"
            f"shares {np.round(shares, 4).tolist()}"
        )
        expected = """
        weights [2.0, 1.3416, 0.9487]
        shares [1.0, 0.2, 0.1, 0.1]
        """
        self.assert_equal(actual, expected, dedent=True)

    def test2(self) -> None:
        """
        The cap leaves a weak chain unchanged.
        """
        W = _chain([0.8, 0.8, 0.8])
        W_capped = tenv.cap_signal_share(W, np.ones(4))
        self.assertTrue(np.array_equal(W, W_capped))

    def test3(self) -> None:
        """
        Path cancellation: edges pass the margin, the marginal correlation is 0.

        In 0 -> 1 -> 2 with 0 -> 2 of weight -1, x2 = e1 + e2, so x0 and x2 are
        uncorrelated although 0 -> 2 is an edge; given x1 it is visible.
        """
        W = np.zeros((3, 3))
        W[0, 1] = 1.0
        W[1, 2] = 1.0
        W[0, 2] = -1.0
        noise_var = np.ones(3)
        strengths = tenv.edge_strengths(W, noise_var)
        actual = (
            f"strengths {np.round(strengths, 4).tolist()}\n"
            f"margin {tenv.passes_edge_margin(W, noise_var)}\n"
            f"min marginal {tenv.min_partial_corr(W, noise_var, max_cond=0):.4f}"
        )
        expected = """
        strengths [0.7071, 0.5774, 0.7071]
        margin True
        min marginal 0.0000
        """
        self.assert_equal(actual, expected, dedent=True)

    def test4(self) -> None:
        """
        A weak edge into a noisy child fails the margin.
        """
        W = np.zeros((2, 2))
        W[0, 1] = 0.05
        self.assertFalse(tenv.passes_edge_margin(W, np.ones(2)))


# #############################################################################
# TestTukeyEnv
# #############################################################################


class TestTukeyEnv(hunitest.TestCase):
    """
    Check the episode API and what reaches an agent.
    """

    def test1(self) -> None:
        """
        Reset gives evidence built from D_train only and a whitelisted `info`.
        """
        env = tenv.TukeyEnv(_LINEAR_CONFIGS["linear_exp"])
        evidence, info = env.reset(seed=3)
        self.assertEqual(
            sorted(info), ["budget", "n_nodes", "n_train", "tool_costs"]
        )
        self.assertEqual(info["budget"], 18)
        x_train = env.d_train.to_numpy()
        self.assertTrue(
            np.array_equal(evidence.corr, np.corrcoef(x_train, rowvar=False))
        )
        # `toolset` is the agent's own configuration, not instance data.
        self.assertEqual(
            sorted(vars(evidence)),
            [
                "_cache",
                "budget",
                "corr",
                "history",
                "n_nodes",
                "n_train",
                "results",
                "toolset",
            ],
        )
        self.assertEqual(evidence.toolset, "m3")
        # The only data the tools hold are the training rows.
        arrays = [
            v for v in vars(evidence._cache).values() if hasattr(v, "shape")
        ]
        for array in arrays:
            self.assertNotEqual(array.shape, env.d_test.shape)
            # No array holds more rows than the training split (e.g., all
            # 3,000 rows with the test rows inside).
            self.assertLessEqual(array.shape[0], max(len(x_train), 1))
        self.assertTrue(np.array_equal(evidence._cache._x, x_train))

    def test2(self) -> None:
        """
        Every tool result is reproduced by a fresh cache on D_train alone.
        """
        env = tenv.TukeyEnv(_LINEAR_CONFIGS["mixed"], budget=300)
        evidence, _ = env.reset(seed=4)
        fresh = ttools.ToolCache(env.d_train.to_numpy())
        for action in ttools.enumerate_tool_actions(6):
            self.assertEqual(evidence.record(action), fresh.run(action))

    def test3(self) -> None:
        """
        Steps cost their price; budget, repeats, STOP and reuse are enforced.
        """
        env = tenv.TukeyEnv(_LINEAR_CONFIGS["linear_gauss"], budget=2)
        env.reset(seed=5)
        action = ttools.ci_test_action(0, 1, (2,))
        _, reward, terminated, truncated, _ = env.step(action)
        self.assertEqual((reward, terminated, truncated), (-0.001, False, False))
        with self.assertRaises(AssertionError):
            env.step(action)
        _, _, _, truncated, _ = env.step(ttools.Action(ttools.NONGAUSS, (0,)))
        self.assertTrue(truncated)
        self.assertEqual(env.legal_actions(), [ttools.Action(ttools.STOP)])
        _, reward, terminated, _, _ = env.step(ttools.Action(ttools.STOP))
        self.assertEqual((reward, terminated), (0.0, True))
        with self.assertRaises(AssertionError):
            env.step(ttools.Action(ttools.NONGAUSS, (1,)))

    def test4(self) -> None:
        """
        Submitting G* scores 0, the empty graph -1; bad edges are dropped.
        """
        env = tenv.TukeyEnv(_LINEAR_CONFIGS["linear_exp"])
        env.reset(seed=6)
        true_edges = [tuple(e) for e in np.argwhere(env._instance.truth.B == 1)]
        scores = env.submit_graph(true_edges + [(0, 0), (0, 9)])
        actual = (
            f"true: score {scores['score']}, shd_target {scores['shd_target']}, "
            f"dropped {scores['n_dropped_edges']}, calls {scores['calls_used']}"
        )
        env.reset(seed=6)
        scores = env.submit_graph([])
        actual += f"\nempty: score {scores['score']}, shd {scores['shd_target']}"
        expected = """
        true: score 0.0, shd_target 0, dropped 2, calls 0
        empty: score -1.0, shd 6
        """
        self.assert_equal(actual, expected, dedent=True)

    def test5(self) -> None:
        """
        A second submission in one episode is refused until the next reset.
        """
        env = tenv.TukeyEnv(_LINEAR_CONFIGS["linear_exp"])
        env.reset(seed=6)
        env.submit_graph([])
        with self.assertRaises(AssertionError):
            env.submit_graph([(0, 1)])
        env.reset(seed=6)
        scores = env.submit_graph([])
        self.assertEqual(scores["score"], -1.0)

    def test6(self) -> None:
        """
        The reward oracle scores any DAG without ending the episode.
        """
        env = tenv.TukeyEnv(_LINEAR_CONFIGS["linear_exp"])
        env.reset(seed=6)
        oracle = env.reward_oracle()
        B_true = env.instance.truth.B
        empty = np.zeros_like(B_true)
        # G* scores 0 and the empty graph -1 (as in `test4`); a repeated graph
        # comes from the cache with the same score.
        actual = [oracle(B_true), oracle(empty), oracle(empty)]
        # The episode goes on: a tool call and the submission still run.
        env.step(ttools.Action(ttools.NONGAUSS, (0,)))
        actual.append(env.submit_graph([])["score"])
        self.assertEqual(actual, [0.0, -1.0, -1.0, -1.0])


# #############################################################################
# Test_more_families
# #############################################################################


_MORE_CONFIGS: Dict[str, tenv.GraphConfig] = {
    "mixed_node": tenv.GraphConfig(family="mixed_node"),
    "nonlinear": tenv.GraphConfig(family="nonlinear", n_nodes=8, n_edges=10),
    "iscm": tenv.GraphConfig(family="iscm"),
    "pgmpy_lg": tenv.GraphConfig(family="pgmpy_lg"),
    "sklearn_reg": tenv.GraphConfig(family="sklearn_reg", n_nodes=7, n_edges=3),
    "sklearn_confounder": tenv.GraphConfig(
        family="sklearn_confounder", n_nodes=6, n_edges=2, confounder="observed"
    ),
    "scale_free": tenv.GraphConfig(n_nodes=8, n_edges=12, topology="scale_free"),
    "chain": tenv.GraphConfig(n_nodes=8, n_edges=7, topology="chain"),
    "star": tenv.GraphConfig(n_nodes=8, n_edges=7, topology="star"),
    "layered": tenv.GraphConfig(n_nodes=9, n_edges=12, topology="layered"),
}


class Test_more_families(hunitest.TestCase):
    """
    Contracts and family-specific checks of the M1 families and topologies.
    """

    def test1(self) -> None:
        """
        Every family: acyclic, exact edge count, standardized, finite; targets.
        """
        lines = []
        for name, cfg in _MORE_CONFIGS.items():
            targets: collections.Counter = collections.Counter()
            for seed in range(20):
                inst = tenv.sample_instance(cfg, seed)
                B = inst.truth.B
                x_train = inst.d_train.to_numpy()
                self.assertTrue(tscore.is_dag(B))
                self.assertEqual(int(B.sum()), cfg.n_edges)
                self.assertTrue(np.isfinite(x_train).all())
                self.assertLess(np.abs(x_train.std(axis=0) - 1).max(), 1e-12)
                if inst.mechanism == "linear":
                    noise_var = inst.noise_std**2
                    cov = tenv.population_covariance(inst.W, noise_var)
                    share = (noise_var / np.diag(cov)).min()
                    self.assertGreaterEqual(share, 0.1 - 1e-9)
                targets[inst.truth.target_class] += 1
            lines.append(f"{name}: {dict(sorted(targets.items()))}")
        actual = "\n".join(lines)
        expected = """
        mixed_node: {'dag': 3, 'ngdag_pattern': 17}
        nonlinear: {'dag': 20}
        iscm: {'cpdag': 5, 'dag': 15}
        pgmpy_lg: {'cpdag': 20}
        sklearn_reg: {'cpdag': 20}
        sklearn_confounder: {'cpdag': 20}
        scale_free: {'cpdag': 4, 'dag': 16}
        chain: {'cpdag': 5, 'dag': 15}
        star: {'cpdag': 5, 'dag': 15}
        layered: {'cpdag': 6, 'dag': 14}
        """
        self.assert_equal(actual, expected, dedent=True)

    def test2(self) -> None:
        """
        Quadratic mechanism: a symmetric parent is uncorrelated with its child.
        """
        cfg = tenv.GraphConfig(
            family="nonlinear", n_nodes=2, n_edges=1, standardize="none"
        )
        found = 0
        for seed in range(200):
            inst = tenv.sample_instance(cfg, seed)
            symmetric = inst.noise_families[0] in ("gaussian", "uniform")
            if inst.mechanism != "quadratic" or not symmetric:
                continue
            (parent,), (child,) = np.nonzero(inst.truth.B)
            x = inst.d_train.to_numpy()
            linear = abs(np.corrcoef(x[:, parent], x[:, child])[0, 1])
            squared = abs(np.corrcoef(x[:, parent] ** 2, x[:, child])[0, 1])
            self.assertLess(linear, 0.1)
            self.assertGreater(squared, 0.2)
            found += 1
        self.assertGreater(found, 10)

    def test3(self) -> None:
        """
        iSCM: every variable has unit variance before any post-processing.
        """
        cfg = tenv.GraphConfig(
            family="iscm", n_nodes=8, n_edges=12, standardize="none"
        )
        for seed in range(10):
            x = tenv.sample_instance(cfg, seed).d_train.to_numpy()
            self.assertLess(np.abs(x.var(axis=0) - 1).max(), 0.15)

    def test4(self) -> None:
        """
        Observed common cause: T1 and T2 are dependent, independent given U.
        """
        cfg = _MORE_CONFIGS["sklearn_confounder"]
        for seed in range(10):
            inst = tenv.sample_instance(cfg, seed)
            (source,) = np.flatnonzero(inst.truth.B.sum(axis=1))
            t_1, t_2 = np.flatnonzero(inst.truth.B[source])
            cache = ttools.ToolCache(inst.d_train.to_numpy())
            marginal = cache.fisher_z(cache.corr[t_1, t_2], 0)[1]
            given_u = cache.run(ttools.ci_test_action(t_1, t_2, (source,)))
            self.assertLess(marginal, 1e-6)
            self.assertGreater(given_u["p"], 0.001)

    def test5(self) -> None:
        """
        Topologies: chain is a path, star has a hub, layers are short.
        """
        for seed in range(10):
            B = tenv.sample_instance(_MORE_CONFIGS["chain"], seed).truth.B
            self.assertEqual(sorted(B.sum(axis=0).tolist()), [0] + [1] * 7)
            self.assertEqual(sorted(B.sum(axis=1).tolist()), [0] + [1] * 7)
            B = tenv.sample_instance(_MORE_CONFIGS["star"], seed).truth.B
            self.assertEqual(int(B.sum(axis=1).max()), 7)
            B = tenv.sample_instance(_MORE_CONFIGS["layered"], seed).truth.B
            # 9 nodes in 3 layers: no directed path longer than 2 edges.
            self.assertEqual(int(np.linalg.matrix_power(B, 3).sum()), 0)


def _fisher_z_p(x: np.ndarray, i: int, j: int, cond: List[int]) -> float:
    """
    Compute the Fisher-z p-value of the partial correlation of i and j.

    :param x: data
    :param i: first variable
    :param j: second variable
    :param cond: conditioning variables
    :return: two-sided p-value
    """
    idx = [i, j] + cond
    precision = np.linalg.inv(np.corrcoef(x[:, idx].T))
    r = -precision[0, 1] / np.sqrt(precision[0, 0] * precision[1, 1])
    z = np.sqrt(x.shape[0] - len(cond) - 3) * np.arctanh(r)
    p_value = float(2 * scipy.stats.norm.sf(abs(z)))
    return p_value


def _hidden_instance(variant: str, seed: int) -> tenv.Instance:
    """
    Sample a 6-variable instance with a hidden common cause.

    :param variant: "hidden_a", "hidden_b" or "hidden_c"
    :param seed: instance seed
    :return: instance
    """
    cfg = tenv.GraphConfig(
        n_nodes=6,
        n_edges=4 if variant == "hidden_b" else 2,
        family="hidden_confounder",
        confounder=variant,
    )
    inst = tenv.sample_instance(cfg, seed)
    return inst


class Test_hidden_confounder(hunitest.TestCase):
    """
    Check the hidden-confounder family: PAG targets and what the data show.
    """

    def test1(self) -> None:
        """
        Variant a: T1 o-o T2 only; dependent given any other observed variable.
        """
        for seed in range(5):
            inst = _hidden_instance("hidden_a", seed)
            ((i, j),) = inst.truth.confounded
            self.assertEqual(inst.truth.target_class, tscore.TARGET_PAG)
            self.assertEqual(int(inst.truth.B.sum()), 0)
            expected = np.zeros((6, 6), dtype=int)
            expected[i, j] = expected[j, i] = tscore.PAG_CIRCLE
            np.testing.assert_array_equal(inst.truth.pag, expected)
            x = inst.d_train.to_numpy()
            for k in sorted(set(range(6)) - {i, j}):
                self.assertLess(_fisher_z_p(x, i, j, [k]), 1e-6)

    def test2(self) -> None:
        """
        Variant b: A o-> T1 <-> T2 <-o B, and A is independent of T2.
        """
        z_scores = []
        for seed in range(200):
            inst = _hidden_instance("hidden_b", seed)
            ((t1, t2),) = inst.truth.confounded
            parents = {int(j): int(i) for i, j in np.argwhere(inst.truth.B)}
            self.assertEqual(sorted(parents), [t1, t2])
            pag = inst.truth.pag
            self.assertEqual(int((pag != 0).sum()), 6)
            self.assertEqual((pag[t1, t2], pag[t2, t1]), (1, 1))
            for target, cause in parents.items():
                self.assertEqual(
                    (pag[cause, target], pag[target, cause]),
                    (tscore.PAG_CIRCLE, tscore.PAG_ARROW),
                )
            x = inst.d_train.to_numpy()
            r = np.corrcoef(x[:, parents[t1]], x[:, t2])[0, 1]
            z_scores.append(np.sqrt(x.shape[0] - 3) * np.arctanh(r))
        # Independence is checked on the spread over seeds: a single seed
        # rejects at alpha = 0.001 once in a thousand draws, while a real
        # effect of A on T2 at the generator's weakest shares gives z ~ 12.
        z_scores = np.array(z_scores)
        self.assertLess(abs(z_scores.mean()), 0.3)
        self.assertLess(np.mean(np.abs(z_scores) > 1.96), 0.1)
        self.assertLess(np.abs(z_scores).max(), 5.0)

    def test3(self) -> None:
        """
        Variant c: every observed variable comes from one non-Gaussian family.
        """
        for seed in range(5):
            inst = _hidden_instance("hidden_c", seed)
            family = inst.noise_families[0]
            self.assertEqual(set(inst.noise_families), {family})
            ((i, j),) = inst.truth.confounded
            x = inst.d_train.to_numpy()
            for k in sorted(set(range(6)) - {i, j}):
                if family == "exponential":
                    self.assertGreater(scipy.stats.skew(x[:, k]), 1.5)
                else:
                    self.assertEqual(family, "uniform")
                    self.assertLess(scipy.stats.kurtosis(x[:, k]), -1.0)

    def test4(self) -> None:
        """
        A DAG output gets PAG metrics, no combined score and a prediction gain.
        """
        inst = _hidden_instance("hidden_a", 0)
        ((i, j),) = inst.truth.confounded
        empty = tscore.score_graph(np.zeros((6, 6), dtype=int), inst.truth)
        claim = np.zeros((6, 6), dtype=int)
        claim[i, j] = 1
        claimed = tscore.score_graph(claim, inst.truth)
        for scores in (empty, claimed):
            self.assertTrue(np.isnan(scores["score"]))
            self.assertTrue(np.isnan(scores["shd_target"]))
        self.assertEqual(empty["confounded_adjacent"], 0)
        self.assertEqual(claimed["confounded_adjacent"], 1)
        self.assertEqual(claimed["confounded_directed"], 1)
        self.assertEqual(empty["pred_gain"], 0.0)
        self.assertGreater(claimed["pred_gain"], 0.0)


class Test_rescale(hunitest.TestCase):
    """
    Check the "rescale" scaling used by the variance-shortcut suites.
    """

    def test1(self) -> None:
        """
        Each raw column is multiplied by one factor in [0.1, 10].
        """
        raw = tenv.sample_instance(tenv.EQVAR_RAW_CONFIG, 3)
        cfg = dataclasses.replace(tenv.EQVAR_RAW_CONFIG, standardize="rescale")
        rescaled = tenv.sample_instance(cfg, 3)
        np.testing.assert_array_equal(rescaled.truth.B, raw.truth.B)
        factors = rescaled.d_train.to_numpy()[0] / raw.d_train.to_numpy()[0]
        self.assertTrue(np.all((factors >= 0.1) & (factors <= 10.0)))
        for split in ("d_train", "d_test"):
            np.testing.assert_allclose(
                getattr(rescaled, split).to_numpy(),
                getattr(raw, split).to_numpy() * factors,
                rtol=1e-12,
            )
        # Equal variances identify the DAG only on the raw scale.
        self.assertEqual(raw.truth.target_class, tscore.TARGET_DAG)
        self.assertEqual(rescaled.truth.target_class, tscore.TARGET_CPDAG)


# #############################################################################
# Test_noise_share
# #############################################################################


_GRID = (0.2, 0.4, 0.6, 0.8)


def _noise_cfg(share: float, **kwargs) -> tenv.GraphConfig:
    """
    Build an iSCM configuration with an exact noise share on the test grid.
    """
    cfg = tenv.GraphConfig(
        family="iscm", noise_share=share, noise_grid=_GRID, **kwargs
    )
    return cfg


def _sample_r2(x: np.ndarray, B: np.ndarray) -> Dict[int, float]:
    """
    Compute each node's sample R2 on its true parents (with an intercept).

    :param x: data, one column per node
    :param B: DAG adjacency over the same columns
    :return: R2 by node, for the nodes with parents
    """
    r2 = {}
    for j in range(B.shape[0]):
        parents = np.flatnonzero(B[:, j])
        if len(parents) == 0:
            continue
        design = np.column_stack([np.ones(len(x)), x[:, parents]])
        coef, *_ = np.linalg.lstsq(design, x[:, j], rcond=None)
        resid = x[:, j] - design @ coef
        r2[j] = 1 - resid.var() / x[:, j].var()
    return r2


class Test_noise_share(hunitest.TestCase):
    """
    The opt-in iSCM noise share survives standardization and pairs the levels.
    """

    def test1(self) -> None:
        """
        Multiplying every noise by one factor only rescales the data.

        Standardization then cancels it, which is why the noise level is set
        as a share of each node's variance instead.
        """
        cfg = tenv.GraphConfig()
        W = np.array([[0.0, 1.5, 0.0], [0.0, 0.0, -0.8], [0.0, 0.0, 0.0]])
        families = ("exponential",) * 3
        noise_std = np.array([1.0, 0.7, 1.2])
        data = []
        for factor in (1.0, 5.0):
            x = tenv._simulate_linear(
                np.random.default_rng(0), W, families, factor * noise_std, 3000
            )
            x_train, _ = tenv._split_and_scale(x, cfg, np.random.default_rng(1))
            data.append(x_train)
        np.testing.assert_allclose(data[0], data[1], atol=1e-12)

    def test2(self) -> None:
        """
        In the population every node has unit variance and the exact share.
        """
        for seed in range(10):
            for share in _GRID:
                inst = tenv.sample_instance(_noise_cfg(share), seed)
                noise_var = inst.noise_std**2
                cov = tenv.population_covariance(inst.W, noise_var)
                np.testing.assert_allclose(np.diag(cov), 1.0, atol=1e-12)
                has_parents = inst.truth.B.any(axis=0)
                np.testing.assert_allclose(
                    noise_var[has_parents], share, atol=1e-12
                )
                np.testing.assert_allclose(noise_var[~has_parents], 1.0)
                self.assertGreaterEqual(
                    tenv.edge_strengths(inst.W, noise_var).min(), 0.1
                )

    def test3(self) -> None:
        """
        The levels of one seed share the graph, draws, split and permutation.
        """
        for seed in range(10):
            insts = [tenv.sample_instance(_noise_cfg(q), seed) for q in _GRID]
            first = insts[0]
            for inst in insts[1:]:
                np.testing.assert_array_equal(inst.truth.B, first.truth.B)
                self.assertEqual(inst.noise_families, first.noise_families)
                self.assertEqual(inst.n_rejected, first.n_rejected)
                self.assertEqual(
                    inst.truth.target_class, first.truth.target_class
                )
                # A node's incoming weights keep their ratios.
                for j in np.flatnonzero(first.truth.B.any(axis=0)):
                    ratio = inst.W[:, j] / np.where(
                        first.W[:, j] == 0, np.nan, first.W[:, j]
                    )
                    ratio = ratio[~np.isnan(ratio)]
                    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-10)

    def test4(self) -> None:
        """
        The historical iSCM of a seed has the same graph, family and target.
        """
        for seed in range(10):
            old = tenv.sample_instance(tenv.GraphConfig(family="iscm"), seed)
            new = tenv.sample_instance(_noise_cfg(0.4), seed)
            np.testing.assert_array_equal(new.truth.B, old.truth.B)
            self.assertEqual(new.noise_families, old.noise_families)
            self.assertEqual(new.truth.target_class, old.truth.target_class)

    def test5(self) -> None:
        """
        On a large sample the R2 on the true parents converges to 1 - share.
        """
        for share in _GRID:
            inst = tenv.sample_instance(_noise_cfg(share, n_samples=200_000), 0)
            r2 = _sample_r2(inst.d_train.to_numpy(), inst.truth.B)
            self.assertTrue(r2)
            for value in r2.values():
                self.assertAlmostEqual(value, 1 - share, delta=0.01)

    def test6(self) -> None:
        """
        Sample R2 is the same after `rescale` as after `post` standardization.
        """
        for share in (0.2, 0.8):
            post = tenv.sample_instance(_noise_cfg(share), 3)
            rescaled = tenv.sample_instance(
                _noise_cfg(share, standardize="rescale"), 3
            )
            r2_post = _sample_r2(post.d_train.to_numpy(), post.truth.B)
            r2_rescaled = _sample_r2(
                rescaled.d_train.to_numpy(), rescaled.truth.B
            )
            for j, value in r2_post.items():
                self.assertAlmostEqual(r2_rescaled[j], value, places=9)

    def test7(self) -> None:
        """
        Invalid settings are refused.
        """
        bad = [
            tenv.GraphConfig(noise_share=0.4, noise_grid=_GRID),
            tenv.GraphConfig(family="iscm", noise_share=0.5, noise_grid=_GRID),
            tenv.GraphConfig(family="iscm", noise_share=0.1, noise_grid=(0.1,)),
            tenv.GraphConfig(family="iscm", noise_grid=_GRID),
        ]
        for cfg in bad:
            with self.assertRaises(AssertionError, msg=str(cfg)):
                tenv.sample_instance(cfg, 0)


class Test_iscm_target(hunitest.TestCase):
    """
    Gaussian iSCM never targets the DAG: its noise is not equal after scaling.
    """

    def test1(self) -> None:
        """
        Equal raw noise on the raw scale identifies the DAG only without iSCM.
        """
        found = 0
        for seed in range(40):
            cfg = tenv.GraphConfig(
                family="iscm", noise_scale="equal", standardize="none"
            )
            inst = tenv.sample_instance(cfg, seed)
            if inst.noise_families[0] != "gaussian":
                continue
            found += 1
            self.assertEqual(inst.truth.target_class, tscore.TARGET_CPDAG)
            # The effective noise variances differ, as the target says.
            self.assertGreater(np.ptp(inst.noise_std), 1e-6)
            share = dataclasses.replace(
                cfg, noise_share=0.4, noise_grid=(0.2, 0.4)
            )
            self.assertEqual(
                tenv.sample_instance(share, seed).truth.target_class,
                tscore.TARGET_CPDAG,
            )
        self.assertGreater(found, 0)
        # The equal-variance linear family keeps its DAG target.
        raw = tenv.sample_instance(tenv.EQVAR_RAW_CONFIG, 0)
        self.assertEqual(raw.truth.target_class, tscore.TARGET_DAG)
