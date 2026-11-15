"""
Test the Tukey toolbox against reference implementations.

Partial correlations are checked against residual regressions, Fisher-z
p-values against causal-learn, regressions against scipy and scikit-learn, and
the pairwise direction statistic against the true direction and DirectLiNGAM.

Import as:

import test.test_tukey_tools as ttetutoo
"""

import logging
from typing import Dict, List, Tuple

import causallearn.utils.cit as cucit
import lingam
import numpy as np
import scipy.stats
import sklearn.linear_model as sklimode

import helpers.hunit_test as hunitest
import tukey_tools as ttools

_LOG = logging.getLogger(__name__)


# #############################################################################
# Fixtures
# #############################################################################


def _simulate(
    *, seed: int, n_samples: int = 3000, noise: str = "exponential"
) -> np.ndarray:
    """
    Sample a 4-variable linear SEM with unequal raw scales and offsets.

    The graph is 0 -> 1 -> 2 -> 3 plus 0 -> 3.

    :param seed: random seed
    :param n_samples: number of rows
    :param noise: "exponential" (centred) or "gaussian"
    :return: data, shape (n_samples, 4)
    """
    rng = np.random.default_rng(seed)
    if noise == "exponential":
        e = rng.exponential(size=(n_samples, 4)) - 1.0
    else:
        e = rng.normal(size=(n_samples, 4))
    x = np.zeros((n_samples, 4))
    x[:, 0] = e[:, 0]
    x[:, 1] = 0.9 * x[:, 0] + e[:, 1]
    x[:, 2] = -1.2 * x[:, 1] + e[:, 2]
    x[:, 3] = 0.7 * x[:, 2] + 0.6 * x[:, 0] + e[:, 3]
    x = x * np.array([1.0, 3.0, 0.2, 10.0]) + np.array([5.0, -2.0, 0.0, 100.0])
    return x


def _simulate_pair(*, seed: int, noise: str) -> Tuple[np.ndarray, int]:
    """
    Sample a two-variable linear SEM as the linear generators do.

    Weight +/-U(0.5, 2), noise scales U(0.5, 1.5), 2,000 rows; the cause is
    put in a random column.

    :param seed: random seed
    :param noise: "exponential" (centred) or "gaussian"
    :return: data and the column index of the cause
    """
    rng = np.random.default_rng(seed)
    n_samples = 2000
    weight = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 2.0)
    scales = rng.uniform(0.5, 1.5, size=2)
    if noise == "exponential":
        e = rng.exponential(size=(n_samples, 2)) - 1.0
    else:
        e = rng.normal(size=(n_samples, 2))
    cause = scales[0] * e[:, 0]
    effect = weight * cause + scales[1] * e[:, 1]
    cause_col = int(rng.integers(2))
    if cause_col == 0:
        x = np.column_stack([cause, effect])
    else:
        x = np.column_stack([effect, cause])
    return x, cause_col


def _standardize(x: np.ndarray) -> np.ndarray:
    """
    Standardize columns with the population standard deviation.

    :param x: data
    :return: standardized data
    """
    z = (x - x.mean(axis=0)) / x.std(axis=0)
    return z


def _all_results(x: np.ndarray) -> Dict[ttools.Action, Dict[str, float]]:
    """
    Run every tool call on a dataset.

    :param x: data
    :return: result of every action
    """
    cache = ttools.ToolCache(x)
    actions = ttools.enumerate_tool_actions(x.shape[1])
    results = {action: cache.run(action) for action in actions}
    return results


# #############################################################################
# Test_enumerate_tool_actions
# #############################################################################


class Test_enumerate_tool_actions(hunitest.TestCase):
    """
    Count and canonicalize the tool calls.
    """

    def test1(self) -> None:
        """
        Counts per tool for 3 and 6 variables, with no duplicates.
        """
        lines: List[str] = []
        for n_nodes in (3, 6):
            actions = ttools.enumerate_tool_actions(n_nodes)
            counts = {
                tool: sum(a.kind == tool for a in actions)
                for tool in ttools.TOOLSETS["m3"]
            }
            lines.append(f"{n_nodes}: {len(actions)} {counts}")
            self.assertEqual(len(set(actions)), len(actions))
        actual = "\n".join(lines)
        expected = (
            "3: 21 {'ci_test': 3, 'regress': 9, 'lingam_dir': 3, "
            "'nongauss': 3, 'variance': 3}\n"
            "6: 267 {'ci_test': 150, 'regress': 90, 'lingam_dir': 15, "
            "'nongauss': 6, 'variance': 6}"
        )
        self.assert_equal(actual, expected)

    def test2(self) -> None:
        """
        Equal calls written in different orders are the same action.
        """
        self.assertEqual(
            ttools.ci_test_action(2, 0, (3, 1)),
            ttools.ci_test_action(0, 2, (1, 3)),
        )
        self.assertEqual(
            ttools.regress_action(1, (3, 0)), ttools.regress_action(1, (0, 3))
        )


# #############################################################################
# Test_ci_test
# #############################################################################


class Test_ci_test(hunitest.TestCase):
    """
    Check partial correlations and Fisher-z p-values.
    """

    def test1(self) -> None:
        """
        Partial correlations equal the correlation of regression residuals.
        """
        x = _simulate(seed=0)
        z = _standardize(x)
        cache = ttools.ToolCache(x)
        errors = []
        for action in ttools.enumerate_tool_actions(4):
            if action.kind != ttools.CI_TEST:
                continue
            i, j, cond = action.args[0], action.args[1], list(action.args[2:])
            z_cond = z[:, cond]
            beta_i = np.linalg.lstsq(z_cond, z[:, i], rcond=None)[0]
            beta_j = np.linalg.lstsq(z_cond, z[:, j], rcond=None)[0]
            resid_i = z[:, i] - z_cond @ beta_i
            resid_j = z[:, j] - z_cond @ beta_j
            expected = np.corrcoef(resid_i, resid_j)[0, 1]
            errors.append(abs(cache.run(action)["pcorr"] - expected))
        self.assertEqual(len(errors), 18)
        self.assertLess(max(errors), 1e-10)

    def test2(self) -> None:
        """
        Fisher-z p-values equal causal-learn's `fisherz` test.
        """
        x = _simulate(seed=1)
        cache = ttools.ToolCache(x)
        cit = cucit.CIT(x, "fisherz")
        actual = []
        expected = []
        for action in ttools.enumerate_tool_actions(4):
            if action.kind != ttools.CI_TEST:
                continue
            i, j, cond = action.args[0], action.args[1], list(action.args[2:])
            actual.append(cache.run(action)["p"])
            expected.append(cit(i, j, cond))
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-12)

    def test3(self) -> None:
        """
        A true conditional independence is not rejected; a true edge is.
        """
        x = _simulate(seed=2)
        cache = ttools.ToolCache(x)
        # x1 and x3 are separated by {x0, x2}; x2 -> x3 is an edge.
        p_independent = cache.run(ttools.ci_test_action(1, 3, (0, 2)))["p"]
        p_edge = cache.run(ttools.ci_test_action(2, 3, (0, 1)))["p"]
        self.assertGreater(p_independent, 0.01)
        self.assertLess(p_edge, 1e-10)


# #############################################################################
# Test_regress
# #############################################################################


class Test_regress(hunitest.TestCase):
    """
    Check standardized regressions against scipy and scikit-learn.
    """

    def test1(self) -> None:
        """
        One predictor: slope, t-statistic and p-value equal `linregress`.
        """
        x = _simulate(seed=3)
        z = _standardize(x)
        cache = ttools.ToolCache(x)
        result = cache.run(ttools.regress_action(2, (1,)))
        ref = scipy.stats.linregress(z[:, 1], z[:, 2])
        actual = [result["coef_1"], result["t_1"], result["p_1"], result["r2"]]
        expected = [ref.slope, ref.slope / ref.stderr, ref.pvalue, ref.rvalue**2]
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-300)

    def test2(self) -> None:
        """
        Two predictors: coefficients and R2 equal scikit-learn; t equals OLS.
        """
        x = _simulate(seed=4)
        z = _standardize(x)
        cache = ttools.ToolCache(x)
        result = cache.run(ttools.regress_action(3, (0, 2)))
        model = sklimode.LinearRegression().fit(z[:, [0, 2]], z[:, 3])
        # Classical OLS standard errors with an intercept column.
        design = np.column_stack([np.ones(len(z)), z[:, [0, 2]]])
        resid = z[:, 3] - model.predict(z[:, [0, 2]])
        sigma2 = resid @ resid / (len(z) - 3)
        cov = sigma2 * np.linalg.inv(design.T @ design)
        t_ref = model.coef_ / np.sqrt(np.diag(cov)[1:])
        actual = [
            result["coef_0"],
            result["coef_2"],
            result["r2"],
            result["t_0"],
            result["t_2"],
        ]
        expected = [
            model.coef_[0],
            model.coef_[1],
            model.score(z[:, [0, 2]], z[:, 3]),
            t_ref[0],
            t_ref[1],
        ]
        np.testing.assert_allclose(actual, expected, rtol=1e-9)

    def test3(self) -> None:
        """
        `resid_dep` equals the prototype's direction statistic on raw data.
        """
        x = _simulate(seed=5)
        cache = ttools.ToolCache(x)
        for cause, effect in [(0, 1), (1, 0), (2, 3)]:
            model = sklimode.LinearRegression().fit(x[:, [cause]], x[:, effect])
            resid = x[:, effect] - model.predict(x[:, [cause]])
            expected = abs(np.corrcoef(x[:, cause], np.abs(resid))[0, 1])
            result = cache.run(ttools.regress_action(effect, (cause,)))
            self.assertAlmostEqual(result["resid_dep"], expected, places=10)


# #############################################################################
# Test_lingam_dir
# #############################################################################


class Test_lingam_dir(hunitest.TestCase):
    """
    Check the pairwise likelihood-ratio direction statistic.
    """

    def test1(self) -> None:
        """
        The sign finds the cause in >= 95% of 200 exponential-noise pairs.
        """
        n_correct = 0
        for seed in range(200):
            x, cause_col = _simulate_pair(seed=seed, noise="exponential")
            action = ttools.Action(ttools.LINGAM_DIR, (0, 1))
            stat = ttools.ToolCache(x).run(action)["stat"]
            predicted = 0 if stat > 0 else 1
            n_correct += int(predicted == cause_col)
        _LOG.info("lingam_dir: %s/200 correct", n_correct)
        self.assertGreaterEqual(n_correct, 190)

    def test2(self) -> None:
        """
        The sign agrees with DirectLiNGAM's causal order on all 200 pairs.
        """
        n_agree = 0
        for seed in range(200):
            x, _ = _simulate_pair(seed=seed, noise="exponential")
            action = ttools.Action(ttools.LINGAM_DIR, (0, 1))
            stat = ttools.ToolCache(x).run(action)["stat"]
            model = lingam.DirectLiNGAM()
            model.fit(x)
            first = 0 if stat > 0 else 1
            n_agree += int(first == model.causal_order_[0])
        self.assertEqual(n_agree, 200)

    def test3(self) -> None:
        """
        Under Gaussian noise the sign is at chance (no direction information).
        """
        n_correct = 0
        for seed in range(200):
            x, cause_col = _simulate_pair(seed=1000 + seed, noise="gaussian")
            action = ttools.Action(ttools.LINGAM_DIR, (0, 1))
            stat = ttools.ToolCache(x).run(action)["stat"]
            n_correct += int((0 if stat > 0 else 1) == cause_col)
        _LOG.info("lingam_dir on Gaussian pairs: %s/200 correct", n_correct)
        self.assertGreaterEqual(n_correct, 70)
        self.assertLessEqual(n_correct, 130)


# #############################################################################
# Test_nongauss_and_variance
# #############################################################################


class Test_nongauss_and_variance(hunitest.TestCase):
    """
    Check the marginal-shape tool and the variance tool.
    """

    def test1(self) -> None:
        """
        Exponential columns are skewed and heavy-tailed; Gaussian ones are not.
        """
        rng = np.random.default_rng(6)
        x = np.column_stack(
            [rng.exponential(size=20000), rng.normal(size=20000)]
        )
        cache = ttools.ToolCache(x)
        expo = cache.run(ttools.Action(ttools.NONGAUSS, (0,)))
        gauss = cache.run(ttools.Action(ttools.NONGAUSS, (1,)))
        # Population values: skew 2 and excess kurtosis 6 for exponential.
        self.assertLess(abs(expo["skew"] - 2.0), 0.15)
        self.assertGreater(expo["kurt"], 4.0)
        self.assertLess(expo["jb_p"], 1e-10)
        self.assertLess(abs(gauss["skew"]), 0.1)
        self.assertLess(abs(gauss["kurt"]), 0.2)
        self.assertGreater(gauss["jb_p"], 0.001)

    def test2(self) -> None:
        """
        The log variance is 0 for a standardized column and tracks rescaling.
        """
        x = _standardize(_simulate(seed=7))
        x[:, 1] = 10.0 * x[:, 1]
        cache = ttools.ToolCache(x)
        log_var_0 = cache.run(ttools.Action(ttools.VARIANCE, (0,)))["log_var"]
        log_var_1 = cache.run(ttools.Action(ttools.VARIANCE, (1,)))["log_var"]
        self.assertLess(abs(log_var_0), 1e-12)
        self.assertLess(abs(log_var_1 - np.log(100.0)), 1e-12)


# #############################################################################
# Test_scale_invariance
# #############################################################################


class Test_scale_invariance(hunitest.TestCase):
    """
    Only `variance` may see the raw scale of a column.
    """

    def test1(self) -> None:
        """
        Rescaling and shifting columns changes no result except `variance`.
        """
        x = _simulate(seed=8)
        rng = np.random.default_rng(8)
        scales = np.exp(rng.uniform(np.log(0.1), np.log(10.0), size=4))
        offsets = rng.normal(scale=50.0, size=4)
        original = _all_results(x)
        rescaled = _all_results(x * scales + offsets)
        n_compared = 0
        for action, result in original.items():
            if action.kind == ttools.VARIANCE:
                expected = result["log_var"] + 2 * np.log(scales[action.args[0]])
                actual = rescaled[action]["log_var"]
                self.assertAlmostEqual(actual, expected, places=9)
                continue
            for key, value in result.items():
                np.testing.assert_allclose(
                    rescaled[action][key], value, rtol=1e-7, atol=1e-9
                )
                n_compared += 1
        self.assertEqual(n_compared, 240)


# #############################################################################
# TestToolCache
# #############################################################################


class TestToolCache(hunitest.TestCase):
    """
    Test the data checks of `tukey_tools.ToolCache`.
    """

    def test1(self) -> None:
        """
        Test that 5 training rows are refused and 6 give a CI test p-value.

        Given 2 variables, Fisher z has `n - 2 - 3` degrees of freedom: 0 with
        5 rows (where it used to report p = 1) and 1 with 6 rows.
        """
        # Prepare inputs.
        rng = np.random.default_rng(0)
        x_short = rng.normal(size=(5, 4))
        x_enough = rng.normal(size=(6, 4))
        action = ttools.ci_test_action(0, 1, (2, 3))
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            ttools.ToolCache(x_short)
        p_value = ttools.ToolCache(x_enough).run(action)["p"]
        self.assertTrue(0.0 < p_value < 1.0)

    def test2(self) -> None:
        """
        Test that duplicated or perfectly correlated columns are refused.
        """
        # Prepare inputs.
        steps = np.arange(10.0)
        cases = {
            "duplicated column": np.column_stack([steps, steps, steps**2]),
            "correlated pair": np.tile([[-1.0, -1.0], [1.0, 1.0]], (10, 1)),
        }
        # Run test and check outputs.
        for name, x in cases.items():
            with self.assertRaises(AssertionError, msg=name):
                ttools.ToolCache(x)

    def test3(self) -> None:
        """
        Test that a single variable is refused.
        """
        # Prepare inputs.
        x = np.random.default_rng(1).normal(size=(100, 1))
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            ttools.ToolCache(x)


# #############################################################################
# TestEvidence
# #############################################################################


class TestEvidence(hunitest.TestCase):
    """
    Check the evidence log: budget, repeats and legal actions.
    """

    def test1(self) -> None:
        """
        Recording updates the history and budget; repeats are refused.
        """
        cache = ttools.ToolCache(_simulate(seed=9))
        evidence = ttools.Evidence(cache, budget=2)
        action = ttools.ci_test_action(0, 1, (2,))
        result = evidence.record(action)
        self.assertEqual(result, cache.run(action))
        self.assertEqual(evidence.calls_used, 1)
        self.assertEqual(evidence.budget_left, 1)
        with self.assertRaises(AssertionError):
            evidence.record(action)
        evidence.record(ttools.Action(ttools.NONGAUSS, (3,)))
        with self.assertRaises(AssertionError):
            evidence.record(ttools.Action(ttools.NONGAUSS, (2,)))
        actual = str(evidence.history)
        expected = (
            "[Action(kind='ci_test', args=(0, 1, 2)), "
            "Action(kind='nongauss', args=(3,))]"
        )
        self.assert_equal(actual, expected)

    def test2(self) -> None:
        """
        Legal actions exclude calls already made and vanish at budget 0.
        """
        cache = ttools.ToolCache(_simulate(seed=10))
        all_actions = ttools.enumerate_tool_actions(4)
        evidence = ttools.Evidence(cache, budget=1)
        self.assertEqual(len(evidence.legal_tool_actions(all_actions)), 56)
        evidence.record(all_actions[0])
        self.assertEqual(evidence.legal_tool_actions(all_actions), [])
        evidence = ttools.Evidence(cache, budget=5)
        evidence.record(all_actions[0])
        legal = evidence.legal_tool_actions(all_actions)
        self.assertEqual(len(legal), 55)
        self.assertNotIn(all_actions[0], legal)

    def test3(self) -> None:
        """
        Non-canonical and out-of-range calls are refused and cost nothing.
        """
        cache = ttools.ToolCache(_simulate(seed=9))
        evidence = ttools.Evidence(cache, budget=5)
        evidence.record(ttools.ci_test_action(0, 1, (2,)))
        refused = [
            # The same test with the pair reversed.
            ttools.Action(ttools.CI_TEST, (1, 0, 2)),
            # A conditioning variable equal to a tested one.
            ttools.Action(ttools.CI_TEST, (0, 1, 1)),
            ttools.Action(ttools.REGRESS, (0, 0)),
            ttools.Action(ttools.LINGAM_DIR, (2, 2)),
            # A variable that does not exist.
            ttools.Action(ttools.NONGAUSS, (cache.n_nodes,)),
        ]
        for action in refused:
            with self.assertRaises(AssertionError, msg=str(action)):
                evidence.record(action)
        self.assertEqual(evidence.calls_used, 1)


class Test_mutual_info(hunitest.TestCase):
    """
    Check the mutual-information tool on dependence it should and should not see.
    """

    @staticmethod
    def _result(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Run the tool on two variables.

        :param x: first variable
        :param y: second variable
        :return: tool result
        """
        cache = ttools.ToolCache(np.column_stack([x, y]))
        result = cache.run(ttools.Action(ttools.MUTUAL_INFO, (0, 1)))
        return result

    def test1(self) -> None:
        """
        Independent variables carry no information.
        """
        rng = np.random.default_rng(0)
        result = self._result(rng.normal(size=2000), rng.normal(size=2000))
        self.assertLess(result["mi"], 0.02)
        self.assertLess(abs(result["mi_excess"]), 0.02)

    def test2(self) -> None:
        """
        Y = x^2 + noise is uncorrelated with x, yet clearly dependent.
        """
        rng = np.random.default_rng(1)
        x = rng.normal(size=2000)
        y = x**2 + 0.3 * rng.normal(size=2000)
        result = self._result(x, y)
        self.assertLess(abs(np.corrcoef(x, y)[0, 1]), 0.1)
        self.assertGreater(result["mi"], 0.3)
        self.assertGreater(result["mi_excess"], 0.3)

    def test3(self) -> None:
        """
        Linear-Gaussian dependence: close to the Gaussian value, no excess.
        """
        rng = np.random.default_rng(2)
        x = rng.normal(size=2000)
        y = 0.7 * x + np.sqrt(0.51) * rng.normal(size=2000)
        result = self._result(x, y)
        self.assertGreater(result["mi"], 0.2)
        self.assertLess(abs(result["mi_excess"]), 0.05)

    def test4(self) -> None:
        """
        The Milestone 4 tool set appends one call per pair to the M3 calls.
        """
        m3 = ttools.enumerate_tool_actions(6)
        m4 = ttools.enumerate_tool_actions(6, toolset="m4")
        self.assertEqual(m4[: len(m3)], m3)
        self.assertEqual(len(m4) - len(m3), 15)
        self.assertEqual({a.kind for a in m4[len(m3) :]}, {ttools.MUTUAL_INFO})

    def test5(self) -> None:
        """
        A Milestone 3 evidence log refuses a mutual-information call.
        """
        rng = np.random.default_rng(3)
        evidence = ttools.Evidence(
            ttools.ToolCache(rng.normal(size=(500, 3))), budget=5
        )
        with self.assertRaises(AssertionError):
            evidence.record(ttools.Action(ttools.MUTUAL_INFO, (0, 1)))
