"""
Import as:

import skill_luck_sim.test.test_estimate as slstee
"""

import logging
import math
import unittest

import numpy as np
import pandas as pd
import pytest

import skill_luck_sim.estimate as slsiesti
import skill_luck_sim.generate as slsigene

_LOG = logging.getLogger(__name__)


def _expected_max_binomial_mean(n_configs: int, n_trials: int) -> float:
    """
    Compute E[max_k X_k / n] for `n_configs` iid Binomial(n, 0.5) variables.

    Uses E[max] = sum_x x (F(x)^K - F(x - 1)^K), with F the binomial CDF.
    """
    pmf = np.array([math.comb(n_trials, x) for x in range(n_trials + 1)]) / (
        2.0**n_trials
    )
    cdf = np.cumsum(pmf)
    cdf_prev = np.concatenate([[0.0], cdf[:-1]])
    x = np.arange(n_trials + 1)
    expected = (
        float((x * (cdf**n_configs - cdf_prev**n_configs)).sum()) / n_trials
    )
    return expected


# #############################################################################
# Test_compute_ranks
# #############################################################################


class Test_compute_ranks(unittest.TestCase):
    """
    Test ranking from best to worst with ties.
    """

    def test1(self) -> None:
        """
        Test the docstring example with a tie.
        """
        # Prepare inputs.
        values = np.array([0.2, 0.5, 0.5])
        # Run test.
        actual = slsiesti.compute_ranks(values)
        # Check outputs.
        np.testing.assert_array_equal(actual, [3.0, 1.5, 1.5])


# #############################################################################
# Test_winners_curse
# #############################################################################


class Test_winners_curse(unittest.TestCase):
    """
    Test the size of the winner's curse in cases with a known answer.
    """

    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test that with all configs equal the curse matches order statistics.

        With sigma_theta = sigma_b = sigma_u = 0 and theta = mu_b, every
        outcome is Bernoulli(0.5), every Ybar is 0.5, and each score is
        Binomial(N S, 0.5) / (N S), independent across configs. So the
        expected curse is E[max of K such scores] - 0.5.
        """
        # Prepare inputs.
        params = slsigene.SimParams(
            n_configs=5,
            n_tasks=50,
            n_seeds=2,
            mu_theta=0.0,
            sigma_theta=0.0,
            sigma_b=0.0,
            sigma_u=0.0,
        )
        n_reps = 4000
        # Prepare outputs. For K = 5 the normal approximation gives
        # 1.163 * 0.5 / sqrt(100) = 0.058.
        expected = _expected_max_binomial_mean(5, 100) - 0.5
        # Run test.
        df = slsiesti.run_replications(params, n_reps, seed=0)
        summary = slsiesti.summarize_replications(df, params)
        # Check outputs.
        self.assertGreater(summary["curse"], 0.0)
        self.assertAlmostEqual(
            summary["curse"], expected, delta=3 * summary["curse_se"]
        )
        self.assertTrue(np.isnan(summary["is_truly_best"]))

    def test2(self) -> None:
        """
        Test that the curse is positive when configs are equal but tasks vary.
        """
        # Prepare inputs.
        params = slsigene.SimParams(
            n_configs=10, n_tasks=25, n_seeds=1, sigma_theta=0.0
        )
        # Run test.
        df = slsiesti.run_replications(params, 1000, seed=0)
        summary = slsiesti.summarize_replications(df, params)
        # Check outputs.
        self.assertGreater(summary["curse"], 5 * summary["curse_se"])

    def test3(self) -> None:
        """
        Test that the curse goes toward zero with many tasks and seeds.
        """
        # Prepare inputs.
        small = slsigene.SimParams(n_configs=10, n_tasks=25, n_seeds=1)
        large = slsigene.SimParams(n_configs=10, n_tasks=2000, n_seeds=20)
        # Run test.
        curse_small = slsiesti.summarize_replications(
            slsiesti.run_replications(small, 300, seed=0), small
        )["curse"]
        curse_large = slsiesti.summarize_replications(
            slsiesti.run_replications(large, 100, seed=0), large
        )["curse"]
        # Check outputs.
        self.assertLess(abs(curse_large), 0.005)
        self.assertLess(abs(curse_large), curse_small / 10)


# #############################################################################
# Test_reevaluation
# #############################################################################


@pytest.mark.slow
class Test_reevaluation(unittest.TestCase):
    """
    Test the bias of re-evaluating the winner on fresh seeds or tasks.
    """

    def helper(self) -> dict:
        """
        Run a setting with a large curse: many configs, few tasks.
        """
        params = slsigene.SimParams(n_configs=20, n_tasks=25, n_seeds=3)
        df = slsiesti.run_replications(params, 3000, seed=1)
        summary = slsiesti.summarize_replications(df, params)
        return summary

    def test1(self) -> None:
        """
        Test that fresh tasks give an unbiased estimate of the winner's Ybar.
        """
        # Run test.
        summary = self.helper()
        # Check outputs.
        self.assertGreater(summary["curse"], 10 * summary["curse_se"])
        self.assertLess(
            abs(summary["fresh_task_error"]), 3 * summary["fresh_task_error_se"]
        )

    def test2(self) -> None:
        """
        Test that fresh seeds on the same tasks remove only part of the curse.

        The winner was also selected for its luck on these tasks (through
        u_ki), and rerunning seeds keeps that luck.
        """
        # Run test.
        summary = self.helper()
        # Check outputs.
        self.assertGreater(
            summary["fresh_seed_error"], 5 * summary["fresh_seed_error_se"]
        )
        self.assertLess(summary["fresh_seed_error"], summary["curse"])


# #############################################################################
# Test_run_replications
# #############################################################################


class Test_run_replications(unittest.TestCase):
    """
    Test reproducibility of a batch of replications.
    """

    def test1(self) -> None:
        """
        Test that the same seed gives identical output.
        """
        # Prepare inputs.
        params = slsigene.SimParams(n_configs=5, n_tasks=20, n_seeds=2)
        # Run test.
        df1 = slsiesti.run_replications(params, 20, seed=5)
        df2 = slsiesti.run_replications(params, 20, seed=5)
        # Check outputs.
        pd.testing.assert_frame_equal(df1, df2)
