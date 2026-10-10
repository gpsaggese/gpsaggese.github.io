"""
Import as:

import skill_luck_sim.test.test_bootstrap as slsteb
"""

import logging
import unittest

import numpy as np
import pandas as pd

import skill_luck_sim.bootstrap as slsiboot
import skill_luck_sim.generate as slsigene

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_bootstrap_scores
# #############################################################################


class Test_bootstrap_scores(unittest.TestCase):
    """
    Test the two resampling schemes on small inputs.
    """

    def test1(self) -> None:
        """
        Test that identical tasks with no seed variation give a constant.
        """
        # Prepare inputs.
        y = np.ones((10, 3), dtype=np.int8)
        rng = np.random.default_rng(0)
        # Run test and check outputs.
        for method in slsiboot.METHODS:
            replicates = slsiboot.bootstrap_scores(y, 50, method, rng)
            np.testing.assert_array_equal(replicates, np.ones(50))

    def test2(self) -> None:
        """
        Test that resampling tasks keeps the seeds of a task together.

        Each task is either always or never solved, so with the `task` scheme
        every replicate is a multiple of 1 / N.
        """
        # Prepare inputs.
        y = np.repeat(np.array([[1], [0], [1], [0]]), 5, axis=1)
        rng = np.random.default_rng(0)
        # Run test.
        replicates = slsiboot.bootstrap_scores(y, 200, "task", rng)
        # Check outputs.
        np.testing.assert_allclose(replicates * 4, np.round(replicates * 4))

    def test3(self) -> None:
        """
        Test that an unknown method raises.
        """
        # Prepare inputs.
        y = np.ones((4, 2), dtype=np.int8)
        rng = np.random.default_rng(0)
        # Run test and check outputs.
        with self.assertRaises(ValueError):
            slsiboot.bootstrap_scores(y, 10, "seed", rng)


# #############################################################################
# Test_run_coverage
# #############################################################################


class Test_run_coverage(unittest.TestCase):
    """
    Test interval coverage in a case with a known answer.
    """

    def helper(self) -> pd.DataFrame:
        """
        Run one config whose outcomes are all iid Bernoulli(0.5).
        """
        params = slsigene.SimParams(
            n_configs=1,
            n_tasks=100,
            n_seeds=2,
            mu_theta=0.0,
            sigma_theta=0.0,
            sigma_b=0.0,
            sigma_u=0.0,
        )
        df = slsiboot.run_coverage(params, 600, 400, seed=0)
        df = df.set_index("method")
        return df

    def test1(self) -> None:
        """
        Test that task resampling covers Ybar = 0.5 about 95% of the time.

        The score is a mean of 200 iid Bernoulli(0.5) draws, so the
        percentile interval is close to exact. The Monte Carlo standard
        error of the coverage is about 0.009.
        """
        # Run test.
        df = self.helper()
        # Check outputs.
        self.assertAlmostEqual(df.loc["task", "coverage"], 0.95, delta=0.03)

    def test2(self) -> None:
        """
        Test that two-stage resampling overcovers, as its variance predicts.

        With S = 2, resampling seeds inside resampled tasks adds a variance of
        (S - 1) / S * 0.25 / (N S) on top of the task-level 0.25 / (N S), so
        the interval is sqrt(1.5) = 1.22 times too wide and should cover
        P(|Z| < 1.96 * 1.22) = 0.984 of the time.
        """
        # Run test.
        df = self.helper()
        # Check outputs.
        self.assertGreater(df.loc["task_seed", "coverage"], 0.97)

    def test3(self) -> None:
        """
        Test that the same seed gives identical output.
        """
        # Prepare inputs.
        params = slsigene.SimParams(n_configs=3, n_tasks=20, n_seeds=2)
        # Run test.
        df1 = slsiboot.run_coverage(params, 5, 50, seed=3)
        df2 = slsiboot.run_coverage(params, 5, 50, seed=3)
        # Check outputs.
        pd.testing.assert_frame_equal(df1, df2)
