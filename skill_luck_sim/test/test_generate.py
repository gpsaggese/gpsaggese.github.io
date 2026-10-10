"""
Import as:

import skill_luck_sim.test.test_generate as slsteg
"""

import logging
import unittest

import numpy as np

import skill_luck_sim.generate as slsigene

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_compute_true_resolve_rate
# #############################################################################


class Test_compute_true_resolve_rate(unittest.TestCase):
    """
    Test the numerical integration of Ybar over the task population.
    """

    def test1(self) -> None:
        """
        Test that theta = mu_b gives exactly 0.5, by symmetry of the logit.
        """
        # Prepare inputs.
        params = slsigene.SimParams(mu_b=0.7, sigma_b=2.0, sigma_u=1.0)
        theta = np.array([0.7])
        # Run test.
        ybar = slsigene.compute_true_resolve_rate(theta, params)
        # Check outputs.
        np.testing.assert_allclose(ybar, [0.5], atol=1e-12)

    def test2(self) -> None:
        """
        Test that with no task spread Ybar is sigmoid(theta - mu_b).
        """
        # Prepare inputs.
        params = slsigene.SimParams(mu_b=0.5, sigma_b=0.0, sigma_u=0.0)
        theta = np.array([-1.0, 0.5, 2.0])
        # Run test.
        ybar = slsigene.compute_true_resolve_rate(theta, params)
        # Check outputs.
        expected = 1.0 / (1.0 + np.exp(-(theta - 0.5)))
        np.testing.assert_allclose(ybar, expected, atol=1e-12)

    def test3(self) -> None:
        """
        Test that the integral matches a large Monte Carlo average.
        """
        # Prepare inputs.
        params = slsigene.SimParams(sigma_b=2.0, sigma_u=1.0)
        theta = np.array([1.3])
        rng = np.random.default_rng(0)
        n_draws = 2_000_000
        # Run test.
        ybar = slsigene.compute_true_resolve_rate(theta, params)[0]
        b = rng.normal(params.mu_b, params.sigma_b, n_draws)
        u = rng.normal(0.0, params.sigma_u, n_draws)
        ybar_mc = slsigene.sigmoid(theta[0] - b + u).mean()
        # Check outputs. The Monte Carlo standard error is about 0.0002.
        self.assertAlmostEqual(ybar, ybar_mc, delta=0.001)


# #############################################################################
# Test_draw_world
# #############################################################################


class Test_draw_world(unittest.TestCase):
    """
    Test shapes and reproducibility of a drawn world.
    """

    def test1(self) -> None:
        """
        Test that the same seed gives identical output.
        """
        # Prepare inputs.
        params = slsigene.SimParams(n_configs=4, n_tasks=30, n_seeds=3)
        # Run test.
        world1 = slsigene.draw_world(params, np.random.default_rng(11))
        world2 = slsigene.draw_world(params, np.random.default_rng(11))
        # Check outputs.
        for field in ["theta", "ybar", "b", "u", "y"]:
            np.testing.assert_array_equal(
                getattr(world1, field), getattr(world2, field)
            )

    def test2(self) -> None:
        """
        Test that different seeds give different outcomes.
        """
        # Prepare inputs.
        params = slsigene.SimParams(n_configs=4, n_tasks=30, n_seeds=3)
        # Run test.
        world1 = slsigene.draw_world(params, np.random.default_rng(1))
        world2 = slsigene.draw_world(params, np.random.default_rng(2))
        # Check outputs.
        self.assertFalse(np.array_equal(world1.y, world2.y))

    def test3(self) -> None:
        """
        Test array shapes and that outcomes are 0 or 1.
        """
        # Prepare inputs.
        params = slsigene.SimParams(n_configs=4, n_tasks=30, n_seeds=3)
        # Run test.
        world = slsigene.draw_world(params, np.random.default_rng(0))
        # Check outputs.
        self.assertEqual(world.theta.shape, (4,))
        self.assertEqual(world.ybar.shape, (4,))
        self.assertEqual(world.b.shape, (30,))
        self.assertEqual(world.u.shape, (4, 30))
        self.assertEqual(world.y.shape, (4, 30, 3))
        self.assertTrue(set(np.unique(world.y)) <= {0, 1})


# #############################################################################
# Test_outcomes_to_frame
# #############################################################################


class Test_outcomes_to_frame(unittest.TestCase):
    """
    Test conversion of the outcome array to the generic long table.
    """

    def test1(self) -> None:
        """
        Test columns and that each row matches the array entry.
        """
        # Prepare inputs.
        y = np.random.default_rng(0).integers(0, 2, size=(2, 3, 4))
        # Run test.
        df = slsigene.outcomes_to_frame(y)
        # Check outputs.
        self.assertEqual(
            list(df.columns), ["config", "task", "seed", "resolved"]
        )
        self.assertEqual(len(df), 24)
        actual = df["resolved"].to_numpy()
        expected = y[df["config"], df["task"], df["seed"]]
        np.testing.assert_array_equal(actual, expected)
