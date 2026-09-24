"""
Tests for racket_trajectory module.
"""

import logging

import numpy as np

import helpers.hunit_test as hunitest
from research.Optimal_strategy_for_racket_sports import racket_params
from research.Optimal_strategy_for_racket_sports import racket_trajectory

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_solve_launch_speed
# #############################################################################


class Test_solve_launch_speed(hunitest.TestCase):
    """
    Test `racket_trajectory.solve_launch_speed()`.
    """

    def test1(self) -> None:
        """
        Test the Figure 1 launch speed: theta=8 deg, d_target=20 m, h0=1 m.
        """
        # Prepare inputs.
        d_target = np.array([20.0])
        theta = np.array([np.radians(8.0)])
        h0 = 1.0
        # Prepare outputs.
        expected = 22.9133
        # Run test.
        actual = racket_trajectory.solve_launch_speed(d_target, theta, h0)
        # Check outputs.
        self.assertAlmostEqual(actual[0], expected, places=3)

    def test2(self) -> None:
        """
        Test that an angle unable to reach `d_target` from `h0` returns NaN.
        """
        # Prepare inputs.
        d_target = np.array([20.0])
        # A steep negative angle drives `h0 + d_target * tan(theta)` negative.
        theta = np.array([np.radians(-60.0)])
        h0 = 1.0
        # Run test.
        actual = racket_trajectory.solve_launch_speed(d_target, theta, h0)
        # Check outputs.
        self.assertTrue(np.isnan(actual[0]))


# #############################################################################
# Test_get_flight_time
# #############################################################################


class Test_get_flight_time(hunitest.TestCase):
    """
    Test `racket_trajectory.get_flight_time()`.
    """

    def test1(self) -> None:
        """
        Test that `h0 = 0` gives the textbook `2 * v0 * sin(theta) / g`.
        """
        # Prepare inputs.
        v0 = np.array([20.0])
        theta = np.array([np.radians(30.0)])
        h0 = 0.0
        # Prepare outputs.
        expected = 2 * v0[0] * np.sin(theta[0]) / racket_params.GRAVITY_MPS2
        # Run test.
        actual = racket_trajectory.get_flight_time(v0, theta, h0)
        # Check outputs.
        self.assertAlmostEqual(actual[0], expected)

    def test2(self) -> None:
        """
        Test the Figure 1 flight time: v0=22.9133, theta=8 deg, h0=1 m.
        """
        # Prepare inputs.
        v0 = np.array([22.9133])
        theta = np.array([np.radians(8.0)])
        h0 = 1.0
        # Prepare outputs.
        expected = 0.8814
        # Run test.
        actual = racket_trajectory.get_flight_time(v0, theta, h0)
        # Check outputs.
        self.assertAlmostEqual(actual[0], expected, places=3)


# #############################################################################
# Test_get_height_at
# #############################################################################


class Test_get_height_at(hunitest.TestCase):
    """
    Test `racket_trajectory.get_height_at()`.
    """

    def test1(self) -> None:
        """
        Test the Figure 1 net clearance: 0.40 m at 12 m from the striker.
        """
        # Prepare inputs.
        x = np.array([12.0])
        v0 = np.array([22.9133])
        theta = np.array([np.radians(8.0)])
        h0 = 1.0
        # Prepare outputs.
        expected_clearance = 0.40
        # Run test.
        height = racket_trajectory.get_height_at(x, v0, theta, h0)
        actual_clearance = (
            height[0] - racket_params.TENNIS.court.net_height_center_m
        )
        # Check outputs.
        self.assertAlmostEqual(actual_clearance, expected_clearance, places=2)


# #############################################################################
# Test_get_feasible_launches
# #############################################################################


class Test_get_feasible_launches(hunitest.TestCase):
    """
    Test `racket_trajectory.get_feasible_launches()`.
    """

    def test1(self) -> None:
        """
        Test that every feasible launch meets the max ball speed and clears
        the net.
        """
        # Prepare inputs.
        striker_xy = (0.0, -11.0)
        target_xy = (0.0, 5.0)
        h0 = 1.0
        sport = racket_params.TENNIS
        theta_grid_rad = np.radians(np.arange(1.0, 45.0, 1.0))
        # Run test.
        actual = racket_trajectory.get_feasible_launches(
            striker_xy,
            target_xy,
            h0,
            sport,
            theta_grid_rad=theta_grid_rad,
        )
        # Check outputs.
        self.assertGreater(len(actual.theta_rad), 0)
        self.assertTrue(np.all(actual.v0_mps <= sport.max_ball_speed_mps))
        self.assertTrue(np.all(actual.net_clearance_m > 0))

    def test2(self) -> None:
        """
        Test that a tiny `max_ball_speed_mps` yields no feasible launches.
        """
        # Prepare inputs.
        striker_xy = (0.0, -11.0)
        target_xy = (0.0, 5.0)
        h0 = 1.0
        sport = racket_params.SportParams(
            name="Tiny",
            court=racket_params.TENNIS.court,
            max_ball_speed_mps=0.5,
        )
        theta_grid_rad = np.radians(np.arange(1.0, 45.0, 1.0))
        # Run test.
        actual = racket_trajectory.get_feasible_launches(
            striker_xy,
            target_xy,
            h0,
            sport,
            theta_grid_rad=theta_grid_rad,
        )
        # Check outputs.
        self.assertEqual(len(actual.theta_rad), 0)


# #############################################################################
# Test_sample_shot_errors
# #############################################################################


class Test_sample_shot_errors(hunitest.TestCase):
    """
    Test `racket_trajectory.sample_shot_errors()`.
    """

    def test1(self) -> None:
        """
        Test that the same seed gives the same draws.
        """
        # Prepare inputs.
        error = racket_params.DEFAULT_ERROR
        n_samples = 10
        # Run test.
        actual1 = racket_trajectory.sample_shot_errors(
            error, n_samples, np.random.default_rng(42)
        )
        actual2 = racket_trajectory.sample_shot_errors(
            error, n_samples, np.random.default_rng(42)
        )
        # Check outputs.
        np.testing.assert_array_equal(actual1.d_theta_rad, actual2.d_theta_rad)
        np.testing.assert_array_equal(actual1.d_v_frac, actual2.d_v_frac)
        np.testing.assert_array_equal(actual1.phi_rad, actual2.phi_rad)

    def test2(self) -> None:
        """
        Test that zero sigmas give zero-valued draws.
        """
        # Prepare inputs.
        error = racket_params.ShotErrorModel(
            sigma_theta_rad=0.0, sigma_v_frac=0.0, sigma_phi_rad=0.0
        )
        n_samples = 5
        # Run test.
        actual = racket_trajectory.sample_shot_errors(
            error, n_samples, np.random.default_rng(0)
        )
        # Check outputs.
        np.testing.assert_array_equal(actual.d_theta_rad, np.zeros(n_samples))
        np.testing.assert_array_equal(actual.d_v_frac, np.zeros(n_samples))
        np.testing.assert_array_equal(actual.phi_rad, np.zeros(n_samples))


# #############################################################################
# Test_simulate_landings
# #############################################################################


class Test_simulate_landings(hunitest.TestCase):
    """
    Test `racket_trajectory.simulate_landings()`.
    """

    def test1(self) -> None:
        """
        Test that zero-error samples land exactly on the nominal target.
        """
        # Prepare inputs.
        striker_xy = (0.0, -11.0)
        target_xy = (0.0, 5.0)
        h0 = 1.0
        sport = racket_params.TENNIS
        theta_grid_rad = np.radians(np.arange(1.0, 45.0, 1.0))
        launches = racket_trajectory.get_feasible_launches(
            striker_xy,
            target_xy,
            h0,
            sport,
            theta_grid_rad=theta_grid_rad,
        )
        n_launches = len(launches.theta_rad)
        errors = racket_trajectory.ShotErrors(
            d_theta_rad=np.zeros(1), d_v_frac=np.zeros(1), phi_rad=np.zeros(1)
        )
        # Run test.
        actual = racket_trajectory.simulate_landings(
            striker_xy, target_xy, h0, sport, launches, errors
        )
        # Check outputs.
        self.assertEqual(actual.x_m.shape, (n_launches, 1))
        np.testing.assert_allclose(
            actual.x_m[:, 0], np.full(n_launches, target_xy[0]), atol=1e-6
        )
        np.testing.assert_allclose(
            actual.y_m[:, 0], np.full(n_launches, target_xy[1]), atol=1e-6
        )

    def test2(self) -> None:
        """
        Test that equal-and-opposite lateral errors land symmetrically about
        the straight striker-to-target line.
        """
        # Prepare inputs.
        striker_xy = (0.0, -11.0)
        target_xy = (0.0, 5.0)
        h0 = 1.0
        sport = racket_params.TENNIS
        launches = racket_trajectory.FeasibleLaunches(
            theta_rad=np.array([np.radians(15.0)]),
            v0_mps=np.array([20.0]),
            flight_time_s=np.array([1.0]),
            net_clearance_m=np.array([1.0]),
        )
        phi = np.radians(3.0)
        errors = racket_trajectory.ShotErrors(
            d_theta_rad=np.zeros(2),
            d_v_frac=np.zeros(2),
            phi_rad=np.array([phi, -phi]),
        )
        # Run test.
        actual = racket_trajectory.simulate_landings(
            striker_xy, target_xy, h0, sport, launches, errors
        )
        # Check outputs.
        self.assertAlmostEqual(actual.x_m[0, 0], -actual.x_m[0, 1])
        self.assertAlmostEqual(actual.y_m[0, 0], actual.y_m[0, 1])
