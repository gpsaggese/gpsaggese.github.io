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

    Tests cover:
    - Happy path: normal inputs (theta=8 deg, d_target=20 m, h0=1 m)
    - Edge case: angle unable to reach target returns NaN
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
        # Prepare outputs.
        # Expected: result is NaN
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

    Tests cover:
    - Edge case: h0=0 gives textbook formula result
    - Happy path: normal inputs (v0=22.9133, theta=8 deg, h0=1 m)
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

    Tests cover:
    - Happy path: net clearance calculation
    - Edge case: x=0 (at striker position) returns h0
    - Edge case: negative angle (downward trajectory)
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

    def test2(self) -> None:
        """
        Test height at x=0 (at the striker position), should equal h0.
        """
        # Prepare inputs.
        x = np.array([0.0])
        v0 = np.array([22.9133])
        theta = np.array([np.radians(8.0)])
        h0 = 2.5
        # Prepare outputs.
        expected = h0
        # Run test.
        height = racket_trajectory.get_height_at(x, v0, theta, h0)
        # Check outputs.
        self.assertAlmostEqual(height[0], expected, places=6)

    def test3(self) -> None:
        """
        Test with negative angle (downward trajectory).
        """
        # Prepare inputs.
        x = np.array([5.0])
        v0 = np.array([15.0])
        theta = np.array([np.radians(-10.0)])
        h0 = 2.0
        # Prepare outputs.
        # Expected: height is less than h0 for downward angle
        # Run test.
        height = racket_trajectory.get_height_at(x, v0, theta, h0)
        # Check outputs.
        self.assertLess(height[0], h0)


# #############################################################################
# Test_get_feasible_launches
# #############################################################################


class Test_get_feasible_launches(hunitest.TestCase):
    """
    Test `racket_trajectory.get_feasible_launches()`.

    Tests cover:
    - Happy path: finds feasible launches within speed and net clearance constraints
    - Edge case: tiny max_ball_speed_mps yields no feasible launches
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
        # Prepare outputs.
        # Expected: some feasible launches found with valid speeds and clearances
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
        # Prepare outputs.
        expected_count = 0
        # Run test.
        actual = racket_trajectory.get_feasible_launches(
            striker_xy,
            target_xy,
            h0,
            sport,
            theta_grid_rad=theta_grid_rad,
        )
        # Check outputs.
        self.assertEqual(len(actual.theta_rad), expected_count)


# #############################################################################
# Test_sample_shot_errors
# #############################################################################


class Test_sample_shot_errors(hunitest.TestCase):
    """
    Test `racket_trajectory.sample_shot_errors()`.

    Tests cover:
    - Happy path: same seed produces reproducible random samples
    - Edge case: zero sigmas produce zero-valued draws
    """

    def test1(self) -> None:
        """
        Test that the same seed gives the same draws.
        """
        # Prepare inputs.
        error = racket_params.DEFAULT_ERROR
        n_samples = 10
        # Prepare outputs.
        # Expected: two calls with same seed produce identical arrays
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
        # Prepare outputs.
        expected_d_theta = np.zeros(n_samples)
        expected_d_v = np.zeros(n_samples)
        expected_phi = np.zeros(n_samples)
        # Run test.
        actual = racket_trajectory.sample_shot_errors(
            error, n_samples, np.random.default_rng(0)
        )
        # Check outputs.
        np.testing.assert_array_equal(actual.d_theta_rad, expected_d_theta)
        np.testing.assert_array_equal(actual.d_v_frac, expected_d_v)
        np.testing.assert_array_equal(actual.phi_rad, expected_phi)


# #############################################################################
# Test_simulate_landings
# #############################################################################


class Test_simulate_landings(hunitest.TestCase):
    """
    Test `racket_trajectory.simulate_landings()`.

    Tests cover:
    - Edge case: zero-error samples land exactly on nominal target
    - Edge case: symmetric lateral errors produce symmetric landings
    - Happy path: realistic simulation with multiple launches and error samples
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
        # Prepare outputs.
        # Expected: all landings match target location exactly
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
        # Prepare outputs.
        # Expected: symmetric errors produce symmetric landing positions
        # Run test.
        actual = racket_trajectory.simulate_landings(
            striker_xy, target_xy, h0, sport, launches, errors
        )
        # Check outputs.
        self.assertAlmostEqual(actual.x_m[0, 0], -actual.x_m[0, 1])
        self.assertAlmostEqual(actual.y_m[0, 0], actual.y_m[0, 1])

    def test3(self) -> None:
        """
        Test realistic landing simulation with multiple launches and error samples.
        """
        # Prepare inputs.
        striker_xy = (0.0, -11.0)
        target_xy = (0.0, 5.0)
        h0 = 1.0
        sport = racket_params.TENNIS
        launches = racket_trajectory.FeasibleLaunches(
            theta_rad=np.array([np.radians(10.0), np.radians(15.0)]),
            v0_mps=np.array([20.0, 22.0]),
            flight_time_s=np.array([0.9, 1.0]),
            net_clearance_m=np.array([0.5, 0.7]),
        )
        errors = racket_trajectory.ShotErrors(
            d_theta_rad=np.array([0.01, -0.01, 0.005]),
            d_v_frac=np.array([0.05, -0.05, 0.02]),
            phi_rad=np.array([np.radians(1.0), np.radians(-1.0), 0.0]),
        )
        # Prepare outputs.
        expected_shape = (2, 3)
        # Run test.
        actual = racket_trajectory.simulate_landings(
            striker_xy, target_xy, h0, sport, launches, errors
        )
        # Check outputs.
        self.assertEqual(actual.x_m.shape, expected_shape)
        self.assertEqual(actual.y_m.shape, expected_shape)
