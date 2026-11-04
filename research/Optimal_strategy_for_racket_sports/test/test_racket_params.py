"""
Tests for racket_params module.
"""

import logging
from typing import List, Literal

import numpy as np

import helpers.hunit_test as hunitest
from research.Optimal_strategy_for_racket_sports import racket_params

_LOG = logging.getLogger(__name__)


# #############################################################################
# TestCourtGeometry
# #############################################################################


class TestCourtGeometry(hunitest.TestCase):
    """
    Test `racket_params.CourtGeometry`.
    """

    def test1(self) -> None:
        """
        Test that a court with valid dimensions is created successfully.
        """
        # Prepare inputs.
        length_m = 23.77
        width_m = 8.23
        net_height_center_m = 0.914
        net_height_post_m = 1.07
        non_volley_zone_m = 0.0
        service_line_m = 6.40
        # Run test.
        court = racket_params.CourtGeometry(
            length_m=length_m,
            width_m=width_m,
            net_height_center_m=net_height_center_m,
            net_height_post_m=net_height_post_m,
            non_volley_zone_m=non_volley_zone_m,
            service_line_m=service_line_m,
        )
        # Check outputs.
        self.assertEqual(court.length_m, length_m)
        self.assertEqual(court.width_m, width_m)


# #############################################################################
# TestCourtRegion
# #############################################################################


class TestCourtRegion(hunitest.TestCase):
    """
    Test `racket_params.CourtRegion`.
    """

    def helper(self, x: np.ndarray, y: np.ndarray, expected: List[bool]) -> None:
        """
        Test helper for `CourtRegion.contains()`.

        :param x: array of x coordinates to test
        :param y: array of y coordinates to test
        :param expected: expected containment result for each (x, y) point
        """
        # Prepare inputs.
        x_min = -1
        x_max = 1
        y_min = 0
        y_max = 2
        region = racket_params.CourtRegion(
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
        )
        # Run test.
        actual = region.contains(x, y)
        # Check outputs.
        self.assert_equal(str(actual.tolist()), str(expected))

    def test1(self) -> None:
        """
        Test that a point strictly inside the region is contained.
        """
        # Prepare inputs.
        x = np.array([0.0])
        y = np.array([1.0])
        expected = [True]
        # Run test and check outputs.
        self.helper(x, y, expected)

    def test2(self) -> None:
        """
        Test that points on the region boundary are contained.
        """
        # Prepare inputs.
        x = np.array([-1.0, 1.0])
        y = np.array([0.0, 2.0])
        expected = [True, True]
        # Run test and check outputs.
        self.helper(x, y, expected)

    def test3(self) -> None:
        """
        Test that points outside the region are not contained.
        """
        # Prepare inputs.
        x = np.array([2.0, 0.0])
        y = np.array([1.0, 3.0])
        expected = [False, False]
        # Run test and check outputs.
        self.helper(x, y, expected)

    def test4(self) -> None:
        """
        Test that contains() evaluates correctly over an array of points.
        """
        # Prepare inputs.
        x = np.array([-1, 0, 1, 2])
        y = np.array([1, 1, 1, 1])
        expected = [True, True, True, False]
        # Run test and check outputs.
        self.helper(x, y, expected)


# #############################################################################
# Test_get_net_height
# #############################################################################


class Test_get_net_height(hunitest.TestCase):
    """
    Test `racket_params.get_net_height()`.
    """

    def helper(self, x: np.ndarray, expected: float) -> None:
        """
        Test helper for `get_net_height()`.

        :param x: single-element array with the lateral position to evaluate
        :param expected: expected net height at `x[0]`
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        # Run test.
        actual = racket_params.get_net_height(court, x)
        # Check outputs.
        self.assertAlmostEqual(actual[0], expected)

    def test1(self) -> None:
        """
        Test that the net height at the center matches net_height_center_m.
        """
        # Prepare inputs.
        x = np.array([0.0])
        expected = racket_params.TENNIS.court.net_height_center_m
        # Run test and check outputs.
        self.helper(x, expected)

    def test2(self) -> None:
        """
        Test that the net height at the sideline matches net_height_post_m.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        x_post = court.width_m / 2
        x = np.array([x_post])
        expected = court.net_height_post_m
        # Run test and check outputs.
        self.helper(x, expected)

    def test3(self) -> None:
        """
        Test that the net height is symmetric around the center line.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        x = np.array([2.0])
        # Run test.
        height_pos = racket_params.get_net_height(court, x)
        height_neg = racket_params.get_net_height(court, -x)
        # Check outputs.
        self.assertAlmostEqual(height_pos[0], height_neg[0])

    def test4(self) -> None:
        """
        Test that the net height interpolates linearly between center and
        post.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        x_post = court.width_m / 2
        x = np.array([x_post / 2])
        expected = court.net_height_center_m + 0.5 * (
            court.net_height_post_m - court.net_height_center_m
        )
        # Run test and check outputs.
        self.helper(x, expected)

    def test5(self) -> None:
        """
        Test that get_net_height() evaluates correctly over an array of
        positions.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        x = np.array([0.0, 1.0, 2.0])
        # Run test.
        heights = racket_params.get_net_height(court, x)
        # Check outputs.
        self.assertEqual(len(heights), 3)
        self.assertAlmostEqual(heights[0], court.net_height_center_m)


# #############################################################################
# Test_get_half_court_region
# #############################################################################


class Test_get_half_court_region(hunitest.TestCase):
    """
    Test `racket_params.get_half_court_region()`.
    """

    def helper(self, court: racket_params.CourtGeometry, expected: str) -> None:
        """
        Test helper for `get_half_court_region()`.

        :param court: court geometry to compute the half-court region for
        :param expected: expected string representation of the region
        """
        # Run test.
        actual = racket_params.get_half_court_region(court)
        # Check outputs.
        self.assert_equal(str(actual), expected)

    def test1(self) -> None:
        """
        Test the half-court region bounds for tennis.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        # Prepare outputs.
        expected = (
            "CourtRegion(x_min=-4.115, x_max=4.115, y_min=0, y_max=11.885)"
        )
        # Run test and check outputs.
        self.helper(court, expected)

    def test2(self) -> None:
        """
        Test the half-court region bounds for pickleball.
        """
        # Prepare inputs.
        court = racket_params.PICKLEBALL.court
        # Prepare outputs.
        expected = "CourtRegion(x_min=-3.05, x_max=3.05, y_min=0, y_max=6.705)"
        # Run test and check outputs.
        self.helper(court, expected)


# #############################################################################
# Test_get_service_box_region
# #############################################################################


class Test_get_service_box_region(hunitest.TestCase):
    """
    Test `racket_params.get_service_box_region()`.
    """

    def helper(
        self,
        court: racket_params.CourtGeometry,
        serve_side: Literal["deuce", "ad"],
        expected: str,
    ) -> None:
        """
        Test helper for `get_service_box_region()`.

        :param court: court geometry to compute the service box for
        :param serve_side: "deuce" or "ad"
        :param expected: expected string representation of the region
        """
        # Run test.
        actual = racket_params.get_service_box_region(court, serve_side)
        # Check outputs.
        self.assert_equal(str(actual), expected)

    def test1(self) -> None:
        """
        Test the deuce-side service box bounds for tennis.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        serve_side: Literal["deuce", "ad"] = "deuce"
        # Prepare outputs.
        expected = "CourtRegion(x_min=-4.115, x_max=0, y_min=0.0, y_max=6.4)"
        # Run test and check outputs.
        self.helper(court, serve_side, expected)

    def test2(self) -> None:
        """
        Test the ad-side service box bounds for tennis.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        serve_side: Literal["deuce", "ad"] = "ad"
        # Prepare outputs.
        expected = "CourtRegion(x_min=0, x_max=4.115, y_min=0.0, y_max=6.4)"
        # Run test and check outputs.
        self.helper(court, serve_side, expected)

    def test3(self) -> None:
        """
        Test the deuce-side service box bounds for pickleball, which start
        after the non-volley zone.
        """
        # Prepare inputs.
        court = racket_params.PICKLEBALL.court
        serve_side: Literal["deuce", "ad"] = "deuce"
        # Prepare outputs.
        expected = "CourtRegion(x_min=-3.05, x_max=0, y_min=2.13, y_max=6.71)"
        # Run test and check outputs.
        self.helper(court, serve_side, expected)


# #############################################################################
# TestTENNIS
# #############################################################################


class TestTENNIS(hunitest.TestCase):
    """
    Test the preset `racket_params.TENNIS`.
    """

    def test1(self) -> None:
        """
        Test the preset TENNIS sport parameters.
        """
        # Prepare inputs.
        sport = racket_params.TENNIS
        # Prepare outputs.
        expected = "name=Tennis length_m=23.77 width_m=8.23 service_line_m=6.4"
        # Run test.
        actual = (
            f"name={sport.name} length_m={sport.court.length_m} "
            f"width_m={sport.court.width_m} "
            f"service_line_m={sport.court.service_line_m}"
        )
        # Check outputs.
        self.assert_equal(actual, expected)


# #############################################################################
# TestPICKLEBALL
# #############################################################################


class TestPICKLEBALL(hunitest.TestCase):
    """
    Test the preset `racket_params.PICKLEBALL`.
    """

    def test1(self) -> None:
        """
        Test the preset PICKLEBALL sport parameters.
        """
        # Prepare inputs.
        sport = racket_params.PICKLEBALL
        # Prepare outputs.
        expected = "name=Pickleball length_m=13.41 non_volley_zone_m=2.13"
        # Run test.
        actual = (
            f"name={sport.name} length_m={sport.court.length_m} "
            f"non_volley_zone_m={sport.court.non_volley_zone_m}"
        )
        # Check outputs.
        self.assert_equal(actual, expected)


# #############################################################################
# TestDEFAULT_PLAYER
# #############################################################################


class TestDEFAULT_PLAYER(hunitest.TestCase):
    """
    Test the preset `racket_params.DEFAULT_PLAYER`.
    """

    def test1(self) -> None:
        """
        Test the preset DEFAULT_PLAYER parameters.
        """
        # Prepare inputs.
        player = racket_params.DEFAULT_PLAYER
        # Prepare outputs.
        expected = "reaction_time_s=0.2 move_speed_mps=1.5 contact_height_m=1.0"
        # Run test.
        actual = (
            f"reaction_time_s={player.reaction_time_s} "
            f"move_speed_mps={player.move_speed_mps} "
            f"contact_height_m={player.contact_height_m}"
        )
        # Check outputs.
        self.assert_equal(actual, expected)


# #############################################################################
# TestDEFAULT_ERROR
# #############################################################################


class TestDEFAULT_ERROR(hunitest.TestCase):
    """
    Test the preset `racket_params.DEFAULT_ERROR`.
    """

    def test1(self) -> None:
        """
        Test the preset DEFAULT_ERROR shot error model.
        """
        # Prepare outputs.
        sigma_theta_deg = 1.5
        expected_sigma_theta_rad = np.radians(sigma_theta_deg)
        # Run test.
        actual_sigma_theta_rad = racket_params.DEFAULT_ERROR.sigma_theta_rad
        # Check outputs.
        self.assertAlmostEqual(actual_sigma_theta_rad, expected_sigma_theta_rad)
