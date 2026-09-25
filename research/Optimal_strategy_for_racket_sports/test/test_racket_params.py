"""
Tests for racket_params module.
"""

import dataclasses
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
        # Run test.
        court = racket_params.CourtGeometry(
            length_m=length_m,
            width_m=width_m,
            net_height_center_m=0.914,
            net_height_post_m=1.07,
            non_volley_zone_m=0.0,
            service_line_m=6.40,
        )
        # Check outputs.
        self.assertEqual(court.length_m, length_m)
        self.assertEqual(court.width_m, width_m)

    def test2(self) -> None:
        """
        Test that a non-positive length_m raises an assertion.
        """
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            racket_params.CourtGeometry(
                length_m=-1,
                width_m=8.23,
                net_height_center_m=0.914,
                net_height_post_m=1.07,
            )

    def test3(self) -> None:
        """
        Test that a CourtGeometry instance cannot be mutated after creation.
        """
        # Prepare inputs.
        court = racket_params.CourtGeometry(
            length_m=23.77,
            width_m=8.23,
            net_height_center_m=0.914,
            net_height_post_m=1.07,
        )
        # Run test and check outputs.
        with self.assertRaises(dataclasses.FrozenInstanceError):
            court.length_m = 24.0


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
        region = racket_params.CourtRegion(x_min=-1, x_max=1, y_min=0, y_max=2)
        # Run test.
        actual = region.contains(x, y)
        # Check outputs.
        self.assert_equal(str(actual.tolist()), str(expected))

    def test1(self) -> None:
        """
        Test that a region with valid bounds is created successfully.
        """
        # Prepare inputs.
        x_min = -1
        x_max = 1
        # Run test.
        region = racket_params.CourtRegion(
            x_min=x_min, x_max=x_max, y_min=0, y_max=2
        )
        # Check outputs.
        self.assertEqual(region.x_min, x_min)
        self.assertEqual(region.x_max, x_max)

    def test2(self) -> None:
        """
        Test that x_min >= x_max raises an assertion.
        """
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            racket_params.CourtRegion(x_min=1, x_max=1, y_min=0, y_max=2)

    def test3(self) -> None:
        """
        Test that a point strictly inside the region is contained.
        """
        # Prepare inputs.
        x = np.array([0.0])
        y = np.array([1.0])
        expected = [True]
        # Run test and check outputs.
        self.helper(x, y, expected)

    def test4(self) -> None:
        """
        Test that points on the region boundary are contained.
        """
        # Prepare inputs.
        x = np.array([-1.0, 1.0])
        y = np.array([0.0, 2.0])
        expected = [True, True]
        # Run test and check outputs.
        self.helper(x, y, expected)

    def test5(self) -> None:
        """
        Test that points outside the region are not contained.
        """
        # Prepare inputs.
        x = np.array([2.0, 0.0])
        y = np.array([1.0, 3.0])
        expected = [False, False]
        # Run test and check outputs.
        self.helper(x, y, expected)

    def test6(self) -> None:
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
        expected = str(
            racket_params.CourtRegion(
                x_min=-4.115, x_max=4.115, y_min=0, y_max=11.885
            )
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
        expected = str(
            racket_params.CourtRegion(
                x_min=-3.05, x_max=3.05, y_min=0, y_max=6.705
            )
        )
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
        serve_side = "deuce"
        # Prepare outputs.
        expected = str(
            racket_params.CourtRegion(
                x_min=-4.115, x_max=0, y_min=0.0, y_max=6.40
            )
        )
        # Run test and check outputs.
        self.helper(court, serve_side, expected)

    def test2(self) -> None:
        """
        Test the ad-side service box bounds for tennis.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        serve_side = "ad"
        # Prepare outputs.
        expected = str(
            racket_params.CourtRegion(
                x_min=0, x_max=4.115, y_min=0.0, y_max=6.40
            )
        )
        # Run test and check outputs.
        self.helper(court, serve_side, expected)

    def test3(self) -> None:
        """
        Test the deuce-side service box bounds for pickleball, which start
        after the non-volley zone.
        """
        # Prepare inputs.
        court = racket_params.PICKLEBALL.court
        serve_side = "deuce"
        # Prepare outputs.
        expected = str(
            racket_params.CourtRegion(
                x_min=-3.05, x_max=0, y_min=2.13, y_max=6.71
            )
        )
        # Run test and check outputs.
        self.helper(court, serve_side, expected)

    def test4(self) -> None:
        """
        Test that an unrecognized serve_side raises an assertion.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        serve_side = "invalid"
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            racket_params.get_service_box_region(court, serve_side)


# #############################################################################
# TestPresets
# #############################################################################


class TestPresets(hunitest.TestCase):
    """
    Test the preset `racket_params.TENNIS`, `racket_params.PICKLEBALL`,
    `racket_params.DEFAULT_PLAYER`, and `racket_params.DEFAULT_ERROR`.
    """

    def test1(self) -> None:
        """
        Test the preset TENNIS sport parameters.
        """
        # Check outputs.
        self.assertEqual(racket_params.TENNIS.name, "Tennis")
        self.assertEqual(racket_params.TENNIS.court.length_m, 23.77)
        self.assertEqual(racket_params.TENNIS.court.width_m, 8.23)
        self.assertEqual(racket_params.TENNIS.court.service_line_m, 6.40)

    def test2(self) -> None:
        """
        Test the preset PICKLEBALL sport parameters.
        """
        # Check outputs.
        self.assertEqual(racket_params.PICKLEBALL.name, "Pickleball")
        self.assertEqual(racket_params.PICKLEBALL.court.length_m, 13.41)
        self.assertEqual(racket_params.PICKLEBALL.court.non_volley_zone_m, 2.13)

    def test3(self) -> None:
        """
        Test the preset DEFAULT_PLAYER parameters.
        """
        # Check outputs.
        self.assertEqual(racket_params.DEFAULT_PLAYER.reaction_time_s, 0.2)
        self.assertEqual(racket_params.DEFAULT_PLAYER.move_speed_mps, 1.5)
        self.assertEqual(racket_params.DEFAULT_PLAYER.contact_height_m, 1.0)

    def test4(self) -> None:
        """
        Test the preset DEFAULT_ERROR shot error model.
        """
        # Check outputs.
        self.assertAlmostEqual(
            racket_params.DEFAULT_ERROR.sigma_theta_rad, np.radians(1.5)
        )
