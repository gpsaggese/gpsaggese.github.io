"""
Tests for racket_shot_probability_utils module.

Import as:

import research.Optimal_strategy_for_racket_sports.test.test_racket_shot_probability_utils as rosfrsttrspu
"""

import logging
from typing import Tuple

import numpy as np

import helpers.hunit_test as hunitest
from research.Optimal_strategy_for_racket_sports import racket_params
from research.Optimal_strategy_for_racket_sports.notebooks import (
    racket_shot_probability_utils,
)

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_get_striker_half_region
# #############################################################################


class Test_get_striker_half_region(hunitest.TestCase):
    """
    Test `racket_shot_probability_utils.get_striker_half_region()`.
    """

    def helper(self, court: racket_params.CourtGeometry, expected: str) -> None:
        """
        Test helper for `get_striker_half_region()`.

        :param court: court geometry to compute player 1's half for
        :param expected: expected string representation of the region
        """
        # Run test.
        actual = racket_shot_probability_utils.get_striker_half_region(court)
        # Check outputs.
        self.assert_equal(str(actual), expected)

    def test1(self) -> None:
        """
        Test player 1's half-court bounds for tennis.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        # Prepare outputs.
        x_min = -4.115
        x_max = 4.115
        y_min = -11.885
        y_max = 0
        expected = str(
            racket_params.CourtRegion(
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
            )
        )
        # Run test and check outputs.
        self.helper(court, expected)

    def test2(self) -> None:
        """
        Test player 1's half-court bounds for pickleball.
        """
        # Prepare inputs.
        court = racket_params.PICKLEBALL.court
        # Prepare outputs.
        x_min = -3.05
        x_max = 3.05
        y_min = -6.705
        y_max = 0
        expected = str(
            racket_params.CourtRegion(
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
            )
        )
        # Run test and check outputs.
        self.helper(court, expected)


# #############################################################################
# Test_compute_p_in
# #############################################################################


class Test_compute_p_in(hunitest.TestCase):
    """
    Test `racket_shot_probability_utils.compute_p_in()`.
    """

    def helper(
        self,
        court: racket_params.CourtGeometry,
        target_xy: Tuple[float, float],
        std_xy: Tuple[float, float],
        expected: float,
    ) -> None:
        """
        Test helper for `compute_p_in()` on player 1's half of `court`.

        :param court: court geometry whose player 1 half is the in region
        :param target_xy: aim point `(x2, y2)` in meters
        :param std_xy: standard deviations `(std_x2, std_y2)` in meters
        :param expected: expected probability, checked to 4 decimal places
        """
        # Prepare inputs.
        region = racket_shot_probability_utils.get_striker_half_region(court)
        # Run test.
        actual = racket_shot_probability_utils.compute_p_in(
            target_xy, std_xy, region
        )
        # Check outputs.
        self.assertAlmostEqual(actual, expected, places=4)

    def test1(self) -> None:
        """
        Test that a tiny std at the center of the half court gives about 1.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        target_xy = (0.0, -court.length_m / 4)
        std_xy = (0.1, 0.1)
        # Prepare outputs.
        expected = 1.0
        # Run test and check outputs.
        self.helper(court, target_xy, std_xy, expected)

    def test2(self) -> None:
        """
        Test that aiming at the baseline gives about 0.5.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        target_xy = (0.0, -court.length_m / 2)
        std_xy = (0.1, 0.1)
        # Prepare outputs.
        expected = 0.5
        # Run test and check outputs.
        self.helper(court, target_xy, std_xy, expected)

    def test3(self) -> None:
        """
        Test that aiming at the sideline gives about 0.5.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        target_xy = (court.width_m / 2, -court.length_m / 4)
        std_xy = (0.1, 0.1)
        # Prepare outputs.
        expected = 0.5
        # Run test and check outputs.
        self.helper(court, target_xy, std_xy, expected)

    def test4(self) -> None:
        """
        Test that aiming at the sideline and baseline corner gives about 0.25.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        target_xy = (court.width_m / 2, -court.length_m / 2)
        std_xy = (0.1, 0.1)
        # Prepare outputs.
        expected = 0.25
        # Run test and check outputs.
        self.helper(court, target_xy, std_xy, expected)

    def test5(self) -> None:
        """
        Test that aiming at the net line gives about 0.5.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        target_xy = (0.0, 0.0)
        std_xy = (0.1, 0.1)
        # Prepare outputs.
        expected = 0.5
        # Run test and check outputs.
        self.helper(court, target_xy, std_xy, expected)

    def test6(self) -> None:
        """
        Test that aiming far outside the sideline gives about 0.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        target_xy = (court.width_m, -court.length_m / 4)
        std_xy = (0.1, 0.1)
        # Prepare outputs.
        expected = 0.0
        # Run test and check outputs.
        self.helper(court, target_xy, std_xy, expected)

    def test7(self) -> None:
        """
        Test that a very large std at the center of the half court gives about 0.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        target_xy = (0.0, -court.length_m / 4)
        std_xy = (1000.0, 1000.0)
        # Prepare outputs.
        expected = 0.0
        # Run test and check outputs.
        self.helper(court, target_xy, std_xy, expected)


# #############################################################################
# Test_sample_landings
# #############################################################################


class Test_sample_landings(hunitest.TestCase):
    """
    Test `racket_shot_probability_utils.sample_landings()`.
    """

    def helper(
        self,
        court: racket_params.CourtGeometry,
        target_xy: Tuple[float, float],
        std_xy: Tuple[float, float],
    ) -> None:
        """
        Test helper comparing the Monte Carlo in-fraction to `compute_p_in()`.

        :param court: court geometry whose player 1 half is the in region
        :param target_xy: aim point `(x2, y2)` in meters
        :param std_xy: standard deviations `(std_x2, std_y2)` in meters
        """
        # Prepare inputs.
        region = racket_shot_probability_utils.get_striker_half_region(court)
        n_samples = 10000
        rng = np.random.default_rng(1)
        # Prepare outputs.
        expected = racket_shot_probability_utils.compute_p_in(
            target_xy, std_xy, region
        )
        # Standard error of a fraction estimated from `n_samples` draws.
        std_err = np.sqrt(expected * (1 - expected) / n_samples)
        # Run test.
        x, y = racket_shot_probability_utils.sample_landings(
            target_xy, std_xy, n_samples, rng
        )
        actual = region.contains(x, y).mean()
        # Check outputs.
        self.assertAlmostEqual(actual, expected, delta=3 * std_err)

    def test1(self) -> None:
        """
        Test that the same seed gives identical samples.
        """
        # Prepare inputs.
        target_xy = (1.0, -5.0)
        std_xy = (0.5, 2.0)
        n_samples = 10
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        # Run test.
        x1, y1 = racket_shot_probability_utils.sample_landings(
            target_xy, std_xy, n_samples, rng1
        )
        x2, y2 = racket_shot_probability_utils.sample_landings(
            target_xy, std_xy, n_samples, rng2
        )
        # Check outputs.
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

    def test2(self) -> None:
        """
        Test that the sample mean and std match `target_xy` and `std_xy`.
        """
        # Prepare inputs.
        target_xy = (1.0, -5.0)
        std_xy = (0.5, 2.0)
        n_samples = 100000
        rng = np.random.default_rng(1)
        # Prepare outputs.
        # With 100,000 samples the sample mean and std are typically off by
        # 0.3% of the std or less, so 1% of the std is a safe tolerance.
        delta_x = 0.01 * std_xy[0]
        delta_y = 0.01 * std_xy[1]
        # Run test.
        x, y = racket_shot_probability_utils.sample_landings(
            target_xy, std_xy, n_samples, rng
        )
        # Check outputs.
        self.assertAlmostEqual(x.mean(), target_xy[0], delta=delta_x)
        self.assertAlmostEqual(y.mean(), target_xy[1], delta=delta_y)
        self.assertAlmostEqual(x.std(), std_xy[0], delta=delta_x)
        self.assertAlmostEqual(y.std(), std_xy[1], delta=delta_y)

    def test3(self) -> None:
        """
        Test Monte Carlo agreement when aiming at the center of the half court.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        target_xy = (0.0, -court.length_m / 4)
        std_xy = (2.0, 3.0)
        # Run test and check outputs.
        self.helper(court, target_xy, std_xy)

    def test4(self) -> None:
        """
        Test Monte Carlo agreement when aiming at the sideline.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        target_xy = (court.width_m / 2, -court.length_m / 4)
        std_xy = (2.0, 3.0)
        # Run test and check outputs.
        self.helper(court, target_xy, std_xy)

    def test5(self) -> None:
        """
        Test Monte Carlo agreement when aiming at the baseline corner.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        target_xy = (court.width_m / 2, -court.length_m / 2)
        std_xy = (2.0, 3.0)
        # Run test and check outputs.
        self.helper(court, target_xy, std_xy)

    def test6(self) -> None:
        """
        Test Monte Carlo agreement when aiming 2 m outside the sideline.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        target_xy = (court.width_m / 2 + 2.0, -court.length_m / 4)
        std_xy = (2.0, 3.0)
        # Run test and check outputs.
        self.helper(court, target_xy, std_xy)
