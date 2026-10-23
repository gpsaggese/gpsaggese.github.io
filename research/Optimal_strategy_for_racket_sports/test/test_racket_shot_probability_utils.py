"""
Tests for racket_shot_probability_utils module.

Import as:

import research.Optimal_strategy_for_racket_sports.test.test_racket_shot_probability_utils as rosfrsttrspu
"""

import dataclasses
import html
import logging
import types
import unittest.mock as umock
from typing import Any, Dict, List, Tuple

import ipywidgets
import matplotlib
import matplotlib.collections as mcollections
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

import helpers.hunit_test as hunitest
from research.Optimal_strategy_for_racket_sports import racket_params
from research.Optimal_strategy_for_racket_sports import racket_scoring
from research.Optimal_strategy_for_racket_sports.notebooks import (
    racket_shot_probability_utils,
)

_LOG = logging.getLogger(__name__)

# Use the non-interactive backend so the widget renders without a display.
matplotlib.use("Agg")


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

    def test3(self) -> None:
        """
        Test that a court with a `nan`, `inf`, or negative size raises an error.
        """
        # Prepare inputs.
        # `CourtGeometry` itself rejects `nan` and negative sizes, so stand-in
        # objects with the same fields reach the check in
        # `get_striker_half_region()`; it accepts `inf`.
        court = racket_params.TENNIS.court
        bad_courts = [
            types.SimpleNamespace(width_m=float("nan"), length_m=23.77),
            types.SimpleNamespace(width_m=8.23, length_m=-1.0),
            dataclasses.replace(court, length_m=float("inf")),
            dataclasses.replace(court, width_m=float("inf")),
        ]
        # Run test and check outputs.
        for bad_court in bad_courts:
            with self.subTest(court=bad_court):
                with self.assertRaises(AssertionError):
                    racket_shot_probability_utils.get_striker_half_region(
                        bad_court
                    )


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

    def test8(self) -> None:
        """
        Test that a `nan` or `inf` target, or a bad std, raises an error.
        """
        # Prepare inputs.
        region = racket_shot_probability_utils.get_striker_half_region(
            racket_params.TENNIS.court
        )
        nan, inf = float("nan"), float("inf")
        target_xy = (0.0, -5.0)
        std_xy = (0.5, 1.0)
        bad_inputs = [
            ((nan, -5.0), std_xy),
            ((0.0, inf), std_xy),
            (target_xy, (nan, 1.0)),
            (target_xy, (0.5, inf)),
            (target_xy, (0.0, 1.0)),
            (target_xy, (0.5, -1.0)),
        ]
        # Run test and check outputs.
        for bad_target_xy, bad_std_xy in bad_inputs:
            with self.subTest(target_xy=bad_target_xy, std_xy=bad_std_xy):
                with self.assertRaises(AssertionError):
                    racket_shot_probability_utils.compute_p_in(
                        bad_target_xy, bad_std_xy, region
                    )


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

    def test7(self) -> None:
        """
        Test that a bad target, std, or `n_samples` raises an error.
        """
        # Prepare inputs.
        nan, inf = float("nan"), float("inf")
        target_xy = (0.0, -5.0)
        std_xy = (0.5, 1.0)
        n_samples = 10
        bad_inputs = [
            ((nan, -5.0), std_xy, n_samples),
            ((inf, -5.0), std_xy, n_samples),
            (target_xy, (nan, 1.0), n_samples),
            (target_xy, (inf, 1.0), n_samples),
            (target_xy, (0.0, 1.0), n_samples),
            (target_xy, (-1.0, 1.0), n_samples),
            (target_xy, std_xy, 0),
            (target_xy, std_xy, -5),
            (target_xy, std_xy, 2.5),
        ]
        # Run test and check outputs.
        for bad_target_xy, bad_std_xy, bad_n_samples in bad_inputs:
            with self.subTest(
                target_xy=bad_target_xy, std_xy=bad_std_xy, n=bad_n_samples
            ):
                with self.assertRaises(AssertionError):
                    racket_shot_probability_utils.sample_landings(
                        bad_target_xy,
                        bad_std_xy,
                        bad_n_samples,
                        np.random.default_rng(1),
                    )


# #############################################################################
# Test_get_player1_xy
# #############################################################################


class Test_get_player1_xy(hunitest.TestCase):
    """
    Test `racket_shot_probability_utils.get_player1_xy()`.
    """

    def helper(self, court: racket_params.CourtGeometry, expected: str) -> None:
        """
        Test helper for `get_player1_xy()`.

        :param court: court geometry to place player 1 on
        :param expected: expected string representation of the position
        """
        # Run test.
        actual = racket_shot_probability_utils.get_player1_xy(court)
        # Check outputs.
        self.assert_equal(str(actual), expected)

    def test1(self) -> None:
        """
        Test that player 1 stands at the center of the tennis baseline.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        # Prepare outputs.
        expected = "(0.0, -11.885)"
        # Run test and check outputs.
        self.helper(court, expected)

    def test2(self) -> None:
        """
        Test that player 1 stands at the center of the pickleball baseline.
        """
        # Prepare inputs.
        court = racket_params.PICKLEBALL.court
        # Prepare outputs.
        expected = "(0.0, -6.705)"
        # Run test and check outputs.
        self.helper(court, expected)

    def test3(self) -> None:
        """
        Test that a court with an `inf`, `nan`, or negative length raises a
        clear error.
        """
        # Prepare inputs.
        # `CourtGeometry` itself rejects `nan` and negative lengths, so
        # stand-in objects with the same field reach the check in
        # `get_player1_xy()`; it accepts `inf`.
        court = racket_params.TENNIS.court
        bad_inputs = [
            (
                dataclasses.replace(court, length_m=float("inf")),
                "Court length_m=inf must be finite",
            ),
            (
                types.SimpleNamespace(length_m=float("nan")),
                "Court length_m=nan must be finite",
            ),
            (
                types.SimpleNamespace(length_m=-1.0),
                "Court length must be positive",
            ),
        ]
        # Run test and check outputs.
        for bad_court, expected in bad_inputs:
            with self.subTest(court=bad_court):
                with self.assertRaises(AssertionError) as cm:
                    racket_shot_probability_utils.get_player1_xy(bad_court)
                self.assertIn(expected, str(cm.exception))


# #############################################################################
# Test_compute_flight_time
# #############################################################################


class Test_compute_flight_time(hunitest.TestCase):
    """
    Test `racket_shot_probability_utils.compute_flight_time()`.
    """

    def helper(
        self,
        hitter_xy: Tuple[float, float],
        x: np.ndarray,
        y: np.ndarray,
        ball_speed_mps: float,
        expected: str,
    ) -> None:
        """
        Test helper for `compute_flight_time()`.

        :param hitter_xy: where player 2 hits the return
        :param x: landing `x` of each sample
        :param y: landing `y` of each sample
        :param ball_speed_mps: speed of the return in m/s
        :param expected: expected flight times in seconds, rounded to 6
            decimals, as the string of a (nested) list
        """
        # Run test.
        actual = racket_shot_probability_utils.compute_flight_time(
            hitter_xy, x, y, ball_speed_mps
        )
        # Check outputs.
        actual = str(np.round(actual, 6).tolist())
        self.assert_equal(actual, expected)

    def test1(self) -> None:
        """
        Test hand-computed flight times of 5 m and 20 m shots at 25 m/s.
        """
        # Prepare inputs.
        # From (0, 6): (3, 2) is 5 m away (a 3-4-5 triangle), (0, -14) is 20 m
        # away, so the times are 5 / 25 = 0.2 s and 20 / 25 = 0.8 s.
        hitter_xy = (0.0, 6.0)
        x = np.array([3.0, 0.0])
        y = np.array([2.0, -14.0])
        ball_speed_mps = 25.0
        # Prepare outputs.
        expected = "[0.2, 0.8]"
        # Run test and check outputs.
        self.helper(hitter_xy, x, y, ball_speed_mps, expected)

    def test2(self) -> None:
        """
        Test that a landing point at player 2 has a flight time of 0.
        """
        # Prepare inputs.
        hitter_xy = (-2.0, 4.8)
        x = np.array([-2.0])
        y = np.array([4.8])
        ball_speed_mps = 25.0
        # Prepare outputs.
        expected = "[0.0]"
        # Run test and check outputs.
        self.helper(hitter_xy, x, y, ball_speed_mps, expected)

    def test3(self) -> None:
        """
        Test that a very fast ball, 1000 m/s over 20 m, flies 0.02 s.
        """
        # Prepare inputs.
        hitter_xy = (0.0, 6.0)
        x = np.array([0.0])
        y = np.array([-14.0])
        ball_speed_mps = 1000.0
        # Prepare outputs.
        expected = "[0.02]"
        # Run test and check outputs.
        self.helper(hitter_xy, x, y, ball_speed_mps, expected)

    def test4(self) -> None:
        """
        Test that a very slow ball, 1 m/s over 20 m, flies 20 s.
        """
        # Prepare inputs.
        hitter_xy = (0.0, 6.0)
        x = np.array([0.0])
        y = np.array([-14.0])
        ball_speed_mps = 1.0
        # Prepare outputs.
        expected = "[20.0]"
        # Run test and check outputs.
        self.helper(hitter_xy, x, y, ball_speed_mps, expected)

    def test5(self) -> None:
        """
        Test that a 2D array of landing points gives times of the same shape.
        """
        # Prepare inputs.
        # One row per target, one column per sample, as a heatmap will pass.
        hitter_xy = (0.0, 6.0)
        x = np.array([[3.0, 0.0], [0.0, 3.0]])
        y = np.array([[2.0, -14.0], [6.0, 6.0]])
        ball_speed_mps = 25.0
        # Prepare outputs.
        expected = "[[0.2, 0.8], [0.0, 0.12]]"
        # Run test and check outputs.
        self.helper(hitter_xy, x, y, ball_speed_mps, expected)

    def test6(self) -> None:
        """
        Test that a bad point, landing arrays, or ball speed raises a clear
        error.
        """
        # Prepare inputs.
        nan, inf = float("nan"), float("inf")
        hitter_xy = (0.0, 6.0)
        x = np.array([3.0, 0.0])
        y = np.array([2.0, -14.0])
        ball_speed_mps = 25.0
        bad_inputs = [
            ((nan, 6.0), x, y, ball_speed_mps, "hitter_xy=(nan, 6.0) must be"),
            ((0.0, 6.0, 1.0), x, y, ball_speed_mps, "must be a point (x, y)"),
            (hitter_xy, [3.0, 0.0], y, ball_speed_mps, "x must be a numpy"),
            (hitter_xy, x, y[:1], ball_speed_mps, "must have the same shape"),
            (hitter_xy, x[:0], y[:0], ball_speed_mps, "at least one landing"),
            (
                hitter_xy,
                np.array([nan, 0.0]),
                y,
                ball_speed_mps,
                "Every landing x must be finite",
            ),
            (
                hitter_xy,
                x,
                np.array([2.0, inf]),
                ball_speed_mps,
                "Every landing y must be finite",
            ),
            (hitter_xy, x, y, 0.0, "ball_speed_mps must be positive"),
            (hitter_xy, x, y, -25.0, "ball_speed_mps must be positive"),
            (hitter_xy, x, y, nan, "ball_speed_mps=nan must be finite"),
            (hitter_xy, x, y, inf, "ball_speed_mps=inf must be finite"),
        ]
        # Run test and check outputs.
        for bad_hitter_xy, bad_x, bad_y, bad_speed, expected in bad_inputs:
            with self.subTest(expected=expected):
                with self.assertRaises(AssertionError) as cm:
                    racket_shot_probability_utils.compute_flight_time(
                        bad_hitter_xy, bad_x, bad_y, bad_speed
                    )
                self.assertIn(expected, str(cm.exception))


# #############################################################################
# Test_compute_is_reached
# #############################################################################


class Test_compute_is_reached(hunitest.TestCase):
    """
    Test `racket_shot_probability_utils.compute_is_reached()`.
    """

    def helper(
        self,
        player_xy: Tuple[float, float],
        x: np.ndarray,
        y: np.ndarray,
        flight_time_s: np.ndarray,
        reaction_time_s: float,
        move_speed_mps: float,
        expected: str,
    ) -> None:
        """
        Test helper for `compute_is_reached()`.

        :param player_xy: where player 1 stands
        :param x: landing `x` of each sample
        :param y: landing `y` of each sample
        :param flight_time_s: flight time of each sample in seconds
        :param reaction_time_s: reaction time `t_r` in seconds
        :param move_speed_mps: running speed `v_p` in m/s
        :param expected: expected reached flags, as the string of a (nested)
            list
        """
        # Run test.
        actual = racket_shot_probability_utils.compute_is_reached(
            player_xy, x, y, flight_time_s, reaction_time_s, move_speed_mps
        )
        # Check outputs.
        self.assert_equal(str(actual.tolist()), expected)

    def test1(self) -> None:
        """
        Test the hand-computed reach of 0.9 m for `T = 0.8 s`, `t_r = 0.2 s`,
        `v_p = 1.5 m/s`.
        """
        # Prepare inputs.
        # Reach: 1.5 * (0.8 - 0.2) = 0.9 m, so the landing points 0.5 m and
        # 0.89 m beside player 1 are reached, and 0.91 m and 2 m are not.
        player_xy = (0.0, -11.885)
        x = np.array([0.5, 0.89, 0.91, 2.0])
        y = np.full(4, -11.885)
        flight_time_s = np.full(4, 0.8)
        reaction_time_s = 0.2
        move_speed_mps = 1.5
        # Prepare outputs.
        expected = "[True, True, False, False]"
        # Run test and check outputs.
        self.helper(
            player_xy,
            x,
            y,
            flight_time_s,
            reaction_time_s,
            move_speed_mps,
            expected,
        )

    def test2(self) -> None:
        """
        Test that a landing point exactly on the edge of the reach is reached.
        """
        # Prepare inputs.
        # Values exact in binary floating point: reach 2.0 * (1.5 - 0.5) = 2 m,
        # and the first landing point is exactly 2 m away.
        player_xy = (0.0, -11.885)
        x = np.array([2.0, 2.000001])
        y = np.full(2, -11.885)
        flight_time_s = np.full(2, 1.5)
        reaction_time_s = 0.5
        move_speed_mps = 2.0
        # Prepare outputs.
        expected = "[True, False]"
        # Run test and check outputs.
        self.helper(
            player_xy,
            x,
            y,
            flight_time_s,
            reaction_time_s,
            move_speed_mps,
            expected,
        )

    def test3(self) -> None:
        """
        Test that a ball landing before player 1 reacts (`T < t_r`) is never
        reached, even 1 cm away.
        """
        # Prepare inputs.
        player_xy = (0.0, -11.885)
        x = np.array([0.01, 0.5, 5.0])
        y = np.full(3, -11.885)
        flight_time_s = np.full(3, 0.1)
        reaction_time_s = 0.2
        move_speed_mps = 1.5
        # Prepare outputs.
        expected = "[False, False, False]"
        # Run test and check outputs.
        self.helper(
            player_xy,
            x,
            y,
            flight_time_s,
            reaction_time_s,
            move_speed_mps,
            expected,
        )

    def test4(self) -> None:
        """
        Test that a player 1 who does not move (`v_p = 0`) reaches nothing,
        even 1 cm away.
        """
        # Prepare inputs.
        player_xy = (0.0, -11.885)
        x = np.array([0.01, 0.5])
        y = np.full(2, -11.885)
        flight_time_s = np.full(2, 0.8)
        reaction_time_s = 0.2
        move_speed_mps = 0.0
        # Prepare outputs.
        expected = "[False, False]"
        # Run test and check outputs.
        self.helper(
            player_xy,
            x,
            y,
            flight_time_s,
            reaction_time_s,
            move_speed_mps,
            expected,
        )

    def test5(self) -> None:
        """
        Test that a ball landing exactly on player 1 is reached.
        """
        # Prepare inputs.
        player_xy = (0.0, -11.885)
        x = np.array([0.0])
        y = np.array([-11.885])
        flight_time_s = np.array([0.8])
        reaction_time_s = 0.2
        move_speed_mps = 1.5
        # Prepare outputs.
        expected = "[True]"
        # Run test and check outputs.
        self.helper(
            player_xy,
            x,
            y,
            flight_time_s,
            reaction_time_s,
            move_speed_mps,
            expected,
        )

    def test6(self) -> None:
        """
        Test that a ball landing exactly on player 1 counts as reached even
        before they react (`T < t_r`).

        The agreed formula gives a reach of 0 and a distance of 0, and
        `0 <= 0`; a Gaussian landing does this with probability 0, so it
        never changes a probability.
        """
        # Prepare inputs.
        player_xy = (0.0, -11.885)
        x = np.array([0.0])
        y = np.array([-11.885])
        flight_time_s = np.array([0.1])
        reaction_time_s = 0.2
        move_speed_mps = 1.5
        # Prepare outputs.
        expected = "[True]"
        # Run test and check outputs.
        self.helper(
            player_xy,
            x,
            y,
            flight_time_s,
            reaction_time_s,
            move_speed_mps,
            expected,
        )

    def test7(self) -> None:
        """
        Test that a very fast ball (1000 m/s) is never reached, even 5 cm from
        player 1.
        """
        # Prepare inputs.
        # From a serve landing point, every flight time on player 1's half is
        # under 0.02 s, well below `t_r`.
        hitter_xy = (-2.0, 4.8)
        player_xy = (0.0, -11.885)
        x = np.array([0.05, 4.0, -4.0, 0.0])
        y = np.array([-11.885, -11.0, -1.0, -6.0])
        ball_speed_mps = 1000.0
        flight_time_s = racket_shot_probability_utils.compute_flight_time(
            hitter_xy, x, y, ball_speed_mps
        )
        reaction_time_s = 0.2
        move_speed_mps = 1.5
        # Prepare outputs.
        expected = "[False, False, False, False]"
        # Run test and check outputs.
        self.helper(
            player_xy,
            x,
            y,
            flight_time_s,
            reaction_time_s,
            move_speed_mps,
            expected,
        )

    def test8(self) -> None:
        """
        Test that a very slow ball (0.1 m/s) is reached everywhere on player
        1's half, corners included.
        """
        # Prepare inputs.
        # The landing point nearest to player 2, the net corner (-4.115, 0),
        # is 5.2 m away, so every flight time is over 52 s and player 1 covers
        # over 77 m, more than the 12.6 m to the farthest corner of their half.
        hitter_xy = (-2.0, 4.8)
        player_xy = (0.0, -11.885)
        x = np.array([-4.115, 4.115, -4.115, 4.115, 0.0])
        y = np.array([0.0, 0.0, -11.885, -11.885, -6.0])
        ball_speed_mps = 0.1
        flight_time_s = racket_shot_probability_utils.compute_flight_time(
            hitter_xy, x, y, ball_speed_mps
        )
        reaction_time_s = 0.2
        move_speed_mps = 1.5
        # Prepare outputs.
        expected = "[True, True, True, True, True]"
        # Run test and check outputs.
        self.helper(
            player_xy,
            x,
            y,
            flight_time_s,
            reaction_time_s,
            move_speed_mps,
            expected,
        )

    def test9(self) -> None:
        """
        Test that a 2D array of landing points gives flags of the same shape,
        with a different flight time per landing point.
        """
        # Prepare inputs.
        # Row 1 uses the reach of `test1()` (0.9 m); row 2 has a longer flight
        # (1.2 s, reach 1.5 m) and a ball that lands before player 1 reacts.
        player_xy = (0.0, -11.885)
        x = np.array([[0.5, 2.0], [1.4, 0.01]])
        y = np.full((2, 2), -11.885)
        flight_time_s = np.array([[0.8, 0.8], [1.2, 0.1]])
        reaction_time_s = 0.2
        move_speed_mps = 1.5
        # Prepare outputs.
        expected = "[[True, False], [True, False]]"
        # Run test and check outputs.
        self.helper(
            player_xy,
            x,
            y,
            flight_time_s,
            reaction_time_s,
            move_speed_mps,
            expected,
        )

    def test10(self) -> None:
        """
        Test that a bad point, flight time, reaction time, or speed raises a
        clear error.
        """
        # Prepare inputs.
        nan, inf = float("nan"), float("inf")
        player_xy = (0.0, -11.885)
        x = np.array([0.5, 2.0])
        y = np.full(2, -11.885)
        flight_time_s = np.full(2, 0.8)
        reaction_time_s = 0.2
        move_speed_mps = 1.5
        bad_inputs = [
            (
                (0.0, nan),
                x,
                flight_time_s,
                reaction_time_s,
                move_speed_mps,
                "player_xy=(0.0, nan) must be finite",
            ),
            (
                player_xy,
                np.array([nan, 2.0]),
                flight_time_s,
                reaction_time_s,
                move_speed_mps,
                "Every landing x must be finite",
            ),
            (
                player_xy,
                x,
                [0.8, 0.8],
                reaction_time_s,
                move_speed_mps,
                "flight_time_s must be a numpy array",
            ),
            (
                player_xy,
                x,
                flight_time_s[:1],
                reaction_time_s,
                move_speed_mps,
                "one time per landing point",
            ),
            (
                player_xy,
                x,
                np.array([0.8, nan]),
                reaction_time_s,
                move_speed_mps,
                "Every flight time must be finite",
            ),
            (
                player_xy,
                x,
                np.array([0.8, -0.1]),
                reaction_time_s,
                move_speed_mps,
                "Every flight time must be non-negative",
            ),
            (
                player_xy,
                x,
                flight_time_s,
                -0.2,
                move_speed_mps,
                "reaction_time_s must be non-negative",
            ),
            (
                player_xy,
                x,
                flight_time_s,
                nan,
                move_speed_mps,
                "reaction_time_s=nan must be finite",
            ),
            (
                player_xy,
                x,
                flight_time_s,
                reaction_time_s,
                -1.5,
                "move_speed_mps must be non-negative",
            ),
            (
                player_xy,
                x,
                flight_time_s,
                reaction_time_s,
                inf,
                "move_speed_mps=inf must be finite",
            ),
        ]
        # Run test and check outputs.
        for (
            bad_player_xy,
            bad_x,
            bad_t,
            bad_t_r,
            bad_v_p,
            expected,
        ) in bad_inputs:
            with self.subTest(expected=expected):
                with self.assertRaises(AssertionError) as cm:
                    racket_shot_probability_utils.compute_is_reached(
                        bad_player_xy, bad_x, y, bad_t, bad_t_r, bad_v_p
                    )
                self.assertIn(expected, str(cm.exception))


# #############################################################################
# Test_compute_winner_probability
# #############################################################################


class Test_compute_winner_probability(hunitest.TestCase):
    """
    Test `racket_shot_probability_utils.compute_winner_probability()`.
    """

    def helper(
        self,
        serve_xy: Tuple[float, float],
        target_xy: Tuple[float, float],
        std_xy: Tuple[float, float],
        ball_speed_mps: float,
        reaction_time_s: float,
        move_speed_mps: float,
        court: racket_params.CourtGeometry,
        n_samples: int,
        seed: int,
    ) -> Dict[str, float]:
        """
        Run `compute_winner_probability()` with a generator seeded by `seed`.

        :param serve_xy: where player 2 hits from, `(x1, y1)`
        :param target_xy: aim point `(x2, y2)`
        :param std_xy: standard deviations `(std_x2, std_y2)`
        :param ball_speed_mps: speed of the return in m/s
        :param reaction_time_s: reaction time `t_r` in seconds
        :param move_speed_mps: running speed `v_p` in m/s
        :param court: court geometry
        :param n_samples: number of Monte Carlo samples
        :param seed: seed of `np.random.default_rng()`
        :return: the result of `compute_winner_probability()`
        """
        rng = np.random.default_rng(seed)
        actual: Dict[str, float] = (
            racket_shot_probability_utils.compute_winner_probability(
                serve_xy,
                target_xy,
                std_xy,
                ball_speed_mps,
                reaction_time_s,
                move_speed_mps,
                court,
                n_samples,
                rng,
            )
        )
        return actual

    def helper2(
        self,
        serve_xy: Tuple[float, float],
        target_xy: Tuple[float, float],
        std_xy: Tuple[float, float],
        ball_speed_mps: float,
        reaction_time_s: float,
        move_speed_mps: float,
        court: racket_params.CourtGeometry,
        n_samples: int,
        seed: int,
    ) -> None:
        """
        Check that the result is the joint event "in AND not reached" on the
        same samples, rebuilt here step by step from the building blocks.

        Each sample gets its own flight time, the in region is player 1's half,
        and player 1 stands at the center of the baseline.

        The parameters are the same as in `helper()`.
        """
        # Prepare outputs.
        # Same seed, so the same landing points as inside the function.
        rng = np.random.default_rng(seed)
        x, y = racket_shot_probability_utils.sample_landings(
            target_xy, std_xy, n_samples, rng
        )
        region = racket_shot_probability_utils.get_striker_half_region(court)
        is_in = region.contains(x, y)
        flight_time_s = racket_shot_probability_utils.compute_flight_time(
            serve_xy, x, y, ball_speed_mps
        )
        player1_xy = racket_shot_probability_utils.get_player1_xy(court)
        is_reached = racket_shot_probability_utils.compute_is_reached(
            player1_xy, x, y, flight_time_s, reaction_time_s, move_speed_mps
        )
        p_winner = float((is_in & ~is_reached).mean())
        expected = {
            "p_winner": p_winner,
            "p_winner_se": float(np.sqrt(p_winner * (1 - p_winner) / n_samples)),
            "p_in": float(is_in.mean()),
            "p_reached": float(is_reached.mean()),
        }
        # Run test.
        actual = self.helper(
            serve_xy,
            target_xy,
            std_xy,
            ball_speed_mps,
            reaction_time_s,
            move_speed_mps,
            court,
            n_samples,
            seed,
        )
        # Check outputs.
        self.assert_equal(str(actual), str(expected))
        # The reach must matter in these cases, or the check above could not
        # tell the joint event from "in" alone or from a product.
        self.assertLess(actual["p_winner"], actual["p_in"])

    def helper3(self, target_xy: Tuple[float, float], expected: float) -> None:
        """
        Check that a tiny spread gives the outcome of the aim point itself, 0
        or 1, as a direct check of that single point says.

        The serve, the ball, and player 1 are the tennis defaults: serve to
        `(-2.1, 3.2)`, 25 m/s, `t_r = 0.2 s`, `v_p = 1.5 m/s`.

        :param target_xy: aim point `(x2, y2)`, where every sample lands
        :param expected: expected `P(winner)`, 0.0 or 1.0
        """
        # Prepare inputs.
        serve_xy = (-2.1, 3.2)
        std_xy = (1e-6, 1e-6)
        ball_speed_mps = 25.0
        reaction_time_s = 0.2
        move_speed_mps = 1.5
        court = racket_params.TENNIS.court
        n_samples = 100
        seed = 1
        # Prepare outputs.
        # Direct check of the aim point alone, with the building blocks.
        x = np.array([target_xy[0]])
        y = np.array([target_xy[1]])
        region = racket_shot_probability_utils.get_striker_half_region(court)
        flight_time_s = racket_shot_probability_utils.compute_flight_time(
            serve_xy, x, y, ball_speed_mps
        )
        player1_xy = racket_shot_probability_utils.get_player1_xy(court)
        is_reached = racket_shot_probability_utils.compute_is_reached(
            player1_xy, x, y, flight_time_s, reaction_time_s, move_speed_mps
        )
        direct = float((region.contains(x, y) & ~is_reached)[0])
        # Run test.
        result = self.helper(
            serve_xy,
            target_xy,
            std_xy,
            ball_speed_mps,
            reaction_time_s,
            move_speed_mps,
            court,
            n_samples,
            seed,
        )
        # Check outputs.
        actual = [result["p_winner"], direct]
        self.assert_equal(str(actual), str([expected, expected]))

    def test1(self) -> None:
        """
        Test that a tennis return with a large reach is the joint event of the
        same samples.
        """
        # Prepare inputs.
        # A slow ball (8 m/s) and a fast player 1 (3 m/s) give a reach of a
        # few meters, so many samples are reached, in and out.
        serve_xy = (-2.1, 3.2)
        target_xy = (0.0, -10.0)
        std_xy = (2.0, 2.0)
        ball_speed_mps = 8.0
        reaction_time_s = 0.2
        move_speed_mps = 3.0
        court = racket_params.TENNIS.court
        n_samples = 2000
        seed = 1
        # Run test and check outputs.
        self.helper2(
            serve_xy,
            target_xy,
            std_xy,
            ball_speed_mps,
            reaction_time_s,
            move_speed_mps,
            court,
            n_samples,
            seed,
        )

    def test2(self) -> None:
        """
        Test that a pickleball return with a large reach is the joint event of
        the same samples.
        """
        # Prepare inputs.
        serve_xy = (-1.5, 4.4)
        target_xy = (0.0, -5.5)
        std_xy = (1.0, 1.0)
        ball_speed_mps = 6.0
        reaction_time_s = 0.2
        move_speed_mps = 3.0
        court = racket_params.PICKLEBALL.court
        n_samples = 2000
        seed = 1
        # Run test and check outputs.
        self.helper2(
            serve_xy,
            target_xy,
            std_xy,
            ball_speed_mps,
            reaction_time_s,
            move_speed_mps,
            court,
            n_samples,
            seed,
        )

    def test3(self) -> None:
        """
        Test that a player 1 who does not move (`v_p = 0`) gives a `P(winner)`
        equal to the in fraction of the same samples, exactly.
        """
        # Prepare inputs.
        serve_xy = (-2.1, 3.2)
        target_xy = (0.0, -11.0)
        std_xy = (1.0, 1.0)
        ball_speed_mps = 25.0
        reaction_time_s = 0.2
        move_speed_mps = 0.0
        court = racket_params.TENNIS.court
        n_samples = 2000
        seed = 1
        # Prepare outputs.
        rng = np.random.default_rng(seed)
        x, y = racket_shot_probability_utils.sample_landings(
            target_xy, std_xy, n_samples, rng
        )
        region = racket_shot_probability_utils.get_striker_half_region(court)
        p_in = float(region.contains(x, y).mean())
        expected = {"p_winner": p_in, "p_in": p_in, "p_reached": 0.0}
        # Run test.
        result = self.helper(
            serve_xy,
            target_xy,
            std_xy,
            ball_speed_mps,
            reaction_time_s,
            move_speed_mps,
            court,
            n_samples,
            seed,
        )
        # Check outputs.
        actual = {k: result[k] for k in ("p_winner", "p_in", "p_reached")}
        self.assert_equal(str(actual), str(expected))

    def test4(self) -> None:
        """
        Test that a very slow ball (0.1 m/s), which player 1 reaches anywhere
        on their half, gives `P(winner) = 0` with a standard error of 0.
        """
        # Prepare inputs.
        # Every in sample is at least 3.2 m from player 2, so it flies over
        # 32 s and player 1 covers over 47 m, more than their whole half.
        serve_xy = (-2.1, 3.2)
        target_xy = (0.0, -6.0)
        std_xy = (2.0, 3.0)
        ball_speed_mps = 0.1
        reaction_time_s = 0.2
        move_speed_mps = 1.5
        court = racket_params.TENNIS.court
        n_samples = 2000
        seed = 1
        # Prepare outputs.
        expected = {"p_winner": 0.0, "p_winner_se": 0.0}
        # Run test.
        result = self.helper(
            serve_xy,
            target_xy,
            std_xy,
            ball_speed_mps,
            reaction_time_s,
            move_speed_mps,
            court,
            n_samples,
            seed,
        )
        # Check outputs.
        actual = {k: result[k] for k in ("p_winner", "p_winner_se")}
        self.assert_equal(str(actual), str(expected))

    def test5(self) -> None:
        """
        Test that a target far from player 1 gives a `P(winner)` close to
        `P(in)`.
        """
        # Prepare inputs.
        # Near the net, about 10 m from player 1, who covers under 1 m with
        # the tennis defaults: no sample is reached.
        serve_xy = (-2.1, 3.2)
        target_xy = (3.0, -2.0)
        std_xy = (0.5, 1.0)
        ball_speed_mps = 25.0
        reaction_time_s = 0.2
        move_speed_mps = 1.5
        court = racket_params.TENNIS.court
        n_samples = 2000
        seed = 1
        # Run test.
        result = self.helper(
            serve_xy,
            target_xy,
            std_xy,
            ball_speed_mps,
            reaction_time_s,
            move_speed_mps,
            court,
            n_samples,
            seed,
        )
        # Check outputs.
        self.assertAlmostEqual(result["p_winner"], result["p_in"], delta=0.001)

    def test6(self) -> None:
        """
        Test that the standard error is `sqrt(p (1 - p) / n)` of the returned
        `P(winner)`.
        """
        # Prepare inputs.
        serve_xy = (-2.1, 3.2)
        target_xy = (0.0, -10.0)
        std_xy = (2.0, 2.0)
        ball_speed_mps = 8.0
        reaction_time_s = 0.2
        move_speed_mps = 3.0
        court = racket_params.TENNIS.court
        n_samples = 2000
        seed = 1
        # Run test.
        result = self.helper(
            serve_xy,
            target_xy,
            std_xy,
            ball_speed_mps,
            reaction_time_s,
            move_speed_mps,
            court,
            n_samples,
            seed,
        )
        # Check outputs.
        # Here `P(winner)` is 0.2155, far from 0 and 1, so a wrong formula
        # shows up.
        p = result["p_winner"]
        expected = float(np.sqrt(p * (1 - p) / n_samples))
        self.assertEqual(result["p_winner_se"], expected)

    def test7(self) -> None:
        """
        Test that the same seed gives identical results, and another seed
        different ones.
        """
        # Prepare inputs.
        serve_xy = (-2.1, 3.2)
        target_xy = (0.0, -10.0)
        std_xy = (2.0, 2.0)
        ball_speed_mps = 8.0
        reaction_time_s = 0.2
        move_speed_mps = 3.0
        court = racket_params.TENNIS.court
        n_samples = 2000
        args = (
            serve_xy,
            target_xy,
            std_xy,
            ball_speed_mps,
            reaction_time_s,
            move_speed_mps,
            court,
            n_samples,
        )
        seed = 7
        other_seed = 8
        # Run test.
        result1 = self.helper(*args, seed)
        result2 = self.helper(*args, seed)
        result3 = self.helper(*args, other_seed)
        # Check outputs.
        self.assert_equal(str(result1), str(result2))
        self.assertNotEqual(str(result1), str(result3))

    def test8(self) -> None:
        """
        Test that a tiny spread aimed in and far from player 1 gives 1.
        """
        # Prepare inputs.
        target_xy = (3.0, -6.0)
        # Prepare outputs.
        expected = 1.0
        # Run test and check outputs.
        self.helper3(target_xy, expected)

    def test9(self) -> None:
        """
        Test that a tiny spread aimed 0.385 m in front of player 1 gives 0.
        """
        # Prepare inputs.
        # By hand: the ball flies hypot(2.1, 14.7) / 25 = 0.594 s, so player 1
        # covers 1.5 * (0.594 - 0.2) = 0.591 m, more than the 0.385 m to
        # `(0, -11.5)`: reached. From the Milestone 1 serve stance
        # `(0.5, -12.185)` it would be 0.848 m away, so not reached.
        target_xy = (0.0, -11.5)
        # Prepare outputs.
        expected = 0.0
        # Run test and check outputs.
        self.helper3(target_xy, expected)

    def test10(self) -> None:
        """
        Test that a tiny spread aimed behind the baseline (out) gives 0.
        """
        # Prepare inputs.
        target_xy = (0.0, -13.0)
        # Prepare outputs.
        expected = 0.0
        # Run test and check outputs.
        self.helper3(target_xy, expected)

    def test11(self) -> None:
        """
        Test the result for the tennis defaults of the Cell 1.1 widget.
        """
        # Prepare inputs.
        serve_xy = (-2.1, 3.2)
        target_xy = (2.1, -10.4)
        std_xy = (0.5, 1.0)
        ball_speed_mps = 25.0
        reaction_time_s = 0.2
        move_speed_mps = 1.5
        court = racket_params.TENNIS.court
        n_samples = 10000
        seed = 1
        # Prepare outputs.
        # Player 1 covers about 0.6 m and the target is 2.6 m away, so almost
        # every in return is a winner; the 2 reached samples are out.
        expected = (
            "{'p_winner': 0.9307, 'p_winner_se': 0.00254, 'p_in': 0.9307, "
            "'p_reached': 0.0002}"
        )
        # Run test.
        result = self.helper(
            serve_xy,
            target_xy,
            std_xy,
            ball_speed_mps,
            reaction_time_s,
            move_speed_mps,
            court,
            n_samples,
            seed,
        )
        # Check outputs.
        actual = str({k: round(v, 6) for k, v in result.items()})
        self.assert_equal(actual, expected)

    def test12(self) -> None:
        """
        Test the result for the pickleball defaults of the Cell 1.1 widget,
        with a 12 m/s ball.
        """
        # Prepare inputs.
        serve_xy = (-1.5, 4.4)
        target_xy = (1.5, -5.2)
        std_xy = (0.5, 1.0)
        ball_speed_mps = 12.0
        reaction_time_s = 0.2
        move_speed_mps = 1.5
        court = racket_params.PICKLEBALL.court
        n_samples = 10000
        seed = 1
        # Prepare outputs.
        # The smaller court puts the target closer to player 1, so about 3%
        # of the in returns are reached.
        expected = (
            "{'p_winner': 0.9044, 'p_winner_se': 0.00294, 'p_in': 0.9323, "
            "'p_reached': 0.0411}"
        )
        # Run test.
        result = self.helper(
            serve_xy,
            target_xy,
            std_xy,
            ball_speed_mps,
            reaction_time_s,
            move_speed_mps,
            court,
            n_samples,
            seed,
        )
        # Check outputs.
        actual = str({k: round(v, 6) for k, v in result.items()})
        self.assert_equal(actual, expected)

    def test13(self) -> None:
        """
        Test that a bad serve point, generator, or any other bad input raises a
        clear error.
        """
        # Prepare inputs.
        nan, inf = float("nan"), float("inf")
        court = racket_params.TENNIS.court
        # The arguments in the order of the signature, passed positionally.
        good = {
            "serve_xy": (-2.1, 3.2),
            "target_xy": (2.1, -10.4),
            "std_xy": (0.5, 1.0),
            "ball_speed_mps": 25.0,
            "reaction_time_s": 0.2,
            "move_speed_mps": 1.5,
            "court": court,
            "n_samples": 100,
            "rng": np.random.default_rng(1),
        }
        bad_inputs = [
            ("serve_xy", (nan, 3.2), "serve_xy=(nan, 3.2) must be finite"),
            ("serve_xy", (-2.1, -1.0), "must land on player 2's half"),
            ("serve_xy", (-2.1, 0.0), "must land on player 2's half"),
            ("rng", 1, "rng must be a numpy random Generator"),
            ("target_xy", (nan, -10.4), "target_xy=(nan, -10.4) must be"),
            ("std_xy", (0.0, 1.0), "std_x2 must be positive"),
            ("n_samples", 0, "n_samples must be positive"),
            ("ball_speed_mps", 0.0, "ball_speed_mps must be positive"),
            ("reaction_time_s", -0.2, "reaction_time_s must be non-negative"),
            ("move_speed_mps", -1.5, "move_speed_mps must be non-negative"),
            (
                "court",
                dataclasses.replace(court, length_m=inf),
                "length_m=inf must be finite",
            ),
        ]
        # Run test and check outputs.
        for name, bad_value, expected in bad_inputs:
            with self.subTest(name=name, value=bad_value):
                kwargs = {**good, name: bad_value}
                with self.assertRaises(AssertionError) as cm:
                    racket_shot_probability_utils.compute_winner_probability(
                        *kwargs.values()
                    )
                self.assertIn(expected, str(cm.exception))


# #############################################################################
# Test_compute_winner_heatmap
# #############################################################################


class Test_compute_winner_heatmap(hunitest.TestCase):
    """
    Test `racket_shot_probability_utils.compute_winner_heatmap()`.
    """

    def helper(
        self,
        ball_speed_mps: float,
        move_speed_mps: float,
        n_x: int,
        n_y: int,
        n_samples: int,
        seed: int,
    ) -> pd.DataFrame:
        """
        Run `compute_winner_heatmap()` on the tennis court, with the Cell 1.1
        serve `(-2.1, 3.2)`, stds `(1.0, 1.0)`, and `t_r = 0.2 s`.

        :param ball_speed_mps: speed of the return in m/s
        :param move_speed_mps: running speed `v_p` in m/s
        :param n_x: number of grid columns
        :param n_y: number of grid rows
        :param n_samples: Monte Carlo samples per target
        :param seed: seed shared by every target
        :return: the heatmap
        """
        serve_xy = (-2.1, 3.2)
        std_xy = (1.0, 1.0)
        reaction_time_s = 0.2
        court = racket_params.TENNIS.court
        heatmap: pd.DataFrame = (
            racket_shot_probability_utils.compute_winner_heatmap(
                serve_xy,
                std_xy,
                ball_speed_mps,
                reaction_time_s,
                move_speed_mps,
                court,
                n_x,
                n_y,
                n_samples,
                seed,
            )
        )
        return heatmap

    def test1(self) -> None:
        """
        Test that each cell equals `compute_winner_probability()` called
        directly for its target with the same seed, exactly.
        """
        # Prepare inputs.
        # A slow ball and a fast player 1, so the cells differ.
        ball_speed_mps = 8.0
        move_speed_mps = 3.0
        n_x = 3
        n_y = 4
        n_samples = 1500
        seed = 5
        # Prepare outputs.
        # The same serve, stds, `t_r`, and court as in `helper()`.
        serve_xy = (-2.1, 3.2)
        std_xy = (1.0, 1.0)
        reaction_time_s = 0.2
        court = racket_params.TENNIS.court
        region = racket_shot_probability_utils.get_striker_half_region(court)
        grid = racket_scoring.make_target_grid(region, n_x, n_y)
        expected = []
        for target_xy in zip(grid["x_m"], grid["y_m"]):
            rng = np.random.default_rng(seed)
            expected.append(
                racket_shot_probability_utils.compute_winner_probability(
                    serve_xy,
                    target_xy,
                    std_xy,
                    ball_speed_mps,
                    reaction_time_s,
                    move_speed_mps,
                    court,
                    n_samples,
                    rng,
                )
            )
        # Run test.
        heatmap = self.helper(
            ball_speed_mps, move_speed_mps, n_x, n_y, n_samples, seed
        )
        # Check outputs.
        columns = ["p_winner", "p_winner_se", "p_in", "p_reached"]
        actual = heatmap[columns].to_dict("records")
        self.assert_equal(str(actual), str(expected))

    def test2(self) -> None:
        """
        Test that the targets are the cell centers of a grid over player 1's
        half, `y` outer from the baseline, `x` inner.
        """
        # Prepare inputs.
        # 2 columns and 3 rows, so swapping them changes the result.
        ball_speed_mps = 25.0
        move_speed_mps = 1.5
        n_x = 2
        n_y = 3
        n_samples = 1500
        seed = 1
        # Prepare outputs.
        # x: -4.115 to 4.115 in 2 cells; y: -11.885 to 0 in 3 cells.
        expected = (
            "[(-2.0575, -9.904167), (2.0575, -9.904167), (-2.0575, -5.9425), "
            "(2.0575, -5.9425), (-2.0575, -1.980833), (2.0575, -1.980833)]"
        )
        # Run test.
        heatmap = self.helper(
            ball_speed_mps, move_speed_mps, n_x, n_y, n_samples, seed
        )
        # Check outputs.
        actual = str(
            [
                (round(x, 6), round(y, 6))
                for x, y in zip(heatmap["x_m"], heatmap["y_m"])
            ]
        )
        self.assert_equal(actual, expected)

    def test3(self) -> None:
        """
        Test the columns of the heatmap.
        """
        # Prepare inputs.
        ball_speed_mps = 25.0
        move_speed_mps = 1.5
        n_x = 2
        n_y = 2
        n_samples = 1500
        seed = 1
        # Prepare outputs.
        expected = (
            "['cell_id', 'x_m', 'y_m', 'x_min', 'x_max', 'y_min', 'y_max', "
            "'p_winner', 'p_winner_se', 'p_in', 'p_reached']"
        )
        # Run test.
        heatmap = self.helper(
            ball_speed_mps, move_speed_mps, n_x, n_y, n_samples, seed
        )
        # Check outputs.
        self.assert_equal(str(list(heatmap.columns)), expected)

    def test4(self) -> None:
        """
        Test that `v_p = 0` gives a map equal to the in fraction map.
        """
        # Prepare inputs.
        ball_speed_mps = 25.0
        move_speed_mps = 0.0
        n_x = 4
        n_y = 5
        n_samples = 1500
        seed = 1
        # Run test.
        heatmap = self.helper(
            ball_speed_mps, move_speed_mps, n_x, n_y, n_samples, seed
        )
        # Check outputs.
        actual = str(heatmap["p_winner"].tolist())
        expected = str(heatmap["p_in"].tolist())
        self.assert_equal(actual, expected)

    def test5(self) -> None:
        """
        Test that a very slow ball (0.01 m/s) gives an all zero map.
        """
        # Prepare inputs.
        ball_speed_mps = 0.01
        move_speed_mps = 1.5
        n_x = 4
        n_y = 5
        n_samples = 1500
        seed = 1
        # Prepare outputs.
        expected = str([0.0] * (n_x * n_y))
        # Run test.
        heatmap = self.helper(
            ball_speed_mps, move_speed_mps, n_x, n_y, n_samples, seed
        )
        # Check outputs.
        self.assert_equal(str(heatmap["p_winner"].tolist()), expected)

    def test6(self) -> None:
        """
        Test that a faster player 1 never increases any cell, and lowers some.
        """
        # Prepare inputs.
        ball_speed_mps = 12.0
        slow_move_speed_mps = 1.5
        fast_move_speed_mps = 6.0
        n_x = 4
        n_y = 5
        n_samples = 1500
        seed = 1
        # Run test.
        slow = self.helper(
            ball_speed_mps, slow_move_speed_mps, n_x, n_y, n_samples, seed
        )
        fast = self.helper(
            ball_speed_mps, fast_move_speed_mps, n_x, n_y, n_samples, seed
        )
        # Check outputs.
        diff = fast["p_winner"] - slow["p_winner"]
        actual = {
            "max_increase": float(diff.max()),
            "lowered": bool(diff.min() < 0),
        }
        expected = {"max_increase": 0.0, "lowered": True}
        self.assert_equal(str(actual), str(expected))

    def test7(self) -> None:
        """
        Test that bad grid sizes or seeds raise a clear error.
        """
        # Prepare inputs.
        bad_inputs: List[Tuple[Dict[str, Any], str]] = [
            ({"n_x": 2.5}, "n_x must be an integer"),
            ({"n_y": 0}, "n_y must be positive"),
            ({"seed": -1}, "seed must be non-negative"),
            ({"seed": 1.5}, "seed must be an integer"),
            ({"n_samples": 0}, "n_samples must be positive"),
        ]
        # The arguments in the order of `helper()`, passed positionally.
        good: Dict[str, Any] = {
            "ball_speed_mps": 25.0,
            "move_speed_mps": 1.5,
            "n_x": 2,
            "n_y": 2,
            "n_samples": 1500,
            "seed": 1,
        }
        # Run test and check outputs.
        for bad, expected in bad_inputs:
            with self.subTest(bad=bad):
                kwargs = {**good, **bad}
                with self.assertRaises(AssertionError) as cm:
                    self.helper(*kwargs.values())
                self.assertIn(expected, str(cm.exception))


# #############################################################################
# Test_find_best_region
# #############################################################################


class Test_find_best_region(hunitest.TestCase):
    """
    Test `racket_shot_probability_utils.find_best_region()`.
    """

    def helper(
        self, p_winner: List[float], p_winner_se: List[float], expected: str
    ) -> None:
        """
        Test helper for `find_best_region()` on targets at `(i, -i)`.

        :param p_winner: `P(winner)` of each target, in grid order
        :param p_winner_se: its standard error
        :param expected: expected flags, as the string of a list
        """
        # Prepare inputs.
        n = len(p_winner)
        heatmap = pd.DataFrame(
            {
                "x_m": [float(i) for i in range(n)],
                "y_m": [float(-i) for i in range(n)],
                "p_winner": p_winner,
                "p_winner_se": p_winner_se,
            }
        )
        # Run test.
        actual = racket_shot_probability_utils.find_best_region(heatmap)
        # Check outputs.
        self.assert_equal(str(actual.tolist()), expected)

    def test1(self) -> None:
        """
        Test that a value within 2 standard errors of the highest is in, and
        values far below are out.
        """
        # Prepare inputs.
        # 0.948 is 0.002 below 0.95, under 2 * sqrt(0.002^2 + 0.002^2) =
        # 0.0057; 0.9 is 0.05 below, over 2 * sqrt(0.002^2 + 0.003^2) = 0.0072.
        p_winner = [0.9, 0.95, 0.948, 0.9]
        p_winner_se = [0.003, 0.002, 0.002, 0.003]
        # Prepare outputs.
        expected = "[False, True, True, False]"
        # Run test and check outputs.
        self.helper(p_winner, p_winner_se, expected)

    def test2(self) -> None:
        """
        Test that the threshold is 2 standard errors, not 1.
        """
        # Prepare inputs.
        # 0.942 is 0.008 below 0.95: over 1 * sqrt(0.004^2 + 0.003^2) =
        # 0.005, under 2 * 0.005 = 0.01.
        p_winner = [0.95, 0.942]
        p_winner_se = [0.004, 0.003]
        # Prepare outputs.
        expected = "[True, True]"
        # Run test and check outputs.
        self.helper(p_winner, p_winner_se, expected)

    def test3(self) -> None:
        """
        Test that with standard errors of 0 only the values equal to the
        highest are in.
        """
        # Prepare inputs.
        # 0.99 is 0.01 below 1.0, over 2 * sqrt(0^2 + 0.0014^2) = 0.0028.
        p_winner = [1.0, 1.0, 0.99, 1.0]
        p_winner_se = [0.0, 0.0, 0.0014, 0.0]
        # Prepare outputs.
        expected = "[True, True, False, True]"
        # Run test and check outputs.
        self.helper(p_winner, p_winner_se, expected)

    def test4(self) -> None:
        """
        Test that a single target is its own best region.
        """
        # Prepare inputs.
        p_winner = [0.3]
        p_winner_se = [0.01]
        # Prepare outputs.
        expected = "[True]"
        # Run test and check outputs.
        self.helper(p_winner, p_winner_se, expected)

    def test5(self) -> None:
        """
        Test that a bad heatmap raises a clear error.
        """
        # Prepare inputs.
        one_row = pd.DataFrame(
            {"x_m": [0.0], "y_m": [0.0], "p_winner": [1.0], "p_winner_se": [0.0]}
        )
        bad_inputs = [
            (one_row.iloc[:0], "There must be at least 1 target"),
            (one_row.drop(columns="p_winner"), "Missing heatmap column"),
            ([1.0, 2.0], "heatmap must be a pandas DataFrame"),
        ]
        # Run test and check outputs.
        for bad_heatmap, expected in bad_inputs:
            with self.subTest(expected=expected):
                with self.assertRaises(AssertionError) as cm:
                    racket_shot_probability_utils.find_best_region(bad_heatmap)
                self.assertIn(expected, str(cm.exception))


# #############################################################################
# Test_find_best_target
# #############################################################################


class Test_find_best_target(hunitest.TestCase):
    """
    Test `racket_shot_probability_utils.find_best_target()`.
    """

    def helper(
        self, p_winner: List[float], p_winner_se: List[float], expected: str
    ) -> None:
        """
        Test helper for `find_best_target()` on targets at `(i, -i)`.

        :param p_winner: `P(winner)` of each target, in grid order
        :param p_winner_se: its standard error
        :param expected: expected result, with values rounded to 6 decimals
        """
        # Prepare inputs.
        n = len(p_winner)
        heatmap = pd.DataFrame(
            {
                "x_m": [float(i) for i in range(n)],
                "y_m": [float(-i) for i in range(n)],
                "p_winner": p_winner,
                "p_winner_se": p_winner_se,
            }
        )
        # Run test.
        result = racket_shot_probability_utils.find_best_target(heatmap)
        # Check outputs.
        actual = str({k: round(v, 6) for k, v in result.items()})
        self.assert_equal(actual, expected)

    def test1(self) -> None:
        """
        Test a clear winner: a region of 1 target, with the gap to the second
        highest in standard errors.
        """
        # Prepare inputs.
        p_winner = [0.5, 0.9, 0.8]
        p_winner_se = [0.01, 0.003, 0.004]
        # Prepare outputs.
        # Gap: (0.9 - 0.8) / sqrt(0.003^2 + 0.004^2) = 0.1 / 0.005 = 20, so 0.8
        # is out of the region (over 2 standard errors).
        expected = (
            "{'x_m': 1.0, 'y_m': -1.0, 'p_winner': 0.9, 'p_winner_se': 0.003, "
            "'n_region_cells': 1, 'max_p_winner': 0.9, 'max_p_winner_se': 0.003, "
            "'second_p_winner': 0.8, 'second_p_winner_se': 0.004, "
            "'gap_in_se': 20.0}"
        )
        # Run test and check outputs.
        self.helper(p_winner, p_winner_se, expected)

    def test2(self) -> None:
        """
        Test that the marked target is the middle of a tied region, not its
        first cell.
        """
        # Prepare inputs.
        # 3 tied targets at (0, 0), (1, -1), (2, -2): the middle is (1, -1).
        p_winner = [0.95, 0.95, 0.95, 0.5]
        p_winner_se = [0.002, 0.002, 0.002, 0.01]
        # Prepare outputs.
        expected = (
            "{'x_m': 1.0, 'y_m': -1.0, 'p_winner': 0.95, 'p_winner_se': 0.002, "
            "'n_region_cells': 3, 'max_p_winner': 0.95, "
            "'max_p_winner_se': 0.002, 'second_p_winner': 0.95, "
            "'second_p_winner_se': 0.002, 'gap_in_se': 0.0}"
        )
        # Run test and check outputs.
        self.helper(p_winner, p_winner_se, expected)

    def test3(self) -> None:
        """
        Test that 2 targets equally close to the middle of the region go to
        the first in grid order.
        """
        # Prepare inputs.
        p_winner = [0.9, 0.95, 0.95]
        p_winner_se = [0.003, 0.002, 0.002]
        # Prepare outputs.
        expected = (
            "{'x_m': 1.0, 'y_m': -1.0, 'p_winner': 0.95, 'p_winner_se': 0.002, "
            "'n_region_cells': 2, 'max_p_winner': 0.95, "
            "'max_p_winner_se': 0.002, 'second_p_winner': 0.95, "
            "'second_p_winner_se': 0.002, 'gap_in_se': 0.0}"
        )
        # Run test and check outputs.
        self.helper(p_winner, p_winner_se, expected)

    def test4(self) -> None:
        """
        Test that 2 standard errors of 0 give a gap of `inf` for different
        probabilities.
        """
        # Prepare inputs.
        p_winner = [0.0, 1.0]
        p_winner_se = [0.0, 0.0]
        # Prepare outputs.
        expected = (
            "{'x_m': 1.0, 'y_m': -1.0, 'p_winner': 1.0, 'p_winner_se': 0.0, "
            "'n_region_cells': 1, 'max_p_winner': 1.0, 'max_p_winner_se': 0.0, "
            "'second_p_winner': 0.0, 'second_p_winner_se': 0.0, "
            "'gap_in_se': inf}"
        )
        # Run test and check outputs.
        self.helper(p_winner, p_winner_se, expected)

    def test5(self) -> None:
        """
        Test that a bad heatmap raises a clear error.
        """
        # Prepare inputs.
        one_row = pd.DataFrame(
            {"x_m": [0.0], "y_m": [0.0], "p_winner": [1.0], "p_winner_se": [0.0]}
        )
        bad_inputs = [
            (one_row, "There must be at least 2 targets to compare"),
            (one_row.drop(columns="p_winner_se"), "Missing heatmap column"),
            ([1.0, 2.0], "heatmap must be a pandas DataFrame"),
        ]
        # Run test and check outputs.
        for bad_heatmap, expected in bad_inputs:
            with self.subTest(expected=expected):
                with self.assertRaises(AssertionError) as cm:
                    racket_shot_probability_utils.find_best_target(bad_heatmap)
                self.assertIn(expected, str(cm.exception))


# #############################################################################
# Test__get_region_outline
# #############################################################################


class Test__get_region_outline(hunitest.TestCase):
    """
    Test `racket_shot_probability_utils._get_region_outline()`.
    """

    def helper(self, in_region: List[List[bool]], expected: str) -> None:
        """
        Test helper for `_get_region_outline()` on a 2 x 2 grid of 1 m cells.

        :param in_region: flags of the 2 rows (lowest `y` first) of 2 cells
        :param expected: expected segments, as a string
        """
        # Prepare inputs.
        x_edges = np.array([0.0, 1.0, 2.0])
        y_edges = np.array([0.0, 1.0, 2.0])
        # Run test.
        actual = racket_shot_probability_utils._get_region_outline(
            np.array(in_region), x_edges, y_edges
        )
        # Check outputs.
        self.assert_equal(str(actual), expected)

    def test1(self) -> None:
        """
        Test that a region of 1 cell has its 4 sides: left, right, bottom, top.
        """
        # Prepare inputs.
        in_region = [[True, False], [False, False]]
        # Prepare outputs.
        expected = (
            "[[(0.0, 0.0), (0.0, 1.0)], [(1.0, 0.0), (1.0, 1.0)], "
            "[(0.0, 0.0), (1.0, 0.0)], [(0.0, 1.0), (1.0, 1.0)]]"
        )
        # Run test and check outputs.
        self.helper(in_region, expected)

    def test2(self) -> None:
        """
        Test that the side shared by 2 cells of the region is not drawn.
        """
        # Prepare inputs.
        in_region = [[True, True], [False, False]]
        # Prepare outputs.
        # Left cell: left, bottom, top; right cell: right, bottom, top.
        expected = (
            "[[(0.0, 0.0), (0.0, 1.0)], [(0.0, 0.0), (1.0, 0.0)], "
            "[(0.0, 1.0), (1.0, 1.0)], [(2.0, 0.0), (2.0, 1.0)], "
            "[(1.0, 0.0), (2.0, 0.0)], [(1.0, 1.0), (2.0, 1.0)]]"
        )
        # Run test and check outputs.
        self.helper(in_region, expected)

    def test3(self) -> None:
        """
        Test that the whole grid is outlined by its border only, and an empty
        region has no outline.
        """
        # Prepare inputs.
        in_region_full = [[True, True], [True, True]]
        in_region_empty = [[False, False], [False, False]]
        # Prepare outputs.
        expected_full = (
            "[[(0.0, 0.0), (0.0, 1.0)], [(0.0, 0.0), (1.0, 0.0)], "
            "[(2.0, 0.0), (2.0, 1.0)], [(1.0, 0.0), (2.0, 0.0)], "
            "[(0.0, 1.0), (0.0, 2.0)], [(0.0, 2.0), (1.0, 2.0)], "
            "[(2.0, 1.0), (2.0, 2.0)], [(1.0, 2.0), (2.0, 2.0)]]"
        )
        expected_empty = "[]"
        # Run test and check outputs.
        self.helper(in_region_full, expected_full)
        self.helper(in_region_empty, expected_empty)


# #############################################################################
# Test_plot_winner_heatmap
# #############################################################################


class Test_plot_winner_heatmap(hunitest.TestCase):
    """
    Test `racket_shot_probability_utils.plot_winner_heatmap()`.
    """

    def helper(self) -> pd.DataFrame:
        """
        Build a 3 x 4 tennis heatmap with `P(winner)` from 0.2 to 0.8 in grid
        order, so the last cell is the best.

        :return: the heatmap
        """
        court = racket_params.TENNIS.court
        region = racket_shot_probability_utils.get_striker_half_region(court)
        heatmap = racket_scoring.make_target_grid(region, 3, 4)
        heatmap["p_winner"] = np.linspace(0.2, 0.8, len(heatmap))
        heatmap["p_winner_se"] = 0.01
        return heatmap

    def helper2(self, heatmap: pd.DataFrame) -> Axes:
        """
        Draw `heatmap` on new axes, with the serve at `(-2.1, 3.2)`.

        :param heatmap: heatmap to draw
        :return: the axes
        """
        court = racket_params.TENNIS.court
        serve_xy = (-2.1, 3.2)
        ax: Axes = plt.subplots()[1]
        racket_shot_probability_utils.plot_winner_heatmap(
            ax, heatmap, court, serve_xy
        )
        return ax

    def test1(self) -> None:
        """
        Test that the color scale is fixed from 0 to 1, whatever the values.
        """
        # Prepare inputs.
        heatmap = self.helper()
        # Prepare outputs.
        expected = "(0.0, 1.0)"
        # Run test.
        ax = self.helper2(heatmap)
        # Check outputs.
        mesh = [
            c for c in ax.collections if isinstance(c, mcollections.QuadMesh)
        ]
        actual = str(tuple(float(v) for v in mesh[0].get_clim()))
        plt.close("all")
        self.assert_equal(actual, expected)

    def test2(self) -> None:
        """
        Test that each cell is drawn at its place: one image row per `y`, from
        the baseline, and the cell edges of the grid.
        """
        # Prepare inputs.
        heatmap = self.helper()
        # Prepare outputs.
        expected = (
            "values=[[0.2, 0.254545, 0.309091], [0.363636, 0.418182, 0.472727], "
            "[0.527273, 0.581818, 0.636364], [0.690909, 0.745455, 0.8]]\n"
            "x_edges=[-4.115, -1.371667, 1.371667, 4.115]\n"
            "y_edges=[-11.885, -8.91375, -5.9425, -2.97125, 0.0]"
        )
        # Run test.
        ax = self.helper2(heatmap)
        # Check outputs.
        mesh = [
            c for c in ax.collections if isinstance(c, mcollections.QuadMesh)
        ]
        values = np.round(np.asarray(mesh[0].get_array()), 6).reshape(4, 3)
        coords = mesh[0].get_coordinates()
        x_edges = np.round(coords[0, :, 0], 6).tolist()
        y_edges = np.round(coords[:, 0, 1], 6).tolist()
        actual = (
            f"values={values.tolist()}\nx_edges={x_edges}\ny_edges={y_edges}"
        )
        plt.close("all")
        self.assert_equal(actual, expected)

    def test3(self) -> None:
        """
        Test the markers: the best target, player 1, and player 2.
        """
        # Prepare inputs.
        heatmap = self.helper()
        # Prepare outputs.
        # The best cell is the last one: x 1.371667 to 4.115, y -2.97125 to 0.
        expected = (
            "[('best target (middle of the region), P(winner) = 0.800 +- 0.010', "
            "[2.743333, -1.485625]), "
            "('player 1', [0.0, -11.885]), ('player 2', [-2.1, 3.2])]"
        )
        # Run test.
        ax = self.helper2(heatmap)
        # Check outputs.
        markers = [
            (line.get_label(), np.round(line.get_xydata()[0], 6).tolist())
            for line in ax.get_lines()
            if not str(line.get_label()).startswith("_")
        ]
        plt.close("all")
        self.assert_equal(str(markers), expected)

    def test4(self) -> None:
        """
        Test that a heatmap with a missing cell raises a clear error.
        """
        # Prepare inputs.
        heatmap = self.helper().iloc[:-1]
        # Prepare outputs.
        expected = "heatmap must hold every cell of its grid"
        # Run test.
        with self.assertRaises(AssertionError) as cm:
            self.helper2(heatmap)
        plt.close("all")
        # Check outputs.
        self.assertIn(expected, str(cm.exception))

    def test5(self) -> None:
        """
        Test the outline of the best region and the star in its middle.
        """
        # Prepare inputs.
        # The middle column of the 3 lowest rows (cells 1, 4, 7) ties at 1.0;
        # the other cells are far below.
        heatmap = self.helper()
        in_region = heatmap["cell_id"].isin([1, 4, 7])
        heatmap["p_winner"] = np.where(in_region, 1.0, 0.5)
        heatmap["p_winner_se"] = np.where(in_region, 0.0, 0.01)
        # Prepare outputs.
        # The 2 long sides of the column (3 segments each), its bottom, and
        # its top; the star at cell 4, the middle of the column.
        expected = (
            "label=best region: within 2 SE of the highest (3 cells)\n"
            "segments=[[(-1.371667, -11.885), (-1.371667, -8.91375)], "
            "[(1.371667, -11.885), (1.371667, -8.91375)], "
            "[(-1.371667, -11.885), (1.371667, -11.885)], "
            "[(-1.371667, -8.91375), (-1.371667, -5.9425)], "
            "[(1.371667, -8.91375), (1.371667, -5.9425)], "
            "[(-1.371667, -5.9425), (-1.371667, -2.97125)], "
            "[(1.371667, -5.9425), (1.371667, -2.97125)], "
            "[(-1.371667, -2.97125), (1.371667, -2.97125)]]\n"
            "star=[0.0, -7.428125]"
        )
        # Run test.
        ax = self.helper2(heatmap)
        # Check outputs.
        outline = [
            c
            for c in ax.collections
            if isinstance(c, mcollections.LineCollection)
        ]
        segments = [
            [tuple(np.round(point, 6).tolist()) for point in segment]
            for segment in outline[0].get_segments()
        ]
        star = [line for line in ax.get_lines() if line.get_marker() == "*"]
        star_xy = np.round(star[0].get_xydata()[0], 6).tolist()
        actual = (
            f"label={outline[0].get_label()}\nsegments={segments}\n"
            f"star={star_xy}"
        )
        plt.close("all")
        self.assert_equal(actual, expected)


# #############################################################################
# Test__round_slider_value
# #############################################################################


class Test__round_slider_value(hunitest.TestCase):
    """
    Test `racket_shot_probability_utils._round_slider_value()`.
    """

    def helper(self, value: float, expected: str) -> None:
        """
        Test helper for `_round_slider_value()`.

        :param value: slider value
        :param expected: expected rounded value, as printed by `str()`
        """
        # Run test.
        actual = str(racket_shot_probability_utils._round_slider_value(value))
        # Check outputs.
        self.assert_equal(actual, expected)

    def test1(self) -> None:
        """
        Test that the drift of 21 `+` clicks from -2.1 rounds to 0.
        """
        # Prepare inputs.
        value = 6.38378239159465e-16
        # Prepare outputs.
        expected = "0.0"
        # Run test and check outputs.
        self.helper(value, expected)

    def test2(self) -> None:
        """
        Test that -0.0 becomes 0.0, so it does not print as "-0.00".
        """
        # Prepare inputs.
        value = -0.0
        # Prepare outputs.
        expected = "0.0"
        # Run test and check outputs.
        self.helper(value, expected)

    def test3(self) -> None:
        """
        Test that a drifted ordinary value rounds back to its slider step.
        """
        # Prepare inputs.
        value = 4.1000000000000005
        # Prepare outputs.
        expected = "4.1"
        # Run test and check outputs.
        self.helper(value, expected)


# #############################################################################
# Test__check_serve
# #############################################################################


class Test__check_serve(hunitest.TestCase):
    """
    Test `racket_shot_probability_utils._check_serve()`.
    """

    def helper(
        self,
        court: racket_params.CourtGeometry,
        side: str,
        x1: float,
        y1: float,
        expected: str,
    ) -> None:
        """
        Test helper for `_check_serve()`.

        :param court: court geometry of the serve
        :param side: "deuce" or "ad", the side the server serves from
        :param x1: where the serve lands, across the court
        :param y1: where the serve lands, along the court
        :param expected: expected fault, "" if the serve is in
        """
        # Run test.
        _, actual = racket_shot_probability_utils._check_serve(
            side, x1, y1, court
        )
        # Check outputs.
        self.assert_equal(actual, expected)

    def test1(self) -> None:
        """
        Test that a deuce serve into the diagonal box (x <= 0) is in.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        side = "deuce"
        x1 = -2.0
        y1 = 3.0
        # Prepare outputs.
        expected = ""
        # Run test and check outputs.
        self.helper(court, side, x1, y1, expected)

    def test2(self) -> None:
        """
        Test that an ad serve into the diagonal box (x >= 0) is in.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        side = "ad"
        x1 = 2.0
        y1 = 3.0
        # Prepare outputs.
        expected = ""
        # Run test and check outputs.
        self.helper(court, side, x1, y1, expected)

    def test3(self) -> None:
        """
        Test that a deuce serve into the same-side box (x > 0) is a fault.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        side = "deuce"
        x1 = 2.0
        y1 = 3.0
        # Prepare outputs.
        expected = "into the wrong box"
        # Run test and check outputs.
        self.helper(court, side, x1, y1, expected)

    def test4(self) -> None:
        """
        Test that an ad serve into the same-side box (x < 0) is a fault.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        side = "ad"
        x1 = -2.0
        y1 = 3.0
        # Prepare outputs.
        expected = "into the wrong box"
        # Run test and check outputs.
        self.helper(court, side, x1, y1, expected)

    def test5(self) -> None:
        """
        Test that a tennis serve past the service line is long.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        side = "deuce"
        x1 = -2.0
        y1 = 7.0
        # Prepare outputs.
        expected = "long"
        # Run test and check outputs.
        self.helper(court, side, x1, y1, expected)

    def test6(self) -> None:
        """
        Test that a serve past the singles sideline is wide.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        side = "deuce"
        x1 = -4.5
        y1 = 3.0
        # Prepare outputs.
        expected = "wide"
        # Run test and check outputs.
        self.helper(court, side, x1, y1, expected)

    def test7(self) -> None:
        """
        Test that a tennis serve on the service line and the sideline is in.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        side = "deuce"
        x1 = -4.115
        y1 = 6.4
        # Prepare outputs.
        expected = ""
        # Run test and check outputs.
        self.helper(court, side, x1, y1, expected)

    def test8(self) -> None:
        """
        Test that a serve on the center line is in, for the deuce box.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        side = "deuce"
        x1 = 0.0
        y1 = 3.0
        # Prepare outputs.
        expected = ""
        # Run test and check outputs.
        self.helper(court, side, x1, y1, expected)

    def test9(self) -> None:
        """
        Test that a pickleball serve short of the kitchen line is a fault.
        """
        # Prepare inputs.
        court = racket_params.PICKLEBALL.court
        side = "deuce"
        x1 = -1.5
        y1 = 1.0
        # Prepare outputs.
        expected = "in the kitchen"
        # Run test and check outputs.
        self.helper(court, side, x1, y1, expected)

    def test10(self) -> None:
        """
        Test that a pickleball serve on the kitchen line is a fault.
        """
        # Prepare inputs.
        court = racket_params.PICKLEBALL.court
        side = "deuce"
        x1 = -1.5
        y1 = 2.13
        # Prepare outputs.
        expected = "in the kitchen"
        # Run test and check outputs.
        self.helper(court, side, x1, y1, expected)

    def test11(self) -> None:
        """
        Test that a pickleball serve just past the kitchen line is in.
        """
        # Prepare inputs.
        court = racket_params.PICKLEBALL.court
        side = "deuce"
        x1 = -1.5
        y1 = 2.14
        # Prepare outputs.
        expected = ""
        # Run test and check outputs.
        self.helper(court, side, x1, y1, expected)

    def test12(self) -> None:
        """
        Test that a pickleball serve 3 mm past the baseline is long.
        """
        # Prepare inputs.
        court = racket_params.PICKLEBALL.court
        side = "deuce"
        x1 = -1.5
        y1 = 6.708
        # Prepare outputs.
        expected = "long"
        # Run test and check outputs.
        self.helper(court, side, x1, y1, expected)

    def test13(self) -> None:
        """
        Test that the deuce box is at x <= 0 on player 2's side.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        side = "deuce"
        x1 = -2.0
        y1 = 3.0
        # Prepare outputs.
        expected = "CourtRegion(x_min=-4.115, x_max=0, y_min=0.0, y_max=6.4)"
        # Run test.
        actual, _ = racket_shot_probability_utils._check_serve(
            side, x1, y1, court
        )
        # Check outputs.
        self.assert_equal(str(actual), expected)


# #############################################################################
# Test__get_server_xy
# #############################################################################


class Test__get_server_xy(hunitest.TestCase):
    """
    Test `racket_shot_probability_utils._get_server_xy()`.
    """

    def helper(
        self, court: racket_params.CourtGeometry, side: str, expected: str
    ) -> None:
        """
        Test helper for `_get_server_xy()`.

        :param court: court geometry of the serve
        :param side: "deuce" or "ad"
        :param expected: expected position, with 3 decimals
        """
        # Run test.
        x, y = racket_shot_probability_utils._get_server_xy(side, court)
        actual = f"({x:.3f}, {y:.3f})"
        # Check outputs.
        self.assert_equal(actual, expected)

    def test1(self) -> None:
        """
        Test the deuce stance: right of the center mark, behind the baseline.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        side = "deuce"
        # Prepare outputs.
        expected = "(0.500, -12.185)"
        # Run test and check outputs.
        self.helper(court, side, expected)

    def test2(self) -> None:
        """
        Test the ad stance: left of the center mark, behind the baseline.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        side = "ad"
        # Prepare outputs.
        expected = "(-0.500, -12.185)"
        # Run test and check outputs.
        self.helper(court, side, expected)

    def test3(self) -> None:
        """
        Test the pickleball deuce stance, behind its shorter baseline.
        """
        # Prepare inputs.
        court = racket_params.PICKLEBALL.court
        side = "deuce"
        # Prepare outputs.
        expected = "(0.500, -7.005)"
        # Run test and check outputs.
        self.helper(court, side, expected)


# #############################################################################
# Test__draw_fault
# #############################################################################


class Test__draw_fault(hunitest.TestCase):
    """
    Test `racket_shot_probability_utils._draw_fault()`.
    """

    def helper(
        self, side: str, serve_xy: Tuple[float, float], expected: str
    ) -> None:
        """
        Test helper for `_draw_fault()`: where the "Fault" label goes.

        :param side: "deuce" or "ad", the side the server serves from
        :param serve_xy: where the serve lands, `(x1, y1)`
        :param expected: expected label text, position and alignment
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        server_xy = racket_shot_probability_utils._get_server_xy(side, court)
        _, ax = plt.subplots()
        # Run test.
        racket_shot_probability_utils._draw_fault(ax, server_xy, serve_xy, court)
        # The label is the only text that is not an arrow.
        label = [t for t in ax.texts if not isinstance(t, mtext.Annotation)][0]
        x, y = label.get_position()
        actual = f"{label.get_text()} at ({x:.2f}, {y:.2f}), ha={label.get_ha()}"
        plt.close("all")
        # Check outputs.
        self.assert_equal(actual, expected)

    def test1(self) -> None:
        """
        Test a long deuce serve near the center line: it lands on the left half,
        so the label starts 1 m right of the server, off the arrow.
        """
        # Prepare inputs.
        side = "deuce"
        serve_xy = (-0.3, 6.6)
        # Prepare outputs.
        expected = "Fault at (1.50, 5.94), ha=left"
        # Run test and check outputs.
        self.helper(side, serve_xy, expected)

    def test2(self) -> None:
        """
        Test a long ad serve: it lands on the right half, so the label ends 1 m
        left of the server.
        """
        # Prepare inputs.
        side = "ad"
        serve_xy = (3.0, 11.8)
        # Prepare outputs.
        expected = "Fault at (-1.50, 5.94), ha=right"
        # Run test and check outputs.
        self.helper(side, serve_xy, expected)

    def test3(self) -> None:
        """
        Test an ad serve into the wrong box, right of the server: the landing
        point, not the server, is the right end of the arrow.
        """
        # Prepare inputs.
        side = "ad"
        serve_xy = (-0.2, 3.2)
        # Prepare outputs.
        expected = "Fault at (0.80, 5.94), ha=left"
        # Run test and check outputs.
        self.helper(side, serve_xy, expected)


# #############################################################################
# Test__build_comments_text
# #############################################################################


class Test__build_comments_text(hunitest.TestCase):
    """
    Test `racket_shot_probability_utils._build_comments_text()`.
    """

    def test1(self) -> None:
        """
        Test that the standard error is not 0 when all dots land in.
        """
        # Prepare inputs.
        serve_xy = (-2.1, 3.2)
        side = "deuce"
        target_xy = (1.0, -6.0)
        std_xy = (0.8, 1.6)
        p_in = 0.9997
        # All 1000 dots land in.
        p_hat = 1.0
        n_samples = 1000
        # Prepare outputs.
        # The standard error comes from `P_in`: sqrt(0.9997 * 0.0003 / 1000).
        expected = """
        Serve to (x1, y1) = (-2.10, 3.20)
          in the deuce box: yes

        Return to (x2, y2) = (1.00, -6.00)
          (std_x2, std_y2) = (0.80, 1.60)

        P_in (closed form) = 0.9997
        P_hat (Monte Carlo, N = 1000) = 1.0000
          standard error = 0.0005"""
        # Run test.
        actual = racket_shot_probability_utils._build_comments_text(
            serve_xy,
            side,
            target_xy,
            std_xy,
            p_in,
            p_hat,
            n_samples,
        )
        # Check outputs.
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test__build_fault_text
# #############################################################################


class Test__build_fault_text(hunitest.TestCase):
    """
    Test `racket_shot_probability_utils._build_fault_text()`.
    """

    def test1(self) -> None:
        """
        Test that a fault names the reason and shows no probability.
        """
        # Prepare inputs.
        serve_xy = (2.0, 3.2)
        side = "deuce"
        fault = "into the wrong box"
        # Prepare outputs.
        expected = """
        Serve to (x1, y1) = (2.00, 3.20)
          in the deuce box: no

        Fault: into the wrong box
          No return, so no P_in and no P_hat"""
        # Run test.
        actual = racket_shot_probability_utils._build_fault_text(
            serve_xy, side, fault
        )
        # Check outputs.
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test__build_shared_controls
# #############################################################################


class Test__build_shared_controls(hunitest.TestCase):
    """
    Test `racket_shot_probability_utils._build_shared_controls()`.
    """

    def helper(self, sport: racket_params.SportParams, expected: str) -> None:
        """
        Test helper that builds the shared controls for `sport`.

        :param sport: sport whose court sets the ranges
        :param expected: expected controls, as `name: (min, max, value)`, and
            the type of each box
        """
        # Run test.
        controls = racket_shot_probability_utils._build_shared_controls(
            sport.court
        )
        # Check outputs.
        lines = []
        for name, (control, box) in controls.items():
            if name == "side":
                lines.append(f"side: {control.options}, {control.value}")
            else:
                lines.append(
                    f"{name}: ({control.min}, {control.max}, {control.value})"
                    f" in {type(box).__name__}"
                )
        actual = "\n".join(lines)
        self.assert_equal(actual, expected, dedent=True)

    def test1(self) -> None:
        """
        Test the shared controls for tennis: the Cell 1.1 ranges and starting
        values.
        """
        # Prepare inputs.
        sport = racket_params.TENNIS
        # Prepare outputs.
        expected = """
        side: ('deuce', 'ad'), deuce
        x1: (-4.115, 4.115, -2.1) in HBox
        y1: (0.1, 11.885, 3.2) in HBox
        std_x2: (0.1, 3.0, 0.5) in HBox
        std_y2: (0.1, 3.0, 1.0) in HBox
        seed: (0, 100, 1) in HBox"""
        # Run test and check outputs.
        self.helper(sport, expected)

    def test2(self) -> None:
        """
        Test the shared controls for pickleball, whose court is smaller.
        """
        # Prepare inputs.
        sport = racket_params.PICKLEBALL
        # Prepare outputs.
        expected = """
        side: ('deuce', 'ad'), deuce
        x1: (-3.05, 3.05, -1.5) in HBox
        y1: (0.1, 6.705, 4.4) in HBox
        std_x2: (0.1, 3.0, 0.5) in HBox
        std_y2: (0.1, 3.0, 1.0) in HBox
        seed: (0, 100, 1) in HBox"""
        # Run test and check outputs.
        self.helper(sport, expected)


# #############################################################################
# Test_cell1_1_plot_shot_widget
# #############################################################################


class Test_cell1_1_plot_shot_widget(hunitest.TestCase):
    """
    Test `racket_shot_probability_utils.cell1_1_plot_shot_widget()`.
    """

    def _build_and_record(
        self, sport: racket_params.SportParams
    ) -> Tuple[ipywidgets.VBox, List[str], List[str]]:
        """
        Build the widget, recording what every redraw shows.

        For this test only, `plt.show()` is replaced by a function that saves,
        for the figure being shown:
        - the text of the comments panel
        - a summary of the court: how many arrows, dot clouds and ellipses it
          has, and its text labels (e.g., "Fault")

        :param sport: sport whose court the widget draws
        :return: the widget, and the lists that fill up with comments texts and
            court summaries
        """
        texts: List[str] = []
        courts: List[str] = []

        def _record_redraw() -> None:
            ax_court, ax_text = plt.gcf().axes[0], plt.gcf().axes[-1]
            texts.append(ax_text.texts[-1].get_text())
            # An arrow is an `Annotation`, which is also a text: split them.
            arrows = [
                t for t in ax_court.texts if isinstance(t, mtext.Annotation)
            ]
            labels = [
                t.get_text()
                for t in ax_court.texts
                if not isinstance(t, mtext.Annotation)
            ]
            ellipses = [
                p for p in ax_court.patches if isinstance(p, mpatches.Ellipse)
            ]
            courts.append(
                f"arrows={len(arrows)} dots={len(ax_court.collections)} "
                f"ellipses={len(ellipses)} labels={labels}"
            )

        self.enterContext(
            umock.patch.object(
                racket_shot_probability_utils.plt,
                "show",
                side_effect=_record_redraw,
            )
        )
        widget = racket_shot_probability_utils.cell1_1_plot_shot_widget(sport)
        return widget, texts, courts

    def helper(self, sport: racket_params.SportParams) -> None:
        """
        Test helper that builds the widget, which also renders its first plot.

        :param sport: sport whose court the widget draws
        """
        # Prepare outputs.
        # The side toggle, 7 slider boxes (x1, y1, x2, y2, std_x2, std_y2,
        # seed), then the plot.
        expected = str(["ToggleButtons"] + ["HBox"] * 7 + ["Output"])
        # Run test.
        widget = racket_shot_probability_utils.cell1_1_plot_shot_widget(sport)
        actual = str([type(child).__name__ for child in widget.children])
        # Check outputs.
        self.assert_equal(actual, expected)
        plt.close("all")

    def test1(self) -> None:
        """
        Test that the widget builds and renders for tennis.
        """
        # Prepare inputs.
        sport = racket_params.TENNIS
        # Run test and check outputs.
        self.helper(sport)

    def test2(self) -> None:
        """
        Test that the widget builds and renders for pickleball.
        """
        # Prepare inputs.
        sport = racket_params.PICKLEBALL
        # Run test and check outputs.
        self.helper(sport)

    # Slow: each of the 21 clicks redraws the figure (about 1.5 s each).
    @pytest.mark.slow
    def test3(self) -> None:
        """
        Test that clicking `+` from the default `x1` up to 0 keeps the serve in.

        Without rounding, `x1` drifts to 6.4e-16, just right of the center line,
        which is outside the deuce box and would be a false fault.
        """
        # Prepare inputs.
        sport = racket_params.TENNIS
        # Prepare outputs.
        expected = """
        Serve to (x1, y1) = (0.00, 3.20)
          in the deuce box: yes"""
        # Run test.
        widget, texts, _ = self._build_and_record(sport)
        # The controls are `[side, x1, y1, ...]`, and each slider box is
        # `[slider, -, text, +]`: index 0 is the slider and index 3 the `+`
        # button.
        x1_box = widget.children[1]
        x1_slider = x1_box.children[0]
        plus_button = x1_box.children[3]
        # From -2.1, 21 clicks of 0.1 drift to 6.4e-16 instead of exactly 0.
        n_clicks = round(-x1_slider.value / x1_slider.step)
        for _ in range(n_clicks):
            plus_button.click()
        actual = "\n".join(texts[-1].splitlines()[:2])
        plt.close("all")
        # Check outputs.
        self.assert_equal(actual, expected, dedent=True)

    def test4(self) -> None:
        """
        Test that a fault hides the return, the ellipses, the dots and `P_in`.
        """
        # Prepare inputs.
        sport = racket_params.TENNIS
        # The default `x1` (-2.1) is in the deuce box, so an ad serve to it
        # goes into the wrong box.
        side = "ad"
        # Prepare outputs.
        # One arrow (the serve), no dots, no ellipses, and a "Fault" label.
        expected = """
        arrows=1 dots=0 ellipses=0 labels=['Fault']
        Serve to (x1, y1) = (-2.10, 3.20)
          in the ad box: no

        Fault: into the wrong box
          No return, so no P_in and no P_hat"""
        # Run test.
        widget, texts, courts = self._build_and_record(sport)
        side_toggle = widget.children[0]
        side_toggle.value = side
        actual = courts[-1] + "\n" + texts[-1]
        plt.close("all")
        # Check outputs.
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test__build_settings_string
# #############################################################################


class Test__build_settings_string(hunitest.TestCase):
    """
    Test `racket_shot_probability_utils._build_settings_string()`.
    """

    def test1(self) -> None:
        """
        Test that every setting is listed once, in the order of the controls,
        then the grid size and the samples per target.
        """
        # Prepare inputs.
        sport_name = "Tennis"
        side = "deuce"
        serve_xy = (-2.1, 3.2)
        std_xy = (0.5, 1.0)
        ball_speed_mps = 25.0
        reaction_time_s = 0.2
        move_speed_mps = 1.5
        seed = 1
        grid_size = (10, 15)
        n_samples = 5000
        # Prepare outputs.
        expected = (
            "sport=Tennis side=deuce x1=-2.1 y1=3.2 std_x2=0.5 std_y2=1.0 "
            "ball_speed=25.0 t_r=0.2 v_p=1.5 seed=1 n_x=10 n_y=15 "
            "n_samples=5000"
        )
        # Run test.
        actual = racket_shot_probability_utils._build_settings_string(
            sport_name,
            side,
            serve_xy,
            std_xy,
            ball_speed_mps,
            reaction_time_s,
            move_speed_mps,
            seed,
            grid_size,
            n_samples,
        )
        # Check outputs.
        self.assert_equal(actual, expected)

    def test2(self) -> None:
        """
        Test that each value reads back as exactly the same number.
        """
        # Prepare inputs.
        # Values that 2 decimals would round: the pickleball maximum ball speed
        # and a reaction time of 3 `+` clicks of 0.05 from 0.2.
        values = {
            "x1": 1.234567,
            "y1": 6.705,
            "std_x2": 0.1,
            "std_y2": 3.0,
            "ball_speed": 24.59,
            "t_r": 0.35,
            "v_p": 0.0,
            "seed": 100,
            "n_x": 9,
            "n_y": 10,
            "n_samples": 5000,
        }
        # Run test.
        settings = racket_shot_probability_utils._build_settings_string(
            "Pickleball",
            "ad",
            (values["x1"], values["y1"]),
            (values["std_x2"], values["std_y2"]),
            values["ball_speed"],
            values["t_r"],
            values["v_p"],
            int(values["seed"]),
            (int(values["n_x"]), int(values["n_y"])),
            int(values["n_samples"]),
        )
        parsed = dict(item.split("=", 1) for item in settings.split(" "))
        # Check outputs.
        self.assert_equal(
            str(list(parsed)),
            str(["sport", "side"] + list(values)),
        )
        for name, value in values.items():
            self.assertEqual(float(parsed[name]), value, name)


# #############################################################################
# Test__format_p_winner
# #############################################################################


class Test__format_p_winner(hunitest.TestCase):
    """
    Test `racket_shot_probability_utils._format_p_winner()`.
    """

    def helper(self, p_winner: float, p_winner_se: float, expected: str) -> None:
        """
        Test helper for `_format_p_winner()` with 5000 samples.

        :param p_winner: probability of a winner
        :param p_winner_se: its standard error
        :param expected: expected text
        """
        # Prepare inputs.
        n_samples = 5000
        # Run test.
        actual = racket_shot_probability_utils._format_p_winner(
            p_winner, p_winner_se, n_samples
        )
        # Check outputs.
        self.assert_equal(actual, expected)

    def test1(self) -> None:
        """
        Test a probability with a standard error above 0.
        """
        # Prepare inputs.
        p_winner = 0.9312
        p_winner_se = 0.003578
        # Prepare outputs.
        expected = "0.9312 +- 0.0036 (N = 5000)"
        # Run test and check outputs.
        self.helper(p_winner, p_winner_se, expected)

    def test2(self) -> None:
        """
        Test that a probability of 0 gives the count of winners, not a standard
        error of 0.
        """
        # Prepare inputs.
        p_winner = 0.0
        p_winner_se = 0.0
        # Prepare outputs.
        expected = "0.0000 (0 of 5000 samples)"
        # Run test and check outputs.
        self.helper(p_winner, p_winner_se, expected)

    def test3(self) -> None:
        """
        Test that a probability of 1 gives the count of winners.
        """
        # Prepare inputs.
        p_winner = 1.0
        p_winner_se = 0.0
        # Prepare outputs.
        expected = "1.0000 (5000 of 5000 samples)"
        # Run test and check outputs.
        self.helper(p_winner, p_winner_se, expected)


# #############################################################################
# Test__format_point
# #############################################################################


class Test__format_point(hunitest.TestCase):
    """
    Test `racket_shot_probability_utils._format_point()`.
    """

    def test1(self) -> None:
        """
        Test that a rounding error around 0 does not print as "-0.00".
        """
        # Prepare inputs.
        # The middle column of the pickleball grid, as `make_target_grid()`
        # computes it.
        xy = (-2.220446049250313e-16, -3.0175)
        # Prepare outputs.
        expected = "(0.00, -3.02)"
        # Run test.
        actual = racket_shot_probability_utils._format_point(xy)
        # Check outputs.
        self.assert_equal(actual, expected)


# #############################################################################
# Test__build_winner_text
# #############################################################################


class Test__build_winner_text(hunitest.TestCase):
    """
    Test `racket_shot_probability_utils._build_winner_text()`.
    """

    def test1(self) -> None:
        """
        Test that the text gives the numbers of the marked target, from its own
        row of the heatmap.
        """
        # Prepare inputs.
        # A 3 x 4 tennis grid with `P(winner)` from 0.2 to 0.8 in grid order,
        # so the last cell is the best and the only one of the best region;
        # `P(in)` and `P(reached)` differ in every row.
        court = racket_params.TENNIS.court
        region = racket_shot_probability_utils.get_striker_half_region(court)
        heatmap = racket_scoring.make_target_grid(region, 3, 4)
        heatmap["p_winner"] = np.linspace(0.2, 0.8, len(heatmap))
        heatmap["p_winner_se"] = 0.01
        heatmap["p_in"] = np.linspace(0.3, 0.9, len(heatmap))
        heatmap["p_reached"] = np.linspace(0.0, 0.11, len(heatmap))
        best = racket_shot_probability_utils.find_best_target(heatmap)
        serve_xy = (-2.1, 3.2)
        side = "deuce"
        n_samples = 5000
        settings = "sport=Tennis side=deuce"
        # Prepare outputs.
        expected = """
        Serve to (x1, y1) = (-2.10, 3.20)
          in the deuce box: yes

        Best target (star) = (2.74, -1.49)
          P(winner) = 0.8000 +- 0.0100 (N = 5000)
          P(in) = 0.9000, P(reached) = 0.1100
        Best region: 1 of 12 cells
          within 2 SE of the highest P(winner)

        Settings:
        sport=Tennis side=deuce"""
        # Run test.
        actual = racket_shot_probability_utils._build_winner_text(
            serve_xy, side, heatmap, best, n_samples, settings
        )
        # Check outputs.
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test__build_winner_fault_text
# #############################################################################


class Test__build_winner_fault_text(hunitest.TestCase):
    """
    Test `racket_shot_probability_utils._build_winner_fault_text()`.
    """

    def test1(self) -> None:
        """
        Test that a fault names the reason, shows no probability, and keeps the
        settings string.
        """
        # Prepare inputs.
        serve_xy = (-2.1, 3.2)
        side = "ad"
        fault = "into the wrong box"
        settings = "sport=Tennis side=ad"
        # Prepare outputs.
        expected = """
        Serve to (x1, y1) = (-2.10, 3.20)
          in the ad box: no

        Fault: into the wrong box
          No return, so no P(winner) and no heatmap

        Settings:
        sport=Tennis side=ad"""
        # Run test.
        actual = racket_shot_probability_utils._build_winner_fault_text(
            serve_xy, side, fault, settings
        )
        # Check outputs.
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_cell2_1_plot_winner_widget
# #############################################################################


class Test_cell2_1_plot_winner_widget(hunitest.TestCase):
    """
    Test `racket_shot_probability_utils.cell2_1_plot_winner_widget()`.
    """

    def _build_and_record(
        self, sport: racket_params.SportParams
    ) -> Tuple[ipywidgets.VBox, Dict[str, Any], List[Dict[str, Any]]]:
        """
        Build the widget, recording what every redraw shows.

        For this test only, `plt.show()` is replaced by a function that saves,
        for the figure being shown:
        - "text": the text of the comments panel
        - "values": the `P(winner)` of the map, flat in the image order (`y`
          outer, `x` inner), or None if there is no map
        - "star": the position of the star, or None
        - "court_labels": the text labels of the court, e.g., "Fault"
        - "legend_panel": the panels with a legend, e.g., "comments", or
          "court+comments" if the court also kept one

        :param sport: sport whose court the widget draws
        :return: the widget, its controls by name, and the list of redraws
        """
        redraws: List[Dict[str, Any]] = []

        def _record_redraw() -> None:
            fig = plt.gcf()
            # The comments panel by its title: the colorbar of the map is also
            # an axes of the figure.
            ax_court = fig.axes[0]
            ax_text = [ax for ax in fig.axes if ax.get_title() == "Comments"][0]
            meshes = [
                c
                for c in ax_court.collections
                if isinstance(c, mcollections.QuadMesh)
            ]
            stars = [line for line in ax_court.lines if line.get_marker() == "*"]
            # Every panel with a legend, so a legend left on the court shows
            # even when the comments panel has one too.
            panels = [("court", ax_court), ("comments", ax_text)]
            legend_panel = "+".join(
                name for name, ax in panels if ax.get_legend() is not None
            )
            redraws.append(
                {
                    "text": ax_text.texts[-1].get_text(),
                    "values": (
                        np.asarray(meshes[0].get_array()).ravel().copy()
                        if meshes
                        else None
                    ),
                    "star": (
                        tuple(stars[0].get_xydata()[0].tolist())
                        if stars
                        else None
                    ),
                    "court_labels": [
                        t.get_text()
                        for t in ax_court.texts
                        if not isinstance(t, mtext.Annotation)
                    ],
                    "legend_panel": legend_panel,
                }
            )

        self.enterContext(
            umock.patch.object(
                racket_shot_probability_utils.plt,
                "show",
                side_effect=_record_redraw,
            )
        )
        widget = racket_shot_probability_utils.cell2_1_plot_winner_widget(sport)
        # The side toggle, then one `[slider, -, text, +]` box per slider,
        # then the plot and the settings line.
        controls: Dict[str, Any] = {"side": widget.children[0]}
        for box in widget.children[1:-2]:
            controls[box.children[0].description] = box.children[0]
        return widget, controls, redraws

    def _get_direct_heatmap(
        self, sport: racket_params.SportParams, controls: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Compute the map with `compute_winner_heatmap()` called directly, for the
        current values of the controls and the default grid.

        :param sport: sport of the widget
        :param controls: controls of the widget, by name
        :return: the heatmap
        """
        n_x, n_y = racket_shot_probability_utils.HEATMAP_GRID_SIZE[sport.name]
        heatmap: pd.DataFrame = (
            racket_shot_probability_utils.compute_winner_heatmap(
                (controls["x1"].value, controls["y1"].value),
                (controls["std_x2"].value, controls["std_y2"].value),
                controls["ball_speed"].value,
                controls["t_r"].value,
                controls["v_p"].value,
                sport.court,
                n_x,
                n_y,
                racket_shot_probability_utils.HEATMAP_N_SAMPLES,
                controls["seed"].value,
            )
        )
        return heatmap

    def helper(self, sport: racket_params.SportParams, expected: str) -> None:
        """
        Test helper that builds the widget, which also renders its first plot,
        and checks the controls and the settings string of that plot.

        :param sport: sport whose court the widget draws
        :param expected: expected controls, as `name: (min, max, value)`, and
            settings string
        """
        # Run test.
        widget, controls, redraws = self._build_and_record(sport)
        # Check outputs.
        # The side toggle, 8 slider boxes, the plot, then the settings line.
        types = [type(child).__name__ for child in widget.children]
        self.assert_equal(
            str(types),
            str(["ToggleButtons"] + ["HBox"] * 8 + ["Output", "HTML"]),
        )
        lines = [f"side: {controls['side'].options}, {controls['side'].value}"]
        for name, control in list(controls.items())[1:]:
            lines.append(
                f"{name}: ({control.min}, {control.max}, {control.value})"
            )
        lines.append(redraws[-1]["text"].splitlines()[-1])
        actual = "\n".join(lines)
        plt.close("all")
        self.assert_equal(actual, expected, dedent=True)

    def test1(self) -> None:
        """
        Test that the widget builds and renders for tennis, with its default
        controls.
        """
        # Prepare inputs.
        sport = racket_params.TENNIS
        # Prepare outputs.
        # A ball speed of 25 m/s up to the tennis maximum of 73.14 m/s.
        expected = """
        side: ('deuce', 'ad'), deuce
        x1: (-4.115, 4.115, -2.1)
        y1: (0.1, 11.885, 3.2)
        std_x2: (0.1, 3.0, 0.5)
        std_y2: (0.1, 3.0, 1.0)
        ball_speed: (5.0, 73.14, 25.0)
        t_r: (0.0, 1.0, 0.2)
        v_p: (0.0, 6.0, 1.5)
        seed: (0, 100, 1)
        sport=Tennis side=deuce x1=-2.1 y1=3.2 std_x2=0.5 std_y2=1.0 ball_speed=25.0 t_r=0.2 v_p=1.5 seed=1 n_x=10 n_y=15 n_samples=5000"""
        # Run test and check outputs.
        self.helper(sport, expected)

    def test2(self) -> None:
        """
        Test that the widget builds and renders for pickleball, with its default
        controls.
        """
        # Prepare inputs.
        sport = racket_params.PICKLEBALL
        # Prepare outputs.
        # A ball speed of 12 m/s up to the pickleball maximum of 24.59 m/s.
        expected = """
        side: ('deuce', 'ad'), deuce
        x1: (-3.05, 3.05, -1.5)
        y1: (0.1, 6.705, 4.4)
        std_x2: (0.1, 3.0, 0.5)
        std_y2: (0.1, 3.0, 1.0)
        ball_speed: (5.0, 24.59, 12.0)
        t_r: (0.0, 1.0, 0.2)
        v_p: (0.0, 6.0, 1.5)
        seed: (0, 100, 1)
        sport=Pickleball side=deuce x1=-1.5 y1=4.4 std_x2=0.5 std_y2=1.0 ball_speed=12.0 t_r=0.2 v_p=1.5 seed=1 n_x=9 n_y=10 n_samples=5000"""
        # Run test and check outputs.
        self.helper(sport, expected)

    def test3(self) -> None:
        """
        Test that a fault shows no map, the fault, and the settings string, with
        the legend in the comments panel.
        """
        # Prepare inputs.
        sport = racket_params.TENNIS
        # The default `x1` (-2.1) is in the deuce box, so an ad serve to it
        # goes into the wrong box.
        side = "ad"
        # Prepare outputs.
        expected = """
        redraws=1 map=False star=False labels=['Fault'] legend=comments
        Serve to (x1, y1) = (-2.10, 3.20)
          in the ad box: no

        Fault: into the wrong box
          No return, so no P(winner) and no heatmap

        Settings:
        sport=Tennis side=ad x1=-2.1 y1=3.2 std_x2=0.5 std_y2=1.0 ball_speed=25.0 t_r=0.2 v_p=1.5 seed=1 n_x=10 n_y=15 n_samples=5000"""
        # Run test.
        _, controls, redraws = self._build_and_record(sport)
        n_before = len(redraws)
        controls["side"].value = side
        last = redraws[-1]
        actual = (
            f"redraws={len(redraws) - n_before} "
            f"map={last['values'] is not None} star={last['star'] is not None} "
            f"labels={last['court_labels']} legend={last['legend_panel']}\n"
            + last["text"]
        )
        plt.close("all")
        # Check outputs.
        self.assert_equal(actual, expected, dedent=True)

    def test4(self) -> None:
        """
        Test that the map, the star, and the comments equal the Phase 3
        functions called directly, with the legend in the comments panel.
        """
        # Prepare inputs.
        sport = racket_params.PICKLEBALL
        # Run test.
        _, controls, redraws = self._build_and_record(sport)
        last = redraws[-1]
        heatmap = self._get_direct_heatmap(sport, controls)
        best = racket_shot_probability_utils.find_best_target(heatmap)
        plt.close("all")
        # Check outputs.
        # The map holds the direct `P(winner)` of every cell, in `cell_id`
        # order.
        expected_values = heatmap.sort_values("cell_id")["p_winner"].to_numpy()
        np.testing.assert_array_equal(last["values"], expected_values)
        # The star and the comments give the same best target.
        self.assertEqual(last["star"], (best["x_m"], best["y_m"]))
        p_winner = racket_shot_probability_utils._format_p_winner(
            best["p_winner"],
            best["p_winner_se"],
            racket_shot_probability_utils.HEATMAP_N_SAMPLES,
        )
        star_xy = racket_shot_probability_utils._format_point(last["star"])
        expected = (
            f"Best target (star) = {star_xy}\n  P(winner) = {p_winner}\n"
            f"legend=comments"
        )
        actual = (
            "\n".join(last["text"].splitlines()[3:5])
            + f"\nlegend={last['legend_panel']}"
        )
        self.assert_equal(actual, expected)

    def test5(self) -> None:
        """
        Test that the `v_p` slider redraws once and reaches the map.
        """
        # Prepare inputs.
        sport = racket_params.PICKLEBALL
        move_speed_mps = 6.0
        # Run test.
        _, controls, redraws = self._build_and_record(sport)
        n_before = len(redraws)
        controls["v_p"].value = move_speed_mps
        last = redraws[-1]
        heatmap = self._get_direct_heatmap(sport, controls)
        plt.close("all")
        # Check outputs.
        self.assertEqual(len(redraws) - n_before, 1)
        self.assertIn("v_p=6.0", last["text"].splitlines()[-1])
        expected_values = heatmap.sort_values("cell_id")["p_winner"].to_numpy()
        np.testing.assert_array_equal(last["values"], expected_values)
        # A faster player 1 reaches more returns, so the map changed.
        self.assertFalse(np.array_equal(last["values"], redraws[0]["values"]))

    def test6(self) -> None:
        """
        Test that the settings line under the figure, which can be copied, is
        the settings string of the comments panel after every redraw.
        """
        # Prepare inputs.
        sport = racket_params.PICKLEBALL
        # Run test.
        widget, controls, redraws = self._build_and_record(sport)
        settings_line = widget.children[-1]
        lines = [settings_line.value]
        texts = [redraws[-1]["text"]]
        controls["seed"].value = 2
        lines.append(settings_line.value)
        texts.append(redraws[-1]["text"])
        # A fault keeps the line too.
        controls["side"].value = "ad"
        lines.append(settings_line.value)
        texts.append(redraws[-1]["text"])
        plt.close("all")
        # Check outputs.
        expected = (
            "settings: <code>sport=Pickleball side=ad x1=-1.5 y1=4.4 "
            "std_x2=0.5 std_y2=1.0 ball_speed=12.0 t_r=0.2 v_p=1.5 "
            "seed=2 n_x=9 n_y=10 n_samples=5000</code>"
        )
        self.assert_equal(lines[-1], expected)
        for line, text in zip(lines, texts):
            settings = text.splitlines()[-1]
            self.assertEqual(
                line, f"settings: <code>{html.escape(settings)}</code>"
            )
