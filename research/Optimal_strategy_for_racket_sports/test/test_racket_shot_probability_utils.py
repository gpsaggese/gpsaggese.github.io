"""
Tests for racket_shot_probability_utils module.

Import as:

import research.Optimal_strategy_for_racket_sports.test.test_racket_shot_probability_utils as rosfrsttrspu
"""

import dataclasses
import logging
import types
import unittest.mock as umock
from typing import List, Tuple

import ipywidgets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import helpers.hunit_test as hunitest
from research.Optimal_strategy_for_racket_sports import racket_params
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
        x1: float,
        y1: float,
        expected: str,
    ) -> None:
        """
        Test helper for `_check_serve()`.

        :param court: court geometry of the serve
        :param x1: where the serve lands, across the court
        :param y1: where the serve lands, along the court
        :param expected: expected `(side, is_serve_in)` as a string
        """
        # Run test.
        side, _, is_serve_in = racket_shot_probability_utils._check_serve(
            x1, y1, court
        )
        actual = str((side, is_serve_in))
        # Check outputs.
        self.assert_equal(actual, expected)

    def test1(self) -> None:
        """
        Test that a pickleball serve on the kitchen line is not in.
        """
        # Prepare inputs.
        court = racket_params.PICKLEBALL.court
        x1 = -1.5
        y1 = court.non_volley_zone_m
        # Prepare outputs.
        expected = "('deuce', False)"
        # Run test and check outputs.
        self.helper(court, x1, y1, expected)

    def test2(self) -> None:
        """
        Test that a pickleball serve just past the kitchen line is in.
        """
        # Prepare inputs.
        court = racket_params.PICKLEBALL.court
        x1 = -1.5
        y1 = court.non_volley_zone_m + 0.01
        # Prepare outputs.
        expected = "('deuce', True)"
        # Run test and check outputs.
        self.helper(court, x1, y1, expected)

    def test3(self) -> None:
        """
        Test that a tennis serve on the service line is in.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        x1 = 1.5
        y1 = court.service_line_m
        # Prepare outputs.
        expected = "('ad', True)"
        # Run test and check outputs.
        self.helper(court, x1, y1, expected)

    def test4(self) -> None:
        """
        Test that a serve on the center line (`x1 = 0`) targets the deuce box.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        x1 = 0.0
        y1 = 3.0
        # Prepare outputs.
        expected = "('deuce', True)"
        # Run test and check outputs.
        self.helper(court, x1, y1, expected)


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
        serve_side = "deuce"
        is_serve_in = True
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
            serve_side,
            is_serve_in,
            target_xy,
            std_xy,
            p_in,
            p_hat,
            n_samples,
        )
        # Check outputs.
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_cell1_1_plot_shot_widget
# #############################################################################


class Test_cell1_1_plot_shot_widget(hunitest.TestCase):
    """
    Test `racket_shot_probability_utils.cell1_1_plot_shot_widget()`.
    """

    def _build_and_record(
        self, sport: racket_params.SportParams
    ) -> Tuple[ipywidgets.VBox, List[str]]:
        """
        Build the widget, recording the comments text of every redraw.

        For this test only, `plt.show()` is replaced by a function that saves
        the text of the comments panel of the figure being shown.

        :param sport: sport whose court the widget draws
        :return: the widget, and the list that fills up with comments texts
        """
        texts: List[str] = []

        def _record_comments() -> None:
            texts.append(plt.gcf().axes[-1].texts[-1].get_text())

        self.enterContext(
            umock.patch.object(
                racket_shot_probability_utils.plt,
                "show",
                side_effect=_record_comments,
            )
        )
        widget = racket_shot_probability_utils.cell1_1_plot_shot_widget(sport)
        return widget, texts

    def helper(self, sport: racket_params.SportParams) -> None:
        """
        Test helper that builds the widget, which also renders its first plot.

        :param sport: sport whose court the widget draws
        """
        # Prepare outputs.
        # 7 slider boxes (x1, y1, x2, y2, std_x2, std_y2, seed), then the plot.
        expected = str(["HBox"] * 7 + ["Output"])
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
        Test that clicking `+` from the default `x1` up to 0 picks the deuce box.
        """
        # Prepare inputs.
        sport = racket_params.TENNIS
        # Prepare outputs.
        expected = """
        Serve to (x1, y1) = (0.00, 3.20)
          in the deuce box: yes"""
        # Run test.
        widget, texts = self._build_and_record(sport)
        # Each slider box is `[slider, -, text, +]`: index 0 is the slider and
        # index 3 the `+` button.
        x1_slider = widget.children[0].children[0]
        plus_button = widget.children[0].children[3]
        # From -2.1, 21 clicks of 0.1 drift to 6.4e-16 instead of exactly 0.
        n_clicks = round(-x1_slider.value / x1_slider.step)
        for _ in range(n_clicks):
            plus_button.click()
        actual = "\n".join(texts[-1].splitlines()[:2])
        plt.close("all")
        # Check outputs.
        self.assert_equal(actual, expected, dedent=True)
