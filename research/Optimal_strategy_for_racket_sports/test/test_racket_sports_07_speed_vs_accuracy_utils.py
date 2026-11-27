"""
Tests for racket_sports_07_speed_vs_accuracy_utils module.

Import as:

import research.Optimal_strategy_for_racket_sports.test.test_racket_sports_07_speed_vs_accuracy_utils as rosfrsttrs0svau
"""

import dataclasses
import logging
import os
import pprint
from typing import Any, Dict, List, Tuple

import helpers.hunit_test as hunitest
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.Optimal_strategy_for_racket_sports.notebooks import (
    racket_sports_05_variable_sensitivity_utils,
    racket_sports_07_speed_vs_accuracy_utils,
)

_LOG = logging.getLogger(__name__)

# Use the non-interactive backend so the figures render without a display.
matplotlib.use("Agg")


def _get_tiny_config(
    **update: Any,
) -> racket_sports_07_speed_vs_accuracy_utils.ExperimentConfig:
    """
    Get a configuration whose sweeps run in a fraction of a second.

    More than 1,000 samples, since fewer is slower (see `HEATMAP_N_SAMPLES` of
    the notebook 01 utils).

    :param update: fields of the configuration to change
    :return: one sport, a tiny grid, 3 ball speeds, the base with one
        variant, and 2 values of `a` for the scaling check
    """
    config = racket_sports_07_speed_vs_accuracy_utils.get_experiment_config(
        False
    )
    config = dataclasses.replace(
        config,
        sports=("Pickleball",),
        grid_size={"Tennis": (3, 4), "Pickleball": (3, 4)},
        n_samples=1100,
        speed_ratios=(0.5, 1.0, 2.0),
        a_values={"base": (0.0, 1.0), "knee": (1.0,)},
        scaling_a_values=(0.0, 0.5),
    )
    config = dataclasses.replace(config, **update)
    return config


def _to_str(df: pd.DataFrame) -> str:
    """
    Convert a table into text, with every float written exactly.

    :param df: the table
    :return: the text, e.g., 0.375 for 0.375 and 0.1 for 0.1
    """
    text: str = df.to_string(float_format=lambda value: str(float(value)))
    return text


def _make_sweep_rows(
    p_winner: List[float],
    p_winner_se: float,
    mean_p_winner: List[float],
    **update: Any,
) -> pd.DataFrame:
    """
    Build the rows of one sweep of `run_speed_sweeps()`, by hand.

    :param p_winner: the highest `P(winner)` per ball speed, from `v0 / 2` in
        steps of `v0 / 2`
    :param p_winner_se: the standard error of each value
    :param mean_p_winner: the mean `P(winner)` over the half per ball speed
    :param update: columns to change, e.g., `sport="Pickleball"`
    :return: one row per ball speed, for the tennis base at `a = 1`
    """
    speed_ratios = [0.5 * (i + 1) for i in range(len(p_winner))]
    sweep = pd.DataFrame(
        {
            "sport": "Tennis",
            "variant": "base",
            "a": 1.0,
            "speed_ratio": speed_ratios,
            "ball_speed": [25.0 * ratio for ratio in speed_ratios],
            "max_p_winner": p_winner,
            "max_p_winner_se": p_winner_se,
            "mean_p_winner": mean_p_winner,
        }
    )
    sweep = sweep.assign(**update)
    return sweep


def _make_best_speed_rows(
    variant: str, a: float, band_ratios: Tuple[float, ...]
) -> pd.DataFrame:
    """
    Build one row of `find_best_speeds()`, by hand.

    :param variant: name of the variant, e.g., "base"
    :param a: coefficient of the sweep
    :param band_ratios: the ball speeds of the band, as `v_b / v0`
    :return: the row, with the first speed of the band as the marked one and
        a loss of the fastest ball of 10 points per speed of the band
    """
    best_speeds = pd.DataFrame(
        {
            "sport": ["Tennis"],
            "variant": [variant],
            "a": [a],
            "best_speed_ratio": [band_ratios[0]],
            "band_low_ratio": [min(band_ratios)],
            "band_high_ratio": [max(band_ratios)],
            "band_ratios": [band_ratios],
            "fastest_loss_points": [10.0 * len(band_ratios)],
        }
    )
    return best_speeds


def _get_tiny_threshold_config(
    **update: Any,
) -> racket_sports_07_speed_vs_accuracy_utils.ThresholdConfig:
    """
    Get a threshold configuration whose runs take a fraction of a second.

    :param update: fields of the configuration to change
    :return: the sweeps of `_get_tiny_config()` for both sports, with the base
        and the knee at `a = 1` at `v0` and `2 v0`, and seeds 2 and 3 for the
        tennis maps of the base
    """
    config = racket_sports_07_speed_vs_accuracy_utils.get_threshold_config(False)
    sweeps = _get_tiny_config(
        sports=("Tennis", "Pickleball"),
        speed_ratios=(1.0, 2.0),
        a_values={"base": (1.0,), "knee": (1.0,)},
        scaling_a_values=(),
    )
    config = dataclasses.replace(
        config, sweeps=sweeps, seed_variants=("base",), extra_seeds=(2, 3)
    )
    config = dataclasses.replace(config, **update)
    return config


def _make_threshold_sweep(
    sport_name: str, columns: Dict[str, List[float]]
) -> pd.DataFrame:
    """
    Build the rows of one sweep of `run_speed_sweeps()` for the threshold.

    :param sport_name: "Tennis" or "Pickleball"
    :param columns: the values and standard errors, one per ball speed from
        `v0 / 2` in steps of `v0 / 2`, e.g., `{"max_p_winner": [0.9, 0.8],
        "max_p_winner_se": [0.002, 0.002]}`
    :return: one row per ball speed, for the base at `a = 1`
    """
    n_speeds = len(next(iter(columns.values())))
    sweep = pd.DataFrame(
        {
            "sport": sport_name,
            "variant": "base",
            "a": 1.0,
            "speed_ratio": [0.5 * (i + 1) for i in range(n_speeds)],
            **columns,
        }
    )
    return sweep


def _make_band_rows(in_band: Dict[float, str], **update: Any) -> pd.DataFrame:
    """
    Build the rows of `find_threshold_bands()` of one sweep per `a`.

    :param in_band: per `a`, the band over the ball speeds `v0 / 2`, `v0`,
        `1.5 v0`, and `2 v0`, e.g., "-xx-" for `v0` and `1.5 v0` in it
    :param update: columns to change, e.g., `variant="knee"`
    :return: one row per `a` and ball speed, for the pickleball base
    """
    rows = [
        {
            "sport": "Pickleball",
            "variant": "base",
            "a": a,
            "speed_ratio": 0.5 * (i + 1),
            "tennis_se": "seeds",
            "in_band": flag == "x",
        }
        for a, flags in in_band.items()
        for i, flag in enumerate(flags)
    ]
    bands = pd.DataFrame(rows).assign(**update)
    return bands


# #############################################################################
# Test_compute_speed_factor
# #############################################################################


class Test_compute_speed_factor(hunitest.TestCase):
    """
    Test `compute_speed_factor()`.

    The speeds are binary fractions of `v0`, so every expected factor is exact.
    """

    def helper(
        self,
        ball_speed_mps: float,
        v0_mps: float,
        a: float,
        expected: Tuple[float, float, float, float],
    ) -> None:
        """
        Check the factor of every shape.

        :param ball_speed_mps: ball speed `v_b`
        :param v0_mps: reference ball speed `v0`
        :param a: coefficient of the factor
        :param expected: the factor of each shape, in the order of
            `SPREAD_SHAPES`: linear, knee, parabola, one-sided parabola
        """
        # Run test.
        actual = tuple(
            racket_sports_07_speed_vs_accuracy_utils.compute_speed_factor(
                ball_speed_mps, v0_mps, a, shape
            )
            for shape in racket_sports_07_speed_vs_accuracy_utils.SPREAD_SHAPES
        )
        # Check outputs.
        self.assertEqual(actual, expected)

    def test1(self) -> None:
        """
        Test that `a = 0` gives 1 for every shape: the Milestone 2 model.
        """
        # Prepare inputs.
        ball_speed_mps = 50.0
        v0_mps = 25.0
        a = 0.0
        # Prepare outputs.
        expected = (1.0, 1.0, 1.0, 1.0)
        # Run test.
        self.helper(ball_speed_mps, v0_mps, a, expected)

    def test2(self) -> None:
        """
        Test that `v_b = v0` gives 1 for every shape, whatever `a` is.
        """
        # Prepare inputs.
        ball_speed_mps = 25.0
        v0_mps = 25.0
        a = 1.0
        # Prepare outputs.
        expected = (1.0, 1.0, 1.0, 1.0)
        # Run test.
        self.helper(ball_speed_mps, v0_mps, a, expected)

    def test3(self) -> None:
        """
        Test a ball slower than `v0`: more accurate with the linear shape,
        less accurate with the parabola, as accurate with the one-sided ones.
        """
        # Prepare inputs.
        ball_speed_mps = 12.5
        v0_mps = 25.0
        a = 1.0
        # Prepare outputs.
        expected = (0.5, 1.0, 1.25, 1.0)
        # Run test.
        self.helper(ball_speed_mps, v0_mps, a, expected)

    def test4(self) -> None:
        """
        Test a ball faster than `v0`: the one-sided shapes equal the two-sided
        ones, and the parabolas grow more slowly than the linear shapes.
        """
        # Prepare inputs.
        ball_speed_mps = 37.5
        v0_mps = 25.0
        a = 1.0
        # Prepare outputs.
        expected = (1.5, 1.5, 1.25, 1.25)
        # Run test.
        self.helper(ball_speed_mps, v0_mps, a, expected)

    def test5(self) -> None:
        """
        Test that every shape gives `1 + a` at `v_b = 2 v0`.
        """
        # Prepare inputs.
        ball_speed_mps = 50.0
        v0_mps = 25.0
        a = 0.5
        # Prepare outputs.
        expected = (1.5, 1.5, 1.5, 1.5)
        # Run test.
        self.helper(ball_speed_mps, v0_mps, a, expected)

    def test6(self) -> None:
        """
        Test that the factor depends on the speeds only through `v_b / v0`,
        with the pickleball `v0` and the same ratio as `test4()`.
        """
        # Prepare inputs.
        ball_speed_mps = 18.0
        v0_mps = 12.0
        a = 1.0
        # Prepare outputs.
        expected = (1.5, 1.5, 1.25, 1.25)
        # Run test.
        self.helper(ball_speed_mps, v0_mps, a, expected)

    def test7(self) -> None:
        """
        Test that `a > 1` is allowed while the factor stays positive.
        """
        # Prepare inputs.
        ball_speed_mps = 12.5
        v0_mps = 25.0
        a = 1.5
        # Prepare outputs.
        expected = (0.25, 1.0, 1.375, 1.0)
        # Run test.
        self.helper(ball_speed_mps, v0_mps, a, expected)

    def test8(self) -> None:
        """
        Test that a factor of 0 is rejected: the linear shape with `a = 2` at
        `v_b = v0 / 2`.
        """
        # Prepare inputs.
        ball_speed_mps = 12.5
        v0_mps = 25.0
        a = 2.0
        shape = "linear"
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            racket_sports_07_speed_vs_accuracy_utils.compute_speed_factor(
                ball_speed_mps, v0_mps, a, shape
            )

    def test9(self) -> None:
        """
        Test that bad inputs are rejected: a negative or `nan` coefficient, an
        unknown shape, and a ball speed that is not positive.
        """
        # Prepare inputs: `(ball_speed_mps, v0_mps, a, shape)`.
        bad_inputs = (
            (50.0, 25.0, -0.5, "linear"),
            (50.0, 25.0, float("nan"), "linear"),
            (50.0, 25.0, 0.5, "cubic"),
            (0.0, 25.0, 0.5, "knee"),
        )
        # Run test and check outputs.
        for inputs in bad_inputs:
            with self.subTest(inputs=inputs), self.assertRaises(AssertionError):
                racket_sports_07_speed_vs_accuracy_utils.compute_speed_factor(
                    *inputs
                )


# #############################################################################
# Test_build_spread_from_ball_speed
# #############################################################################


class Test_build_spread_from_ball_speed(hunitest.TestCase):
    """
    Test `build_spread_from_ball_speed()`.

    The spread at `v0 = 25` m/s is the Milestone 2 default, `(0.5, 1.0)` m.
    """

    def helper(
        self,
        ball_speed_mps: float,
        a_x: float,
        a_y: float,
        shape: str,
        expected: Tuple[float, float],
    ) -> None:
        """
        Check the spread at a ball speed.

        :param ball_speed_mps: ball speed `v_b`
        :param a_x: coefficient of the width factor
        :param a_y: coefficient of the depth factor
        :param shape: one of `SPREAD_SHAPES`
        :param expected: `(std_x2, std_y2)`
        """
        # Prepare inputs.
        std0_x = 0.5
        std0_y = 1.0
        v0_mps = 25.0
        # Run test.
        actual = racket_sports_07_speed_vs_accuracy_utils.build_spread_from_ball_speed(
            ball_speed_mps, std0_x, std0_y, v0_mps, a_x, a_y, shape
        )
        # Check outputs.
        self.assertEqual(actual, expected)

    def test1(self) -> None:
        """
        Test that `a_x = a_y = 0` returns the spread at `v0` exactly.
        """
        # Prepare inputs.
        ball_speed_mps = 50.0
        a_x = 0.0
        a_y = 0.0
        shape = "linear"
        # Prepare outputs.
        expected = (0.5, 1.0)
        # Run test.
        self.helper(ball_speed_mps, a_x, a_y, shape, expected)

    def test2(self) -> None:
        """
        Test that `a_y` acts only on the depth: with `a_x = 0` the width keeps
        its spread at `v0`.
        """
        # Prepare inputs.
        ball_speed_mps = 50.0
        a_x = 0.0
        a_y = 1.0
        shape = "linear"
        # Prepare outputs.
        expected = (0.5, 2.0)
        # Run test.
        self.helper(ball_speed_mps, a_x, a_y, shape, expected)

    def test3(self) -> None:
        """
        Test `a_x = a / 2` and `a_y = a` with `a = 1`: each direction gets its
        own factor.
        """
        # Prepare inputs.
        ball_speed_mps = 37.5
        a_x = 0.5
        a_y = 1.0
        shape = "linear"
        # Prepare outputs.
        expected = (0.625, 1.5)
        # Run test.
        self.helper(ball_speed_mps, a_x, a_y, shape, expected)

    def test4(self) -> None:
        """
        Test that `a_x = a_y` gives both directions the same factor, here the
        parabola below `v0`.
        """
        # Prepare inputs.
        ball_speed_mps = 12.5
        a_x = 1.0
        a_y = 1.0
        shape = "parabola"
        # Prepare outputs.
        expected = (0.625, 1.25)
        # Run test.
        self.helper(ball_speed_mps, a_x, a_y, shape, expected)

    def test5(self) -> None:
        """
        Test that a spread at `v0` that is not positive is rejected.
        """
        # Prepare inputs.
        ball_speed_mps = 50.0
        std0_y = 1.0
        v0_mps = 25.0
        a_x = 0.5
        a_y = 1.0
        shape = "linear"
        # Run test and check outputs.
        for std0_x in (0.0, float("nan")):
            with self.subTest(std0_x=std0_x), self.assertRaises(AssertionError):
                racket_sports_07_speed_vs_accuracy_utils.build_spread_from_ball_speed(
                    ball_speed_mps, std0_x, std0_y, v0_mps, a_x, a_y, shape
                )


# #############################################################################
# Test_plot_speed_factors
# #############################################################################


class Test_plot_speed_factors(hunitest.TestCase):
    """
    Test `plot_speed_factors()`.
    """

    def test1(self) -> None:
        """
        Test one panel per `a`, with one curve per shape, each at `1 + a` at
        `2 v0`.
        """
        # Prepare inputs.
        a_values = [0.5, 1.0]
        # Prepare outputs.
        expected = "[[1.5, 1.5, 1.5, 1.5], [2.0, 2.0, 2.0, 2.0]]"
        # Run test.
        fig = racket_sports_07_speed_vs_accuracy_utils.plot_speed_factors(
            a_values
        )
        # Check outputs.
        self.assertEqual(len(fig.axes), 2)
        # The 4 shapes, then the vertical line at `v0`.
        actual = [
            [float(line.get_ydata()[-1]) for line in ax.get_lines()[:4]]
            for ax in fig.axes
        ]
        self.assert_equal(str(actual), expected)
        plt.close("all")

    def test2(self) -> None:
        """
        Test that no value of `a` is rejected.
        """
        # Prepare inputs.
        a_values: List[float] = []
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            racket_sports_07_speed_vs_accuracy_utils.plot_speed_factors(a_values)


# #############################################################################
# Test_get_experiment_config
# #############################################################################


class Test_get_experiment_config(hunitest.TestCase):
    """
    Test `get_experiment_config()`.
    """

    def test1(self) -> None:
        """
        Test the full run: 16 ball speeds from `v0 / 2` to `2 v0`, the base at
        5 values of `a`, and 10 variants.
        """
        # Prepare inputs.
        run_full = True
        # Prepare outputs.
        expected = """
        {'n_samples': 5000,
         'speed_ratios': (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                          1.6, 1.7, 1.8, 1.9, 2.0),
         'a_values': {'base': (0.0, 0.25, 0.5, 0.75, 1.0),
                      'knee': (0.5, 1.0),
                      'parabola': (0.5, 1.0),
                      'one_sided_parabola': (0.5, 1.0),
                      'a_x_eq_a_y': (0.5, 1.0),
                      'std0_half': (0.5, 1.0),
                      'std0_double': (0.5, 1.0),
                      'v_p_1.9': (0.5, 1.0),
                      'v_p_3.0': (0.5, 1.0),
                      'serve_T_deep': (0.5, 1.0),
                      'seed_2': (1.0,)}}
        """
        # Run test.
        config = racket_sports_07_speed_vs_accuracy_utils.get_experiment_config(
            run_full
        )
        # Check outputs.
        actual = {
            "n_samples": config.n_samples,
            "speed_ratios": config.speed_ratios,
            "a_values": config.a_values,
        }
        actual = pprint.pformat(actual, sort_dicts=False)
        self.assert_equal(actual, expected, dedent=True, fuzzy_match=True)

    def test2(self) -> None:
        """
        Test that the reduced run is a part of the full run, with `v0` and
        both ends of the ball speeds.
        """
        # Prepare inputs.
        full = racket_sports_07_speed_vs_accuracy_utils.get_experiment_config(
            True
        )
        # Run test.
        reduced = racket_sports_07_speed_vs_accuracy_utils.get_experiment_config(
            False
        )
        # Check outputs.
        self.assertLess(reduced.n_samples, full.n_samples)
        self.assertLess(set(reduced.speed_ratios), set(full.speed_ratios))
        self.assertLess({0.5, 1.0, 2.0}, set(reduced.speed_ratios))
        self.assertLess(set(reduced.a_values), set(full.a_values))
        for variant, a_values in reduced.a_values.items():
            with self.subTest(variant=variant):
                self.assertLessEqual(set(a_values), set(full.a_values[variant]))

    def test3(self) -> None:
        """
        Test that a value that is not a bool is rejected.
        """
        # Prepare inputs.
        run_full = 1
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            racket_sports_07_speed_vs_accuracy_utils.get_experiment_config(
                run_full
            )


# #############################################################################
# Test_count_maps
# #############################################################################


class Test_count_maps(hunitest.TestCase):
    """
    Test `count_maps()`.
    """

    def helper(self, run_full: bool, expected: int) -> None:
        """
        Check the number of maps of a run.

        :param run_full: True for the full run, False for the reduced one
        :param expected: the number of maps
        """
        # Prepare inputs.
        config = racket_sports_07_speed_vs_accuracy_utils.get_experiment_config(
            run_full
        )
        # Run test.
        actual = racket_sports_07_speed_vs_accuracy_utils.count_maps(config)
        # Check outputs.
        self.assertEqual(actual, expected)

    def test1(self) -> None:
        """
        Test the full run: 160 maps for the base, 576 for the 9 variants at 2
        values of `a`, 32 for the second seed, and 18 for the scaling check.
        """
        # Prepare inputs.
        run_full = True
        # Prepare outputs.
        expected = 160 + 576 + 32 + 18
        # Run test.
        self.helper(run_full, expected)

    def test2(self) -> None:
        """
        Test the reduced run: 54 maps for the base and 126 for the 7 variants,
        all on 9 ball speeds, and 18 for the scaling check.
        """
        # Prepare inputs.
        run_full = False
        # Prepare outputs.
        expected = 54 + 126 + 18
        # Run test.
        self.helper(run_full, expected)


# #############################################################################
# Test_build_sweep_spec
# #############################################################################


class Test_build_sweep_spec(hunitest.TestCase):
    """
    Test `build_sweep_spec()`.
    """

    def helper(
        self, sport_name: str, variant: str, a: float, speed_ratio: float
    ) -> Dict[str, Any]:
        """
        Build one run on the tiny grid, with its settings as top-level items.

        :param sport_name: "Tennis" or "Pickleball"
        :param variant: name of the variant, e.g., "base"
        :param a: coefficient of the depth factor
        :param speed_ratio: ball speed as `v_b / v0`
        :return: the builder inputs and the settings of the map, in one
            dictionary
        """
        config = _get_tiny_config()
        spec = racket_sports_07_speed_vs_accuracy_utils.build_sweep_spec(
            sport_name, variant, a, speed_ratio, config
        )
        flat_spec = {
            name: value for name, value in spec.items() if name != "settings"
        }
        flat_spec.update(spec["settings"])
        return flat_spec

    def test1(self) -> None:
        """
        Test the base in tennis at `1.5 v0`: `a_x = a / 2`, the spread from
        the builder, and the other settings at the defaults of notebook 05.
        """
        # Prepare inputs.
        sport_name = "Tennis"
        variant = "base"
        a = 1.0
        speed_ratio = 1.5
        # Prepare outputs.
        expected = """
        {'variant': 'base',
         'shape': 'linear',
         'a': 1.0,
         'a_x': 0.5,
         'a_y': 1.0,
         'v0': 25.0,
         'std0_x': 0.5,
         'std0_y': 1.0,
         'speed_ratio': 1.5,
         'sport': 'Tennis',
         'side': 'deuce',
         'x1': -2.1,
         'y1': 3.2,
         'std_x2': 0.625,
         'std_y2': 1.5,
         'ball_speed': 37.5,
         't_r': 0.2,
         'v_p': 1.5,
         'seed': 1,
         'n_x': 3,
         'n_y': 4,
         'n_samples': 1100}
        """
        # Run test.
        actual = self.helper(sport_name, variant, a, speed_ratio)
        # Check outputs.
        actual = pprint.pformat(actual, sort_dicts=False)
        self.assert_equal(actual, expected, dedent=True, fuzzy_match=True)

    def test2(self) -> None:
        """
        Test that each variant changes one thing from the base, and nothing
        else; the 2 spreads are left out, since the builder computes them.
        """
        # Prepare inputs.
        sport_name = "Pickleball"
        a = 1.0
        speed_ratio = 0.5
        variants = (
            racket_sports_07_speed_vs_accuracy_utils.get_experiment_config(
                True
            ).a_values
        )
        # Prepare outputs.
        expected = """
        {'knee': ['shape'],
         'parabola': ['shape'],
         'one_sided_parabola': ['shape'],
         'a_x_eq_a_y': ['a_x'],
         'std0_half': ['std0_x', 'std0_y'],
         'std0_double': ['std0_x', 'std0_y'],
         'v_p_1.9': ['v_p'],
         'v_p_3.0': ['v_p'],
         'serve_T_deep': ['x1', 'y1'],
         'seed_2': ['seed']}
        """
        # Run test.
        base = self.helper(sport_name, "base", a, speed_ratio)
        actual = {}
        for variant in variants:
            spec = self.helper(sport_name, variant, a, speed_ratio)
            actual[variant] = [
                name
                for name in base
                if spec[name] != base[name]
                and name not in ("variant", "std_x2", "std_y2")
            ]
        del actual["base"]
        # Check outputs.
        actual = pprint.pformat(actual, sort_dicts=False)
        self.assert_equal(actual, expected, dedent=True, fuzzy_match=True)

    def test3(self) -> None:
        """
        Test that the spread of every variant comes from its own inputs: the
        spread at `v0` at `v_b = v0`, and `1 + a` times it at `2 v0`, where
        all the shapes agree.
        """
        # Prepare inputs.
        sport_name = "Tennis"
        a = 1.0
        variants = (
            racket_sports_07_speed_vs_accuracy_utils.get_experiment_config(
                True
            ).a_values
        )
        # Run test and prepare outputs: the expected spread comes from the
        # inputs of each run.
        actual = {}
        expected = {}
        for variant in variants:
            at_v0 = self.helper(sport_name, variant, a, 1.0)
            at_2v0 = self.helper(sport_name, variant, a, 2.0)
            actual[variant] = [
                (at_v0["std_x2"], at_v0["std_y2"]),
                (at_2v0["std_x2"], at_2v0["std_y2"]),
            ]
            expected[variant] = [
                (at_v0["std0_x"], at_v0["std0_y"]),
                (
                    (1 + at_2v0["a_x"]) * at_2v0["std0_x"],
                    (1 + at_2v0["a_y"]) * at_2v0["std0_y"],
                ),
            ]
        # Check outputs.
        self.assert_equal(str(actual), str(expected))

    def test4(self) -> None:
        """
        Test that the ball speed reads as a round number: 8.4 m/s for `0.7 *
        12`, and not 8.399999999999999.
        """
        # Prepare inputs.
        sport_name = "Pickleball"
        variant = "base"
        a = 0.5
        speed_ratio = 0.7
        # Prepare outputs.
        expected = 8.4
        # Run test.
        actual = self.helper(sport_name, variant, a, speed_ratio)
        # Check outputs.
        self.assertEqual(actual["ball_speed"], expected)

    def test5(self) -> None:
        """
        Test that the settings of a run survive the settings string of
        notebook 05, so the map can be rebuilt from it exactly.
        """
        # Prepare inputs.
        sport_name = "Pickleball"
        variant = "std0_half"
        a = 0.75
        speed_ratio = 0.7
        config = _get_tiny_config()
        # Run test.
        spec = racket_sports_07_speed_vs_accuracy_utils.build_sweep_spec(
            sport_name, variant, a, speed_ratio, config
        )
        text = racket_sports_05_variable_sensitivity_utils.settings_to_string(
            spec["settings"]
        )
        actual = (
            racket_sports_05_variable_sensitivity_utils.parse_settings_string(
                text
            )
        )
        # Check outputs.
        self.assert_equal(str(actual), str(spec["settings"]))

    def test6(self) -> None:
        """
        Test that an unknown variant and a negative coefficient are rejected.
        """
        # Prepare inputs: `(variant, a)`.
        bad_inputs = (("cubic", 0.5), ("base", -0.5))
        sport_name = "Tennis"
        speed_ratio = 1.0
        # Run test and check outputs.
        for variant, a in bad_inputs:
            with (
                self.subTest(variant=variant),
                self.assertRaises(AssertionError),
            ):
                self.helper(sport_name, variant, a, speed_ratio)


# #############################################################################
# Test_run_speed_sweeps
# #############################################################################


class Test_run_speed_sweeps(hunitest.TestCase):
    """
    Test `run_speed_sweeps()`.
    """

    def helper(self) -> pd.DataFrame:
        """
        Run the sweeps of the tiny configuration.

        :return: 9 maps: the base at `a` = 0 and 1 and the knee at `a = 1`,
            each at `v0 / 2`, `v0`, and `2 v0`
        """
        config = _get_tiny_config()
        sweeps = racket_sports_07_speed_vs_accuracy_utils.run_speed_sweeps(
            config
        )
        return sweeps

    def test1(self) -> None:
        """
        Test the rows: one per sport, variant, `a`, and ball speed, with the
        builder inputs and the spread of each map as columns.
        """
        # Prepare inputs.
        columns = ["variant", "shape", "a", "a_x", "a_y", "v0", "std0_x"]
        columns += ["std0_y", "speed_ratio", "ball_speed", "std_x2", "std_y2"]
        # Prepare outputs.
        expected = """
          variant   shape    a  a_x  a_y    v0  std0_x  std0_y  speed_ratio  ball_speed  std_x2  std_y2
        0    base  linear  0.0  0.0  0.0  12.0     0.5     1.0          0.5         6.0     0.5     1.0
        1    base  linear  0.0  0.0  0.0  12.0     0.5     1.0          1.0        12.0     0.5     1.0
        2    base  linear  0.0  0.0  0.0  12.0     0.5     1.0          2.0        24.0     0.5     1.0
        3    base  linear  1.0  0.5  1.0  12.0     0.5     1.0          0.5         6.0   0.375     0.5
        4    base  linear  1.0  0.5  1.0  12.0     0.5     1.0          1.0        12.0     0.5     1.0
        5    base  linear  1.0  0.5  1.0  12.0     0.5     1.0          2.0        24.0    0.75     2.0
        6    knee    knee  1.0  0.5  1.0  12.0     0.5     1.0          0.5         6.0     0.5     1.0
        7    knee    knee  1.0  0.5  1.0  12.0     0.5     1.0          1.0        12.0     0.5     1.0
        8    knee    knee  1.0  0.5  1.0  12.0     0.5     1.0          2.0        24.0    0.75     2.0
        """
        # Run test.
        actual = self.helper()
        # Check outputs.
        actual = _to_str(actual[columns])
        self.assert_equal(actual, expected, dedent=True, fuzzy_match=True)

    def test2(self) -> None:
        """
        Test that the rows at `a = 0` are the Milestone 2 model: the runs of
        `compute_run()` of notebook 05 at the same ball speeds, with its
        default spread.
        """
        # Prepare inputs.
        settings = (
            racket_sports_05_variable_sensitivity_utils.get_default_settings(
                "Pickleball", n_samples=1100
            )
        )
        settings.update(n_x=3, n_y=4)
        ball_speeds = [6.0, 12.0, 24.0]
        # Prepare outputs.
        expected = pd.DataFrame(
            [
                racket_sports_05_variable_sensitivity_utils.compute_run(
                    dict(settings, ball_speed=ball_speed)
                )[0]
                for ball_speed in ball_speeds
            ]
        )
        # Run test.
        actual = self.helper()
        # Check outputs.
        actual = actual.loc[actual["a"] == 0.0, expected.columns]
        self.assert_equal(_to_str(actual), _to_str(expected))

    def test3(self) -> None:
        """
        Test that at `v_b = v0` the map depends neither on `a` nor on the
        shape, and that all the maps share their seed.
        """
        # Prepare outputs.
        expected = (
            "sport=Pickleball side=deuce x1=-1.5 y1=4.4 std_x2=0.5 std_y2=1.0"
            " ball_speed=12.0 t_r=0.2 v_p=1.5 seed=1 n_x=3 n_y=4 n_samples=1100"
        )
        # Run test.
        actual = self.helper()
        # Check outputs.
        at_v0 = actual.loc[actual["speed_ratio"] == 1.0, "settings"]
        self.assertEqual(len(at_v0), 3)
        self.assert_equal("\n".join(at_v0.unique()), expected)
        self.assertEqual(actual["seed"].unique().tolist(), [1])

    def test4(self) -> None:
        """
        Test that the knee is the Milestone 2 model below `v0` and the linear
        shape above it, map by map.
        """
        # Run test.
        actual = self.helper()
        # Check outputs.
        maps = actual.set_index(["variant", "a", "speed_ratio"])["settings"]
        knee = maps["knee", 1.0]
        self.assert_equal(knee[0.5], maps["base", 0.0, 0.5])
        self.assert_equal(knee[2.0], maps["base", 1.0, 2.0])
        self.assertNotEqual(knee[0.5], maps["base", 1.0, 0.5])

    def test5(self) -> None:
        """
        Test that the settings string of a row rebuilds its map in notebook
        05, with the same summary.
        """
        # Run test.
        actual = self.helper()
        # Check outputs.
        row = actual.iloc[5]
        settings = (
            racket_sports_05_variable_sensitivity_utils.parse_settings_string(
                row["settings"]
            )
        )
        expected, _ = racket_sports_05_variable_sensitivity_utils.compute_run(
            settings
        )
        actual_summary = row[list(expected)].to_dict()
        self.assert_equal(str(actual_summary), str(expected))

    def test6(self) -> None:
        """
        Test that speed costs accuracy with the linear shape at `a = 1`: the
        mean `P(winner)` falls at `2 v0` and rises at `v0 / 2`, against the
        Milestone 2 model.
        """
        # Run test.
        actual = self.helper()
        # Check outputs.
        is_base = actual["variant"] == "base"
        mean = actual[is_base].set_index(["a", "speed_ratio"])["mean_p_winner"]
        self.assertLess(mean[1.0, 2.0], mean[0.0, 2.0] - 0.05)
        self.assertGreater(mean[1.0, 0.5], mean[0.0, 0.5] + 0.02)

    def test7(self) -> None:
        """
        Test that `mean_p_winner_se` of a row is the mean of the standard
        errors of the cells of its map.
        """
        # Run test.
        actual = self.helper()
        # Check outputs.
        row = actual.iloc[5]
        settings = (
            racket_sports_05_variable_sensitivity_utils.parse_settings_string(
                row["settings"]
            )
        )
        _, heatmap = racket_sports_05_variable_sensitivity_utils.compute_run(
            settings
        )
        expected = float(heatmap["p_winner_se"].mean())
        self.assertGreater(expected, 0.0)
        self.assertEqual(row["mean_p_winner_se"], expected)


# #############################################################################
# Test_plot_sweep_maps
# #############################################################################


class Test_plot_sweep_maps(hunitest.TestCase):
    """
    Test `plot_sweep_maps()`.
    """

    def test1(self) -> None:
        """
        Test one map per ball speed, each titled with its speed and spread.
        """
        # Prepare inputs.
        sport_name = "Pickleball"
        variant = "base"
        a = 1.0
        speed_ratios = [0.5, 2.0]
        config = _get_tiny_config()
        # Prepare outputs.
        expected = """
        v_b = 0.5 v0 = 6 m/s
        spread (0.38, 0.50) m
        v_b = 2 v0 = 24 m/s
        spread (0.75, 2.00) m
        """
        # Run test.
        fig = racket_sports_07_speed_vs_accuracy_utils.plot_sweep_maps(
            sport_name, variant, a, speed_ratios, config
        )
        # Check outputs.
        # One map and one colorbar per panel, as in `plot_montage()`.
        self.assertEqual(len(fig.axes), 2 + 2)
        # The first 2 lines of each map title; the third has the result.
        actual = "\n".join(
            "\n".join(ax.get_title().split("\n")[:2]) for ax in fig.axes[:2]
        )
        self.assert_equal(actual, expected, dedent=True)
        plt.close("all")


# #############################################################################
# Test_find_best_speeds
# #############################################################################


class Test_find_best_speeds(hunitest.TestCase):
    """
    Test `find_best_speeds()`.
    """

    def helper(
        self,
        p_winner: List[float],
        p_winner_se: float,
        mean_p_winner: List[float],
    ) -> pd.Series:
        """
        Find the best ball speed of one sweep, at `v0 / 2`, `v0`, `1.5 v0`,
        and `2 v0`, with `v0` = 25 m/s.

        :param p_winner: the highest `P(winner)` per ball speed
        :param p_winner_se: the standard error of each value
        :param mean_p_winner: the mean `P(winner)` over the half per ball speed
        :return: the only row of `find_best_speeds()`
        """
        sweeps = _make_sweep_rows(p_winner, p_winner_se, mean_p_winner)
        best_speeds = racket_sports_07_speed_vs_accuracy_utils.find_best_speeds(
            sweeps
        )
        self.assertEqual(len(best_speeds), 1)
        row: pd.Series = best_speeds.round(6).iloc[0]
        return row

    def test1(self) -> None:
        """
        Test a clear best speed: the others are more than 2 SE away, and the
        mean over the half peaks at another speed.
        """
        # Prepare inputs.
        p_winner = [0.9, 0.99, 0.95, 0.8]
        p_winner_se = 0.002
        mean_p_winner = [0.6, 0.7, 0.8, 0.75]
        # Prepare outputs.
        expected = """
        sport                       Tennis
        variant                       base
        a                              1.0
        best_speed_ratio               1.0
        best_ball_speed               25.0
        band_low_ratio                 1.0
        band_high_ratio                1.0
        band_ratios                 (1.0,)
        n_band                           1
        n_speeds                         4
        slowest_in_band              False
        fastest_in_band              False
        top_p_winner                  0.99
        fastest_p_winner               0.8
        fastest_loss_points           19.0
        best_mean_speed_ratio          1.5
        best_mean_p_winner             0.8
        fastest_mean_loss_points       5.0
        """
        # Run test.
        actual = self.helper(p_winner, p_winner_se, mean_p_winner)
        # Check outputs.
        actual = actual.to_string()
        self.assert_equal(actual, expected, dedent=True, fuzzy_match=True)

    def test2(self) -> None:
        """
        Test that every speed is in the band when all tie at 1.000 with no
        standard error, as in tennis: the fastest ball loses nothing.
        """
        # Prepare inputs.
        p_winner = [1.0, 1.0, 1.0, 1.0]
        p_winner_se = 0.0
        mean_p_winner = [0.6, 0.7, 0.8, 0.9]
        names = ["band_ratios", "fastest_in_band", "fastest_loss_points"]
        # Prepare outputs.
        expected = """
        band_ratios            (0.5, 1.0, 1.5, 2.0)
        fastest_in_band                        True
        fastest_loss_points                     0.0
        """
        # Run test.
        actual = self.helper(p_winner, p_winner_se, mean_p_winner)
        # Check outputs.
        actual = actual[names].to_string()
        self.assert_equal(actual, expected, dedent=True, fuzzy_match=True)

    def test3(self) -> None:
        """
        Test a band of 3 speeds within 2 SE of the highest: the marked speed
        is the middle of the band, not the highest value.
        """
        # Prepare inputs.
        p_winner = [0.985, 0.985, 0.99, 0.8]
        p_winner_se = 0.004
        mean_p_winner = [0.9, 0.8, 0.7, 0.6]
        names = ["band_ratios", "best_speed_ratio", "slowest_in_band"]
        # Prepare outputs.
        expected = """
        band_ratios         (0.5, 1.0, 1.5)
        best_speed_ratio                1.0
        slowest_in_band                True
        """
        # Run test.
        actual = self.helper(p_winner, p_winner_se, mean_p_winner)
        # Check outputs.
        actual = actual[names].to_string()
        self.assert_equal(actual, expected, dedent=True, fuzzy_match=True)

    def test4(self) -> None:
        """
        Test that with no standard error only an equal value is in the band,
        so a band can skip a speed.
        """
        # Prepare inputs.
        p_winner = [1.0, 0.999, 1.0, 0.5]
        p_winner_se = 0.0
        mean_p_winner = [0.6, 0.7, 0.8, 0.75]
        names = ["band_ratios", "n_band"]
        # Prepare outputs.
        expected = """
        band_ratios    (0.5, 1.5)
        n_band                  2
        """
        # Run test.
        actual = self.helper(p_winner, p_winner_se, mean_p_winner)
        # Check outputs.
        actual = actual[names].to_string()
        self.assert_equal(actual, expected, dedent=True, fuzzy_match=True)

    def test5(self) -> None:
        """
        Test one row per sport, variant, and `a`, in the order of the runs,
        with the rows of each sweep in any order: the pickleball rows come
        fastest first, and its fastest ball still loses 20 points.
        """
        # Prepare inputs.
        p_winner_se = 0.002
        mean_p_winner = [0.5, 0.5, 0.5]
        sweeps = pd.concat(
            [
                _make_sweep_rows([0.9, 0.8, 0.7], p_winner_se, mean_p_winner),
                _make_sweep_rows(
                    [0.7, 0.9, 0.8], p_winner_se, mean_p_winner, variant="knee"
                ),
                _make_sweep_rows(
                    [0.7, 0.8, 0.9], p_winner_se, mean_p_winner, a=0.5
                ),
                _make_sweep_rows(
                    [0.9, 0.8, 0.7],
                    p_winner_se,
                    mean_p_winner,
                    sport="Pickleball",
                ).iloc[::-1],
            ],
            ignore_index=True,
        )
        # Prepare outputs.
        expected = """
                sport variant    a  best_speed_ratio  fastest_loss_points
        0      Tennis    base  1.0               0.5                 20.0
        1      Tennis    knee  1.0               1.0                 10.0
        2      Tennis    base  0.5               1.5                  0.0
        3  Pickleball    base  1.0               0.5                 20.0
        """
        # Run test.
        actual = racket_sports_07_speed_vs_accuracy_utils.find_best_speeds(
            sweeps
        )
        # Check outputs.
        columns = ["sport", "variant", "a", "best_speed_ratio"]
        columns += ["fastest_loss_points"]
        actual = _to_str(actual[columns].round(6))
        self.assert_equal(actual, expected, dedent=True, fuzzy_match=True)

    def test6(self) -> None:
        """
        Test that no run, and a sweep with one ball speed, are rejected.
        """
        # Prepare inputs.
        bad_inputs = (pd.DataFrame(), _make_sweep_rows([0.9], 0.002, [0.5]))
        # Run test and check outputs.
        for sweeps in bad_inputs:
            with (
                self.subTest(n_rows=len(sweeps)),
                self.assertRaises(AssertionError),
            ):
                racket_sports_07_speed_vs_accuracy_utils.find_best_speeds(sweeps)


# #############################################################################
# Test_summarize_robustness
# #############################################################################


class Test_summarize_robustness(hunitest.TestCase):
    """
    Test `summarize_robustness()`.
    """

    def helper(
        self, base_band: Tuple[float, ...], variant_band: Tuple[float, ...]
    ) -> pd.Series:
        """
        Compare the band of a variant with the band of the base, at `a = 1`.

        :param base_band: the ball speeds of the band of the base
        :param variant_band: the ball speeds of the band of the variant
        :return: the only row of `summarize_robustness()`
        """
        best_speeds = pd.concat(
            [
                _make_best_speed_rows("base", 1.0, base_band),
                _make_best_speed_rows("knee", 1.0, variant_band),
            ],
            ignore_index=True,
        )
        robustness = (
            racket_sports_07_speed_vs_accuracy_utils.summarize_robustness(
                best_speeds
            )
        )
        self.assertEqual(len(robustness), 1)
        row: pd.Series = robustness.iloc[0]
        return row

    def test1(self) -> None:
        """
        Test 2 bands that share ball speeds: the best speed is robust to the
        variant.
        """
        # Prepare inputs.
        base_band = (0.5, 1.0, 1.5)
        variant_band = (1.0, 1.5, 2.0, 2.5)
        # Prepare outputs.
        expected = """
        sport                    Tennis
        variant                    knee
        a                           1.0
        base_band_low_ratio         0.5
        base_band_high_ratio        1.5
        band_low_ratio              1.0
        band_high_ratio             2.5
        n_shared                      2
        overlaps                   True
        d_best_speed_ratio          0.5
        d_fastest_loss_points      10.0
        """
        # Run test.
        actual = self.helper(base_band, variant_band)
        # Check outputs.
        actual = actual.to_string()
        self.assert_equal(actual, expected, dedent=True, fuzzy_match=True)

    def test2(self) -> None:
        """
        Test 2 bands apart: the best speed is not robust to the variant.
        """
        # Prepare inputs.
        base_band = (1.5, 2.0)
        variant_band = (0.5,)
        names = ["n_shared", "overlaps", "d_best_speed_ratio"]
        # Prepare outputs.
        expected = """
        n_shared                  0
        overlaps              False
        d_best_speed_ratio     -1.0
        """
        # Run test.
        actual = self.helper(base_band, variant_band)
        # Check outputs.
        actual = actual[names].to_string()
        self.assert_equal(actual, expected, dedent=True, fuzzy_match=True)

    def test3(self) -> None:
        """
        Test that a band inside the gap of the other one shares no ball
        speed: the bands compare speed by speed, not by their ends.
        """
        # Prepare inputs.
        base_band = (0.5, 1.5)
        variant_band = (1.0,)
        names = ["n_shared", "overlaps"]
        # Prepare outputs.
        expected = """
        n_shared        0
        overlaps    False
        """
        # Run test.
        actual = self.helper(base_band, variant_band)
        # Check outputs.
        actual = actual[names].to_string()
        self.assert_equal(actual, expected, dedent=True, fuzzy_match=True)

    def test4(self) -> None:
        """
        Test that a variant with no base sweep at its `a` is rejected.
        """
        # Prepare inputs.
        best_speeds = pd.concat(
            [
                _make_best_speed_rows("base", 1.0, (0.5,)),
                _make_best_speed_rows("knee", 0.5, (0.5,)),
            ],
            ignore_index=True,
        )
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            racket_sports_07_speed_vs_accuracy_utils.summarize_robustness(
                best_speeds
            )


# #############################################################################
# Test_plot_speed_sweeps
# #############################################################################


class Test_plot_speed_sweeps(hunitest.TestCase):
    """
    Test `plot_speed_sweeps()`.
    """

    def test1(self) -> None:
        """
        Test one row of 2 panels per sport, with one curve per `a` of the
        variant.
        """
        # Prepare inputs.
        config = _get_tiny_config()
        sweeps = racket_sports_07_speed_vs_accuracy_utils.run_speed_sweeps(
            config
        )
        best_speeds = racket_sports_07_speed_vs_accuracy_utils.find_best_speeds(
            sweeps
        )
        variant = "base"
        # Prepare outputs.
        expected = "['a = 0', 'a = 1']"
        # Run test.
        fig = racket_sports_07_speed_vs_accuracy_utils.plot_speed_sweeps(
            sweeps, best_speeds, variant
        )
        # Check outputs.
        self.assertEqual(len(fig.axes), 2)
        labels = [text.get_text() for text in fig.axes[0].get_legend().texts]
        self.assert_equal(str(labels), expected)
        # The curve of `a = 1` holds only the 3 maps of the base, not the knee.
        curve = [
            line
            for line in fig.axes[0].get_lines()
            if line.get_label() == "a = 1"
        ][0]
        is_base = (sweeps["variant"] == "base") & (sweeps["a"] == 1.0)
        self.assertEqual(curve.get_xdata().tolist(), [0.5, 1.0, 2.0])
        self.assertEqual(
            curve.get_ydata().tolist(),
            sweeps.loc[is_base, "max_p_winner"].tolist(),
        )
        plt.close("all")

    def test2(self) -> None:
        """
        Test that a variant without runs is rejected.
        """
        # Prepare inputs.
        sweeps = _make_sweep_rows([0.9, 0.8], 0.002, [0.5, 0.5])
        variant = "knee"
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            racket_sports_07_speed_vs_accuracy_utils.plot_speed_sweeps(
                sweeps, pd.DataFrame(), variant
            )


# #############################################################################
# Test_plot_best_speeds
# #############################################################################


class Test_plot_best_speeds(hunitest.TestCase):
    """
    Test `plot_best_speeds()`.
    """

    def test1(self) -> None:
        """
        Test the base as a line through its marked speeds, not the ends of
        its bands, and the knee as markers with error bars; the other variants
        are not drawn.
        """
        # Prepare inputs.
        # The first speed of a band is its marked speed, here not its low end.
        best_speeds = pd.concat(
            [
                _make_best_speed_rows("base", 0.0, (2.0, 1.5)),
                _make_best_speed_rows("base", 1.0, (1.0, 0.5)),
                _make_best_speed_rows("knee", 1.0, (1.0, 1.5)),
                _make_best_speed_rows("v_p_3.0", 1.0, (1.5,)),
            ],
            ignore_index=True,
        )
        # Prepare outputs.
        expected = "['linear (base)', 'knee']"
        # Run test.
        fig = racket_sports_07_speed_vs_accuracy_utils.plot_best_speeds(
            best_speeds
        )
        # Check outputs.
        self.assertEqual(len(fig.axes), 1)
        ax = fig.axes[0]
        labels = [text.get_text() for text in ax.get_legend().texts]
        self.assert_equal(str(labels), expected)
        base = ax.get_lines()[0]
        self.assertEqual(base.get_xdata().tolist(), [0.0, 1.0])
        self.assertEqual(base.get_ydata().tolist(), [2.0, 1.0])
        plt.close("all")

    def test2(self) -> None:
        """
        Test that rows without the base are rejected.
        """
        # Prepare inputs.
        best_speeds = _make_best_speed_rows("knee", 1.0, (1.0,))
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            racket_sports_07_speed_vs_accuracy_utils.plot_best_speeds(
                best_speeds
            )


# #############################################################################
# Test_check_speed_scaling
# #############################################################################


class Test_check_speed_scaling(hunitest.TestCase):
    """
    Test `check_speed_scaling()`.
    """

    def helper(
        self, variant: str, a: float, k: float, t_r: float
    ) -> Dict[str, Any]:
        """
        Run the check at `v_b = v0`, where player 1 reaches many landing
        points.

        :param variant: name of the variant, e.g., "base"
        :param a: coefficient of the depth factor
        :param k: the factor of the speeds
        :param t_r: reaction time of player 1, in seconds
        :return: the output of `check_speed_scaling()`
        """
        config = _get_tiny_config()
        spec = racket_sports_07_speed_vs_accuracy_utils.build_sweep_spec(
            "Pickleball", variant, a, 1.0, config
        )
        spec["settings"].update(v_p=6.0, t_r=t_r)
        actual: Dict[str, Any] = (
            racket_sports_07_speed_vs_accuracy_utils.check_speed_scaling(spec, k)
        )
        return actual

    def test1(self) -> None:
        """
        Test Proposition 1 at `a = 0`: with `t_r = 0`, scaling both speeds
        leaves the map unchanged, cell by cell, with or without `v0`.
        """
        # Prepare inputs.
        variant = "base"
        a = 0.0
        k = 2.0
        t_r = 0.0
        # Run test.
        actual = self.helper(variant, a, k, t_r)
        # Check outputs.
        self.assertEqual(actual["max_diff_speeds"], 0.0)
        self.assertEqual(actual["max_diff_speeds_and_v0"], 0.0)
        # The reach rule is exercised: a difference could show.
        self.assertGreater(actual["mean_p_reached"], 0.01)

    def test2(self) -> None:
        """
        Test that with `a > 0` scaling both speeds changes the map, and
        scaling `v0` with them does not, for every shape and with `a_x` equal
        to `a_y` or not.
        """
        # Prepare inputs.
        variants = (
            "base",
            "knee",
            "parabola",
            "one_sided_parabola",
            "a_x_eq_a_y",
        )
        a = 0.5
        k = 2.0
        t_r = 0.0
        # Run test and check outputs.
        for variant in variants:
            with self.subTest(variant=variant):
                actual = self.helper(variant, a, k, t_r)
                self.assertGreater(actual["max_diff_speeds"], 0.05)
                self.assertEqual(actual["max_diff_speeds_and_v0"], 0.0)
                self.assertGreater(actual["mean_p_reached"], 0.01)

    def test3(self) -> None:
        """
        Test the planted failure: at `t_r > 0` scaling the 2 speeds and `v0`
        still changes the map, since the reaction distance `v_p * t_r`
        changes.
        """
        # Prepare inputs.
        variant = "base"
        a = 0.5
        k = 2.0
        t_r = 0.2
        # Run test.
        actual = self.helper(variant, a, k, t_r)
        # Check outputs.
        self.assertGreater(actual["max_diff_speeds_and_v0"], 0.05)

    def test4(self) -> None:
        """
        Test that a factor that is not positive is rejected.
        """
        # Prepare inputs.
        variant = "base"
        a = 0.5
        t_r = 0.0
        # Run test and check outputs.
        for k in (0.0, -1.0, float("nan")):
            with self.subTest(k=k), self.assertRaises(AssertionError):
                self.helper(variant, a, k, t_r)


# #############################################################################
# Test_run_scaling_checks
# #############################################################################


class Test_run_scaling_checks(hunitest.TestCase):
    """
    Test `run_scaling_checks()`.
    """

    def test1(self) -> None:
        """
        Test one row per sport and `a`, all with `t_r = 0` at `v_b = v0`: the
        map changes with the 2 speeds only when `a > 0`, and never when `v0`
        is scaled too.
        """
        # Prepare inputs.
        config = _get_tiny_config(sports=("Tennis", "Pickleball"))
        # Prepare outputs.
        expected = """
                sport    a    k  speeds_differ  max_diff_speeds_and_v0  at_v0_with_t_r_0
        0      Tennis  0.0  2.0          False                     0.0              True
        1      Tennis  0.5  2.0           True                     0.0              True
        2  Pickleball  0.0  2.0          False                     0.0              True
        3  Pickleball  0.5  2.0           True                     0.0              True
        """
        # Run test.
        actual = racket_sports_07_speed_vs_accuracy_utils.run_scaling_checks(
            config
        )
        # Check outputs.
        actual["speeds_differ"] = actual["max_diff_speeds"] > 0.05
        actual["at_v0_with_t_r_0"] = actual["settings"].str.contains(
            r"std_x2=0\.5 std_y2=1\.0 ball_speed=(?:25|12)\.0 t_r=0\.0 "
        )
        columns = ["sport", "a", "k", "speeds_differ", "max_diff_speeds_and_v0"]
        columns += ["at_v0_with_t_r_0"]
        actual = _to_str(actual[columns])
        self.assert_equal(actual, expected, dedent=True, fuzzy_match=True)


# #############################################################################
# Test_get_threshold_config
# #############################################################################


class Test_get_threshold_config(hunitest.TestCase):
    """
    Test `get_threshold_config()`.
    """

    def test1(self) -> None:
        """
        Test the full run: `a` in steps of 0.025, up to 0.5 for the 3 other
        shapes and up to 1 for the linear sweeps, on the ball speeds and
        samples of the full run of notebook 07, and seeds 2 to 5 for the
        tennis maps of the 4 shapes.
        """
        # Prepare inputs.
        run_full = True
        # Prepare outputs.
        expected = """
        {'seed_sport': 'Tennis',
         'seed_variants': ('base', 'knee', 'parabola', 'one_sided_parabola'),
         'extra_seeds': (2, 3, 4, 5),
         'speeds_and_samples_of_notebook_07': True,
         'scaling_a_values': (),
         'steps': [0.025],
         'a_grids': {'base': (41, 0.0, 1.0),
                     'knee': (21, 0.0, 0.5),
                     'parabola': (21, 0.0, 0.5),
                     'one_sided_parabola': (21, 0.0, 0.5),
                     'std0_half': (41, 0.0, 1.0),
                     'std0_double': (41, 0.0, 1.0),
                     'v_p_1.9': (41, 0.0, 1.0),
                     'v_p_3.0': (41, 0.0, 1.0)},
         'knee_grid': (0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2,
                       0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425,
                       0.45, 0.475, 0.5)}
        """
        # Run test.
        config = racket_sports_07_speed_vs_accuracy_utils.get_threshold_config(
            run_full
        )
        # Check outputs.
        notebook_07 = (
            racket_sports_07_speed_vs_accuracy_utils.get_experiment_config(
                run_full
            )
        )
        a_values = config.sweeps.a_values
        actual = {
            "seed_sport": config.seed_sport,
            "seed_variants": config.seed_variants,
            "extra_seeds": config.extra_seeds,
            "speeds_and_samples_of_notebook_07": (
                config.sweeps.speed_ratios == notebook_07.speed_ratios
                and config.sweeps.n_samples == notebook_07.n_samples
            ),
            "scaling_a_values": config.sweeps.scaling_a_values,
            "steps": sorted(
                {
                    round(high - low, 12)
                    for grid in a_values.values()
                    for low, high in zip(grid, grid[1:])
                }
            ),
            "a_grids": {
                variant: (len(grid), min(grid), max(grid))
                for variant, grid in a_values.items()
            },
            "knee_grid": a_values["knee"],
        }
        actual = pprint.pformat(actual, sort_dicts=False)
        self.assert_equal(actual, expected, dedent=True, fuzzy_match=True)

    def test2(self) -> None:
        """
        Test that the reduced run is a part of the full run, with fewer seeds
        and without the variants of the base.
        """
        # Prepare inputs.
        full = racket_sports_07_speed_vs_accuracy_utils.get_threshold_config(
            True
        )
        # Run test.
        reduced = racket_sports_07_speed_vs_accuracy_utils.get_threshold_config(
            False
        )
        # Check outputs.
        self.assertEqual(reduced.seed_sport, full.seed_sport)
        self.assertEqual(reduced.seed_variants, full.seed_variants)
        self.assertLess(set(reduced.extra_seeds), set(full.extra_seeds))
        self.assertLess(
            set(reduced.sweeps.speed_ratios), set(full.sweeps.speed_ratios)
        )
        self.assertEqual(set(reduced.sweeps.a_values), set(full.seed_variants))
        self.assertLess(set(reduced.sweeps.a_values), set(full.sweeps.a_values))
        for variant, a_values in reduced.sweeps.a_values.items():
            with self.subTest(variant=variant):
                self.assertLess(
                    set(a_values), set(full.sweeps.a_values[variant])
                )

    def test3(self) -> None:
        """
        Test that a value that is not a bool is rejected.
        """
        # Prepare inputs.
        run_full = 1
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            racket_sports_07_speed_vs_accuracy_utils.get_threshold_config(
                run_full
            )


# #############################################################################
# Test_count_threshold_maps
# #############################################################################


class Test_count_threshold_maps(hunitest.TestCase):
    """
    Test `count_threshold_maps()`.
    """

    def helper(self, run_full: bool, expected: int) -> None:
        """
        Check the number of maps of a threshold run.

        :param run_full: True for the full run, False for the reduced one
        :param expected: the number of maps
        """
        # Prepare inputs.
        config = racket_sports_07_speed_vs_accuracy_utils.get_threshold_config(
            run_full
        )
        # Run test.
        actual = racket_sports_07_speed_vs_accuracy_utils.count_threshold_maps(
            config
        )
        # Check outputs.
        self.assertEqual(actual, expected)

    def test1(self) -> None:
        """
        Test the full run: 268 sweeps of 16 ball speeds per sport, and the
        1,664 tennis maps of the 104 sweeps of the 4 shapes again with each of
        4 seeds.
        """
        # Prepare inputs.
        run_full = True
        # Prepare outputs.
        expected = 2 * 268 * 16 + 4 * 104 * 16
        # Run test.
        self.helper(run_full, expected)

    def test2(self) -> None:
        """
        Test the reduced run: 14 sweeps of the 4 shapes on 5 ball speeds per
        sport, and the 70 tennis maps again with each of 2 seeds.
        """
        # Prepare inputs.
        run_full = False
        # Prepare outputs.
        expected = 2 * 14 * 5 + 2 * 14 * 5
        # Run test.
        self.helper(run_full, expected)


# #############################################################################
# Test_add_seed_mean_se
# #############################################################################


class Test_add_seed_mean_se(hunitest.TestCase):
    """
    Test `add_seed_mean_se()`.
    """

    def helper(self) -> pd.DataFrame:
        """
        Run the sweeps of the tiny threshold configuration.

        :return: 8 maps: the base and the knee at `a = 1`, at `v0` and `2 v0`,
            for each sport, all with seed 1
        """
        config = _get_tiny_threshold_config()
        sweeps = racket_sports_07_speed_vs_accuracy_utils.run_speed_sweeps(
            config.sweeps
        )
        return sweeps

    def test1(self) -> None:
        """
        Test that a tennis map gets the average of its mean `P(winner)` with
        seeds 1 to 3, and their standard deviation over `sqrt(3)`.
        """
        # Prepare inputs.
        config = _get_tiny_threshold_config()
        sweeps = self.helper()
        # Prepare outputs: the first map, a tennis one, with each seed.
        settings = (
            racket_sports_05_variable_sensitivity_utils.parse_settings_string(
                sweeps.loc[0, "settings"]
            )
        )
        means = [
            racket_sports_05_variable_sensitivity_utils.compute_run(
                dict(settings, seed=seed)
            )[0]["mean_p_winner"]
            for seed in (1, 2, 3)
        ]
        expected = [np.mean(means), np.std(means, ddof=1) / np.sqrt(3)]
        # Run test.
        actual = racket_sports_07_speed_vs_accuracy_utils.add_seed_mean_se(
            sweeps, config
        )
        # Check outputs.
        row = actual.loc[0]
        self.assertEqual(row["sport"], "Tennis")
        self.assertGreater(expected[1], 0.0)
        np.testing.assert_allclose(
            [row["mean_p_winner_seeds"], row["mean_p_winner_seeds_se"]],
            expected,
            rtol=1e-12,
        )

    def test2(self) -> None:
        """
        Test that only the tennis maps of the seed variants get the 2 columns,
        and that the input rows are left as they were.
        """
        # Prepare inputs.
        config = _get_tiny_threshold_config()
        sweeps = self.helper()
        sweeps_before = sweeps.copy()
        # Prepare outputs.
        expected = """
        {'tennis_base_maps_with_se': 2,
         'tennis_knee_maps_with_se': 0,
         'pickleball_maps_with_se': 0,
         'same_input_columns': True,
         'input_unchanged': True}
        """
        # Run test.
        actual = racket_sports_07_speed_vs_accuracy_utils.add_seed_mean_se(
            sweeps, config
        )
        # Check outputs.
        is_tennis = actual["sport"] == "Tennis"
        is_base = actual["variant"] == "base"
        has_se = actual["mean_p_winner_seeds_se"].notna()
        actual = {
            "tennis_base_maps_with_se": int(
                (is_tennis & is_base & has_se).sum()
            ),
            "tennis_knee_maps_with_se": int(
                (is_tennis & ~is_base & has_se).sum()
            ),
            "pickleball_maps_with_se": int((~is_tennis & has_se).sum()),
            "same_input_columns": actual[sweeps.columns].equals(sweeps),
            "input_unchanged": sweeps.equals(sweeps_before),
        }
        actual = pprint.pformat(actual, sort_dicts=False)
        self.assert_equal(actual, expected, dedent=True, fuzzy_match=True)

    def test3(self) -> None:
        """
        Test that seed 1 of the sweeps as an extra seed, a repeated seed, no
        extra seed, and a sport or variants without maps are rejected.
        """
        # Prepare inputs.
        sweeps = self.helper()
        bad_updates = (
            {"extra_seeds": (1, 2)},
            {"extra_seeds": (2, 2)},
            {"extra_seeds": ()},
            {"seed_sport": "Badminton"},
            {"seed_variants": ("parabola",)},
        )
        # Run test and check outputs.
        for update in bad_updates:
            config = _get_tiny_threshold_config(**update)
            with (
                self.subTest(update=update),
                self.assertRaises(AssertionError),
            ):
                racket_sports_07_speed_vs_accuracy_utils.add_seed_mean_se(
                    sweeps, config
                )


# #############################################################################
# Test_find_threshold_bands
# #############################################################################


class Test_find_threshold_bands(hunitest.TestCase):
    """
    Test `find_threshold_bands()`.
    """

    def helper(self) -> pd.DataFrame:
        """
        Build one sweep per sport, at `v0 / 2`, `v0`, `1.5 v0`, and `2 v0`.

        Each column ranks the speeds in its own way, so the band shows which
        column is used:
        - Pickleball: the highest cell ties at `v0` and `1.5 v0`, while the
          mean peaks at `v0 / 2`
        - Tennis: the average of the seeds has a narrow band at `v0` only, and
          seed 1 with the bound of the cells a wide one at `v0` and `1.5 v0`

        :return: the rows of both sweeps, the pickleball ones fastest first
        """
        pickleball = _make_threshold_sweep(
            "Pickleball",
            {
                "max_p_winner": [0.9, 0.99, 0.98, 0.8],
                "max_p_winner_se": [0.004] * 4,
                "mean_p_winner": [0.9, 0.8, 0.7, 0.6],
                "mean_p_winner_se": [0.0] * 4,
            },
        )
        tennis = _make_threshold_sweep(
            "Tennis",
            {
                "max_p_winner": [1.0] * 4,
                "max_p_winner_se": [0.0] * 4,
                "mean_p_winner": [0.8, 0.85, 0.84, 0.7],
                "mean_p_winner_se": [0.01] * 4,
                "mean_p_winner_seeds": [0.8, 0.85, 0.84, 0.7],
                "mean_p_winner_seeds_se": [0.002] * 4,
            },
        )
        sweeps = pd.concat([pickleball.iloc[::-1], tennis], ignore_index=True)
        return sweeps

    def test1(self) -> None:
        """
        Test the bands with the seeds: the highest cell in pickleball, and the
        average of the seeds in tennis, slowest speed first.
        """
        # Prepare inputs.
        sweeps = self.helper()
        tennis_se = "seeds"
        # Prepare outputs.
        expected = """
                sport variant    a  speed_ratio tennis_se  value  value_se  in_band
        0  Pickleball    base  1.0          0.5     seeds    0.9     0.004    False
        1  Pickleball    base  1.0          1.0     seeds   0.99     0.004     True
        2  Pickleball    base  1.0          1.5     seeds   0.98     0.004     True
        3  Pickleball    base  1.0          2.0     seeds    0.8     0.004    False
        4      Tennis    base  1.0          0.5     seeds    0.8     0.002    False
        5      Tennis    base  1.0          1.0     seeds   0.85     0.002     True
        6      Tennis    base  1.0          1.5     seeds   0.84     0.002    False
        7      Tennis    base  1.0          2.0     seeds    0.7     0.002    False
        """
        # Run test.
        actual = racket_sports_07_speed_vs_accuracy_utils.find_threshold_bands(
            sweeps, tennis_se
        )
        # Check outputs.
        actual = _to_str(actual)
        self.assert_equal(actual, expected, dedent=True, fuzzy_match=True)

    def test2(self) -> None:
        """
        Test that only the tennis band changes with the bound of the cells.
        """
        # Prepare inputs.
        sweeps = self.helper()
        # Prepare outputs.
        expected = """
        {'seeds': {'Pickleball': [False, True, True, False],
                   'Tennis': [False, True, False, False]},
         'cells': {'Pickleball': [False, True, True, False],
                   'Tennis': [False, True, True, False]}}
        """
        # Run test.
        actual = {}
        for tennis_se in ("seeds", "cells"):
            bands = (
                racket_sports_07_speed_vs_accuracy_utils.find_threshold_bands(
                    sweeps, tennis_se
                )
            )
            actual[tennis_se] = {
                sport: grp["in_band"].tolist()
                for sport, grp in bands.groupby("sport")
            }
        # Check outputs.
        actual = pprint.pformat(actual, sort_dicts=False)
        self.assert_equal(actual, expected, dedent=True, fuzzy_match=True)

    def test3(self) -> None:
        """
        Test that an unknown method, the seeds without their columns, and a
        missing standard error are rejected.
        """
        # Prepare inputs: `(sweeps, tennis_se)`.
        sweeps = self.helper()
        without_seeds = sweeps.drop(
            columns=["mean_p_winner_seeds", "mean_p_winner_seeds_se"]
        )
        with_nan = sweeps.copy()
        with_nan.loc[5, "mean_p_winner_seeds_se"] = np.nan
        bad_inputs = (
            (sweeps, "bootstrap"),
            (without_seeds, "seeds"),
            (with_nan, "seeds"),
        )
        # Run test and check outputs.
        for i, (bad_sweeps, tennis_se) in enumerate(bad_inputs):
            with self.subTest(i=i), self.assertRaises(AssertionError):
                racket_sports_07_speed_vs_accuracy_utils.find_threshold_bands(
                    bad_sweeps, tennis_se
                )


# #############################################################################
# Test_find_speed_thresholds
# #############################################################################


class Test_find_speed_thresholds(hunitest.TestCase):
    """
    Test `find_speed_thresholds()`.
    """

    def helper(self, in_band: Dict[float, str], expected: str) -> None:
        """
        Check the thresholds of one pickleball sweep per `a`.

        :param in_band: per `a`, the band over `v0 / 2`, `v0`, `1.5 v0`, and
            `2 v0`, e.g., "-xx-"
        :param expected: the thresholds, as a dictionary
        """
        # Prepare inputs.
        bands = _make_band_rows(in_band)
        names = ["a_star", "a_star_prev", "fastest_reenters"]
        names += ["band_low_ratio_at_a_star", "band_high_ratio_at_a_star"]
        names += ["a_star_star"]
        # Run test.
        thresholds = (
            racket_sports_07_speed_vs_accuracy_utils.find_speed_thresholds(bands)
        )
        # Check outputs.
        self.assertEqual(len(thresholds), 1)
        actual = pprint.pformat(
            thresholds.iloc[0][names].to_dict(), sort_dicts=False
        )
        self.assert_equal(actual, expected, dedent=True, fuzzy_match=True)

    def test1(self) -> None:
        """
        Test a band that slides to the slow end: `2 v0` leaves it at 0.25 and
        stays out, and the band falls below `v0` at 1.
        """
        # Prepare inputs.
        in_band = {
            0.0: "-xxx",
            0.25: "xxx-",
            0.5: "xx--",
            0.75: "xx--",
            1.0: "x---",
        }
        # Prepare outputs.
        expected = """
        {'a_star': 0.25,
         'a_star_prev': 0.0,
         'fastest_reenters': False,
         'band_low_ratio_at_a_star': 0.5,
         'band_high_ratio_at_a_star': 1.5,
         'a_star_star': 1.0}
        """
        # Run test.
        self.helper(in_band, expected)

    def test2(self) -> None:
        """
        Test that a fastest ball always in the band gives no thresholds.
        """
        # Prepare inputs.
        in_band = {0.0: "xxxx", 0.5: "-xxx"}
        # Prepare outputs.
        expected = """
        {'a_star': nan,
         'a_star_prev': nan,
         'fastest_reenters': False,
         'band_low_ratio_at_a_star': nan,
         'band_high_ratio_at_a_star': nan,
         'a_star_star': nan}
        """
        # Run test.
        self.helper(in_band, expected)

    def test3(self) -> None:
        """
        Test that a fastest ball back in the band after `a*` is flagged, and
        `a*` stays the first value out.
        """
        # Prepare inputs.
        in_band = {0.0: "xxxx", 0.25: "xxx-", 0.5: "xxxx", 0.75: "xx--"}
        # Prepare outputs.
        expected = """
        {'a_star': 0.25,
         'a_star_prev': 0.0,
         'fastest_reenters': True,
         'band_low_ratio_at_a_star': 0.5,
         'band_high_ratio_at_a_star': 1.5,
         'a_star_star': nan}
        """
        # Run test.
        self.helper(in_band, expected)

    def test4(self) -> None:
        """
        Test that `v0` out of a band of faster speeds is not `a**`, and that
        `a*` at the first value of `a` has no value before it.
        """
        # Prepare inputs.
        in_band = {0.0: "--x-", 0.5: "-xx-"}
        # Prepare outputs.
        expected = """
        {'a_star': 0.0,
         'a_star_prev': nan,
         'fastest_reenters': False,
         'band_low_ratio_at_a_star': 1.5,
         'band_high_ratio_at_a_star': 1.5,
         'a_star_star': nan}
        """
        # Run test.
        self.helper(in_band, expected)

    def test5(self) -> None:
        """
        Test one row per sport and variant, in the order of the bands.
        """
        # Prepare inputs.
        bands = pd.concat(
            [
                _make_band_rows({0.0: "xxxx", 0.5: "xx--"}),
                _make_band_rows({0.0: "xxxx", 0.5: "-xx-"}, variant="knee"),
                _make_band_rows({0.0: "xxxx", 0.5: "xxxx"}, sport="Tennis"),
            ],
            ignore_index=True,
        )
        # Prepare outputs.
        expected = """
                sport variant tennis_se  n_a  a_max  a_star
        0  Pickleball    base     seeds    2    0.5     0.5
        1  Pickleball    knee     seeds    2    0.5     0.5
        2      Tennis    base     seeds    2    0.5     NaN
        """
        # Run test.
        actual = racket_sports_07_speed_vs_accuracy_utils.find_speed_thresholds(
            bands
        )
        # Check outputs.
        columns = ["sport", "variant", "tennis_se", "n_a", "a_max", "a_star"]
        actual = _to_str(actual[columns])
        self.assert_equal(actual, expected, dedent=True, fuzzy_match=True)

    def test6(self) -> None:
        """
        Test that bands without `v0`, and with a ball speed missing at one
        `a`, are rejected.
        """
        # Prepare inputs.
        bands = _make_band_rows({0.0: "xxxx", 0.5: "xx--"})
        bad_inputs = (
            bands[bands["speed_ratio"] != 1.0],
            bands.drop(index=7),
        )
        # Run test and check outputs.
        for i, bad_bands in enumerate(bad_inputs):
            with self.subTest(i=i), self.assertRaises(AssertionError):
                racket_sports_07_speed_vs_accuracy_utils.find_speed_thresholds(
                    bad_bands
                )


# #############################################################################
# Test_compare_tennis_se
# #############################################################################


class Test_compare_tennis_se(hunitest.TestCase):
    """
    Test `compare_tennis_se()`.
    """

    def helper(self, tennis_se: str, tennis_band: str) -> pd.DataFrame:
        """
        Find the thresholds of a tennis and a pickleball sweep.

        :param tennis_se: the method, "seeds" or "cells"
        :param tennis_band: the tennis band at `a = 0.5`, e.g., "xx--"
        :return: the rows of `find_speed_thresholds()`, pickleball first
        """
        bands = pd.concat(
            [
                _make_band_rows({0.0: "xxxx", 0.5: "xxx-"}),
                _make_band_rows({0.0: "xxxx", 0.5: tennis_band}, sport="Tennis"),
            ],
            ignore_index=True,
        ).assign(tennis_se=tennis_se)
        thresholds = (
            racket_sports_07_speed_vs_accuracy_utils.find_speed_thresholds(bands)
        )
        return thresholds

    def test1(self) -> None:
        """
        Test that the thresholds with the bound of the cells join the rows of
        the seeds sweep by sweep, in the order of the seeds.
        """
        # Prepare inputs.
        thresholds_seeds = self.helper("seeds", "xx--")
        thresholds_cells = self.helper("cells", "xxxx").iloc[::-1]
        # Prepare outputs.
        expected = """
                sport variant  a_star  a_star_cells  a_star_star  a_star_star_cells
        0  Pickleball    base     0.5           0.5          NaN                NaN
        1      Tennis    base     0.5           NaN          NaN                NaN
        """
        # Run test.
        actual = racket_sports_07_speed_vs_accuracy_utils.compare_tennis_se(
            thresholds_seeds, thresholds_cells
        )
        # Check outputs.
        columns = ["sport", "variant", "a_star", "a_star_cells", "a_star_star"]
        columns += ["a_star_star_cells"]
        self.assertNotIn("tennis_se", actual.columns)
        actual = _to_str(actual[columns])
        self.assert_equal(actual, expected, dedent=True, fuzzy_match=True)

    def test2(self) -> None:
        """
        Test that the methods in the wrong order, and tables of different
        sweeps, are rejected.
        """
        # Prepare inputs: `(thresholds_seeds, thresholds_cells)`.
        thresholds_seeds = self.helper("seeds", "xx--")
        thresholds_cells = self.helper("cells", "xxxx")
        bad_inputs = (
            (thresholds_cells, thresholds_seeds),
            (thresholds_seeds, thresholds_cells.iloc[:1]),
        )
        # Run test and check outputs.
        for i, (seeds, cells) in enumerate(bad_inputs):
            with self.subTest(i=i), self.assertRaises(AssertionError):
                racket_sports_07_speed_vs_accuracy_utils.compare_tennis_se(
                    seeds, cells
                )


# #############################################################################
# Test_plot_threshold_bands
# #############################################################################


class Test_plot_threshold_bands(hunitest.TestCase):
    """
    Test `plot_threshold_bands()`.
    """

    def test1(self) -> None:
        """
        Test one panel per variant, the speeds of the band as filled squares,
        and a line for each threshold found.
        """
        # Prepare inputs.
        bands = pd.concat(
            [
                _make_band_rows({0.0: "-xxx", 0.5: "x---"}),
                _make_band_rows({0.0: "-xxx", 0.5: "-xxx"}, variant="knee"),
            ],
            ignore_index=True,
        )
        thresholds = (
            racket_sports_07_speed_vs_accuracy_utils.find_speed_thresholds(bands)
        )
        variants = ["base", "knee"]
        # Prepare outputs.
        expected = """
        {'titles': ['Pickleball: base', 'Pickleball: knee'],
         'legends': [['in the band', 'out of the band', 'a* = 0.5', 'a** = 0.5'],
                     ['in the band', 'out of the band']],
         'base_in_band': [(0.0, 1.0), (0.0, 1.5), (0.0, 2.0), (0.5, 0.5)]}
        """
        # Run test.
        fig = racket_sports_07_speed_vs_accuracy_utils.plot_threshold_bands(
            bands, thresholds, variants
        )
        # Check outputs.
        filled = fig.axes[0].get_lines()[0]
        actual = {
            "titles": [ax.get_title() for ax in fig.axes],
            "legends": [
                [text.get_text() for text in ax.get_legend().texts]
                for ax in fig.axes
            ],
            "base_in_band": list(
                zip(filled.get_xdata().tolist(), filled.get_ydata().tolist())
            ),
        }
        actual = pprint.pformat(actual, sort_dicts=False)
        self.assert_equal(actual, expected, dedent=True, fuzzy_match=True)
        plt.close("all")

    def test2(self) -> None:
        """
        Test that a variant without maps is rejected.
        """
        # Prepare inputs.
        bands = _make_band_rows({0.0: "-xxx", 0.5: "x---"})
        thresholds = (
            racket_sports_07_speed_vs_accuracy_utils.find_speed_thresholds(bands)
        )
        variants = ["base", "parabola"]
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            racket_sports_07_speed_vs_accuracy_utils.plot_threshold_bands(
                bands, thresholds, variants
            )
        plt.close("all")


# #############################################################################
# Test_save_paper_figure_data
# #############################################################################


class Test_save_paper_figure_data(hunitest.TestCase):
    """
    Test `save_paper_figure_data()`.
    """

    def test1(self) -> None:
        """
        Test that the file holds only the base and the other shapes, with the
        columns of the figure.
        """
        # Prepare inputs.
        out_dir = self.get_scratch_space()
        best_speeds = pd.concat(
            [
                _make_best_speed_rows("base", 1.0, (0.5, 1.0)),
                _make_best_speed_rows("parabola", 1.0, (1.0,)),
                _make_best_speed_rows("seed_2", 1.0, (0.5,)),
            ],
            ignore_index=True,
        )
        # Prepare outputs.
        expected = """
        sport,variant,a,best_speed_ratio,band_low_ratio,band_high_ratio
        Tennis,base,1.0,0.5,0.5,1.0
        Tennis,parabola,1.0,1.0,1.0,1.0
        """
        # Run test.
        path = racket_sports_07_speed_vs_accuracy_utils.save_paper_figure_data(
            best_speeds, out_dir
        )
        # Check outputs.
        self.assertEqual(
            os.path.basename(path), "speed_vs_accuracy_best_speed.csv"
        )
        with open(path) as f:
            actual = f.read()
        self.assert_equal(actual, expected, dedent=True)

    def test2(self) -> None:
        """
        Test that a directory that does not exist is rejected.
        """
        # Prepare inputs.
        out_dir = os.path.join(self.get_scratch_space(), "missing")
        best_speeds = _make_best_speed_rows("base", 1.0, (0.5,))
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            racket_sports_07_speed_vs_accuracy_utils.save_paper_figure_data(
                best_speeds, out_dir
            )
