"""
Tests for racket_sports_05_variable_sensitivity_utils module.

Import as:

import research.Optimal_strategy_for_racket_sports.test.test_racket_sports_05_variable_sensitivity_utils as rosfrsttrs0vsu
"""

import dataclasses
import logging
import os
from typing import Any, Dict

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import helpers.hunit_test as hunitest
from research.Optimal_strategy_for_racket_sports import racket_params
from research.Optimal_strategy_for_racket_sports.notebooks import (
    racket_sports_01_shot_probability_utils,
)
from research.Optimal_strategy_for_racket_sports.notebooks import (
    racket_sports_05_variable_sensitivity_utils,
)

_LOG = logging.getLogger(__name__)

# Use the non-interactive backend so the figures render without a display.
matplotlib.use("Agg")


def _get_tiny_config(
    **update: Any,
) -> racket_sports_05_variable_sensitivity_utils.ExperimentConfig:
    """
    Get a configuration that runs in a fraction of a second per experiment.

    More than 1,000 samples, since fewer is slower (see `HEATMAP_N_SAMPLES`).

    :param update: fields of the configuration to change
    :return: one sport, a tiny grid, and a few samples
    """
    config = racket_sports_05_variable_sensitivity_utils.get_experiment_config(
        False
    )
    config = dataclasses.replace(
        config,
        sports=("Pickleball",),
        grid_size={"Pickleball": (3, 4)},
        n_samples=1100,
        serve_names=("wide-deep", "T-deep"),
        sweep_step=4,
        serve_map_step=4,
        n_random=2,
        r_values=(0.3, 0.6),
    )
    config = dataclasses.replace(config, **update)
    return config


def _get_tiny_settings(sport_name: str, **update: Any) -> Dict[str, Any]:
    """
    Get the default settings of a sport, on a tiny map.

    :param sport_name: "Tennis" or "Pickleball"
    :param update: settings to change
    :return: the settings
    """
    settings: Dict[str, Any] = (
        racket_sports_05_variable_sensitivity_utils.get_default_settings(
            sport_name, n_samples=1100
        )
    )
    settings.update(n_x=3, n_y=4)
    settings.update(update)
    return settings


def _make_sweep_rows() -> pd.DataFrame:
    """
    Build the rows of 2 sweeps of `run_all_sweeps()`, by hand.

    :return: `v_p` moves the highest `P(winner)` by 10 points, `t_r` by 1
    """
    df = pd.DataFrame(
        {
            "experiment": ["sweep_defaults"] * 4,
            "sport": ["Tennis"] * 4,
            "sweep_var": ["t_r", "t_r", "v_p", "v_p"],
            "t_r": [0.0, 0.5, 0.2, 0.2],
            "v_p": [1.5, 1.5, 0.0, 6.0],
            "max_p_winner": [0.99, 1.0, 1.0, 0.9],
            "mean_p_winner": [0.7, 0.8, 0.8, 0.5],
        }
    )
    return df


# #############################################################################
# Test_get_experiment_config
# #############################################################################


class Test_get_experiment_config(hunitest.TestCase):
    """
    Test `get_experiment_config()`.
    """

    def test1(self) -> None:
        """
        Test that the full runs use the defaults of the widget.
        """
        # Run test.
        config = (
            racket_sports_05_variable_sensitivity_utils.get_experiment_config(
                True
            )
        )
        # Check outputs.
        self.assertEqual(config.n_samples, 5000)
        self.assertEqual(config.sides, ("deuce", "ad"))
        self.assertEqual(config.n_random, 500)
        self.assertEqual(len(config.serve_names), 9)
        self.assertEqual(len(config.r_values), 12)

    def test2(self) -> None:
        """
        Test that the reduced runs keep the cells of the full ones.
        """
        # Prepare inputs.
        module = racket_sports_05_variable_sensitivity_utils
        # Run test.
        full = module.get_experiment_config(True)
        reduced = module.get_experiment_config(False)
        # Check outputs.
        self.assertEqual(reduced.grid_size, full.grid_size)
        self.assertLess(reduced.n_samples, full.n_samples)
        self.assertLess(reduced.n_random, full.n_random)

    def test3(self) -> None:
        """
        Test that a value that is not a bool is rejected.
        """
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            racket_sports_05_variable_sensitivity_utils.get_experiment_config(1)


# #############################################################################
# Test_get_default_settings
# #############################################################################


class Test_get_default_settings(hunitest.TestCase):
    """
    Test `get_default_settings()`.
    """

    def helper(self, sport_name: str, expected: str) -> None:
        """
        Test helper for `get_default_settings()`.

        :param sport_name: name of the sport
        :param expected: expected settings string
        """
        # Run test.
        settings = (
            racket_sports_05_variable_sensitivity_utils.get_default_settings(
                sport_name
            )
        )
        actual = racket_sports_05_variable_sensitivity_utils.settings_to_string(
            settings
        )
        # Check outputs.
        self.assert_equal(actual, expected)

    def test1(self) -> None:
        """
        Test that the tennis defaults give the settings string of the Gdoc.
        """
        # Prepare outputs.
        expected = (
            "sport=Tennis side=deuce x1=-2.1 y1=3.2 std_x2=0.5 std_y2=1.0"
            " ball_speed=25.0 t_r=0.2 v_p=1.5 seed=1 n_x=10 n_y=15"
            " n_samples=5000"
        )
        # Run test and check outputs.
        self.helper("Tennis", expected)

    def test2(self) -> None:
        """
        Test that the pickleball defaults give the settings string of the Gdoc.
        """
        # Prepare outputs.
        expected = (
            "sport=Pickleball side=deuce x1=-1.5 y1=4.4 std_x2=0.5 std_y2=1.0"
            " ball_speed=12.0 t_r=0.2 v_p=1.5 seed=1 n_x=9 n_y=10"
            " n_samples=5000"
        )
        # Run test and check outputs.
        self.helper("Pickleball", expected)

    def test3(self) -> None:
        """
        Test that an unknown sport is rejected.
        """
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            racket_sports_05_variable_sensitivity_utils.get_default_settings(
                "Squash"
            )


# #############################################################################
# Test_settings_to_string
# #############################################################################


class Test_settings_to_string(hunitest.TestCase):
    """
    Test `settings_to_string()`.
    """

    def test1(self) -> None:
        """
        Test that the settings of a run are listed in the widget's order.
        """
        # Prepare inputs.
        settings = _get_tiny_settings("Pickleball", v_p=6.0)
        # Prepare outputs.
        expected = (
            "sport=Pickleball side=deuce x1=-1.5 y1=4.4 std_x2=0.5 std_y2=1.0"
            " ball_speed=12.0 t_r=0.2 v_p=6.0 seed=1 n_x=3 n_y=4"
            " n_samples=1100"
        )
        # Run test.
        actual = racket_sports_05_variable_sensitivity_utils.settings_to_string(
            settings
        )
        # Check outputs.
        self.assert_equal(actual, expected)

    def test2(self) -> None:
        """
        Test that missing settings are rejected.
        """
        # Prepare inputs.
        settings = _get_tiny_settings("Pickleball")
        del settings["seed"]
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            racket_sports_05_variable_sensitivity_utils.settings_to_string(
                settings
            )


# #############################################################################
# Test_parse_settings_string
# #############################################################################


class Test_parse_settings_string(hunitest.TestCase):
    """
    Test `parse_settings_string()`.
    """

    def test1(self) -> None:
        """
        Test that a settings string parses back to the same settings.
        """
        # Prepare inputs.
        settings = _get_tiny_settings("Tennis", x1=-0.3, ball_speed=73.14)
        text = racket_sports_05_variable_sensitivity_utils.settings_to_string(
            settings
        )
        # Run test.
        actual = (
            racket_sports_05_variable_sensitivity_utils.parse_settings_string(
                text
            )
        )
        # Check outputs.
        self.assertEqual(actual, settings)

    def test2(self) -> None:
        """
        Test that a string with an unknown setting is rejected.
        """
        # Prepare inputs.
        settings = _get_tiny_settings("Tennis")
        text = racket_sports_05_variable_sensitivity_utils.settings_to_string(
            settings
        )
        text = text.replace("seed=1", "seed=1 v_x=2.0")
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            racket_sports_05_variable_sensitivity_utils.parse_settings_string(
                text
            )


# #############################################################################
# Test_compute_run
# #############################################################################


class Test_compute_run(hunitest.TestCase):
    """
    Test `compute_run()`.
    """

    def test1(self) -> None:
        """
        Test that the summary agrees with the map, on a map where the marked
        target is not the highest cell.
        """
        # Prepare inputs.
        settings = _get_tiny_settings("Tennis", v_p=6.0, ball_speed=5.0)
        # Run test.
        summary, heatmap = (
            racket_sports_05_variable_sensitivity_utils.compute_run(settings)
        )
        # Check outputs.
        self.assertEqual(summary["n_cells"], 12)
        self.assertEqual(len(heatmap), 12)
        self.assertEqual(summary["max_p_winner"], heatmap["p_winner"].max())
        self.assertEqual(summary["mean_p_winner"], heatmap["p_winner"].mean())
        self.assertLess(summary["best_p_winner"], summary["max_p_winner"])
        self.assertGreater(summary["n_region_cells"], 0)

    def test2(self) -> None:
        """
        Test that with `v_p = 0` the map is the `P(in)` map on the same
        samples.
        """
        # Prepare inputs.
        settings = _get_tiny_settings("Tennis", v_p=0.0)
        # Run test.
        _, heatmap = racket_sports_05_variable_sensitivity_utils.compute_run(
            settings
        )
        # Check outputs.
        np.testing.assert_array_equal(heatmap["p_winner"], heatmap["p_in"])

    def test3(self) -> None:
        """
        Test that a fault is rejected: this serve lands past the service line.
        """
        # Prepare inputs.
        settings = _get_tiny_settings("Tennis", y1=8.0)
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            racket_sports_05_variable_sensitivity_utils.compute_run(settings)


# #############################################################################
# Test_get_sweep_values
# #############################################################################


class Test_get_sweep_values(hunitest.TestCase):
    """
    Test `get_sweep_values()`.
    """

    def test1(self) -> None:
        """
        Test that the ball speeds end at the fastest ball of the sport.
        """
        # Prepare outputs.
        expected = racket_params.TENNIS.max_ball_speed_mps
        # Run test.
        actual = racket_sports_05_variable_sensitivity_utils.get_sweep_values(
            "Tennis"
        )
        # Check outputs.
        self.assertEqual(actual["ball_speed"][-1], expected)
        self.assertEqual(
            sorted(actual), ["ball_speed", "std_x2", "std_y2", "t_r", "v_p"]
        )

    def test2(self) -> None:
        """
        Test that thinning a sweep keeps both of its ends.
        """
        # Run test.
        actual = racket_sports_05_variable_sensitivity_utils.get_sweep_values(
            "Tennis", sweep_step=2
        )
        # Check outputs.
        self.assertEqual(actual["v_p"], [0.0, 1.0, 2.0, 3.0, 5.0, 6.0])

    def test3(self) -> None:
        """
        Test that a step of 0 is rejected.
        """
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            racket_sports_05_variable_sensitivity_utils.get_sweep_values(
                "Tennis", sweep_step=0
            )


# #############################################################################
# Test_run_sweep
# #############################################################################


class Test_run_sweep(hunitest.TestCase):
    """
    Test `run_sweep()`.
    """

    def test1(self) -> None:
        """
        Test that a faster player 1 never raises `P(winner)` of the best cell,
        since they reach a superset of the same samples.
        """
        # Prepare inputs.
        settings = _get_tiny_settings("Pickleball", ball_speed=5.0)
        # Run test.
        actual = racket_sports_05_variable_sensitivity_utils.run_sweep(
            settings, "v_p", [0.0, 3.0, 6.0], "defaults"
        )
        # Check outputs.
        self.assertEqual(
            actual["run_id"].tolist(),
            [
                "sweep_defaults_pickleball_v_p_0",
                "sweep_defaults_pickleball_v_p_3",
                "sweep_defaults_pickleball_v_p_6",
            ],
        )
        p_winner = actual["max_p_winner"].to_numpy()
        self.assertTrue(np.all(np.diff(p_winner) <= 0))
        self.assertLess(p_winner[-1], p_winner[0])

    def test2(self) -> None:
        """
        Test that the setting of each run is the swept value, and the other
        settings do not move.
        """
        # Prepare inputs.
        settings = _get_tiny_settings("Tennis")
        # Run test.
        actual = racket_sports_05_variable_sensitivity_utils.run_sweep(
            settings, "std_x2", [0.2, 2.0], "fast_p1"
        )
        # Check outputs.
        self.assertEqual(actual["std_x2"].tolist(), [0.2, 2.0])
        self.assertEqual(actual["std_y2"].tolist(), [1.0, 1.0])
        self.assertEqual(
            actual["experiment"].unique().tolist(), ["sweep_fast_p1"]
        )

    def test3(self) -> None:
        """
        Test that an unknown setting is rejected.
        """
        # Prepare inputs.
        settings = _get_tiny_settings("Tennis")
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            racket_sports_05_variable_sensitivity_utils.run_sweep(
                settings, "v_x", [1.0], "defaults"
            )


# #############################################################################
# Test_run_all_sweeps
# #############################################################################


class Test_run_all_sweeps(hunitest.TestCase):
    """
    Test `run_all_sweeps()`.
    """

    def test1(self) -> None:
        """
        Test both bases, with the `v_p` sweep only around the defaults.
        """
        # Prepare inputs.
        config = _get_tiny_config()
        # Run test.
        actual = racket_sports_05_variable_sensitivity_utils.run_all_sweeps(
            config
        )
        # Check outputs.
        vars_per_base = actual.groupby("experiment")["sweep_var"].nunique()
        self.assertEqual(vars_per_base["sweep_defaults"], 5)
        self.assertEqual(vars_per_base["sweep_fast_p1"], 4)
        is_fast = actual["experiment"] == "sweep_fast_p1"
        self.assertEqual(set(actual.loc[is_fast, "v_p"]), {6.0})
        self.assertEqual(actual["n_samples"].unique().tolist(), [1100])


# #############################################################################
# Test_run_corners
# #############################################################################


class Test_run_corners(hunitest.TestCase):
    """
    Test `run_corners()`.
    """

    def test1(self) -> None:
        """
        Test that every serve meets every ball speed and every `v_p`.
        """
        # Prepare inputs.
        config = _get_tiny_config()
        # Run test.
        actual = racket_sports_05_variable_sensitivity_utils.run_corners(config)
        # Check outputs.
        self.assertEqual(len(actual), 2 * 3 * 3)
        self.assertEqual(
            sorted(actual["ball_speed"].unique()), [5.0, 12.0, 24.59]
        )
        self.assertEqual(sorted(actual["v_p"].unique()), [0.0, 1.5, 6.0])
        self.assertIn(
            "corner_pickleball_deuce_T-deep_ballmin_vpmax",
            actual["run_id"].tolist(),
        )

    def test2(self) -> None:
        """
        Test that the ad serves mirror the deuce serves.
        """
        # Prepare inputs.
        config = _get_tiny_config(sides=("deuce", "ad"), serve_names=("T-deep",))
        # Run test.
        actual = racket_sports_05_variable_sensitivity_utils.run_corners(config)
        # Check outputs.
        by_side = actual.groupby("side")["x1"].unique()
        self.assertEqual(by_side["deuce"].tolist(), [-0.3])
        self.assertEqual(by_side["ad"].tolist(), [0.3])


# #############################################################################
# Test_find_best_serves
# #############################################################################


class Test_find_best_serves(hunitest.TestCase):
    """
    Test `find_best_serves()`.
    """

    def helper(self, p: Dict[str, float], se: float) -> pd.DataFrame:
        """
        Run `find_best_serves()` on one corner.

        :param p: the highest `P(winner)` per serve
        :param se: the standard error of each value
        :return: the output of `find_best_serves()`
        """
        corners = pd.DataFrame(
            {
                "sport": "Tennis",
                "side": "deuce",
                "group": "ball=min v_p=max",
                "serve_name": list(p),
                "max_p_winner": list(p.values()),
                "max_p_winner_se": se,
            }
        )
        actual = racket_sports_05_variable_sensitivity_utils.find_best_serves(
            corners
        )
        return actual

    def test1(self) -> None:
        """
        Test a clear best serve: the others are more than 2 SE away.
        """
        # Prepare inputs.
        p = {"T-deep": 0.1, "wide-deep": 0.5, "T-short": 0.9}
        # Run test.
        actual = self.helper(p, 0.01)
        # Check outputs.
        row = actual.iloc[0]
        self.assertEqual(row["best_serve"], "T-deep")
        self.assertEqual(row["max_p_winner"], 0.1)
        self.assertEqual(row["highest_over_serves"], 0.9)
        self.assertEqual(row["n_tied"], 1)

    def test2(self) -> None:
        """
        Test that a value within 2 SE of the difference is a tie.
        """
        # Prepare inputs.
        p = {"T-deep": 0.5, "wide-deep": 0.51, "T-short": 0.9}
        # Run test.
        actual = self.helper(p, 0.01)
        # Check outputs.
        self.assertEqual(actual.iloc[0]["n_tied"], 2)

    def test3(self) -> None:
        """
        Test that no run is rejected.
        """
        # Prepare inputs.
        corners = pd.DataFrame()
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            racket_sports_05_variable_sensitivity_utils.find_best_serves(corners)


# #############################################################################
# Test_run_serve_map
# #############################################################################


class Test_run_serve_map(hunitest.TestCase):
    """
    Test `run_serve_map()`.
    """

    def test1(self) -> None:
        """
        Test that the serves are the thinned grid of legal deuce serves.
        """
        # Prepare inputs.
        config = _get_tiny_config()
        # Run test.
        actual = racket_sports_05_variable_sensitivity_utils.run_serve_map(
            config
        )
        # Check outputs.
        self.assertEqual(actual["x1"].unique().tolist(), [-2.8, -0.8])
        self.assertEqual(actual["y1"].unique().tolist(), [6.4, 4.4, 2.4])
        self.assertEqual(
            actual["group"].unique().tolist(), ["fast_p1_slow_ball"]
        )
        self.assertEqual(actual["v_p"].unique().tolist(), [6.0])

    def test2(self) -> None:
        """
        Test that the deep serve leaves player 2 less than the short one, in
        the corner where the serve matters.
        """
        # Prepare inputs.
        config = _get_tiny_config()
        # Run test.
        actual = racket_sports_05_variable_sensitivity_utils.run_serve_map(
            config
        )
        # Check outputs.
        by_depth = actual.groupby("y1")["max_p_winner"].mean()
        self.assertLess(by_depth[6.4], by_depth[2.4])


# #############################################################################
# Test_check_ratio_invariance
# #############################################################################


class Test_check_ratio_invariance(hunitest.TestCase):
    """
    Test `check_ratio_invariance()`.
    """

    def helper(self, sport_name: str, k: float, **update: Any) -> Dict[str, Any]:
        """
        Run the check where player 1 reaches many landing points.

        :param sport_name: name of the sport
        :param k: the factor of both speeds
        :param update: settings to change
        :return: the output of `check_ratio_invariance()`
        """
        settings = _get_tiny_settings(
            sport_name, v_p=6.0, ball_speed=5.0, **update
        )
        actual: Dict[str, Any] = (
            racket_sports_05_variable_sensitivity_utils.check_ratio_invariance(
                settings, k
            )
        )
        return actual

    def test1(self) -> None:
        """
        Test the exact invariance at `t_r = 0`, cell by cell, in tennis.
        """
        # Run test.
        actual = self.helper("Tennis", 2.0, t_r=0.0)
        # Check outputs.
        self.assertEqual(actual["max_diff"], 0.0)
        # The reach rule is exercised: a difference could show.
        self.assertGreater(actual["mean_p_reached"], 0.01)

    def test2(self) -> None:
        """
        Test the exact invariance at `t_r = 0` in pickleball, for a factor that
        is not a power of 2.
        """
        # Run test.
        actual = self.helper("Pickleball", 3.7, t_r=0.0)
        # Check outputs.
        self.assertEqual(actual["max_diff"], 0.0)
        self.assertGreater(actual["mean_p_reached"], 0.01)

    def test3(self) -> None:
        """
        Test the planted failure: at `t_r > 0` the same scaling changes the map
        cell by cell, since the reaction distance `v_p * t_r` changes.
        """
        # Run test.
        actual = self.helper("Pickleball", 2.0, t_r=0.2)
        # Check outputs.
        self.assertGreater(actual["max_diff"], 0.1)
        self.assertGreater(actual["max_diff_p_winner"], 0.1)

    def test4(self) -> None:
        """
        Test that a factor that is not positive is rejected.
        """
        # Prepare inputs.
        settings = _get_tiny_settings("Tennis", t_r=0.0)
        # Run test and check outputs.
        for k in (0.0, -1.0, float("nan")):
            with self.subTest(k=k), self.assertRaises(AssertionError):
                racket_sports_05_variable_sensitivity_utils.check_ratio_invariance(
                    settings, k
                )


# #############################################################################
# Test_run_ratio_checks
# #############################################################################


class Test_run_ratio_checks(hunitest.TestCase):
    """
    Test `run_ratio_checks()`.
    """

    def test1(self) -> None:
        """
        Test that the 5 settings give identical maps at `t_r = 0`.
        """
        # Prepare inputs.
        config = _get_tiny_config()
        # Run test.
        actual = racket_sports_05_variable_sensitivity_utils.run_ratio_checks(
            config
        )
        # Check outputs.
        self.assertEqual(
            actual["setting"].tolist(),
            [
                "defaults",
                "v_p_6",
                "slow_ball_fast_p1",
                "T_deep_mid",
                "wide_short_slow",
            ],
        )
        self.assertEqual(actual["max_diff"].max(), 0.0)


# #############################################################################
# Test_run_random_ratio_checks
# #############################################################################


class Test_run_random_ratio_checks(hunitest.TestCase):
    """
    Test `run_random_ratio_checks()`.
    """

    def test1(self) -> None:
        """
        Test that the first draw is the first row of the saved 500 runs of
        #789, so a full run repeats them.
        """
        # Prepare inputs.
        config = _get_tiny_config(n_random=1)
        # Prepare outputs.
        expected = (
            "sport=Pickleball side=deuce x1=-1.7 y1=4.3 std_x2=0.6 std_y2=0.1"
            " ball_speed=6.62 t_r=0.0 v_p=3.7 seed=9 n_x=4 n_y=5"
            " n_samples=2000"
        )
        # Run test.
        actual = (
            racket_sports_05_variable_sensitivity_utils.run_random_ratio_checks(
                config
            )
        )
        # Check outputs.
        self.assert_equal(actual.iloc[0]["settings"], expected)

    def test2(self) -> None:
        """
        Test that the maps are identical at random settings and factors.
        """
        # Prepare inputs.
        config = _get_tiny_config(n_random=3)
        # Run test.
        actual = (
            racket_sports_05_variable_sensitivity_utils.run_random_ratio_checks(
                config
            )
        )
        # Check outputs.
        self.assertEqual(len(actual), 3)
        self.assertEqual(actual["max_diff"].max(), 0.0)


# #############################################################################
# Test_run_speed_grid
# #############################################################################


class Test_run_speed_grid(hunitest.TestCase):
    """
    Test `run_speed_grid()`.
    """

    def test1(self) -> None:
        """
        Test one row and one map per `r` x `v_p` x `t_r`.
        """
        # Prepare inputs.
        config = _get_tiny_config()
        # Run test.
        grid, p_winner_maps = (
            racket_sports_05_variable_sensitivity_utils.run_speed_grid(config)
        )
        # Check outputs.
        self.assertEqual(len(grid), 2 * 3 * 3)
        self.assertEqual(sorted(p_winner_maps), grid["run_id"].tolist())
        self.assertEqual(len(p_winner_maps[0]), 12)
        np.testing.assert_allclose(grid["v_p"] / grid["ball_speed"], grid["r"])


# #############################################################################
# Test_summarize_ratio_collapse
# #############################################################################


class Test_summarize_ratio_collapse(hunitest.TestCase):
    """
    Test `summarize_ratio_collapse()`.
    """

    def test1(self) -> None:
        """
        Test one curve per ratio at `t_r = 0`, and the planted failure at
        `t_r > 0`.
        """
        # Prepare inputs.
        config = _get_tiny_config()
        module = racket_sports_05_variable_sensitivity_utils
        grid, p_winner_maps = module.run_speed_grid(config)
        # Run test.
        actual = module.summarize_ratio_collapse(grid, p_winner_maps)
        # Check outputs.
        self.assertEqual(actual["t_r"].tolist(), [0.0, 0.2, 0.25])
        at_zero = actual.iloc[0]
        self.assertEqual(at_zero["max_p_winner_spread"], 0.0)
        self.assertEqual(at_zero["mean_p_winner_spread"], 0.0)
        self.assertEqual(at_zero["max_cell_diff"], 0.0)
        self.assertGreater(actual.iloc[1]["max_cell_diff"], 0.05)
        self.assertGreater(actual.iloc[2]["max_cell_diff"], 0.05)


# #############################################################################
# Test_compare_reaction_time
# #############################################################################


class Test_compare_reaction_time(hunitest.TestCase):
    """
    Test `compare_reaction_time()`.
    """

    def test1(self) -> None:
        """
        Test that a reaction time never lowers `P(winner)`: it only shrinks
        player 1's reach.
        """
        # Prepare inputs.
        config = _get_tiny_config()
        module = racket_sports_05_variable_sensitivity_utils
        grid, p_winner_maps = module.run_speed_grid(config)
        # Run test.
        actual = module.compare_reaction_time(grid, p_winner_maps)
        # Check outputs.
        self.assertEqual(len(actual), 2 * 3 * 2)
        self.assertLessEqual(actual["d_best"].max(), 0.0)
        self.assertLess(actual["d_mean"].max(), 0.0)
        self.assertEqual(sorted(actual["t_r"].unique()), [0.2, 0.25])

    def test2(self) -> None:
        """
        Test the ball speeds within the widget's range: 1.5 m/s with `r = 0.3`
        is a 5 m/s ball, and with `r = 0.6` a 2.5 m/s ball.
        """
        # Prepare inputs.
        config = _get_tiny_config()
        module = racket_sports_05_variable_sensitivity_utils
        grid, p_winner_maps = module.run_speed_grid(config)
        # Run test.
        actual = module.compare_reaction_time(grid, p_winner_maps)
        # Check outputs.
        rows = actual[actual["v_p"] == 1.5].set_index("r")
        self.assertTrue(bool(rows.loc[0.3, "in_domain"].all()))
        self.assertFalse(bool(rows.loc[0.6, "in_domain"].any()))

    def test3(self) -> None:
        """
        Test that a grid without `t_r = 0` is rejected.
        """
        # Prepare inputs.
        grid = pd.DataFrame({"t_r": [0.2]})
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            racket_sports_05_variable_sensitivity_utils.compare_reaction_time(
                grid, {}
            )


# #############################################################################
# Test_summarize_reaction_time
# #############################################################################


class Test_summarize_reaction_time(hunitest.TestCase):
    """
    Test `summarize_reaction_time()`.
    """

    def test1(self) -> None:
        """
        Test the largest change, within the widget's ball speeds only.
        """
        # Prepare inputs.
        comparison = pd.DataFrame(
            {
                "sport": ["Tennis"] * 3,
                "v_p": [1.5] * 3,
                "t_r": [0.2] * 3,
                "d_best": [-0.01, -0.02, -0.5],
                "d_mean": [-0.03, -0.01, -0.5],
                "d_region": [-1, 2, 9],
                "d_centroid_m": [0.1, 0.2, 9.0],
                "max_cell_diff": [0.1, 0.05, 0.9],
                "in_domain": [True, True, False],
            }
        )
        # Run test.
        actual = (
            racket_sports_05_variable_sensitivity_utils.summarize_reaction_time(
                comparison
            )
        )
        # Check outputs.
        self.assertEqual(len(actual), 1)
        row = actual.iloc[0]
        self.assertAlmostEqual(row["best_points"], 2.0)
        self.assertAlmostEqual(row["mean_points"], 3.0)
        self.assertEqual(row["region_cells"], 2)
        self.assertEqual(row["centroid_m"], 0.2)
        self.assertEqual(row["max_cell_diff"], 0.1)


# #############################################################################
# Test_summarize_sensitivity
# #############################################################################


class Test_summarize_sensitivity(hunitest.TestCase):
    """
    Test `summarize_sensitivity()`.
    """

    def test1(self) -> None:
        """
        Test the ranges, with the setting that moves the most first.
        """
        # Prepare inputs.
        sweeps = _make_sweep_rows()
        # Run test.
        actual = (
            racket_sports_05_variable_sensitivity_utils.summarize_sensitivity(
                sweeps
            )
        )
        # Check outputs.
        self.assertEqual(actual["setting"].tolist(), ["v_p", "t_r"])
        np.testing.assert_allclose(actual["best_range_points"], [10.0, 1.0])
        np.testing.assert_allclose(actual["mean_range_points"], [30.0, 10.0])
        self.assertEqual(actual.iloc[0]["best_low"], 0.9)
        self.assertEqual(actual.iloc[0]["best_high"], 1.0)


# #############################################################################
# Test_save_paper_figure_data
# #############################################################################


class Test_save_paper_figure_data(hunitest.TestCase):
    """
    Test `save_paper_figure_data()`.
    """

    def test1(self) -> None:
        """
        Test that each file holds only the rows of its figure.
        """
        # Prepare inputs.
        scratch_dir = self.get_scratch_space()
        sweeps = pd.concat(
            [
                _make_sweep_rows(),
                _make_sweep_rows().assign(experiment="sweep_fast_p1"),
            ]
        )
        serve_map = pd.DataFrame(
            {
                "sport": ["Tennis", "Tennis"],
                "group": ["defaults", "fast_p1_slow_ball"],
                "x1": [-0.3, -0.3],
                "y1": [6.1, 6.1],
                "max_p_winner": [1.0, 0.8164],
            }
        )
        # Run test.
        paths = (
            racket_sports_05_variable_sensitivity_utils.save_paper_figure_data(
                sweeps, serve_map, scratch_dir
            )
        )
        # Check outputs.
        sweep_df = pd.read_csv(paths["sweeps"])
        self.assertEqual(len(sweep_df), 4)
        self.assertEqual(sweep_df["value"].tolist(), [0.0, 0.5, 0.0, 6.0])
        serve_df = pd.read_csv(paths["serve_depth"])
        self.assertEqual(serve_df["max_p_winner"].tolist(), [0.8164])

    def test2(self) -> None:
        """
        Test that a directory that does not exist is rejected.
        """
        # Prepare inputs.
        out_dir = os.path.join(self.get_scratch_space(), "missing")
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            racket_sports_05_variable_sensitivity_utils.save_paper_figure_data(
                _make_sweep_rows(), pd.DataFrame(), out_dir
            )


# #############################################################################
# Test_plot_sweeps
# #############################################################################


class Test_plot_sweeps(hunitest.TestCase):
    """
    Test `plot_sweeps()`.
    """

    def test1(self) -> None:
        """
        Test one row per swept setting, with 3 panels each.
        """
        # Prepare inputs.
        module = racket_sports_05_variable_sensitivity_utils
        settings = _get_tiny_settings("Pickleball")
        sweeps = pd.concat(
            [
                module.run_sweep(settings, "v_p", [0.0, 6.0], "defaults"),
                module.run_sweep(settings, "t_r", [0.0, 1.0], "defaults"),
            ]
        )
        # Run test.
        fig = module.plot_sweeps(sweeps, "sweep_defaults")
        # Check outputs.
        self.assertEqual(len(fig.axes), 2 * 3)
        line = fig.axes[0].get_lines()[-1]
        self.assertEqual(line.get_xdata().tolist(), [0.0, 6.0])
        plt.close("all")

    def test2(self) -> None:
        """
        Test that an experiment without runs is rejected.
        """
        # Prepare inputs.
        sweeps = _make_sweep_rows()
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            racket_sports_05_variable_sensitivity_utils.plot_sweeps(
                sweeps, "sweep_fast_p1"
            )


# #############################################################################
# Test_plot_montage
# #############################################################################


class Test_plot_montage(hunitest.TestCase):
    """
    Test `plot_montage()`.
    """

    def test1(self) -> None:
        """
        Test one map and one colorbar per panel, cropped to player 1's half.
        """
        # Prepare inputs.
        settings_list = [
            _get_tiny_settings("Pickleball", v_p=0.0),
            _get_tiny_settings("Pickleball", v_p=6.0),
        ]
        # Run test.
        fig = racket_sports_05_variable_sensitivity_utils.plot_montage(
            settings_list, ["v_p = 0", "v_p = 6"], "v_p sweep"
        )
        # Check outputs.
        self.assertEqual(len(fig.axes), 2 + 2)
        self.assertIn("v_p = 6", fig.axes[1].get_title())
        self.assertEqual(fig.axes[0].get_ylim()[1], 1.0)
        plt.close("all")

    def test2(self) -> None:
        """
        Test that a missing title is rejected.
        """
        # Prepare inputs.
        settings_list = [_get_tiny_settings("Tennis")] * 2
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            racket_sports_05_variable_sensitivity_utils.plot_montage(
                settings_list, ["one"], "title"
            )


# #############################################################################
# Test_plot_serve_table
# #############################################################################


class Test_plot_serve_table(hunitest.TestCase):
    """
    Test `plot_serve_table()`.
    """

    def helper(self, p: Dict[str, float], se: float) -> plt.Figure:
        """
        Plot one row of the table.

        :param p: the highest `P(winner)` per serve
        :param se: the standard error of each value
        :return: the figure
        """
        corners = pd.DataFrame(
            {
                "sport": "Tennis",
                "side": "deuce",
                "group": "ball=min v_p=max",
                "serve_name": list(p),
                "max_p_winner": list(p.values()),
                "max_p_winner_se": se,
                "mean_p_winner": 0.5,
            }
        )
        fig = racket_sports_05_variable_sensitivity_utils.plot_serve_table(
            corners, "Tennis"
        )
        return fig

    def test1(self) -> None:
        """
        Test that only a serve lower than the highest by more than 2 SE of the
        difference gets a box.
        """
        # Prepare inputs.
        p = {"wide-deep": 0.5, "T-deep": 0.995, "T-short": 1.0}
        # Run test.
        fig = self.helper(p, 0.002)
        # Check outputs.
        boxes = [
            patch
            for patch in fig.axes[0].patches
            if isinstance(patch, mpatches.Rectangle)
        ]
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].get_x(), -0.5)
        plt.close("all")

    def test2(self) -> None:
        """
        Test that a tie within noise has no box, and that each cell shows both
        numbers.
        """
        # Prepare inputs.
        p = {"wide-deep": 0.999, "T-deep": 1.0}
        # Run test.
        fig = self.helper(p, 0.002)
        # Check outputs.
        self.assertEqual(len(fig.axes[0].patches), 0)
        texts = [text.get_text() for text in fig.axes[0].texts]
        self.assertEqual(texts, ["0.999", "mean 0.500", "1.000", "mean 0.500"])
        plt.close("all")

    def test3(self) -> None:
        """
        Test that the table uses the colormap of `plot_winner_heatmap()`.
        """
        # Prepare inputs.
        module = racket_sports_01_shot_probability_utils
        sport = racket_params.PICKLEBALL
        heatmap = module.compute_winner_heatmap(
            (-1.5, 4.4), (0.5, 1.0), 12.0, 0.2, 1.5, sport.court, 3, 4, 1100, 1
        )
        _, ax = plt.subplots()
        module.plot_winner_heatmap(ax, heatmap, sport.court, (-1.5, 4.4))
        expected = ax.collections[0].get_cmap().name
        # Run test.
        fig = self.helper({"wide-deep": 0.5, "T-deep": 1.0}, 0.002)
        # Check outputs.
        self.assert_equal(fig.axes[0].images[0].get_cmap().name, expected)
        plt.close("all")


# #############################################################################
# Test_plot_serve_maps
# #############################################################################


class Test_plot_serve_maps(hunitest.TestCase):
    """
    Test `plot_serve_maps()`.
    """

    def test1(self) -> None:
        """
        Test one panel with a fixed scale from 0 to 1, and one shared colorbar.
        """
        # Prepare inputs.
        config = _get_tiny_config()
        module = racket_sports_05_variable_sensitivity_utils
        serve_map = module.run_serve_map(config)
        # Run test.
        fig = module.plot_serve_maps(serve_map)
        # Check outputs.
        self.assertEqual(len(fig.axes), 1 + 1)
        mesh = fig.axes[0].collections[0]
        self.assertEqual(mesh.get_clim(), (0.0, 1.0))
        self.assertIn("min", fig.axes[0].get_title())
        plt.close("all")


# #############################################################################
# Test_plot_ratio_collapse
# #############################################################################


class Test_plot_ratio_collapse(hunitest.TestCase):
    """
    Test `plot_ratio_collapse()`.
    """

    def test1(self) -> None:
        """
        Test that the 3 speed pairs draw one curve at `t_r = 0` but not at
        `t_r = 0.2` s.
        """
        # Prepare inputs.
        config = _get_tiny_config()
        module = racket_sports_05_variable_sensitivity_utils
        grid, _ = module.run_speed_grid(config)
        # Run test.
        fig = module.plot_ratio_collapse(grid, "Pickleball")
        # Check outputs.
        self.assertEqual(len(fig.axes), 6)
        # The 3rd panel of each row is the mean `P(winner)`.
        lines_t0 = [line.get_ydata() for line in fig.axes[2].get_lines()]
        lines_t2 = [line.get_ydata() for line in fig.axes[5].get_lines()]
        self.assertEqual(len(lines_t0), 3)
        np.testing.assert_array_equal(lines_t0[0], lines_t0[2])
        self.assertFalse(np.array_equal(lines_t2[0], lines_t2[2]))
        plt.close("all")


# #############################################################################
# Test_plot_reaction_time
# #############################################################################


class Test_plot_reaction_time(hunitest.TestCase):
    """
    Test `plot_reaction_time()`.
    """

    def test1(self) -> None:
        """
        Test 3 panels per sport and one line per reaction time.
        """
        # Prepare inputs.
        config = _get_tiny_config()
        module = racket_sports_05_variable_sensitivity_utils
        grid, _ = module.run_speed_grid(config)
        # Run test.
        fig = module.plot_reaction_time(grid, 1.5)
        # Check outputs.
        self.assertEqual(len(fig.axes), 3)
        self.assertEqual(len(fig.axes[0].get_lines()), 3)
        plt.close("all")

    def test2(self) -> None:
        """
        Test that a `v_p` outside the grid is rejected.
        """
        # Prepare inputs.
        grid = pd.DataFrame({"v_p": [1.5]})
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            racket_sports_05_variable_sensitivity_utils.plot_reaction_time(
                grid, 3.0
            )
