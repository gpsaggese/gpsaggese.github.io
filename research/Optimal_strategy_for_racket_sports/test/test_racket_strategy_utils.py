"""
Tests for racket_strategy_utils module.
"""

import logging
from typing import Dict, List, Optional, Tuple

import helpers.hunit_test as hunitest
import ipywidgets
import matplotlib
import pandas as pd

from research.Optimal_strategy_for_racket_sports import (
    racket_params,
    racket_scoring,
    racket_strategy_utils,
)

_LOG = logging.getLogger(__name__)

# Use non-interactive backend for testing.
matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def _make_scores(n_x: int, n_y: int, score_values: List[float]) -> pd.DataFrame:
    """
    Build the scores of a grid over the returner's half of a tennis court.

    :param n_x: number of grid columns
    :param n_y: number of grid rows
    :param score_values: score of each cell, in grid order
    :return: the cells of the grid, with a "score" column
    """
    sport = racket_params.TENNIS
    region = racket_params.get_half_court_region(sport.court)
    targets = racket_scoring.make_target_grid(region, n_x, n_y)
    scores = targets.copy()
    scores["score"] = score_values
    return scores


def _make_launch_table(n_x: int, n_y: int) -> pd.DataFrame:
    """
    Estimate the launch table of a tennis rally shot from `(0, -11)` to a grid
    over the returner's half, with 50 samples.

    :param n_x: number of grid columns
    :param n_y: number of grid rows
    :return: the launch table
    """
    sport = racket_params.TENNIS
    striker = racket_params.DEFAULT_PLAYER
    striker_pos = (0.0, -11.0)
    situation = racket_scoring.make_rally_situation(sport, striker, striker_pos)
    targets = racket_scoring.make_target_grid(
        racket_params.get_half_court_region(sport.court),
        n_x,
        n_y,
    )
    n_samples = 50
    seed = 0
    config = racket_scoring.ScoringConfig(n_samples=n_samples, seed=seed)
    launch_table = racket_scoring.estimate_launch_table(
        sport, situation, targets, config
    )
    return launch_table


# #############################################################################
# Test_draw_court
# #############################################################################


class Test_draw_court(hunitest.TestCase):
    """
    Test `racket_strategy_utils.draw_court()`.
    """

    def helper(
        self, court: racket_params.CourtGeometry, expected_lines: int
    ) -> None:
        """
        Helper for draw_court.

        :param court: Court object to draw
        :param expected_lines: Expected number of lines drawn
        """
        # Prepare inputs.
        _, ax = plt.subplots()
        # Run test.
        racket_strategy_utils.draw_court(ax, court)
        # Check outputs.
        self.assertEqual(len(ax.lines), expected_lines)
        plt.close("all")

    def test1(self) -> None:
        """
        Test that the tennis court draws the outer boundary, net, and
        service lines.
        """
        # Prepare inputs.
        court = racket_params.TENNIS.court
        # Prepare outputs.
        # Boundary (1) + net (1) + 2 service lines = 4 lines.
        expected_lines = 4
        # Run test.
        self.helper(court, expected_lines)

    def test2(self) -> None:
        """
        Test that the pickleball court also draws the non-volley zone
        lines.
        """
        # Prepare inputs.
        court = racket_params.PICKLEBALL.court
        # Prepare outputs.
        # Boundary (1) + net (1) + 2 service lines + 2 non-volley lines = 6.
        expected_lines = 6
        # Run test.
        self.helper(court, expected_lines)


# #############################################################################
# Test_plot_trajectory_fan
# #############################################################################


class Test_plot_trajectory_fan(hunitest.TestCase):
    """
    Test `racket_strategy_utils.plot_trajectory_fan()`.
    """

    def test1(self) -> None:
        """
        Test that the Figure 1 setup runs without error.
        """
        # Prepare inputs.
        sport = racket_params.TENNIS
        player = racket_params.PlayerParams(
            reaction_time_s=0.2,
            move_speed_mps=1.5,
            contact_height_m=1.0,
            error=racket_params.DEFAULT_ERROR,
        )
        start_pos = (0.0, -12.0)
        end_pos = (0.0, 8.0)
        time_s = 8.0
        _, ax = plt.subplots()
        # Run test.
        racket_strategy_utils.plot_trajectory_fan(
            sport, player, start_pos, end_pos, time_s, ax=ax
        )
        # Check outputs.
        self.assertGreater(len(ax.lines), 0)
        plt.close("all")


# #############################################################################
# Test_plot_court_heatmap
# #############################################################################


class Test_plot_court_heatmap(hunitest.TestCase):
    """
    Test `racket_strategy_utils.plot_court_heatmap()`.
    """

    def helper(
        self,
        scores: pd.DataFrame,
        markers: Optional[Dict[str, Tuple[float, float]]],
        expected: str,
    ) -> None:
        """
        Helper for plot_court_heatmap.

        :param scores: DataFrame with score data
        :param markers: marker positions by label, or None for no markers
        :param expected: expected number of cell rectangles and of markers
        """
        # Prepare inputs.
        column_name = "score"
        sport = racket_params.TENNIS
        _, ax = plt.subplots()
        # Run test.
        racket_strategy_utils.plot_court_heatmap(
            scores, column_name, sport, markers=markers, ax=ax
        )
        # Check outputs.
        n_rectangles = sum(
            1 for p in ax.patches if isinstance(p, mpatches.Rectangle)
        )
        # Each marker is a scatter of one point.
        actual = f"rectangles={n_rectangles} markers={len(ax.collections)}"
        plt.close("all")
        self.assert_equal(actual, expected)

    def test1(self) -> None:
        """
        Test that a 2x2 grid draws one patch per cell.
        """
        # Prepare inputs.
        scores = _make_scores(2, 2, [0.1, 0.4, 0.7, 0.9])
        markers = None
        # Prepare outputs.
        expected = "rectangles=4 markers=0"
        # Run test.
        self.helper(scores, markers, expected)

    def test2(self) -> None:
        """
        Test that markers add one scatter point per label.
        """
        # Prepare inputs.
        scores = _make_scores(2, 2, [0.1, 0.4, 0.7, 0.9])
        striker_pos = (0.0, -1.0)
        returner_pos = (1.0, 3.0)
        markers = {"Striker": striker_pos, "Returner": returner_pos}
        # Prepare outputs.
        expected = "rectangles=4 markers=2"
        # Run test.
        self.helper(scores, markers, expected)

    def test3(self) -> None:
        """
        Test edge case with minimal 1x1 grid draws one patch.
        """
        # Prepare inputs.
        scores = _make_scores(1, 1, [0.5])
        markers = None
        # Prepare outputs.
        expected = "rectangles=1 markers=0"
        # Run test.
        self.helper(scores, markers, expected)


# #############################################################################
# Test_plot_launch_tradeoff
# #############################################################################


class Test_plot_launch_tradeoff(hunitest.TestCase):
    """
    Test `racket_strategy_utils.plot_launch_tradeoff()`.
    """

    def test1(self) -> None:
        """
        Test that a small launch table plots without error.
        """
        # Prepare inputs.
        launch_table = _make_launch_table(1, 1)
        cell_id = int(launch_table["cell_id"].iloc[0])
        _, ax = plt.subplots()
        # Run test.
        racket_strategy_utils.plot_launch_tradeoff(launch_table, cell_id, ax=ax)
        # Check outputs.
        self.assertGreater(len(ax.lines), 0)
        plt.close("all")


# #############################################################################
# Test_build_score_widget
# #############################################################################


class Test_build_score_widget(hunitest.TestCase):
    """
    Test `racket_strategy_utils.build_score_widget()`.
    """

    def helper(self, launch_table: pd.DataFrame) -> None:
        """
        Helper for build_score_widget.

        :param launch_table: Launch table from estimate_launch_table
        """
        # Prepare inputs.
        sport = racket_params.TENNIS
        # Prepare outputs.
        expected = "widget=VBox controls=VBox n_controls=4"
        # Run test.
        widget = racket_strategy_utils.build_score_widget(
            sport, launch_table, racket_params.DEFAULT_PLAYER
        )
        # Check outputs.
        controls = widget.children[0]
        actual = (
            f"widget={type(widget).__name__} "
            f"controls={type(controls).__name__} "
            f"n_controls={len(controls.children)}"
        )
        plt.close("all")
        self.assert_equal(actual, expected)

    def test1(self) -> None:
        """
        Test that the widget builds without running the notebook UI.
        """
        # Prepare inputs.
        launch_table = _make_launch_table(2, 2)
        # Run test.
        self.helper(launch_table)

    def test2(self) -> None:
        """
        Test edge case with minimal 1x1 grid.
        """
        # Prepare inputs.
        launch_table = _make_launch_table(1, 1)
        # Run test.
        self.helper(launch_table)


# #############################################################################
# Test_build_exploration_widget
# #############################################################################


class Test_build_exploration_widget(hunitest.TestCase):
    """
    Test `racket_strategy_utils.build_exploration_widget()`.
    """

    def test1(self) -> None:
        """
        Test that the widget builds and renders a static default run.
        """
        # Prepare inputs.
        sport = racket_params.TENNIS
        n_samples = 20
        seed = 0
        config = racket_scoring.ScoringConfig(n_samples=n_samples, seed=seed)
        # Run test.
        widget = racket_strategy_utils.build_exploration_widget(sport, config)
        # Check outputs.
        self.assertIsInstance(widget, ipywidgets.VBox)
        controls, output = widget.children
        self.assertIsInstance(controls, ipywidgets.VBox)
        self.assertIsInstance(output, ipywidgets.Output)
        # 7 sliders + 1 run button.
        expected_children = 8
        self.assertEqual(len(controls.children), expected_children)
        self.assertIsInstance(controls.children[-1], ipywidgets.Button)
        plt.close("all")
