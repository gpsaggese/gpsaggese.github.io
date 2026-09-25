"""
Tests for racket_strategy_utils module.
"""

import logging

import ipywidgets
import matplotlib

import helpers.hunit_test as hunitest
from research.Optimal_strategy_for_racket_sports import racket_params
from research.Optimal_strategy_for_racket_sports import racket_scoring
from research.Optimal_strategy_for_racket_sports import racket_strategy_utils

_LOG = logging.getLogger(__name__)

# Use non-interactive backend for testing.
matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


# #############################################################################
# Test_draw_court
# #############################################################################


class Test_draw_court(hunitest.TestCase):
    """
    Test `racket_strategy_utils.draw_court()`.
    """

    def test1(self) -> None:
        """
        Test that the tennis court draws the outer boundary, net, and
        service lines.
        """
        # Prepare inputs.
        _, ax = plt.subplots()
        court = racket_params.TENNIS.court
        # Run test.
        racket_strategy_utils.draw_court(ax, court)
        # Check outputs.
        # Boundary (1) + net (1) + 2 service lines = 4 lines.
        self.assertEqual(len(ax.lines), 4)
        plt.close("all")

    def test2(self) -> None:
        """
        Test that the pickleball court also draws the non-volley zone
        lines.
        """
        # Prepare inputs.
        _, ax = plt.subplots()
        court = racket_params.PICKLEBALL.court
        # Run test.
        racket_strategy_utils.draw_court(ax, court)
        # Check outputs.
        # Boundary (1) + net (1) + 2 service lines + 2 non-volley lines = 6.
        self.assertEqual(len(ax.lines), 6)
        plt.close("all")


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
        _, ax = plt.subplots()
        sport = racket_params.TENNIS
        player = racket_params.PlayerParams(
            reaction_time_s=0.2,
            move_speed_mps=1.5,
            contact_height_m=1.0,
            error=racket_params.DEFAULT_ERROR,
        )
        # Run test.
        racket_strategy_utils.plot_trajectory_fan(
            sport, player, (0.0, -12.0), (0.0, 8.0), 8.0, ax=ax
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

    def test1(self) -> None:
        """
        Test that a 2x2 grid draws one patch per cell.
        """
        # Prepare inputs.
        sport = racket_params.TENNIS
        region = racket_params.get_half_court_region(sport.court)
        targets = racket_scoring.make_target_grid(region, 2, 2)
        scores = targets.copy()
        scores["score"] = [0.1, 0.4, 0.7, 0.9]
        _, ax = plt.subplots()
        # Run test.
        racket_strategy_utils.plot_court_heatmap(scores, "score", sport, ax=ax)
        # Check outputs.
        n_rectangles = sum(
            1 for p in ax.patches if isinstance(p, mpatches.Rectangle)
        )
        self.assertEqual(n_rectangles, 4)
        plt.close("all")

    def test2(self) -> None:
        """
        Test that markers add one scatter point per label.
        """
        # Prepare inputs.
        sport = racket_params.TENNIS
        region = racket_params.get_half_court_region(sport.court)
        targets = racket_scoring.make_target_grid(region, 2, 2)
        scores = targets.copy()
        scores["score"] = [0.1, 0.4, 0.7, 0.9]
        markers = {"Striker": (0.0, -1.0), "Returner": (1.0, 3.0)}
        _, ax = plt.subplots()
        # Run test.
        racket_strategy_utils.plot_court_heatmap(
            scores, "score", sport, markers=markers, ax=ax
        )
        # Check outputs.
        self.assertGreater(len(ax.collections), 0)
        plt.close("all")


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
        sport = racket_params.TENNIS
        striker = racket_params.DEFAULT_PLAYER
        situation = racket_scoring.make_rally_situation(
            sport, striker, (0.0, -11.0)
        )
        targets = racket_scoring.make_target_grid(
            racket_params.get_half_court_region(sport.court), 1, 1
        )
        config = racket_scoring.ScoringConfig(n_samples=50, seed=0)
        launch_table = racket_scoring.estimate_launch_table(
            sport, situation, targets, config
        )
        cell_id = int(targets["cell_id"].iloc[0])
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

    def test1(self) -> None:
        """
        Test that the widget builds without running the notebook UI.
        """
        # Prepare inputs.
        sport = racket_params.TENNIS
        striker = racket_params.DEFAULT_PLAYER
        situation = racket_scoring.make_rally_situation(
            sport, striker, (0.0, -11.0)
        )
        targets = racket_scoring.make_target_grid(
            racket_params.get_half_court_region(sport.court), 2, 2
        )
        config = racket_scoring.ScoringConfig(n_samples=50, seed=0)
        launch_table = racket_scoring.estimate_launch_table(
            sport, situation, targets, config
        )
        # Run test.
        widget = racket_strategy_utils.build_score_widget(
            sport, launch_table, racket_params.DEFAULT_PLAYER
        )
        # Check outputs.
        self.assertIsInstance(widget, ipywidgets.VBox)
        controls = widget.children[0]
        self.assertIsInstance(controls, ipywidgets.VBox)
        self.assertEqual(len(controls.children), 4)
        plt.close("all")


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
        config = racket_scoring.ScoringConfig(n_samples=20, seed=0)
        # Run test.
        widget = racket_strategy_utils.build_exploration_widget(sport, config)
        # Check outputs.
        self.assertIsInstance(widget, ipywidgets.VBox)
        controls, output = widget.children
        self.assertIsInstance(controls, ipywidgets.VBox)
        self.assertIsInstance(output, ipywidgets.Output)
        # 7 sliders + 1 run button.
        self.assertEqual(len(controls.children), 8)
        self.assertIsInstance(controls.children[-1], ipywidgets.Button)
        plt.close("all")
