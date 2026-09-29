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

    def helper(self, court, expected_lines: int) -> None:
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

    def helper(
        self,
        sport,
        player,
        start_pos,
        end_pos,
        time_s,
    ) -> None:
        """
        Helper for plot_trajectory_fan.

        :param sport: Sport configuration
        :param player: Player parameters
        :param start_pos: Starting position
        :param end_pos: Ending position
        :param time_s: Time in seconds
        """
        # Prepare inputs.
        _, ax = plt.subplots()
        # Run test.
        racket_strategy_utils.plot_trajectory_fan(
            sport, player, start_pos, end_pos, time_s, ax=ax
        )
        # Check outputs.
        self.assertGreater(len(ax.lines), 0)
        plt.close("all")

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
        # Run test.
        self.helper(sport, player, start_pos, end_pos, time_s)

    def test2(self) -> None:
        """
        Test edge case with minimal player speed and reaction time.
        """
        # Prepare inputs.
        sport = racket_params.TENNIS
        player = racket_params.PlayerParams(
            reaction_time_s=0.05,
            move_speed_mps=0.5,
            contact_height_m=1.0,
            error=racket_params.DEFAULT_ERROR,
        )
        start_pos = (0.0, -12.0)
        end_pos = (0.0, 8.0)
        time_s = 8.0
        # Run test.
        self.helper(sport, player, start_pos, end_pos, time_s)


# #############################################################################
# Test_plot_court_heatmap
# #############################################################################


class Test_plot_court_heatmap(hunitest.TestCase):
    """
    Test `racket_strategy_utils.plot_court_heatmap()`.
    """

    def helper(
        self,
        scores,
        column_name: str,
        sport,
        expected_rectangles: int = None,
        expected_collections: bool = False,
        markers=None,
    ) -> None:
        """
        Helper for plot_court_heatmap.

        :param scores: DataFrame with score data
        :param column_name: Name of the score column
        :param sport: Sport configuration
        :param expected_rectangles: Expected number of rectangles (None to skip check)
        :param expected_collections: Whether to expect collections (for markers)
        :param markers: Optional marker positions
        """
        # Prepare inputs.
        _, ax = plt.subplots()
        # Run test.
        racket_strategy_utils.plot_court_heatmap(
            scores, column_name, sport, markers=markers, ax=ax
        )
        # Check outputs.
        if expected_rectangles is not None:
            n_rectangles = sum(
                1 for p in ax.patches if isinstance(p, mpatches.Rectangle)
            )
            self.assertEqual(n_rectangles, expected_rectangles)
        if expected_collections:
            self.assertGreater(len(ax.collections), 0)
        plt.close("all")

    def test1(self) -> None:
        """
        Test that a 2x2 grid draws one patch per cell.
        """
        # Prepare inputs.
        sport = racket_params.TENNIS
        region = racket_params.get_half_court_region(sport.court)
        grid_rows = 2
        grid_cols = 2
        targets = racket_scoring.make_target_grid(region, grid_rows, grid_cols)
        scores = targets.copy()
        score_values = [0.1, 0.4, 0.7, 0.9]
        scores["score"] = score_values
        column_name = "score"
        # Prepare outputs.
        expected_rectangles = 4
        # Run test.
        self.helper(
            scores, column_name, sport, expected_rectangles=expected_rectangles
        )

    def test2(self) -> None:
        """
        Test that markers add one scatter point per label.
        """
        # Prepare inputs.
        sport = racket_params.TENNIS
        region = racket_params.get_half_court_region(sport.court)
        grid_rows = 2
        grid_cols = 2
        targets = racket_scoring.make_target_grid(region, grid_rows, grid_cols)
        scores = targets.copy()
        score_values = [0.1, 0.4, 0.7, 0.9]
        scores["score"] = score_values
        striker_pos = (0.0, -1.0)
        returner_pos = (1.0, 3.0)
        markers = {"Striker": striker_pos, "Returner": returner_pos}
        column_name = "score"
        # Run test.
        self.helper(
            scores,
            column_name,
            sport,
            expected_collections=True,
            markers=markers,
        )

    def test3(self) -> None:
        """
        Test edge case with minimal 1x1 grid draws one patch.
        """
        # Prepare inputs.
        sport = racket_params.TENNIS
        region = racket_params.get_half_court_region(sport.court)
        grid_rows = 1
        grid_cols = 1
        targets = racket_scoring.make_target_grid(region, grid_rows, grid_cols)
        scores = targets.copy()
        score_values = [0.5]
        scores["score"] = score_values
        column_name = "score"
        # Prepare outputs.
        expected_rectangles = 1
        # Run test.
        self.helper(
            scores, column_name, sport, expected_rectangles=expected_rectangles
        )


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
        striker_pos = (0.0, -11.0)
        situation = racket_scoring.make_rally_situation(
            sport, striker, striker_pos
        )
        grid_rows = 1
        grid_cols = 1
        targets = racket_scoring.make_target_grid(
            racket_params.get_half_court_region(sport.court),
            grid_rows,
            grid_cols,
        )
        n_samples = 50
        seed = 0
        config = racket_scoring.ScoringConfig(n_samples=n_samples, seed=seed)
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

    def helper(
        self,
        launch_table,
        expected_children_count: int = None,
        check_controls: bool = False,
    ) -> None:
        """
        Helper for build_score_widget.

        :param launch_table: Launch table from estimate_launch_table
        :param expected_children_count: Expected number of widget children (None to skip)
        :param check_controls: Whether to check controls structure
        """
        # Prepare inputs.
        sport = racket_params.TENNIS
        # Run test.
        widget = racket_strategy_utils.build_score_widget(
            sport, launch_table, racket_params.DEFAULT_PLAYER
        )
        # Check outputs.
        self.assertIsInstance(widget, ipywidgets.VBox)
        if check_controls:
            controls = widget.children[0]
            self.assertIsInstance(controls, ipywidgets.VBox)
            self.assertEqual(len(controls.children), 4)
        if expected_children_count is not None:
            self.assertGreater(len(widget.children), expected_children_count)
        plt.close("all")

    def test1(self) -> None:
        """
        Test that the widget builds without running the notebook UI.
        """
        # Prepare inputs.
        sport = racket_params.TENNIS
        striker = racket_params.DEFAULT_PLAYER
        striker_pos = (0.0, -11.0)
        situation = racket_scoring.make_rally_situation(
            sport, striker, striker_pos
        )
        grid_rows = 2
        grid_cols = 2
        targets = racket_scoring.make_target_grid(
            racket_params.get_half_court_region(sport.court),
            grid_rows,
            grid_cols,
        )
        n_samples = 50
        seed = 0
        config = racket_scoring.ScoringConfig(n_samples=n_samples, seed=seed)
        launch_table = racket_scoring.estimate_launch_table(
            sport, situation, targets, config
        )
        # Run test.
        self.helper(launch_table, check_controls=True)

    def test2(self) -> None:
        """
        Test edge case with minimal 1x1 grid.
        """
        # Prepare inputs.
        sport = racket_params.TENNIS
        striker = racket_params.DEFAULT_PLAYER
        striker_pos = (0.0, -11.0)
        situation = racket_scoring.make_rally_situation(
            sport, striker, striker_pos
        )
        grid_rows = 1
        grid_cols = 1
        targets = racket_scoring.make_target_grid(
            racket_params.get_half_court_region(sport.court),
            grid_rows,
            grid_cols,
        )
        n_samples = 50
        seed = 0
        config = racket_scoring.ScoringConfig(n_samples=n_samples, seed=seed)
        launch_table = racket_scoring.estimate_launch_table(
            sport, situation, targets, config
        )
        # Run test.
        self.helper(launch_table, expected_children_count=0)


# #############################################################################
# Test_build_exploration_widget
# #############################################################################


class Test_build_exploration_widget(hunitest.TestCase):
    """
    Test `racket_strategy_utils.build_exploration_widget()`.
    """

    def helper(
        self,
        sport,
        config,
        check_controls_count: bool = False,
    ) -> None:
        """
        Helper for build_exploration_widget.

        :param sport: Sport configuration
        :param config: Scoring configuration
        :param check_controls_count: Whether to verify controls has 8 children
        """
        # Run test.
        widget = racket_strategy_utils.build_exploration_widget(sport, config)
        # Check outputs.
        self.assertIsInstance(widget, ipywidgets.VBox)
        controls, output = widget.children
        self.assertIsInstance(controls, ipywidgets.VBox)
        self.assertIsInstance(output, ipywidgets.Output)
        if check_controls_count:
            # 7 sliders + 1 run button.
            expected_children = 8
            self.assertEqual(len(controls.children), expected_children)
            self.assertIsInstance(controls.children[-1], ipywidgets.Button)
        plt.close("all")

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
        self.helper(sport, config, check_controls_count=True)

    def test2(self) -> None:
        """
        Test edge case with minimal sample size.
        """
        # Prepare inputs.
        sport = racket_params.TENNIS
        n_samples = 5
        seed = 0
        config = racket_scoring.ScoringConfig(n_samples=n_samples, seed=seed)
        # Run test.
        self.helper(sport, config)
