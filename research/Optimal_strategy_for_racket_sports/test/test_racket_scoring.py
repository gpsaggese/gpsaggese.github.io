"""
Tests for racket_scoring module.
"""

import logging

import numpy as np
import pandas as pd

import helpers.hunit_test as hunitest
from research.Optimal_strategy_for_racket_sports import racket_params
from research.Optimal_strategy_for_racket_sports import racket_scoring

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_make_target_grid
# #############################################################################


class Test_make_target_grid(hunitest.TestCase):
    """
    Test `racket_scoring.make_target_grid()`.
    """

    def test1(self) -> None:
        """
        Test a 2x3 grid on the unit region: cell centers and bounds.
        """
        # Prepare inputs.
        region = racket_params.CourtRegion(
            x_min=0.0, x_max=2.0, y_min=0.0, y_max=3.0
        )
        n_x = 2
        n_y = 3
        # Run test.
        actual = racket_scoring.make_target_grid(region, n_x, n_y)
        # Check outputs.
        self.assertEqual(len(actual), n_x * n_y)
        self.assertEqual(
            sorted(actual["cell_id"].tolist()), list(range(n_x * n_y))
        )
        first_row = actual.iloc[0]
        self.assertAlmostEqual(first_row["x_min"], 0.0)
        self.assertAlmostEqual(first_row["x_max"], 1.0)
        self.assertAlmostEqual(first_row["x_m"], 0.5)
        self.assertAlmostEqual(first_row["y_min"], 0.0)
        self.assertAlmostEqual(first_row["y_max"], 1.0)
        self.assertAlmostEqual(first_row["y_m"], 0.5)
        last_row = actual.iloc[-1]
        self.assertAlmostEqual(last_row["x_min"], 1.0)
        self.assertAlmostEqual(last_row["x_max"], 2.0)
        self.assertAlmostEqual(last_row["y_min"], 2.0)
        self.assertAlmostEqual(last_row["y_max"], 3.0)

    def test2(self) -> None:
        """
        Test a single cell (1x1 grid): edge case of minimal grid size.
        """
        # Prepare inputs.
        region = racket_params.CourtRegion(
            x_min=0.0, x_max=4.0, y_min=0.0, y_max=6.0
        )
        n_x = 1
        n_y = 1
        # Run test.
        actual = racket_scoring.make_target_grid(region, n_x, n_y)
        # Check outputs.
        self.assertEqual(len(actual), 1)
        self.assertEqual(actual.iloc[0]["cell_id"], 0)
        self.assertAlmostEqual(actual.iloc[0]["x_m"], 2.0)
        self.assertAlmostEqual(actual.iloc[0]["y_m"], 3.0)


# #############################################################################
# Test_compute_reachability
# #############################################################################


class Test_compute_reachability(hunitest.TestCase):
    """
    Test `racket_scoring.compute_reachability()`, reproducing Table II.
    """

    def helper(self, flight_time_s: float, expected: list) -> None:
        """
        Test helper for `compute_reachability()`.

        :param flight_time_s: nominal ball flight time, in seconds
        :param expected: expected reachability per cell
        """
        # Prepare inputs.
        dist_m = np.array([0.3, 0.7, 1.1, 1.5, 1.9])
        flight_time = np.full(5, flight_time_s)
        returner = racket_params.PlayerParams(
            reaction_time_s=0.2,
            move_speed_mps=1.5,
            contact_height_m=1.0,
            error=racket_params.DEFAULT_ERROR,
        )
        # Run test.
        actual = racket_scoring.compute_reachability(
            dist_m, flight_time, returner
        )
        # Check outputs.
        self.assertEqual(actual.tolist(), expected)

    def test1(self) -> None:
        """
        Test the tennis regime: `T_f = 0.8` s, reach radius 0.9 m.
        """
        # Prepare inputs.
        flight_time_s = 0.8
        # Prepare outputs.
        expected = [True, True, False, False, False]
        # Run test and check outputs.
        self.helper(flight_time_s, expected)

    def test2(self) -> None:
        """
        Test the pickleball regime: `T_f = 0.2` s, no time to move.
        """
        # Prepare inputs.
        flight_time_s = 0.2
        # Prepare outputs.
        expected = [False, False, False, False, False]
        # Run test and check outputs.
        self.helper(flight_time_s, expected)

    def test3(self) -> None:
        """
        Test that a `NaN` flight time (no feasible launch) is unreachable.
        """
        # Prepare inputs.
        dist_m = np.array([0.0])
        flight_time = np.array([np.nan])
        returner = racket_params.DEFAULT_PLAYER
        # Run test.
        actual = racket_scoring.compute_reachability(
            dist_m, flight_time, returner
        )
        # Check outputs.
        self.assertFalse(bool(actual[0]))


# #############################################################################
# Test_compute_score
# #############################################################################


class Test_compute_score(hunitest.TestCase):
    """
    Test `racket_scoring.compute_score()`, reproducing Table II.
    """

    def test1(self) -> None:
        """
        Test the tennis regime: argmax is `c_3`.
        """
        # Prepare inputs.
        p_in = np.array([0.95, 0.91, 0.85, 0.77, 0.65])
        reachable = np.array([True, True, False, False, False])
        # Prepare outputs.
        expected = [0.0, 0.0, 0.85, 0.77, 0.65]
        expected_argmax = 2
        # Run test.
        actual = racket_scoring.compute_score(p_in, reachable)
        # Check outputs.
        np.testing.assert_allclose(actual, expected, atol=1e-9)
        self.assertEqual(int(np.argmax(actual)), expected_argmax)

    def test2(self) -> None:
        """
        Test the pickleball regime: argmax is `c_1`.
        """
        # Prepare inputs.
        p_in = np.array([0.95, 0.91, 0.85, 0.77, 0.65])
        reachable = np.array([False, False, False, False, False])
        # Prepare outputs.
        expected_argmax = 0
        # Run test.
        actual = racket_scoring.compute_score(p_in, reachable)
        # Check outputs.
        np.testing.assert_allclose(actual, p_in, atol=1e-9)
        self.assertEqual(int(np.argmax(actual)), expected_argmax)


# #############################################################################
# Test_estimate_launch_table
# #############################################################################


class Test_estimate_launch_table(hunitest.TestCase):
    """
    Test `racket_scoring.estimate_launch_table()`.
    """

    def test1(self) -> None:
        """
        Test that zero error gives `p_in = 1` for a target inside the legal
        region and `p_in = 0` for a target outside it.
        """
        # Prepare inputs.
        sport = racket_params.TENNIS
        zero_error = racket_params.ShotErrorModel(
            sigma_theta_rad=0.0, sigma_v_frac=0.0, sigma_phi_rad=0.0
        )
        striker = racket_params.PlayerParams(
            reaction_time_s=0.2,
            move_speed_mps=1.5,
            contact_height_m=1.0,
            error=zero_error,
        )
        # Legal region only covers the near half of the returner's court.
        legal_region = racket_params.CourtRegion(
            x_min=-4.0, x_max=4.0, y_min=0.0, y_max=5.0
        )
        situation = racket_scoring.ShotSituation(
            striker=striker, striker_xy=(0.0, -11.0), legal_region=legal_region
        )
        targets = racket_scoring.make_target_grid(
            racket_params.CourtRegion(
                x_min=-0.5, x_max=0.5, y_min=3.0, y_max=8.0
            ),
            1,
            2,
        )
        config = racket_scoring.ScoringConfig(n_samples=1, seed=0)
        # Prepare outputs.
        expected_p_in_inside = 1.0
        expected_p_in_outside = 0.0
        min_rows_required = 0
        # Run test.
        actual = racket_scoring.estimate_launch_table(
            sport, situation, targets, config
        )
        # Check outputs.
        inside_cell_id = targets.loc[targets["y_m"] < 5.0, "cell_id"].iloc[0]
        outside_cell_id = targets.loc[targets["y_m"] > 5.0, "cell_id"].iloc[0]
        inside_rows = actual[actual["cell_id"] == inside_cell_id]
        outside_rows = actual[actual["cell_id"] == outside_cell_id]
        self.assertGreater(len(inside_rows), min_rows_required)
        self.assertTrue((inside_rows["p_in"] == expected_p_in_inside).all())
        if len(outside_rows) > 0:
            self.assertTrue(
                (outside_rows["p_in"] == expected_p_in_outside).all()
            )

    def test2(self) -> None:
        """
        Test that lateral-only error at the sideline gives `p_in ~= 0.5`.
        """
        # Prepare inputs.
        sport = racket_params.TENNIS
        lateral_error = racket_params.ShotErrorModel(
            sigma_theta_rad=0.0,
            sigma_v_frac=0.0,
            sigma_phi_rad=np.radians(3.0),
        )
        striker = racket_params.PlayerParams(
            reaction_time_s=0.2,
            move_speed_mps=1.5,
            contact_height_m=1.0,
            error=lateral_error,
        )
        legal_region = racket_params.get_half_court_region(sport.court)
        # A single cell centered exactly on the singles sideline.
        x_side = sport.court.width_m / 2
        situation = racket_scoring.ShotSituation(
            striker=striker, striker_xy=(0.0, -11.0), legal_region=legal_region
        )
        targets = racket_scoring.make_target_grid(
            racket_params.CourtRegion(
                x_min=x_side - 0.01, x_max=x_side + 0.01, y_min=4.0, y_max=6.0
            ),
            1,
            1,
        )
        config = racket_scoring.ScoringConfig(n_samples=5000, seed=1)
        # Prepare outputs.
        expected_p_in = 0.5
        num_std_errs = 3
        # Run test.
        actual = racket_scoring.estimate_launch_table(
            sport, situation, targets, config
        )
        # Check outputs.
        p_in = actual["p_in"].mean()
        p_in_se = actual["p_in_se"].mean()
        self.assertLess(abs(p_in - expected_p_in), num_std_errs * p_in_se)

    def test3(self) -> None:
        """
        Test that a cell with no feasible angle gets `p_in = 0`, `theta_deg
        = NaN`.
        """
        # Prepare inputs.
        sport = racket_params.SportParams(
            name="Tiny",
            court=racket_params.TENNIS.court,
            max_ball_speed_mps=0.5,
        )
        striker = racket_params.DEFAULT_PLAYER
        situation = racket_scoring.make_rally_situation(
            sport, striker, (0.0, -11.0)
        )
        targets = racket_scoring.make_target_grid(
            racket_params.get_half_court_region(sport.court), 1, 1
        )
        config = racket_scoring.ScoringConfig(n_samples=10, seed=0)
        # Prepare outputs.
        expected_len = 1
        expected_p_in = 0.0
        # Run test.
        actual = racket_scoring.estimate_launch_table(
            sport, situation, targets, config
        )
        # Check outputs.
        self.assertEqual(len(actual), expected_len)
        self.assertEqual(actual.iloc[0]["p_in"], expected_p_in)
        self.assertTrue(np.isnan(actual.iloc[0]["theta_deg"]))


# #############################################################################
# Test_score_targets
# #############################################################################


class Test_score_targets(hunitest.TestCase):
    """
    Test `racket_scoring.score_targets()`.
    """

    def test1(self) -> None:
        """
        Test a tiny grid with a fixed seed against the whole output frame.
        """
        # Prepare inputs.
        sport = racket_params.TENNIS
        striker = racket_params.DEFAULT_PLAYER
        situation = racket_scoring.make_rally_situation(
            sport, striker, (0.0, -11.0)
        )
        region = racket_params.get_half_court_region(sport.court)
        targets = racket_scoring.make_target_grid(region, 2, 2)
        config = racket_scoring.ScoringConfig(n_samples=200, seed=1)
        launch_table = racket_scoring.estimate_launch_table(
            sport, situation, targets, config
        )
        returner = racket_params.DEFAULT_PLAYER
        returner_xy = (0.0, 6.0)
        # Prepare outputs.
        min_score = 0
        max_score_tolerance = 1e-9
        # Run test.
        actual = racket_scoring.score_targets(
            launch_table, returner, returner_xy
        )
        # Check outputs.
        self.assertEqual(
            sorted(actual["cell_id"].tolist()),
            sorted(targets["cell_id"].tolist()),
        )
        self.assertTrue((actual["score"] >= min_score).all())
        self.assertTrue(
            (actual["score"] <= actual["p_in"] + max_score_tolerance).all()
        )

    def test2(self) -> None:
        """
        Test edge case with a single target cell.
        """
        # Prepare inputs.
        sport = racket_params.TENNIS
        striker = racket_params.DEFAULT_PLAYER
        situation = racket_scoring.make_rally_situation(
            sport, striker, (0.0, -11.0)
        )
        region = racket_params.get_half_court_region(sport.court)
        targets = racket_scoring.make_target_grid(region, 1, 1)
        config = racket_scoring.ScoringConfig(n_samples=200, seed=1)
        launch_table = racket_scoring.estimate_launch_table(
            sport, situation, targets, config
        )
        returner = racket_params.DEFAULT_PLAYER
        returner_xy = (0.0, 6.0)
        # Prepare outputs.
        expected_len = 1
        min_score = 0
        max_score_tolerance = 1e-9
        # Run test.
        actual = racket_scoring.score_targets(
            launch_table, returner, returner_xy
        )
        # Check outputs.
        self.assertEqual(len(actual), expected_len)
        self.assertTrue(actual["score"].iloc[0] >= min_score)
        self.assertTrue(
            actual["score"].iloc[0]
            <= actual["p_in"].iloc[0] + max_score_tolerance
        )


# #############################################################################
# Test_select_best_cell
# #############################################################################


class Test_select_best_cell(hunitest.TestCase):
    """
    Test `racket_scoring.select_best_cell()`.
    """

    def test1(self) -> None:
        """
        Test that the row with the highest score is selected.
        """
        # Prepare inputs.
        scores = pd.DataFrame(
            {
                "cell_id": [0, 1, 2],
                "score": [0.1, 0.9, 0.5],
            }
        )
        # Prepare outputs.
        expected_cell_id = 1
        expected_score = 0.9
        # Run test.
        actual = racket_scoring.select_best_cell(scores)
        # Check outputs.
        self.assertEqual(int(actual["cell_id"]), expected_cell_id)
        self.assertAlmostEqual(float(actual["score"]), expected_score)

    def test2(self) -> None:
        """
        Test edge case with a single row.
        """
        # Prepare inputs.
        scores = pd.DataFrame(
            {
                "cell_id": [42],
                "score": [0.7],
            }
        )
        # Prepare outputs.
        expected_cell_id = 42
        expected_score = 0.7
        # Run test.
        actual = racket_scoring.select_best_cell(scores)
        # Check outputs.
        self.assertEqual(int(actual["cell_id"]), expected_cell_id)
        self.assertAlmostEqual(float(actual["score"]), expected_score)

    def test3(self) -> None:
        """
        Test edge case with tied scores: returns first occurrence.
        """
        # Prepare inputs.
        scores = pd.DataFrame(
            {
                "cell_id": [0, 1, 2],
                "score": [0.5, 0.8, 0.8],
            }
        )
        # Prepare outputs.
        expected_cell_id = 1
        expected_score = 0.8
        # Run test.
        actual = racket_scoring.select_best_cell(scores)
        # Check outputs: should return first row with max score
        self.assertAlmostEqual(float(actual["score"]), expected_score)
        self.assertEqual(int(actual["cell_id"]), expected_cell_id)


# #############################################################################
# Test_make_serve_situation
# #############################################################################


class Test_make_serve_situation(hunitest.TestCase):
    """
    Test `racket_scoring.make_serve_situation()`.
    """

    def helper(self, sport: racket_params.SportParams, side: str) -> None:
        """
        Test helper for `make_serve_situation()`: every grid cell center
        lies in the service box.

        :param sport: sport parameters to build the serve situation for
        :param side: court side ("deuce" or "ad")
        """
        # Prepare inputs.
        server = racket_params.SERVE_PLAYER_TENNIS
        grid_size = 3
        striker_y_index = 1
        expected_striker_y_sign = -1
        # Run test.
        situation = racket_scoring.make_serve_situation(sport, server, side)
        targets = racket_scoring.make_target_grid(
            situation.legal_region, grid_size, grid_size
        )
        # Check outputs.
        service_box = racket_params.get_service_box_region(sport.court, side)
        in_box = service_box.contains(
            targets["x_m"].to_numpy(), targets["y_m"].to_numpy()
        )
        self.assertTrue(bool(in_box.all()))
        self.assertLess(
            situation.striker_xy[striker_y_index], expected_striker_y_sign
        )

    def test1(self) -> None:
        """
        Test the deuce-side serve situation for tennis.
        """
        # Prepare inputs.
        side = "deuce"
        # Run test.
        self.helper(racket_params.TENNIS, side)

    def test2(self) -> None:
        """
        Test the deuce-side serve situation for pickleball.
        """
        # Prepare inputs.
        side = "deuce"
        # Run test.
        self.helper(racket_params.PICKLEBALL, side)

    def test3(self) -> None:
        """
        Test the ad-side serve situation for tennis: edge case of different court side.
        """
        # Prepare inputs.
        side = "ad"
        # Run test.
        self.helper(racket_params.TENNIS, side)
