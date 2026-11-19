"""
Tests for racket_sports_06_rl_return_utils module.

Import as:

import research.Optimal_strategy_for_racket_sports.test.test_racket_sports_06_rl_return_utils as rosfrsttrs0rru
"""

import logging
import pprint
import warnings
from typing import Tuple

import pytest

# Skip this file if `gymnasium` is not installed: the image does not have it,
# and the notebook installs it in its pip cell.
pytest.importorskip("gymnasium")

import gymnasium.utils.env_checker  # noqa: E402
import helpers.hunit_test as hunitest  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research.Optimal_strategy_for_racket_sports import (  # noqa: E402
    racket_params,
    racket_scoring,
)
from research.Optimal_strategy_for_racket_sports.notebooks import (  # noqa: E402
    racket_sports_01_shot_probability_utils,
    racket_sports_06_rl_return_utils,
)

_LOG = logging.getLogger(__name__)

# Use the non-interactive backend so the figures render without a display.
matplotlib.use("Agg")

# A legal tennis deuce serve, near the middle of the box.
_SERVE_XY = (-2.0, 3.0)


def _make_mixed_env() -> Tuple[racket_sports_06_rl_return_utils.ReturnEnv, int]:
    """
    Build a tennis env and a target near player 1 that give all 3 outcomes.

    A wide spread puts some returns out past the baseline, and a fast player 1
    reaches many of the others, but not all.

    :return: the env, and the cell closest to player 1
    """
    std_xy = (2.0, 2.0)
    move_speed_mps = 5.0
    env = racket_sports_06_rl_return_utils.build_env(
        racket_params.TENNIS, std_xy=std_xy, move_speed_mps=move_speed_mps
    )
    action = racket_sports_06_rl_return_utils.get_nearest_action(
        env, env.player1_xy
    )
    return env, action


def _make_heatmap_pair() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a brute force map and a Milestone 2 map of 4 cells that disagree on
    the top.

    - Brute force: the highest cell is 1 (0.875), and its best region is
      cells 1 and 2, since 0.75 is within 2 SE
    - Milestone 2: the highest cell is 2 (0.875), and its best region is cell
      2 only, since the standard errors are small
    - Ranks: `[2, 4, 3, 1]` and `[2, 3, 4, 1]`, so Spearman is
      `1 - 6 * 2 / (4 * 15) = 0.8`

    :return: the brute force map and the Milestone 2 map
    """
    cells = {
        "cell_id": [0, 1, 2, 3],
        "x_m": [-1.0, 1.0, -1.0, 1.0],
        "y_m": [-3.0, -3.0, -1.0, -1.0],
    }
    env_heatmap = pd.DataFrame(
        {
            **cells,
            "p_winner": [0.5, 0.875, 0.75, 0.25],
            "p_winner_se": [0.0625] * 4,
        }
    )
    m2_heatmap = pd.DataFrame(
        {
            **cells,
            "p_winner": [0.5, 0.75, 0.875, 0.125],
            "p_winner_se": [0.001] * 4,
        }
    )
    return env_heatmap, m2_heatmap


# #############################################################################
# TestReturnEnv
# #############################################################################


class TestReturnEnv(hunitest.TestCase):
    """
    Test `ReturnEnv`.
    """

    def test1(self) -> None:
        """
        Test that the env passes `check_env()` with warnings as errors, in
        both sports.
        """
        # Prepare inputs.
        envs = [
            racket_sports_06_rl_return_utils.build_env(racket_params.TENNIS),
            racket_sports_06_rl_return_utils.build_env(racket_params.PICKLEBALL),
        ]
        # Run test and check outputs.
        for env in envs:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                gymnasium.utils.env_checker.check_env(
                    env, skip_render_check=True
                )

    def test2(self) -> None:
        """
        Test that the reward is 1 exactly for a winner: in and not reached.
        """
        # Prepare inputs.
        env, action = _make_mixed_env()
        n_points = 200
        # Prepare outputs.
        # `(reward, is_in, is_reached)`: out, out and reached, reached, winner;
        # each occurs, and only a winner gets 1.
        expected = [
            (0.0, False, False),
            (0.0, False, True),
            (0.0, True, True),
            (1.0, True, False),
        ]
        # Run test.
        outcomes = set()
        for seed in range(n_points):
            env.reset(seed=seed, options={"serve_xy": _SERVE_XY})
            _, reward, _, _, info = env.step(action)
            outcomes.add((reward, info["is_in"], info["is_reached"]))
        # Check outputs.
        self.assert_equal(str(sorted(outcomes)), str(expected))

    def test3(self) -> None:
        """
        Test that the same seeds give the same points, from different serves.
        """
        # Prepare inputs.
        (env_1, action), (env_2, _) = _make_mixed_env(), _make_mixed_env()
        n_points = 50
        # Prepare outputs.
        # The rewards vary and every serve differs, so the 2 runs cannot agree
        # by chance.
        expected = str(([0.0, 1.0], n_points))
        # Run test.
        runs = []
        for env in (env_1, env_2):
            points = []
            for seed in range(n_points):
                obs, _ = env.reset(seed=seed)
                _, reward, _, _, info = env.step(action)
                points.append((obs.tolist(), reward, info["is_in"]))
            runs.append(points)
        # Check outputs.
        self.assert_equal(str(runs[1]), str(runs[0]))
        rewards = sorted({reward for _, reward, _ in runs[0]})
        n_serves = len({tuple(obs) for obs, _, _ in runs[0]})
        self.assert_equal(str((rewards, n_serves)), expected)

    def test4(self) -> None:
        """
        Test that one step ends the point, aimed at the center of the cell of
        the action.
        """
        # Prepare inputs.
        env = racket_sports_06_rl_return_utils.build_env(racket_params.TENNIS)
        # Prepare outputs.
        # The Milestone 2 grid of tennis, built without the env.
        region = racket_sports_01_shot_probability_utils.get_striker_half_region(
            racket_params.TENNIS.court
        )
        n_x, n_y = racket_sports_01_shot_probability_utils.HEATMAP_GRID_SIZE[
            "Tennis"
        ]
        grid = racket_scoring.make_target_grid(region, n_x, n_y)
        expected = [
            ((float(x), float(y)), True, False)
            for x, y in zip(grid["x_m"], grid["y_m"])
        ]
        # Run test.
        actual = []
        for action in range(len(env.targets)):
            env.reset(seed=0, options={"serve_xy": _SERVE_XY})
            _, _, terminated, truncated, info = env.step(action)
            actual.append((info["target_xy"], terminated, truncated))
        # Check outputs.
        self.assert_equal(str(actual), str(expected))

    def test5(self) -> None:
        """
        Test that a serve passed in `options` fixes the observation: player 1
        at the center of the baseline, player 2 and the serve at `serve_xy`.
        """
        # Prepare inputs.
        env = racket_sports_06_rl_return_utils.build_env(racket_params.TENNIS)
        # Prepare outputs.
        expected = str(([0.0, -11.885, -2.0, 3.0, -2.0, 3.0], {}))
        # Run test.
        obs, info = env.reset(seed=0, options={"serve_xy": _SERVE_XY})
        # Check outputs.
        # Round away the float32 digits, e.g., -11.885 is -11.88500023.
        actual = str(([round(float(value), 6) for value in obs], info))
        self.assert_equal(actual, expected)


# #############################################################################
# Test_get_nearest_action
# #############################################################################


class Test_get_nearest_action(hunitest.TestCase):
    """
    Test `get_nearest_action()`.
    """

    def test1(self) -> None:
        """
        Test that the center of a cell gives that cell.
        """
        # Prepare inputs.
        env = racket_sports_06_rl_return_utils.build_env(racket_params.TENNIS)
        actions = [0, 7, 149]
        # Prepare outputs.
        expected = str(actions)
        # Run test.
        actual = [
            racket_sports_06_rl_return_utils.get_nearest_action(
                env, (env.targets["x_m"][a], env.targets["y_m"][a])
            )
            for a in actions
        ]
        # Check outputs.
        self.assert_equal(str(actual), expected)


# #############################################################################
# Test_compute_env_heatmap
# #############################################################################


class Test_compute_env_heatmap(hunitest.TestCase):
    """
    Test `compute_env_heatmap()`.
    """

    def test1(self) -> None:
        """
        Test that the map has the cells and the columns of the Milestone 2 map,
        and agrees with it within 4 standard errors in every cell.

        The standard error of the difference of the 2 fractions uses their
        pooled fraction, the usual test for 2 proportions.
        """
        # Prepare inputs.
        # 3 x 4 cells, and a wide spread that keeps every `P(winner)` away from
        # 0 and 1, so the cells differ and every standard error is positive.
        grid_step_m = 3.0
        ball_speed_mps = 25.0
        std_xy = (3.0, 3.0)
        env = racket_sports_06_rl_return_utils.ReturnEnv(
            racket_params.TENNIS, grid_step_m, ball_speed_mps, std_xy=std_xy
        )
        n_samples = 200
        seed = 1000
        m2_n_samples = 5000
        m2_seed = 2
        # Run test.
        env_heatmap = racket_sports_06_rl_return_utils.compute_env_heatmap(
            env, _SERVE_XY, n_samples, seed
        )
        m2_heatmap = racket_sports_06_rl_return_utils.compute_m2_heatmap(
            env, _SERVE_XY, m2_n_samples, m2_seed
        )
        # Check outputs.
        cells = ["cell_id", "x_m", "y_m"]
        self.assert_equal(
            str((env_heatmap.columns.tolist(), env_heatmap[cells])),
            str((m2_heatmap.columns.tolist(), m2_heatmap[cells])),
        )
        p_env = env_heatmap["p_winner"].to_numpy()
        p_m2 = m2_heatmap["p_winner"].to_numpy()
        p_pool = (n_samples * p_env + m2_n_samples * p_m2) / (
            n_samples + m2_n_samples
        )
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_samples + 1 / m2_n_samples))
        z = np.abs(p_env - p_m2) / se
        _LOG.debug("z=%s", z)
        self.assertLess(z.max(), 4.0)


# #############################################################################
# Test_compute_m2_heatmap
# #############################################################################


class Test_compute_m2_heatmap(hunitest.TestCase):
    """
    Test `compute_m2_heatmap()`.
    """

    def test1(self) -> None:
        """
        Test that the map is the Milestone 2 map with the settings of the env.
        """
        # Prepare inputs.
        move_speed_mps = 3.0
        env = racket_sports_06_rl_return_utils.build_env(
            racket_params.PICKLEBALL, move_speed_mps=move_speed_mps
        )
        serve_xy = (-1.5, 4.4)
        n_samples = 1100
        seed = 2
        # Prepare outputs.
        # The pickleball defaults and grid, with the `v_p` of the env.
        std_xy = (0.5, 1.0)
        ball_speed_mps = 12.0
        reaction_time_s = 0.2
        n_x, n_y = (9, 10)
        expected = (
            racket_sports_01_shot_probability_utils.compute_winner_heatmap(
                serve_xy,
                std_xy,
                ball_speed_mps,
                reaction_time_s,
                move_speed_mps,
                racket_params.PICKLEBALL.court,
                n_x,
                n_y,
                n_samples,
                seed,
            )
        )
        # Run test.
        actual = racket_sports_06_rl_return_utils.compute_m2_heatmap(
            env, serve_xy, n_samples, seed
        )
        # Check outputs.
        pd.testing.assert_frame_equal(actual, expected)


# #############################################################################
# Test_get_eval_serves
# #############################################################################


class Test_get_eval_serves(hunitest.TestCase):
    """
    Test `get_eval_serves()`.
    """

    def test1(self) -> None:
        """
        Test that 3 x 3 serves in pickleball are 9 legal deuce serves, in a box
        capped at the baseline.
        """
        # Prepare inputs.
        env = racket_sports_06_rl_return_utils.build_env(
            racket_params.PICKLEBALL
        )
        n_x = 3
        n_y = 3
        court = racket_params.PICKLEBALL.court
        # Prepare outputs.
        expected = [""] * 9
        # Run test.
        serves = racket_sports_06_rl_return_utils.get_eval_serves(env, n_x, n_y)
        # Check outputs.
        actual = [
            racket_sports_01_shot_probability_utils.check_serve(
                "deuce", x1, y1, court
            )[1]
            for x1, y1 in serves
        ]
        self.assert_equal(str(actual), str(expected))


# #############################################################################
# Test_compare_heatmaps
# #############################################################################


class Test_compare_heatmaps(hunitest.TestCase):
    """
    Test `compare_heatmaps()`.
    """

    def test1(self) -> None:
        """
        Test the containment, the Spearman correlation, and the region sizes
        of 2 hand-made maps.
        """
        # Prepare inputs.
        env_heatmap, m2_heatmap = _make_heatmap_pair()
        # Prepare outputs.
        expected = {
            "env_best_in_m2_region": False,
            "m2_best_in_env_region": True,
            "spearman": 0.8,
            "n_env_region": 2,
            "n_m2_region": 1,
        }
        # Run test.
        actual = racket_sports_06_rl_return_utils.compare_heatmaps(
            env_heatmap, m2_heatmap
        )
        # Check outputs.
        # Spearman is 0.7999999999999999 in floating point.
        actual["spearman"] = round(actual["spearman"], 6)
        self.assert_equal(pprint.pformat(actual), pprint.pformat(expected))


# #############################################################################
# Test_score_target
# #############################################################################


class Test_score_target(hunitest.TestCase):
    """
    Test `score_target()`.
    """

    def test1(self) -> None:
        """
        Test the score of a target in the brute force region only.
        """
        # Prepare inputs.
        env_heatmap, m2_heatmap = _make_heatmap_pair()
        action = 1
        # Prepare outputs.
        # The regret is 0.875 - 0.75, and the random regret is 0.875 minus the
        # mean 0.5625, in points.
        expected = {
            "in_env_region": True,
            "in_m2_region": False,
            "regret_points": 12.5,
            "random_regret_points": 31.25,
        }
        # Run test.
        actual = racket_sports_06_rl_return_utils.score_target(
            env_heatmap, m2_heatmap, action
        )
        # Check outputs.
        self.assert_equal(pprint.pformat(actual), pprint.pformat(expected))


# #############################################################################
# Test_evaluate_policy
# #############################################################################


class Test_evaluate_policy(hunitest.TestCase):
    """
    Test `evaluate_policy()`.
    """

    def test1(self) -> None:
        """
        Test the greedy cell and the expected regret of a policy on hand-made
        maps, and that the policy sees the serve.
        """
        # Prepare inputs.
        env = racket_sports_06_rl_return_utils.build_env(racket_params.TENNIS)
        env_heatmap, m2_heatmap = _make_heatmap_pair()
        serves = [_SERVE_XY]
        observations = []

        def _get_probs(obs: np.ndarray) -> np.ndarray:
            observations.append(obs.tolist())
            probs: np.ndarray = np.array([0.25, 0.5, 0.25, 0.0])
            return probs

        # Prepare outputs.
        # The greedy cell is 1; the expected P(winner) on the M2 map is
        # 0.25 * 0.5 + 0.5 * 0.75 + 0.25 * 0.875 = 0.71875.
        expected = {
            "x1": -2.0,
            "y1": 3.0,
            "action": 1,
            "x_m": 1.0,
            "y_m": -3.0,
            "in_env_region": True,
            "in_m2_region": False,
            "regret_points": 12.5,
            "random_regret_points": 31.25,
            "top_prob": 0.5,
            "expected_p_winner": 0.71875,
            "expected_regret_points": 15.625,
        }
        # The last 2 values of an observation are the serve.
        expected_serves = [[-2.0, 3.0]]
        # Run test.
        actual = racket_sports_06_rl_return_utils.evaluate_policy(
            env, _get_probs, serves, [env_heatmap], [m2_heatmap]
        )
        # Check outputs.
        self.assert_equal(
            pprint.pformat(actual.to_dict("records")), pprint.pformat([expected])
        )
        actual_serves = [obs[4:] for obs in observations]
        self.assert_equal(str(actual_serves), str(expected_serves))


# #############################################################################
# Test_summarize_pass_rule
# #############################################################################


class Test_summarize_pass_rule(hunitest.TestCase):
    """
    Test `summarize_pass_rule()`.
    """

    def test1(self) -> None:
        """
        Test a run that passes, one that passes with the 1-point tolerance
        only, and one that fails both.
        """
        # Prepare inputs.
        results = pd.DataFrame(
            {
                "sport": ["Tennis"] * 5,
                "seed": [3, 3, 4, 4, 5],
                "in_m2_region": [True, True, True, False, False],
                "regret_points": [0.0, 0.02, 0.0, 0.62, 2.72],
                "expected_regret_points": [0.2, 0.25, 0.3, 0.5, 1.5],
            }
        )
        # Prepare outputs.
        expected = [
            {
                "sport": "Tennis",
                "seed": 3,
                "n_serves": 2,
                "n_in_m2_region": 2,
                "max_regret_points": 0.02,
                "max_expected_regret_points": 0.25,
                "strict_pass": True,
                "pass_1_point": True,
            },
            {
                "sport": "Tennis",
                "seed": 4,
                "n_serves": 2,
                "n_in_m2_region": 1,
                "max_regret_points": 0.62,
                "max_expected_regret_points": 0.5,
                "strict_pass": False,
                "pass_1_point": True,
            },
            {
                "sport": "Tennis",
                "seed": 5,
                "n_serves": 1,
                "n_in_m2_region": 0,
                "max_regret_points": 2.72,
                "max_expected_regret_points": 1.5,
                "strict_pass": False,
                "pass_1_point": False,
            },
        ]
        # Run test.
        actual = racket_sports_06_rl_return_utils.summarize_pass_rule(results)
        # Check outputs.
        self.assert_equal(
            pprint.pformat(actual.to_dict("records")), pprint.pformat(expected)
        )


# #############################################################################
# Test_plot_heatmaps
# #############################################################################


class Test_plot_heatmaps(hunitest.TestCase):
    """
    Test `plot_heatmaps()`.
    """

    def test1(self) -> None:
        """
        Test one map and one colorbar per panel, cropped to player 1's half,
        with the target in the legend.
        """
        # Prepare inputs.
        env = racket_sports_06_rl_return_utils.build_env(racket_params.TENNIS)
        n_samples = 1100
        seed = 2
        m2_heatmap = racket_sports_06_rl_return_utils.compute_m2_heatmap(
            env, _SERVE_XY, n_samples, seed
        )
        heatmaps = {"Brute force": m2_heatmap, "Milestone 2": m2_heatmap}
        targets_xy = ((0.0, -5.0),)
        # Prepare outputs.
        expected = str((4, ["Brute force", "Milestone 2"], 1.0, True))
        # Run test.
        fig = racket_sports_06_rl_return_utils.plot_heatmaps(
            heatmaps,
            racket_params.TENNIS.court,
            _SERVE_XY,
            targets_xy=targets_xy,
        )
        # Check outputs.
        # The labels that the legend of the first panel lists.
        _, legend = fig.axes[0].get_legend_handles_labels()
        actual = str(
            (
                len(fig.axes),
                [ax.get_title() for ax in fig.axes[:2]],
                float(fig.axes[0].get_ylim()[1]),
                "PPO target" in legend,
            )
        )
        plt.close("all")
        self.assert_equal(actual, expected)
