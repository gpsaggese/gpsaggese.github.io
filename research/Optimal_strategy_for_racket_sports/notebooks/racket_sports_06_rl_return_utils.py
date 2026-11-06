"""
Utilities for the `racket_sports_06_rl_return` notebook.

A one-shot RL env for the return of serve (#713), whose step is one call of the
Milestone 2 function `compute_winner_probability()` with 1 sample, and the
functions that score every target through the env (brute force) and compare it
with the Milestone 2 map.

Import as:

import research.Optimal_strategy_for_racket_sports.notebooks.racket_sports_06_rl_return_utils as rosfrsnrs0rru
"""

import dataclasses
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebo
import helpers.hprint as hprint
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

import research.Optimal_strategy_for_racket_sports.notebooks.racket_sports_01_shot_probability_utils as rosfrsnrs0spu
from research.Optimal_strategy_for_racket_sports import (
    racket_params,
    racket_scoring,
)

_LOG = logging.getLogger(__name__)


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utils logger.

    :param notebook_log: logger owned by the notebook
    """
    _LOG.debug(hprint.to_str("notebook_log"))
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# Defaults
# #############################################################################


# Grid step per sport in meters: the nearest whole number of cells gives the
# Milestone 2 grids of `HEATMAP_GRID_SIZE`, 10 x 15 cells in tennis and 9 x 10
# in pickleball.
GRID_STEP_M = {"Tennis": 0.8, "Pickleball": 0.67}

# Default spread of the return in meters, the slider default of Cells 1.1 and
# 2.1 of notebook 01.
_STD_DEFAULT_M = (0.5, 1.0)


# #############################################################################
# ReturnEnv
# #############################################################################


class ReturnEnv(gym.Env):
    """
    One point with one shot: player 2 aims the return of a serve, and gets a
    reward of 1 for a winner, else 0.

    - Observation: 6 floats in meters, player 1 `(x, y)`, player 2 `(x, y)`,
      and the serve landing `(x1, y1)`
      - In v0 player 1 stays at `get_player1_xy()` and player 2 hits from the
        serve landing, as in Milestone 2, so only the serve varies
    - Action: a cell of `racket_scoring.make_target_grid()` over player 1's
      half; the return aims at its center
    - Transition: `compute_winner_probability()` with 1 sample, so a winner
      is the Milestone 2 event, in and not reached
    - Episode: 1 step, which ends as terminated; the serve is the context, so
      the env is a contextual bandit
    """

    def __init__(
        self,
        sport: racket_params.SportParams,
        grid_step_m: float,
        ball_speed_mps: float,
        *,
        std_xy: Tuple[float, float] = _STD_DEFAULT_M,
        reaction_time_s: float = racket_params.DEFAULT_PLAYER.reaction_time_s,
        move_speed_mps: float = racket_params.DEFAULT_PLAYER.move_speed_mps,
    ) -> None:
        """
        Build the grid of targets, the serve box, and the 2 spaces.

        The model parameters are checked by the Milestone 2 functions at the
        first `step()`.

        :param sport: sport, e.g., `racket_params.TENNIS`
        :param grid_step_m: wanted cell size of the grid of targets in meters,
            e.g., `GRID_STEP_M["Tennis"]`
        :param ball_speed_mps: speed of the return in m/s, e.g.,
            `WINNER_BALL_SPEED_MPS["Tennis"]`
        :param std_xy: standard deviations `(std_x2, std_y2)` of the return in
            meters
        :param reaction_time_s: reaction time `t_r` of player 1 in seconds
        :param move_speed_mps: running speed `v_p` of player 1 in m/s
        """
        _LOG.debug(
            hprint.to_str(
                "sport.name grid_step_m ball_speed_mps std_xy reaction_time_s"
                " move_speed_mps"
            )
        )
        super().__init__()
        self.sport = sport
        court = sport.court
        # The grid of targets over player 1's half, as in Milestone 2. Each side
        # gets the whole number of cells closest to its length over the step,
        # so the cells tile the half, e.g., 8.23 m / 0.8 m = 10.3 gives 10
        # cells of 0.823 m; a step that gives 0 cells fails in
        # `make_target_grid()`.
        half = rosfrsnrs0spu.get_striker_half_region(court)
        n_x = round((half.x_max - half.x_min) / grid_step_m)
        n_y = round((half.y_max - half.y_min) / grid_step_m)
        self.grid_size = (n_x, n_y)
        self.targets = racket_scoring.make_target_grid(half, n_x, n_y)
        # The centers as an array, so that `step()` does not index a DataFrame.
        self._targets_xy = self.targets[["x_m", "y_m"]].to_numpy()
        # The deuce box, capped at the baseline as `check_serve()` does: the
        # pickleball service line is 5 mm past the baseline.
        box = racket_params.get_service_box_region(court, "deuce")
        self.serve_box = dataclasses.replace(
            box, y_max=min(box.y_max, court.length_m / 2)
        )
        self.player1_xy = rosfrsnrs0spu.get_player1_xy(court)
        self.std_xy = std_xy
        self.ball_speed_mps = ball_speed_mps
        self.reaction_time_s = reaction_time_s
        self.move_speed_mps = move_speed_mps
        # Player 1, player 2, and the serve, each bounded by the court.
        low = [-court.width_m / 2, -court.length_m / 2] * 3
        high = [court.width_m / 2, court.length_m / 2] * 3
        self.observation_space = gym.spaces.Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(len(self.targets))
        # `reset()` sets the serve; `compute_winner_probability()` rejects this
        # `nan` serve in a step before any reset.
        self._serve_xy = (np.nan, np.nan)

    def _get_obs(self) -> np.ndarray:
        """
        Build the observation: player 1, player 2, and the serve landing.

        :return: 6 floats, e.g., `[0.0, -11.885, -2.0, 3.0, -2.0, 3.0]` in
            tennis for a serve to `(-2.0, 3.0)`
        """
        # In v0 player 2 hits from where the serve lands.
        player2_xy = self._serve_xy
        obs: np.ndarray = np.array(
            [*self.player1_xy, *player2_xy, *self._serve_xy], dtype=np.float32
        )
        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Start a point: draw the serve, or take it from `options`.

        :param seed: seed of `self.np_random`, the only source of randomness of
            the env; `None` keeps the current generator, as in `gym.Env`
        :param options: `{"serve_xy": (x1, y1)}` fixes the serve, which must
            be a legal deuce serve, and draws nothing; without it the serve is
            uniform in the deuce box
        :return: the observation, and an empty info dict
        """
        # Lazy formatting, since this runs once per point.
        _LOG.debug("seed=%s options=%s", seed, options)
        super().reset(seed=seed)
        options = options or {}
        hdbg.dassert_is_subset(options.keys(), {"serve_xy"}, "Unknown options")
        if "serve_xy" in options:
            x1, y1 = (float(value) for value in options["serve_xy"])
            _, fault = rosfrsnrs0spu.check_serve(
                "deuce", x1, y1, self.sport.court
            )
            hdbg.dassert_eq(fault, "", "serve_xy=%s is a fault", (x1, y1))
        else:
            # A uniform draw lands on an edge of the box (the net, the kitchen
            # line) with probability 0.
            box = self.serve_box
            x1 = float(self.np_random.uniform(box.x_min, box.x_max))
            y1 = float(self.np_random.uniform(box.y_min, box.y_max))
        self._serve_xy = (x1, y1)
        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Play the return: aim at the center of cell `action`.

        :param action: index of the target cell, a row of `self.targets`
        :return: the observation, the reward (1.0 for a winner, else 0.0),
            `terminated=True`, `truncated=False`, and the info with `is_in`,
            `is_reached`, and `target_xy`
        """
        hdbg.dassert(
            self.action_space.contains(action), "Invalid action=%s", action
        )
        x2, y2 = self._targets_xy[int(action)]
        target_xy = (float(x2), float(y2))
        # One return: the Milestone 2 winner event on 1 sample, drawn from
        # `self.np_random`. It uses its own `get_player1_xy()`, the same point
        # as the observation in v0.
        result = rosfrsnrs0spu.compute_winner_probability(
            self._serve_xy,
            target_xy,
            self.std_xy,
            self.ball_speed_mps,
            self.reaction_time_s,
            self.move_speed_mps,
            self.sport.court,
            1,
            self.np_random,
        )
        reward = result["p_winner"]
        # With 1 sample, each fraction is 0 or 1.
        info = {
            "is_in": result["p_in"] == 1.0,
            "is_reached": result["p_reached"] == 1.0,
            "target_xy": target_xy,
        }
        _LOG.debug("action=%s reward=%s info=%s", action, reward, info)
        # One shot ends the point. `check_env()` asserts that `truncated` is
        # the bool `False`.
        terminated = True
        truncated = False
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info


# #############################################################################
# Brute force through the env
# #############################################################################


def compute_env_heatmap(
    env: ReturnEnv, serve_xy: Tuple[float, float], n_samples: int, seed: int
) -> pd.DataFrame:
    """
    Estimate the probability of a winner of every target by playing the env.

    Each target plays `n_samples` points from `serve_xy`, with the reset seeds
    `seed`, `seed + 1`, ..., the same for every target: common random numbers,
    as in `compute_winner_heatmap()`.

    :param env: env to play
    :param serve_xy: serve landing `(x1, y1)`, a legal deuce serve
    :param n_samples: points per target, e.g., 1000
    :param seed: first reset seed, e.g., 1000
    :return: one row per target, with the columns of `env.targets` and
        `p_winner`, `p_winner_se`, `p_in`, `p_reached`, the columns of
        `compute_winner_heatmap()`
    """
    _LOG.debug(hprint.to_str("serve_xy n_samples seed"))
    # 0 samples would give a map of `nan` with only a warning.
    hdbg.dassert_lt(0, n_samples, "n_samples must be positive")
    # One action per row of `env.targets`.
    n_targets = len(env.targets)
    rewards: np.ndarray = np.zeros((n_targets, n_samples))
    is_in: np.ndarray = np.zeros((n_targets, n_samples), dtype=bool)
    is_reached: np.ndarray = np.zeros((n_targets, n_samples), dtype=bool)
    for action in tqdm(range(n_targets), desc="targets"):
        for k in range(n_samples):
            env.reset(seed=seed + k, options={"serve_xy": serve_xy})
            _, reward, _, _, info = env.step(action)
            rewards[action, k] = reward
            is_in[action, k] = info["is_in"]
            is_reached[action, k] = info["is_reached"]
    # Same columns as `compute_winner_heatmap()`, in the grid order.
    heatmap: pd.DataFrame = env.targets.copy()
    p_winner = rewards.mean(axis=1)
    heatmap["p_winner"] = p_winner
    heatmap["p_winner_se"] = np.sqrt(p_winner * (1 - p_winner) / n_samples)
    heatmap["p_in"] = is_in.mean(axis=1)
    heatmap["p_reached"] = is_reached.mean(axis=1)
    _LOG.debug("return: %d targets", len(heatmap))
    return heatmap


def compute_m2_heatmap(
    env: ReturnEnv, serve_xy: Tuple[float, float], n_samples: int, seed: int
) -> pd.DataFrame:
    """
    Compute the Milestone 2 heatmap with the settings and the grid of the env.

    It calls `compute_winner_heatmap()` afresh, so a comparison with
    `compute_env_heatmap()` reads no saved Milestone 2 result.

    :param env: env whose settings and grid to use
    :param serve_xy: serve landing `(x1, y1)`
    :param n_samples: Monte Carlo samples per target, e.g., 5000
    :param seed: seed of the generator, the same for every target, e.g., 2
    :return: the heatmap of `compute_winner_heatmap()`, one row per target
        of `env.targets`, in the same order
    """
    _LOG.debug(hprint.to_str("serve_xy n_samples seed"))
    n_x, n_y = env.grid_size
    heatmap: pd.DataFrame = rosfrsnrs0spu.compute_winner_heatmap(
        serve_xy,
        env.std_xy,
        env.ball_speed_mps,
        env.reaction_time_s,
        env.move_speed_mps,
        env.sport.court,
        n_x,
        n_y,
        n_samples,
        seed,
    )
    return heatmap


# #############################################################################
# Evaluation
# #############################################################################


def get_eval_serves(
    env: ReturnEnv, n_x: int, n_y: int
) -> List[Tuple[float, float]]:
    """
    Get the serves to evaluate on: the cell centers of a grid over the deuce
    box.

    :param env: env whose serve box to use
    :param n_x: number of serves across the box, e.g., 3
    :param n_y: number of serves along the box, e.g., 3
    :return: serves `(x1, y1)` in the grid order of `make_target_grid()`, e.g.,
        `[(-2.0575, 3.2)]`, the middle of the box in tennis, for 1 x 1
    """
    _LOG.debug(hprint.to_str("n_x n_y"))
    grid = racket_scoring.make_target_grid(env.serve_box, n_x, n_y)
    serves = [(float(x), float(y)) for x, y in zip(grid["x_m"], grid["y_m"])]
    _LOG.debug("return: %d serves, first=%s", len(serves), serves[0])
    return serves


def get_greedy_targets(
    env: ReturnEnv,
    policy: Callable[[np.ndarray], int],
    serves: List[Tuple[float, float]],
) -> pd.DataFrame:
    """
    Get the target that `policy` picks for each serve.

    :param env: env that builds the observation of each serve
    :param policy: map from an observation to an action, e.g., the greedy
        action of a trained agent
    :param serves: serves `(x1, y1)`, e.g., from `get_eval_serves()`
    :return: one row per serve, with the columns `x1`, `y1` (the serve),
        `action`, `x_m`, `y_m` (the center of the target cell)
    """
    _LOG.debug(hprint.to_str("serves"))
    rows = []
    for serve_xy in serves:
        obs, _ = env.reset(options={"serve_xy": serve_xy})
        action = int(policy(obs))
        rows.append(
            {
                "x1": serve_xy[0],
                "y1": serve_xy[1],
                "action": action,
                "x_m": float(env.targets["x_m"].iloc[action]),
                "y_m": float(env.targets["y_m"].iloc[action]),
            }
        )
    targets = pd.DataFrame(rows)
    _LOG.debug("return: %d serves", len(targets))
    return targets


def compare_heatmaps(
    env_heatmap: pd.DataFrame, m2_heatmap: pd.DataFrame
) -> Dict[str, Any]:
    """
    Compare the brute force map of the env with the Milestone 2 map.

    The best region of a map is `find_best_region()`, the cells within 2
    standard errors of its highest `P(winner)`:
    - Containment: the highest cell of each map is in the best region of the
      other; it holds with ties at the top and with different sample counts
    - Spearman: the rank correlation of the 2 maps over every cell

    :param env_heatmap: from `compute_env_heatmap()`
    :param m2_heatmap: from `compute_m2_heatmap()`, on the same grid
    :return: `env_best_in_m2_region`, `m2_best_in_env_region`, `spearman`,
        `n_env_region`, `n_m2_region` (the sizes of the 2 best regions)
    """
    _LOG.debug(hprint.to_str("env_heatmap.shape m2_heatmap.shape"))
    # Same cells in the same order, since `Series.corr()` aligns on the index
    # and would drop the cells of one map silently.
    hdbg.dassert(
        env_heatmap[["x_m", "y_m"]].equals(m2_heatmap[["x_m", "y_m"]]),
        "The 2 maps must have the same cells, in the same order",
    )
    env_p = env_heatmap["p_winner"].to_numpy()
    m2_p = m2_heatmap["p_winner"].to_numpy()
    env_region = rosfrsnrs0spu.find_best_region(env_heatmap)
    m2_region = rosfrsnrs0spu.find_best_region(m2_heatmap)
    # The highest cell of each map; `argmax()` keeps the first of a tie.
    env_best = int(np.argmax(env_p))
    m2_best = int(np.argmax(m2_p))
    # Spearman is the correlation of the ranks, with ties at their mean rank.
    spearman = pd.Series(env_p).corr(pd.Series(m2_p), method="spearman")
    result = {
        "env_best_in_m2_region": bool(m2_region[env_best]),
        "m2_best_in_env_region": bool(env_region[m2_best]),
        "spearman": float(spearman),
        "n_env_region": int(env_region.sum()),
        "n_m2_region": int(m2_region.sum()),
    }
    _LOG.debug("return=%s", result)
    return result


def score_target(
    env_heatmap: pd.DataFrame, m2_heatmap: pd.DataFrame, action: int
) -> Dict[str, Any]:
    """
    Score a target picked for a serve, e.g., by a trained agent.

    - The regret is the highest `P(winner)` of the Milestone 2 map minus its
      value at the target, in points, since that map has the most samples
    - The random regret is the same for a uniformly random target: the
      highest value minus the mean over the cells

    :param env_heatmap: from `compute_env_heatmap()`
    :param m2_heatmap: from `compute_m2_heatmap()`, on the same grid
    :param action: index of the target cell
    :return: `in_env_region`, `in_m2_region`, `regret_points`,
        `random_regret_points`
    """
    _LOG.debug(hprint.to_str("action"))
    # Same cells in the same order, so `action` is the same cell in both.
    hdbg.dassert(
        env_heatmap[["x_m", "y_m"]].equals(m2_heatmap[["x_m", "y_m"]]),
        "The 2 maps must have the same cells, in the same order",
    )
    # A negative action would index the last cells.
    hdbg.dassert_in(action, range(len(m2_heatmap)), "action must be a cell")
    env_region = rosfrsnrs0spu.find_best_region(env_heatmap)
    m2_region = rosfrsnrs0spu.find_best_region(m2_heatmap)
    m2_p = m2_heatmap["p_winner"].to_numpy()
    result = {
        "in_env_region": bool(env_region[action]),
        "in_m2_region": bool(m2_region[action]),
        "regret_points": float(100 * (m2_p.max() - m2_p[action])),
        "random_regret_points": float(100 * (m2_p.max() - m2_p.mean())),
    }
    _LOG.debug("return=%s", result)
    return result
