"""
Utilities for the `racket_shot_probability` notebook.

Player 1 serves to `(x1, y1)`, player 2 moves there and aims a return at
`(x2, y2)` on player 1's half of the court. The return lands at a 2D Gaussian
point around `(x2, y2)` with independent `x` and `y` errors, so the
probability that the return is in has a closed form.

For Milestone 2 (#624), player 1 then stands at the center of their baseline
and tries to reach each landing point before the ball lands, after a reaction
time. The return is a winner if it lands in and player 1 does not reach it,
estimated by Monte Carlo as one event.

Import as:

import research.Optimal_strategy_for_racket_sports.notebooks.racket_shot_probability_utils as rosfrsnrspu
"""

import dataclasses
import html
import logging
from typing import Any, Dict, List, Optional, Tuple

import ipywidgets
import matplotlib.collections as mcollections
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from IPython.display import clear_output, display
from matplotlib.axes import Axes

import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebo
import helpers.hprint as hprint
import helpers.htutorial as htutori
from research.Optimal_strategy_for_racket_sports import racket_params
from research.Optimal_strategy_for_racket_sports import racket_scoring
from research.Optimal_strategy_for_racket_sports import racket_strategy_utils

_LOG = logging.getLogger(__name__)


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utils logger.

    :param notebook_log: logger owned by the notebook
    """
    _LOG.debug(hprint.to_str("notebook_log"))
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# In-court probability
# #############################################################################


def get_striker_half_region(
    court: racket_params.CourtGeometry,
) -> racket_params.CourtRegion:
    """
    Get the region of player 1's half of the court (y < 0).

    Mirror of `racket_params.get_half_court_region()`, which returns player 2's
    half (y > 0). Player 2's return must land in this region to be in.

    :param court: court geometry, e.g., `racket_params.TENNIS.court`
    :return: rectangle with `x` in `[-W/2, W/2]` and `y` in `[-L/2, 0]`, where
        `W` is the court width and `L` is the court length
    """
    _LOG.debug(hprint.to_str("court"))
    # `CourtGeometry` rejects sizes that are not positive, but accepts `inf`,
    # which would give an infinite half court.
    hdbg.dassert(
        np.isfinite(court.width_m) and np.isfinite(court.length_m),
        "Court width_m=%s and length_m=%s must be finite",
        court.width_m,
        court.length_m,
    )
    hdbg.dassert_lt(0, court.width_m, "Court width must be positive")
    hdbg.dassert_lt(0, court.length_m, "Court length must be positive")
    # Mirror of `racket_params.get_half_court_region()` across the net (y = 0).
    region = racket_params.CourtRegion(
        x_min=-court.width_m / 2,
        x_max=court.width_m / 2,
        y_min=-court.length_m / 2,
        y_max=0,
    )
    _LOG.debug("return=%s", region)
    return region


def _dassert_target_and_std(
    target_xy: Tuple[float, float], std_xy: Tuple[float, float]
) -> None:
    """
    Check that the target and the stds are finite, and the stds positive.

    `scipy` and `numpy` do not raise on these values, they return a wrong
    number instead:
    - A `nan` target gives a `nan` probability and `nan` samples
    - An `inf` target or std gives a probability of 0 and `inf` samples
    - A std of 0 gives NaN in `scipy` and the target every time in `numpy`

    :param target_xy: aim point `(x2, y2)` in meters
    :param std_xy: standard deviations `(std_x2, std_y2)` in meters
    """
    _LOG.debug(hprint.to_str("target_xy std_xy"))
    hdbg.dassert(
        np.all(np.isfinite(target_xy)), "target_xy=%s must be finite", target_xy
    )
    hdbg.dassert(np.all(np.isfinite(std_xy)), "std_xy=%s must be finite", std_xy)
    hdbg.dassert_lt(0, std_xy[0], "std_x2 must be positive")
    hdbg.dassert_lt(0, std_xy[1], "std_y2 must be positive")


def compute_p_in(
    target_xy: Tuple[float, float],
    std_xy: Tuple[float, float],
    region: racket_params.CourtRegion,
) -> float:
    """
    Compute the probability that a Gaussian shot lands inside `region`.

    The landing point is `X ~ N(x2, std_x2^2)` and `Y ~ N(y2, std_y2^2)`, with
    `X` and `Y` independent. Since `region` is an axis-aligned rectangle, the
    probability is the `x` part times the `y` part:
    ```
    P_in = P(x_min <= X <= x_max) * P(y_min <= Y <= y_max)
    ```
    Each part is a difference of two normal CDFs, e.g.,
    `P(x_min <= X <= x_max) = CDF_X(x_max) - CDF_X(x_min)`.

    :param target_xy: aim point `(x2, y2)` in meters, e.g., `(0.0, -5.94)`
    :param std_xy: standard deviations `(std_x2, std_y2)` in meters, e.g.,
        `(0.5, 1.0)`
    :param region: rectangle where the shot counts as in
    :return: probability in `[0, 1]`, e.g., about 0.25 when aiming at a corner
        of `region` with a small std
    """
    _LOG.debug(hprint.to_str("target_xy std_xy region"))
    _dassert_target_and_std(target_xy, std_xy)
    x2, y2 = target_xy
    std_x2, std_y2 = std_xy
    # Probability that `X` lands between `x_min` and `x_max`.
    cdf_x_max = scipy.stats.norm.cdf(region.x_max, loc=x2, scale=std_x2)
    cdf_x_min = scipy.stats.norm.cdf(region.x_min, loc=x2, scale=std_x2)
    p_x = cdf_x_max - cdf_x_min
    # Probability that `Y` lands between `y_min` and `y_max`.
    cdf_y_max = scipy.stats.norm.cdf(region.y_max, loc=y2, scale=std_y2)
    cdf_y_min = scipy.stats.norm.cdf(region.y_min, loc=y2, scale=std_y2)
    p_y = cdf_y_max - cdf_y_min
    # Multiply the two parts since `X` and `Y` are independent.
    p_in = float(p_x * p_y)
    _LOG.debug("return=%s", p_in)
    return p_in


# #############################################################################
# Monte Carlo sampling
# #############################################################################


def sample_landings(
    target_xy: Tuple[float, float],
    std_xy: Tuple[float, float],
    n_samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw `n_samples` Gaussian landing points around `target_xy`.

    Same model as `compute_p_in()`: `X ~ N(x2, std_x2^2)` and
    `Y ~ N(y2, std_y2^2)`, drawn independently. The samples give a Monte Carlo
    check of `compute_p_in()` and the scatter cloud in the notebook.

    :param target_xy: aim point `(x2, y2)` in meters
    :param std_xy: standard deviations `(std_x2, std_y2)` in meters
    :param n_samples: number of landing points to draw, a positive integer
    :param rng: random generator to draw from, passed in explicitly so that the
        same seed gives the same points, e.g., `np.random.default_rng(1)`
    :return: arrays `(x, y)` of landing coordinates, each of shape
        `(n_samples,)`
    """
    _LOG.debug(hprint.to_str("target_xy std_xy n_samples"))
    _dassert_target_and_std(target_xy, std_xy)
    # `numpy` silently returns empty arrays for 0 samples.
    hdbg.dassert_isinstance(
        n_samples, (int, np.integer), "n_samples must be an integer"
    )
    hdbg.dassert_lt(0, n_samples, "n_samples must be positive")
    x2, y2 = target_xy
    std_x2, std_y2 = std_xy
    # Draw each coordinate on its own since `X` and `Y` are independent.
    x = rng.normal(x2, std_x2, n_samples)
    y = rng.normal(y2, std_y2, n_samples)
    _LOG.debug("return: %d landing points", len(x))
    return x, y


# #############################################################################
# Player 1's reach
# #############################################################################


def get_player1_xy(court: racket_params.CourtGeometry) -> Tuple[float, float]:
    """
    Get where player 1 stands when the return comes back.

    Player 1 serves and then stays at the center of their baseline,
    `(0, -L/2)`, as agreed with GP in #624; where a player should stand is
    left to a later model. This is not the serve stance of `_get_server_xy()`,
    which is only where the serve starts.

    :param court: court geometry, e.g., `racket_params.TENNIS.court`
    :return: position `(0.0, -L/2)` in meters, where `L` is the court length,
        e.g., `(0.0, -11.885)` for tennis
    """
    _LOG.debug(hprint.to_str("court"))
    # `CourtGeometry` rejects a length that is not positive, but accepts `inf`,
    # which would put player 1 infinitely far away.
    hdbg.dassert(
        np.isfinite(court.length_m),
        "Court length_m=%s must be finite",
        court.length_m,
    )
    hdbg.dassert_lt(0, court.length_m, "Court length must be positive")
    player1_xy = (0.0, -court.length_m / 2)
    _LOG.debug("return=%s", player1_xy)
    return player1_xy


def _dassert_point(xy: Tuple[float, float], name: str) -> None:
    """
    Check that `xy` is a point `(x, y)` with finite coordinates.

    :param xy: point to check, e.g., `(-2.0, 4.8)`
    :param name: name of the point in the error message, e.g., "hitter_xy"
    """
    _LOG.debug(hprint.to_str("xy name"))
    hdbg.dassert_eq(len(xy), 2, "%s=%s must be a point (x, y)", name, xy)
    hdbg.dassert(np.all(np.isfinite(xy)), "%s=%s must be finite", name, xy)


def _dassert_landing_points(x: np.ndarray, y: np.ndarray) -> None:
    """
    Check that the landing coordinates are finite arrays of the same shape.

    `numpy` does not raise on these inputs, it returns a wrong answer instead:
    - Arrays of different shapes can broadcast, pairing each `x` with the
      wrong `y`
    - A `nan` coordinate gives a `nan` distance, which counts as not reached

    :param x: landing `x` of each sample, any shape
    :param y: landing `y` of each sample, same shape as `x`
    """
    hdbg.dassert_isinstance(x, np.ndarray, "x must be a numpy array")
    hdbg.dassert_isinstance(y, np.ndarray, "y must be a numpy array")
    # Log the shapes only once they are known to exist.
    _LOG.debug(hprint.to_str("x.shape y.shape"))
    hdbg.dassert_eq(x.shape, y.shape, "x and y must have the same shape")
    hdbg.dassert_lt(0, x.size, "There must be at least one landing point")
    hdbg.dassert(np.all(np.isfinite(x)), "Every landing x must be finite")
    hdbg.dassert(np.all(np.isfinite(y)), "Every landing y must be finite")


def compute_flight_time(
    hitter_xy: Tuple[float, float],
    x: np.ndarray,
    y: np.ndarray,
    ball_speed_mps: float,
) -> np.ndarray:
    """
    Compute the flight time of the return to each landing point.

    The ball goes in a straight line, at a constant speed, from player 2 at
    `hitter_xy = (x1, y1)` to each landing point `(X, Y)`:
    ```
    T = distance((x1, y1), (X, Y)) / ball_speed
    ```
    The time uses the actual landing point, not the aim point `(x2, y2)`:
    the aim point is a latent variable, while the landing point is where the
    ball goes (GP on #690). The model is 2D: no arc, no bounce, no slowdown.

    :param hitter_xy: where player 2 hits the return, the serve landing point
        `(x1, y1)` in meters, e.g., `(-2.0, 4.8)`
    :param x: landing `x` of each sample in meters, e.g., from
        `sample_landings()`; any shape
    :param y: landing `y` of each sample in meters, same shape as `x`
    :param ball_speed_mps: speed of the return in m/s, positive, e.g., 25.0
    :return: flight time of each sample in seconds, same shape as `x`, e.g.,
        `[0.2, 0.8]` for landing points 5 m and 20 m away at 25 m/s
    """
    # `x` and `y` are logged by `_dassert_landing_points()`, once checked.
    _LOG.debug(hprint.to_str("hitter_xy ball_speed_mps"))
    _dassert_point(hitter_xy, "hitter_xy")
    _dassert_landing_points(x, y)
    hdbg.dassert(
        np.isfinite(ball_speed_mps),
        "ball_speed_mps=%s must be finite",
        ball_speed_mps,
    )
    # A speed of 0 would give infinite times, with only a `numpy` warning.
    hdbg.dassert_lt(0, ball_speed_mps, "ball_speed_mps must be positive")
    # Straight-line distance from player 2 to each landing point.
    x1, y1 = hitter_xy
    dist_m = np.hypot(x - x1, y - y1)
    flight_time_s: np.ndarray = dist_m / ball_speed_mps
    _LOG.debug(
        "return: %d flight times, first=%s",
        flight_time_s.size,
        flight_time_s.flat[0],
    )
    return flight_time_s


def compute_is_reached(
    player_xy: Tuple[float, float],
    x: np.ndarray,
    y: np.ndarray,
    flight_time_s: np.ndarray,
    reaction_time_s: float,
    move_speed_mps: float,
) -> np.ndarray:
    """
    Find the landing points that player 1 reaches before the ball lands.

    Player 1 at `player_xy` first reacts for `t_r` seconds, the time to decide
    where the ball will land, and then runs in a straight line at a constant
    speed `v_p` (GP on #624). They reach a landing point `(X, Y)` with flight
    time `T` if:
    ```
    distance(player_xy, (X, Y)) <= v_p * max(0, T - t_r)
    ```
    - `max(0, ...)`: a ball that lands before player 1 reacts (`T < t_r`) gives
      a reach radius of 0, not a negative one
    - `<=`: a point exactly on the edge of the reach counts as reached, as in
      `racket_scoring.compute_reachability()`
    - With a reach radius of 0 (`T <= t_r` or `v_p = 0`), only a ball landing
      exactly on player 1 counts as reached, which a Gaussian landing does
      with probability 0

    :param player_xy: where player 1 stands in meters, e.g., from
        `get_player1_xy()`
    :param x: landing `x` of each sample in meters; any shape
    :param y: landing `y` of each sample in meters, same shape as `x`
    :param flight_time_s: flight time of each sample in seconds, same shape
        as `x`, e.g., from `compute_flight_time()`
    :param reaction_time_s: reaction time `t_r` in seconds, e.g., 0.2
    :param move_speed_mps: running speed `v_p` in m/s, e.g., 1.5; 0 means that
        player 1 does not move
    :return: boolean array, same shape as `x`, True where player 1 reaches the
        landing point; e.g., `T = 0.8`, `t_r = 0.2`, `v_p = 1.5` give a reach of
        0.9 m, so a landing point 0.5 m away is True and one 1 m away is False
    """
    # `x` and `y` are logged by `_dassert_landing_points()`, once checked.
    _LOG.debug(hprint.to_str("player_xy reaction_time_s move_speed_mps"))
    _dassert_point(player_xy, "player_xy")
    _dassert_landing_points(x, y)
    # One finite, non-negative flight time per landing point.
    hdbg.dassert_isinstance(
        flight_time_s, np.ndarray, "flight_time_s must be a numpy array"
    )
    hdbg.dassert_eq(
        flight_time_s.shape,
        x.shape,
        "flight_time_s must have one time per landing point",
    )
    hdbg.dassert(
        np.all(np.isfinite(flight_time_s)), "Every flight time must be finite"
    )
    hdbg.dassert_lte(
        0, flight_time_s.min(), "Every flight time must be non-negative"
    )
    # A negative reaction time or speed would make player 1 faster than
    # possible, and a `nan` would count every point as not reached.
    hdbg.dassert(
        np.isfinite(reaction_time_s),
        "reaction_time_s=%s must be finite",
        reaction_time_s,
    )
    hdbg.dassert_lte(0, reaction_time_s, "reaction_time_s must be non-negative")
    hdbg.dassert(
        np.isfinite(move_speed_mps),
        "move_speed_mps=%s must be finite",
        move_speed_mps,
    )
    hdbg.dassert_lte(0, move_speed_mps, "move_speed_mps must be non-negative")
    # How far player 1 can run in the time left once they have reacted.
    run_time_s = np.maximum(0.0, flight_time_s - reaction_time_s)
    reach_m = move_speed_mps * run_time_s
    # Compare with the straight-line distance to each landing point.
    dist_m = np.hypot(x - player_xy[0], y - player_xy[1])
    is_reached: np.ndarray = dist_m <= reach_m
    _LOG.debug(
        "return: %d of %d landing points reached",
        is_reached.sum(),
        is_reached.size,
    )
    return is_reached


# #############################################################################
# Winner probability
# #############################################################################


def compute_winner_probability(
    serve_xy: Tuple[float, float],
    target_xy: Tuple[float, float],
    std_xy: Tuple[float, float],
    ball_speed_mps: float,
    reaction_time_s: float,
    move_speed_mps: float,
    court: racket_params.CourtGeometry,
    n_samples: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Estimate the probability that player 2's return is a winner.

    Player 2 hits from the serve landing point `serve_xy = (x1, y1)` and aims
    at `target_xy = (x2, y2)`; player 1 waits at `get_player1_xy()`. Each of
    the `n_samples` Gaussian landing points of `sample_landings()` is checked
    for 2 things, on the same sample:
    - In: it lands on player 1's half, `get_striker_half_region()`
    - Reached: player 1 gets there in time, `compute_is_reached()`, with the
      sample's own flight time from `compute_flight_time()`
    A sample is a winner if it is in and not reached, as one event (GP on
    #690): `P(winner)` is the fraction of winners, never a product of the
    probabilities of 2 events like `P(in) * P(not reached)`. `P(in)` and
    `P(reached)` come from the same samples, for display only.

    The serve is assumed legal: the caller (the widget) checks the serve rules
    with `_check_serve()` and computes no return for a fault. Only the side of
    the court of `serve_xy` is checked here.

    :param serve_xy: where the serve lands and player 2 hits from, `(x1, y1)`
        in meters, on player 2's half (`y1 > 0`), e.g., `(-2.1, 3.2)`
    :param target_xy: aim point of the return `(x2, y2)` in meters, e.g.,
        `(2.1, -10.4)`
    :param std_xy: standard deviations `(std_x2, std_y2)` of the return in
        meters, e.g., `(0.5, 1.0)`
    :param ball_speed_mps: speed of the return in m/s, e.g., 25.0 for tennis
    :param reaction_time_s: reaction time `t_r` of player 1 in seconds, e.g.,
        0.2
    :param move_speed_mps: running speed `v_p` of player 1 in m/s, e.g., 1.5
    :param court: court geometry, e.g., `racket_params.TENNIS.court`
    :param n_samples: number of Monte Carlo landing points, e.g., 10000
    :param rng: random generator, e.g., `np.random.default_rng(1)`; the same
        seed gives the same samples, so the same result
    :return: probabilities estimated from the same samples:
        - "p_winner": fraction of samples that land in and are not reached
        - "p_winner_se": standard error of "p_winner",
          `sqrt(p_winner * (1 - p_winner) / n_samples)`; it is 0 when every
          sample agrees, e.g., when player 1 reaches every sample
        - "p_in": fraction of samples that land in
        - "p_reached": fraction of samples that player 1 reaches, in or out
        e.g., `{"p_winner": 0.93, "p_winner_se": 0.0026, "p_in": 0.93,
        "p_reached": 0.0}`
    """
    _LOG.debug(
        hprint.to_str(
            "serve_xy target_xy std_xy ball_speed_mps reaction_time_s "
            "move_speed_mps n_samples"
        )
    )
    # Player 2 hits from their own half; the other serve rules are checked by
    # the caller.
    _dassert_point(serve_xy, "serve_xy")
    hdbg.dassert_lt(
        0, serve_xy[1], "The serve must land on player 2's half (y1 > 0)"
    )
    hdbg.dassert_isinstance(
        rng, np.random.Generator, "rng must be a numpy random Generator"
    )
    # Draw the landing points first, so the same seed gives the same points
    # as `sample_landings()` alone; it also checks the target, the stds, and
    # `n_samples`.
    x, y = sample_landings(target_xy, std_xy, n_samples, rng)
    # Event 1, per sample: the return lands in.
    region = get_striker_half_region(court)
    is_in = region.contains(x, y)
    # Event 2, per sample: player 1 reaches the landing point before the ball
    # lands. The reach rule lives only in `compute_is_reached()`.
    player1_xy = get_player1_xy(court)
    flight_time_s = compute_flight_time(serve_xy, x, y, ball_speed_mps)
    is_reached = compute_is_reached(
        player1_xy, x, y, flight_time_s, reaction_time_s, move_speed_mps
    )
    # A winner is one event on each sample: in AND not reached.
    is_winner = is_in & ~is_reached
    p_winner = float(is_winner.mean())
    # Standard error of a fraction of `n_samples` independent samples. Unlike
    # `P_in` in Cell 1.1 there is no closed form to plug in, so it uses the
    # estimate itself.
    p_winner_se = float(np.sqrt(p_winner * (1 - p_winner) / n_samples))
    result = {
        "p_winner": p_winner,
        "p_winner_se": p_winner_se,
        "p_in": float(is_in.mean()),
        "p_reached": float(is_reached.mean()),
    }
    _LOG.debug("return=%s", result)
    return result


# #############################################################################
# Winner probability heatmap
# #############################################################################


# Default grid `(n_x, n_y)` per sport and samples per target, chosen so one
# full map takes under 2 s (decision D24):
# - Each target costs about 8 to 10 ms almost whatever `n_samples`, mostly the
#   debug strings of the building blocks, so a map can hold about 150 targets:
#   cells of about 0.8 m (tennis, 150 targets) and 0.68 m (pickleball, 90)
# - More than 1,000 samples, since `racket_params.CourtRegion.contains()`
#   formats arrays of up to 1,000 values in a debug log even when logging is
#   off; 5,000 costs about the same as 2,000 and halves the noise of 1,000
HEATMAP_GRID_SIZE = {"Tennis": (10, 15), "Pickleball": (9, 10)}
HEATMAP_N_SAMPLES = 5000


def compute_winner_heatmap(
    serve_xy: Tuple[float, float],
    std_xy: Tuple[float, float],
    ball_speed_mps: float,
    reaction_time_s: float,
    move_speed_mps: float,
    court: racket_params.CourtGeometry,
    n_x: int,
    n_y: int,
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    """
    Compute the probability of a winner for every target of a grid over player
    1's half.

    The targets are the cell centers of `racket_scoring.make_target_grid()`
    over `get_striker_half_region()`. Each target calls
    `compute_winner_probability()` with a new `np.random.default_rng(seed)`,
    the same `seed` for all: common random numbers, so every target uses the
    same standard normal draws, shifted to its own aim point. Two targets then
    differ only because of where they aim, not because of sampling noise, so
    the map is smooth and the best target is a fair comparison.

    :param serve_xy: where the serve lands and player 2 hits from, `(x1, y1)`
        in meters, on player 2's half, e.g., `(-2.1, 3.2)`
    :param std_xy: standard deviations `(std_x2, std_y2)` of the return in
        meters, e.g., `(0.5, 1.0)`
    :param ball_speed_mps: speed of the return in m/s, e.g., 25.0 for tennis
    :param reaction_time_s: reaction time `t_r` of player 1 in seconds, e.g.,
        0.2
    :param move_speed_mps: running speed `v_p` of player 1 in m/s, e.g., 1.5
    :param court: court geometry, e.g., `racket_params.TENNIS.court`
    :param n_x: number of grid columns, across the court, e.g., 10
    :param n_y: number of grid rows, along the court, e.g., 15
    :param n_samples: Monte Carlo samples per target, e.g., 5000; more than
        1,000 keeps each target fast (see `HEATMAP_N_SAMPLES`)
    :param seed: seed of the generator, the same for every target, e.g., 1
    :return: one row per target, in the grid order of `make_target_grid()`
        (`y` outer from the baseline, `x` inner), with the columns `cell_id`,
        `x_m`, `y_m` (the target), `x_min`, `x_max`, `y_min`, `y_max` (the
        cell), and `p_winner`, `p_winner_se`, `p_in`, `p_reached`
    """
    _LOG.debug(hprint.to_str("serve_xy std_xy n_x n_y n_samples seed"))
    # `make_target_grid()` checks only that the sizes are positive, and
    # `default_rng()` also accepts arrays and `None` as a seed.
    hdbg.dassert_isinstance(n_x, (int, np.integer), "n_x must be an integer")
    hdbg.dassert_isinstance(n_y, (int, np.integer), "n_y must be an integer")
    hdbg.dassert_isinstance(seed, (int, np.integer), "seed must be an integer")
    hdbg.dassert_lte(0, seed, "seed must be non-negative")
    # One target per cell center of a uniform grid over player 1's half.
    region = get_striker_half_region(court)
    grid = racket_scoring.make_target_grid(region, n_x, n_y)
    # Common random numbers: a new generator with the same seed per target.
    rows = []
    for target_xy in zip(grid["x_m"], grid["y_m"]):
        rng = np.random.default_rng(seed)
        result = compute_winner_probability(
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
        rows.append(result)
    # Same row order as `grid`, so the columns line up target by target.
    heatmap = pd.concat([grid, pd.DataFrame(rows)], axis=1)
    _LOG.debug("return: %d targets", len(heatmap))
    return heatmap


# A target is in the best region if it is within this many standard errors
# of the highest `P(winner)` (decision D28).
_BEST_REGION_N_SE = 2.0


def find_best_region(heatmap: pd.DataFrame) -> np.ndarray:
    """
    Find the targets whose probability of a winner is within 2 standard errors
    of the highest one: the best region.

    Many targets can tie at the top (e.g., every sample a winner, so
    `P(winner) = 1.0`), so the best is a region rather than one cell. A target
    is in it if `p_max - p <= 2 * sqrt(se_max^2 + se^2)`, the standard error
    of the difference of 2 independent estimates, as in `find_best_target()`.
    With both standard errors 0 only an equal value is in.

    :param heatmap: targets with `x_m`, `y_m`, `p_winner`, `p_winner_se`,
        e.g., from `compute_winner_heatmap()`, at least 1 row
    :return: boolean array, one value per row of `heatmap` in its order, True
        for the targets of the best region, e.g., `[False, True, True]` for
        `P(winner)` of `[0.9, 0.95, 0.948]` with standard errors of about 0.002
    """
    hdbg.dassert_isinstance(
        heatmap, pd.DataFrame, "heatmap must be a pandas DataFrame"
    )
    # Log the shape only once it is known to exist.
    _LOG.debug(hprint.to_str("heatmap.shape"))
    for column in ("x_m", "y_m", "p_winner", "p_winner_se"):
        hdbg.dassert_in(column, heatmap.columns, "Missing heatmap column")
    hdbg.dassert_lte(1, len(heatmap), "There must be at least 1 target")
    p_winner = heatmap["p_winner"].to_numpy(dtype=float)
    p_winner_se = heatmap["p_winner_se"].to_numpy(dtype=float)
    # The highest `P(winner)`; `argmax()` keeps the first of a tie.
    i_max = int(np.argmax(p_winner))
    # Gap to the highest, against the standard error of that difference.
    gap = p_winner[i_max] - p_winner
    gap_se = np.hypot(p_winner_se[i_max], p_winner_se)
    in_region: np.ndarray = gap <= _BEST_REGION_N_SE * gap_se
    _LOG.debug("return: %d of %d targets", in_region.sum(), in_region.size)
    return in_region


def find_best_target(heatmap: pd.DataFrame) -> Dict[str, float]:
    """
    Find the target to mark as the best: the middle of the best region, and
    how far the second highest probability is from the highest one.

    - The best region is `find_best_region()`; the marked target is its cell
      closest to the region's centroid, a real target with its own
      `P(winner)`, since the centroid itself can fall outside a region that is
      not convex; a tie goes to the first in grid order
    - The gap is `(p_max - p_second) / sqrt(se_max^2 + se_second^2)` between
      the highest and the second highest `P(winner)`, the usual formula for 2
      independent estimates; with common random numbers the 2 estimates move
      together, so the real standard error of the gap is smaller, and this
      number understates how clearly the highest wins
    - If both standard errors are 0 (both probabilities 0 or 1), the gap is 0
      for equal probabilities, and `inf` otherwise

    :param heatmap: targets with `x_m`, `y_m`, `p_winner`, `p_winner_se`,
        e.g., from `compute_winner_heatmap()`, at least 2 rows
    :return:
        - `x_m`, `y_m`, `p_winner`, `p_winner_se`: the marked target
        - `n_region_cells`: the number of targets in the best region
        - `max_p_winner`, `max_p_winner_se`: the highest `P(winner)`
        - `second_p_winner`, `second_p_winner_se`: the second highest
        - `gap_in_se`: the gap between the 2 highest, in standard errors
        e.g., `{"x_m": 0.0, "y_m": -5.5, "p_winner": 1.0, "p_winner_se": 0.0,
        "n_region_cells": 35, ..., "gap_in_se": 0.0}`
    """
    # It also checks the type and the columns of `heatmap`.
    in_region = find_best_region(heatmap)
    hdbg.dassert_lte(
        2, len(heatmap), "There must be at least 2 targets to compare"
    )
    # The cell of the best region closest to the region's centroid.
    x = heatmap["x_m"].to_numpy(dtype=float)
    y = heatmap["y_m"].to_numpy(dtype=float)
    x_center = x[in_region].mean()
    y_center = y[in_region].mean()
    dist_m = np.where(in_region, np.hypot(x - x_center, y - y_center), np.inf)
    marked = heatmap.iloc[int(np.argmin(dist_m))]
    # Highest `P(winner)` first; a stable sort keeps the grid order of ties.
    ranked = heatmap.sort_values("p_winner", ascending=False, kind="stable")
    best = ranked.iloc[0]
    second = ranked.iloc[1]
    # Gap in standard errors of the difference of 2 independent estimates.
    gap = float(best["p_winner"] - second["p_winner"])
    gap_se = float(np.hypot(best["p_winner_se"], second["p_winner_se"]))
    if gap_se > 0:
        gap_in_se = gap / gap_se
    else:
        gap_in_se = 0.0 if gap == 0 else float("inf")
    result = {
        "x_m": float(marked["x_m"]),
        "y_m": float(marked["y_m"]),
        "p_winner": float(marked["p_winner"]),
        "p_winner_se": float(marked["p_winner_se"]),
        "n_region_cells": int(in_region.sum()),
        "max_p_winner": float(best["p_winner"]),
        "max_p_winner_se": float(best["p_winner_se"]),
        "second_p_winner": float(second["p_winner"]),
        "second_p_winner_se": float(second["p_winner_se"]),
        "gap_in_se": gap_in_se,
    }
    _LOG.debug("return=%s", result)
    return result


# #############################################################################
# Court drawing
# #############################################################################


def _draw_center_service_line(
    ax: Axes, court: racket_params.CourtGeometry
) -> None:
    """
    Draw the center service line (x = 0) that splits the service boxes.

    `racket_strategy_utils.draw_court()` does not draw it. On each side of the
    net it spans the same `y` range as `racket_params.get_service_box_region()`:
    - Tennis: from the net to the service line
    - Pickleball: from the kitchen line to the baseline

    :param ax: axes to draw on
    :param court: court geometry giving the line positions
    """
    _LOG.debug(hprint.to_str("court"))
    # One segment per side of the net: `sign = -1` is player 1's side.
    for sign in (-1, 1):
        ax.plot(
            [0, 0],
            [sign * court.non_volley_zone_m, sign * court.service_line_m],
            color="gray",
            linestyle="--",
            linewidth=1.0,
        )


# #############################################################################
# Winner heatmap drawing
# #############################################################################


def _get_region_outline(
    in_region: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray
) -> List[List[Tuple[float, float]]]:
    """
    Find the segments of the outline of a region of grid cells.

    A side of a cell of the region is on the outline if the cell on the other
    side is not in the region, or if there is no cell there (the grid edge).

    :param in_region: True for the cells of the region, shape `(n_y, n_x)`,
        one row per `y` from the lowest
    :param x_edges: the `n_x + 1` cell edges along `x`
    :param y_edges: the `n_y + 1` cell edges along `y`
    :return: one segment `[(x_start, y_start), (x_end, y_end)]` per side on
        the outline, e.g., the 4 sides of a region of 1 cell
    """
    _LOG.debug(hprint.to_str("in_region.shape"))
    n_y, n_x = in_region.shape
    hdbg.dassert_eq(len(x_edges), n_x + 1, "One more x edge than cells")
    hdbg.dassert_eq(len(y_edges), n_y + 1, "One more y edge than cells")
    segments = []
    for iy in range(n_y):
        for ix in range(n_x):
            if not in_region[iy, ix]:
                continue
            x0, x1 = float(x_edges[ix]), float(x_edges[ix + 1])
            y0, y1 = float(y_edges[iy]), float(y_edges[iy + 1])
            # Left, right, bottom, and top sides, each kept if the neighbor on
            # that side is outside the region or outside the grid.
            if ix == 0 or not in_region[iy, ix - 1]:
                segments.append([(x0, y0), (x0, y1)])
            if ix == n_x - 1 or not in_region[iy, ix + 1]:
                segments.append([(x1, y0), (x1, y1)])
            if iy == 0 or not in_region[iy - 1, ix]:
                segments.append([(x0, y0), (x1, y0)])
            if iy == n_y - 1 or not in_region[iy + 1, ix]:
                segments.append([(x0, y1), (x1, y1)])
    _LOG.debug("return: %d segments", len(segments))
    return segments


def plot_winner_heatmap(
    ax: Axes,
    heatmap: pd.DataFrame,
    court: racket_params.CourtGeometry,
    serve_xy: Tuple[float, float],
) -> None:
    """
    Draw the probability of a winner of every target on the court.

    - Each grid cell is colored by its `P(winner)` on a fixed scale from 0 to
      1, so maps for different settings or sports compare by color; GP's
      `racket_strategy_utils.plot_court_heatmap()` scales to each map's own
      range instead
    - The court lines, with the center service line
    - The best region (`find_best_region()`) outlined in red, and the best
      target (`find_best_target()`), its middle, as a red star
    - Player 1 at `get_player1_xy()` and player 2 at `serve_xy`

    :param ax: axes to draw on
    :param heatmap: a full grid from `compute_winner_heatmap()`
    :param court: court geometry, the same as for `heatmap`
    :param serve_xy: where player 2 hits from, `(x1, y1)`, the same as for
        `heatmap`
    """
    hdbg.dassert_isinstance(
        heatmap, pd.DataFrame, "heatmap must be a pandas DataFrame"
    )
    # Log the shape only once it is known to exist.
    _LOG.debug(hprint.to_str("heatmap.shape serve_xy"))
    _dassert_point(serve_xy, "serve_xy")
    for column in ("cell_id", "x_min", "x_max", "y_min", "y_max"):
        hdbg.dassert_in(column, heatmap.columns, "Missing heatmap column")
    # Cell edges of the regular grid of `make_target_grid()`.
    x_edges: np.ndarray = np.append(
        np.unique(heatmap["x_min"]), heatmap["x_max"].max()
    )
    y_edges: np.ndarray = np.append(
        np.unique(heatmap["y_min"]), heatmap["y_max"].max()
    )
    n_x = len(x_edges) - 1
    n_y = len(y_edges) - 1
    hdbg.dassert_eq(
        n_x * n_y, len(heatmap), "heatmap must hold every cell of its grid"
    )
    # `cell_id` runs `y` outer, `x` inner: one row of the image per `y`.
    order = heatmap["cell_id"].to_numpy().argsort(kind="stable")
    values = heatmap["p_winner"].to_numpy()[order]
    values = values.reshape(n_y, n_x)
    # Fixed color scale from 0 to 1, below the court lines.
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        values,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        shading="flat",
        zorder=0,
    )
    plt.colorbar(mesh, ax=ax, label="P(winner)")
    racket_strategy_utils.draw_court(ax, court)
    _draw_center_service_line(ax, court)
    # The outline of the best region, in the same cell order as `values`;
    # `find_best_region()` also checks the other columns.
    in_region = find_best_region(heatmap)[order].reshape(n_y, n_x)
    segments = _get_region_outline(in_region, x_edges, y_edges)
    outline = mcollections.LineCollection(
        segments,
        colors="tab:red",
        linewidths=2.0,
        label=(
            f"best region: within {_BEST_REGION_N_SE:g} SE of the highest"
            f" ({int(in_region.sum())} cells)"
        ),
        zorder=3,
    )
    ax.add_collection(outline)
    # The best target and the two players, above the cells and the lines.
    best = find_best_target(heatmap)
    ax.plot(
        best["x_m"],
        best["y_m"],
        "*",
        markersize=18,
        color="tab:red",
        markeredgecolor="black",
        label=(
            f"best target (middle of the region), P(winner) ="
            f" {best['p_winner']:.3f} +- {best['p_winner_se']:.3f}"
        ),
        zorder=4,
    )
    ax.plot(*get_player1_xy(court), "o", color="tab:blue", label="player 1")
    ax.plot(*serve_xy, "o", color="tab:orange", label="player 2")
    ax.legend(loc="upper right", fontsize=9)


# #############################################################################
# Cell 0.1: The two courts
# #############################################################################


def cell0_1_plot_courts(*, figsize: Tuple[float, float] = (10.0, 7.0)) -> None:
    """
    Draw the tennis and pickleball courts side by side, in the same frame.

    :param figsize: figure size (width, height) in inches
    """
    _LOG.debug(hprint.to_str("figsize"))
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    # Same drawing for both sports, plus the center service line.
    for ax, sport in zip(axes, [racket_params.TENNIS, racket_params.PICKLEBALL]):
        racket_strategy_utils.draw_court(ax, sport.court)
        _draw_center_service_line(ax, sport.court)
        ax.set_title(sport.name)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


# #############################################################################
# Cell 1.1: Serve and return widget
# #############################################################################


def _round_slider_value(value: float) -> float:
    """
    Round a float slider value to 6 decimals, before any logic uses it.

    The `+` / `-` buttons add the 0.1 step in floating point, so the value
    drifts: 21 clicks of `+` from -2.1 give 6.4e-16 instead of 0, which
    `x1 <= 0` would put in the ad box. Adding 0.0 turns -0.0 into 0.0, so the
    value prints as "0.00" and not "-0.00".

    :param value: slider value, e.g., 6.4e-16
    :return: rounded value, e.g., 0.0
    """
    _LOG.debug(hprint.to_str("value"))
    rounded = round(value, 6) + 0.0
    _LOG.debug("return=%s", rounded)
    return rounded


def _get_server_xy(
    side: str, court: racket_params.CourtGeometry
) -> Tuple[float, float]:
    """
    Find where player 1 stands to serve from `side`.

    The rules only ask the server to stand behind the baseline, between the
    center mark and the sideline, on the half that matches the side (tennis
    and pickleball alike). The numbers are a typical singles stance, not rule
    values:
    - 0.5 m beside the center mark, so the diagonal of the serve is visible
    - 0.3 m behind the baseline, since the feet may not touch it

    :param side: "deuce" or "ad"
    :param court: court geometry, e.g., `racket_params.TENNIS.court`
    :return: server position, e.g., `(0.5, -12.185)` for a tennis deuce serve
    """
    _LOG.debug(hprint.to_str("side court"))
    hdbg.dassert_in(side, ("deuce", "ad"), "Invalid serve side")
    # Player 1 faces +y, so their right-hand (deuce) half is at x > 0.
    x = 0.5 if side == "deuce" else -0.5
    server_xy = (x, -court.length_m / 2 - 0.3)
    _LOG.debug("return=%s", server_xy)
    return server_xy


def _check_serve(
    side: str, x1: float, y1: float, court: racket_params.CourtGeometry
) -> Tuple[racket_params.CourtRegion, str]:
    """
    Find the legal service box for `side`, and why a serve to `(x1, y1)` is a
    fault, if it is one.

    Orientation, checked against `racket_params.get_service_box_region()` (the
    PR1 spec warns that deuce and ad are easy to flip): player 2 faces -y, so
    their right-hand (deuce) service box is at `x <= 0`. In this frame a deuce
    server stands at `x > 0` and serves cross-court into the box at `x <= 0`
    on player 2's side; ad is the mirror image.

    The legal box is only the one diagonal from the server. Every line is in,
    except the pickleball kitchen (non-volley) line, which is a fault.

    :param side: "deuce" or "ad", the side the server serves from
    :param x1: where the serve lands, across the court, e.g., -2.1
    :param y1: where the serve lands, along the court, e.g., 3.2
    :param court: court geometry, e.g., `racket_params.PICKLEBALL.court`
    :return: the legal box, and the fault: "" if the serve is in, otherwise
        "wide", "into the wrong box", "long", "in the kitchen", or "into the
        net", e.g., `(CourtRegion(...), "long")`
    """
    _LOG.debug(hprint.to_str("side x1 y1 court"))
    hdbg.dassert_in(side, ("deuce", "ad"), "Invalid serve side")
    box = racket_params.get_service_box_region(court, side)
    # Cap the box at the baseline: the pickleball preset's service line
    # (6.71 m) is 5 mm past its baseline (13.41 / 2 = 6.705 m), so without the
    # cap a serve just past the baseline would count as in.
    # `racket_params.py` is left as is; the issue is raised with GP.
    box = dataclasses.replace(box, y_max=min(box.y_max, court.length_m / 2))
    # Name one fault per serve, checking in this order.
    if abs(x1) > court.width_m / 2:
        fault = "wide"
    elif not box.x_min <= x1 <= box.x_max:
        fault = "into the wrong box"
    elif y1 > box.y_max:
        fault = "long"
    elif y1 <= court.non_volley_zone_m:
        # Pickleball: short of the kitchen line or on it. Tennis has no
        # kitchen (0 m), so only a serve on the net line, which the `y1` slider
        # cannot reach.
        has_kitchen = court.non_volley_zone_m > 0
        fault = "in the kitchen" if has_kitchen else "into the net"
    else:
        fault = ""
    _LOG.debug("return=%s", (box, fault))
    return box, fault


def _draw_service_box(ax: Axes, box: racket_params.CourtRegion) -> None:
    """
    Shade the legal service box: the one diagonal from the server.

    :param ax: court axes to draw on
    :param box: legal service box, from `_check_serve()`
    """
    _LOG.debug(hprint.to_str("box"))
    box_rect = mpatches.Rectangle(
        (box.x_min, box.y_min),
        box.x_max - box.x_min,
        box.y_max - box.y_min,
        color="tab:blue",
        alpha=0.1,
        label="service box",
    )
    ax.add_patch(box_rect)


def _draw_samples(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    is_in: np.ndarray,
    court: racket_params.CourtGeometry,
) -> None:
    """
    Draw the Monte Carlo landing points, and widen the view to include them.

    :param ax: court axes to draw on
    :param x: landing `x` of each sample, from `sample_landings()`
    :param y: landing `y` of each sample
    :param is_in: True where the sample lands in player 1's half
    :param court: court geometry, for the default view
    """
    _LOG.debug(hprint.to_str("court"))
    # Green in, red out, and see-through so the ellipses stay visible.
    ax.scatter(x[is_in], y[is_in], s=4, c="tab:green", alpha=0.4, label="in")
    ax.scatter(x[~is_in], y[~is_in], s=4, c="tab:red", alpha=0.4, label="out")
    # View: the court plus 1 m, widened to include every dot, so the cloud is
    # not cut off for targets and stds at the slider limits. The extra 0.5 m
    # keeps the outermost dots off the frame of the plot.
    half_width = court.width_m / 2
    half_length = court.length_m / 2
    ax.set_xlim(
        min(-half_width - 1, x.min() - 0.5), max(half_width + 1, x.max() + 0.5)
    )
    ax.set_ylim(
        min(-half_length - 1, y.min() - 0.5),
        max(half_length + 1, y.max() + 0.5),
    )


def _draw_sigma_ellipses(
    ax: Axes, target_xy: Tuple[float, float], std_xy: Tuple[float, float]
) -> None:
    """
    Draw the 1 and 2 sigma ellipses of the return around `target_xy`.

    The `k` sigma ellipse is `k` std away from `(x2, y2)` on each axis. In 2D
    the 1 sigma ellipse holds about 39% of the samples and the 2 sigma ellipse
    about 86% (`1 - exp(-k^2 / 2)`), not the 68% and 95% of 1D.

    :param ax: court axes to draw on
    :param target_xy: aim point `(x2, y2)`
    :param std_xy: standard deviations `(std_x2, std_y2)`
    """
    _LOG.debug(hprint.to_str("target_xy std_xy"))
    for k, linestyle in ((1, "-"), (2, "--")):
        ellipse = mpatches.Ellipse(
            target_xy,
            2 * k * std_xy[0],
            2 * k * std_xy[1],
            fill=False,
            # Explicit color: the `seaborn` style of the notebook makes patch
            # edges white by default.
            edgecolor="black",
            linestyle=linestyle,
            label=f"{k} sigma",
            # Above the dots (`zorder` 1) and the lines and markers (`zorder`
            # 2), so the ellipses stay visible in a dense cloud.
            zorder=3,
        )
        ax.add_patch(ellipse)


def _draw_players_and_arrows(
    ax: Axes,
    server_xy: Tuple[float, float],
    serve_xy: Tuple[float, float],
    target_xy: Tuple[float, float],
) -> None:
    """
    Draw the serve and return arrows, and the two players.

    :param ax: court axes to draw on
    :param server_xy: where player 1 serves from
    :param serve_xy: where the serve lands, `(x1, y1)`, where player 2 stands
    :param target_xy: where player 2 aims the return, `(x2, y2)`
    """
    _LOG.debug(hprint.to_str("server_xy serve_xy target_xy"))
    # An annotation with empty text is just an arrow, from `xytext` to `xy`.
    ax.annotate(
        "",
        xy=serve_xy,
        xytext=server_xy,
        arrowprops={"arrowstyle": "->", "color": "tab:blue"},
    )
    ax.annotate(
        "",
        xy=target_xy,
        xytext=serve_xy,
        arrowprops={"arrowstyle": "->", "color": "tab:orange"},
    )
    ax.plot(*server_xy, "o", color="tab:blue", label="player 1")
    ax.plot(*serve_xy, "o", color="tab:orange", label="player 2")


def _draw_fault(
    ax: Axes,
    server_xy: Tuple[float, float],
    serve_xy: Tuple[float, float],
    court: racket_params.CourtGeometry,
) -> None:
    """
    Draw a serve that is a fault: the serve arrow, a red cross where it lands,
    and a red "Fault" label on player 2's half, beside the arrow.

    :param ax: court axes to draw on
    :param server_xy: where player 1 serves from
    :param serve_xy: where the serve lands, `(x1, y1)`
    :param court: court geometry, to place the label in player 2's half
    """
    _LOG.debug(hprint.to_str("server_xy serve_xy"))
    # An annotation with empty text is just an arrow, from `xytext` to `xy`.
    ax.annotate(
        "",
        xy=serve_xy,
        xytext=server_xy,
        arrowprops={"arrowstyle": "->", "color": "tab:blue"},
    )
    ax.plot(*server_xy, "o", color="tab:blue", label="player 1")
    ax.plot(*serve_xy, "x", color="tab:red", markersize=10, mew=2, label="fault")
    # Label halfway up player 2's half, on a white box so it reads over the
    # court lines. The arrow's `x` always stays between the server's and the
    # landing point's, so the label starts 1 m past that range, on the half
    # the serve did not land in: it never covers the arrow.
    arrow_x = (server_xy[0], serve_xy[0])
    if serve_xy[0] <= 0:
        # Landed on the left half: the label goes right, growing rightward.
        label_x, ha = max(arrow_x) + 1.0, "left"
    else:
        label_x, ha = min(arrow_x) - 1.0, "right"
    ax.text(
        label_x,
        court.length_m / 4,
        "Fault",
        color="tab:red",
        fontsize=20,
        fontweight="bold",
        ha=ha,
        va="center",
        bbox={"facecolor": "white", "edgecolor": "tab:red"},
    )


def _build_comments_text(
    serve_xy: Tuple[float, float],
    side: str,
    target_xy: Tuple[float, float],
    std_xy: Tuple[float, float],
    p_in: float,
    p_hat: float,
    n_samples: int,
) -> str:
    """
    Build the text of the comments panel for a serve that is in.

    :param serve_xy: where the serve lands, `(x1, y1)`
    :param side: "deuce" or "ad", the side the server serves from
    :param target_xy: aim point of the return, `(x2, y2)`
    :param std_xy: standard deviations `(std_x2, std_y2)`
    :param p_in: closed-form probability that the return is in
    :param p_hat: fraction of the `n_samples` Monte Carlo samples that are in
    :param n_samples: number of Monte Carlo samples
    :return: multi-line text, e.g., starting with
        "Serve to (x1, y1) = (-2.10, 3.20)"
    """
    _LOG.debug(hprint.to_str("serve_xy side p_in p_hat"))
    # Standard error of `P_hat` from the true probability `P_in`, not from
    # `P_hat` itself: `sqrt(P_hat * (1 - P_hat) / N)` is 0 when all dots land
    # on one side, which would claim no uncertainty at all.
    p_hat_se = np.sqrt(p_in * (1 - p_in) / n_samples)
    text = (
        f"Serve to (x1, y1) = ({serve_xy[0]:.2f}, {serve_xy[1]:.2f})\n"
        f"  in the {side} box: yes\n\n"
        f"Return to (x2, y2) = ({target_xy[0]:.2f}, {target_xy[1]:.2f})\n"
        f"  (std_x2, std_y2) = ({std_xy[0]:.2f}, {std_xy[1]:.2f})\n\n"
        f"P_in (closed form) = {p_in:.4f}\n"
        f"P_hat (Monte Carlo, N = {n_samples}) = {p_hat:.4f}\n"
        f"  standard error = {p_hat_se:.4f}"
    )
    _LOG.debug("return=%s", text)
    return text


def _build_fault_text(
    serve_xy: Tuple[float, float], side: str, fault: str
) -> str:
    """
    Build the text of the comments panel for a serve that is a fault.

    :param serve_xy: where the serve lands, `(x1, y1)`
    :param side: "deuce" or "ad", the side the server serves from
    :param fault: why the serve is a fault, from `_check_serve()`, e.g., "long"
    :return: multi-line text, e.g., ending with "Fault: long"
    """
    _LOG.debug(hprint.to_str("serve_xy side fault"))
    text = (
        f"Serve to (x1, y1) = ({serve_xy[0]:.2f}, {serve_xy[1]:.2f})\n"
        f"  in the {side} box: no\n\n"
        f"Fault: {fault}\n"
        f"  No return, so no P_in and no P_hat"
    )
    _LOG.debug("return=%s", text)
    return text


def _build_shared_controls(
    court: racket_params.CourtGeometry,
) -> Dict[str, Tuple[Any, Any]]:
    """
    Build the controls that Cell 1.1 and Cell 2.1 share.

    - `side`: deuce or ad, the side player 1 serves from
    - `x1`, `y1`: where the serve lands
    - `std_x2`, `std_y2`: the spread of the return
    - `seed`: the seed of the Monte Carlo samples, placed last by both widgets

    :param court: court geometry, for the slider ranges and the start of the
        serve
    :return: `{name: (control, box)}` for "side", "x1", "y1", "std_x2",
        "std_y2", and "seed": the control to read, and the box to display;
        the side toggle is both
    """
    _LOG.debug(hprint.to_str("court"))
    half_width = court.width_m / 2
    half_length = court.length_m / 2
    # Side of the serve: it decides where the server stands and which box is
    # legal.
    side_toggle = ipywidgets.ToggleButtons(
        options=["deuce", "ad"], value="deuce", description="side"
    )
    # Serve sliders: `(x1, y1)` anywhere on player 2's half, so faults are
    # possible, starting in the middle of the deuce service box. `x1` spans the
    # singles court from sideline to sideline; `y1` goes from 0.1 m past the
    # net (never on the net line) to the baseline.
    x1_init = round(-half_width / 2, 1)
    x1_slider, x1_box = htutori.build_widget_control(
        "x1", "serve x", -half_width, half_width, 0.1, x1_init
    )
    y1_init = round((court.non_volley_zone_m + court.service_line_m) / 2, 1)
    y1_slider, y1_box = htutori.build_widget_control(
        "y1", "serve y", 0.1, half_length, 0.1, y1_init
    )
    # Spread of the return: a std must be strictly positive. From 0.1 m (a
    # very precise player) to 3 m (a wild shot, wider than a service box).
    std_x2_slider, std_x2_box = htutori.build_widget_control(
        "std_x2", "return std x", 0.1, 3.0, 0.1, 0.5
    )
    std_y2_slider, std_y2_box = htutori.build_widget_control(
        "std_y2", "return std y", 0.1, 3.0, 0.1, 1.0
    )
    # Seed of the Monte Carlo samples, placed last as the notebook rules ask;
    # 101 seeds are plenty to see the sampling noise.
    seed_slider, seed_box = htutori.build_widget_control(
        "seed", "random seed", 0, 100, 1, 1, is_float=False
    )
    controls: Dict[str, Tuple[Any, Any]] = {
        "side": (side_toggle, side_toggle),
        "x1": (x1_slider, x1_box),
        "y1": (y1_slider, y1_box),
        "std_x2": (std_x2_slider, std_x2_box),
        "std_y2": (std_y2_slider, std_y2_box),
        "seed": (seed_slider, seed_box),
    }
    return controls


def cell1_1_plot_shot_widget(
    sport: racket_params.SportParams,
    *,
    figsize: Tuple[float, float] = (12.0, 7.0),
) -> ipywidgets.VBox:
    """
    Build the serve and return widget for `sport` and display it.

    Player 1 serves from the chosen side (deuce or ad), behind the baseline
    beside the center mark, to `(x1, y1)`. The serve is in only if it lands in
    the service box diagonal from the server. If it is in, player 2 moves to
    `(x1, y1)` and aims a return at `(x2, y2)` with Gaussian spread
    `(std_x2, std_y2)`. Every control change redraws a 1 x 2 figure:
    - Court: the serve, and for a serve that is in, the return, the 1 and 2
      sigma ellipses, and Monte Carlo landing points colored by in or out; for
      a fault, a red cross and a "Fault" label instead
    - Comments: whether the serve is in; for a serve that is in, the
      closed-form `P_in` and the Monte Carlo estimate with its standard error;
      and the legend of the court

    :param sport: sport whose court is drawn, e.g., `racket_params.TENNIS`
    :param figsize: figure size (width, height) in inches
    :return: the displayed widget: the controls, then the plot output
    """
    _LOG.debug(hprint.to_str("sport"))
    court = sport.court
    half_width = court.width_m / 2
    half_length = court.length_m / 2
    # Enough Monte Carlo points for a standard error of at most 0.016, few
    # enough to keep the scatter readable.
    n_samples = 1000
    # The side of the serve, the serve sliders, the spread of the return, and
    # the seed, shared with Cell 2.1.
    shared = _build_shared_controls(court)
    side_toggle = shared["side"][0]
    x1_slider, x1_box = shared["x1"]
    y1_slider, y1_box = shared["y1"]
    # Return sliders: `(x2, y2)` on player 1's side (`y2 <= 0`), up to 2 m
    # outside the sidelines and the baseline, so out shots can be explored,
    # starting 1.5 m inside the baseline.
    x2_init = round(half_width / 2, 1)
    x2_slider, x2_box = htutori.build_widget_control(
        "x2", "return x", -half_width - 2, half_width + 2, 0.1, x2_init
    )
    y2_init = round(-half_length + 1.5, 1)
    y2_slider, y2_box = htutori.build_widget_control(
        "y2", "return y", -half_length - 2, 0.0, 0.1, y2_init
    )
    std_x2_slider, std_x2_box = shared["std_x2"]
    std_y2_slider, std_y2_box = shared["std_y2"]
    seed_slider, seed_box = shared["seed"]
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw both panels from the current control values.

        :param change: change event sent by `observe()` (unused)
        """
        _ = change
        with output:
            clear_output(wait=True)
            # Read the slider values, rounded before any logic uses them.
            side = side_toggle.value
            x1 = _round_slider_value(x1_slider.value)
            y1 = _round_slider_value(y1_slider.value)
            x2 = _round_slider_value(x2_slider.value)
            y2 = _round_slider_value(y2_slider.value)
            std_xy = (
                _round_slider_value(std_x2_slider.value),
                _round_slider_value(std_y2_slider.value),
            )
            # The side, not `x1`, decides the stance and the legal box.
            server_xy = _get_server_xy(side, court)
            box, fault = _check_serve(side, x1, y1, court)
            # Panel 1: the court and the legal box, shaded.
            fig, (ax_court, ax_text) = plt.subplots(1, 2, figsize=figsize)
            racket_strategy_utils.draw_court(ax_court, court)
            _draw_center_service_line(ax_court, court)
            _draw_service_box(ax_court, box)
            if fault:
                # A fault ends the point: no return, so no dots, no ellipses,
                # and no probability.
                _draw_fault(ax_court, server_xy, (x1, y1), court)
                text = _build_fault_text((x1, y1), side, fault)
            else:
                # Closed form and Monte Carlo estimate on player 1's half.
                region = get_striker_half_region(court)
                p_in = compute_p_in((x2, y2), std_xy, region)
                rng = np.random.default_rng(seed_slider.value)
                x, y = sample_landings((x2, y2), std_xy, n_samples, rng)
                is_in = region.contains(x, y)
                p_hat = is_in.mean()
                # The dots, the ellipses, and the two shots. Within the same
                # `zorder` later artists are drawn on top, so the arrows
                # (`zorder` 3) end up above the ellipses (`zorder` 3); the
                # player markers (`zorder` 2) stay below them.
                _draw_samples(ax_court, x, y, is_in, court)
                _draw_sigma_ellipses(ax_court, (x2, y2), std_xy)
                _draw_players_and_arrows(ax_court, server_xy, (x1, y1), (x2, y2))
                text = _build_comments_text(
                    (x1, y1), side, (x2, y2), std_xy, p_in, p_hat, n_samples
                )
            ax_court.set_title(sport.name)
            # Panel 2: comments with the current state only.
            ax_text.axis("off")
            ax_text.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
            htutori.add_fitted_text_box(
                ax_text, text, max_fontsize=13, min_fontsize=9
            )
            # Legend in the free lower part of the comments panel, so it never
            # covers the court.
            handles, labels = ax_court.get_legend_handles_labels()
            ax_text.legend(handles, labels, loc="lower left", fontsize=10)
            plt.tight_layout()
            plt.show()
            # A figure left open inside an `Output` widget can stall the kernel
            # on rerun, so close it once shown.
            plt.close(fig)

    # Redraw on every control change: no Run button.
    controls = [
        side_toggle,
        x1_slider,
        y1_slider,
        x2_slider,
        y2_slider,
        std_x2_slider,
        std_y2_slider,
        seed_slider,
    ]
    for control in controls:
        control.observe(update_plot, names="value")
    # Draw once now, so the cell has output even without interaction.
    update_plot()
    # Controls on top, the plot below.
    boxes = [x1_box, y1_box, x2_box, y2_box, std_x2_box, std_y2_box, seed_box]
    widget = ipywidgets.VBox([side_toggle] + boxes + [output])
    display(widget)
    return widget


# #############################################################################
# Cell 2.1: Winner probability widget
# #############################################################################


# Default ball speed of the return per sport, in m/s (decision D8):
# illustrative values, well under each sport's `max_ball_speed_mps`.
WINNER_BALL_SPEED_MPS = {"Tennis": 25.0, "Pickleball": 12.0}


def _build_settings_string(
    sport_name: str,
    side: str,
    serve_xy: Tuple[float, float],
    std_xy: Tuple[float, float],
    ball_speed_mps: float,
    reaction_time_s: float,
    move_speed_mps: float,
    seed: int,
    grid_size: Tuple[int, int],
    n_samples: int,
) -> str:
    """
    Build the one-line settings string of the winner widget.

    It lists every control as `name=value`, in the order of the controls, then
    the grid size and the samples per target, so a result copied into the
    Gdoc can be reproduced exactly (GP on #694). Each float is written with
    `repr()`, the shortest text that reads back as the same float, e.g.,
    "-2.1" and not "-2.10".

    :param sport_name: name of the sport, e.g., "Tennis"
    :param side: "deuce" or "ad", the side the server serves from
    :param serve_xy: where the serve lands, `(x1, y1)`
    :param std_xy: standard deviations `(std_x2, std_y2)` of the return
    :param ball_speed_mps: speed of the return in m/s
    :param reaction_time_s: reaction time `t_r` of player 1 in seconds
    :param move_speed_mps: running speed `v_p` of player 1 in m/s
    :param seed: seed of the Monte Carlo samples
    :param grid_size: grid `(n_x, n_y)` of the map, e.g., `(10, 15)`
    :param n_samples: Monte Carlo samples per target, e.g., 5000
    :return: e.g., "sport=Tennis side=deuce x1=-2.1 y1=3.2 std_x2=0.5
        std_y2=1.0 ball_speed=25.0 t_r=0.2 v_p=1.5 seed=1 n_x=10 n_y=15
        n_samples=5000"
    """
    _LOG.debug(
        hprint.to_str("sport_name side serve_xy std_xy seed grid_size n_samples")
    )
    values = [
        ("sport", sport_name),
        ("side", side),
        ("x1", repr(float(serve_xy[0]))),
        ("y1", repr(float(serve_xy[1]))),
        ("std_x2", repr(float(std_xy[0]))),
        ("std_y2", repr(float(std_xy[1]))),
        ("ball_speed", repr(float(ball_speed_mps))),
        ("t_r", repr(float(reaction_time_s))),
        ("v_p", repr(float(move_speed_mps))),
        ("seed", str(int(seed))),
        ("n_x", str(int(grid_size[0]))),
        ("n_y", str(int(grid_size[1]))),
        ("n_samples", str(int(n_samples))),
    ]
    settings = " ".join(f"{name}={value}" for name, value in values)
    _LOG.debug("return=%s", settings)
    return settings


def _format_p_winner(p_winner: float, p_winner_se: float, n_samples: int) -> str:
    """
    Format a probability of a winner with its standard error.

    When every sample agrees (`P(winner)` is 0 or 1) the standard error
    `sqrt(p * (1 - p) / N)` is 0, which does not mean that there is no
    uncertainty (decision D18): the count of winners is shown instead.

    :param p_winner: probability of a winner, from `find_best_target()`
    :param p_winner_se: its standard error
    :param n_samples: number of Monte Carlo samples per target
    :return: e.g., "0.9312 +- 0.0036 (N = 5000)", or "1.0000 (5000 of 5000
        samples)" when the standard error is 0
    """
    _LOG.debug(hprint.to_str("p_winner p_winner_se n_samples"))
    if p_winner_se > 0:
        text = f"{p_winner:.4f} +- {p_winner_se:.4f} (N = {n_samples})"
    else:
        n_winners = int(round(p_winner * n_samples))
        text = f"{p_winner:.4f} ({n_winners} of {n_samples} samples)"
    _LOG.debug("return=%s", text)
    return text


def _format_point(xy: Tuple[float, float]) -> str:
    """
    Format a point as "(x, y)" with 2 decimals, never as "-0.00".

    Grid cell centers can be off by a rounding error, e.g., -2.2e-16 instead
    of 0 for the middle column of the pickleball grid, which would print as
    "-0.00".

    :param xy: point in meters, e.g., `(-2.2e-16, -3.0175)`
    :return: e.g., "(0.00, -3.02)"
    """
    _LOG.debug(hprint.to_str("xy"))
    # Adding 0.0 turns the -0.0 of `round()` into 0.0.
    x, y = (round(float(value), 6) + 0.0 for value in xy)
    text = f"({x:.2f}, {y:.2f})"
    _LOG.debug("return=%s", text)
    return text


def _build_winner_text(
    serve_xy: Tuple[float, float],
    side: str,
    heatmap: pd.DataFrame,
    best: Dict[str, float],
    n_samples: int,
    settings: str,
) -> str:
    """
    Build the text of the comments panel of the winner widget for a serve
    that is in.

    :param serve_xy: where the serve lands, `(x1, y1)`
    :param side: "deuce" or "ad", the side the server serves from
    :param heatmap: the map from `compute_winner_heatmap()`
    :param best: the best target of `heatmap`, from `find_best_target()`
    :param n_samples: number of Monte Carlo samples per target
    :param settings: settings string, from `_build_settings_string()`
    :return: multi-line text, e.g., starting with
        "Serve to (x1, y1) = (-2.10, 3.20)"
    """
    _LOG.debug(hprint.to_str("serve_xy side best n_samples settings"))
    # The row of the marked target, for its `P(in)` and `P(reached)`: the star
    # is a cell of `heatmap`, so its coordinates match exactly.
    is_best = (heatmap["x_m"] == best["x_m"]) & (heatmap["y_m"] == best["y_m"])
    hdbg.dassert_eq(
        int(is_best.sum()), 1, "The best target must be one cell of heatmap"
    )
    row = heatmap[is_best].iloc[0]
    p_winner = _format_p_winner(best["p_winner"], best["p_winner_se"], n_samples)
    text = (
        f"Serve to (x1, y1) = ({serve_xy[0]:.2f}, {serve_xy[1]:.2f})\n"
        f"  in the {side} box: yes\n\n"
        f"Best target (star) = {_format_point((best['x_m'], best['y_m']))}\n"
        f"  P(winner) = {p_winner}\n"
        f"  P(in) = {row['p_in']:.4f}, P(reached) = {row['p_reached']:.4f}\n"
        f"Best region: {int(best['n_region_cells'])} of {len(heatmap)} cells\n"
        f"  within {_BEST_REGION_N_SE:g} SE of the highest P(winner)\n\n"
        f"Settings:\n{settings}"
    )
    _LOG.debug("return=%s", text)
    return text


def _build_winner_fault_text(
    serve_xy: Tuple[float, float], side: str, fault: str, settings: str
) -> str:
    """
    Build the text of the comments panel of the winner widget for a serve
    that is a fault.

    :param serve_xy: where the serve lands, `(x1, y1)`
    :param side: "deuce" or "ad", the side the server serves from
    :param fault: why the serve is a fault, from `_check_serve()`, e.g., "long"
    :param settings: settings string, from `_build_settings_string()`
    :return: multi-line text, e.g., with the line "Fault: long"
    """
    _LOG.debug(hprint.to_str("serve_xy side fault settings"))
    text = (
        f"Serve to (x1, y1) = ({serve_xy[0]:.2f}, {serve_xy[1]:.2f})\n"
        f"  in the {side} box: no\n\n"
        f"Fault: {fault}\n"
        f"  No return, so no P(winner) and no heatmap\n\n"
        f"Settings:\n{settings}"
    )
    _LOG.debug("return=%s", text)
    return text


def cell2_1_plot_winner_widget(
    sport: racket_params.SportParams,
    *,
    figsize: Tuple[float, float] = (18.0, 7.0),
) -> ipywidgets.VBox:
    """
    Build the winner probability widget for `sport` and display it.

    Player 1 serves from the chosen side to `(x1, y1)`, with the serve rules
    of Cell 1.1. If the serve is in, player 2 returns from `(x1, y1)`, and
    every target of a grid over player 1's half gets its probability of a
    winner, from the spread `(std_x2, std_y2)`, the ball speed, and player 1's
    reaction time `t_r` and running speed `v_p`. Every control change redraws
    a 1 x 2 figure:
    - Court: the map of `P(winner)` from `plot_winner_heatmap()`, with the
      best region and the best target; for a fault, the fault as in Cell 1.1,
      and no map
    - Comments: whether the serve is in; for a serve that is in, the best
      target with its `P(winner)`, `P(in)`, and `P(reached)`, and the size of
      the best region; the settings string; and the legend of the court
    Under the figure, the settings string is repeated as text, which can be
    selected and copied into the Gdoc: the comments panel is part of the
    image (decision D34).

    :param sport: sport whose court is drawn, e.g., `racket_params.TENNIS`
    :param figsize: figure size (width, height) in inches; wide enough that
        the settings string fits on one line
    :return: the displayed widget: the controls, the plot output, then the
        settings line
    """
    _LOG.debug(hprint.to_str("sport"))
    court = sport.court
    # Grid and samples per target of the map (decision D24).
    n_x, n_y = HEATMAP_GRID_SIZE[sport.name]
    n_samples = HEATMAP_N_SAMPLES
    # The side of the serve, the serve sliders, the spread of the return, and
    # the seed, as in Cell 1.1; the spread does not depend on the ball speed
    # (decision D9, #723).
    shared = _build_shared_controls(court)
    side_toggle = shared["side"][0]
    x1_slider, x1_box = shared["x1"]
    y1_slider, y1_box = shared["y1"]
    std_x2_slider, std_x2_box = shared["std_x2"]
    std_y2_slider, std_y2_box = shared["std_y2"]
    # Ball speed of the return, from a slow 5 m/s to the sport's fastest ball.
    ball_speed_slider, ball_speed_box = htutori.build_widget_control(
        "ball_speed",
        "ball speed (m/s)",
        5.0,
        sport.max_ball_speed_mps,
        0.5,
        WINNER_BALL_SPEED_MPS[sport.name],
    )
    # Player 1's reaction time and running speed, starting at the values of
    # `racket_params.DEFAULT_PLAYER` (decision D8); `v_p = 0` means that
    # player 1 does not move.
    t_r_slider, t_r_box = htutori.build_widget_control(
        "t_r", "reaction time (s)", 0.0, 1.0, 0.05, 0.2
    )
    v_p_slider, v_p_box = htutori.build_widget_control(
        "v_p", "player 1 speed (m/s)", 0.0, 6.0, 0.1, 1.5
    )
    seed_slider, seed_box = shared["seed"]
    output = ipywidgets.Output()
    # The settings string as text under the figure, so it can be copied.
    settings_line = ipywidgets.HTML()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw both panels and the settings line from the current control
        values.

        :param change: change event sent by `observe()` (unused)
        """
        _ = change
        with output:
            clear_output(wait=True)
            # Read the control values, rounded before any logic uses them.
            side = side_toggle.value
            serve_xy = (
                _round_slider_value(x1_slider.value),
                _round_slider_value(y1_slider.value),
            )
            std_xy = (
                _round_slider_value(std_x2_slider.value),
                _round_slider_value(std_y2_slider.value),
            )
            ball_speed_mps = _round_slider_value(ball_speed_slider.value)
            reaction_time_s = _round_slider_value(t_r_slider.value)
            move_speed_mps = _round_slider_value(v_p_slider.value)
            seed = int(seed_slider.value)
            settings = _build_settings_string(
                sport.name,
                side,
                serve_xy,
                std_xy,
                ball_speed_mps,
                reaction_time_s,
                move_speed_mps,
                seed,
                (n_x, n_y),
                n_samples,
            )
            settings_line.value = (
                f"settings: <code>{html.escape(settings)}</code>"
            )
            # The serve rules of Cell 1.1, for the chosen side.
            box, fault = _check_serve(side, serve_xy[0], serve_xy[1], court)
            # The comments panel is wider, so the settings string fits on one
            # line.
            fig, (ax_court, ax_text) = plt.subplots(
                1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 1.6]}
            )
            if fault:
                # A fault ends the point: the fault of Cell 1.1, and no map.
                racket_strategy_utils.draw_court(ax_court, court)
                _draw_center_service_line(ax_court, court)
                _draw_service_box(ax_court, box)
                server_xy = _get_server_xy(side, court)
                _draw_fault(ax_court, server_xy, serve_xy, court)
                text = _build_winner_fault_text(serve_xy, side, fault, settings)
            else:
                # `P(winner)` of every target of the grid, then the map with
                # the best region, the best target, and the two players.
                heatmap = compute_winner_heatmap(
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
                plot_winner_heatmap(ax_court, heatmap, court, serve_xy)
                _draw_service_box(ax_court, box)
                # The legend goes to the comments panel, below. The star's
                # label repeats `P(winner)` with a standard error of 0 when
                # every sample agrees, so the comments give the number
                # instead (decision D18).
                ax_court.get_legend().remove()
                for line in ax_court.get_lines():
                    if line.get_marker() == "*":
                        line.set_label("best target (middle of the region)")
                best = find_best_target(heatmap)
                text = _build_winner_text(
                    serve_xy, side, heatmap, best, n_samples, settings
                )
            ax_court.set_title(sport.name)
            # Panel 2: comments with the current state only. At most 9 pt, so
            # the longest settings string stays on one line at any DPI.
            ax_text.axis("off")
            ax_text.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
            htutori.add_fitted_text_box(
                ax_text, text, max_fontsize=9, min_fontsize=8
            )
            # Legend in the free lower part of the comments panel, so it never
            # covers the court.
            handles, labels = ax_court.get_legend_handles_labels()
            ax_text.legend(handles, labels, loc="lower left", fontsize=10)
            plt.tight_layout()
            plt.show()
            # A figure left open inside an `Output` widget can stall the kernel
            # on rerun, so close it once shown.
            plt.close(fig)

    # Redraw on every control change: no Run button.
    controls = [
        side_toggle,
        x1_slider,
        y1_slider,
        std_x2_slider,
        std_y2_slider,
        ball_speed_slider,
        t_r_slider,
        v_p_slider,
        seed_slider,
    ]
    for control in controls:
        control.observe(update_plot, names="value")
    # Draw once now, so the cell has output even without interaction.
    update_plot()
    # Controls on top, the plot below, then the settings line.
    boxes = [
        x1_box,
        y1_box,
        std_x2_box,
        std_y2_box,
        ball_speed_box,
        t_r_box,
        v_p_box,
        seed_box,
    ]
    widget = ipywidgets.VBox([side_toggle] + boxes + [output, settings_line])
    display(widget)
    return widget
