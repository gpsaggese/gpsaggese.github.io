"""
Utilities for the `racket_shot_probability` notebook.

Player 1 serves to `(x1, y1)`, player 2 moves there and aims a return at
`(x2, y2)` on player 1's half of the court. The return lands at a 2D Gaussian
point around `(x2, y2)` with independent `x` and `y` errors, so the
probability that the return is in has a closed form.

Import as:

import research.Optimal_strategy_for_racket_sports.notebooks.racket_shot_probability_utils as rosfrsnrspu
"""

import logging
from typing import Tuple

import numpy as np
import scipy.stats

import helpers.hdbg as hdbg
import helpers.hprint as hprint
from research.Optimal_strategy_for_racket_sports import racket_params

_LOG = logging.getLogger(__name__)


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
    # Mirror of `racket_params.get_half_court_region()` across the net (y = 0).
    region = racket_params.CourtRegion(
        x_min=-court.width_m / 2,
        x_max=court.width_m / 2,
        y_min=-court.length_m / 2,
        y_max=0,
    )
    _LOG.debug("return=%s", region)
    return region


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
    x2, y2 = target_xy
    std_x2, std_y2 = std_xy
    # `scipy` returns NaN instead of raising for a std that is not positive.
    hdbg.dassert_lt(0, std_x2, "std_x2 must be positive")
    hdbg.dassert_lt(0, std_y2, "std_y2 must be positive")
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
    :param n_samples: number of landing points to draw
    :param rng: random generator to draw from, passed in explicitly so that the
        same seed gives the same points, e.g., `np.random.default_rng(1)`
    :return: arrays `(x, y)` of landing coordinates, each of shape
        `(n_samples,)`
    """
    _LOG.debug(hprint.to_str("target_xy std_xy n_samples"))
    x2, y2 = target_xy
    std_x2, std_y2 = std_xy
    # Same check as `compute_p_in()`: `rng.normal()` silently returns the target
    # every time for a std of 0.
    hdbg.dassert_lt(0, std_x2, "std_x2 must be positive")
    hdbg.dassert_lt(0, std_y2, "std_y2 must be positive")
    # Draw each coordinate on its own since `X` and `Y` are independent.
    x = rng.normal(x2, std_x2, n_samples)
    y = rng.normal(y2, std_y2, n_samples)
    _LOG.debug("return: %d landing points", len(x))
    return x, y
