"""
Utilities for the `racket_sports_07_speed_vs_accuracy` notebook.

The spread of the return as a function of the ball speed (#723). The Milestone 2
model keeps `std_x2`, `std_y2`, and the ball speed as 3 independent inputs
(design principle P1): `build_spread_from_ball_speed()` computes the 2 spreads
from the ball speed with one model, for the sweeps only.

Import as:

import research.Optimal_strategy_for_racket_sports.notebooks.racket_sports_07_speed_vs_accuracy_utils as rosfrsnrs0svau
"""

import logging
from typing import Tuple

import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebo
import helpers.hprint as hprint
import numpy as np

_LOG = logging.getLogger(__name__)


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utils logger.

    :param notebook_log: logger owned by the notebook
    """
    _LOG.debug(hprint.to_str("notebook_log"))
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# Spread model
# #############################################################################

# Shapes of the factor `f` that multiplies the spread at the reference ball
# speed `v0`, with `u = (v_b - v0) / v0`:
# - "linear": `1 + a * u`, the base; a ball slower than `v0` is more accurate
# - "knee": `1 + a * max(0, u)`; a ball slower than `v0` is as accurate as at
#   `v0`
# - "parabola": `1 + a * u^2`; `v0` is the most accurate speed
# - "one_sided_parabola": `1 + a * max(0, u)^2`
# All 4 give `f = 1` at `v_b = v0` or `a = 0`, and `f = 1 + a` at `v_b = 2 v0`.
SPREAD_SHAPES = ("linear", "knee", "parabola", "one_sided_parabola")


def compute_speed_factor(
    ball_speed_mps: float, v0_mps: float, a: float, shape: str
) -> float:
    """
    Compute the factor that multiplies the spread at a ball speed.

    The factor depends on the speeds only through `v_b / v0`, so multiplying
    `v_b` and `v0` by the same number leaves it unchanged.

    :param ball_speed_mps: ball speed `v_b` of the return, e.g., 37.5
    :param v0_mps: reference ball speed `v0`, where the factor is 1, e.g.,
        25.0 for tennis
    :param a: coefficient of the factor, at least 0; `a = 0` gives 1, the
        Milestone 2 model
    :param shape: one of `SPREAD_SHAPES`
    :return: the factor, e.g., 1.5 for 37.5 m/s with `v0 = 25`, `a = 1`, and
        "linear"
    """
    _LOG.debug(hprint.to_str("ball_speed_mps v0_mps a shape"))
    for name, value in (("ball_speed_mps", ball_speed_mps), ("v0_mps", v0_mps)):
        hdbg.dassert(np.isfinite(value), "%s=%s must be finite", name, value)
        hdbg.dassert_lt(0.0, value, "%s must be positive", name)
    # Any `a >= 0` is allowed, so a sweep can perturb it.
    hdbg.dassert(np.isfinite(a), "a=%s must be finite", a)
    hdbg.dassert_lte(0.0, a, "a must not be negative")
    hdbg.dassert_in(shape, SPREAD_SHAPES, "Unknown shape")
    u = (ball_speed_mps - v0_mps) / v0_mps
    if shape == "linear":
        factor = 1.0 + a * u
    elif shape == "knee":
        factor = 1.0 + a * max(0.0, u)
    elif shape == "parabola":
        factor = 1.0 + a * u**2
    else:
        # The shape is "one_sided_parabola", checked above.
        factor = 1.0 + a * max(0.0, u) ** 2
    # Only "linear" with `a > 1` can reach 0, at a ball speed of at most
    # `v0 * (1 - 1 / a)`, e.g., `a = 2` at `v_b = v0 / 2`.
    hdbg.dassert_lt(
        0.0,
        factor,
        "The factor must be positive: ball_speed_mps=%s v0_mps=%s a=%s shape=%s",
        ball_speed_mps,
        v0_mps,
        a,
        shape,
    )
    return float(factor)


def build_spread_from_ball_speed(
    ball_speed_mps: float,
    std0_x: float,
    std0_y: float,
    v0_mps: float,
    a_x: float,
    a_y: float,
    shape: str,
) -> Tuple[float, float]:
    """
    Compute the spread of the return at a ball speed, for the sweeps.

    The width and the depth get their own factor, with the same shape and ball
    speed: `std_x2 = f(a_x) * std0_x` and `std_y2 = f(a_y) * std0_y`, with `f`
    from `compute_speed_factor()`. The model does not change: it gets the
    result as its 2 independent inputs `std_x2` and `std_y2`.

    :param ball_speed_mps: ball speed `v_b` of the return, e.g., 37.5
    :param std0_x: spread across the court (the width) at `v0`, in meters,
        e.g., 0.5
    :param std0_y: spread along the court (the depth) at `v0`, in meters,
        e.g., 1.0
    :param v0_mps: reference ball speed `v0`, where the spread is `(std0_x,
        std0_y)`, e.g., 25.0 for tennis
    :param a_x: coefficient of the width factor, at least 0
    :param a_y: coefficient of the depth factor, at least 0
    :param shape: one of `SPREAD_SHAPES`
    :return: `(std_x2, std_y2)` in meters, e.g., `(0.625, 1.5)` for 37.5 m/s
        with `std0 = (0.5, 1.0)`, `v0 = 25`, `a_x = 0.5`, `a_y = 1`, and
        "linear"
    """
    _LOG.debug(
        hprint.to_str("ball_speed_mps std0_x std0_y v0_mps a_x a_y shape")
    )
    for name, value in (("std0_x", std0_x), ("std0_y", std0_y)):
        hdbg.dassert(np.isfinite(value), "%s=%s must be finite", name, value)
        hdbg.dassert_lt(0.0, value, "%s must be positive", name)
    factor_x = compute_speed_factor(ball_speed_mps, v0_mps, a_x, shape)
    factor_y = compute_speed_factor(ball_speed_mps, v0_mps, a_y, shape)
    spread = (factor_x * std0_x, factor_y * std0_y)
    return spread
