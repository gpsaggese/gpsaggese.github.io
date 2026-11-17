"""
Utilities for the `racket_sports_07_speed_vs_accuracy` notebook.

The spread of the return as a function of the ball speed (#723). The Milestone 2
model keeps `std_x2`, `std_y2`, and the ball speed as 3 independent inputs
(design principle P1): `build_spread_from_ball_speed()` computes the 2 spreads
from the ball speed with one model, for the sweeps only.

- The spread model: `compute_speed_factor()`, `build_spread_from_ball_speed()`
- The sweeps: the ball speed for each coefficient `a`, then one variable at a
  time; every map is a run of `compute_run()` of the notebook 05 utils

Import as:

import research.Optimal_strategy_for_racket_sports.notebooks.racket_sports_07_speed_vs_accuracy_utils as rosfrsnrs0svau
"""

import dataclasses
import logging
from typing import Any, Dict, Tuple

import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebo
import helpers.hprint as hprint
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

import research.Optimal_strategy_for_racket_sports.notebooks.racket_sports_01_shot_probability_utils as rosfrsnrs0spu
import research.Optimal_strategy_for_racket_sports.notebooks.racket_sports_05_variable_sensitivity_utils as rosfrsnrs0vsu

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


# #############################################################################
# Sweep settings
# #############################################################################

# The base of the sweeps: the linear shape, with `a_y = a` for the depth and
# `a_x = a / 2` for the width.
_BASE_VARIANT = "base"
# What each variant changes from the base, one variable at a time (design
# principle P3). The values are set to something reasonable and then perturbed,
# to see how stable the best ball speed is (GP, Q4):
# - `shape`: the shape of the factor; the base is "linear"
# - `a_x_fraction`: `a_x / a`; the base is 0.5, our assumption, between a width
#   that does not depend on the ball speed (0) and a width as sensitive as the
#   depth (1)
# - `std0_x`, `std0_y`: the spread at `v0`, in meters; the base is the default
#   spread of notebook 05, `(0.5, 1.0)`, and the 2 variants halve and double it
# - `v_p`: the speed of player 1, in m/s; the base is the default, 1.5; 1.9 is
#   the top of the split-step speeds in the paper's references, and 3.0 is an
#   assumption
# - `serve_name`: a named serve of notebook 05; the base is "body-mid", the
#   default serve, and "T-deep" helped player 1 most in its corner runs
# - `seed`: the base is the default, 1
_VARIANTS: Dict[str, Dict[str, Any]] = {
    _BASE_VARIANT: {},
    "knee": {"shape": "knee"},
    "parabola": {"shape": "parabola"},
    "one_sided_parabola": {"shape": "one_sided_parabola"},
    "a_x_eq_a_y": {"a_x_fraction": 1.0},
    "std0_half": {"std0_x": 0.25, "std0_y": 0.5},
    "std0_double": {"std0_x": 1.0, "std0_y": 2.0},
    "v_p_1.9": {"v_p": 1.9},
    "v_p_3.0": {"v_p": 3.0},
    "serve_T_deep": {"serve_name": "T-deep"},
    "seed_2": {"seed": 2},
}


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    """
    How much of the speed sweeps to run.

    Reducing a run removes whole maps and lowers `n_samples`, but keeps the
    cells of each map, so a reduced run compares cell by cell with a full one.
    """

    # Sports to run.
    sports: Tuple[str, ...]
    # Grid `(n_x, n_y)` of a map, per sport.
    grid_size: Dict[str, Tuple[int, int]]
    # Monte Carlo samples per target.
    n_samples: int
    # Ball speeds of a sweep as `v_b / v0`, e.g., 0.5 for `v0 / 2`.
    speed_ratios: Tuple[float, ...]
    # Values of the coefficient `a` per variant of `_VARIANTS`; a variant that
    # is not listed is not run.
    a_values: Dict[str, Tuple[float, ...]]


# 768 maps. The range of `a`, from 0 (the Milestone 2 model) to 1, is our
# assumption: no study gives a value. The variants run at `a` = 0.5 and 1, and
# the second seed only at `a = 1`.
_FULL_CONFIG = ExperimentConfig(
    sports=("Tennis", "Pickleball"),
    grid_size=rosfrsnrs0spu.HEATMAP_GRID_SIZE,
    n_samples=rosfrsnrs0spu.HEATMAP_N_SAMPLES,
    # From `v0 / 2` to `2 v0` in steps of `v0 / 10`.
    speed_ratios=tuple(tenth / 10 for tenth in range(5, 21)),
    a_values={
        _BASE_VARIANT: (0.0, 0.25, 0.5, 0.75, 1.0),
        "knee": (0.5, 1.0),
        "parabola": (0.5, 1.0),
        "one_sided_parabola": (0.5, 1.0),
        "a_x_eq_a_y": (0.5, 1.0),
        "std0_half": (0.5, 1.0),
        "std0_double": (0.5, 1.0),
        "v_p_1.9": (0.5, 1.0),
        "v_p_3.0": (0.5, 1.0),
        "serve_T_deep": (0.5, 1.0),
        "seed_2": (1.0,),
    },
)
# 180 maps. The speeds keep both ends and `v0`, so the loss of the fastest ball
# and the map of the Milestone 2 defaults are still in the run. Every kind of
# variant runs at `a = 1`: a shape, the direction, a spread at `v0`, and `v_p`.
_REDUCED_CONFIG = dataclasses.replace(
    _FULL_CONFIG,
    n_samples=2000,
    speed_ratios=(0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0),
    a_values={
        _BASE_VARIANT: (0.0, 0.5, 1.0),
        "knee": (1.0,),
        "parabola": (1.0,),
        "one_sided_parabola": (1.0,),
        "a_x_eq_a_y": (1.0,),
        "std0_half": (1.0,),
        "std0_double": (1.0,),
        "v_p_3.0": (1.0,),
    },
)


def get_experiment_config(run_full: bool) -> ExperimentConfig:
    """
    Get how much of the speed sweeps to run.

    :param run_full: True for the full runs used in the paper, False for the
        reduced runs that take about a minute
    :return: the configuration
    """
    _LOG.debug(hprint.to_str("run_full"))
    hdbg.dassert_isinstance(run_full, bool, "run_full must be a bool")
    config = _FULL_CONFIG if run_full else _REDUCED_CONFIG
    return config


def count_maps(config: ExperimentConfig) -> int:
    """
    Count the maps of the sweeps of a run.

    :param config: how much to run
    :return: the number of maps, e.g., 768 for the full run and 180 for the
        reduced one
    """
    _LOG.debug(hprint.to_str("config"))
    # One sweep per sport, variant, and `a`, with one map per ball speed.
    n_a_values = sum(len(a_values) for a_values in config.a_values.values())
    n_maps = len(config.sports) * n_a_values * len(config.speed_ratios)
    return n_maps


# #############################################################################
# Sweeps
# #############################################################################


def build_sweep_spec(
    sport_name: str,
    variant: str,
    a: float,
    speed_ratio: float,
    config: ExperimentConfig,
) -> Dict[str, Any]:
    """
    Build one run of the sweeps: the builder inputs and the settings of a map.

    The settings are the ones of notebook 05, so the map is a run of its
    `compute_run()`: `std_x2`, `std_y2`, and the ball speed stay 3 independent
    inputs of the model (P1), and the builder only sets their values here.

    :param sport_name: "Tennis" or "Pickleball"
    :param variant: a name of `_VARIANTS`, e.g., "base" or "knee"
    :param a: coefficient of the depth factor, at least 0; the width gets
        `a_x_fraction * a`
    :param speed_ratio: ball speed as `v_b / v0`, e.g., 1.5
    :param config: how much to run, for the grid and the samples
    :return: the builder inputs and `settings`, the settings of the map, e.g.,
        `{"variant": "base", "shape": "linear", "a": 1.0, "a_x": 0.5, "a_y":
        1.0, "v0": 25.0, "std0_x": 0.5, "std0_y": 1.0, "speed_ratio": 1.5,
        "settings": {"sport": "Tennis", ..., "std_x2": 0.625, "std_y2": 1.5,
        "ball_speed": 37.5, ...}}`
    """
    _LOG.debug(hprint.to_str("sport_name variant a speed_ratio"))
    hdbg.dassert_in(variant, _VARIANTS, "Unknown variant")
    hdbg.dassert_in(sport_name, config.grid_size, "No grid for the sport")
    # The defaults of notebook 05 on the grid of `config`: their ball speed is
    # `v0`, and their spread is the spread at `v0`.
    settings = rosfrsnrs0vsu.get_default_settings(
        sport_name, n_samples=config.n_samples
    )
    n_x, n_y = config.grid_size[sport_name]
    settings.update(n_x=n_x, n_y=n_y)
    v0 = settings["ball_speed"]
    # The base values of what a variant can change, then the variant.
    inputs: Dict[str, Any] = {
        "shape": "linear",
        "a_x_fraction": 0.5,
        "std0_x": settings["std_x2"],
        "std0_y": settings["std_y2"],
        "v_p": settings["v_p"],
        "serve_name": "body-mid",
        "seed": settings["seed"],
    }
    inputs.update(_VARIANTS[variant])
    # Rounding to 1e-6 m/s keeps the settings string readable, e.g., 8.4 m/s
    # for `0.7 * 12` and not 8.399999999999999.
    ball_speed = round(speed_ratio * v0, 6)
    # The builder turns the ball speed into the 2 spreads of this map.
    a_x = inputs["a_x_fraction"] * a
    a_y = a
    std_x2, std_y2 = build_spread_from_ball_speed(
        ball_speed,
        inputs["std0_x"],
        inputs["std0_y"],
        v0,
        a_x,
        a_y,
        inputs["shape"],
    )
    # The serve comes from notebook 05 by name, on the default side.
    ((_, x1, y1),) = rosfrsnrs0vsu.get_serves(
        sport_name, settings["side"], (inputs["serve_name"],)
    )
    settings.update(
        x1=x1,
        y1=y1,
        std_x2=std_x2,
        std_y2=std_y2,
        ball_speed=ball_speed,
        v_p=inputs["v_p"],
        seed=inputs["seed"],
    )
    spec: Dict[str, Any] = {
        "variant": variant,
        "shape": inputs["shape"],
        "a": a,
        "a_x": a_x,
        "a_y": a_y,
        "v0": v0,
        "std0_x": inputs["std0_x"],
        "std0_y": inputs["std0_y"],
        "speed_ratio": speed_ratio,
        "settings": settings,
    }
    return spec


def run_speed_sweeps(config: ExperimentConfig) -> pd.DataFrame:
    """
    Run the sweeps: every ball speed, for each sport, variant, and `a`.

    All the maps of a sweep use the same seed: common random numbers, so a
    change between 2 ball speeds comes from the speed and its spread, and not
    from sampling noise.

    :param config: how much to run
    :return: one row per map, with:
        - The builder inputs of `build_sweep_spec()`: `variant`, `shape`, `a`,
          `a_x`, `a_y`, `v0`, `std0_x`, `std0_y`, and `speed_ratio`
        - The settings of the map, one column each, e.g., `ball_speed`,
          `std_x2`, and `std_y2`
        - The summary of `compute_run()` of notebook 05, e.g., `max_p_winner`;
          its `settings` column is the settings string that rebuilds the map
    """
    _LOG.debug(hprint.to_str("config"))
    specs = [
        build_sweep_spec(sport_name, variant, a, speed_ratio, config)
        for sport_name in config.sports
        for variant, a_values in config.a_values.items()
        for a in a_values
        for speed_ratio in config.speed_ratios
    ]
    rows = []
    for spec in tqdm(specs, desc="speed sweeps"):
        summary, _ = rosfrsnrs0vsu.compute_run(spec["settings"])
        # The settings string of the summary takes the place of the settings.
        row = {name: value for name, value in spec.items() if name != "settings"}
        row.update(spec["settings"])
        row.update(summary)
        rows.append(row)
    sweeps = pd.DataFrame(rows)
    return sweeps
