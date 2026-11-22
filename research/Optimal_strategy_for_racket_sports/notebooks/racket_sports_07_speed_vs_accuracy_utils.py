"""
Utilities for the `racket_sports_07_speed_vs_accuracy` notebook.

The spread of the return as a function of the ball speed (#723). The Milestone 2
model keeps `std_x2`, `std_y2`, and the ball speed as 3 independent inputs
(design principle P1): `build_spread_from_ball_speed()` computes the 2 spreads
from the ball speed with one model, for the sweeps only.

- The spread model: `compute_speed_factor()`, `build_spread_from_ball_speed()`
- The sweeps: the ball speed for each coefficient `a`, then one variable at a
  time; every map is a run of `compute_run()` of the notebook 05 utils
- The best ball speed as a band of tied speeds, and what it is robust to
- The speed scaling check: Proposition 1 with a spread that depends on the
  ball speed
- The threshold `a*`: the smallest `a` at which hitting at `2 v0` stops being
  among the best speeds, on a fine grid of `a`
- The plots of the notebook, and the data of the paper's figure

Import as:

import research.Optimal_strategy_for_racket_sports.notebooks.racket_sports_07_speed_vs_accuracy_utils as rosfrsnrs0svau
"""

import dataclasses
import logging
import os
from typing import Any, Dict, List, Sequence, Tuple

import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebo
import helpers.hprint as hprint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
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


# One style per shape: a color, and a marker and a line style as a second
# encoding.
_SHAPE_STYLES = {
    "linear": {"color": "#2a78d6", "marker": "o", "linestyle": "-"},
    "knee": {"color": "#eb6834", "marker": "s", "linestyle": "--"},
    "parabola": {"color": "#1baf7a", "marker": "^", "linestyle": ":"},
    "one_sided_parabola": {"color": "#8e5cc4", "marker": "D", "linestyle": "-."},
}


def plot_speed_factors(a_values: Sequence[float]) -> Figure:
    """
    Plot the factor of the spread against the ball speed, for each shape.

    One panel per coefficient `a`. The factor depends on the speeds only
    through `v_b / v0`, so the x axis is that ratio, for every sport.

    :param a_values: coefficients, one panel each, e.g., `[0.5, 1.0]`
    :return: the figure
    """
    _LOG.debug(hprint.to_str("a_values"))
    hdbg.dassert_lte(1, len(a_values), "There must be at least 1 value of a")
    speed_ratios = np.linspace(0.5, 2.0, 151)
    fig: Figure
    fig, axes = plt.subplots(
        1, len(a_values), figsize=(5 * len(a_values), 3.8), squeeze=False
    )
    for ax, a in zip(axes[0], a_values):
        for shape in SPREAD_SHAPES:
            # With `v0 = 1` the ball speed is the ratio itself.
            factors = [
                compute_speed_factor(ratio, 1.0, a, shape)
                for ratio in speed_ratios
            ]
            style = _SHAPE_STYLES[shape]
            ax.plot(
                speed_ratios,
                factors,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2,
                label=shape,
            )
        # At `v0` every shape gives the spread at `v0`.
        ax.axvline(1.0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_xlabel("ball speed v_b / v0")
        ax.set_ylabel("factor f of the spread at v0")
        ax.set_title(f"a = {a:g}", fontsize=11)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=9, frameon=False)
    fig.suptitle(
        "The spread grows with the ball speed: f = 1 + a g(u), u = (v_b - v0)"
        " / v0",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


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
    # Values of `a` of the speed scaling check.
    scaling_a_values: Tuple[float, ...]


# 786 maps: 768 for the sweeps and 18 for the speed scaling check. The range of
# `a`, from 0 (the Milestone 2 model) to 1, is our assumption: no study gives a
# value. The variants run at `a` = 0.5 and 1, and the second seed only at
# `a = 1`.
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
    scaling_a_values=(0.0, 0.5, 1.0),
)
# 198 maps: 180 for the sweeps and the same 18 for the speed scaling check. The
# speeds keep both ends and `v0`, so the loss of the fastest ball and the map of
# the Milestone 2 defaults are still in the run. Every kind of variant runs at
# `a = 1`: a shape, the direction, a spread at `v0`, and `v_p`.
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
    Count the maps of a run: the sweeps and the speed scaling check.

    :param config: how much to run
    :return: the number of maps, e.g., 786 for the full run and 198 for the
        reduced one
    """
    _LOG.debug(hprint.to_str("config"))
    # One sweep per sport, variant, and `a`, with one map per ball speed.
    n_a_values = sum(len(a_values) for a_values in config.a_values.values())
    n_sweep_maps = len(config.sports) * n_a_values * len(config.speed_ratios)
    # A check compares 3 maps: the base, both speeds scaled, and `v0` scaled too.
    n_scaling_maps = 3 * len(config.sports) * len(config.scaling_a_values)
    n_maps = n_sweep_maps + n_scaling_maps
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
        - `mean_p_winner_se`: the mean of the standard errors of the cells,
          an upper bound of the standard error of `mean_p_winner`
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
        summary, heatmap = rosfrsnrs0vsu.compute_run(spec["settings"])
        # The settings string of the summary takes the place of the settings.
        row = {name: value for name, value in spec.items() if name != "settings"}
        row.update(spec["settings"])
        row.update(summary)
        # The standard deviation of a mean is at most the mean of the standard
        # deviations, whatever the correlation of the cells, which share their
        # draws.
        row["mean_p_winner_se"] = float(heatmap["p_winner_se"].mean())
        rows.append(row)
    sweeps = pd.DataFrame(rows)
    return sweeps


def plot_sweep_maps(
    sport_name: str,
    variant: str,
    a: float,
    speed_ratios: Sequence[float],
    config: ExperimentConfig,
) -> Figure:
    """
    Plot the maps of one sweep at a few ball speeds, side by side.

    Each map is the one of `run_speed_sweeps()` at that ball speed, drawn by
    `plot_montage()` of notebook 05.

    :param sport_name: "Tennis" or "Pickleball"
    :param variant: a name of `_VARIANTS`, e.g., "base"
    :param a: coefficient of the depth factor
    :param speed_ratios: ball speeds as `v_b / v0`, one map each, e.g.,
        `[0.5, 1.0, 2.0]`
    :param config: how much to run, for the grid and the samples
    :return: the figure
    """
    _LOG.debug(hprint.to_str("sport_name variant a speed_ratios"))
    specs = [
        build_sweep_spec(sport_name, variant, a, speed_ratio, config)
        for speed_ratio in speed_ratios
    ]
    settings_list = [spec["settings"] for spec in specs]
    titles = [
        f"v_b = {spec['speed_ratio']:g} v0 = {spec['settings']['ball_speed']:g}"
        f" m/s\nspread ({spec['settings']['std_x2']:.2f},"
        f" {spec['settings']['std_y2']:.2f}) m"
        for spec in specs
    ]
    suptitle = (
        f"{sport_name}: {variant} at a = {a:g}, the spread from the ball speed,"
        " other settings at the defaults"
    )
    fig: Figure = rosfrsnrs0vsu.plot_montage(settings_list, titles, suptitle)
    return fig


# #############################################################################
# Best ball speed
# #############################################################################


def _build_speed_map(
    runs: pd.DataFrame, value_col: str, se_col: str
) -> pd.DataFrame:
    """
    Build a map with one row of cells from the ball speeds of a sweep.

    The rule of the best region of notebook 01 then applies to the speeds as it
    is: `x_m` is the speed as `v_b / v0`, and `p_winner` the value that ranks
    the speeds.

    :param runs: the rows of one sweep of `run_speed_sweeps()`, sorted by
        `speed_ratio`
    :param value_col: the column of the value, e.g., "max_p_winner"
    :param se_col: the column of its standard error, e.g., "max_p_winner_se"
    :return: one row per speed, with `x_m`, `y_m`, `p_winner`, `p_winner_se`
    """
    _LOG.debug(hprint.to_str("value_col se_col"))
    for column in (value_col, se_col):
        hdbg.dassert_in(column, runs.columns, "Missing column")
    speed_map = pd.DataFrame(
        {
            "x_m": runs["speed_ratio"].to_numpy(),
            "y_m": 0.0,
            "p_winner": runs[value_col].to_numpy(dtype=float),
            "p_winner_se": runs[se_col].to_numpy(dtype=float),
        }
    )
    return speed_map


def find_best_speeds(sweeps: pd.DataFrame) -> pd.DataFrame:
    """
    Find the best ball speed of each sweep, as a band of tied speeds.

    The value of a ball speed is the highest `P(winner)` of its map. Many
    speeds can tie at the top, e.g., at 1.000 in tennis, so the best speed is a
    band: every speed within 2 standard errors of the highest value, with the
    rule of `find_best_region()` of notebook 01. The marked speed is the speed
    of the band closest to its middle, as for the best region. The maps of a
    sweep share their seed, so their values move together and the band is
    wider than it needs to be.

    :param sweeps: the rows of `run_speed_sweeps()`, with at least 2 ball
        speeds per sport, variant, and `a`
    :return: one row per sport, variant, and `a`, with:
        - `best_speed_ratio`, `best_ball_speed`: the marked speed, as
          `v_b / v0` and in m/s
        - `band_low_ratio`, `band_high_ratio`, `band_ratios`, `n_band`: the
          ends of the band, its speeds, and how many they are, of `n_speeds`
        - `slowest_in_band`, `fastest_in_band`: True if the band reaches that
          end of the sweep, so the best speed may lie beyond it
        - `top_p_winner`: the highest value over the speeds
        - `fastest_p_winner`, `fastest_loss_points`: the value of the fastest
          ball, and what it loses against the highest, in percentage points
        - `best_mean_speed_ratio`, `best_mean_p_winner`: the speed with the
          highest mean `P(winner)` over the half, which does not saturate
        - `fastest_mean_loss_points`: what the fastest ball loses on that mean
    """
    _LOG.debug(hprint.to_str("sweeps.shape"))
    hdbg.dassert_lte(1, len(sweeps), "There must be at least 1 run")
    rows = []
    for (sport, variant, a), runs in sweeps.groupby(
        ["sport", "variant", "a"], sort=False
    ):
        runs = runs.sort_values("speed_ratio")
        hdbg.dassert_lte(2, len(runs), "A sweep needs at least 2 ball speeds")
        # The speeds of a sweep form a map with one row of cells, so the rule
        # of the best region applies to them as it is: `x_m` is the speed.
        speed_map = _build_speed_map(runs, "max_p_winner", "max_p_winner_se")
        in_band = rosfrsnrs0spu.find_best_region(speed_map)
        best = rosfrsnrs0spu.find_best_target(speed_map)
        band = runs[in_band]
        marked = runs[runs["speed_ratio"] == best["x_m"]].iloc[0]
        # The fastest ball, and the speed with the highest mean over the half.
        fastest = runs.iloc[-1]
        best_mean = runs.loc[runs["mean_p_winner"].idxmax()]
        rows.append(
            {
                "sport": sport,
                "variant": variant,
                "a": a,
                "best_speed_ratio": marked["speed_ratio"],
                "best_ball_speed": marked["ball_speed"],
                "band_low_ratio": band["speed_ratio"].min(),
                "band_high_ratio": band["speed_ratio"].max(),
                "band_ratios": tuple(band["speed_ratio"].tolist()),
                "n_band": len(band),
                "n_speeds": len(runs),
                "slowest_in_band": bool(in_band[0]),
                "fastest_in_band": bool(in_band[-1]),
                "top_p_winner": best["max_p_winner"],
                "fastest_p_winner": fastest["max_p_winner"],
                "fastest_loss_points": 100
                * (best["max_p_winner"] - fastest["max_p_winner"]),
                "best_mean_speed_ratio": best_mean["speed_ratio"],
                "best_mean_p_winner": best_mean["mean_p_winner"],
                "fastest_mean_loss_points": 100
                * (best_mean["mean_p_winner"] - fastest["mean_p_winner"]),
            }
        )
    best_speeds = pd.DataFrame(rows)
    return best_speeds


def summarize_robustness(best_speeds: pd.DataFrame) -> pd.DataFrame:
    """
    Compare the band of each variant with the band of the base.

    The best speed is robust to a variant if the 2 bands share at least one
    ball speed, for the same sport and `a`: the variant does not move the best
    speed by more than the sampling noise can explain (P3).

    :param best_speeds: the rows of `find_best_speeds()`, with the base rows
        of every sport and `a` of the variants
    :return: one row per sport, variant, and `a`, without the base, with:
        - `base_band_low_ratio`, `base_band_high_ratio`: the band of the base
        - `band_low_ratio`, `band_high_ratio`: the band of the variant
        - `n_shared`, `overlaps`: the ball speeds in both bands, and whether
          there is at least one
        - `d_best_speed_ratio`: how far the marked speed moves, as `v_b / v0`
        - `d_fastest_loss_points`: how much more the fastest ball loses
    """
    _LOG.debug(hprint.to_str("best_speeds.shape"))
    hdbg.dassert_lte(1, len(best_speeds), "There must be at least 1 sweep")
    is_base = best_speeds["variant"] == _BASE_VARIANT
    base_rows = best_speeds[is_base].set_index(["sport", "a"])
    rows = []
    for _, row in best_speeds[~is_base].iterrows():
        key = (row["sport"], row["a"])
        hdbg.dassert_in(key, base_rows.index, "The variant has no base sweep")
        base = base_rows.loc[key]
        # Both sweeps run the same ball speeds, so the bands compare as sets.
        n_shared = len(set(row["band_ratios"]) & set(base["band_ratios"]))
        rows.append(
            {
                "sport": row["sport"],
                "variant": row["variant"],
                "a": row["a"],
                "base_band_low_ratio": base["band_low_ratio"],
                "base_band_high_ratio": base["band_high_ratio"],
                "band_low_ratio": row["band_low_ratio"],
                "band_high_ratio": row["band_high_ratio"],
                "n_shared": n_shared,
                "overlaps": n_shared > 0,
                "d_best_speed_ratio": row["best_speed_ratio"]
                - base["best_speed_ratio"],
                "d_fastest_loss_points": row["fastest_loss_points"]
                - base["fastest_loss_points"],
            }
        )
    robustness = pd.DataFrame(rows)
    return robustness


def _get_a_color(a: float) -> Tuple[float, float, float, float]:
    """
    Get the color of a coefficient `a`, from light (0) to dark (1 or more).

    :param a: coefficient, at least 0
    :return: an RGBA color of the "viridis" colormap
    """
    cmap = plt.get_cmap("viridis_r")
    color: Tuple[float, float, float, float] = cmap(0.15 + 0.8 * min(a, 1.0))
    return color


def plot_speed_sweeps(
    sweeps: pd.DataFrame, best_speeds: pd.DataFrame, variant: str
) -> Figure:
    """
    Plot the highest and the mean `P(winner)` against the ball speed.

    One row per sport and one curve per `a`. The left panel has a 2 SE band
    around each curve; a filled marker is a speed of the band of the best
    speed, a hollow one a speed outside it, and the star the marked speed.

    :param sweeps: the rows of `run_speed_sweeps()`
    :param best_speeds: the rows of `find_best_speeds()` of the same sweeps
    :param variant: the variant to plot, e.g., "base"
    :return: the figure
    """
    _LOG.debug(hprint.to_str("variant"))
    hdbg.dassert_in(variant, set(sweeps["variant"]), "Unknown variant")
    runs = sweeps[sweeps["variant"] == variant]
    sports = list(runs["sport"].unique())
    fig: Figure
    fig, axes = plt.subplots(
        len(sports), 2, figsize=(13, 4 * len(sports)), squeeze=False
    )
    for i, sport_name in enumerate(sports):
        for a, grp in runs[runs["sport"] == sport_name].groupby("a"):
            grp = grp.sort_values("speed_ratio")
            best = best_speeds[
                (best_speeds["sport"] == sport_name)
                & (best_speeds["variant"] == variant)
                & (best_speeds["a"] == a)
            ].iloc[0]
            x = grp["speed_ratio"].to_numpy()
            p = grp["max_p_winner"].to_numpy()
            se = grp["max_p_winner_se"].to_numpy()
            color = _get_a_color(float(a))
            in_band = np.isin(x, best["band_ratios"])
            # Left: the highest `P(winner)` with its band of tied speeds.
            ax = axes[i, 0]
            ax.plot(x, p, color=color, linewidth=1.5, label=f"a = {a:g}")
            ax.fill_between(x, p - 2 * se, p + 2 * se, color=color, alpha=0.15)
            ax.plot(x[in_band], p[in_band], "o", color=color, ms=5)
            ax.plot(
                x[~in_band],
                p[~in_band],
                "o",
                markerfacecolor="white",
                color=color,
                ms=5,
            )
            ax.plot(
                best["best_speed_ratio"],
                p[x == best["best_speed_ratio"]][0],
                "*",
                color=color,
                ms=14,
                markeredgecolor="black",
            )
            # Right: the mean over the half, which does not saturate.
            axes[i, 1].plot(
                x, grp["mean_p_winner"], "o-", color=color, linewidth=1.5, ms=4
            )
        for j, label in enumerate(
            ("highest P(winner), +- 2 SE", "mean P(winner) over the half")
        ):
            axes[i, j].axvline(1.0, color="gray", linewidth=0.8, linestyle=":")
            axes[i, j].set_xlabel("ball speed v_b / v0")
            axes[i, j].set_ylabel(label)
            axes[i, j].set_title(sport_name, fontsize=11)
            axes[i, j].grid(alpha=0.3)
        axes[i, 0].legend(fontsize=9, frameon=False)
    fig.suptitle(
        f"{variant}: the best ball speed when speed costs accuracy (star: the"
        " marked speed, filled: its band)",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


# The variants of the paper's figure: the base and the 3 other shapes.
_FIGURE_VARIANTS = (_BASE_VARIANT, "knee", "parabola", "one_sided_parabola")


def plot_best_speeds(best_speeds: pd.DataFrame) -> Figure:
    """
    Plot the best ball speed against `a`, for the base and the other shapes.

    One panel per sport. The base is a line with its band shaded; each other
    shape is a marker with its band as an error bar, shifted a little in `a`
    so the bars do not hide each other.

    :param best_speeds: the rows of `find_best_speeds()`, with the base
    :return: the figure
    """
    _LOG.debug(hprint.to_str("best_speeds.shape"))
    hdbg.dassert_in(
        _BASE_VARIANT, set(best_speeds["variant"]), "The base is missing"
    )
    sports = list(best_speeds["sport"].unique())
    fig: Figure
    fig, axes = plt.subplots(
        1, len(sports), figsize=(6 * len(sports), 4.2), squeeze=False
    )
    for ax, sport_name in zip(axes[0], sports):
        rows = best_speeds[best_speeds["sport"] == sport_name]
        for k, variant in enumerate(_FIGURE_VARIANTS):
            grp = rows[rows["variant"] == variant].sort_values("a")
            if grp.empty:
                continue
            shape = "linear" if variant == _BASE_VARIANT else variant
            style = _SHAPE_STYLES[shape]
            a_values = grp["a"].to_numpy()
            best = grp["best_speed_ratio"].to_numpy()
            low = grp["band_low_ratio"].to_numpy()
            high = grp["band_high_ratio"].to_numpy()
            if variant == _BASE_VARIANT:
                ax.plot(
                    a_values,
                    best,
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=2,
                    label="linear (base)",
                )
                ax.fill_between(
                    a_values, low, high, color=style["color"], alpha=0.15
                )
            else:
                shift = 0.02 * k
                ax.errorbar(
                    a_values + shift,
                    best,
                    yerr=[best - low, high - best],
                    color=style["color"],
                    marker=style["marker"],
                    linestyle="none",
                    capsize=3,
                    label=variant,
                )
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_xlabel("coefficient a of the spread")
        ax.set_ylabel("best ball speed v_b / v0")
        ax.set_ylim(0.45, 2.05)
        ax.set_title(sport_name, fontsize=11)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=9, frameon=False)
    fig.suptitle(
        "Best ball speed against a, with its band of tied speeds", fontsize=12
    )
    fig.tight_layout()
    return fig


# #############################################################################
# Speed scaling check
# #############################################################################


def _compute_scaled_map(
    spec: Dict[str, Any], k_speeds: float, k_v0: float
) -> pd.DataFrame:
    """
    Compute the map of a run with its speeds and its `v0` multiplied.

    The builder computes the spread again, from the new ball speed and `v0`.

    :param spec: a run, from `build_sweep_spec()`
    :param k_speeds: factor of the ball speed and of the speed of player 1
    :param k_v0: factor of the reference ball speed `v0`
    :return: the map of `compute_run()`
    """
    settings = spec["settings"]
    ball_speed = settings["ball_speed"] * k_speeds
    std_x2, std_y2 = build_spread_from_ball_speed(
        ball_speed,
        spec["std0_x"],
        spec["std0_y"],
        spec["v0"] * k_v0,
        spec["a_x"],
        spec["a_y"],
        spec["shape"],
    )
    scaled = dict(
        settings,
        ball_speed=ball_speed,
        v_p=settings["v_p"] * k_speeds,
        std_x2=std_x2,
        std_y2=std_y2,
    )
    _, heatmap = rosfrsnrs0vsu.compute_run(scaled)
    return heatmap


def check_speed_scaling(spec: Dict[str, Any], k: float) -> Dict[str, Any]:
    """
    Compare the map of a run with 2 maps whose speeds are multiplied by `k`.

    With a spread from the builder, the landing points depend on the ball
    speed through `v_b / v0`, and player 1 reaches a landing point through
    `v_p / v_b` and the reaction distance `v_p * t_r`. So, with `t_r = 0`:
    - Multiplying `v_p` and `v_b` by `k` changes `v_b / v0`: the maps differ
      when `a > 0`, and Proposition 1 needs a spread that does not depend on
      the ball speed
    - Multiplying `v_p`, `v_b`, and `v0` by `k` changes neither ratio: the
      maps are identical, for every shape and any `a_x`, `a_y`

    :param spec: a run, from `build_sweep_spec()`
    :param k: the factor, positive; a power of 2 scales the speeds exactly,
        e.g., 2.0
    :return: `max_diff_speeds`, the largest cell difference with both speeds
        scaled, over `p_winner`, `p_winner_se`, `p_in`, and `p_reached`;
        `max_diff_speeds_and_v0`, the same with `v0` scaled too; `k`; the
        settings string of the run; and `mean_p_reached`, which shows that the
        reach rule is exercised
    """
    _LOG.debug(hprint.to_str("k"))
    hdbg.dassert(np.isfinite(k), "k=%s must be finite", k)
    hdbg.dassert_lt(0, k, "k must be positive")
    # Same seed and same grid: the 3 maps line up cell by cell.
    summary, heatmap = rosfrsnrs0vsu.compute_run(spec["settings"])
    scaled_maps = {
        "speeds": _compute_scaled_map(spec, k, 1.0),
        "speeds_and_v0": _compute_scaled_map(spec, k, k),
    }
    row: Dict[str, Any] = {"settings": summary["settings"], "k": k}
    columns = ["p_winner", "p_winner_se", "p_in", "p_reached"]
    for name, scaled_map in scaled_maps.items():
        diff = (heatmap[columns] - scaled_map[columns]).abs()
        row[f"max_diff_{name}"] = float(diff.to_numpy().max())
    row["mean_p_reached"] = float(heatmap["p_reached"].mean())
    return row


def run_scaling_checks(config: ExperimentConfig) -> pd.DataFrame:
    """
    Run `check_speed_scaling()` on the base of each sport, for each `a`.

    Each run is the base at `v_b = v0` with `t_r = 0`, and both speeds are
    doubled, so the scaled ball is at `2 v0`, where the spread is `1 + a`
    times larger unless `v0` is doubled too.

    :param config: how much to run
    :return: one row per sport and `a`, with `sport`, `a`, and the output of
        `check_speed_scaling()`
    """
    _LOG.debug(hprint.to_str("config"))
    rows = []
    for sport_name in config.sports:
        for a in config.scaling_a_values:
            spec = build_sweep_spec(sport_name, _BASE_VARIANT, a, 1.0, config)
            spec["settings"]["t_r"] = 0.0
            row = {"sport": sport_name, "a": a}
            row.update(check_speed_scaling(spec, 2.0))
            rows.append(row)
    scaling_checks = pd.DataFrame(rows)
    return scaling_checks


# #############################################################################
# Threshold of `a`
# #############################################################################


@dataclasses.dataclass(frozen=True)
class ThresholdConfig:
    """
    How much of the threshold run to run: the sweeps, then the seed runs.
    """

    # The sweeps on a fine grid of `a`, without the speed scaling check.
    sweeps: ExperimentConfig
    # The sport whose maps run again with other seeds, for the standard error
    # of their mean `P(winner)`.
    seed_sport: str
    # The variants of those maps; the other variants keep the bound of the
    # cells.
    seed_variants: Tuple[str, ...]
    # The seeds of those runs, besides seed 1 of the sweeps.
    extra_seeds: Tuple[int, ...]


# The fine grid of `a`, in steps of 0.025: up to 0.5 for every shape, and up to
# 1 for the sweeps of the linear shape, where `v0` itself leaves the band of the
# best speed only above 0.5 in pickleball.
_A_UP_TO_HALF = tuple(step / 40 for step in range(21))
_A_UP_TO_ONE = tuple(step / 40 for step in range(41))
# The 4 shapes, whose tennis maps run with the extra seeds.
_SHAPE_VARIANTS = (_BASE_VARIANT, "knee", "parabola", "one_sided_parabola")
# 15,232 maps: 8,576 for the sweeps, the 4 shapes up to `a = 0.5` and the base
# with its 4 variants of the spread at `v0` and `v_p` up to `a = 1`, on the 16
# ball speeds of the full run; then 6,656 for the tennis maps of the 4 shapes
# with seeds 2 to 5.
_THRESHOLD_FULL_CONFIG = ThresholdConfig(
    sweeps=dataclasses.replace(
        _FULL_CONFIG,
        a_values={
            _BASE_VARIANT: _A_UP_TO_ONE,
            "knee": _A_UP_TO_HALF,
            "parabola": _A_UP_TO_HALF,
            "one_sided_parabola": _A_UP_TO_HALF,
            "std0_half": _A_UP_TO_ONE,
            "std0_double": _A_UP_TO_ONE,
            "v_p_1.9": _A_UP_TO_ONE,
            "v_p_3.0": _A_UP_TO_ONE,
        },
        scaling_a_values=(),
    ),
    seed_sport="Tennis",
    seed_variants=_SHAPE_VARIANTS,
    extra_seeds=(2, 3, 4, 5),
)
# 280 maps: 140 for the sweeps of the 4 shapes, on 5 ball speeds with `v0` and
# both ends, in steps of 0.25 of `a` and with the base up to `a = 1`; then 140
# for the tennis maps with seeds 2 and 3. The variants of the base run only in
# the full run.
_THRESHOLD_REDUCED_CONFIG = ThresholdConfig(
    sweeps=dataclasses.replace(
        _REDUCED_CONFIG,
        speed_ratios=(0.5, 0.8, 1.0, 1.5, 2.0),
        a_values={
            _BASE_VARIANT: (0.0, 0.25, 0.5, 0.75, 1.0),
            "knee": (0.0, 0.25, 0.5),
            "parabola": (0.0, 0.25, 0.5),
            "one_sided_parabola": (0.0, 0.25, 0.5),
        },
        scaling_a_values=(),
    ),
    seed_sport="Tennis",
    seed_variants=_SHAPE_VARIANTS,
    extra_seeds=(2, 3),
)


def get_threshold_config(run_full: bool) -> ThresholdConfig:
    """
    Get how much of the threshold run to run.

    :param run_full: True for the full run of the threshold table, False for
        the reduced run that takes about a minute
    :return: the configuration
    """
    _LOG.debug(hprint.to_str("run_full"))
    hdbg.dassert_isinstance(run_full, bool, "run_full must be a bool")
    config = _THRESHOLD_FULL_CONFIG if run_full else _THRESHOLD_REDUCED_CONFIG
    return config


def count_threshold_maps(config: ThresholdConfig) -> int:
    """
    Count the maps of the threshold run: the sweeps and the seed runs.

    :param config: how much to run
    :return: the number of maps, e.g., 15,232 for the full run and 280 for the
        reduced one
    """
    _LOG.debug(hprint.to_str("config.extra_seeds"))
    n_sweep_maps = count_maps(config.sweeps)
    # Each seed runs again the maps of one sport and of the seed variants.
    seed_a_values = {
        variant: config.sweeps.a_values[variant]
        for variant in config.seed_variants
    }
    seed_sweeps = dataclasses.replace(
        config.sweeps, sports=(config.seed_sport,), a_values=seed_a_values
    )
    n_seed_maps = len(config.extra_seeds) * count_maps(seed_sweeps)
    n_maps = n_sweep_maps + n_seed_maps
    return n_maps


def add_seed_mean_se(
    sweeps: pd.DataFrame, config: ThresholdConfig
) -> pd.DataFrame:
    """
    Estimate the standard error of the mean `P(winner)` of maps from seeds.

    The cells of a map share their standard normal draws, so their standard
    errors do not give the one of their mean. Each map of `config.seed_sport`
    and `config.seed_variants` runs again with each of `config.extra_seeds`,
    all else equal: with its own seed, that is `n = 1 + len(extra_seeds)`
    independent estimates of its mean.

    :param sweeps: the rows of `run_speed_sweeps()`, none with an extra seed
    :param config: the sport, the variants, and the seeds of the runs
    :return: a copy of `sweeps` with 2 columns, NaN for the other maps:
        - `mean_p_winner_seeds`: the average of the `n` estimates
        - `mean_p_winner_seeds_se`: their standard deviation over `sqrt(n)`,
          the standard error of that average
    """
    _LOG.debug(hprint.to_str("sweeps.shape config.extra_seeds"))
    hdbg.dassert_lte(1, len(config.extra_seeds), "There must be an extra seed")
    hdbg.dassert_no_duplicates(list(config.extra_seeds), "Repeated seeds")
    is_seeded = (sweeps["sport"] == config.seed_sport) & sweeps["variant"].isin(
        config.seed_variants
    )
    hdbg.dassert_lte(1, int(is_seeded.sum()), "No map to run again")
    estimates = []
    for _, row in tqdm(
        sweeps[is_seeded].iterrows(), total=int(is_seeded.sum()), desc="seeds"
    ):
        # The settings string rebuilds the map exactly, and only the seed
        # changes.
        settings = rosfrsnrs0vsu.parse_settings_string(row["settings"])
        hdbg.dassert_not_in(
            settings["seed"],
            config.extra_seeds,
            "The seed of a map must not be an extra seed",
        )
        means = [row["mean_p_winner"]]
        for seed in config.extra_seeds:
            summary, _ = rosfrsnrs0vsu.compute_run(dict(settings, seed=seed))
            means.append(summary["mean_p_winner"])
        estimates.append(means)
    # One row per map, one column per seed.
    estimates_array = np.array(estimates)
    n_seeds = estimates_array.shape[1]
    seeded = sweeps.copy()
    seeded["mean_p_winner_seeds"] = np.nan
    seeded["mean_p_winner_seeds_se"] = np.nan
    seeded.loc[is_seeded, "mean_p_winner_seeds"] = estimates_array.mean(axis=1)
    seeded.loc[is_seeded, "mean_p_winner_seeds_se"] = estimates_array.std(
        axis=1, ddof=1
    ) / np.sqrt(n_seeds)
    return seeded


def find_threshold_bands(sweeps: pd.DataFrame, tennis_se: str) -> pd.DataFrame:
    """
    Find the band of the best ball speed of each sweep, for the threshold.

    The band is every speed within 2 standard errors of the highest value, the
    rule of `find_best_region()` of notebook 01, as in `find_best_speeds()`
    (P9). The value is the highest cell in pickleball, as there, and the mean
    over the half in tennis, since its highest cell is 1.000 at almost every
    speed.

    :param sweeps: the rows of `run_speed_sweeps()`, with the columns of
        `add_seed_mean_se()` for "seeds"
    :param tennis_se: how tennis gets its value and its standard error:
        - "seeds": the average of the seeds, with the standard error from
          their spread
        - "cells": seed 1 alone, with the mean of the standard errors of the
          cells, an upper bound
    :return: one row per map, with `sport`, `variant`, `a`, `speed_ratio`,
        `tennis_se`, the value and its standard error as `value` and
        `value_se`, and `in_band`
    """
    _LOG.debug(hprint.to_str("sweeps.shape tennis_se"))
    # The columns of the value and of its standard error, per sport.
    metrics = {
        "Pickleball": {
            "seeds": ("max_p_winner", "max_p_winner_se"),
            "cells": ("max_p_winner", "max_p_winner_se"),
        },
        "Tennis": {
            "seeds": ("mean_p_winner_seeds", "mean_p_winner_seeds_se"),
            "cells": ("mean_p_winner", "mean_p_winner_se"),
        },
    }
    hdbg.dassert_in(tennis_se, ("seeds", "cells"), "Unknown method")
    hdbg.dassert_lte(1, len(sweeps), "There must be at least 1 run")
    parts = []
    for (sport, variant, a), runs in sweeps.groupby(
        ["sport", "variant", "a"], sort=False
    ):
        runs = runs.sort_values("speed_ratio")
        hdbg.dassert_lte(2, len(runs), "A sweep needs at least 2 ball speeds")
        hdbg.dassert_in(sport, metrics, "Unknown sport")
        value_col, se_col = metrics[sport][tennis_se]
        speed_map = _build_speed_map(runs, value_col, se_col)
        hdbg.dassert(
            speed_map.notna().all().all(),
            "Missing values: sport=%s variant=%s a=%s",
            sport,
            variant,
            a,
        )
        parts.append(
            pd.DataFrame(
                {
                    "sport": sport,
                    "variant": variant,
                    "a": a,
                    "speed_ratio": speed_map["x_m"],
                    "tennis_se": tennis_se,
                    "value": speed_map["p_winner"],
                    "value_se": speed_map["p_winner_se"],
                    "in_band": rosfrsnrs0spu.find_best_region(speed_map),
                }
            )
        )
    bands = pd.concat(parts, ignore_index=True)
    return bands


def find_speed_thresholds(bands: pd.DataFrame) -> pd.DataFrame:
    """
    Find the 2 thresholds of `a` of each sport and variant.

    - `a*`: the smallest `a` at which the fastest ball, `2 v0`, is out of the
      band of the best speed; below it, hitting as hard as possible is among
      the best speeds
    - `a**`: the smallest `a` at which every speed of the band is slower than
      `v0`, so `v0` itself is out of the band, and not because faster balls
      win

    On a grid of `a`, the true threshold is between the value of `a` before
    the one found and the one found.

    :param bands: the rows of `find_threshold_bands()`, with `v0` and the same
        ball speeds at every `a`
    :return: one row per sport and variant, with:
        - `tennis_se`, `n_a`, `a_max`: the method, how many values of `a`, and
          the largest
        - `a_star`, NaN if the fastest ball is in the band up to `a_max`
        - `a_star_prev`: the value of `a` before `a_star`, NaN if `a_star` is
          the first
        - `fastest_reenters`: True if the fastest ball is in the band again at
          a larger `a`
        - `band_low_ratio_at_a_star`, `band_high_ratio_at_a_star`: the band at
          `a_star`
        - `a_star_star`, NaN if the band reaches `v0` up to `a_max`
    """
    _LOG.debug(hprint.to_str("bands.shape"))
    hdbg.dassert_lte(1, len(bands), "There must be at least 1 map")
    rows = []
    for (sport, variant), grp in bands.groupby(["sport", "variant"], sort=False):
        a_values = np.sort(grp["a"].unique())
        speed_ratios = grp["speed_ratio"].unique()
        hdbg.dassert_in(1.0, speed_ratios, "The sweeps must have v0")
        hdbg.dassert_eq(
            len(grp),
            len(a_values) * len(speed_ratios),
            "Every a must have the same ball speeds",
        )
        # Per `a`: whether the fastest ball is in the band, and the band.
        is_fastest = grp["speed_ratio"] == speed_ratios.max()
        fastest_in = grp[is_fastest].set_index("a")["in_band"].loc[a_values]
        band = grp[grp["in_band"]].groupby("a")["speed_ratio"]
        band_low = band.min().loc[a_values].to_numpy()
        band_high = band.max().loc[a_values].to_numpy()
        # The first `a` with the fastest ball out, and with the band below `v0`.
        i_out = np.flatnonzero(~fastest_in.to_numpy(dtype=bool))
        i_below = np.flatnonzero(band_high < 1.0)
        has_star = i_out.size > 0
        i_star = int(i_out[0]) if has_star else -1
        # The fastest ball is in the band again if some `a` from `a*` on does
        # not have it out.
        reenters = has_star and i_out.size < len(a_values) - i_star
        rows.append(
            {
                "sport": sport,
                "variant": variant,
                "tennis_se": grp["tennis_se"].iloc[0],
                "n_a": len(a_values),
                "a_max": a_values[-1],
                "a_star": a_values[i_star] if has_star else np.nan,
                "a_star_prev": a_values[i_star - 1] if i_star > 0 else np.nan,
                "fastest_reenters": bool(reenters),
                "band_low_ratio_at_a_star": (
                    band_low[i_star] if has_star else np.nan
                ),
                "band_high_ratio_at_a_star": (
                    band_high[i_star] if has_star else np.nan
                ),
                "a_star_star": (
                    a_values[i_below[0]] if i_below.size > 0 else np.nan
                ),
            }
        )
    thresholds = pd.DataFrame(rows)
    return thresholds


def compare_tennis_se(
    thresholds_seeds: pd.DataFrame, thresholds_cells: pd.DataFrame
) -> pd.DataFrame:
    """
    Put the thresholds of the 2 standard errors of the tennis mean side by
    side.

    :param thresholds_seeds: the rows of `find_speed_thresholds()` with the
        standard error from the seeds
    :param thresholds_cells: the rows of the same sweeps with the bound of the
        cells
    :return: the rows of `thresholds_seeds`, without `tennis_se`, and with
        `a_star_cells` and `a_star_star_cells`, the thresholds with the bound
        of the cells; in pickleball they are the same, since it uses the
        highest cell
    """
    _LOG.debug(hprint.to_str("thresholds_seeds.shape thresholds_cells.shape"))
    for thresholds, tennis_se in (
        (thresholds_seeds, "seeds"),
        (thresholds_cells, "cells"),
    ):
        hdbg.dassert_eq(
            set(thresholds["tennis_se"]), {tennis_se}, "Wrong method"
        )
    keys = ["sport", "variant"]
    hdbg.dassert_set_eq(
        set(thresholds_seeds[keys].itertuples(index=False)),
        set(thresholds_cells[keys].itertuples(index=False)),
        "The 2 tables must have the same sweeps",
    )
    cells = thresholds_cells[keys + ["a_star", "a_star_star"]].rename(
        columns={"a_star": "a_star_cells", "a_star_star": "a_star_star_cells"}
    )
    # The rows stay in the order of `thresholds_seeds`.
    comparison = thresholds_seeds.drop(columns="tennis_se").merge(
        cells, on=keys, how="left", validate="one_to_one"
    )
    return comparison


def plot_threshold_bands(
    bands: pd.DataFrame, thresholds: pd.DataFrame, variants: Sequence[str]
) -> Figure:
    """
    Plot the band of the best ball speed against `a`, with the thresholds.

    One row of panels per sport and one column per variant. A filled square is
    a speed of the band at that `a`, a hollow one a speed outside it; the solid
    line is `a*`, and the dashed one `a**`.

    :param bands: the rows of `find_threshold_bands()`
    :param thresholds: the rows of `find_speed_thresholds()` of the same bands
    :param variants: the variants to plot, e.g., the 4 shapes
    :return: the figure
    """
    _LOG.debug(hprint.to_str("variants"))
    hdbg.dassert_lte(1, len(variants), "There must be at least 1 variant")
    sports = list(bands["sport"].unique())
    fig: Figure
    fig, axes = plt.subplots(
        len(sports),
        len(variants),
        figsize=(3.6 * len(variants), 3.4 * len(sports)),
        squeeze=False,
        sharey=True,
    )
    for i, sport_name in enumerate(sports):
        for j, variant in enumerate(variants):
            ax = axes[i, j]
            grp = bands[
                (bands["sport"] == sport_name) & (bands["variant"] == variant)
            ]
            hdbg.dassert_lte(1, len(grp), "No map: %s %s", sport_name, variant)
            row = thresholds[
                (thresholds["sport"] == sport_name)
                & (thresholds["variant"] == variant)
            ].iloc[0]
            shape = variant if variant in SPREAD_SHAPES else "linear"
            color = _SHAPE_STYLES[shape]["color"]
            in_band = grp["in_band"].to_numpy(dtype=bool)
            ax.plot(
                grp["a"][in_band],
                grp["speed_ratio"][in_band],
                "s",
                color=color,
                ms=3.5,
                label="in the band",
            )
            ax.plot(
                grp["a"][~in_band],
                grp["speed_ratio"][~in_band],
                "s",
                markerfacecolor="white",
                color="lightgray",
                ms=3.5,
                label="out of the band",
            )
            # The thresholds found on the grid, if any.
            for name, label, linestyle in (
                ("a_star", "a*", "-"),
                ("a_star_star", "a**", "--"),
            ):
                if np.isfinite(row[name]):
                    ax.axvline(
                        row[name],
                        color="black",
                        linestyle=linestyle,
                        linewidth=1.2,
                        label=f"{label} = {row[name]:g}",
                    )
            ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
            ax.set_xlabel("coefficient a of the spread")
            ax.set_title(f"{sport_name}: {variant}", fontsize=10)
            ax.legend(fontsize=7, frameon=False, loc="upper right")
        axes[i, 0].set_ylabel("ball speed v_b / v0")
    fig.suptitle(
        "The band of the best ball speed against a: 2 v0 leaves it at a*, and"
        " v0 at a**",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


# #############################################################################
# Data of the paper's figure
# #############################################################################

# The `figures` directory of the paper, the same as for notebook 05.
PAPER_FIGURES_DIR = rosfrsnrs0vsu.PAPER_FIGURES_DIR


def save_paper_figure_data(best_speeds: pd.DataFrame, out_dir: str) -> str:
    """
    Save the data of the paper's figure, for `figures/make_figures.py`.

    The figure shows the best ball speed against `a` for the base and the 3
    other shapes, so only their rows are saved.

    :param best_speeds: the rows of `find_best_speeds()` of the full run
    :param out_dir: the `figures` directory of the paper, e.g.,
        `PAPER_FIGURES_DIR`
    :return: the path of the file
    """
    _LOG.debug(hprint.to_str("out_dir"))
    hdbg.dassert_dir_exists(out_dir)
    is_shown = best_speeds["variant"].isin(_FIGURE_VARIANTS)
    columns: List[str] = ["sport", "variant", "a", "best_speed_ratio"]
    columns += ["band_low_ratio", "band_high_ratio"]
    df = best_speeds.loc[is_shown, columns]
    path = os.path.join(out_dir, "speed_vs_accuracy_best_speed.csv")
    df.to_csv(path, index=False)
    return path
