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


# #############################################################################
# Best ball speed
# #############################################################################


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
        speed_map = pd.DataFrame(
            {
                "x_m": runs["speed_ratio"].to_numpy(),
                "y_m": 0.0,
                "p_winner": runs["max_p_winner"].to_numpy(),
                "p_winner_se": runs["max_p_winner_se"].to_numpy(),
            }
        )
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
