"""
Utilities for the `racket_sports_05_variable_sensitivity` notebook.

The notebook asks which settings of the Milestone 2 model matter for the
probability of a winner (#694, #789). Every run calls the Milestone 2 functions
of `racket_sports_01_shot_probability_utils`, with no copied math:
`compute_winner_heatmap()`, `find_best_region()`, and `find_best_target()`.

- Part 1: one setting at a time, around the defaults and a fast player 1
- Part 2: the corners of the settings and the best serve for player 1
- Part 3: the speed ratio check and the reaction time comparison
- Part 4: which variables matter

Import as:

import research.Optimal_strategy_for_racket_sports.notebooks.racket_sports_05_variable_sensitivity_utils as rosfrsnrs0vsu
"""

import dataclasses
import logging
import os
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.patheffects as mpatheff
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tqdm.autonotebook import tqdm

import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebo
import helpers.hprint as hprint
import research.Optimal_strategy_for_racket_sports.notebooks.racket_sports_01_shot_probability_utils as rosfrsnrs0spu
from research.Optimal_strategy_for_racket_sports import racket_params
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
# Settings
# #############################################################################

_SPORTS = {
    sport.name: sport
    for sport in (racket_params.TENNIS, racket_params.PICKLEBALL)
}

# Slider ranges of Cell 2.1 of the `racket_sports_01_shot_probability` notebook:
# the corners are the slider ends.
BALL_SPEED_MIN_MPS = 5.0
V_P_MIN_MPS = 0.0
V_P_MAX_MPS = 6.0
_STD_DEFAULT_M = (0.5, 1.0)
_SEED_DEFAULT = 1

# Serve positions in the deuce box, per sport, on the 0.1 m grid of the
# sliders; the ad side is the mirror (`x1 -> -x1`).
# - x: wide is about 0.3 m inside the singles sideline, T about 0.3 m from the
#   center service line, body is the middle of the box (the widget default)
# - y: short is about 20% of the box depth past its near edge (the net in
#   tennis, the kitchen line in pickleball), mid is the widget default, deep is
#   about 0.3 m inside the service line (the baseline in pickleball)
_SERVE_X = {
    "Tennis": {"wide": -3.8, "body": -2.1, "T": -0.3},
    "Pickleball": {"wide": -2.8, "body": -1.5, "T": -0.3},
}
_SERVE_Y = {
    "Tennis": {"short": 1.3, "mid": 3.2, "deep": 6.1},
    "Pickleball": {"short": 3.0, "mid": 4.4, "deep": 6.4},
}
_SERVE_NAMES = tuple(
    f"{x_name}-{y_name}"
    for x_name in ("wide", "body", "T")
    for y_name in ("short", "mid", "deep")
)

# Settings of the serve maps: where `P(winner)` is not saturated at 1, so the
# serve can matter.
_SERVE_MAP_SETTINGS = {
    "defaults": {},
    "fast_p1": {"v_p": V_P_MAX_MPS},
    "fast_p1_slow_ball": {
        "v_p": V_P_MAX_MPS,
        "ball_speed": BALL_SPEED_MIN_MPS,
    },
}

# Reaction times of Part 3: none, and the 190 to 250 ms of simple visual
# reaction time quoted in the paper (`woods2015reaction`).
_T_R_VALUES = (0.0, 0.2, 0.25)
# Ratios `r = v_p / v_b`: the widget's range, from `v_p` 1.5 m/s with the
# fastest tennis ball (0.02) to `v_p` 6 m/s with a 5 m/s ball (1.2).
_R_VALUES = (0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.8, 1.2)
# Player speeds with the same ratio: powers of 2 apart, so `v_b = v_p / r`
# scales exactly in floating point.
_V_P_VALUES = (1.5, 3.0, 6.0)


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    """
    How much of each experiment to run.

    Reducing a run removes whole runs and lowers `n_samples`, but keeps the
    cells of each map, so a reduced run compares cell by cell with a full one.
    """

    # Sports to run.
    sports: Tuple[str, ...]
    # Grid `(n_x, n_y)` of a map, per sport.
    grid_size: Dict[str, Tuple[int, int]]
    # Monte Carlo samples per target.
    n_samples: int
    # Serve sides of the corners.
    sides: Tuple[str, ...]
    # Named serves of the corners, e.g., "T-deep".
    serve_names: Tuple[str, ...]
    # Keep every `sweep_step`-th value of a sweep; both ends are always kept.
    sweep_step: int
    # Keep every `serve_map_step`-th point per axis of the 0.5 m serve grid.
    serve_map_step: int
    # Settings of the serve maps, from `_SERVE_MAP_SETTINGS`.
    serve_map_names: Tuple[str, ...]
    # Factors of the speed scaling check.
    ks: Tuple[float, ...]
    # Random settings of the speed scaling check.
    n_random: int
    # Ratios `r = v_p / v_b` of the speed grid.
    r_values: Tuple[float, ...]


_FULL_CONFIG = ExperimentConfig(
    sports=tuple(_SPORTS),
    grid_size=rosfrsnrs0spu.HEATMAP_GRID_SIZE,
    n_samples=rosfrsnrs0spu.HEATMAP_N_SAMPLES,
    sides=("deuce", "ad"),
    serve_names=_SERVE_NAMES,
    sweep_step=1,
    serve_map_step=1,
    serve_map_names=tuple(_SERVE_MAP_SETTINGS),
    ks=(0.5, 2.0, 4.0),
    n_random=500,
    r_values=_R_VALUES,
)
# About 430 maps instead of about 1,230. A map costs about the same with 2,000
# and 5,000 samples, so the time comes from the number of maps; the lower
# `n_samples` gives an independent check of the full results.
_REDUCED_CONFIG = dataclasses.replace(
    _FULL_CONFIG,
    n_samples=2000,
    sides=("deuce",),
    sweep_step=2,
    serve_map_step=2,
    serve_map_names=("fast_p1_slow_ball",),
    ks=(2.0,),
    n_random=30,
    r_values=(0.03, 0.1, 0.3, 0.6, 1.2),
)


def get_experiment_config(run_full: bool) -> ExperimentConfig:
    """
    Get how much of each experiment to run.

    :param run_full: True for the full runs used in the paper, False for the
        reduced runs that take a few minutes
    :return: the configuration
    """
    _LOG.debug(hprint.to_str("run_full"))
    hdbg.dassert_isinstance(run_full, bool, "run_full must be a bool")
    config = _FULL_CONFIG if run_full else _REDUCED_CONFIG
    return config


def get_default_settings(
    sport_name: str,
    *,
    n_samples: int = rosfrsnrs0spu.HEATMAP_N_SAMPLES,
) -> Dict[str, Any]:
    """
    Get the default settings of Cell 2.1 for a sport.

    :param sport_name: "Tennis" or "Pickleball"
    :param n_samples: Monte Carlo samples per target
    :return: settings with the keys of the settings string, e.g., `{"sport":
        "Tennis", "side": "deuce", "x1": -2.1, ..., "n_samples": 5000}`
    """
    _LOG.debug(hprint.to_str("sport_name n_samples"))
    hdbg.dassert_in(sport_name, _SPORTS, "Unknown sport")
    n_x, n_y = rosfrsnrs0spu.HEATMAP_GRID_SIZE[sport_name]
    player = racket_params.DEFAULT_PLAYER
    settings = {
        "sport": sport_name,
        "side": "deuce",
        "x1": _SERVE_X[sport_name]["body"],
        "y1": _SERVE_Y[sport_name]["mid"],
        "std_x2": _STD_DEFAULT_M[0],
        "std_y2": _STD_DEFAULT_M[1],
        "ball_speed": rosfrsnrs0spu.WINNER_BALL_SPEED_MPS[sport_name],
        "t_r": player.reaction_time_s,
        "v_p": player.move_speed_mps,
        "seed": _SEED_DEFAULT,
        "n_x": n_x,
        "n_y": n_y,
        "n_samples": n_samples,
    }
    return settings


def settings_to_string(settings: Dict[str, Any]) -> str:
    """
    Build the settings string of Cell 2.1 with the widget's own function.

    :param settings: settings, e.g., from `get_default_settings()`
    :return: e.g., "sport=Tennis side=deuce x1=-2.1 y1=3.2 std_x2=0.5
        std_y2=1.0 ball_speed=25.0 t_r=0.2 v_p=1.5 seed=1 n_x=10 n_y=15
        n_samples=5000"
    """
    _LOG.debug(hprint.to_str("settings"))
    hdbg.dassert_set_eq(
        settings.keys(),
        get_default_settings("Tennis").keys(),
        "Unexpected settings",
    )
    text: str = rosfrsnrs0spu.build_settings_string(
        settings["sport"],
        settings["side"],
        (settings["x1"], settings["y1"]),
        (settings["std_x2"], settings["std_y2"]),
        settings["ball_speed"],
        settings["t_r"],
        settings["v_p"],
        settings["seed"],
        (settings["n_x"], settings["n_y"]),
        settings["n_samples"],
    )
    return text


def parse_settings_string(text: str) -> Dict[str, Any]:
    """
    Parse a settings string of Cell 2.1 back into settings.

    :param text: e.g., "sport=Tennis side=deuce x1=-2.1 ... n_samples=5000"
    :return: settings with floats and ints, the inverse of
        `settings_to_string()`
    """
    _LOG.debug(hprint.to_str("text"))
    settings: Dict[str, Any] = dict(item.split("=", 1) for item in text.split())
    for name in ("x1", "y1", "std_x2", "std_y2", "ball_speed", "t_r", "v_p"):
        settings[name] = float(settings[name])
    for name in ("seed", "n_x", "n_y", "n_samples"):
        settings[name] = int(settings[name])
    # The string round trips exactly: `repr()` of each float reads back as the
    # same float. It also checks the names of the settings.
    hdbg.dassert_eq(settings_to_string(settings), text)
    return settings


# #############################################################################
# One run
# #############################################################################


def _compute_map(settings: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute the winner heatmap of one set of settings.

    :param settings: settings, e.g., from `get_default_settings()`
    :return: the map of `compute_winner_heatmap()`
    """
    _LOG.debug(hprint.to_str("settings"))
    heatmap = rosfrsnrs0spu.compute_winner_heatmap(
        (settings["x1"], settings["y1"]),
        (settings["std_x2"], settings["std_y2"]),
        settings["ball_speed"],
        settings["t_r"],
        settings["v_p"],
        _SPORTS[settings["sport"]].court,
        settings["n_x"],
        settings["n_y"],
        settings["n_samples"],
        settings["seed"],
    )
    return heatmap


def _dassert_legal_serve(settings: Dict[str, Any]) -> None:
    """
    Check that the serve of `settings` is in the legal box.

    :param settings: settings, e.g., from `get_default_settings()`
    """
    _, fault = rosfrsnrs0spu.check_serve(
        settings["side"],
        settings["x1"],
        settings["y1"],
        _SPORTS[settings["sport"]].court,
    )
    hdbg.dassert_eq(fault, "", "The serve must be legal: %s", settings)


def _summarize_map(
    settings: Dict[str, Any], heatmap: pd.DataFrame
) -> Dict[str, Any]:
    """
    Summarize a map: its highest value, best target, and best region.

    :param settings: settings of the map
    :param heatmap: the map, from `compute_winner_heatmap()`
    :return: the settings string, and:
        - `best_*`: the marked best target and its statistics
        - `max_p_winner`, `max_p_winner_se`: the highest `P(winner)`
        - `mean_p_winner`: the mean over the cells of the half
        - `n_region_cells`, `region_centroid_*`: the best region
    """
    best = rosfrsnrs0spu.find_best_target(heatmap)
    region = heatmap[rosfrsnrs0spu.find_best_region(heatmap)]
    # The marked target is one cell of the map: its own `P(in)`, `P(reached)`.
    is_best = (heatmap["x_m"] == best["x_m"]) & (heatmap["y_m"] == best["y_m"])
    best_row = heatmap[is_best].iloc[0]
    summary = {
        "settings": settings_to_string(settings),
        "best_x": best["x_m"],
        "best_y": best["y_m"],
        "best_p_winner": best["p_winner"],
        "best_p_winner_se": best["p_winner_se"],
        "best_p_in": float(best_row["p_in"]),
        "best_p_reached": float(best_row["p_reached"]),
        "max_p_winner": best["max_p_winner"],
        "max_p_winner_se": best["max_p_winner_se"],
        "mean_p_winner": float(heatmap["p_winner"].mean()),
        "n_region_cells": best["n_region_cells"],
        "n_cells": len(heatmap),
        "region_centroid_x": float(region["x_m"].mean()),
        "region_centroid_y": float(region["y_m"].mean()),
    }
    return summary


def compute_run(
    settings: Dict[str, Any],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Compute the map of one set of settings and summarize it.

    :param settings: settings, e.g., from `get_default_settings()`, with a
        legal serve
    :return: the summary of `_summarize_map()`, and the map
    """
    _LOG.debug(hprint.to_str("settings"))
    _dassert_legal_serve(settings)
    heatmap = _compute_map(settings)
    summary = _summarize_map(settings, heatmap)
    return summary, heatmap


def _run_specs(specs: List[Dict[str, Any]], *, desc: str) -> pd.DataFrame:
    """
    Run a list of specs.

    :param specs: the runs, each with `run_id`, `experiment`, `group`,
        `serve_name`, `sweep_var`, and `settings`
    :param desc: label of the progress bar
    :return: one row per run: the spec, the settings, and the summary
    """
    _LOG.debug(hprint.to_str("len(specs) desc"))
    rows = []
    for spec in tqdm(specs, desc=desc):
        summary, _ = compute_run(spec["settings"])
        row = {name: value for name, value in spec.items() if name != "settings"}
        row.update(spec["settings"])
        row.update(summary)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def _apply_config_to_settings(
    settings: Dict[str, Any], config: ExperimentConfig
) -> Dict[str, Any]:
    """
    Set the grid and the samples of `config` in `settings`.

    :param settings: settings, e.g., from `get_default_settings()`
    :param config: how much to run
    :return: a new dictionary
    """
    n_x, n_y = config.grid_size[settings["sport"]]
    new_settings = dict(settings, n_x=n_x, n_y=n_y, n_samples=config.n_samples)
    return new_settings


# #############################################################################
# Part 1: One setting at a time
# #############################################################################


# Sweep values, on the slider steps of Cell 2.1.
_SWEEP_BALL_SPEED = {
    "Tennis": [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 73.14],
    "Pickleball": [
        5.0,
        7.0,
        9.0,
        10.0,
        12.0,
        14.0,
        16.0,
        18.0,
        20.0,
        22.0,
        24.59,
    ],
}
_SWEEP_V_P = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
_SWEEP_T_R = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
_SWEEP_STD = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]


def get_sweep_values(
    sport_name: str, *, sweep_step: int = 1
) -> Dict[str, List[float]]:
    """
    Get the values of each setting to sweep.

    :param sport_name: "Tennis" or "Pickleball"
    :param sweep_step: keep every `sweep_step`-th value; the first and the
        last value are always kept
    :return: values per setting, e.g., `{"ball_speed": [5.0, ...], "v_p": ...}`
    """
    _LOG.debug(hprint.to_str("sport_name sweep_step"))
    hdbg.dassert_in(sport_name, _SPORTS, "Unknown sport")
    hdbg.dassert_lte(1, sweep_step, "sweep_step must be positive")
    sweeps = {
        "ball_speed": _SWEEP_BALL_SPEED[sport_name],
        "v_p": _SWEEP_V_P,
        "t_r": _SWEEP_T_R,
        "std_x2": _SWEEP_STD,
        "std_y2": _SWEEP_STD,
    }
    sweep_values = {}
    for var, values in sweeps.items():
        kept = values[::sweep_step]
        # The ends of a sweep are the most informative values.
        if kept[-1] != values[-1]:
            kept = kept + [values[-1]]
        sweep_values[var] = kept
    return sweep_values


def run_sweep(
    settings: Dict[str, Any],
    var: str,
    values: Sequence[float],
    base_name: str,
) -> pd.DataFrame:
    """
    Vary one setting, keeping the others fixed.

    Every run uses the same seed: common random numbers, so a change between 2
    values comes from the setting and not from sampling noise.

    :param settings: the settings kept fixed, e.g., from
        `get_default_settings()`
    :param var: the setting to vary, e.g., "v_p"
    :param values: its values, e.g., `[0.0, 1.5, 6.0]`
    :param base_name: name of the fixed settings, e.g., "defaults"
    :return: one row per value, as in `_run_specs()`
    """
    _LOG.debug(hprint.to_str("var values base_name"))
    hdbg.dassert_in(var, settings, "Unknown setting")
    hdbg.dassert_lte(1, len(values), "There must be at least 1 value")
    sport_tag = settings["sport"].lower()
    specs = [
        {
            "run_id": f"sweep_{base_name}_{sport_tag}_{var}_{value:g}",
            "experiment": f"sweep_{base_name}",
            "group": var,
            "serve_name": "",
            "sweep_var": var,
            "settings": dict(settings, **{var: value}),
        }
        for value in values
    ]
    df = _run_specs(specs, desc=f"{sport_tag} {base_name} {var}")
    return df


def run_all_sweeps(config: ExperimentConfig) -> pd.DataFrame:
    """
    Run every sweep, around the defaults and around a fast player 1.

    :param config: how much to run
    :return: one row per run, as in `run_sweep()`
    """
    _LOG.debug(hprint.to_str("config"))
    bases = {"defaults": {}, "fast_p1": {"v_p": V_P_MAX_MPS}}
    dfs = []
    for sport_name in config.sports:
        sweeps = get_sweep_values(sport_name, sweep_step=config.sweep_step)
        for base_name, update in bases.items():
            settings = get_default_settings(sport_name)
            settings = _apply_config_to_settings(settings, config)
            settings.update(update)
            for var, values in sweeps.items():
                # The `v_p` sweep is the same around both bases.
                if base_name == "fast_p1" and var == "v_p":
                    continue
                dfs.append(run_sweep(settings, var, values, base_name))
    sweeps_df = pd.concat(dfs, ignore_index=True)
    return sweeps_df


# Order and colors of the charts: one color per sport.
_SWEEP_VARS = ("ball_speed", "v_p", "t_r", "std_x2", "std_y2")
_SPORT_COLORS = {"Tennis": "#2a78d6", "Pickleball": "#eb6834"}


def plot_sweeps(sweeps: pd.DataFrame, experiment: str) -> Figure:
    """
    Plot how the best value and the best region move with each setting.

    One row per swept setting; columns: the highest `P(winner)` with a 2 SE
    band, the fraction of cells in the best region, and the region centroid
    `y` (negative on player 1's half).

    :param sweeps: the rows of `run_all_sweeps()`
    :param experiment: "sweep_defaults" or "sweep_fast_p1"
    :return: the figure
    """
    _LOG.debug(hprint.to_str("experiment"))
    hdbg.dassert_in(experiment, set(sweeps["experiment"]), "Unknown experiment")
    runs = sweeps[sweeps["experiment"] == experiment]
    variables = [var for var in _SWEEP_VARS if var in set(runs["sweep_var"])]
    fig: Figure
    fig, axes = plt.subplots(
        len(variables), 3, figsize=(14, 2.6 * len(variables)), squeeze=False
    )
    for i, var in enumerate(variables):
        for sport_name, color in _SPORT_COLORS.items():
            grp = runs[
                (runs["sweep_var"] == var) & (runs["sport"] == sport_name)
            ]
            grp = grp.sort_values(var)
            x = grp[var].to_numpy()
            p = grp["max_p_winner"].to_numpy()
            se = grp["max_p_winner_se"].to_numpy()
            style: Dict[str, Any] = {"color": color, "lw": 1.5, "ms": 4}
            axes[i, 0].plot(x, p, "o-", label=sport_name, **style)
            axes[i, 0].fill_between(
                x, p - 2 * se, p + 2 * se, color=color, alpha=0.2
            )
            axes[i, 1].plot(
                x, grp["n_region_cells"] / grp["n_cells"], "o-", **style
            )
            axes[i, 2].plot(x, grp["region_centroid_y"], "o-", **style)
        axes[i, 0].set_ylabel(f"sweep of {var}", fontsize=10, fontweight="bold")
        for j in range(3):
            axes[i, j].set_xlabel(var)
            axes[i, j].grid(alpha=0.3)
        axes[i, 0].set_title("highest P(winner), +- 2 SE", fontsize=10)
        axes[i, 1].set_title("best region, fraction of the cells", fontsize=10)
        axes[i, 2].set_title("best region centroid y (m)", fontsize=10)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(
        f"{experiment}: one setting at a time, the others at their base values",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


def plot_montage(
    settings_list: Sequence[Dict[str, Any]],
    titles: Sequence[str],
    suptitle: str,
) -> Figure:
    """
    Plot several maps side by side, cropped to player 1's half.

    :param settings_list: the settings of each map, left to right, all for the
        same sport
    :param titles: one title per map
    :param suptitle: title of the figure, e.g., with the fixed settings
    :return: the figure
    """
    _LOG.debug(hprint.to_str("titles suptitle"))
    hdbg.dassert_eq(len(settings_list), len(titles), "One title per map")
    hdbg.dassert_lte(2, len(titles), "A montage has at least 2 maps")
    fig: Figure
    fig, axes = plt.subplots(
        1, len(titles), figsize=(3.6 * len(titles), 5.5), squeeze=False
    )
    for ax, settings, title in zip(axes[0], settings_list, titles):
        court = _SPORTS[settings["sport"]].court
        summary, heatmap = compute_run(settings)
        rosfrsnrs0spu.plot_winner_heatmap(
            ax, heatmap, court, (settings["x1"], settings["y1"])
        )
        ax.get_legend().remove()
        ax.set_ylim(-court.length_m / 2 - 1, 1.0)
        ax.set_title(
            f"{title}\nbest {summary['max_p_winner']:.3f}, region"
            f" {summary['n_region_cells']} cells",
            fontsize=10,
        )
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return fig


# #############################################################################
# Part 2: Corners and the best serve
# #############################################################################


def _get_serves(
    sport_name: str, side: str, serve_names: Sequence[str]
) -> List[Tuple[str, float, float]]:
    """
    Get the named serves of a side.

    :param sport_name: "Tennis" or "Pickleball"
    :param side: "deuce" or "ad"
    :param serve_names: names of the serves, e.g., `["T-deep"]`
    :return: `(name, x1, y1)`, e.g., `("T-deep", -0.3, 6.1)`
    """
    sign = 1.0 if side == "deuce" else -1.0
    serves = []
    for name in serve_names:
        x_name, y_name = name.split("-")
        x1 = _SERVE_X[sport_name][x_name]
        # Adding 0.0 avoids a -0.0 in the settings string.
        serves.append((name, sign * x1 + 0.0, _SERVE_Y[sport_name][y_name]))
    return serves


def _get_corner_values(
    sport_name: str,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Get the slider ends and the defaults of the ball speed and of `v_p`.

    :param sport_name: "Tennis" or "Pickleball"
    :return: the ball speeds and the `v_p` values, each keyed by "min", "def",
        and "max"
    """
    defaults = get_default_settings(sport_name)
    ball_speeds = {
        "min": BALL_SPEED_MIN_MPS,
        "def": defaults["ball_speed"],
        "max": _SPORTS[sport_name].max_ball_speed_mps,
    }
    v_ps = {"min": V_P_MIN_MPS, "def": defaults["v_p"], "max": V_P_MAX_MPS}
    return ball_speeds, v_ps


def run_corners(config: ExperimentConfig) -> pd.DataFrame:
    """
    Run the corners: serves x ball speed x `v_p`, per side and sport.

    :param config: how much to run
    :return: one row per run, as in `_run_specs()`; `group` names the corner,
        e.g., "ball=min v_p=max"
    """
    _LOG.debug(hprint.to_str("config"))
    specs = []
    for sport_name in config.sports:
        ball_speeds, v_ps = _get_corner_values(sport_name)
        base = get_default_settings(sport_name)
        base = _apply_config_to_settings(base, config)
        for side in config.sides:
            for name, x1, y1 in _get_serves(
                sport_name, side, config.serve_names
            ):
                for ball_name, ball_speed in ball_speeds.items():
                    for v_p_name, v_p in v_ps.items():
                        run_id = (
                            f"corner_{sport_name.lower()}_{side}_{name}"
                            f"_ball{ball_name}_vp{v_p_name}"
                        )
                        specs.append(
                            {
                                "run_id": run_id,
                                "experiment": "corners",
                                "group": f"ball={ball_name} v_p={v_p_name}",
                                "serve_name": name,
                                "sweep_var": "",
                                "settings": dict(
                                    base,
                                    side=side,
                                    x1=x1,
                                    y1=y1,
                                    ball_speed=ball_speed,
                                    v_p=v_p,
                                ),
                            }
                        )
    df = _run_specs(specs, desc="corners")
    return df


def find_best_serves(corners: pd.DataFrame) -> pd.DataFrame:
    """
    Find the best serve for player 1 in each corner.

    The best serve has the lowest highest `P(winner)`: player 2's best answer
    is the weakest. Another serve is tied with it if its value is within 2
    standard errors of the difference, as for the best region.

    :param corners: the rows of `run_corners()`
    :return: one row per sport, side, and corner, with `best_serve`, its
        `max_p_winner`, the highest value over the serves, and `n_tied`, the
        number of serves tied with the best one, itself included
    """
    _LOG.debug(hprint.to_str("corners.shape"))
    hdbg.dassert_lte(1, len(corners), "There must be at least 1 run")
    rows = []
    for (sport, side, group), grp in corners.groupby(
        ["sport", "side", "group"], sort=False
    ):
        best = grp.loc[grp["max_p_winner"].idxmin()]
        gap = grp["max_p_winner"] - best["max_p_winner"]
        gap_se = np.hypot(grp["max_p_winner_se"], best["max_p_winner_se"])
        rows.append(
            {
                "sport": sport,
                "side": side,
                "corner": group,
                "best_serve": best["serve_name"],
                "max_p_winner": best["max_p_winner"],
                "highest_over_serves": grp["max_p_winner"].max(),
                "n_tied": int((gap <= 2 * gap_se).sum()),
            }
        )
    df = pd.DataFrame(rows)
    return df


def _get_text_color(rgba: Tuple[float, float, float, float]) -> str:
    """
    Pick the text color that reads best on a background color.

    :param rgba: the background color
    :return: "black" on a light background, "white" on a dark one
    """
    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    text_color = "black" if luminance > 0.5 else "white"
    return text_color


def _draw_serve_table(ax: Axes, runs: pd.DataFrame, title: str) -> None:
    """
    Draw the table of one sport and side: corners as rows, serves as columns.

    :param ax: axes to draw on
    :param runs: the rows of `run_corners()` of one sport and side
    :param title: title of the panel
    """
    corner_order = [
        corner
        for corner in (
            f"ball={ball} v_p={v_p}"
            for ball in ("min", "def", "max")
            for v_p in ("min", "def", "max")
        )
        if corner in set(runs["group"])
    ]
    serve_order = [
        name for name in _SERVE_NAMES if name in set(runs["serve_name"])
    ]

    def get_table(column: str) -> np.ndarray:
        table = runs.pivot(index="group", columns="serve_name", values=column)
        values: np.ndarray = table.loc[corner_order, serve_order].to_numpy()
        return values

    values = get_table("max_p_winner")
    ses = get_table("max_p_winner_se")
    means = get_table("mean_p_winner")
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap(rosfrsnrs0spu.HEATMAP_CMAP)
    ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    ax.grid(False)
    for i in range(values.shape[0]):
        i_top = int(np.argmax(values[i]))
        for j in range(values.shape[1]):
            color = _get_text_color(cmap(norm(values[i, j])))
            ax.text(
                j,
                i - 0.12,
                f"{values[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color=color,
            )
            ax.text(
                j,
                i + 0.22,
                f"mean {means[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color=color,
                fontstyle="italic",
            )
            # Box a serve only if it is lower than the highest of its row by
            # more than 2 standard errors of the difference.
            gap = values[i, i_top] - values[i, j]
            if gap > 2 * np.hypot(ses[i, i_top], ses[i, j]):
                box = plt.Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="black",
                    lw=2,
                    # A white edge keeps the box visible on every color.
                    path_effects=[
                        mpatheff.withStroke(linewidth=4, foreground="white")
                    ],
                )
                ax.add_patch(box)
    ax.set_xticks(range(len(serve_order)))
    ax.set_xticklabels(serve_order, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(corner_order)))
    ax.set_yticklabels(corner_order, fontsize=9)
    ax.set_title(title, fontsize=12)


def plot_serve_table(corners: pd.DataFrame, sport_name: str) -> Figure:
    """
    Plot the highest `P(winner)` for every named serve and corner.

    One panel per side; rows are the corners, columns the serves. Each cell
    shows the highest `P(winner)` and the mean over the half court. A box marks
    a serve that is lower than the highest value of its row by more than 2
    standard errors of the difference, so a tie within noise has no box.

    :param corners: the rows of `run_corners()`
    :param sport_name: "Tennis" or "Pickleball"
    :return: the figure
    """
    _LOG.debug(hprint.to_str("sport_name"))
    hdbg.dassert_in(sport_name, set(corners["sport"]), "Unknown sport")
    runs = corners[corners["sport"] == sport_name]
    sides = list(runs["side"].unique())
    fig: Figure
    fig, axes = plt.subplots(
        1, len(sides), figsize=(8 * len(sides), 6.5), squeeze=False
    )
    for ax, side in zip(axes[0], sides):
        _draw_serve_table(
            ax, runs[runs["side"] == side], f"{sport_name}, {side} side"
        )
    fig.suptitle(
        f"{sport_name}: highest P(winner) of player 2 per serve (top) and mean"
        " P(winner) over the half court (bottom)\nbox: serve lower than the"
        " highest value in its row by more than 2 standard errors of the"
        " difference",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


def run_serve_map(config: ExperimentConfig) -> pd.DataFrame:
    """
    Run the serve maps: the best answer of player 2 for a grid of deuce serves.

    :param config: how much to run
    :return: one row per run, as in `_run_specs()`; `group` names the settings,
        e.g., "fast_p1_slow_ball"
    """
    _LOG.debug(hprint.to_str("config"))
    specs = []
    for sport_name in config.sports:
        court = _SPORTS[sport_name].court
        base = get_default_settings(sport_name)
        base = _apply_config_to_settings(base, config)
        # From the wide serve to the T on a 0.5 m grid, and from the deep serve
        # toward the net, down to 0.2 m past the near edge of the box.
        x_values: np.ndarray = np.round(
            np.arange(_SERVE_X[sport_name]["wide"], 0.0, 0.5), 1
        )
        y_values = np.asarray(
            np.round(
                np.arange(
                    _SERVE_Y[sport_name]["deep"],
                    court.non_volley_zone_m + 0.2,
                    -0.5,
                ),
                1,
            )
        )
        for map_name in config.serve_map_names:
            for x1 in x_values[:: config.serve_map_step]:
                for y1 in y_values[:: config.serve_map_step]:
                    _, fault = rosfrsnrs0spu.check_serve(
                        "deuce", float(x1), float(y1), court
                    )
                    if fault:
                        continue
                    settings = dict(base, **_SERVE_MAP_SETTINGS[map_name])
                    # Adding 0.0 avoids a -0.0 in the settings string.
                    settings.update(x1=float(x1) + 0.0, y1=float(y1))
                    run_id = (
                        f"servemap_{map_name}_{sport_name.lower()}"
                        f"_x{x1:g}_y{y1:g}"
                    )
                    specs.append(
                        {
                            "run_id": run_id,
                            "experiment": "serve_map",
                            "group": map_name,
                            "serve_name": "",
                            "sweep_var": "",
                            "settings": settings,
                        }
                    )
    df = _run_specs(specs, desc="serve map")
    return df


def _draw_serve_map(ax: Axes, runs: pd.DataFrame, court: Any) -> Any:
    """
    Draw one serve map: `P(winner)` by where the serve lands.

    :param ax: axes to draw on
    :param runs: the rows of `run_serve_map()` of one sport and one group
    :param court: court geometry of the sport
    :return: the mesh, for a shared colorbar
    """
    table = runs.pivot(index="y1", columns="x1", values="max_p_winner")
    xs = table.columns.to_numpy()
    ys = table.index.to_numpy()
    # The serve points are the centers of the cells of a regular grid.
    half_dx = np.diff(xs).min() / 2 if len(xs) > 1 else 0.25
    half_dy = np.diff(ys).min() / 2 if len(ys) > 1 else 0.25
    mesh = ax.pcolormesh(
        np.append(xs - half_dx, xs[-1] + half_dx),
        np.append(ys - half_dy, ys[-1] + half_dy),
        np.ma.masked_invalid(table.to_numpy()),
        cmap=rosfrsnrs0spu.HEATMAP_CMAP,
        vmin=0.0,
        vmax=1.0,
        shading="flat",
    )
    ax.grid(False)
    racket_strategy_utils.draw_court(ax, court)
    rosfrsnrs0spu.draw_center_service_line(ax, court)
    best = runs.loc[runs["max_p_winner"].idxmin()]
    worst = runs.loc[runs["max_p_winner"].idxmax()]
    # The white edge keeps the marker visible on the dark cells of the map.
    ax.plot(
        best["x1"],
        best["y1"],
        "v",
        ms=12,
        color=rosfrsnrs0spu.HEATMAP_BEST_COLOR,
        markeredgecolor="white",
        label="best serve for player 1",
    )
    ax.set_xlim(-court.width_m / 2 - 0.5, 0.5)
    ax.set_ylim(court.non_volley_zone_m - 0.5, court.length_m / 2 + 0.5)
    ax.set_title(
        f"{best['sport']}, {best['group']}\nball={best['ball_speed']:g}"
        f" v_p={best['v_p']:g}; min {best['max_p_winner']:.3f} at"
        f" ({best['x1']:g}, {best['y1']:g}), max {worst['max_p_winner']:.3f}",
        fontsize=9,
    )
    return mesh


def plot_serve_maps(serve_map: pd.DataFrame) -> Figure:
    """
    Plot the highest `P(winner)` of player 2 by where the serve lands.

    One row per sport and one column per group of settings, with one color
    scale from 0 to 1 in all the panels, so flat panels look flat.

    :param serve_map: the rows of `run_serve_map()`
    :return: the figure
    """
    _LOG.debug(hprint.to_str("serve_map.shape"))
    hdbg.dassert_lte(1, len(serve_map), "There must be at least 1 run")
    sports = list(serve_map["sport"].unique())
    groups = list(serve_map["group"].unique())
    fig: Figure
    fig, axes = plt.subplots(
        len(sports),
        len(groups),
        figsize=(4.8 * len(groups) + 1.5, 5.0 * len(sports)),
        squeeze=False,
    )
    meshes = []
    for i, sport_name in enumerate(sports):
        court = _SPORTS[sport_name].court
        for j, group in enumerate(groups):
            is_panel = (serve_map["sport"] == sport_name) & (
                serve_map["group"] == group
            )
            meshes.append(
                _draw_serve_map(axes[i, j], serve_map[is_panel], court)
            )
    fig.suptitle(
        "Best P(winner) player 2 can get, by where the deuce serve lands (the"
        " ad side is the mirror)",
        fontsize=12,
    )
    # One legend below the panels and one colorbar for all of them.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 0.92, 1))
    cax = fig.add_axes((0.93, 0.25, 0.015, 0.5))
    fig.colorbar(
        meshes[-1], cax=cax, label="highest P(winner), same scale in all"
    )
    return fig


# #############################################################################
# Part 3: Speed ratio and reaction time
# #############################################################################


# Columns of a map compared cell by cell.
_MAP_COLUMNS = ("p_winner", "p_winner_se", "p_in", "p_reached")


def check_ratio_invariance(settings: Dict[str, Any], k: float) -> Dict[str, Any]:
    """
    Compare a map with the map where both speeds are multiplied by `k`.

    Player 1 reaches a landing point at distance `dist`, with flight distance
    `d`, if `dist <= v_p * max(0, d / v_b - t_r)`, which is
    `dist <= max(0, r * d - v_p * t_r)` with `r = v_p / v_b`. The landing points
    do not depend on the speeds. So:
    - With `t_r = 0` the map depends on the speeds only through `r`, and the 2
      maps are identical
    - With `t_r > 0` the reaction distance `v_p * t_r` changes with `k`, and
      the maps differ

    :param settings: settings, e.g., from `get_default_settings()`; the serve
        can be anywhere on player 2's half
    :param k: the factor, positive, e.g., 2.0
    :return: `max_diff`, the largest cell difference over the columns of
        `_MAP_COLUMNS`, 0.0 for identical maps; the same per column as
        `max_diff_<column>`; `k`; the settings string of the base map; and
        `mean_p_reached`, which shows that the reach rule is exercised
    """
    _LOG.debug(hprint.to_str("settings k"))
    hdbg.dassert(np.isfinite(k), "k=%s must be finite", k)
    hdbg.dassert_lt(0, k, "k must be positive")
    scaled = dict(
        settings, v_p=settings["v_p"] * k, ball_speed=settings["ball_speed"] * k
    )
    heatmap = _compute_map(settings)
    heatmap_scaled = _compute_map(scaled)
    row: Dict[str, Any] = {"settings": settings_to_string(settings), "k": k}
    # Same seed and same grid: the 2 maps line up cell by cell.
    for column in _MAP_COLUMNS:
        diff = (heatmap[column] - heatmap_scaled[column]).abs().max()
        row[f"max_diff_{column}"] = float(diff)
    row["max_diff"] = max(row[f"max_diff_{c}"] for c in _MAP_COLUMNS)
    row["mean_p_reached"] = float(heatmap["p_reached"].mean())
    return row


def _get_ratio_settings(
    sport_name: str, config: ExperimentConfig
) -> Dict[str, Dict[str, Any]]:
    """
    Get the 5 settings of the speed scaling check, all with `t_r = 0`.

    :param sport_name: "Tennis" or "Pickleball"
    :param config: how much to run
    :return: settings per name, e.g., `{"defaults": {...}, ...}`
    """
    base = get_default_settings(sport_name)
    base = _apply_config_to_settings(base, config)
    base["t_r"] = 0.0
    # Serves that are not the default one, with other speeds and spreads.
    t_deep_xy = (_SERVE_X[sport_name]["T"], _SERVE_Y[sport_name]["deep"])
    wide_short_xy = {"Tennis": (-3.8, 1.3), "Pickleball": (-2.7, 3.0)}
    settings = {
        "defaults": base,
        "v_p_6": dict(base, v_p=V_P_MAX_MPS),
        "slow_ball_fast_p1": dict(
            base, v_p=V_P_MAX_MPS, ball_speed=BALL_SPEED_MIN_MPS
        ),
        "T_deep_mid": dict(
            base,
            x1=t_deep_xy[0],
            y1=t_deep_xy[1],
            v_p=3.0,
            ball_speed=10.0,
            std_x2=1.0,
            std_y2=2.0,
            seed=7,
        ),
        "wide_short_slow": dict(
            base,
            x1=wide_short_xy[sport_name][0],
            y1=wide_short_xy[sport_name][1],
            v_p=4.5,
            ball_speed=7.5,
            std_x2=0.3,
            std_y2=0.7,
            seed=3,
        ),
    }
    return settings


def run_ratio_checks(config: ExperimentConfig) -> pd.DataFrame:
    """
    Run `check_ratio_invariance()` on 5 settings per sport and on each `k`.

    :param config: how much to run
    :return: one row per pair of maps, with `sport` and `setting`
    """
    _LOG.debug(hprint.to_str("config"))
    rows = []
    for sport_name in config.sports:
        settings = _get_ratio_settings(sport_name, config)
        for name, one_settings in settings.items():
            for k in config.ks:
                row = check_ratio_invariance(one_settings, k)
                row.update(sport=sport_name, setting=name)
                rows.append(row)
    df = pd.DataFrame(rows)
    return df


def run_random_ratio_checks(config: ExperimentConfig) -> pd.DataFrame:
    """
    Run `check_ratio_invariance()` on random settings and a random `k`.

    The settings are drawn from a fixed seed, in the same order for every
    `n_random`, so a reduced run is the first part of a full run. The maps are
    small (4 x 5 cells, 2,000 samples), since each one only needs some
    landing points near the edge of the reach.

    :param config: how much to run
    :return: one row per pair of maps
    """
    _LOG.debug(hprint.to_str("config"))
    rng = np.random.default_rng(789)
    rows = []
    for _ in tqdm(range(config.n_random), desc="random settings"):
        sport_name = str(rng.choice(list(_SPORTS)))
        sport = _SPORTS[sport_name]
        court = sport.court
        settings = get_default_settings(sport_name, n_samples=2000)
        settings.update(
            t_r=0.0,
            x1=round(
                float(rng.uniform(-court.width_m / 2, court.width_m / 2)), 1
            ),
            y1=round(float(rng.uniform(0.1, court.length_m / 2)), 1),
            std_x2=round(float(rng.uniform(0.1, 3.0)), 1),
            std_y2=round(float(rng.uniform(0.1, 3.0)), 1),
            v_p=round(float(rng.uniform(0.1, V_P_MAX_MPS)), 2),
            ball_speed=round(
                float(rng.uniform(BALL_SPEED_MIN_MPS, sport.max_ball_speed_mps)),
                2,
            ),
            seed=int(rng.integers(0, 101)),
            n_x=4,
            n_y=5,
        )
        k = round(float(np.exp(rng.uniform(np.log(0.1), np.log(10.0)))), 4)
        rows.append(check_ratio_invariance(settings, k))
    df = pd.DataFrame(rows)
    return df


def run_speed_grid(
    config: ExperimentConfig,
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray]]:
    """
    Run the grid of ratios `r` x `v_p` x `t_r`, with the default serve.

    Both `compare_reaction_time()` and `summarize_ratio_collapse()` read this
    grid, so each map is computed once.

    :param config: how much to run
    :return:
        - One row per map, with `run_id`, `sport`, `v_p`, `ball_speed`, `t_r`,
          `r`, and the summary of `_summarize_map()`
        - The `p_winner` of every cell, per `run_id`
    """
    _LOG.debug(hprint.to_str("config"))
    rows = []
    p_winner_maps = {}
    specs = [
        (sport_name, v_p, t_r, r)
        for sport_name in config.sports
        for v_p in _V_P_VALUES
        for t_r in _T_R_VALUES
        for r in config.r_values
    ]
    for run_id, (sport_name, v_p, t_r, r) in enumerate(
        tqdm(specs, desc="speed grid")
    ):
        settings = get_default_settings(sport_name)
        settings = _apply_config_to_settings(settings, config)
        settings.update(v_p=v_p, t_r=t_r, ball_speed=v_p / r)
        summary, heatmap = compute_run(settings)
        rows.append(
            {
                "run_id": run_id,
                "sport": sport_name,
                "v_p": v_p,
                "ball_speed": settings["ball_speed"],
                "t_r": t_r,
                "r": r,
                **summary,
            }
        )
        p_winner_maps[run_id] = heatmap["p_winner"].to_numpy()
    grid = pd.DataFrame(rows)
    return grid, p_winner_maps


def summarize_ratio_collapse(
    grid: pd.DataFrame, p_winner_maps: Dict[int, np.ndarray]
) -> pd.DataFrame:
    """
    Measure how far the speed pairs of each ratio are from one curve.

    :param grid: the rows of `run_speed_grid()`
    :param p_winner_maps: the maps of `run_speed_grid()`
    :return: one row per `t_r`, each the largest over the sports and ratios of
        the spread across the speed pairs: `max_p_winner_spread`,
        `mean_p_winner_spread`, `n_region_cells_spread`, and `max_cell_diff`;
        all 0 for `t_r = 0`
    """
    _LOG.debug(hprint.to_str("grid.shape"))
    hdbg.dassert_lte(1, len(grid), "There must be at least 1 map")
    rows = []
    for t_r, runs in grid.groupby("t_r"):
        spreads = runs.groupby(["sport", "r"])[
            ["max_p_winner", "mean_p_winner", "n_region_cells"]
        ].agg(lambda values: values.max() - values.min())
        # The largest cell difference within the speed pairs of one ratio.
        max_cell_diff = 0.0
        for _, pairs in runs.groupby(["sport", "r"]):
            maps = [p_winner_maps[run_id] for run_id in pairs["run_id"]]
            diffs = [np.abs(one_map - maps[0]).max() for one_map in maps]
            max_cell_diff = max(max_cell_diff, float(max(diffs)))
        rows.append(
            {
                "t_r": t_r,
                "max_p_winner_spread": spreads["max_p_winner"].max(),
                "mean_p_winner_spread": spreads["mean_p_winner"].max(),
                "n_region_cells_spread": spreads["n_region_cells"].max(),
                "max_cell_diff": max_cell_diff,
            }
        )
    df = pd.DataFrame(rows)
    return df


def compare_reaction_time(
    grid: pd.DataFrame, p_winner_maps: Dict[int, np.ndarray]
) -> pd.DataFrame:
    """
    Compare each map of the speed grid with its map at `t_r = 0`.

    :param grid: the rows of `run_speed_grid()`, with `t_r = 0` among them
    :param p_winner_maps: the maps of `run_speed_grid()`
    :return: one row per sport, `v_p`, `r`, and `t_r > 0`:
        - `best_t0`, `best_t`, and their difference `d_best`: the highest
          `P(winner)` without and with the reaction time
        - `mean_t0`, `mean_t`, `d_mean`: the same for the mean over the half
        - `region_t0`, `region_t`, `d_region`: the same for the best region
        - `d_centroid_m`: how far the centroid of the best region moves
        - `max_cell_diff`: the largest change of one cell
        - `in_domain`: True if the ball speed is within the widget's range,
          from 5 m/s to the sport's fastest ball
    """
    _LOG.debug(hprint.to_str("grid.shape"))
    hdbg.dassert_in(0.0, set(grid["t_r"]), "The grid must include t_r = 0")
    rows = []
    for (sport, v_p, r), runs in grid.groupby(["sport", "v_p", "r"], sort=False):
        base = runs[runs["t_r"] == 0.0].iloc[0]
        max_ball_speed = _SPORTS[str(sport)].max_ball_speed_mps
        for _, other in runs[runs["t_r"] > 0.0].iterrows():
            cell_diff = (
                p_winner_maps[base["run_id"]] - p_winner_maps[other["run_id"]]
            )
            rows.append(
                {
                    "sport": sport,
                    "v_p": v_p,
                    "r": r,
                    "ball_speed": base["ball_speed"],
                    "t_r": other["t_r"],
                    "reaction_dist_m": v_p * other["t_r"],
                    "best_t0": base["max_p_winner"],
                    "best_t": other["max_p_winner"],
                    "d_best": base["max_p_winner"] - other["max_p_winner"],
                    "mean_t0": base["mean_p_winner"],
                    "mean_t": other["mean_p_winner"],
                    "d_mean": base["mean_p_winner"] - other["mean_p_winner"],
                    "region_t0": base["n_region_cells"],
                    "region_t": other["n_region_cells"],
                    "d_region": base["n_region_cells"] - other["n_region_cells"],
                    "d_centroid_m": float(
                        np.hypot(
                            base["region_centroid_x"]
                            - other["region_centroid_x"],
                            base["region_centroid_y"]
                            - other["region_centroid_y"],
                        )
                    ),
                    "max_cell_diff": float(np.abs(cell_diff).max()),
                    # With a small margin for the rounding of `v_p / r`.
                    "in_domain": v_p / max_ball_speed - 1e-12
                    <= r
                    <= v_p / BALL_SPEED_MIN_MPS + 1e-12,
                }
            )
    df = pd.DataFrame(rows)
    return df


def summarize_reaction_time(comparison: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize `compare_reaction_time()` within the widget's ball speeds.

    :param comparison: the rows of `compare_reaction_time()`
    :return: one row per sport, `v_p`, and `t_r`, with the largest change:
        `best_points` and `mean_points` in percentage points of `P(winner)`,
        `region_cells`, `centroid_m`, and `max_cell_diff`
    """
    _LOG.debug(hprint.to_str("comparison.shape"))
    in_domain = comparison[comparison["in_domain"]]
    hdbg.dassert_lte(1, len(in_domain), "There must be at least 1 row")
    rows = []
    for (sport, v_p, t_r), runs in in_domain.groupby(
        ["sport", "v_p", "t_r"], sort=False
    ):
        rows.append(
            {
                "sport": sport,
                "v_p": v_p,
                "t_r": t_r,
                "best_points": 100 * runs["d_best"].abs().max(),
                "mean_points": 100 * runs["d_mean"].abs().max(),
                "region_cells": runs["d_region"].abs().max(),
                "centroid_m": runs["d_centroid_m"].max(),
                "max_cell_diff": runs["max_cell_diff"].max(),
            }
        )
    df = pd.DataFrame(rows)
    return df


# Categorical slots 1 to 3, with markers and line styles as a second encoding.
_SERIES_STYLE = (
    {"color": "#2a78d6", "marker": "o", "linestyle": "-", "markersize": 8},
    {"color": "#eb6834", "marker": "s", "linestyle": "--", "markersize": 6},
    {"color": "#1baf7a", "marker": "^", "linestyle": ":", "markersize": 5},
)
_METRICS = (
    ("max_p_winner", "highest P(winner)"),
    ("n_region_cells", "best region size (cells)"),
    ("mean_p_winner", "mean P(winner) over the half"),
)


def _style_ratio_axes(ax: Axes, ylabel: str) -> None:
    """
    Style one panel with the ratio `r` on a log scale.

    :param ax: axes to style
    :param ylabel: label of the metric
    """
    ax.set_xscale("log")
    ax.set_xlabel("r = v_p / v_b")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_ratio_collapse(grid: pd.DataFrame, sport_name: str) -> Figure:
    """
    Plot each metric against `r` for the speed pairs with the same `r`.

    At `t_r = 0` the speed pairs fall on one curve; at `t_r = 0.2` s they do
    not.

    :param grid: the rows of `run_speed_grid()`
    :param sport_name: "Tennis" or "Pickleball"
    :return: the figure
    """
    _LOG.debug(hprint.to_str("sport_name"))
    hdbg.dassert_in(sport_name, set(grid["sport"]), "Unknown sport")
    runs = grid[grid["sport"] == sport_name]
    t_rs = (0.0, 0.2)
    fig: Figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for i, t_r in enumerate(t_rs):
        for j, (column, label) in enumerate(_METRICS):
            ax = axes[i, j]
            for style, v_p in zip(_SERIES_STYLE, _V_P_VALUES):
                series = runs[(runs["v_p"] == v_p) & (runs["t_r"] == t_r)]
                series = series.sort_values("r")
                ax.plot(
                    series["r"],
                    series[column],
                    linewidth=2,
                    label=f"v_p = {v_p:g} m/s",
                    **style,
                )
            _style_ratio_axes(ax, label)
            ax.set_title(f"t_r = {t_r:g} s", fontsize=11)
    axes[0, 0].legend(loc="lower left", fontsize=9, frameon=False)
    fig.suptitle(
        f"{sport_name}: 3 speed pairs per ratio r (v_b = v_p / r); at t_r = 0"
        " the 3 curves are one curve",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


def plot_reaction_time(grid: pd.DataFrame, v_p: float) -> Figure:
    """
    Plot each metric against `r` for each reaction time, at one `v_p`.

    :param grid: the rows of `run_speed_grid()`
    :param v_p: the speed of player 1, e.g., 1.5
    :return: the figure
    """
    _LOG.debug(hprint.to_str("v_p"))
    hdbg.dassert_in(v_p, set(grid["v_p"]), "Unknown v_p")
    sports = list(grid["sport"].unique())
    fig: Figure
    fig, axes = plt.subplots(
        len(sports),
        len(_METRICS),
        figsize=(15, 4 * len(sports)),
        sharex=True,
        squeeze=False,
    )
    for i, sport_name in enumerate(sports):
        for j, (column, label) in enumerate(_METRICS):
            ax = axes[i, j]
            for style, t_r in zip(_SERIES_STYLE, _T_R_VALUES):
                is_series = (
                    (grid["sport"] == sport_name)
                    & (grid["v_p"] == v_p)
                    & (grid["t_r"] == t_r)
                )
                series = grid[is_series].sort_values("r")
                ax.plot(
                    series["r"],
                    series[column],
                    linewidth=2,
                    label=f"t_r = {t_r:g} s",
                    **style,
                )
            _style_ratio_axes(ax, label)
            ax.set_title(sport_name, fontsize=11)
    axes[0, 0].legend(loc="lower left", fontsize=9, frameon=False)
    fig.suptitle(
        f"Reaction time 0 vs realistic, v_p = {v_p:g} m/s, ball speed"
        " v_b = v_p / r",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


# #############################################################################
# Part 4: Which variables matter
# #############################################################################


def summarize_sensitivity(sweeps: pd.DataFrame) -> pd.DataFrame:
    """
    Measure how much each swept setting moves the results.

    :param sweeps: the rows of `run_all_sweeps()`
    :return: one row per experiment, sport, and setting, with the lowest and
        the highest `P(winner)` over the sweep (`best_low`, `best_high`) and
        `best_range_points` in percentage points, and the same for the mean
        over the half (`mean_low`, `mean_high`, `mean_range_points`); within
        an experiment and a sport, the setting that moves the highest
        `P(winner)` most comes first
    """
    _LOG.debug(hprint.to_str("sweeps.shape"))
    hdbg.dassert_lte(1, len(sweeps), "There must be at least 1 run")
    rows = []
    for (experiment, sport, var), runs in sweeps.groupby(
        ["experiment", "sport", "sweep_var"], sort=False
    ):
        best = runs["max_p_winner"]
        mean = runs["mean_p_winner"]
        rows.append(
            {
                "experiment": experiment,
                "sport": sport,
                "setting": var,
                "best_low": best.min(),
                "best_high": best.max(),
                "best_range_points": 100 * (best.max() - best.min()),
                "mean_low": mean.min(),
                "mean_high": mean.max(),
                "mean_range_points": 100 * (mean.max() - mean.min()),
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["experiment", "sport", "best_range_points"],
        ascending=[True, True, False],
        kind="stable",
    ).reset_index(drop=True)
    return df


# The `figures` directory of the paper, from this file.
PAPER_FIGURES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    "papers",
    "Optimal_strategy_for_racket_sports",
    "figures",
)


def save_paper_figure_data(
    sweeps: pd.DataFrame, serve_map: pd.DataFrame, out_dir: str
) -> Dict[str, str]:
    """
    Save the data of the paper's figures, for `figures/make_figures.py`.

    :param sweeps: the rows of `run_all_sweeps()`
    :param serve_map: the rows of `run_serve_map()`
    :param out_dir: the `figures` directory of the paper
    :return: the path of each file, per name
    """
    _LOG.debug(hprint.to_str("out_dir"))
    hdbg.dassert_dir_exists(out_dir)
    # The sweeps around the defaults: each row is one value of one setting.
    runs = sweeps[sweeps["experiment"] == "sweep_defaults"]
    rows = [
        (
            row["sport"],
            row["sweep_var"],
            row[row["sweep_var"]],
            row["max_p_winner"],
        )
        for _, row in runs.iterrows()
    ]
    sweep_df = pd.DataFrame(
        rows, columns=["sport", "setting", "value", "max_p_winner"]
    )
    # The serve map where the serve matters.
    corner = serve_map[serve_map["group"] == "fast_p1_slow_ball"]
    serve_df = corner[["sport", "x1", "y1", "max_p_winner"]]
    paths = {
        "sweeps": os.path.join(out_dir, "variable_sensitivity_sweeps.csv"),
        "serve_depth": os.path.join(
            out_dir, "variable_sensitivity_serve_depth.csv"
        ),
    }
    sweep_df.to_csv(paths["sweeps"], index=False)
    serve_df.to_csv(paths["serve_depth"], index=False)
    return paths
