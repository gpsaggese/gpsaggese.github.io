"""
Grid-based Monte Carlo scoring for shot placement in racket sports.

Import as:

import research.Optimal_strategy_for_racket_sports.racket_scoring as rosfrsrsc
"""

import dataclasses
import logging
from typing import Literal, Tuple

import numpy as np
import pandas as pd

import helpers.hdbg as hdbg
import helpers.hprint as hprint
from research.Optimal_strategy_for_racket_sports import racket_params
from research.Optimal_strategy_for_racket_sports import racket_trajectory

_LOG = logging.getLogger(__name__)


# #############################################################################
# ShotSituation
# #############################################################################


@dataclasses.dataclass(frozen=True)
class ShotSituation:
    """
    A shot to be scored: who is hitting, from where, into which region.
    """

    # Striker's parameters (contact height, error model).
    striker: racket_params.PlayerParams
    # Striker's court position (x, y), in meters.
    striker_xy: Tuple[float, float]
    # Legal landing region on the other side of the net.
    legal_region: racket_params.CourtRegion

    def __post_init__(self) -> None:
        """
        Validate that the striker is on the negative-y side of the net.
        """
        hdbg.dassert_lt(
            self.striker_xy[1],
            0,
            "striker_xy must be on the negative-y side of the net",
        )


# #############################################################################
# ScoringConfig
# #############################################################################


@dataclasses.dataclass(frozen=True)
class ScoringConfig:
    """
    Monte Carlo sampling configuration for `estimate_launch_table()`.
    """

    # Number of shot-error samples drawn per cell (K in the paper).
    n_samples: int = 2000
    # Candidate launch angles to sweep, in degrees.
    theta_grid_deg: Tuple[float, ...] = tuple(float(d) for d in range(-10, 46))
    # Seed for the shared (common-random-numbers) error draw.
    seed: int = 1

    def __post_init__(self) -> None:
        """
        Validate the scoring configuration.
        """
        hdbg.dassert_lt(0, self.n_samples, "n_samples must be positive")
        hdbg.dassert_lt(
            0, len(self.theta_grid_deg), "theta_grid_deg cannot be empty"
        )


# #############################################################################
# Target grid
# #############################################################################


def make_target_grid(
    region: racket_params.CourtRegion, n_x: int, n_y: int
) -> pd.DataFrame:
    """
    Discretize a court region into a uniform grid of candidate target cells.

    :param region: court region to discretize
    :param n_x: number of grid columns, along x
    :param n_y: number of grid rows, along y
    :return: one row per cell, columns `cell_id`, `x_m`, `y_m`, `x_min`,
        `x_max`, `y_min`, `y_max`; `cell_id` runs row-major, `y` outer, `x`
        inner
    """
    _LOG.debug(hprint.to_str("region n_x n_y"))
    hdbg.dassert_lt(0, n_x, "n_x must be positive")
    hdbg.dassert_lt(0, n_y, "n_y must be positive")
    # Cell boundaries along each axis.
    x_edges = np.linspace(region.x_min, region.x_max, n_x + 1)
    y_edges = np.linspace(region.y_min, region.y_max, n_y + 1)
    rows = []
    cell_id = 0
    for iy in range(n_y):
        for ix in range(n_x):
            rows.append(
                {
                    "cell_id": cell_id,
                    "x_m": (x_edges[ix] + x_edges[ix + 1]) / 2,
                    "y_m": (y_edges[iy] + y_edges[iy + 1]) / 2,
                    "x_min": x_edges[ix],
                    "x_max": x_edges[ix + 1],
                    "y_min": y_edges[iy],
                    "y_max": y_edges[iy + 1],
                }
            )
            cell_id += 1
    grid = pd.DataFrame(rows)
    _LOG.debug("return=%s", hprint.to_str("grid.shape"))
    return grid


# #############################################################################
# Opponent-independent launch table
# #############################################################################


def estimate_launch_table(
    sport: racket_params.SportParams,
    situation: ShotSituation,
    targets: pd.DataFrame,
    config: ScoringConfig,
) -> pd.DataFrame:
    """
    Monte Carlo launch table for every (cell, feasible angle) pair.

    Opponent-independent (Section IV-B): draws one shared (common random
    numbers) set of `config.n_samples` shot errors, then reuses it across
    every cell and every feasible angle so that cell-to-cell differences in
    `p_in` reflect target geometry, not sampling noise.

    :param sport: sport parameters (court geometry, max ball speed)
    :param situation: striker position, contact height, error model, and
        legal landing region
    :param targets: candidate cells, from `make_target_grid()`
    :param config: sampling configuration
    :return: one row per `(cell_id, theta_deg)` feasible for that cell,
        columns `cell_id`, `theta_deg`, `x_m`, `y_m`, `v0_mps`,
        `flight_time_s`, `p_in`, `p_in_se`, `p_hit_cell`
        - `p_in`: fraction of samples that clear the net and land inside
          `situation.legal_region` (Section IV-B, in-bounds reading)
        - `p_hit_cell`: fraction of samples landing inside this cell
          specifically
        - A cell with no feasible angle gets one row with `theta_deg`,
          `v0_mps`, `flight_time_s` all `NaN` and `p_in = p_hit_cell = 0`
    """
    _LOG.debug(hprint.to_str("sport situation config"))
    theta_grid_rad = np.radians(np.array(config.theta_grid_deg, dtype=float))
    rng = np.random.default_rng(config.seed)
    # Common random numbers: one shared draw for every cell and angle.
    errors = racket_trajectory.sample_shot_errors(
        situation.striker.error, config.n_samples, rng
    )
    h0 = situation.striker.contact_height_m
    rows = []
    for _, cell in targets.iterrows():
        cell_id = cell["cell_id"]
        x_m = float(cell["x_m"])
        y_m = float(cell["y_m"])
        target_xy = (x_m, y_m)
        launches = racket_trajectory.get_feasible_launches(
            situation.striker_xy,
            target_xy,
            h0,
            sport,
            theta_grid_rad=theta_grid_rad,
        )
        if len(launches.theta_rad) == 0:
            # No feasible angle for this cell.
            rows.append(
                {
                    "cell_id": cell_id,
                    "theta_deg": np.nan,
                    "x_m": x_m,
                    "y_m": y_m,
                    "v0_mps": np.nan,
                    "flight_time_s": np.nan,
                    "p_in": 0.0,
                    "p_in_se": np.nan,
                    "p_hit_cell": 0.0,
                }
            )
            continue
        landings = racket_trajectory.simulate_landings(
            situation.striker_xy, target_xy, h0, sport, launches, errors
        )
        # In-bounds: clears the net and lands in the legal region.
        is_in = landings.clears_net & situation.legal_region.contains(
            landings.x_m, landings.y_m
        )
        cell_region = racket_params.CourtRegion(
            float(cell["x_min"]),
            float(cell["x_max"]),
            float(cell["y_min"]),
            float(cell["y_max"]),
        )
        is_in_cell = landings.clears_net & cell_region.contains(
            landings.x_m, landings.y_m
        )
        p_in = is_in.mean(axis=1)
        p_in_se = np.sqrt(p_in * (1 - p_in) / config.n_samples)
        p_hit_cell = is_in_cell.mean(axis=1)
        for i, theta in enumerate(launches.theta_rad):
            rows.append(
                {
                    "cell_id": cell_id,
                    "theta_deg": np.degrees(theta),
                    "x_m": x_m,
                    "y_m": y_m,
                    "v0_mps": launches.v0_mps[i],
                    "flight_time_s": launches.flight_time_s[i],
                    "p_in": p_in[i],
                    "p_in_se": p_in_se[i],
                    "p_hit_cell": p_hit_cell[i],
                }
            )
    columns = [
        "cell_id",
        "theta_deg",
        "x_m",
        "y_m",
        "v0_mps",
        "flight_time_s",
        "p_in",
        "p_in_se",
        "p_hit_cell",
    ]
    launch_table = pd.DataFrame(rows).reindex(columns=columns)
    _LOG.debug("return=%s", hprint.to_str("launch_table.shape"))
    return launch_table


# #############################################################################
# Reachability and score
# #############################################################################


def compute_reachability(
    dist_m: np.ndarray,
    flight_time_s: np.ndarray,
    returner: racket_params.PlayerParams,
) -> np.ndarray:
    """
    Whether the returner can reach a cell before the ball's flight ends.

    Deterministic indicator of Section IV-C: the returner reaches the cell
    iff their minimum possible arrival time (reaction time plus travel time
    at top speed) is within the ball's flight time budget. `NaN` values in
    `flight_time_s` (no feasible launch) evaluate to not reachable.

    :param dist_m: distance(s) from the returner to the candidate cell(s),
        in meters
    :param flight_time_s: ball flight time(s) to the candidate cell(s), in
        seconds
    :param returner: returner parameters, providing `reaction_time_s` and
        `move_speed_mps`
    :return: boolean array, True where the returner can reach the cell
    """
    _LOG.debug(hprint.to_str("dist_m flight_time_s"))
    arrival_time = returner.reaction_time_s + dist_m / returner.move_speed_mps
    reachable = arrival_time <= flight_time_s
    return reachable


def compute_score(p_in: np.ndarray, reachable: np.ndarray) -> np.ndarray:
    """
    Composite score `S(c) = P_in(c) * (1 - R(c))` (Section IV-D).

    :param p_in: in-bounds probability per candidate
    :param reachable: reachability indicator per candidate
    :return: composite score per candidate
    """
    _LOG.debug(hprint.to_str("p_in reachable"))
    score = p_in * (1 - reachable.astype(float))
    return score


def score_targets(
    launch_table: pd.DataFrame,
    returner: racket_params.PlayerParams,
    returner_xy: Tuple[float, float],
) -> pd.DataFrame:
    """
    Reachability and composite score for one fixed returner position.

    For each cell, keeps only the angle that maximizes the score, since the
    best angle depends on the opponent (Section IV-D). Cheap relative to
    `estimate_launch_table()`: no new sampling, so this can be re-run for
    many returner positions against one cached `launch_table`.

    :param launch_table: launch table, from `estimate_launch_table()`
    :param returner: returner parameters (reaction time, move speed)
    :param returner_xy: returner's current court position (x, y)
    :return: one row per `cell_id`, columns `cell_id`, `theta_deg`, `x_m`,
        `y_m`, `p_in`, `reachable`, `score`
    """
    _LOG.debug(hprint.to_str("returner returner_xy"))
    scores = launch_table.copy()
    dist_m = np.sqrt(
        (scores["x_m"] - returner_xy[0]) ** 2
        + (scores["y_m"] - returner_xy[1]) ** 2
    )
    reachable = compute_reachability(
        dist_m.to_numpy(), scores["flight_time_s"].to_numpy(), returner
    )
    score = compute_score(scores["p_in"].to_numpy(), reachable)
    scores["reachable"] = reachable
    scores["score"] = score
    # Keep the best angle per cell.
    best_idx = scores.groupby("cell_id")["score"].idxmax()
    columns = [
        "cell_id",
        "theta_deg",
        "x_m",
        "y_m",
        "p_in",
        "reachable",
        "score",
    ]
    best = scores.loc[best_idx, columns].reset_index(drop=True)
    _LOG.debug("return=%s", hprint.to_str("best.shape"))
    return best


def select_best_cell(scores: pd.DataFrame) -> pd.Series:
    """
    Argmax cell of `score_targets()`'s output (`c^star` of Section IV-D).

    :param scores: per-cell scores, from `score_targets()`
    :return: row of `scores` with the highest `score`
    """
    _LOG.debug(hprint.to_str("scores"))
    best_cell = scores.loc[scores["score"].idxmax()]
    return best_cell


# #############################################################################
# Situations
# #############################################################################


def make_rally_situation(
    sport: racket_params.SportParams,
    striker: racket_params.PlayerParams,
    striker_xy: Tuple[float, float],
) -> ShotSituation:
    """
    Rally shot situation: the returner's full half court is the legal region.

    :param sport: sport parameters, for the court geometry
    :param striker: striker parameters (contact height, error model)
    :param striker_xy: striker's court position (x, y), with `y < 0`
    :return: shot situation targeting the returner's half court
    """
    _LOG.debug(hprint.to_str("sport striker striker_xy"))
    legal_region = racket_params.get_half_court_region(sport.court)
    situation = ShotSituation(
        striker=striker, striker_xy=striker_xy, legal_region=legal_region
    )
    return situation


def make_serve_situation(
    sport: racket_params.SportParams,
    server: racket_params.PlayerParams,
    serve_side: Literal["deuce", "ad"],
) -> ShotSituation:
    """
    Serve situation: the legal region is the service box (Section IV-E).

    The server stands behind their own baseline, on the same lateral side
    as the target service box.

    :param sport: sport parameters, for the court geometry
    :param server: server parameters (contact height, error model)
    :param serve_side: which service box to serve into, "deuce" or "ad"
    :return: shot situation targeting the given service box
    """
    _LOG.debug(hprint.to_str("sport server serve_side"))
    legal_region = racket_params.get_service_box_region(sport.court, serve_side)
    x_mid = (legal_region.x_min + legal_region.x_max) / 2
    striker_xy = (x_mid, -sport.court.length_m / 2)
    situation = ShotSituation(
        striker=server, striker_xy=striker_xy, legal_region=legal_region
    )
    return situation
