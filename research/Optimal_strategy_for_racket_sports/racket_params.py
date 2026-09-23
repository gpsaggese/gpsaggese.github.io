"""
Parameters and geometry for racket sports models.

Import as:

import research.Optimal_strategy_for_racket_sports.racket_params as rosfrsrpa
"""

import dataclasses
import logging
from typing import Literal

import numpy as np

import helpers.hdbg as hdbg
import helpers.hprint as hprint

_LOG = logging.getLogger(__name__)

# #############################################################################
# Constants
# #############################################################################


GRAVITY_MPS2 = 9.81


# #############################################################################
# CourtGeometry
# #############################################################################


@dataclasses.dataclass(frozen=True)
class CourtGeometry:
    """
    Court dimensions and net geometry.
    """

    length_m: float
    width_m: float
    net_height_center_m: float
    net_height_post_m: float
    non_volley_zone_m: float = 0.0
    service_line_m: float = 0.0

    def __post_init__(self) -> None:
        """
        Validate the court geometry parameters.
        """
        hdbg.dassert_lt(0, self.length_m, "length_m must be positive")
        hdbg.dassert_lt(0, self.width_m, "width_m must be positive")
        hdbg.dassert_lt(
            0,
            self.net_height_center_m,
            "net_height_center_m must be positive",
        )
        hdbg.dassert_lt(
            0, self.net_height_post_m, "net_height_post_m must be positive"
        )


# #############################################################################
# CourtRegion
# #############################################################################


@dataclasses.dataclass(frozen=True)
class CourtRegion:
    """
    Rectangular region on the court.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        """
        Validate the region bounds.
        """
        hdbg.dassert_lt(self.x_min, self.x_max, "x_min must be less than x_max")
        hdbg.dassert_lt(self.y_min, self.y_max, "y_min must be less than y_max")

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Check if points (x, y) are in the region.

        :param x: array of x coordinates to test
        :param y: array of y coordinates to test
        :return: boolean array, True where the (x, y) point lies within
            the region
        """
        _LOG.debug(hprint.to_str("x y"))
        return (
            (x >= self.x_min)
            & (x <= self.x_max)
            & (y >= self.y_min)
            & (y <= self.y_max)
        )


# #############################################################################
# SportParams
# #############################################################################


@dataclasses.dataclass(frozen=True)
class SportParams:
    """
    Sport-specific parameters.
    """

    name: str
    court: CourtGeometry
    max_ball_speed_mps: float

    def __post_init__(self) -> None:
        """
        Validate the sport parameters.
        """
        hdbg.dassert_ne(self.name, "", "name cannot be empty")
        hdbg.dassert_lt(
            0,
            self.max_ball_speed_mps,
            "max_ball_speed_mps must be positive",
        )


# #############################################################################
# ShotErrorModel
# #############################################################################


@dataclasses.dataclass(frozen=True)
class ShotErrorModel:
    """
    Error model for shot execution.
    """

    sigma_theta_rad: float
    sigma_v_frac: float
    sigma_phi_rad: float

    def __post_init__(self) -> None:
        """
        Validate the shot error parameters.
        """
        hdbg.dassert_lte(
            0, self.sigma_theta_rad, "sigma_theta_rad must be non-negative"
        )
        hdbg.dassert_lte(
            0, self.sigma_v_frac, "sigma_v_frac must be non-negative"
        )
        hdbg.dassert_lte(
            0, self.sigma_phi_rad, "sigma_phi_rad must be non-negative"
        )


# #############################################################################
# PlayerParams
# #############################################################################


@dataclasses.dataclass(frozen=True)
class PlayerParams:
    """
    Player-specific parameters.
    """

    reaction_time_s: float
    move_speed_mps: float
    contact_height_m: float
    error: ShotErrorModel

    def __post_init__(self) -> None:
        """
        Validate the player parameters.
        """
        hdbg.dassert_lte(
            0, self.reaction_time_s, "reaction_time_s must be non-negative"
        )
        hdbg.dassert_lt(
            0, self.move_speed_mps, "move_speed_mps must be positive"
        )
        hdbg.dassert_lt(
            0, self.contact_height_m, "contact_height_m must be positive"
        )


# #############################################################################
# Court geometry helpers
# #############################################################################


def get_net_height(court: CourtGeometry, x: np.ndarray) -> np.ndarray:
    """
    Net height as a function of lateral position x.

    Linear interpolation from center height to post height at the singles
    sideline. Origin at net center; x is lateral (0 on center line).

    :param court: court geometry providing the net height parameters
    :param x: lateral position(s) at which to evaluate the net height
    :return: net height(s) at the given lateral position(s)
    """
    _LOG.debug(hprint.to_str("court x"))
    x_abs = np.abs(x)
    x_post = court.width_m / 2
    # Interpolate linearly between the center and post heights up to the
    # sideline, then hold flat beyond it.
    height = np.where(
        x_abs <= x_post,
        court.net_height_center_m
        + (court.net_height_post_m - court.net_height_center_m)
        * (x_abs / x_post),
        court.net_height_post_m,
    )
    return height


def get_half_court_region(court: CourtGeometry) -> CourtRegion:
    """
    Region of the returner's half of the court (y > 0).

    :param court: court geometry to compute the half-court region for
    :return: rectangular region covering the returner's half of the court
    """
    _LOG.debug(hprint.to_str("court"))
    region = CourtRegion(
        x_min=-court.width_m / 2,
        x_max=court.width_m / 2,
        y_min=0,
        y_max=court.length_m / 2,
    )
    _LOG.debug("return=%s", region)
    return region


def get_service_box_region(
    court: CourtGeometry, serve_side: Literal["deuce", "ad"]
) -> CourtRegion:
    """
    Service box region.

    :param court: court geometry to compute the service box for
    :param serve_side: "deuce" -> x in [-W/2, 0], "ad" -> x in [0, W/2]
    :return: rectangular region covering the specified service box
    """
    _LOG.debug(hprint.to_str("court serve_side"))
    hdbg.dassert_in(serve_side, ("deuce", "ad"), "Invalid serve_side")
    service_line_y = court.service_line_m
    nv_zone_y = court.non_volley_zone_m
    width_half = court.width_m / 2
    # Deuce side spans the left half of the court width, ad side the right
    # half.
    if serve_side == "deuce":
        x_min = -width_half
        x_max = 0
    else:  # ad
        x_min = 0
        x_max = width_half
    region = CourtRegion(
        x_min=x_min, x_max=x_max, y_min=nv_zone_y, y_max=service_line_y
    )
    _LOG.debug("return=%s", region)
    return region


# #############################################################################
# Presets
# #############################################################################


TENNIS = SportParams(
    name="Tennis",
    court=CourtGeometry(
        length_m=23.77,
        width_m=8.23,
        net_height_center_m=0.914,
        net_height_post_m=1.07,
        non_volley_zone_m=0.0,
        service_line_m=6.40,
    ),
    max_ball_speed_mps=73.14,
)

PICKLEBALL = SportParams(
    name="Pickleball",
    court=CourtGeometry(
        length_m=13.41,
        width_m=6.10,
        net_height_center_m=0.867,
        net_height_post_m=0.867,
        non_volley_zone_m=2.13,
        service_line_m=6.71,
    ),
    max_ball_speed_mps=24.59,
)

DEFAULT_ERROR = ShotErrorModel(
    sigma_theta_rad=np.radians(1.5),
    sigma_v_frac=0.05,
    sigma_phi_rad=np.radians(1.5),
)

DEFAULT_PLAYER = PlayerParams(
    reaction_time_s=0.2,
    move_speed_mps=1.5,
    contact_height_m=1.0,
    error=DEFAULT_ERROR,
)
