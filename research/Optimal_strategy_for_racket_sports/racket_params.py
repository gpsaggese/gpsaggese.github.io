"""Parameters and geometry for racket sports models."""

import dataclasses
from typing import Literal

import numpy as np

GRAVITY_MPS2 = 9.81


@dataclasses.dataclass(frozen=True)
class CourtGeometry:
    """Court dimensions and net geometry."""

    length_m: float
    width_m: float
    net_height_center_m: float
    net_height_post_m: float
    non_volley_zone_m: float = 0.0
    service_line_m: float = 0.0

    def __post_init__(self) -> None:
        if self.length_m <= 0:
            raise ValueError(f"length_m must be positive, got {self.length_m}")
        if self.width_m <= 0:
            raise ValueError(f"width_m must be positive, got {self.width_m}")
        if self.net_height_center_m <= 0:
            raise ValueError(
                f"net_height_center_m must be positive, got {self.net_height_center_m}"
            )
        if self.net_height_post_m <= 0:
            raise ValueError(
                f"net_height_post_m must be positive, got {self.net_height_post_m}"
            )


@dataclasses.dataclass(frozen=True)
class CourtRegion:
    """Rectangular region on the court."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        if self.x_min >= self.x_max:
            raise ValueError(
                f"x_min must be < x_max, got {self.x_min} >= {self.x_max}"
            )
        if self.y_min >= self.y_max:
            raise ValueError(
                f"y_min must be < y_max, got {self.y_min} >= {self.y_max}"
            )

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Check if points (x, y) are in the region."""
        return (
            (x >= self.x_min)
            & (x <= self.x_max)
            & (y >= self.y_min)
            & (y <= self.y_max)
        )


@dataclasses.dataclass(frozen=True)
class SportParams:
    """Sport-specific parameters."""

    name: str
    court: CourtGeometry
    max_ball_speed_mps: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name cannot be empty")
        if self.max_ball_speed_mps <= 0:
            raise ValueError(
                f"max_ball_speed_mps must be positive, got {self.max_ball_speed_mps}"
            )


@dataclasses.dataclass(frozen=True)
class ShotErrorModel:
    """Error model for shot execution."""

    sigma_theta_rad: float
    sigma_v_frac: float
    sigma_phi_rad: float

    def __post_init__(self) -> None:
        if self.sigma_theta_rad < 0:
            raise ValueError(
                f"sigma_theta_rad must be non-negative, got {self.sigma_theta_rad}"
            )
        if self.sigma_v_frac < 0:
            raise ValueError(
                f"sigma_v_frac must be non-negative, got {self.sigma_v_frac}"
            )
        if self.sigma_phi_rad < 0:
            raise ValueError(
                f"sigma_phi_rad must be non-negative, got {self.sigma_phi_rad}"
            )


@dataclasses.dataclass(frozen=True)
class PlayerParams:
    """Player-specific parameters."""

    reaction_time_s: float
    move_speed_mps: float
    contact_height_m: float
    error: ShotErrorModel

    def __post_init__(self) -> None:
        if self.reaction_time_s < 0:
            raise ValueError(
                f"reaction_time_s must be non-negative, got {self.reaction_time_s}"
            )
        if self.move_speed_mps <= 0:
            raise ValueError(
                f"move_speed_mps must be positive, got {self.move_speed_mps}"
            )
        if self.contact_height_m <= 0:
            raise ValueError(
                f"contact_height_m must be positive, got {self.contact_height_m}"
            )


def get_net_height(court: CourtGeometry, x: np.ndarray) -> np.ndarray:
    """Net height as a function of lateral position x.

    Linear interpolation from center height to post height at the singles sideline.
    Origin at net center; x is lateral (0 on center line).
    """
    x_abs = np.abs(x)
    x_post = court.width_m / 2

    height = np.where(
        x_abs <= x_post,
        court.net_height_center_m
        + (court.net_height_post_m - court.net_height_center_m)
        * (x_abs / x_post),
        court.net_height_post_m,
    )
    return height


def get_half_court_region(court: CourtGeometry) -> CourtRegion:
    """Region of the returner's half of the court (y > 0)."""
    return CourtRegion(
        x_min=-court.width_m / 2,
        x_max=court.width_m / 2,
        y_min=0,
        y_max=court.length_m / 2,
    )


def get_service_box_region(
    court: CourtGeometry, serve_side: Literal["deuce", "ad"]
) -> CourtRegion:
    """Service box region.

    Args:
        serve_side: "deuce" -> x in [-W/2, 0], "ad" -> x in [0, W/2]
    """
    if serve_side not in ("deuce", "ad"):
        raise ValueError(f"serve_side must be 'deuce' or 'ad', got {serve_side}")

    service_line_y = court.service_line_m
    nv_zone_y = court.non_volley_zone_m
    width_half = court.width_m / 2

    if serve_side == "deuce":
        x_min = -width_half
        x_max = 0
    else:  # ad
        x_min = 0
        x_max = width_half

    return CourtRegion(
        x_min=x_min, x_max=x_max, y_min=nv_zone_y, y_max=service_line_y
    )


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
