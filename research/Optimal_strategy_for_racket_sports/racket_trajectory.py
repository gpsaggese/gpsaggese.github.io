"""
Closed-form ball flight and error propagation for racket sports.

Import as:

import research.Optimal_strategy_for_racket_sports.racket_trajectory as rosfrsrtr
"""

import dataclasses
import logging
from typing import Tuple

import numpy as np

import helpers.hdbg as hdbg
import helpers.hprint as hprint
from research.Optimal_strategy_for_racket_sports import racket_params

_LOG = logging.getLogger(__name__)


# #############################################################################
# FeasibleLaunches
# #############################################################################


@dataclasses.dataclass(frozen=True)
class FeasibleLaunches:
    """
    Feasible (theta, v0) launch candidates aimed at one target, shape (L,).
    """

    # Launch angles, in radians.
    theta_rad: np.ndarray
    # Launch speeds, in m/s.
    v0_mps: np.ndarray
    # Time of flight for each launch, in seconds.
    flight_time_s: np.ndarray
    # Vertical clearance over the net at the net crossing point, in meters.
    net_clearance_m: np.ndarray


# #############################################################################
# ShotErrors
# #############################################################################


@dataclasses.dataclass(frozen=True)
class ShotErrors:
    """
    Sampled shot execution errors, shape (K,).
    """

    # Sampled launch-angle error, in radians.
    d_theta_rad: np.ndarray
    # Sampled launch-speed error, as a fraction of the nominal speed.
    d_v_frac: np.ndarray
    # Sampled lateral aim-angle error, in radians.
    phi_rad: np.ndarray


# #############################################################################
# Landings
# #############################################################################


@dataclasses.dataclass(frozen=True)
class Landings:
    """
    Simulated landing points for every (launch, error sample) pair, shape
    (L, K).
    """

    # Landing x coordinate, in meters.
    x_m: np.ndarray
    # Landing y coordinate, in meters.
    y_m: np.ndarray
    # Time of flight to landing, in seconds.
    flight_time_s: np.ndarray
    # True where the trajectory cleared the net.
    clears_net: np.ndarray


# #############################################################################
# Vertical-plane trajectory
# #############################################################################


def get_flight_time(v0: np.ndarray, theta: np.ndarray, h0: float) -> np.ndarray:
    """
    Time of flight until the ball returns to ground level (`z = 0`).

    :param v0: launch speed(s), in m/s
    :param theta: launch angle(s), in radians
    :param h0: contact height, in meters
    :return: flight time(s), in seconds
    """
    _LOG.debug(hprint.to_str("h0"))
    # Vertical component of the launch velocity.
    vz0 = v0 * np.sin(theta)
    # Discriminant of the quadratic equation for `z(t) = 0`.
    discriminant = vz0**2 + 2 * racket_params.GRAVITY_MPS2 * h0
    # Positive root of the quadratic equation, i.e., the landing time.
    flight_time = (vz0 + np.sqrt(discriminant)) / racket_params.GRAVITY_MPS2
    return flight_time


def get_height_at(
    x: np.ndarray, v0: np.ndarray, theta: np.ndarray, h0: float
) -> np.ndarray:
    """
    Ball height at horizontal distance `x` along the flight path.

    :param x: horizontal distance(s) traveled, in meters
    :param v0: launch speed(s), in m/s
    :param theta: launch angle(s), in radians
    :param h0: contact height, in meters
    :return: ball height(s) at `x`, in meters
    """
    _LOG.debug(hprint.to_str("h0"))
    # Projectile height at `x`: linear term minus the gravity drop term.
    height = (
        h0
        + x * np.tan(theta)
        - racket_params.GRAVITY_MPS2 * x**2 / (2 * v0**2 * np.cos(theta) ** 2)
    )
    return height


def solve_launch_speed(
    d_target: np.ndarray, theta: np.ndarray, h0: float
) -> np.ndarray:
    """
    Launch speed that lands the ball exactly at `d_target`.

    :param d_target: target horizontal distance(s), in meters
    :param theta: launch angle(s), in radians
    :param h0: contact height, in meters
    :return: required launch speed(s), in m/s
        - NaN where the angle cannot reach `d_target` from height `h0` (i.e.,
          `h0 + d_target * tan(theta) <= 0`)
    """
    _LOG.debug(hprint.to_str("h0"))
    # Denominator of the launch-speed formula; must be positive for `theta`
    # to be able to reach `d_target` from height `h0`.
    denom = h0 + d_target * np.tan(theta)
    # Whether each angle can reach `d_target` at all.
    is_feasible = denom > 0
    # Avoid `sqrt` of a non-positive value; the invalid branch is masked out
    # below.
    safe_denom = np.where(is_feasible, denom, 1.0)
    # Launch speed that lands exactly at `d_target`.
    v0 = (d_target / np.cos(theta)) * np.sqrt(
        racket_params.GRAVITY_MPS2 / (2 * safe_denom)
    )
    # Mask out the infeasible angles.
    v0 = np.where(is_feasible, v0, np.nan)
    return v0


# #############################################################################
# Feasible launch set and Monte Carlo landing simulation
# #############################################################################


def get_feasible_launches(
    striker_xy: Tuple[float, float],
    target_xy: Tuple[float, float],
    h0: float,
    sport: racket_params.SportParams,
    *,
    theta_grid_rad: np.ndarray,
) -> FeasibleLaunches:
    """
    Feasible launch angles/speeds landing at `target_xy` from `striker_xy`.

    A launch is feasible when it clears the net, lands at `target_xy`, and
    requires a speed at or below `sport.max_ball_speed_mps`.

    :param striker_xy: striker contact point (x, y), in meters
    :param target_xy: target landing point (x, y), in meters
    :param h0: contact height, in meters
    :param sport: sport parameters, providing `max_ball_speed_mps` and the
        court geometry for the net height
    :param theta_grid_rad: grid of launch angles to evaluate, in radians
    :return: feasible launches, one entry per angle in `theta_grid_rad` that
        satisfies the speed and net-clearance constraints
    """
    _LOG.debug(hprint.to_str("striker_xy target_xy h0"))
    # Striker contact point coordinates.
    x_s, y_s = striker_xy
    # Target landing point coordinates.
    x_t, y_t = target_xy
    hdbg.dassert_lt(y_s, 0, "Striker must be on the negative-y side of the net")
    # Straight-line ground distance from the striker to the target.
    d_target = np.sqrt((x_t - x_s) ** 2 + (y_t - y_s) ** 2)
    # Launch speed required for each candidate angle to reach `d_target`.
    v0 = solve_launch_speed(d_target, theta_grid_rad, h0)
    # Time of flight for each candidate launch.
    flight_time = get_flight_time(v0, theta_grid_rad, h0)
    # Net crossing point along the straight ground track from striker to
    # target: a scalar fraction of `d_target`, independent of theta.
    s_net = -y_s / (y_t - y_s)
    # Distance from the striker to the net crossing point.
    d_net = s_net * d_target
    # Lateral (x) position where the trajectory crosses the net.
    x_net = x_s + s_net * (x_t - x_s)
    # Ball height at the net crossing point, for each candidate launch.
    height_at_net = get_height_at(d_net, v0, theta_grid_rad, h0)
    # Net height at the lateral crossing position.
    net_height = racket_params.get_net_height(sport.court, x_net)
    # Vertical clearance over the net (positive means the ball clears it).
    net_clearance = height_at_net - net_height
    # Keep only angles that are feasible, clear the net, and stay within the
    # sport's max ball speed.
    is_feasible = (
        ~np.isnan(v0) & (v0 <= sport.max_ball_speed_mps) & (net_clearance > 0)
    )
    # Collect the surviving launches.
    result = FeasibleLaunches(
        theta_rad=theta_grid_rad[is_feasible],
        v0_mps=v0[is_feasible],
        flight_time_s=flight_time[is_feasible],
        net_clearance_m=net_clearance[is_feasible],
    )
    _LOG.debug("return=%s", result)
    return result


def sample_shot_errors(
    error: racket_params.ShotErrorModel,
    n_samples: int,
    rng: np.random.Generator,
) -> ShotErrors:
    """
    Draw `n_samples` shot execution errors from `error`.

    :param error: shot error model giving the three error standard deviations
    :param n_samples: number of samples to draw
    :param rng: random generator to draw from, passed in explicitly so the
        same draws (common random numbers) can be reused across targets
    :return: sampled errors, each array of shape `(n_samples,)`
    """
    _LOG.debug(hprint.to_str("n_samples"))
    # Sampled launch-angle error.
    d_theta_rad = rng.normal(0, error.sigma_theta_rad, n_samples)
    # Sampled launch-speed error, as a fraction of the nominal speed.
    d_v_frac = rng.normal(0, error.sigma_v_frac, n_samples)
    # Sampled lateral aim-angle error.
    phi_rad = rng.normal(0, error.sigma_phi_rad, n_samples)
    # Collect the sampled errors.
    result = ShotErrors(
        d_theta_rad=d_theta_rad, d_v_frac=d_v_frac, phi_rad=phi_rad
    )
    return result


def simulate_landings(
    striker_xy: Tuple[float, float],
    target_xy: Tuple[float, float],
    h0: float,
    sport: racket_params.SportParams,
    launches: FeasibleLaunches,
    errors: ShotErrors,
) -> Landings:
    """
    Simulate landing points for every (launch, error sample) pair.

    Each of the `L` nominal launches (aimed at `target_xy`) is perturbed by
    each of the `K` sampled errors: `theta` and `v0` shift the vertical-plane
    trajectory, `phi` rotates the azimuth away from the straight line to
    `target_xy`.

    :param striker_xy: striker contact point (x, y), in meters
    :param target_xy: nominal target landing point (x, y), in meters
    :param h0: contact height, in meters
    :param sport: sport parameters, providing the court geometry for the net
        height
    :param launches: `L` feasible nominal launches aimed at `target_xy`
    :param errors: `K` sampled shot execution errors
    :return: landings of shape `(L, K)`
    """
    _LOG.debug(hprint.to_str("striker_xy target_xy h0"))
    # Striker contact point coordinates.
    x_s, y_s = striker_xy
    # Nominal target landing point coordinates.
    x_t, y_t = target_xy
    hdbg.dassert_lt(y_s, 0, "Striker must be on the negative-y side of the net")
    # Nominal azimuth from the striker to the target, measured from the `+y`
    # axis.
    psi = np.arctan2(x_t - x_s, y_t - y_s)
    # Broadcast the `L` nominal launches against the `K` error samples to get
    # (L, K) perturbed trajectories.
    # Perturbed launch angle for every (launch, error sample) pair.
    theta = launches.theta_rad[:, np.newaxis] + errors.d_theta_rad[np.newaxis, :]
    # Perturbed launch speed for every (launch, error sample) pair.
    v0 = launches.v0_mps[:, np.newaxis] * (1 + errors.d_v_frac[np.newaxis, :])
    # Perturbed azimuth for every (launch, error sample) pair.
    alpha = np.broadcast_to(psi + errors.phi_rad[np.newaxis, :], theta.shape)
    # Landing point: horizontal distance traveled at the time the ball
    # returns to ground level, along the perturbed azimuth `alpha`.
    # Time of flight for every perturbed trajectory.
    flight_time = get_flight_time(v0, theta, h0)
    # Ground distance traveled along the perturbed azimuth.
    d_land = v0 * np.cos(theta) * flight_time
    # Landing x coordinate.
    x_land = x_s + d_land * np.sin(alpha)
    # Landing y coordinate.
    y_land = y_s + d_land * np.cos(alpha)
    # Net clearance along the same perturbed azimuth (paper Section III-C):
    # the net crossing distance and lateral position depend on `alpha`.
    # Cosine of the perturbed azimuth, used to project onto the net plane.
    cos_alpha = np.cos(alpha)
    hdbg.dassert(
        bool((cos_alpha > 0).all()),
        "Every sampled shot azimuth must cross the net",
    )
    # Distance from the striker to the net crossing point, along `alpha`.
    d_net = -y_s / cos_alpha
    # Lateral (x) position where the trajectory crosses the net.
    x_net = x_s - y_s * np.tan(alpha)
    # Ball height at the net crossing point.
    height_at_net = get_height_at(d_net, v0, theta, h0)
    # Net height at the lateral crossing position.
    net_height = racket_params.get_net_height(sport.court, x_net)
    # True where the trajectory clears the net.
    clears_net = height_at_net > net_height
    # Collect the simulated landings.
    result = Landings(
        x_m=x_land,
        y_m=y_land,
        flight_time_s=flight_time,
        clears_net=clears_net,
    )
    return result
