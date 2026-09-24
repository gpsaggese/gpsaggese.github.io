"""
Plotting helpers for the `racket_trajectory.API` notebook.

Import as:

import research.Optimal_strategy_for_racket_sports.notebooks.racket_trajectory_utils as rosfrsntrtu
"""

import logging
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import helpers.hdbg as hdbg
import helpers.hprint as hprint
from research.Optimal_strategy_for_racket_sports import racket_params
from research.Optimal_strategy_for_racket_sports import racket_trajectory

_LOG = logging.getLogger(__name__)


# #############################################################################
# Cell 2.4: Vertical-plane trajectory
# #############################################################################


def plot_trajectory(
    v0_mps: float,
    theta_rad: float,
    h0_m: float,
    d_net_m: float,
    net_height_m: float,
    d_target_m: float,
    *,
    figsize: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Plot the vertical-plane arc `z(x)` for one launch, with the net and
    target landing point marked.

    :param v0_mps: launch speed, in m/s
    :param theta_rad: launch angle, in radians
    :param h0_m: contact height, in meters
    :param d_net_m: horizontal distance from the striker to the net, in
        meters
    :param net_height_m: net height at the lateral crossing position, in
        meters
    :param d_target_m: horizontal distance from the striker to the target,
        in meters
    :param figsize: figure size
        - Default: `plt.rcParams["figure.figsize"]`
    """
    _LOG.debug(hprint.to_str("v0_mps theta_rad h0_m d_net_m"))
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    # Sample the arc from the striker to the target, in a grid of horizontal
    # distances.
    x_grid = np.linspace(0, d_target_m, 100)
    v0_grid = np.full_like(x_grid, v0_mps)
    theta_grid = np.full_like(x_grid, theta_rad)
    z_grid = racket_trajectory.get_height_at(x_grid, v0_grid, theta_grid, h0_m)
    _, ax = plt.subplots(figsize=figsize)
    ax.plot(x_grid, z_grid, color="C0", label="Ball trajectory")
    # Mark the net as a vertical segment from the ground to its height.
    ax.plot([d_net_m, d_net_m], [0, net_height_m], color="C1", label="Net")
    # Mark the target landing point on the ground.
    ax.scatter([d_target_m], [0], color="C2", zorder=5, label="Target")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Horizontal distance from striker (m)")
    ax.set_ylabel("Height (m)")
    ax.set_title("Vertical-plane trajectory")
    ax.legend()
    plt.show()


# #############################################################################
# Cell 4.3: Landing dispersion
# #############################################################################


def plot_landing_scatter(
    x_m: np.ndarray,
    y_m: np.ndarray,
    target_xy: Tuple[float, float],
    court: racket_params.CourtGeometry,
    *,
    figsize: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Scatter plot of simulated landing points around a nominal target.

    :param x_m: landing x coordinates, any shape (flattened for plotting)
    :param y_m: landing y coordinates, same shape as `x_m`
    :param target_xy: nominal target (x, y), in meters
    :param court: court geometry, used to draw the sideline bounds
    :param figsize: figure size
        - Default: `plt.rcParams["figure.figsize"]`
    """
    _LOG.debug(hprint.to_str("target_xy"))
    hdbg.dassert_eq(x_m.shape, y_m.shape, "x_m and y_m must have the same shape")
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    _, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        x=x_m.flatten(), y=y_m.flatten(), alpha=0.5, ax=ax, label="Landings"
    )
    ax.scatter(
        [target_xy[0]],
        [target_xy[1]],
        color="C3",
        marker="x",
        s=100,
        zorder=5,
        label="Nominal target",
    )
    # Draw the singles sidelines and the net line for spatial context.
    width_half = court.width_m / 2
    ax.axvline(-width_half, color="black", linewidth=0.5, linestyle="--")
    ax.axvline(width_half, color="black", linewidth=0.5, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Simulated landing dispersion")
    ax.legend()
    plt.show()
