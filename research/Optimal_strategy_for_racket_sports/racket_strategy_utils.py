"""
Plotting and widget helpers for the `racket_strategy` notebooks.

Import as:

import research.Optimal_strategy_for_racket_sports.racket_strategy_utils as rosfrsrsu
"""

import dataclasses
import logging
from typing import Dict, Optional, Tuple

import ipywidgets
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output, display
from matplotlib.axes import Axes

import helpers.hdbg as hdbg
import helpers.hprint as hprint
import helpers.htutorial as htutori
from research.Optimal_strategy_for_racket_sports import racket_params
from research.Optimal_strategy_for_racket_sports import racket_scoring
from research.Optimal_strategy_for_racket_sports import racket_trajectory

_LOG = logging.getLogger(__name__)


# #############################################################################
# Court drawing
# #############################################################################


def draw_court(ax: Axes, court: racket_params.CourtGeometry) -> None:
    """
    Draw the court boundary, net, and service/non-volley lines on `ax`.

    :param ax: axes to draw on
    :param court: court geometry to draw
    """
    _LOG.debug(hprint.to_str("court"))
    half_length = court.length_m / 2
    half_width = court.width_m / 2
    # Outer court boundary.
    ax.plot(
        [-half_width, half_width, half_width, -half_width, -half_width],
        [-half_length, -half_length, half_length, half_length, -half_length],
        color="black",
        linewidth=1.5,
    )
    # Net line.
    ax.axhline(0, color="black", linewidth=2.0)
    # Service lines, if the sport defines them.
    if court.service_line_m > 0:
        ax.axhline(
            court.service_line_m, color="gray", linestyle="--", linewidth=1.0
        )
        ax.axhline(
            -court.service_line_m, color="gray", linestyle="--", linewidth=1.0
        )
    # Non-volley zone ("kitchen") lines, if the sport defines them.
    if court.non_volley_zone_m > 0:
        ax.axhline(
            court.non_volley_zone_m, color="gray", linestyle=":", linewidth=1.0
        )
        ax.axhline(
            -court.non_volley_zone_m, color="gray", linestyle=":", linewidth=1.0
        )
    ax.set_xlim(-half_width - 1, half_width + 1)
    ax.set_ylim(-half_length - 1, half_length + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


# #############################################################################
# Trajectory fan
# #############################################################################


def plot_trajectory_fan(
    sport: racket_params.SportParams,
    player: racket_params.PlayerParams,
    striker_xy: Tuple[float, float],
    target_xy: Tuple[float, float],
    theta_deg: float,
    *,
    ax: Optional[Axes] = None,
) -> None:
    """
    Plot the nominal trajectory plus a fan of perturbed ones (Figure 1).

    :param sport: sport parameters, for the net geometry
    :param player: striking player, for contact height and error model
    :param striker_xy: striker contact point (x, y), in meters
    :param target_xy: target landing point (x, y), in meters
    :param theta_deg: nominal launch angle, in degrees
    :param ax: axes to draw on
        - Default: a new figure and axes
    """
    _LOG.debug(hprint.to_str("striker_xy target_xy theta_deg"))
    if ax is None:
        _, ax = plt.subplots(figsize=plt.rcParams["figure.figsize"])
    h0 = player.contact_height_m
    x_s, y_s = striker_xy
    x_t, y_t = target_xy
    d_target = np.sqrt((x_t - x_s) ** 2 + (y_t - y_s) ** 2)
    theta_rad = np.radians(theta_deg)
    v0 = racket_trajectory.solve_launch_speed(
        np.array([d_target]), np.array([theta_rad]), h0
    )[0]
    x_grid = np.linspace(0, d_target, 200)
    z_nominal = racket_trajectory.get_height_at(
        x_grid, np.full_like(x_grid, v0), np.full_like(x_grid, theta_rad), h0
    )
    ax.plot(x_grid, z_nominal, color="C0", linewidth=2.0, label="Nominal")
    # Fan of trajectories perturbed by +/-1 and +/-2 sigma in angle and speed.
    sigma_theta = player.error.sigma_theta_rad
    sigma_v = player.error.sigma_v_frac
    perturbations = [
        (sigma_theta, -sigma_v),
        (-sigma_theta, sigma_v),
        (2 * sigma_theta, -2 * sigma_v),
        (-2 * sigma_theta, 2 * sigma_v),
    ]
    for i, (d_theta, d_v_frac) in enumerate(perturbations):
        theta_p = theta_rad + d_theta
        v0_p = v0 * (1 + d_v_frac)
        z_p = racket_trajectory.get_height_at(
            x_grid,
            np.full_like(x_grid, v0_p),
            np.full_like(x_grid, theta_p),
            h0,
        )
        label = "Perturbed" if i == 0 else None
        ax.plot(
            x_grid,
            z_p,
            color="C0",
            linewidth=1.0,
            linestyle="--",
            alpha=0.5,
            label=label,
        )
    # Net crossing point and height, along the striker-to-target line.
    s_net = -y_s / (y_t - y_s)
    d_net = s_net * d_target
    x_net = x_s + s_net * (x_t - x_s)
    net_height = racket_params.get_net_height(sport.court, np.array([x_net]))[0]
    ax.plot(
        [d_net, d_net], [0, net_height], color="C1", linewidth=2.0, label="Net"
    )
    ax.scatter([d_target], [0], color="C2", zorder=5, label="Target")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Horizontal distance from striker (m)")
    ax.set_ylabel("Height (m)")
    ax.set_title(f"Trajectory fan: theta={theta_deg:.1f} deg, v0={v0:.2f} m/s")
    ax.legend()


# #############################################################################
# Court heatmap
# #############################################################################


def plot_court_heatmap(
    scores: pd.DataFrame,
    column: str,
    sport: racket_params.SportParams,
    *,
    markers: Optional[Dict[str, Tuple[float, float]]] = None,
    ax: Optional[Axes] = None,
) -> None:
    """
    Draw one colored patch per cell in `scores`, over the court outline.

    :param scores: per-cell values, from `racket_scoring.score_targets()`,
        columns include `x_m`, `y_m`, and `column`; cells are assumed to sit
        on a regular grid, so the patch size is inferred from the spacing
        between cell centers
    :param column: name of the column in `scores` to color cells by
    :param sport: sport parameters, for the court outline
    :param markers: optional `{label: (x, y)}` points to overlay (e.g.,
        striker/returner positions)
    :param ax: axes to draw on
        - Default: a new figure and axes
    """
    _LOG.debug(hprint.to_str("column"))
    hdbg.dassert_lt(0, len(scores), "scores cannot be empty")
    hdbg.dassert_in(column, scores.columns, "column must be in scores")
    if ax is None:
        _, ax = plt.subplots(figsize=plt.rcParams["figure.figsize"])
    x_vals = np.unique(scores["x_m"].to_numpy(dtype=float))
    y_vals = np.unique(scores["y_m"].to_numpy(dtype=float))
    x_step = float(np.min(np.diff(x_vals))) if len(x_vals) > 1 else 1.0
    y_step = float(np.min(np.diff(y_vals))) if len(y_vals) > 1 else 1.0
    values = scores[column].to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    vmin = float(np.min(finite_values)) if len(finite_values) else 0.0
    vmax = float(np.max(finite_values)) if len(finite_values) else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    for _, row in scores.iterrows():
        value = row[column]
        color = cmap(norm(value)) if np.isfinite(value) else "lightgray"
        rect = mpatches.Rectangle(
            (row["x_m"] - x_step / 2, row["y_m"] - y_step / 2),
            x_step,
            y_step,
            facecolor=color,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.add_patch(rect)
    draw_court(ax, sport.court)
    scalar_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar_mappable.set_array([])
    plt.colorbar(scalar_mappable, ax=ax, label=column)
    if markers:
        for label, (marker_x, marker_y) in markers.items():
            ax.scatter(
                [marker_x],
                [marker_y],
                marker="*",
                s=200,
                color="red",
                edgecolor="black",
                label=label,
                zorder=6,
            )
        ax.legend()
    ax.set_title(f"{column} heatmap")


# #############################################################################
# Launch angle trade-off
# #############################################################################


def plot_launch_tradeoff(
    launch_table: pd.DataFrame, cell_id: int, *, ax: Optional[Axes] = None
) -> None:
    """
    Plot `p_in` and `v0_mps` against `theta_deg`, for one cell.

    :param launch_table: launch table, from
        `racket_scoring.estimate_launch_table()`
    :param cell_id: which cell's angle sweep to plot
    :param ax: axes to draw on
        - Default: a new figure and axes
    """
    _LOG.debug(hprint.to_str("cell_id"))
    if ax is None:
        _, ax = plt.subplots(figsize=plt.rcParams["figure.figsize"])
    mask = launch_table["cell_id"] == cell_id
    rows = launch_table.loc[mask].sort_values(by="theta_deg")
    ax.plot(rows["theta_deg"], rows["p_in"], marker="o", color="C0")
    ax.set_xlabel("Launch angle theta (deg)")
    ax.set_ylabel("P_in", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax_speed = ax.twinx()
    ax_speed.plot(rows["theta_deg"], rows["v0_mps"], marker="s", color="C1")
    ax_speed.set_ylabel("v0 (m/s)", color="C1")
    ax_speed.tick_params(axis="y", labelcolor="C1")
    ax.set_title(f"Launch angle trade-off: cell {cell_id}")


# #############################################################################
# Score widget (cheap re-score over a cached launch table)
# #############################################################################


def build_score_widget(
    sport: racket_params.SportParams,
    launch_table: pd.DataFrame,
    returner: racket_params.PlayerParams,
) -> ipywidgets.Widget:
    """
    Live widget: returner position, `t_r`, `v_p` re-score a cached table.

    Cheap: only `racket_scoring.score_targets()` re-runs on every slider
    move, no new Monte Carlo sampling.

    :param sport: sport parameters, for the court outline
    :param launch_table: cached launch table, from
        `racket_scoring.estimate_launch_table()`
    :param returner: initial returner parameters (reaction time, move speed)
    :return: the displayed widget container
    """
    _LOG.debug(hprint.to_str("returner"))
    x_max = sport.court.width_m / 2
    y_max = sport.court.length_m / 2
    returner_x_slider, returner_x_box = htutori.build_widget_control(
        name="returner_x",
        description="Returner x (m)",
        min_val=-x_max,
        max_val=x_max,
        step=0.1,
        initial_value=0.0,
    )
    returner_y_slider, returner_y_box = htutori.build_widget_control(
        name="returner_y",
        description="Returner y (m)",
        min_val=0.5,
        max_val=y_max,
        step=0.1,
        initial_value=y_max / 2,
    )
    t_r_slider, t_r_box = htutori.build_widget_control(
        name="t_r",
        description="Reaction time (s)",
        min_val=0.0,
        max_val=1.0,
        step=0.05,
        initial_value=returner.reaction_time_s,
    )
    v_p_slider, v_p_box = htutori.build_widget_control(
        name="v_p",
        description="Move speed (m/s)",
        min_val=0.5,
        max_val=6.0,
        step=0.1,
        initial_value=returner.move_speed_mps,
    )
    output = ipywidgets.Output()

    def update(change: Optional[object] = None) -> None:
        """
        Re-score the cached launch table and redraw the heatmaps.
        """
        _ = change
        with output:
            clear_output(wait=True)
            current_returner = dataclasses.replace(
                returner,
                reaction_time_s=t_r_slider.value,
                move_speed_mps=v_p_slider.value,
            )
            returner_xy = (returner_x_slider.value, returner_y_slider.value)
            scores = racket_scoring.score_targets(
                launch_table, current_returner, returner_xy
            )
            markers = {"Returner": returner_xy}
            fig, (ax_p_in, ax_score) = plt.subplots(1, 2, figsize=(12, 5))
            plot_court_heatmap(
                scores, "p_in", sport, markers=markers, ax=ax_p_in
            )
            plot_court_heatmap(
                scores, "score", sport, markers=markers, ax=ax_score
            )
            plt.tight_layout()
            plt.show()
            # Close the figure: an inline-backend figure left open inside an
            # `Output` widget stalls the kernel's idle handshake on rerun.
            plt.close(fig)

    for slider in (
        returner_x_slider,
        returner_y_slider,
        t_r_slider,
        v_p_slider,
    ):
        slider.observe(update, names="value")
    update()
    controls = ipywidgets.VBox(
        [returner_x_box, returner_y_box, t_r_box, v_p_box]
    )
    widget = ipywidgets.VBox([controls, output])
    display(widget)
    return widget


# #############################################################################
# Exploration widget (click-to-run, resamples the launch table)
# #############################################################################


def build_exploration_widget(
    sport: racket_params.SportParams, config: racket_scoring.ScoringConfig
) -> ipywidgets.Widget:
    """
    Click-to-run widget over ball speed, error scale, move speed, positions.

    Every control feeds `racket_scoring.estimate_launch_table()`, the costly
    Monte Carlo step, so this recomputes only on a "Run" click, not on every
    slider drag (contrast `build_score_widget()`, which is live).

    :param sport: sport parameters (max ball speed, court geometry)
    :param config: sampling configuration for the Monte Carlo re-sample
    :return: the displayed widget container
    """
    _LOG.debug(hprint.to_str("sport config"))
    x_max = sport.court.width_m / 2
    y_max = sport.court.length_m / 2
    max_speed_slider, max_speed_box = htutori.build_widget_control(
        name="max_ball_speed_mps",
        description="Max ball speed (m/s)",
        min_val=5.0,
        max_val=sport.max_ball_speed_mps,
        step=1.0,
        initial_value=min(30.0, sport.max_ball_speed_mps),
    )
    error_scale_slider, error_scale_box = htutori.build_widget_control(
        name="error_scale",
        description="Shot error scale",
        min_val=0.2,
        max_val=3.0,
        step=0.1,
        initial_value=1.0,
    )
    move_speed_slider, move_speed_box = htutori.build_widget_control(
        name="move_speed_mps",
        description="Returner move speed (m/s)",
        min_val=0.5,
        max_val=6.0,
        step=0.1,
        initial_value=racket_params.DEFAULT_PLAYER.move_speed_mps,
    )
    striker_x_slider, striker_x_box = htutori.build_widget_control(
        name="striker_x",
        description="Striker x (m)",
        min_val=-x_max,
        max_val=x_max,
        step=0.1,
        initial_value=0.0,
    )
    striker_y_slider, striker_y_box = htutori.build_widget_control(
        name="striker_y",
        description="Striker y (m)",
        min_val=-y_max,
        max_val=-0.5,
        step=0.1,
        initial_value=-(y_max - 1.0),
    )
    returner_x_slider, returner_x_box = htutori.build_widget_control(
        name="returner_x",
        description="Returner x (m)",
        min_val=-x_max,
        max_val=x_max,
        step=0.1,
        initial_value=0.0,
    )
    returner_y_slider, returner_y_box = htutori.build_widget_control(
        name="returner_y",
        description="Returner y (m)",
        min_val=0.5,
        max_val=y_max,
        step=0.1,
        initial_value=y_max - 1.0,
    )
    run_button = ipywidgets.Button(description="Run", button_style="primary")
    output = ipywidgets.Output()

    def run(_button: Optional[object] = None) -> None:
        """
        Resample the launch table for the current controls and redraw.
        """
        _ = _button
        with output:
            clear_output(wait=True)
            custom_sport = dataclasses.replace(
                sport, max_ball_speed_mps=max_speed_slider.value
            )
            base_error = racket_params.DEFAULT_ERROR
            error_scale = error_scale_slider.value
            scaled_error = racket_params.ShotErrorModel(
                sigma_theta_rad=base_error.sigma_theta_rad * error_scale,
                sigma_v_frac=base_error.sigma_v_frac * error_scale,
                sigma_phi_rad=base_error.sigma_phi_rad * error_scale,
            )
            striker = dataclasses.replace(
                racket_params.DEFAULT_PLAYER, error=scaled_error
            )
            returner = dataclasses.replace(
                racket_params.DEFAULT_PLAYER,
                move_speed_mps=move_speed_slider.value,
            )
            striker_xy = (striker_x_slider.value, striker_y_slider.value)
            returner_xy = (returner_x_slider.value, returner_y_slider.value)
            situation = racket_scoring.make_rally_situation(
                custom_sport, striker, striker_xy
            )
            targets = racket_scoring.make_target_grid(
                situation.legal_region, 8, 8
            )
            launch_table = racket_scoring.estimate_launch_table(
                custom_sport, situation, targets, config
            )
            scores = racket_scoring.score_targets(
                launch_table, returner, returner_xy
            )
            scores = scores.copy()
            # Winner probability for the striker: 1 - R(c).
            scores["winner_prob"] = 1 - scores["reachable"].astype(float)
            markers = {"Striker": striker_xy, "Returner": returner_xy}
            fig, (ax_p_in, ax_winner, ax_score) = plt.subplots(
                1, 3, figsize=(16, 5)
            )
            plot_court_heatmap(
                scores, "p_in", custom_sport, markers=markers, ax=ax_p_in
            )
            plot_court_heatmap(
                scores,
                "winner_prob",
                custom_sport,
                markers=markers,
                ax=ax_winner,
            )
            plot_court_heatmap(
                scores, "score", custom_sport, markers=markers, ax=ax_score
            )
            plt.tight_layout()
            plt.show()
            # Close the figure: an inline-backend figure left open inside an
            # `Output` widget stalls the kernel's idle handshake on rerun.
            plt.close(fig)

    run_button.on_click(run)
    # Render one static default run so the cell has output even without a
    # click (e.g., when run headlessly under nbconvert).
    run()
    controls = ipywidgets.VBox(
        [
            max_speed_box,
            error_scale_box,
            move_speed_box,
            striker_x_box,
            striker_y_box,
            returner_x_box,
            returner_y_box,
            run_button,
        ]
    )
    widget = ipywidgets.VBox([controls, output])
    display(widget)
    return widget
