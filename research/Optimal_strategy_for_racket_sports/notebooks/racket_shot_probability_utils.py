"""
Utilities for the `racket_shot_probability` notebook.

Player 1 serves to `(x1, y1)`, player 2 moves there and aims a return at
`(x2, y2)` on player 1's half of the court. The return lands at a 2D Gaussian
point around `(x2, y2)` with independent `x` and `y` errors, so the
probability that the return is in has a closed form.

Import as:

import research.Optimal_strategy_for_racket_sports.notebooks.racket_shot_probability_utils as rosfrsnrspu
"""

import logging
from typing import Any, Optional, Tuple

import ipywidgets
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from IPython.display import clear_output, display
from matplotlib.axes import Axes

import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebo
import helpers.hprint as hprint
import helpers.htutorial as htutori
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
# In-court probability
# #############################################################################


def get_striker_half_region(
    court: racket_params.CourtGeometry,
) -> racket_params.CourtRegion:
    """
    Get the region of player 1's half of the court (y < 0).

    Mirror of `racket_params.get_half_court_region()`, which returns player 2's
    half (y > 0). Player 2's return must land in this region to be in.

    :param court: court geometry, e.g., `racket_params.TENNIS.court`
    :return: rectangle with `x` in `[-W/2, W/2]` and `y` in `[-L/2, 0]`, where
        `W` is the court width and `L` is the court length
    """
    _LOG.debug(hprint.to_str("court"))
    # Mirror of `racket_params.get_half_court_region()` across the net (y = 0).
    region = racket_params.CourtRegion(
        x_min=-court.width_m / 2,
        x_max=court.width_m / 2,
        y_min=-court.length_m / 2,
        y_max=0,
    )
    _LOG.debug("return=%s", region)
    return region


def compute_p_in(
    target_xy: Tuple[float, float],
    std_xy: Tuple[float, float],
    region: racket_params.CourtRegion,
) -> float:
    """
    Compute the probability that a Gaussian shot lands inside `region`.

    The landing point is `X ~ N(x2, std_x2^2)` and `Y ~ N(y2, std_y2^2)`, with
    `X` and `Y` independent. Since `region` is an axis-aligned rectangle, the
    probability is the `x` part times the `y` part:
    ```
    P_in = P(x_min <= X <= x_max) * P(y_min <= Y <= y_max)
    ```
    Each part is a difference of two normal CDFs, e.g.,
    `P(x_min <= X <= x_max) = CDF_X(x_max) - CDF_X(x_min)`.

    :param target_xy: aim point `(x2, y2)` in meters, e.g., `(0.0, -5.94)`
    :param std_xy: standard deviations `(std_x2, std_y2)` in meters, e.g.,
        `(0.5, 1.0)`
    :param region: rectangle where the shot counts as in
    :return: probability in `[0, 1]`, e.g., about 0.25 when aiming at a corner
        of `region` with a small std
    """
    _LOG.debug(hprint.to_str("target_xy std_xy region"))
    x2, y2 = target_xy
    std_x2, std_y2 = std_xy
    # `scipy` returns NaN instead of raising for a std that is not positive.
    hdbg.dassert_lt(0, std_x2, "std_x2 must be positive")
    hdbg.dassert_lt(0, std_y2, "std_y2 must be positive")
    # Probability that `X` lands between `x_min` and `x_max`.
    cdf_x_max = scipy.stats.norm.cdf(region.x_max, loc=x2, scale=std_x2)
    cdf_x_min = scipy.stats.norm.cdf(region.x_min, loc=x2, scale=std_x2)
    p_x = cdf_x_max - cdf_x_min
    # Probability that `Y` lands between `y_min` and `y_max`.
    cdf_y_max = scipy.stats.norm.cdf(region.y_max, loc=y2, scale=std_y2)
    cdf_y_min = scipy.stats.norm.cdf(region.y_min, loc=y2, scale=std_y2)
    p_y = cdf_y_max - cdf_y_min
    # Multiply the two parts since `X` and `Y` are independent.
    p_in = float(p_x * p_y)
    _LOG.debug("return=%s", p_in)
    return p_in


# #############################################################################
# Monte Carlo sampling
# #############################################################################


def sample_landings(
    target_xy: Tuple[float, float],
    std_xy: Tuple[float, float],
    n_samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw `n_samples` Gaussian landing points around `target_xy`.

    Same model as `compute_p_in()`: `X ~ N(x2, std_x2^2)` and
    `Y ~ N(y2, std_y2^2)`, drawn independently. The samples give a Monte Carlo
    check of `compute_p_in()` and the scatter cloud in the notebook.

    :param target_xy: aim point `(x2, y2)` in meters
    :param std_xy: standard deviations `(std_x2, std_y2)` in meters
    :param n_samples: number of landing points to draw
    :param rng: random generator to draw from, passed in explicitly so that the
        same seed gives the same points, e.g., `np.random.default_rng(1)`
    :return: arrays `(x, y)` of landing coordinates, each of shape
        `(n_samples,)`
    """
    _LOG.debug(hprint.to_str("target_xy std_xy n_samples"))
    x2, y2 = target_xy
    std_x2, std_y2 = std_xy
    # Same check as `compute_p_in()`: `rng.normal()` silently returns the target
    # every time for a std of 0.
    hdbg.dassert_lt(0, std_x2, "std_x2 must be positive")
    hdbg.dassert_lt(0, std_y2, "std_y2 must be positive")
    # Draw each coordinate on its own since `X` and `Y` are independent.
    x = rng.normal(x2, std_x2, n_samples)
    y = rng.normal(y2, std_y2, n_samples)
    _LOG.debug("return: %d landing points", len(x))
    return x, y


# #############################################################################
# Court drawing
# #############################################################################


def _draw_center_service_line(
    ax: Axes, court: racket_params.CourtGeometry
) -> None:
    """
    Draw the center service line (x = 0) that splits the service boxes.

    `racket_strategy_utils.draw_court()` does not draw it. On each side of the
    net it spans the same `y` range as `racket_params.get_service_box_region()`:
    - Tennis: from the net to the service line
    - Pickleball: from the kitchen line to the baseline

    :param ax: axes to draw on
    :param court: court geometry giving the line positions
    """
    _LOG.debug(hprint.to_str("court"))
    # One segment per side of the net: `sign = -1` is player 1's side.
    for sign in (-1, 1):
        ax.plot(
            [0, 0],
            [sign * court.non_volley_zone_m, sign * court.service_line_m],
            color="gray",
            linestyle="--",
            linewidth=1.0,
        )


# #############################################################################
# Cell 0.1: The two courts
# #############################################################################


def cell0_1_plot_courts(*, figsize: Tuple[float, float] = (10.0, 7.0)) -> None:
    """
    Draw the tennis and pickleball courts side by side, in the same frame.

    :param figsize: figure size (width, height) in inches
    """
    _LOG.debug(hprint.to_str("figsize"))
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    # Same drawing for both sports, plus the center service line.
    for ax, sport in zip(axes, [racket_params.TENNIS, racket_params.PICKLEBALL]):
        racket_strategy_utils.draw_court(ax, sport.court)
        _draw_center_service_line(ax, sport.court)
        ax.set_title(sport.name)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


# #############################################################################
# Cell 1.1: Serve and return widget
# #############################################################################


def _round_slider_value(value: float) -> float:
    """
    Round a float slider value to 6 decimals, before any logic uses it.

    The `+` / `-` buttons add the 0.1 step in floating point, so the value
    drifts: 21 clicks of `+` from -2.1 give 6.4e-16 instead of 0, which
    `x1 <= 0` would put in the ad box. Adding 0.0 turns -0.0 into 0.0, so the
    value prints as "0.00" and not "-0.00".

    :param value: slider value, e.g., 6.4e-16
    :return: rounded value, e.g., 0.0
    """
    _LOG.debug(hprint.to_str("value"))
    rounded = round(value, 6) + 0.0
    _LOG.debug("return=%s", rounded)
    return rounded


def _check_serve(
    x1: float, y1: float, court: racket_params.CourtGeometry
) -> Tuple[str, racket_params.CourtRegion, bool]:
    """
    Find the target service box of a serve to `(x1, y1)`, and whether it is in.

    The target box follows the sign of `x1`, as in
    `racket_params.get_service_box_region()`: `x1 <= 0` is the deuce box. A
    serve on the kitchen (non-volley) line is a fault, while
    `CourtRegion.contains()` counts every line as in, so the serve must also
    land strictly past the kitchen line.

    :param x1: where the serve lands, across the court, e.g., -2.1
    :param y1: where the serve lands, along the court, e.g., 3.2
    :param court: court geometry, e.g., `racket_params.PICKLEBALL.court`
    :return: side ("deuce" or "ad"), its service box, and whether the serve is
        in, e.g., `("deuce", CourtRegion(...), True)`
    """
    _LOG.debug(hprint.to_str("x1 y1 court"))
    serve_side = "deuce" if x1 <= 0 else "ad"
    box = racket_params.get_service_box_region(court, serve_side)
    in_box = bool(box.contains(np.array([x1]), np.array([y1]))[0])
    # For tennis the kitchen line is the net (0 m), so this never bites there.
    is_serve_in = in_box and y1 > court.non_volley_zone_m
    _LOG.debug("return=%s", (serve_side, box, is_serve_in))
    return serve_side, box, is_serve_in


def _draw_service_box(ax: Axes, box: racket_params.CourtRegion) -> None:
    """
    Shade the target service box of the serve.

    :param ax: court axes to draw on
    :param box: service box, from `_check_serve()`
    """
    _LOG.debug(hprint.to_str("box"))
    box_rect = mpatches.Rectangle(
        (box.x_min, box.y_min),
        box.x_max - box.x_min,
        box.y_max - box.y_min,
        color="tab:blue",
        alpha=0.1,
        label="service box",
    )
    ax.add_patch(box_rect)


def _draw_samples(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    is_in: np.ndarray,
    court: racket_params.CourtGeometry,
) -> None:
    """
    Draw the Monte Carlo landing points, and widen the view to include them.

    :param ax: court axes to draw on
    :param x: landing `x` of each sample, from `sample_landings()`
    :param y: landing `y` of each sample
    :param is_in: True where the sample lands in player 1's half
    :param court: court geometry, for the default view
    """
    _LOG.debug(hprint.to_str("court"))
    # Green in, red out, and see-through so the ellipses stay visible.
    ax.scatter(x[is_in], y[is_in], s=4, c="tab:green", alpha=0.4, label="in")
    ax.scatter(x[~is_in], y[~is_in], s=4, c="tab:red", alpha=0.4, label="out")
    # View: the court plus 1 m, widened to include every dot, so the cloud is
    # not cut off for targets and stds at the slider limits.
    half_width = court.width_m / 2
    half_length = court.length_m / 2
    ax.set_xlim(
        min(-half_width - 1, x.min() - 0.5), max(half_width + 1, x.max() + 0.5)
    )
    ax.set_ylim(
        min(-half_length - 1, y.min() - 0.5),
        max(half_length + 1, y.max() + 0.5),
    )


def _draw_sigma_ellipses(
    ax: Axes, target_xy: Tuple[float, float], std_xy: Tuple[float, float]
) -> None:
    """
    Draw the 1 and 2 sigma ellipses of the return around `target_xy`.

    The `k` sigma ellipse is `k` std away from `(x2, y2)` on each axis. In 2D
    the 1 sigma ellipse holds about 39% of the samples and the 2 sigma ellipse
    about 86% (`1 - exp(-k^2 / 2)`), not the 68% and 95% of 1D.

    :param ax: court axes to draw on
    :param target_xy: aim point `(x2, y2)`
    :param std_xy: standard deviations `(std_x2, std_y2)`
    """
    _LOG.debug(hprint.to_str("target_xy std_xy"))
    for k, linestyle in ((1, "-"), (2, "--")):
        ellipse = mpatches.Ellipse(
            target_xy,
            2 * k * std_xy[0],
            2 * k * std_xy[1],
            fill=False,
            # Explicit color: the `seaborn` style of the notebook makes patch
            # edges white by default.
            edgecolor="black",
            linestyle=linestyle,
            label=f"{k} sigma",
            zorder=3,
        )
        ax.add_patch(ellipse)


def _draw_players_and_arrows(
    ax: Axes,
    server_xy: Tuple[float, float],
    serve_xy: Tuple[float, float],
    target_xy: Tuple[float, float],
) -> None:
    """
    Draw the serve and return arrows, and the two players.

    :param ax: court axes to draw on
    :param server_xy: where player 1 serves from
    :param serve_xy: where the serve lands, `(x1, y1)`, where player 2 stands
    :param target_xy: where player 2 aims the return, `(x2, y2)`
    """
    _LOG.debug(hprint.to_str("server_xy serve_xy target_xy"))
    # An annotation with empty text is just an arrow, from `xytext` to `xy`.
    ax.annotate(
        "",
        xy=serve_xy,
        xytext=server_xy,
        arrowprops={"arrowstyle": "->", "color": "tab:blue"},
    )
    ax.annotate(
        "",
        xy=target_xy,
        xytext=serve_xy,
        arrowprops={"arrowstyle": "->", "color": "tab:orange"},
    )
    ax.plot(*server_xy, "o", color="tab:blue", label="player 1")
    ax.plot(*serve_xy, "o", color="tab:orange", label="player 2")


def _build_comments_text(
    serve_xy: Tuple[float, float],
    serve_side: str,
    is_serve_in: bool,
    target_xy: Tuple[float, float],
    std_xy: Tuple[float, float],
    p_in: float,
    p_hat: float,
    n_samples: int,
) -> str:
    """
    Build the text of the comments panel from the current state.

    :param serve_xy: where the serve lands, `(x1, y1)`
    :param serve_side: "deuce" or "ad", from `_check_serve()`
    :param is_serve_in: whether the serve lands in its service box
    :param target_xy: aim point of the return, `(x2, y2)`
    :param std_xy: standard deviations `(std_x2, std_y2)`
    :param p_in: closed-form probability that the return is in
    :param p_hat: fraction of the `n_samples` Monte Carlo samples that are in
    :param n_samples: number of Monte Carlo samples
    :return: multi-line text, e.g., starting with
        "Serve to (x1, y1) = (-2.10, 3.20)"
    """
    _LOG.debug(hprint.to_str("serve_xy serve_side is_serve_in p_in p_hat"))
    # Standard error of `P_hat` from the true probability `P_in`, not from
    # `P_hat` itself: `sqrt(P_hat * (1 - P_hat) / N)` is 0 when all dots land
    # on one side, which would claim no uncertainty at all.
    p_hat_se = np.sqrt(p_in * (1 - p_in) / n_samples)
    serve_in_str = "yes" if is_serve_in else "no"
    text = (
        f"Serve to (x1, y1) = ({serve_xy[0]:.2f}, {serve_xy[1]:.2f})\n"
        f"  in the {serve_side} box: {serve_in_str}\n\n"
        f"Return to (x2, y2) = ({target_xy[0]:.2f}, {target_xy[1]:.2f})\n"
        f"  (std_x2, std_y2) = ({std_xy[0]:.2f}, {std_xy[1]:.2f})\n\n"
        f"P_in (closed form) = {p_in:.4f}\n"
        f"P_hat (Monte Carlo, N = {n_samples}) = {p_hat:.4f}\n"
        f"  standard error = {p_hat_se:.4f}"
    )
    _LOG.debug("return=%s", text)
    return text


def cell1_1_plot_shot_widget(
    sport: racket_params.SportParams,
    *,
    figsize: Tuple[float, float] = (12.0, 7.0),
) -> ipywidgets.VBox:
    """
    Build the serve and return widget for `sport` and display it.

    Player 1 serves from beside the center mark to `(x1, y1)`. Player 2 moves
    there and aims a return at `(x2, y2)` with Gaussian spread
    `(std_x2, std_y2)`. Every slider change redraws a 1 x 2 figure:
    - Court: both shots, the 1 and 2 sigma ellipses of the return, and Monte
      Carlo landing points colored by in or out
    - Comments: closed-form `P_in`, the Monte Carlo estimate with its standard
      error, whether the serve lands in the diagonal service box, and the
      legend of the court

    :param sport: sport whose court is drawn, e.g., `racket_params.TENNIS`
    :param figsize: figure size (width, height) in inches
    :return: the displayed widget: the sliders, then the plot output
    """
    _LOG.debug(hprint.to_str("sport"))
    court = sport.court
    half_width = court.width_m / 2
    half_length = court.length_m / 2
    # Enough Monte Carlo points for a standard error of at most 0.016, few
    # enough to keep the scatter readable.
    n_samples = 1000
    # Serve sliders: `(x1, y1)` on player 2's side (`y1 > 0`), starting in the
    # middle of the deuce service box.
    x1_init = round(-half_width / 2, 1)
    x1_slider, x1_box = htutori.build_widget_control(
        "x1", "serve x", -half_width, half_width, 0.1, x1_init
    )
    y1_init = round((court.non_volley_zone_m + court.service_line_m) / 2, 1)
    y1_slider, y1_box = htutori.build_widget_control(
        "y1", "serve y", 0.1, half_length, 0.1, y1_init
    )
    # Return sliders: `(x2, y2)` on player 1's side (`y2 <= 0`), up to 2 m
    # outside the lines, starting 1.5 m inside the baseline.
    x2_init = round(half_width / 2, 1)
    x2_slider, x2_box = htutori.build_widget_control(
        "x2", "return x", -half_width - 2, half_width + 2, 0.1, x2_init
    )
    y2_init = round(-half_length + 1.5, 1)
    y2_slider, y2_box = htutori.build_widget_control(
        "y2", "return y", -half_length - 2, 0.0, 0.1, y2_init
    )
    # Spread of the return: a std must be strictly positive.
    std_x2_slider, std_x2_box = htutori.build_widget_control(
        "std_x2", "return std x", 0.1, 3.0, 0.1, 0.5
    )
    std_y2_slider, std_y2_box = htutori.build_widget_control(
        "std_y2", "return std y", 0.1, 3.0, 0.1, 1.0
    )
    # Seed of the Monte Carlo samples, placed last as the notebook rules ask.
    seed_slider, seed_box = htutori.build_widget_control(
        "seed", "random seed", 0, 100, 1, 1, is_float=False
    )
    output = ipywidgets.Output()

    def update_plot(change: Optional[Any] = None) -> None:
        """
        Redraw both panels from the current slider values.

        :param change: change event sent by `observe()` (unused)
        """
        _ = change
        with output:
            clear_output(wait=True)
            # Read the slider values, rounded before any logic uses them.
            x1 = _round_slider_value(x1_slider.value)
            y1 = _round_slider_value(y1_slider.value)
            x2 = _round_slider_value(x2_slider.value)
            y2 = _round_slider_value(y2_slider.value)
            std_xy = (
                _round_slider_value(std_x2_slider.value),
                _round_slider_value(std_y2_slider.value),
            )
            serve_side, box, is_serve_in = _check_serve(x1, y1, court)
            # Player 1 stands on the baseline just beside the center mark, on
            # the side diagonal to the target box. The 0.5 m offset is a
            # typical stance, not a rule value: it keeps the server off the
            # center line so the diagonal is visible.
            server_xy = (0.5 if serve_side == "deuce" else -0.5, -half_length)
            # Closed form and Monte Carlo estimate on player 1's half.
            region = get_striker_half_region(court)
            p_in = compute_p_in((x2, y2), std_xy, region)
            rng = np.random.default_rng(seed_slider.value)
            x, y = sample_landings((x2, y2), std_xy, n_samples, rng)
            is_in = region.contains(x, y)
            p_hat = is_in.mean()
            # Panel 1: the court, the target box, the dots, the ellipses, and
            # the two shots, in this order so the shots are drawn on top.
            fig, (ax_court, ax_text) = plt.subplots(1, 2, figsize=figsize)
            racket_strategy_utils.draw_court(ax_court, court)
            _draw_center_service_line(ax_court, court)
            _draw_service_box(ax_court, box)
            _draw_samples(ax_court, x, y, is_in, court)
            _draw_sigma_ellipses(ax_court, (x2, y2), std_xy)
            _draw_players_and_arrows(ax_court, server_xy, (x1, y1), (x2, y2))
            ax_court.set_title(sport.name)
            # Panel 2: comments with the current state only.
            ax_text.axis("off")
            ax_text.set_title("Comments", fontsize=14, fontweight="bold", pad=20)
            text = _build_comments_text(
                (x1, y1),
                serve_side,
                is_serve_in,
                (x2, y2),
                std_xy,
                p_in,
                p_hat,
                n_samples,
            )
            htutori.add_fitted_text_box(
                ax_text, text, max_fontsize=13, min_fontsize=9
            )
            # Legend in the free lower part of the comments panel, so it never
            # covers the court.
            handles, labels = ax_court.get_legend_handles_labels()
            ax_text.legend(handles, labels, loc="lower left", fontsize=10)
            plt.tight_layout()
            plt.show()
            # A figure left open inside an `Output` widget can stall the kernel
            # on rerun, so close it once shown.
            plt.close(fig)

    # Redraw on every slider change: no Run button.
    sliders = [
        x1_slider,
        y1_slider,
        x2_slider,
        y2_slider,
        std_x2_slider,
        std_y2_slider,
        seed_slider,
    ]
    for slider in sliders:
        slider.observe(update_plot, names="value")
    # Draw once now, so the cell has output even without interaction.
    update_plot()
    # Sliders on top, the plot below.
    boxes = [x1_box, y1_box, x2_box, y2_box, std_x2_box, std_y2_box, seed_box]
    widget = ipywidgets.VBox(boxes + [output])
    display(widget)
    return widget
