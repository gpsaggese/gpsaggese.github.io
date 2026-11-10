"""
Utility functions to generate the figures of Lesson 01.01 "Why Finance Is
Different".

Import as:

import L01_01_why_finance_is_different_utils as l0wfdiut
"""

import logging
from typing import Any, List, Optional, Tuple

import matplotlib.figure as mfigure
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.stats as sstats
import seaborn as sns

import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebo

_LOG = logging.getLogger(__name__)


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utils logger.

    :param notebook_log: logger owned by the notebook
    """
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# Plot helpers
# #############################################################################


# Colors of the TikZ figures of the slides, so that a figure generated here
# looks like the diagrams around it.
_BLUE = "#3C6FB0"
_CORAL = "#C0455B"
_ORANGE = "#D98E2B"
_GREEN = "#4F9A5C"
_GRAY = "#7B8794"


def _new_figure(
    figsize: Optional[Tuple[float, float]],
    *,
    nrows: int = 1,
    ncols: int = 1,
    **kwargs: Any,
) -> Tuple[mfigure.Figure, np.ndarray]:
    """
    Create a figure with a clean style and no top / right spines.

    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :param nrows: number of rows of subplots
    :param ncols: number of columns of subplots
    :param kwargs: additional arguments passed to `plt.subplots()`
    :return: figure and array of axes with shape (nrows, ncols)
    """
    if figsize is None:
        figsize = tuple(plt.rcParams["figure.figsize"])
    # Use a white background without the default seaborn grid, so that the
    # figure blends into the slide.
    with sns.axes_style("ticks"):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            squeeze=False,
            **kwargs,
        )
    sns.despine(fig=fig)
    return fig, axes


# #############################################################################
# Cell 1.1: Futures and options payoffs
# #############################################################################


def cell1_1_plot_option_payoffs(
    *,
    strike: float,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot the payoff at expiration of a long call and a long put.

    :param strike: strike price $K$ shared by the call and the put
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with the call and the put payoff curves
    """
    # Payoff of each option on a grid of spot prices at expiration.
    spot = np.linspace(0.0, 2.0 * strike, 201)
    payoff_df = pd.DataFrame(
        {
            "spot": spot,
            "call": np.maximum(spot - strike, 0.0),
            "put": np.maximum(strike - spot, 0.0),
        }
    )
    payoff_df = payoff_df.melt(
        id_vars="spot", var_name="option", value_name="payoff"
    )
    #
    fig, axes = _new_figure(figsize)
    ax = axes[0, 0]
    sns.lineplot(
        data=payoff_df,
        x="spot",
        y="payoff",
        hue="option",
        palette={"call": _BLUE, "put": _CORAL},
        linewidth=2.5,
        legend=False,
        ax=ax,
    )
    # Mark the strike, where both payoffs have a kink.
    ax.axvline(strike, color=_GRAY, linestyle="--", linewidth=1.0)
    ax.set_xticks([0.0, strike, 2.0 * strike])
    ax.set_xticklabels(["0", "$K$", "$2K$"])
    ax.text(1.55 * strike, 0.75 * strike, "long call", color=_BLUE, ha="center")
    ax.text(0.45 * strike, 0.75 * strike, "long put", color=_CORAL, ha="center")
    ax.set_xlabel("spot price at expiration $S_T$")
    ax.set_ylabel("payoff")
    return fig


# #############################################################################
# Cell 1.2: Equity curve, maximum drawdown, and drawdown duration
# #############################################################################


def _simulate_equity_curve(
    *, n_days: int, mean_return: float, volatility: float, seed: int
) -> pd.Series:
    """
    Simulate the equity curve of a strategy with IID Gaussian daily returns.

    :param n_days: number of days to simulate
    :param mean_return: mean daily return
    :param volatility: daily volatility of the return
    :param seed: seed of the random number generator
    :return: equity curve of length `n_days + 1` that starts at 1.0
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(mean_return, volatility, size=n_days)
    # Compound the gross returns, starting from one unit of capital.
    equity = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    equity = pd.Series(equity, name="equity")
    return equity


def _compute_drawdown(equity: pd.Series) -> pd.DataFrame:
    """
    Compute the high watermark and the drawdown of an equity curve.

    :param equity: equity curve
    :return: dataframe with columns `equity`, `high_watermark`, and
        `drawdown`, where the drawdown is the fractional loss from the
        high watermark
    """
    high_watermark = equity.cummax()
    drawdown = (high_watermark - equity) / high_watermark
    drawdown_df = pd.DataFrame(
        {
            "equity": equity,
            "high_watermark": high_watermark,
            "drawdown": drawdown,
        }
    )
    return drawdown_df


def _find_longest_drawdown(drawdown_df: pd.DataFrame) -> Tuple[int, int]:
    """
    Find the longest time that the equity stays below a previous peak.

    :param drawdown_df: output of `_compute_drawdown()`
    :return: positions of the peak that starts the period and of the day
        that recovers it (or the last day when it is never recovered)
    """
    # The equity sets a new high watermark at each peak.
    peak_pos = np.flatnonzero(drawdown_df["drawdown"].to_numpy() == 0.0)
    # A drawdown lasts from a peak to the next one, or to the end of the sample.
    recover_pos = np.append(peak_pos[1:], len(drawdown_df) - 1)
    longest = int(np.argmax(recover_pos - peak_pos))
    start_pos = int(peak_pos[longest])
    end_pos = int(recover_pos[longest])
    return start_pos, end_pos


def cell1_2_plot_drawdown(
    *,
    n_days: int,
    mean_return: float,
    volatility: float,
    seed: int,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot an equity curve with its high watermark and its drawdowns.

    The figure marks the maximum drawdown (depth) and the longest drawdown
    (duration), which can be two different episodes.

    :param n_days: number of days to simulate
    :param mean_return: mean daily return
    :param volatility: daily volatility of the return
    :param seed: seed of the random number generator
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with the equity curve and the drawdown annotations
    """
    equity = _simulate_equity_curve(
        n_days=n_days,
        mean_return=mean_return,
        volatility=volatility,
        seed=seed,
    )
    drawdown_df = _compute_drawdown(equity)
    # Depth of the worst drawdown and the day when it hits the bottom.
    trough_pos = int(drawdown_df["drawdown"].to_numpy().argmax())
    max_drawdown = drawdown_df["drawdown"].iloc[trough_pos]
    peak_value = drawdown_df["high_watermark"].iloc[trough_pos]
    trough_value = drawdown_df["equity"].iloc[trough_pos]
    # Duration of the longest drawdown.
    start_pos, end_pos = _find_longest_drawdown(drawdown_df)
    duration = end_pos - start_pos
    #
    fig, axes = _new_figure(figsize)
    ax = axes[0, 0]
    sns.lineplot(
        data=drawdown_df,
        x=drawdown_df.index,
        y="equity",
        color=_BLUE,
        linewidth=1.5,
        ax=ax,
    )
    ax.plot(
        drawdown_df.index,
        drawdown_df["high_watermark"],
        color=_GRAY,
        linestyle="--",
        linewidth=1.0,
    )
    # Vertical arrow for the maximum drawdown.
    ax.annotate(
        "",
        xy=(trough_pos, trough_value),
        xytext=(trough_pos, peak_value),
        arrowprops={"arrowstyle": "<->", "color": _CORAL, "linewidth": 2.0},
    )
    ax.text(
        trough_pos - 0.01 * n_days,
        trough_value - 0.04 * (equity.max() - equity.min()),
        f"max drawdown\n{max_drawdown:.0%}",
        color=_CORAL,
        ha="left",
        va="top",
    )
    # Horizontal arrow below the curve for the longest drawdown.
    y_min, y_max = equity.min(), equity.max()
    y_arrow = y_min - 0.12 * (y_max - y_min)
    ax.annotate(
        "",
        xy=(end_pos, y_arrow),
        xytext=(start_pos, y_arrow),
        arrowprops={"arrowstyle": "<->", "color": _CORAL, "linewidth": 2.0},
    )
    for pos in (start_pos, end_pos):
        ax.plot(
            [pos, pos],
            [y_arrow, drawdown_df["equity"].iloc[pos]],
            color=_GRAY,
            linestyle=":",
            linewidth=1.0,
        )
    ax.text(
        0.5 * (start_pos + end_pos),
        y_arrow - 0.03 * (y_max - y_min),
        f"longest drawdown: {duration} days",
        color=_CORAL,
        ha="center",
        va="top",
    )
    ax.set_ylim(y_arrow - 0.2 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    ax.set_xlabel("day")
    ax.set_ylabel("equity")
    return fig


# #############################################################################
# Cell 1.3: Sharpe ratio and leverage
# #############################################################################


def cell1_3_plot_sharpe_leverage(
    *,
    risk_free: float,
    excess_return: float,
    volatility: float,
    leverages: List[float],
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot the return and volatility of a strategy at different leverages.

    All the leveraged strategies are on the same line of slope equal to the
    Sharpe ratio, since leverage scales return and risk together.

    :param risk_free: risk-free rate per year
    :param excess_return: expected excess return per year at 1x leverage
    :param volatility: volatility per year at 1x leverage
    :param leverages: leverage multiples to mark on the line
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with the line of constant Sharpe ratio
    """
    sharpe = excess_return / volatility
    # Leverage scales the excess return and the volatility, not the Sharpe
    # ratio.
    points_df = pd.DataFrame({"leverage": leverages})
    points_df["volatility"] = points_df["leverage"] * volatility
    points_df["expected_return"] = (
        risk_free + points_df["leverage"] * excess_return
    )
    # Line of constant Sharpe ratio, from cash to beyond the highest leverage.
    max_volatility = 1.2 * points_df["volatility"].max()
    line_df = pd.DataFrame({"volatility": [0.0, max_volatility]})
    line_df["expected_return"] = risk_free + sharpe * line_df["volatility"]
    #
    fig, axes = _new_figure(figsize)
    ax = axes[0, 0]
    sns.lineplot(
        data=line_df,
        x="volatility",
        y="expected_return",
        color=_BLUE,
        linewidth=2.0,
        ax=ax,
    )
    sns.scatterplot(
        data=points_df,
        x="volatility",
        y="expected_return",
        color=_BLUE,
        s=80,
        zorder=3,
        ax=ax,
    )
    for _, row in points_df.iterrows():
        ax.annotate(
            f"{row['leverage']:g}x",
            xy=(row["volatility"], row["expected_return"]),
            xytext=(8, -16),
            textcoords="offset points",
        )
    # The risk-free asset is the point with no volatility.
    ax.scatter([0.0], [risk_free], color=_ORANGE, s=80, zorder=3, clip_on=False)
    ax.annotate(
        "$R_f$",
        xy=(0.0, risk_free),
        xytext=(8, -16),
        textcoords="offset points",
        color=_ORANGE,
    )
    ax.text(
        0.35 * max_volatility,
        risk_free + sharpe * 0.35 * max_volatility + 0.01,
        f"slope = $SR$ = {sharpe:.1f}",
        color=_BLUE,
        ha="right",
    )
    # Use ticks every 5% so that the percent labels are exact.
    percent = mticker.PercentFormatter(xmax=1.0, decimals=0)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(mticker.MultipleLocator(0.05))
        axis.set_major_formatter(percent)
    ax.set_xlabel("volatility (per year)")
    ax.set_ylabel("expected return (per year)")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    return fig


# #############################################################################
# Cell 1.4: Market impact around a buy order
# #############################################################################


def cell1_4_plot_market_impact(
    *,
    arrival_price: float,
    permanent: float,
    temporary: float,
    start: float,
    end: float,
    decay_time: float,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot the price path around a buy order, split into its impact components.

    The price rises while the order executes. The temporary part reverts after
    the execution, the permanent part stays.

    :param arrival_price: price when the order arrives
    :param permanent: permanent price impact, in price units
    :param temporary: peak temporary price impact, in price units
    :param start: time when the execution starts, in minutes
    :param end: time when the execution ends, in minutes
    :param decay_time: time scale of the reversion of the temporary impact,
        in minutes
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with the price path and the impact components
    """
    time = np.linspace(0.0, 2.2 * end, 400)
    # Impact accumulates linearly while the order executes. After the end,
    # the temporary impact decays and the permanent impact stays.
    progress = np.clip((time - start) / (end - start), 0.0, 1.0)
    decay = np.exp(-np.maximum(time - end, 0.0) / decay_time)
    price = arrival_price + permanent * progress + temporary * progress * decay
    price_df = pd.DataFrame({"time": time, "price": price})
    #
    fig, axes = _new_figure(figsize)
    ax = axes[0, 0]
    ax.axvspan(start, end, color=_GRAY, alpha=0.15, linewidth=0.0)
    sns.lineplot(
        data=price_df, x="time", y="price", color=_BLUE, linewidth=2.5, ax=ax
    )
    # Reference levels: arrival price and price after the temporary part decays.
    final_price = arrival_price + permanent
    t_right = time[-1]
    ax.hlines(arrival_price, start, t_right, color=_GRAY, linestyle="--")
    ax.hlines(final_price, end, t_right, color=_GRAY, linestyle="--")
    # Arrow for the permanent impact on the right, for the temporary one at
    # the end of the execution.
    ax.annotate(
        "",
        xy=(t_right, final_price),
        xytext=(t_right, arrival_price),
        arrowprops={"arrowstyle": "<->", "color": _CORAL, "linewidth": 2.0},
    )
    ax.text(
        t_right + 0.02 * t_right,
        0.5 * (arrival_price + final_price),
        "permanent",
        color=_CORAL,
        va="center",
    )
    ax.annotate(
        "",
        xy=(end, final_price + temporary),
        xytext=(end, final_price),
        arrowprops={"arrowstyle": "<->", "color": _CORAL, "linewidth": 2.0},
    )
    ax.text(
        end + 0.02 * t_right,
        final_price + 0.25 * temporary,
        "temporary",
        color=_CORAL,
        va="center",
    )
    ax.text(
        0.5 * (start + end),
        final_price + 1.2 * temporary,
        "execution",
        color=_GRAY,
        ha="center",
    )
    ax.text(
        t_right - 0.1 * t_right,
        arrival_price + 0.04 * temporary,
        "arrival price",
        ha="right",
        va="bottom",
    )
    ax.set_ylim(arrival_price - 0.1 * temporary, final_price + 1.45 * temporary)
    ax.set_xlabel("time (minutes)")
    ax.set_ylabel("price ($)")
    return fig


# #############################################################################
# Cell 1.5: Strategy capacity
# #############################################################################


def cell1_5_plot_strategy_capacity(
    *,
    gross_edge: float,
    capacity: float,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot gross PnL, impact cost, and net PnL as a function of the book size.

    Gross PnL is linear in the book size and the impact cost is quadratic, so
    the net PnL peaks at the capacity.

    :param gross_edge: gross return of the strategy per year on the book
    :param capacity: book size that maximizes the net PnL, in $ million
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with the three curves
    """
    # Net PnL = a * B - c * B^2 peaks at B = a / (2 * c).
    impact_coef = gross_edge / (2.0 * capacity)
    book_size = np.linspace(0.0, 2.0 * capacity, 201)
    pnl_df = pd.DataFrame(
        {
            "book_size": book_size,
            "gross edge": gross_edge * book_size,
            "impact cost": impact_coef * book_size**2,
        }
    )
    pnl_df["net PnL"] = pnl_df["gross edge"] - pnl_df["impact cost"]
    max_net = pnl_df["net PnL"].max()
    #
    fig, axes = _new_figure(figsize)
    ax = axes[0, 0]
    colors = {"gross edge": _GREEN, "impact cost": _CORAL, "net PnL": _BLUE}
    for column, color in colors.items():
        sns.lineplot(
            data=pnl_df,
            x="book_size",
            y=column,
            color=color,
            linewidth=3.0 if column == "net PnL" else 1.5,
            label=column,
            ax=ax,
        )
    ax.legend(loc="upper left", frameon=False)
    # Mark the capacity, where the net PnL peaks.
    ax.vlines(capacity, 0.0, max_net, color=_GRAY, linestyle="--")
    ax.hlines(max_net, 0.0, capacity, color=_GRAY, linestyle="--")
    ax.scatter([capacity], [max_net], color=_BLUE, s=60, zorder=3)
    ax.set_xticks([0.0, capacity, 2.0 * capacity])
    ax.set_xticklabels(["0", f"{capacity:g}", f"{2.0 * capacity:g}"])
    ax.text(
        capacity + 0.02 * 2.0 * capacity,
        0.08 * max_net,
        "capacity",
        color=_GRAY,
        ha="left",
    )
    ax.set_xlabel("book size ($ million)")
    ax.set_ylabel("PnL ($ million per year)")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    return fig


# #############################################################################
# Cell 2.1: Prices vs. returns
# #############################################################################


def cell2_1_plot_prices_vs_returns(
    *,
    n_days: int,
    mean_return: float,
    volatility: float,
    seed: int,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot a simulated price path and its returns on two aligned panels.

    :param n_days: number of days to simulate
    :param mean_return: mean daily return
    :param volatility: daily volatility of the return
    :param seed: seed of the random number generator
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with the price and the return panels
    """
    # Simulate IID returns and compound them into a price path.
    rng = np.random.default_rng(seed)
    returns = rng.normal(mean_return, volatility, size=n_days)
    start_price = 100.0
    data_df = pd.DataFrame(
        {
            "price": start_price * np.cumprod(1.0 + returns),
            "return": returns,
        }
    )
    #
    fig, axes = _new_figure(
        figsize, nrows=2, sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    ax_price, ax_return = axes[0, 0], axes[1, 0]
    sns.lineplot(
        data=data_df,
        x=data_df.index,
        y="price",
        color=_ORANGE,
        linewidth=1.5,
        ax=ax_price,
    )
    ax_price.axhline(start_price, color=_GRAY, linestyle="--", linewidth=1.0)
    ax_price.set_ylabel("price")
    #
    sns.lineplot(
        data=data_df,
        x=data_df.index,
        y="return",
        color=_BLUE,
        linewidth=0.8,
        ax=ax_return,
    )
    ax_return.axhline(
        data_df["return"].mean(), color=_GRAY, linestyle="--", linewidth=1.0
    )
    ax_return.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_return.set_ylabel("return")
    ax_return.set_xlabel("day")
    return fig


# #############################################################################
# Cell 2.2: Kurtosis and fat tails
# #############################################################################


def cell2_2_plot_fat_tails(
    *,
    degrees_of_freedom: int,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot the log density of a Gaussian and a Student-t with unit variance.

    :param degrees_of_freedom: degrees of freedom of the Student-t, larger
        than 2 so that the variance is finite
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with the two log densities
    """
    # Rescale the Student-t so that its variance is 1, like the Gaussian.
    scale = np.sqrt((degrees_of_freedom - 2.0) / degrees_of_freedom)
    x = np.linspace(-4.5, 4.5, 361)
    log_density_df = pd.DataFrame(
        {
            "x": x,
            "Gaussian": sstats.norm.logpdf(x),
            "Student-t": sstats.t.logpdf(x, degrees_of_freedom, scale=scale),
        }
    )
    #
    fig, axes = _new_figure(figsize)
    ax = axes[0, 0]
    colors = {"Gaussian": _BLUE, "Student-t": _CORAL}
    labels = {
        "Gaussian": "Gaussian",
        "Student-t": f"Student-t ({degrees_of_freedom} d.o.f.)",
    }
    for column, color in colors.items():
        sns.lineplot(
            data=log_density_df,
            x="x",
            y=column,
            color=color,
            linewidth=2.5,
            label=labels[column],
            ax=ax,
        )
    ax.legend(loc="lower center", frameon=False)
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_xticklabels(
        [r"$-4\sigma$", r"$-2\sigma$", "0", r"$2\sigma$", r"$4\sigma$"]
    )
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(bottom=-10.0)
    ax.set_xlabel("return")
    ax.set_ylabel("log density")
    return fig


# #############################################################################
# Cell 2.3: Volatility clustering
# #############################################################################


def _simulate_garch11(
    *, n_obs: int, alpha: float, beta: float, seed: int, burn_in: int = 500
) -> pd.Series:
    """
    Simulate returns from a GARCH(1,1) process with unit unconditional variance.

    The conditional variance follows
    `sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2`.

    :param n_obs: number of returns to return
    :param alpha: weight of the previous squared return
    :param beta: weight of the previous variance
    :param seed: seed of the random number generator
    :param burn_in: number of initial returns to drop
    :return: simulated returns
    """
    hdbg.dassert_lte(0.0, alpha)
    hdbg.dassert_lte(0.0, beta)
    # The process is stationary only when alpha + beta < 1.
    hdbg.dassert_lt(alpha + beta, 1.0)
    # Choose omega so that the unconditional variance omega / (1 - alpha -
    # beta) equals 1.
    omega = 1.0 - alpha - beta
    rng = np.random.default_rng(seed)
    n_total = n_obs + burn_in
    shocks = rng.standard_normal(n_total)
    returns = np.zeros(n_total)
    variance = 1.0
    for t in range(n_total):
        returns[t] = np.sqrt(variance) * shocks[t]
        variance = omega + alpha * returns[t] ** 2 + beta * variance
    returns = pd.Series(returns[burn_in:], name="return")
    return returns


def cell2_3_plot_volatility_clustering(
    *,
    n_obs: int,
    alpha: float,
    beta: float,
    seed: int,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot simulated GARCH(1,1) returns, where large moves follow large moves.

    :param n_obs: number of returns to simulate
    :param alpha: weight of the previous squared return
    :param beta: weight of the previous variance
    :param seed: seed of the random number generator
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with the return series
    """
    returns = _simulate_garch11(n_obs=n_obs, alpha=alpha, beta=beta, seed=seed)
    #
    fig, axes = _new_figure(figsize)
    ax = axes[0, 0]
    sns.lineplot(x=returns.index, y=returns, color=_BLUE, linewidth=0.8, ax=ax)
    ax.axhline(0.0, color=_GRAY, linestyle="--", linewidth=1.0)
    ax.set_xlabel("day")
    ax.set_ylabel("return (%)")
    return fig


# #############################################################################
# Cell 2.4: Returns vs. squared returns
# #############################################################################


def _compute_acf(x: pd.Series, *, n_lags: int) -> pd.Series:
    """
    Compute the sample autocorrelation function for lags 1 to `n_lags`.

    :param x: time series
    :param n_lags: number of lags
    :return: autocorrelation indexed by lag
    """
    values = x.to_numpy() - x.mean()
    variance_sum = np.sum(values**2)
    lags = np.arange(1, n_lags + 1)
    acf = [np.sum(values[:-lag] * values[lag:]) / variance_sum for lag in lags]
    acf = pd.Series(acf, index=lags, name="acf")
    return acf


def _compute_garch11_squared_acf(
    *, alpha: float, beta: float, n_lags: int
) -> pd.Series:
    """
    Compute the theoretical autocorrelation of squared GARCH(1,1) returns.

    The autocorrelation is
    `rho_1 = alpha * (1 - alpha * beta - beta^2) / (1 - 2 * alpha * beta -
    beta^2)` at lag 1, and `rho_k = rho_1 * (alpha + beta)^(k - 1)` after.

    :param alpha: weight of the previous squared return
    :param beta: weight of the previous variance
    :param n_lags: number of lags
    :return: theoretical autocorrelation indexed by lag
    """
    rho_1 = alpha * (1.0 - alpha * beta - beta**2)
    rho_1 = rho_1 / (1.0 - 2.0 * alpha * beta - beta**2)
    lags = np.arange(1, n_lags + 1)
    acf = pd.Series(rho_1 * (alpha + beta) ** (lags - 1), index=lags, name="acf")
    return acf


def cell2_4_plot_returns_vs_squared_returns(
    *,
    n_obs: int,
    alpha: float,
    beta: float,
    n_lags: int,
    seed: int,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot the autocorrelation of returns and of squared returns side by side.

    Returns look like white noise, squared returns have a slowly decaying
    autocorrelation. The panel of squared returns also shows the theoretical
    GARCH(1,1) autocorrelation.

    :param n_obs: number of returns to simulate
    :param alpha: weight of the previous squared return
    :param beta: weight of the previous variance
    :param n_lags: number of lags to plot
    :param seed: seed of the random number generator
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with the two autocorrelation panels
    """
    returns = _simulate_garch11(n_obs=n_obs, alpha=alpha, beta=beta, seed=seed)
    acf_returns = _compute_acf(returns, n_lags=n_lags)
    acf_squared = _compute_acf(returns**2, n_lags=n_lags)
    acf_theory = _compute_garch11_squared_acf(
        alpha=alpha, beta=beta, n_lags=n_lags
    )
    # 95% band of the autocorrelation of white noise.
    band = 1.96 / np.sqrt(n_obs)
    #
    fig, axes = _new_figure(figsize, ncols=2, sharey=True)
    panels = [
        (axes[0, 0], acf_returns, _BLUE, "returns"),
        (axes[0, 1], acf_squared, _CORAL, "squared returns"),
    ]
    for ax, acf, color, title in panels:
        sns.barplot(x=acf.index, y=acf.to_numpy(), color=color, width=0.6, ax=ax)
        ax.axhline(band, color=_GRAY, linestyle="--", linewidth=1.0)
        ax.axhline(-band, color=_GRAY, linestyle="--", linewidth=1.0)
        ax.set_title(title, color=color)
        ax.set_xlabel("lag")
        ax.set_xticks(np.arange(4, n_lags, 5))
    # Theory is dotted and light, empirical is solid (see the plot rules).
    axes[0, 1].plot(
        np.arange(n_lags),
        acf_theory.to_numpy(),
        color=_CORAL,
        linestyle=":",
        linewidth=2.0,
        alpha=0.6,
    )
    axes[0, 0].set_ylabel("autocorrelation")
    axes[0, 0].text(
        0.97,
        0.95,
        f"$T$ = {n_obs}\n95% band = $\\pm${band:.3f}",
        transform=axes[0, 0].transAxes,
        ha="right",
        va="top",
    )
    return fig


# #############################################################################
# Cell 2.5: Low signal-to-noise
# #############################################################################


def cell2_5_plot_weak_signal(
    *,
    n_obs: int,
    information_coefficient: float,
    seed: int,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot realized returns against forecasts for a given information coefficient.

    :param n_obs: number of forecast and return pairs to simulate
    :param information_coefficient: true correlation between forecast and
        realized return
    :param seed: seed of the random number generator
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with the scatter plot and the fitted line
    """
    # Standardized forecast and return with correlation equal to the IC.
    rng = np.random.default_rng(seed)
    forecast = rng.standard_normal(n_obs)
    noise = rng.standard_normal(n_obs)
    ic = information_coefficient
    realized = ic * forecast + np.sqrt(1.0 - ic**2) * noise
    data_df = pd.DataFrame({"forecast": forecast, "return": realized})
    sample_ic = float(np.corrcoef(forecast, realized)[0, 1])
    #
    fig, axes = _new_figure(figsize)
    ax = axes[0, 0]
    sns.scatterplot(
        data=data_df,
        x="forecast",
        y="return",
        color=_BLUE,
        alpha=0.4,
        s=15,
        linewidth=0.0,
        ax=ax,
    )
    sns.regplot(
        data=data_df,
        x="forecast",
        y="return",
        scatter=False,
        ci=None,
        line_kws={"color": _CORAL, "linewidth": 2.5},
        ax=ax,
    )
    ax.text(
        0.03,
        0.97,
        f"true IC = {ic:.2f}\nsample IC = {sample_ic:.2f}\n"
        f"$R^2$ = {sample_ic**2:.2%}",
        transform=ax.transAxes,
        va="top",
    )
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel("forecast")
    ax.set_ylabel("realized return")
    return fig


# #############################################################################
# Cell 2.6: Overfitting noise
# #############################################################################


def _compute_overfitting_errors(
    *,
    n_train: int,
    n_test: int,
    n_features: int,
    n_signal: int,
    coef: float,
    n_trials: int,
    seed: int,
) -> pd.DataFrame:
    """
    Compute the in-sample and out-of-sample error of nested linear models.

    Only the first `n_signal` features drive the target. The model with `k`
    features uses the first `k` columns, so each model adds noise features to
    the previous one.

    :param n_train: number of training observations
    :param n_test: number of test observations
    :param n_features: number of candidate features
    :param n_signal: number of features with a non-zero coefficient
    :param coef: coefficient of each signal feature
    :param n_trials: number of simulated datasets to average over
    :param seed: seed of the random number generator
    :return: dataframe indexed by the number of features, with columns
        `in-sample` and `out-of-sample` for the mean squared error
    """
    rng = np.random.default_rng(seed)
    true_coef = np.zeros(n_features)
    true_coef[:n_signal] = coef
    errors = np.zeros((n_trials, n_features, 2))
    for trial in range(n_trials):
        # Training and test sets share the same data-generating process.
        x_train = rng.standard_normal((n_train, n_features))
        y_train = x_train @ true_coef + rng.standard_normal(n_train)
        x_test = rng.standard_normal((n_test, n_features))
        y_test = x_test @ true_coef + rng.standard_normal(n_test)
        for k in range(1, n_features + 1):
            # Fit OLS on the first `k` features.
            fit = np.linalg.lstsq(x_train[:, :k], y_train, rcond=None)[0]
            in_sample = np.mean((y_train - x_train[:, :k] @ fit) ** 2)
            out_of_sample = np.mean((y_test - x_test[:, :k] @ fit) ** 2)
            errors[trial, k - 1] = (in_sample, out_of_sample)
    error_df = pd.DataFrame(
        errors.mean(axis=0),
        index=pd.RangeIndex(1, n_features + 1, name="n_features"),
        columns=pd.Index(["in-sample", "out-of-sample"]),
    )
    return error_df


def cell2_6_plot_overfitting(
    *,
    n_train: int,
    n_features: int,
    n_signal: int,
    seed: int,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot in-sample and out-of-sample error against the model complexity.

    :param n_train: number of training observations
    :param n_features: number of candidate features
    :param n_signal: number of features that carry signal
    :param seed: seed of the random number generator
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with the two error curves
    """
    error_df = _compute_overfitting_errors(
        n_train=n_train,
        n_test=2000,
        n_features=n_features,
        n_signal=n_signal,
        coef=0.4,
        n_trials=200,
        seed=seed,
    )
    best_k = int(np.argmin(error_df["out-of-sample"].to_numpy())) + 1
    #
    fig, axes = _new_figure(figsize)
    ax = axes[0, 0]
    colors = {"in-sample": _BLUE, "out-of-sample": _CORAL}
    for column, color in colors.items():
        sns.lineplot(
            data=error_df,
            x=error_df.index,
            y=column,
            color=color,
            linewidth=2.5,
            ax=ax,
        )
    # Mark the complexity with the lowest out-of-sample error.
    ax.axvline(best_k, color=_GRAY, linestyle="--", linewidth=1.0)
    ax.text(
        best_k + 0.01 * n_features,
        0.03,
        f"optimal: {best_k}",
        color=_GRAY,
        ha="left",
        va="bottom",
    )
    ax.text(
        n_features,
        error_df["in-sample"].iloc[-1] + 0.2,
        "in-sample",
        color=_BLUE,
        ha="right",
        va="bottom",
    )
    ax.text(
        n_features,
        error_df["out-of-sample"].iloc[-1] + 0.1,
        "out-of-sample",
        color=_CORAL,
        ha="right",
        va="bottom",
    )
    ax.set_xlabel("model complexity (number of features)")
    ax.set_ylabel("mean squared error")
    ax.set_xlim(0, n_features + 1)
    ax.set_ylim(0.0, 1.15 * error_df["out-of-sample"].max())
    return fig
