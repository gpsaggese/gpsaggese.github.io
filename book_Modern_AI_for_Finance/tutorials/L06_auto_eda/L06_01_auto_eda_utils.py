"""
Utility functions to generate the figures of Lesson 06.01 "Auto EDA".

Import as:

import L06_01_auto_eda_utils as l0auedut
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.figure as mfigure
import matplotlib.pyplot as plt
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


# Colors of the figures of Lessons 01.01 and 02.01, so that the figures of the
# lessons look alike.
_BLUE = "#3C6FB0"
_CORAL = "#C0455B"
_ORANGE = "#D98E2B"
_GREEN = "#4F9A5C"
_GRAY = "#7B8794"
_LIGHT_GRAY = "#D5DBE1"

# Number of trading days in one year and in one month.
_DAYS_PER_YEAR = 252
_DAYS_PER_MONTH = 21

# 5% critical value of the supremum of the absolute value of a Brownian
# bridge, used by both CUSUM tests.
_BRIDGE_CRITICAL_5PCT = 1.358


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


def _compute_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple returns without filling the missing prices.

    :param prices_df: prices with dates as rows and assets as columns
    :return: simple returns, NaN when the price or the previous price is
        missing
    """
    returns_df = prices_df.pct_change(fill_method=None)
    return returns_df


# #############################################################################
# Running example: a daily equity panel
# #############################################################################


def simulate_garch_returns(
    *,
    n_days: int,
    daily_vol: float,
    alpha: float,
    beta: float,
    dof: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate daily returns with GARCH(1,1) volatility and Student-t shocks.

    :param n_days: number of days
    :param daily_vol: unconditional daily volatility
    :param alpha: weight of the last squared return in the variance
    :param beta: weight of the last variance in the variance
    :param dof: degrees of freedom of the Student-t shocks, above 2
    :param rng: random number generator
    :return: array of daily returns
    """
    hdbg.dassert_lt(alpha + beta, 1.0)
    hdbg.dassert_lt(2.0, dof)
    omega = daily_vol**2 * (1.0 - alpha - beta)
    # Scale the Student-t shocks to unit variance.
    shocks = rng.standard_t(dof, size=n_days) * np.sqrt((dof - 2.0) / dof)
    returns = np.empty(n_days)
    variance = daily_vol**2
    for day in range(n_days):
        returns[day] = np.sqrt(variance) * shocks[day]
        variance = omega + alpha * returns[day] ** 2 + beta * variance
    return returns


def simulate_panel(
    *,
    n_stocks: int,
    n_days: int,
    ipo_fraction: float,
    delisting_price: float,
    outage_fraction: float,
    outage_start: int,
    outage_days: int,
    seed: int,
) -> pd.DataFrame:
    """
    Simulate the daily close prices of a panel of stocks.

    - The return of each stock is its beta times a market return with
      GARCH(1,1) volatility, plus Student-t idiosyncratic noise
    - A fraction of the stocks lists after the start of the panel (IPO)
    - A stock delists the first day its price falls below `delisting_price`
      times its listing price
    - A vendor outage removes the prices of a fraction of the stocks for some
      days
    - Prices are rounded to the cent, as in a real database

    :param n_stocks: number of stocks
    :param n_days: number of business days
    :param ipo_fraction: fraction of the stocks that list after the start
    :param delisting_price: price, relative to the listing price, that
        triggers a delisting
    :param outage_fraction: fraction of the stocks hit by the vendor outage
    :param outage_start: first day of the vendor outage
    :param outage_days: number of days of the vendor outage
    :param seed: seed of the random number generator
    :return: close prices with dates as rows and stocks as columns, NaN when a
        stock is not listed or its price is missing
    """
    hdbg.dassert_lt(outage_start + outage_days, n_days)
    rng = np.random.default_rng(seed)
    market = simulate_garch_returns(
        n_days=n_days, daily_vol=0.01, alpha=0.08, beta=0.90, dof=5.0, rng=rng
    )
    betas = rng.uniform(0.6, 1.4, size=n_stocks)
    idio_vols = rng.uniform(0.01, 0.025, size=n_stocks)
    dof = 4.0
    noise = rng.standard_t(dof, size=(n_days, n_stocks))
    noise *= np.sqrt((dof - 2.0) / dof)
    returns = 0.0003 + market[:, None] * betas[None, :]
    returns += noise * idio_vols[None, :]
    # A simple return below -100% would make the price negative.
    returns = np.maximum(returns, -0.9)
    # Day of the listing: the first day, or a random day for an IPO.
    is_ipo = rng.random(n_stocks) < ipo_fraction
    ipo_day = rng.integers(1, int(0.8 * n_days), size=n_stocks)
    listing_day = np.where(is_ipo, ipo_day, 0)
    days = np.arange(n_days)[:, None]
    returns = np.where(days > listing_day[None, :], returns, 0.0)
    prices = 20.0 * np.cumprod(1.0 + returns, axis=0)
    # Delist a stock the first day its price falls below the threshold.
    below = (prices < 20.0 * delisting_price) & (days >= listing_day[None, :])
    delisting_day = np.where(below.any(axis=0), below.argmax(axis=0), n_days)
    is_listed = (days >= listing_day[None, :]) & (days <= delisting_day[None, :])
    prices = np.where(is_listed, np.round(prices, 2), np.nan)
    # Remove the prices of some stocks during the vendor outage.
    is_hit = rng.random(n_stocks) < outage_fraction
    in_outage = (days >= outage_start) & (days < outage_start + outage_days)
    prices = np.where(in_outage & is_hit[None, :], np.nan, prices)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    columns = [f"S{idx:03d}" for idx in range(n_stocks)]
    prices_df = pd.DataFrame(prices, index=dates, columns=columns)
    return prices_df


# #############################################################################
# Cell 1.1: Panel coverage by asset and date
# #############################################################################


def cell1_1_plot_panel_coverage(
    prices_df: pd.DataFrame,
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot the coverage of a panel by stock and month, and the listed stocks.

    :param prices_df: close prices with dates as rows and stocks as columns
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with the coverage heatmap and the number of stocks with a
        price in the full and in the survivor-only database
    """
    is_present = prices_df.notna()
    # Fraction of the business days of each month with a price.
    months = prices_df.index.to_period("M")
    coverage_df = is_present.groupby(months).mean()
    # Sort the stocks by first date with a price, then by last date.
    spans_df = pd.DataFrame(
        {"first": is_present.idxmax(), "last": is_present[::-1].idxmax()}
    )
    order = spans_df.sort_values(
        ["first", "last"], ascending=[True, False]
    ).index
    # A survivor-only database keeps the stocks with a price on the last day.
    survivors = prices_df.columns[is_present.iloc[-1]]
    n_exits = len(prices_df.columns) - len(survivors)
    #
    fig, axes = _new_figure(
        figsize,
        nrows=2,
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    ax = axes[0, 0]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "coverage", ["#FFFFFF", _BLUE]
    )
    x_start = mdates.date2num(prices_df.index[0])
    x_end = mdates.date2num(prices_df.index[-1])
    ax.imshow(
        coverage_df[order].T.to_numpy(),
        aspect="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        extent=(x_start, x_end, len(order), 0),
        interpolation="nearest",
    )
    ax.set_ylabel("stocks by listing date")
    ax.set_yticks([])
    #
    ax = axes[1, 0]
    ax.plot(
        prices_df.index,
        is_present.sum(axis=1),
        color=_BLUE,
        linewidth=1.8,
        label=f"full database: {n_exits} exits",
    )
    ax.plot(
        prices_df.index,
        is_present[survivors].sum(axis=1),
        color=_CORAL,
        linewidth=1.8,
        label="survivors only: 0 exits",
    )
    # Mark the vendor outage at the largest one-day drop of the stock count.
    count_change = is_present.sum(axis=1).diff()
    outage_date = count_change.idxmin()
    ax.annotate(
        "vendor outage",
        (
            mdates.date2num(outage_date),
            float(is_present.sum(axis=1)[outage_date]),
        ),
        xytext=(-20, -18),
        textcoords="offset points",
        ha="right",
        color=_GRAY,
        arrowprops={"arrowstyle": "->", "color": _GRAY},
    )
    ax.set_ylabel("stocks with a price")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.xaxis_date()
    return fig


# #############################################################################
# Cell 2.1: Stale prices
# #############################################################################


def add_stale_prices(
    prices_df: pd.DataFrame,
    *,
    stocks: List[str],
    stale_prob: float,
    seed: int,
) -> pd.DataFrame:
    """
    Replace prices with the previous price, as for a stock that did not trade.

    :param prices_df: close prices with dates as rows and stocks as columns
    :param stocks: stocks whose prices become stale
    :param stale_prob: probability that a stock does not trade on a day
    :param seed: seed of the random number generator
    :return: copy of the prices with stale prices for `stocks`
    """
    hdbg.dassert_lte(0.0, stale_prob)
    hdbg.dassert_lt(stale_prob, 1.0)
    rng = np.random.default_rng(seed)
    stale_df = prices_df.copy()
    is_stale = rng.random((len(prices_df), len(stocks))) < stale_prob
    stocks_df = prices_df[stocks]
    # Carry the last traded price forward, but keep missing the days when the
    # stock is not listed.
    filled_df = stocks_df.mask(is_stale).ffill()
    stale_df[stocks] = filled_df.where(stocks_df.notna())
    return stale_df


def _longest_zero_run(returns: pd.Series) -> int:
    """
    Compute the longest run of consecutive zero returns.

    :param returns: returns of one asset
    :return: number of days of the longest run of zero returns
    """
    is_zero = returns == 0.0
    # Each non-zero return starts a new run.
    run_id = (~is_zero).cumsum()
    longest = int(is_zero.groupby(run_id).sum().max())
    return longest


def compute_stale_stats(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the statistics of a stale-price detector for each stock.

    :param prices_df: close prices with dates as rows and stocks as columns
    :return: dataframe with one row per stock and the columns
        - `zero_fraction`: fraction of the returns equal to 0
        - `longest_run`: longest run of consecutive zero returns, in days
    """
    returns_df = _compute_returns(prices_df)
    n_returns = returns_df.notna().sum()
    zero_fraction = (returns_df == 0.0).sum() / n_returns
    longest_run = returns_df.apply(_longest_zero_run)
    stats_df = pd.DataFrame(
        {"zero_fraction": zero_fraction, "longest_run": longest_run}
    )
    return stats_df


def compute_index_autocorrelation(prices_df: pd.DataFrame) -> float:
    """
    Compute the lag-1 autocorrelation of an equal-weighted index.

    :param prices_df: close prices with dates as rows and stocks as columns
    :return: lag-1 autocorrelation of the daily returns of the index
    """
    index_returns = _compute_returns(prices_df).mean(axis=1)
    autocorr = float(index_returns.autocorr(lag=1))
    return autocorr


def cell2_1_plot_stale_prices(
    clean_df: pd.DataFrame,
    stale_df: pd.DataFrame,
    *,
    stale_stocks: List[str],
    stock: str,
    n_days_shown: int,
    zero_fraction_threshold: float,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot one stock with stale prices and a stale-price detector.

    :param clean_df: close prices of the stocks that trade every day
    :param stale_df: the same prices with stale prices for `stale_stocks`
    :param stale_stocks: stocks with stale prices
    :param stock: stale stock to plot
    :param n_days_shown: number of days of the price path to plot
    :param zero_fraction_threshold: fraction of zero returns above which a
        stock is flagged
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with the price path and the detector scatter
    """
    hdbg.dassert_in(stock, stale_stocks)
    fig, axes = _new_figure(
        figsize, nrows=2, gridspec_kw={"height_ratios": [1.0, 1.2]}
    )
    # Plot a window of the stale stock, starting on its first listed day.
    ax = axes[0, 0]
    first_day = stale_df[stock].first_valid_index()
    window = stale_df.loc[first_day:, stock].iloc[:n_days_shown]
    true_window = clean_df.loc[window.index, stock]
    ax.plot(
        true_window.index,
        true_window,
        color=_LIGHT_GRAY,
        linewidth=1.5,
        label="true price",
    )
    ax.plot(
        window.index,
        window,
        color=_BLUE,
        linewidth=1.8,
        drawstyle="steps-post",
        label="recorded price",
    )
    is_zero = window.diff() == 0.0
    ax.plot(
        window.index[is_zero],
        window[is_zero],
        "o",
        color=_CORAL,
        markersize=3,
        label="zero return",
    )
    ax.set_ylabel("price")
    ax.legend(frameon=False, loc="best", fontsize=8)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    # Scatter the detector statistics of all the stocks.
    ax = axes[1, 0]
    stats_df = compute_stale_stats(stale_df)
    is_stale = stats_df.index.isin(stale_stocks)
    for flag, color, label in [
        (False, _BLUE, "trades daily"),
        (True, _CORAL, "stale"),
    ]:
        subset_df = stats_df[is_stale == flag]
        ax.scatter(
            subset_df["zero_fraction"],
            subset_df["longest_run"],
            color=color,
            s=14,
            alpha=0.8,
            label=label,
        )
    ax.axvline(
        zero_fraction_threshold, color=_GRAY, linestyle="--", linewidth=1.0
    )
    ax.set_xlabel("fraction of zero returns")
    ax.set_ylabel("longest flat run (days)")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


# #############################################################################
# Cell 2.2: Unadjusted split
# #############################################################################


def add_unadjusted_split(
    prices_df: pd.DataFrame,
    *,
    stock: str,
    split_date: str,
    ratio: float,
) -> pd.DataFrame:
    """
    Divide the prices of a stock from a split date on, without adjusting.

    :param prices_df: split-adjusted close prices
    :param stock: stock that splits
    :param split_date: first date with the post-split price
    :param ratio: number of new shares per old share (e.g., 2 for a 2-for-1
        split)
    :return: copy of the prices with the unadjusted split
    """
    hdbg.dassert_lt(1.0, ratio)
    raw_df = prices_df.copy()
    is_after = raw_df.index >= pd.Timestamp(split_date)
    raw_df.loc[is_after, stock] = np.round(
        raw_df.loc[is_after, stock] / ratio, 2
    )
    return raw_df


def _robust_z_scores(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute z-scores with the median and the MAD of each column.

    :param returns_df: returns with dates as rows and assets as columns
    :return: robust z-scores with the same shape as `returns_df`
    """
    median = returns_df.median()
    mad = (returns_df - median).abs().median()
    # Scale the MAD so that it estimates the standard deviation of a Gaussian.
    z_scores_df = (returns_df - median) / (1.4826 * mad)
    return z_scores_df


def detect_split_candidates(
    prices_df: pd.DataFrame,
    *,
    z_threshold: float,
    ratios: List[float],
    tolerance: float,
) -> pd.DataFrame:
    """
    Flag extreme returns and check whether they match a split ratio.

    A split with ratio `k` without price adjustment produces a return of
    `1 / k - 1`, e.g., -50% for a 2-for-1 split.

    :param prices_df: close prices with dates as rows and stocks as columns
    :param z_threshold: absolute robust z-score above which a return is flagged
    :param ratios: split ratios to test
    :param tolerance: largest distance between a return and `1 / k - 1`
    :return: dataframe with one row per flagged return and the columns
        `date`, `stock`, `return`, `robust_z`, `split_ratio` (NaN when no
        ratio matches)
    """
    returns_df = _compute_returns(prices_df)
    z_scores_df = _robust_z_scores(returns_df)
    is_flagged = z_scores_df.abs() > z_threshold
    flagged = returns_df[is_flagged].stack()
    rows = []
    for (date, stock), ret in flagged.items():
        matches = [k for k in ratios if abs(ret - (1.0 / k - 1.0)) < tolerance]
        rows.append(
            {
                "date": date,
                "stock": stock,
                "return": ret,
                "robust_z": z_scores_df.loc[date, stock],
                "split_ratio": matches[0] if matches else np.nan,
            }
        )
    candidates_df = pd.DataFrame(rows)
    return candidates_df


def cell2_2_plot_split_detection(
    adjusted_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    *,
    stock: str,
    ratio: float,
    z_threshold: float,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot a stock with an unadjusted split and the robust return band.

    :param adjusted_df: split-adjusted close prices
    :param raw_df: close prices with the unadjusted split
    :param stock: stock that splits
    :param ratio: split ratio, used in the annotation
    :param z_threshold: absolute robust z-score of the band
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with the raw and adjusted price, and the daily returns
    """
    fig, axes = _new_figure(figsize, nrows=2, sharex=True)
    ax = axes[0, 0]
    ax.plot(
        adjusted_df.index,
        adjusted_df[stock],
        color=_BLUE,
        linewidth=1.5,
        label="adjusted price",
    )
    ax.plot(
        raw_df.index,
        raw_df[stock],
        color=_CORAL,
        linewidth=1.5,
        label="raw price",
    )
    ax.set_ylabel("price")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    # Plot the returns of the raw prices with the robust band.
    ax = axes[1, 0]
    returns = _compute_returns(raw_df[[stock]])[stock]
    median = returns.median()
    mad = (returns - median).abs().median()
    band = z_threshold * 1.4826 * mad
    ax.plot(returns.index, returns, ".", color=_GRAY, markersize=2)
    for level in [median - band, median + band]:
        ax.axhline(level, color=_BLUE, linestyle="--", linewidth=1.0)
    split_date = returns.idxmin()
    split_return = float(returns[split_date])
    ax.plot(split_date, split_return, "o", color=_CORAL, markersize=6)
    ax.annotate(
        f"{split_return:.1%}: {ratio:g}-for-1 split?",
        (mdates.date2num(split_date), split_return),
        xytext=(10, 4),
        textcoords="offset points",
        color=_CORAL,
    )
    ax.set_ylabel("daily return")
    ax.xaxis_date()
    return fig


# #############################################################################
# Cell 3.1: Structural breaks with CUSUM
# #############################################################################


def simulate_strategy_returns(
    *,
    segments: List[Tuple[int, float, float]],
    seed: int,
) -> pd.Series:
    """
    Simulate Gaussian daily returns of a strategy with regimes.

    :param segments: list of `(n_days, annual_sharpe, annual_vol)` for each
        regime, in time order
    :param seed: seed of the random number generator
    :return: daily returns indexed by business days
    """
    rng = np.random.default_rng(seed)
    chunks = []
    for n_days, annual_sharpe, annual_vol in segments:
        daily_vol = annual_vol / np.sqrt(_DAYS_PER_YEAR)
        daily_mean = annual_sharpe * annual_vol / _DAYS_PER_YEAR
        chunks.append(rng.normal(daily_mean, daily_vol, size=n_days))
    returns = np.concatenate(chunks)
    dates = pd.bdate_range("2015-01-01", periods=len(returns))
    returns_srs = pd.Series(returns, index=dates)
    return returns_srs


def compute_cusum_stats(returns: pd.Series) -> pd.DataFrame:
    """
    Compute the CUSUM statistics of a break in the mean and in the variance.

    - Mean: cumulative sum of the demeaned returns, scaled by the standard
      deviation and `sqrt(T)` (OLS-CUSUM)
    - Variance: centered cumulative sum of the squared demeaned returns
      (cumulative sum of squares)
    - Without a break both statistics converge to a Brownian bridge, so the
      supremum of their absolute value has 5% critical value 1.358

    :param returns: daily returns
    :return: dataframe indexed like `returns` with the columns `mean_cusum` and
        `variance_cusum`
    """
    values = returns.to_numpy()
    n_obs = len(values)
    hdbg.dassert_lt(2, n_obs)
    resid = values - values.mean()
    mean_cusum = np.cumsum(resid) / (resid.std(ddof=1) * np.sqrt(n_obs))
    squares = resid**2
    fraction = np.arange(1, n_obs + 1) / n_obs
    variance_cusum = np.sqrt(n_obs / 2.0) * (
        np.cumsum(squares) / squares.sum() - fraction
    )
    stats_df = pd.DataFrame(
        {"mean_cusum": mean_cusum, "variance_cusum": variance_cusum},
        index=returns.index,
    )
    return stats_df


def cell3_1_plot_structural_breaks(
    returns: pd.Series,
    *,
    break_days: List[int],
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot the cumulative return of a strategy and its CUSUM statistics.

    :param returns: daily returns of the strategy
    :param break_days: days of the true breaks, marked with vertical lines
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with the cumulative return and the two CUSUM statistics
    """
    stats_df = compute_cusum_stats(returns)
    fig, axes = _new_figure(figsize, nrows=2, sharex=True)
    ax = axes[0, 0]
    ax.plot(returns.index, returns.cumsum(), color=_BLUE, linewidth=1.5)
    ax.set_ylabel("cumulative return")
    for ax_ in axes[:, 0]:
        for day in break_days:
            ax_.axvline(
                returns.index[day], color=_GRAY, linestyle=":", linewidth=1.0
            )
    ax = axes[1, 0]
    curves = [
        ("mean_cusum", _BLUE, "break in mean"),
        ("variance_cusum", _CORAL, "break in variance"),
    ]
    for column, color, label in curves:
        statistic = stats_df[column].abs()
        verdict = (
            "detected"
            if statistic.max() > _BRIDGE_CRITICAL_5PCT
            else "not detected"
        )
        ax.plot(
            statistic.index,
            statistic,
            color=color,
            linewidth=1.5,
            label=f"{label}: max {statistic.max():.2f}, {verdict}",
        )
    ax.axhline(_BRIDGE_CRITICAL_5PCT, color=_GRAY, linestyle="--", linewidth=1.0)
    # Leave room above the curves for the legend.
    ax.set_ylim(0.0, 1.5 * float(stats_df.abs().max().max()))
    ax.set_ylabel("|CUSUM|")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.xaxis_date()
    return fig


# #############################################################################
# Cell 3.2: Regime-aware summaries
# #############################################################################


def cell3_2_summarize_by_period(
    returns: pd.Series,
    *,
    boundaries: List[int],
    labels: List[str],
) -> pd.DataFrame:
    """
    Summarize the returns of a strategy over the full sample and by period.

    :param returns: daily returns of the strategy
    :param boundaries: first day of each period after the first one
    :param labels: label of each period, one more than `boundaries`
    :return: dataframe with one row per period and the columns `years`,
        `annual_return`, `annual_vol`, `sharpe`
    """
    hdbg.dassert_eq(len(labels), len(boundaries) + 1)
    edges = [0] + list(boundaries) + [len(returns)]
    periods = {"full sample": returns}
    for label, start, end in zip(labels, edges[:-1], edges[1:]):
        periods[label] = returns.iloc[start:end]
    rows = []
    for label, period in periods.items():
        annual_return = period.mean() * _DAYS_PER_YEAR
        annual_vol = period.std(ddof=1) * np.sqrt(_DAYS_PER_YEAR)
        rows.append(
            {
                "period": label,
                "years": len(period) / _DAYS_PER_YEAR,
                "annual_return": annual_return,
                "annual_vol": annual_vol,
                "sharpe": annual_return / annual_vol,
            }
        )
    summary_df = pd.DataFrame(rows).set_index("period")
    return summary_df


# #############################################################################
# Cell 4.1: Stylized facts test suite
# #############################################################################


def build_corrupted_prices(
    *,
    n_days: int,
    missing_prob: float,
    split_ratio: float,
    seed: int,
) -> Dict[str, pd.Series]:
    """
    Build a clean index price series and corrupted versions of it.

    - `clean`: GARCH(1,1) returns with Student-t shocks
    - `stale`: a random fraction of the days repeat the previous price
    - `interpolated`: the same days are filled by linear interpolation
    - `shuffled`: the clean returns in random order
    - `unadjusted split`: the price is divided by `split_ratio` halfway

    :param n_days: number of business days
    :param missing_prob: fraction of the days that are stale or interpolated
    :param split_ratio: ratio of the unadjusted split
    :param seed: seed of the random number generator
    :return: map from the name of the version to its price series
    """
    rng = np.random.default_rng(seed)
    returns = simulate_garch_returns(
        n_days=n_days, daily_vol=0.01, alpha=0.08, beta=0.90, dof=5.0, rng=rng
    )
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    clean = pd.Series(100.0 * np.cumprod(1.0 + returns), index=dates)
    is_missing = rng.random(n_days) < missing_prob
    # Keep the first and the last price, so that every gap can be filled.
    is_missing[0] = is_missing[-1] = False
    shuffled_returns = rng.permutation(returns)
    split = clean.copy()
    split.iloc[n_days // 2 :] /= split_ratio
    prices = {
        "clean": clean,
        "stale": clean.mask(is_missing).ffill(),
        "interpolated": clean.mask(is_missing).interpolate(method="linear"),
        "shuffled": pd.Series(
            100.0 * np.cumprod(1.0 + shuffled_returns), index=dates
        ),
        "unadjusted split": split,
    }
    return prices


def compute_stylized_facts(prices: pd.Series, *, max_lag: int) -> pd.Series:
    """
    Compute the statistics of a stylized-facts test suite.

    :param prices: price series of one asset
    :param max_lag: largest lag of the autocorrelation of absolute returns
    :return: series with the statistics
        - `excess_kurtosis_daily`: excess kurtosis of daily returns
        - `excess_kurtosis_monthly`: excess kurtosis of 21-day returns
        - `acf1_returns`: lag-1 autocorrelation of daily returns
        - `acf_abs_returns`: mean autocorrelation of absolute returns over
          lags 1 to `max_lag`
        - `zero_return_fraction`: fraction of daily returns equal to 0
    """
    returns = prices.pct_change(fill_method=None).dropna()
    abs_returns = returns.abs()
    acf_abs = np.mean(
        [abs_returns.autocorr(lag) for lag in range(1, max_lag + 1)]
    )
    # Non-overlapping monthly returns.
    month_id = np.arange(len(returns)) // _DAYS_PER_MONTH
    monthly = (1.0 + returns).groupby(month_id).prod() - 1.0
    facts = pd.Series(
        {
            "excess_kurtosis_daily": sstats.kurtosis(returns),
            "excess_kurtosis_monthly": sstats.kurtosis(monthly),
            "acf1_returns": returns.autocorr(1),
            "acf_abs_returns": acf_abs,
            "zero_return_fraction": float((returns == 0.0).mean()),
        }
    )
    return facts


def cell4_1_compute_stylized_facts_table(
    prices: Dict[str, pd.Series],
    *,
    max_lag: int,
) -> pd.DataFrame:
    """
    Compute the stylized-facts test suite on each version of a price series.

    :param prices: map from the name of the version to its price series
    :param max_lag: largest lag of the autocorrelation of absolute returns
    :return: dataframe with the statistics as rows and the versions as columns
    """
    facts_df = pd.DataFrame(
        {
            name: compute_stylized_facts(series, max_lag=max_lag)
            for name, series in prices.items()
        }
    )
    return facts_df


# #############################################################################
# Cell 5.1: Every look is a trial
# #############################################################################


def _column_correlations(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Compute the correlation of each column of `features` with `target`.

    :param features: array with shape (n_obs, n_features)
    :param target: array with shape (n_obs,)
    :return: array with shape (n_features,)
    """
    features = (features - features.mean(axis=0)) / features.std(axis=0)
    target = (target - target.mean()) / target.std()
    correlations = features.T @ target / len(target)
    return correlations


def simulate_snooping(
    *,
    n_in: int,
    n_out: int,
    looks: List[int],
    n_reps: int,
    seed: int,
) -> pd.DataFrame:
    """
    Simulate the IC of the best of many noise features.

    The target and the features are independent Gaussian noise, so every
    feature has a true IC of 0. Each look tests one more feature on the
    in-sample period and keeps the one with the largest absolute IC.

    :param n_in: number of in-sample days
    :param n_out: number of holdout days
    :param looks: numbers of looks to evaluate, in increasing order
    :param n_reps: number of repetitions of the simulation
    :param seed: seed of the random number generator
    :return: dataframe with one row per repetition and number of looks, and
        the columns `looks`, `rep`, `ic_in`, `ic_out`
    """
    rng = np.random.default_rng(seed)
    max_looks = max(looks)
    rows = []
    for rep in range(n_reps):
        target_in = rng.normal(size=n_in)
        target_out = rng.normal(size=n_out)
        features_in = rng.normal(size=(n_in, max_looks))
        features_out = rng.normal(size=(n_out, max_looks))
        ic_in = _column_correlations(features_in, target_in)
        ic_out = _column_correlations(features_out, target_out)
        for n_looks in looks:
            best = int(np.argmax(np.abs(ic_in[:n_looks])))
            # Trade the best feature with the sign found in-sample.
            sign = np.sign(ic_in[best])
            rows.append(
                {
                    "looks": n_looks,
                    "rep": rep,
                    "ic_in": abs(ic_in[best]),
                    "ic_out": sign * ic_out[best],
                }
            )
    snooping_df = pd.DataFrame(rows)
    return snooping_df


def cell5_1_plot_snooping(
    snooping_df: pd.DataFrame,
    *,
    n_in: int,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot the in-sample and holdout IC of the best feature against the looks.

    :param snooping_df: output of `simulate_snooping()`
    :param n_in: number of in-sample days, used for the 5% threshold
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with the mean IC and its 10%-90% band
    """
    grouped = snooping_df.groupby("looks")
    fig, axes = _new_figure(figsize)
    ax = axes[0, 0]
    curves = [
        ("ic_in", _BLUE, "best feature, in-sample"),
        ("ic_out", _CORAL, "same feature, holdout"),
    ]
    for column, color, label in curves:
        mean = grouped[column].mean()
        low = grouped[column].quantile(0.1)
        high = grouped[column].quantile(0.9)
        ax.plot(mean.index, mean, "o-", color=color, linewidth=2.0, label=label)
        ax.fill_between(mean.index, low, high, color=color, alpha=0.15)
    # Absolute IC that one test of one feature exceeds with probability 5%.
    threshold = 1.96 / np.sqrt(n_in)
    ax.axhline(threshold, color=_GRAY, linestyle="--", linewidth=1.0)
    ax.text(
        float(snooping_df["looks"].min()),
        threshold,
        "  single-test 5% threshold",
        va="bottom",
        color=_GRAY,
        fontsize=8,
    )
    ax.axhline(0.0, color=_GRAY, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("number of looks")
    ax.set_ylabel("IC of the selected feature")
    ax.legend(frameon=False, loc="upper left")
    return fig
