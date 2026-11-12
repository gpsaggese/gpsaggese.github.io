"""
Utility functions to generate the figures of Lesson 02.01 "Strategy
Backtesting".

Import as:

import L02_01_strategy_backtesting_utils as l0stbaut
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.figure as mfigure
import matplotlib.patches as mpatches
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


# Colors of the figures of Lesson 01.01, so that the figures of the two
# lessons look alike.
_BLUE = "#3C6FB0"
_CORAL = "#C0455B"
_ORANGE = "#D98E2B"
_GREEN = "#4F9A5C"
_GRAY = "#7B8794"
_LIGHT_GRAY = "#D5DBE1"

# Number of trading days in one year.
_DAYS_PER_YEAR = 252

# Euler-Mascheroni constant used by the false strategy theorem.
_EULER_GAMMA = 0.5772156649


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


def _annualized_sharpe(returns: np.ndarray, periods_per_year: int) -> float:
    """
    Compute the annualized Sharpe ratio of a series of returns.

    :param returns: returns of one period each
    :param periods_per_year: number of periods in one year
    :return: annualized Sharpe ratio
    """
    sharpe = np.mean(returns) / np.std(returns, ddof=1)
    sharpe = float(sharpe * np.sqrt(periods_per_year))
    return sharpe


def expected_max_sharpe(n_trials: np.ndarray, sharpe_std: float) -> np.ndarray:
    """
    Compute the expected maximum Sharpe ratio of zero-skill trials.

    The false strategy theorem approximates the expected maximum of
    `n_trials` independent Sharpe ratio estimates with true value 0 and
    standard deviation `sharpe_std`.

    :param n_trials: number of independent trials, each at least 1
    :param sharpe_std: standard deviation of the Sharpe ratio estimates
    :return: expected maximum Sharpe ratio for each number of trials
    """
    n_trials = np.asarray(n_trials, dtype=float)
    hdbg.dassert_lte(1.0, float(np.min(n_trials)))
    # With one trial, the expected value of the only estimate is 0.
    n_safe = np.maximum(n_trials, 2.0)
    z_1 = sstats.norm.ppf(1.0 - 1.0 / n_safe)
    z_2 = sstats.norm.ppf(1.0 - 1.0 / (n_safe * np.e))
    max_sharpe = sharpe_std * ((1.0 - _EULER_GAMMA) * z_1 + _EULER_GAMMA * z_2)
    max_sharpe = np.where(n_trials < 2.0, 0.0, max_sharpe)
    return max_sharpe


def deflated_sharpe(
    *,
    sharpe: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
    n_trials: np.ndarray,
) -> np.ndarray:
    """
    Compute the deflated Sharpe ratio of the best of `n_trials` trials.

    All the Sharpe ratios are per period (e.g., daily), not annualized. The
    benchmark Sharpe ratio is the expected maximum of zero-skill trials whose
    estimates have standard deviation `1 / sqrt(n_obs)`.

    :param sharpe: observed Sharpe ratio per period of the selected trial
    :param n_obs: number of return observations of the backtest
    :param skew: skewness of the returns
    :param kurtosis: kurtosis of the returns (3 for a Gaussian)
    :param n_trials: number of independent trials
    :return: probability that the true Sharpe ratio is above the benchmark
    """
    hdbg.dassert_lt(1, n_obs)
    benchmark = expected_max_sharpe(n_trials, 1.0 / np.sqrt(n_obs))
    # Standard error of the Sharpe ratio estimate with non-Gaussian returns.
    variance = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    z_score = (sharpe - benchmark) * np.sqrt(n_obs - 1) / np.sqrt(variance)
    dsr = sstats.norm.cdf(z_score)
    return dsr


# #############################################################################
# Cell 1.1: Transaction costs and net Sharpe ratio
# #############################################################################


def cell1_1_plot_cost_sensitivity(
    *,
    gross_return: float,
    volatility: float,
    turnovers: Dict[str, float],
    max_cost_bps: float,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot the net Sharpe ratio against the cost per unit traded.

    :param gross_return: annual return of the strategy before costs
    :param volatility: annual volatility of the strategy
    :param turnovers: map from a label to the annual turnover, in multiples of
        the book size
    :param max_cost_bps: largest cost per unit traded on the x axis, in basis
        points
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with one net Sharpe ratio curve per turnover
    """
    hdbg.dassert_lt(0.0, volatility)
    cost_bps = np.linspace(0.0, max_cost_bps, 201)
    colors = [_BLUE, _ORANGE, _CORAL, _GREEN]
    hdbg.dassert_lte(len(turnovers), len(colors))
    #
    fig, axes = _new_figure(figsize)
    ax = axes[0, 0]
    for color, (label, turnover) in zip(colors, turnovers.items()):
        # Costs scale with the notional traded in one year.
        annual_cost = turnover * cost_bps / 1e4
        net_sharpe = (gross_return - annual_cost) / volatility
        ax.plot(cost_bps, net_sharpe, color=color, linewidth=2.5, label=label)
        # Mark the cost per trade that erases the gross edge.
        breakeven_bps = gross_return / turnover * 1e4
        if breakeven_bps <= max_cost_bps:
            ax.plot(breakeven_bps, 0.0, "o", color=color, markersize=6)
            ax.annotate(
                f"{breakeven_bps:.0f} bps",
                (breakeven_bps, 0.0),
                textcoords="offset points",
                xytext=(4, 6),
                color=color,
            )
    ax.axhline(0.0, color=_GRAY, linewidth=1.0)
    ax.set_xlim(0.0, max_cost_bps)
    ax.set_ylim(-1.0, 1.05 * gross_return / volatility)
    ax.set_xlabel("cost per unit traded (bps)")
    ax.set_ylabel("net Sharpe ratio")
    ax.legend(
        frameon=False,
        ncol=len(turnovers),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
    )
    return fig


# #############################################################################
# Cell 1.2: Look-ahead bias
# #############################################################################


def cell1_2_plot_look_ahead_bias(
    *,
    n_days: int,
    volatility: float,
    window: int,
    seed: int,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot a trend rule on a random walk with and without a one-bar look-ahead.

    The rule is long when the log price is above its moving average and short
    otherwise. The correct backtest applies the position decided at the close
    of day t to the return of day t + 1. The biased backtest applies it to
    the return of day t, which the rule already used.

    :param n_days: number of simulated days
    :param volatility: daily volatility of the returns
    :param window: number of days of the moving average
    :param seed: seed of the random number generator
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with the cumulative PnL of the two backtests
    """
    hdbg.dassert_lt(window, n_days)
    # Random walk: the returns contain no signal at all.
    rng = np.random.default_rng(seed)
    returns = pd.Series(rng.normal(0.0, volatility, size=n_days))
    log_price = returns.cumsum()
    # The moving average includes the close of the current day.
    moving_average = log_price.rolling(window).mean()
    position = np.sign(log_price - moving_average)
    position = position.fillna(0.0)
    pnl_df = pd.DataFrame(
        {
            "correct": position.shift(1).fillna(0.0) * returns,
            "look-ahead": position * returns,
        }
    )
    #
    fig, axes = _new_figure(figsize)
    ax = axes[0, 0]
    colors = {"correct": _BLUE, "look-ahead": _CORAL}
    years = np.arange(n_days) / _DAYS_PER_YEAR
    for name, color in colors.items():
        sharpe = _annualized_sharpe(pnl_df[name].to_numpy(), _DAYS_PER_YEAR)
        label = f"{name}: SR = {sharpe:.1f}"
        ax.plot(years, 100.0 * pnl_df[name].cumsum(), color=color, label=label)
    ax.axhline(0.0, color=_GRAY, linewidth=1.0)
    ax.set_xlabel("years")
    ax.set_ylabel("cumulative log return (%)")
    ax.legend(frameon=False, loc="upper left")
    return fig


# #############################################################################
# Cell 1.3: Survivorship bias
# #############################################################################


def _simulate_universe(
    *,
    n_stocks: int,
    n_months: int,
    mean_return: float,
    volatility: float,
    delisting_price: float,
    delisting_return: float,
    seed: int,
) -> pd.DataFrame:
    """
    Simulate monthly simple returns of stocks that delist after a large fall.

    A stock delists in the first month its price falls below
    `delisting_price`, with an extra loss `delisting_return` in that month.
    The returns after the delisting are NaN.

    :param n_stocks: number of stocks in the universe
    :param n_months: number of months
    :param mean_return: mean monthly log return
    :param volatility: monthly volatility of the log returns
    :param delisting_price: price level, relative to the start, that triggers
        a delisting
    :param delisting_return: return of a stock in its delisting month, on top
        of the return of that month
    :param seed: seed of the random number generator
    :return: dataframe of simple returns, with months as rows and stocks as
        columns
    """
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(mean_return, volatility, size=(n_months, n_stocks))
    log_price = np.cumsum(log_returns, axis=0)
    below = log_price < np.log(delisting_price)
    # Month of the delisting, or `n_months` for a stock that survives.
    delist_month = np.where(below.any(axis=0), below.argmax(axis=0), n_months)
    returns = np.expm1(log_returns)
    months = np.arange(n_months)[:, None]
    is_delisting = months == delist_month[None, :]
    returns = np.where(
        is_delisting, (1.0 + returns) * (1.0 + delisting_return) - 1.0, returns
    )
    returns = np.where(months > delist_month[None, :], np.nan, returns)
    returns_df = pd.DataFrame(returns)
    return returns_df


def _backtest_buy_losers(returns_df: pd.DataFrame, quantile: float) -> pd.Series:
    """
    Backtest a long-only strategy that buys the worst stocks of the last month.

    :param returns_df: monthly simple returns, NaN for a stock not listed
    :param quantile: fraction of the listed stocks to buy each month
    :return: monthly returns of the equal-weighted portfolio of losers
    """
    strategy_returns = []
    for month in range(1, len(returns_df)):
        past = returns_df.iloc[month - 1].dropna()
        n_buy = max(1, int(quantile * len(past)))
        losers = past.nsmallest(n_buy).index
        strategy_returns.append(float(returns_df.iloc[month][losers].mean()))
    strategy_returns = pd.Series(strategy_returns)
    return strategy_returns


def cell1_3_plot_survivorship_bias(
    *,
    n_stocks: int,
    n_months: int,
    mean_return: float,
    volatility: float,
    delisting_price: float,
    delisting_return: float,
    quantile: float,
    seed: int,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot a buy-the-losers backtest on the full universe and on survivors.

    :param n_stocks: number of stocks in the universe
    :param n_months: number of months
    :param mean_return: mean monthly log return of every stock
    :param volatility: monthly volatility of the log returns
    :param delisting_price: price level, relative to the start, that triggers
        a delisting
    :param delisting_return: extra return of a stock in its delisting month
    :param quantile: fraction of the listed stocks to buy each month
    :param seed: seed of the random number generator
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with the equity curve of the two backtests
    """
    returns_df = _simulate_universe(
        n_stocks=n_stocks,
        n_months=n_months,
        mean_return=mean_return,
        volatility=volatility,
        delisting_price=delisting_price,
        delisting_return=delisting_return,
        seed=seed,
    )
    # A survivor-only database keeps the stocks listed at the end.
    survivors = returns_df.columns[returns_df.iloc[-1].notna()]
    universes = {
        "full universe": returns_df,
        "survivors only": returns_df[survivors],
    }
    colors = {"full universe": _BLUE, "survivors only": _CORAL}
    n_delisted = n_stocks - len(survivors)
    #
    fig, axes = _new_figure(figsize)
    ax = axes[0, 0]
    years = np.arange(1, n_months) / 12.0
    for name, universe_df in universes.items():
        strategy_returns = _backtest_buy_losers(universe_df, quantile)
        equity = (1.0 + strategy_returns).cumprod()
        annual_return = float(equity.iloc[-1] ** (12.0 / len(equity)) - 1.0)
        label = f"{name}: {annual_return:+.1%} per year"
        ax.plot(years, equity, color=colors[name], linewidth=2.0, label=label)
    ax.axhline(1.0, color=_GRAY, linewidth=1.0)
    ax.text(
        0.97,
        0.35,
        f"{n_delisted} of {n_stocks} stocks delist",
        transform=ax.transAxes,
        ha="right",
        color=_GRAY,
    )
    ax.set_xlabel("years")
    ax.set_ylabel("equity")
    ax.legend(frameon=False, loc="upper left")
    return fig


# #############################################################################
# Cell 2.1: In-sample vs. out-of-sample Sharpe ratio
# #############################################################################


def cell2_1_plot_is_vs_oos_sharpe(
    *,
    n_strategies: int,
    n_days_is: int,
    n_days_oos: int,
    volatility: float,
    seed: int,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot in-sample against out-of-sample Sharpe ratio of zero-skill strategies.

    :param n_strategies: number of strategies, all with true Sharpe ratio 0
    :param n_days_is: number of in-sample days
    :param n_days_oos: number of out-of-sample days
    :param volatility: daily volatility of the strategy returns
    :param seed: seed of the random number generator
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with the scatter plot and the selected strategy
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(
        0.0, volatility, size=(n_days_is + n_days_oos, n_strategies)
    )
    sharpe_df = pd.DataFrame(
        {
            "in-sample": [
                _annualized_sharpe(returns[:n_days_is, i], _DAYS_PER_YEAR)
                for i in range(n_strategies)
            ],
            "out-of-sample": [
                _annualized_sharpe(returns[n_days_is:, i], _DAYS_PER_YEAR)
                for i in range(n_strategies)
            ],
        }
    )
    best = int(np.argmax(sharpe_df["in-sample"].to_numpy()))
    best_is = float(sharpe_df.loc[best, "in-sample"])
    best_oos = float(sharpe_df.loc[best, "out-of-sample"])
    #
    fig, axes = _new_figure(figsize)
    ax = axes[0, 0]
    sns.scatterplot(
        data=sharpe_df,
        x="in-sample",
        y="out-of-sample",
        color=_BLUE,
        alpha=0.5,
        s=18,
        linewidth=0.0,
        ax=ax,
    )
    ax.plot(best_is, best_oos, "o", color=_CORAL, markersize=9)
    ax.annotate(
        f"selected\nin: {best_is:.2f}\nout: {best_oos:.2f}",
        (best_is, best_oos),
        textcoords="data",
        xytext=(best_is + 0.05, 0.65),
        ha="right",
        va="center",
        color=_CORAL,
        arrowprops={"arrowstyle": "-", "color": _CORAL},
    )
    ax.axhline(0.0, color=_GRAY, linewidth=1.0)
    ax.axvline(0.0, color=_GRAY, linewidth=1.0)
    ax.set_xlabel("in-sample Sharpe ratio")
    ax.set_ylabel("out-of-sample Sharpe ratio")
    return fig


# #############################################################################
# Cell 2.2: Expected maximum Sharpe ratio
# #############################################################################


def cell2_2_plot_expected_max_sharpe(
    *,
    n_years: float,
    trial_grid: List[int],
    n_sims: int,
    seed: int,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot the expected maximum Sharpe ratio of zero-skill trials.

    :param n_years: length of each backtest in years
    :param trial_grid: numbers of trials for the Monte Carlo estimate
    :param n_sims: number of Monte Carlo repetitions for each number of trials
    :param seed: seed of the random number generator
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with the theoretical curve and the Monte Carlo points
    """
    # An annualized Sharpe ratio estimate of a zero-skill strategy has standard
    # deviation of about `1 / sqrt(years)`.
    sharpe_std = 1.0 / np.sqrt(n_years)
    # The formula is smooth in the number of trials, so draw it on a fine grid.
    n_trials = np.logspace(np.log10(2.0), np.log10(max(trial_grid)), 200)
    theory = expected_max_sharpe(n_trials, sharpe_std)
    # Monte Carlo: maximum of `n` Gaussian Sharpe ratio estimates.
    rng = np.random.default_rng(seed)
    mc_mean = []
    for n in trial_grid:
        draws = rng.normal(0.0, sharpe_std, size=(n_sims, n))
        mc_mean.append(float(draws.max(axis=1).mean()))
    #
    fig, axes = _new_figure(figsize)
    ax = axes[0, 0]
    ax.plot(
        n_trials,
        theory,
        color=_BLUE,
        linewidth=2.5,
        label="false strategy theorem",
    )
    ax.plot(
        trial_grid,
        mc_mean,
        "o",
        color=_CORAL,
        markersize=6,
        label="Monte Carlo",
    )
    ax.set_xscale("log")
    ax.set_xlabel("number of trials $N$")
    ax.set_ylabel("expected maximum Sharpe ratio")
    ax.text(
        0.97,
        0.05,
        f"true Sharpe ratio = 0\nbacktest = {n_years:g} years",
        transform=ax.transAxes,
        ha="right",
        color=_GRAY,
    )
    ax.legend(frameon=False, loc="upper left")
    return fig


# #############################################################################
# Cell 2.3: Bonferroni and Benjamini-Hochberg
# #############################################################################


def cell2_3_plot_bh_procedure(
    *,
    n_null: int,
    n_true: int,
    true_t_mean: float,
    alpha: float,
    seed: int,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot sorted p-values against the Bonferroni and Benjamini-Hochberg lines.

    :param n_null: number of strategies with no skill
    :param n_true: number of strategies with skill
    :param true_t_mean: mean t-statistic of a strategy with skill
    :param alpha: target FWER for Bonferroni and target FDR for
        Benjamini-Hochberg
    :param seed: seed of the random number generator
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with the sorted p-values and the two thresholds
    """
    rng = np.random.default_rng(seed)
    # One-sided p-values: uniform under the null, small for true strategies.
    t_stat = np.concatenate(
        [
            rng.standard_normal(n_null),
            rng.normal(true_t_mean, 1.0, size=n_true),
        ]
    )
    is_true = np.concatenate([np.zeros(n_null), np.ones(n_true)]).astype(bool)
    p_value = sstats.norm.sf(t_stat)
    order = np.argsort(p_value)
    p_sorted = p_value[order]
    true_sorted = is_true[order]
    n_tests = n_null + n_true
    rank = np.arange(1, n_tests + 1)
    # Benjamini-Hochberg rejects the hypotheses up to the largest rank under
    # the sloped line.
    under_line = np.flatnonzero(p_sorted <= alpha * rank / n_tests)
    n_bh = int(under_line.max() + 1) if len(under_line) > 0 else 0
    n_bonferroni = int(np.sum(p_sorted <= alpha / n_tests))
    n_false_bh = int(np.sum(~true_sorted[:n_bh]))
    n_false_raw = int(np.sum(~true_sorted[p_sorted <= alpha]))
    n_raw = int(np.sum(p_sorted <= alpha))
    #
    fig, axes = _new_figure(figsize)
    ax = axes[0, 0]
    n_show = min(n_tests, 2 * max(n_true, n_raw))
    shown = rank <= n_show
    ax.scatter(
        rank[shown & true_sorted],
        p_sorted[shown & true_sorted],
        color=_GREEN,
        s=22,
        label="true signal",
        zorder=3,
    )
    ax.scatter(
        rank[shown & ~true_sorted],
        p_sorted[shown & ~true_sorted],
        color=_GRAY,
        s=22,
        label="no skill",
        zorder=3,
    )
    x_line = np.arange(1, n_show + 1)
    ax.plot(
        x_line,
        alpha * x_line / n_tests,
        color=_BLUE,
        label="BH: $\\alpha i / m$",
    )
    ax.axhline(
        alpha / n_tests,
        color=_CORAL,
        linestyle="--",
        label="Bonferroni: $\\alpha / m$",
    )
    ax.axhline(
        alpha, color=_ORANGE, linestyle=":", label="no correction: $\\alpha$"
    )
    ax.axvline(n_bh + 0.5, color=_BLUE, linewidth=0.8, linestyle="--")
    ax.set_yscale("log")
    ax.set_xlabel("rank $i$ of the sorted p-value")
    ax.set_ylabel("p-value")
    ax.text(
        0.97,
        0.27,
        f"m = {n_tests} strategies, {n_true} with skill\n"
        f"no correction: {n_raw} picks, {n_false_raw} false\n"
        f"Bonferroni: {n_bonferroni} picks\n"
        f"BH: {n_bh} picks, {n_false_bh} false",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize="small",
    )
    ax.legend(
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        fontsize="small",
    )
    return fig


# #############################################################################
# Cell 2.4: Purged and embargoed k-fold cross-validation
# #############################################################################


def _purged_kfold_labels(
    *, n_obs: int, n_folds: int, label_horizon: int, embargo: int
) -> np.ndarray:
    """
    Assign each observation of each fold to train, test, purge, or embargo.

    The label of observation `i` uses the returns of the observations from
    `i + 1` to `i + label_horizon`. A training observation is purged when its
    label window overlaps the label windows of the test block. The embargo
    then drops the next `embargo` observations after the purged ones.

    :param n_obs: number of observations
    :param n_folds: number of folds
    :param label_horizon: number of observations spanned by one label
    :param embargo: number of observations embargoed after the test block
    :return: array of shape (n_folds, n_obs) with 0 = train, 1 = test,
        2 = purged, 3 = embargoed
    """
    labels = np.zeros((n_folds, n_obs), dtype=int)
    bounds = np.linspace(0, n_obs, n_folds + 1).astype(int)
    for fold in range(n_folds):
        start, end = bounds[fold], bounds[fold + 1] - 1
        labels[fold, start : end + 1] = 1
        # Training labels before the test block reach into the test block.
        purge_before = range(max(0, start - label_horizon), start)
        # Training labels after the test block overlap the test labels.
        purge_after = range(end + 1, min(n_obs, end + 1 + label_horizon))
        for i in list(purge_before) + list(purge_after):
            labels[fold, i] = 2
        embargo_start = end + 1 + label_horizon
        for i in range(embargo_start, min(n_obs, embargo_start + embargo)):
            labels[fold, i] = 3
    return labels


def cell2_4_plot_purged_kfold(
    *,
    n_obs: int,
    n_folds: int,
    label_horizon: int,
    embargo: int,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot the train, test, purged, and embargoed observations of each fold.

    :param n_obs: number of observations
    :param n_folds: number of folds
    :param label_horizon: number of observations spanned by one label
    :param embargo: number of observations embargoed after the test block
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with one row of observations per fold
    """
    labels = _purged_kfold_labels(
        n_obs=n_obs,
        n_folds=n_folds,
        label_horizon=label_horizon,
        embargo=embargo,
    )
    colors = [_BLUE, _ORANGE, _LIGHT_GRAY, _CORAL]
    names = ["train", "test", "purged", "embargoed"]
    #
    fig, axes = _new_figure(figsize)
    ax = axes[0, 0]
    for fold in range(n_folds):
        for i in range(n_obs):
            ax.add_patch(
                mpatches.Rectangle(
                    (i, fold - 0.35),
                    0.9,
                    0.7,
                    color=colors[labels[fold, i]],
                    linewidth=0.0,
                )
            )
    ax.set_xlim(0, n_obs)
    ax.set_ylim(n_folds - 0.5, -0.5)
    ax.set_yticks(range(n_folds))
    ax.set_yticklabels([f"fold {k + 1}" for k in range(n_folds)])
    ax.set_xlabel("time (observation)")
    sns.despine(ax=ax, left=True)
    ax.tick_params(axis="y", length=0)
    handles = [mpatches.Patch(color=c, label=n) for c, n in zip(colors, names)]
    ax.legend(
        handles=handles,
        frameon=False,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
    )
    return fig


# #############################################################################
# Cell 2.5: Deflated Sharpe ratio
# #############################################################################


def cell2_5_plot_deflated_sharpe(
    *,
    annual_sharpes: List[float],
    n_years: float,
    skew: float,
    kurtosis: float,
    max_trials: int,
    threshold: float,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot the deflated Sharpe ratio against the number of trials.

    :param annual_sharpes: observed annualized Sharpe ratios of the selected
        strategy, one curve each
    :param n_years: length of the backtest in years of daily returns
    :param skew: skewness of the daily returns
    :param kurtosis: kurtosis of the daily returns (3 for a Gaussian)
    :param max_trials: largest number of trials on the x axis
    :param threshold: confidence level drawn as a horizontal line
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with one deflated Sharpe ratio curve per Sharpe ratio
    """
    n_obs = int(n_years * _DAYS_PER_YEAR)
    n_trials = np.unique(np.logspace(0, np.log10(max_trials), 300).round())
    colors = [_GREEN, _BLUE, _CORAL, _ORANGE]
    hdbg.dassert_lte(len(annual_sharpes), len(colors))
    #
    fig, axes = _new_figure(figsize)
    ax = axes[0, 0]
    for color, annual_sharpe in zip(colors, annual_sharpes):
        sharpe = annual_sharpe / np.sqrt(_DAYS_PER_YEAR)
        dsr = deflated_sharpe(
            sharpe=sharpe,
            n_obs=n_obs,
            skew=skew,
            kurtosis=kurtosis,
            n_trials=n_trials,
        )
        ax.plot(
            n_trials,
            dsr,
            color=color,
            linewidth=2.5,
            label=f"SR = {annual_sharpe:g}",
        )
        # Mark the number of trials where the strategy stops being significant.
        below = np.flatnonzero(dsr < threshold)
        if len(below) > 0:
            n_cross = n_trials[below[0]]
            ax.plot(n_cross, threshold, "o", color=color, markersize=6)
            ax.annotate(
                f"N = {n_cross:.0f}",
                (n_cross, threshold),
                textcoords="offset points",
                xytext=(4, -14),
                color=color,
            )
    ax.axhline(threshold, color=_GRAY, linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("number of trials $N$")
    ax.set_ylabel("deflated Sharpe ratio")
    ax.text(
        0.03,
        0.05,
        f"{n_years:g} years of daily returns\nskew = {skew:g}, "
        f"kurtosis = {kurtosis:g}",
        transform=ax.transAxes,
        color=_GRAY,
    )
    ax.legend(frameon=False, loc="lower right")
    return fig
