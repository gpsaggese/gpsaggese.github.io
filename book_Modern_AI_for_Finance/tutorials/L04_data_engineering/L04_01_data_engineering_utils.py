"""
Utility functions to generate the figures of Lesson 04.01 "Data Engineering".

Import as:

import L04_01_data_engineering_utils as l0daendut
"""

import logging
from typing import Any, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.figure as mfigure
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
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

# Number of trading days in one year and in one quarter.
_DAYS_PER_YEAR = 252
_DAYS_PER_QUARTER = 63


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
# Running example: fundamentals with release dates and restatements
# #############################################################################


def simulate_fundamentals(
    *,
    n_stocks: int,
    n_quarters: int,
    first_quarter_end: str,
    persistence: float,
    restated_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """
    Simulate the quarterly fundamentals of a panel of stocks.

    - Each stock-quarter has a persistent value component `v` (AR(1) with unit
      variance) and an earnings surprise `surprise` (unit variance)
    - The reported signal is `signal_first = v + surprise`
    - The release date `rdq` is the quarter end plus a lag of 25 days plus a
      gamma noise, and 15 more days for the fourth quarter (annual report)
    - A fraction of the records is restated some time after the release, with
      a negative revision `delta`

    :param n_stocks: number of stocks
    :param n_quarters: number of quarters
    :param first_quarter_end: first fiscal quarter end, e.g., `2005-03-31`
    :param persistence: quarterly AR(1) coefficient of `v`
    :param restated_fraction: fraction of the records that are restated
    :param seed: seed of the random number generator
    :return: one row per stock and quarter with the columns `stock`,
        `datadate`, `rdq`, `lag_days`, `v`, `surprise`, `signal_first`,
        `restate_date`, `delta`, and `signal_restated`
    """
    hdbg.dassert_lt(persistence, 1.0)
    hdbg.dassert_is_proportion(restated_fraction)
    rng = np.random.default_rng(seed)
    quarter_ends = pd.date_range(
        first_quarter_end, periods=n_quarters, freq="QE"
    )
    # Simulate the persistent value component.
    v = np.empty((n_stocks, n_quarters))
    v[:, 0] = rng.standard_normal(n_stocks)
    innovation_scale = np.sqrt(1.0 - persistence**2)
    for q in range(1, n_quarters):
        v[:, q] = persistence * v[
            :, q - 1
        ] + innovation_scale * rng.standard_normal(n_stocks)
    surprise = rng.standard_normal((n_stocks, n_quarters))
    # Simulate the release lag in days.
    lag = 25.0 + rng.gamma(3.0, 10.0, size=(n_stocks, n_quarters))
    is_fourth_quarter = quarter_ends.to_series().dt.month.to_numpy() == 12
    lag = lag + 15.0 * is_fourth_quarter[None, :]
    lag_days = np.round(lag).astype(int)
    datadate = np.tile(quarter_ends.values, (n_stocks, 1))
    rdq = datadate + lag_days.astype("timedelta64[D]")
    # Simulate the restatements, which come 20-80 days after the release.
    is_restated = rng.random((n_stocks, n_quarters)) < restated_fraction
    restate_lag = rng.integers(20, 80, size=(n_stocks, n_quarters))
    restate_date = rdq + restate_lag.astype("timedelta64[D]")
    restate_date = np.where(is_restated, restate_date, np.datetime64("NaT"))
    delta = -np.abs(rng.normal(1.5, 0.5, size=(n_stocks, n_quarters)))
    delta = np.where(is_restated, delta, 0.0)
    signal_first = v + surprise
    fund_df = pd.DataFrame(
        {
            "stock": np.repeat(
                [f"S{i:04d}" for i in range(n_stocks)], n_quarters
            ),
            "datadate": datadate.ravel(),
            "rdq": rdq.ravel(),
            "lag_days": lag_days.ravel(),
            "v": v.ravel(),
            "surprise": surprise.ravel(),
            "signal_first": signal_first.ravel(),
            "restate_date": restate_date.ravel(),
            "delta": delta.ravel(),
            "signal_restated": (signal_first + delta).ravel(),
        }
    )
    return fund_df


def make_month_ends(
    fund_df: pd.DataFrame, *, n_extra_months: int
) -> pd.DatetimeIndex:
    """
    Make the month ends at which a strategy decides.

    :param fund_df: fundamentals from `simulate_fundamentals()`
    :param n_extra_months: months after the last quarter end, so that late
        releases and restatements are inside the sample
    :return: month ends from the first to the last quarter end plus the extra
        months
    """
    first = fund_df["datadate"].min()
    last = fund_df["datadate"].max() + pd.DateOffset(months=n_extra_months)
    month_ends = pd.date_range(first, last, freq="ME")
    return month_ends


def _ceil_month_rows(
    dates: np.ndarray, month_ends: pd.DatetimeIndex
) -> np.ndarray:
    """
    Map each date to the row of the first month end on or after it.

    :param dates: array of `datetime64` values, `NaT` allowed
    :param month_ends: month ends
    :return: array of row numbers, `len(month_ends)` for a `NaT` or a date
        after the last month end
    """
    rows = np.searchsorted(month_ends.values, dates, side="left")
    rows = np.where(np.isnat(dates), len(month_ends), rows)
    return rows


def _pivot_by_quarter(fund_df: pd.DataFrame, column: str) -> np.ndarray:
    """
    Reshape a column of the long fundamentals into a stocks x quarters array.

    :param fund_df: fundamentals from `simulate_fundamentals()`
    :param column: column to reshape
    :return: array with one row per stock and one column per quarter
    """
    pivot_df = fund_df.pivot(index="stock", columns="datadate", values=column)
    return pivot_df.to_numpy()


def _build_signal_matrix(
    avail_rows: np.ndarray,
    values: np.ndarray,
    restate_rows: np.ndarray,
    restate_values: np.ndarray,
    *,
    n_months: int,
) -> np.ndarray:
    """
    Build the signal of each stock as known at each month end.

    - The signal of a stock is the value of its latest quarter that is
      available at the month end
    - A restatement replaces the value of the latest quarter, if it arrives
      before the next release

    :param avail_rows: row of the month end at which each quarter becomes
        available, with shape (n_stocks, n_quarters)
    :param values: value of each quarter once available
    :param restate_rows: row of the month end at which each quarter is
        restated, `n_months` for no restatement
    :param restate_values: value of each quarter after the restatement
    :param n_months: number of month ends
    :return: signal with shape (n_months, n_stocks), NaN before the first
        release
    """
    n_stocks, n_quarters = avail_rows.shape
    signal = np.full((n_months, n_stocks), np.nan)
    # Releases of a stock are ordered in time, even when one lag is long.
    avail_rows = np.maximum.accumulate(avail_rows, axis=1)
    for i in range(n_stocks):
        for q in range(n_quarters):
            start = avail_rows[i, q]
            if start >= n_months:
                break
            stop = n_months
            if q + 1 < n_quarters:
                stop = min(avail_rows[i, q + 1], n_months)
            signal[start:stop, i] = values[i, q]
            if restate_rows[i, q] < stop:
                first_row = max(restate_rows[i, q], start)
                signal[first_row:stop, i] = restate_values[i, q]
    return signal


def build_signal(
    fund_df: pd.DataFrame,
    month_ends: pd.DatetimeIndex,
    *,
    mode: str,
    assumed_lag_days: int,
) -> pd.DataFrame:
    """
    Build the signal as known at each month end under one alignment rule.

    The rules are:
    - `period_end_lag`: the value is available `assumed_lag_days` after the
      quarter end, as in a join on the period end with a fixed lag
    - `first`: the value is available on the release date, first reported
      value only
    - `pit`: point in time, the first value on the release date, and the
      restated value on the restatement date
    - `latest`: the restated value from the release date on, as a database
      that keeps only the latest vintage

    :param fund_df: fundamentals from `simulate_fundamentals()`
    :param month_ends: month ends at which the strategy decides
    :param mode: alignment rule
    :param assumed_lag_days: lag after the quarter end, used by
        `period_end_lag` only
    :return: signal with month ends as rows and stocks as columns
    """
    hdbg.dassert_in(mode, ["period_end_lag", "first", "pit", "latest"])
    n_months = len(month_ends)
    datadate = _pivot_by_quarter(fund_df, "datadate").astype("datetime64[ns]")
    rdq = _pivot_by_quarter(fund_df, "rdq").astype("datetime64[ns]")
    restate_date = _pivot_by_quarter(fund_df, "restate_date").astype(
        "datetime64[ns]"
    )
    signal_first = _pivot_by_quarter(fund_df, "signal_first").astype(float)
    signal_restated = _pivot_by_quarter(fund_df, "signal_restated").astype(float)
    # By default there is no restatement.
    restate_rows = np.full(rdq.shape, n_months)
    restate_values = signal_restated
    if mode == "period_end_lag":
        lag = np.timedelta64(assumed_lag_days, "D")
        avail_rows = _ceil_month_rows(datadate + lag, month_ends)
        values = signal_first
    elif mode == "first":
        avail_rows = _ceil_month_rows(rdq, month_ends)
        values = signal_first
    elif mode == "pit":
        avail_rows = _ceil_month_rows(rdq, month_ends)
        values = signal_first
        restate_rows = _ceil_month_rows(restate_date, month_ends)
    else:
        avail_rows = _ceil_month_rows(rdq, month_ends)
        values = signal_restated
    signal = _build_signal_matrix(
        avail_rows,
        values,
        restate_rows,
        restate_values,
        n_months=n_months,
    )
    stocks = sorted(fund_df["stock"].unique())
    signal_df = pd.DataFrame(signal, index=month_ends, columns=pd.Index(stocks))
    return signal_df


def simulate_monthly_returns(
    fund_df: pd.DataFrame,
    month_ends: pd.DatetimeIndex,
    *,
    monthly_drift: float,
    monthly_vol: float,
    dof: float,
    value_beta: float,
    surprise_beta: float,
    restate_beta: float,
    seed: int,
) -> pd.DataFrame:
    """
    Simulate monthly returns that react to the fundamentals when released.

    - The return of a month is a drift, plus `value_beta` times the value
      component of the latest released quarter, plus Student-t noise
    - The month with the release date also has `surprise_beta` times the
      earnings surprise, since the price moves when the news is public
    - The month with a restatement date also has `restate_beta` times the
      revision

    :param fund_df: fundamentals from `simulate_fundamentals()`
    :param month_ends: month ends of the returns
    :param monthly_drift: mean monthly return
    :param monthly_vol: standard deviation of the noise
    :param dof: degrees of freedom of the Student-t noise, above 2
    :param value_beta: monthly return per unit of the value component
    :param surprise_beta: monthly return per unit of the earnings surprise
    :param restate_beta: monthly return per unit of the revision
    :param seed: seed of the random number generator
    :return: returns with month ends as rows and stocks as columns
    """
    hdbg.dassert_lt(2.0, dof)
    rng = np.random.default_rng(seed)
    n_months = len(month_ends)
    stocks = sorted(fund_df["stock"].unique())
    n_stocks = len(stocks)
    rdq = _pivot_by_quarter(fund_df, "rdq").astype("datetime64[ns]")
    restate_date = _pivot_by_quarter(fund_df, "restate_date").astype(
        "datetime64[ns]"
    )
    v = _pivot_by_quarter(fund_df, "v").astype(float)
    surprise = _pivot_by_quarter(fund_df, "surprise").astype(float)
    delta = _pivot_by_quarter(fund_df, "delta").astype(float)
    release_rows = _ceil_month_rows(rdq, month_ends)
    restate_rows = _ceil_month_rows(restate_date, month_ends)
    # The value component is known from the release month on, and it drives
    # the returns of the following months.
    no_restate_rows = np.full(rdq.shape, n_months)
    v_known = _build_signal_matrix(
        release_rows, v, no_restate_rows, v, n_months=n_months
    )
    v_prev = np.vstack([np.full((1, n_stocks), np.nan), v_known[:-1]])
    v_prev = np.nan_to_num(v_prev, nan=0.0)
    # Add the price reaction to the news in the month of the release and of the
    # restatement.
    reaction = np.zeros((n_months, n_stocks))
    stock_idx = np.broadcast_to(np.arange(n_stocks)[:, None], rdq.shape)
    is_in = release_rows < n_months
    np.add.at(
        reaction,
        (release_rows[is_in], stock_idx[is_in]),
        surprise_beta * surprise[is_in],
    )
    is_in = restate_rows < n_months
    np.add.at(
        reaction,
        (restate_rows[is_in], stock_idx[is_in]),
        restate_beta * delta[is_in],
    )
    shocks = rng.standard_t(dof, size=(n_months, n_stocks))
    shocks = shocks * np.sqrt((dof - 2.0) / dof)
    returns = (
        monthly_drift + value_beta * v_prev + reaction + monthly_vol * shocks
    )
    returns_df = pd.DataFrame(
        returns, index=month_ends, columns=pd.Index(stocks)
    )
    return returns_df


def compute_monthly_ic(
    signal_df: pd.DataFrame, returns_df: pd.DataFrame, *, min_stocks: int
) -> pd.Series:
    """
    Compute the rank information coefficient (IC) month by month.

    The IC of month `t` is the cross-sectional Spearman correlation between
    the signal at the end of month `t` and the return of month `t + 1`.

    :param signal_df: signal as known at each month end
    :param returns_df: returns of each month
    :param min_stocks: minimum number of stocks with a signal to report a month
    :return: IC of each month, indexed by the month end of the signal
    """
    hdbg.dassert_eq(signal_df.shape, returns_df.shape)
    signal = signal_df.iloc[:-1]
    forward = returns_df.iloc[1:].set_axis(signal.index, axis=0)
    # Keep the stocks with both a signal and a return.
    is_valid = signal.notna() & forward.notna()
    signal = signal.where(is_valid)
    forward = forward.where(is_valid)
    ic_s = signal.rank(axis=1).corrwith(forward.rank(axis=1), axis=1)
    ic_s = ic_s[is_valid.sum(axis=1) >= min_stocks]
    return ic_s


# #############################################################################
# Cell 1.2: As-of join on a small example
# #############################################################################


def build_asof_example() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a stock with three quarters of fundamentals and monthly decisions.

    :return: fundamentals with `datadate` (period end), `rdq` (release date),
        and `earnings_yield`, and decision dates at each month end
    """
    fund_df = pd.DataFrame(
        {
            "stock": "AAA",
            "datadate": pd.to_datetime(
                ["2023-12-31", "2024-03-31", "2024-06-30"]
            ),
            "rdq": pd.to_datetime(["2024-02-22", "2024-05-02", "2024-08-01"]),
            "earnings_yield": [4.1, 4.6, 5.2],
        }
    )
    decision_df = pd.DataFrame(
        {
            "stock": "AAA",
            "decision_date": pd.date_range(
                "2024-01-31", "2024-08-31", freq="ME"
            ),
        }
    )
    return fund_df, decision_df


# #############################################################################
# Cell 1.3: Release lag and look-ahead
# #############################################################################


def compute_ic_by_assumed_lag(
    fund_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    *,
    lags: List[int],
    min_stocks: int,
) -> pd.Series:
    """
    Compute the mean IC when the value is available a fixed lag after the
    quarter end.

    :param fund_df: fundamentals from `simulate_fundamentals()`
    :param returns_df: returns from `simulate_monthly_returns()`
    :param lags: assumed lags in days after the quarter end
    :param min_stocks: minimum number of stocks with a signal to report a month
    :return: mean IC indexed by the assumed lag
    """
    month_ends = pd.DatetimeIndex(returns_df.index)
    mean_ic = {}
    for lag in lags:
        signal_df = build_signal(
            fund_df, month_ends, mode="period_end_lag", assumed_lag_days=lag
        )
        ic_s = compute_monthly_ic(signal_df, returns_df, min_stocks=min_stocks)
        mean_ic[lag] = ic_s.mean()
    mean_ic_s = pd.Series(mean_ic, name="mean_ic")
    mean_ic_s.index.name = "assumed_lag_days"
    return mean_ic_s


def cell1_3_plot_release_lag(
    fund_df: pd.DataFrame,
    ic_by_lag_s: pd.Series,
    ic_asof: float,
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot the distribution of release lags and the IC against the assumed lag.

    :param fund_df: fundamentals from `simulate_fundamentals()`
    :param ic_by_lag_s: mean IC indexed by the assumed lag in days
    :param ic_asof: mean IC when the value is available on the true release
        date
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with the histogram of lags and the IC curve
    """
    fig, axes = _new_figure(figsize, nrows=2)
    # Plot the release lags.
    ax = axes[0, 0]
    lag_days = fund_df["lag_days"]
    ax.hist(
        lag_days,
        bins=np.arange(0, lag_days.max() + 5, 5),
        color=_BLUE,
        alpha=0.85,
    )
    ax.axvline(lag_days.mean(), color=_CORAL, linestyle="--", linewidth=1.2)
    ax.text(
        lag_days.mean() + 2,
        ax.get_ylim()[1] * 0.85,
        f"mean {lag_days.mean():.0f} days",
        color=_CORAL,
        fontsize=8,
    )
    ax.set_xlabel("days from quarter end to release")
    ax.set_ylabel("records")
    # Plot the IC against the assumed lag.
    ax = axes[1, 0]
    ax.plot(
        ic_by_lag_s.index,
        ic_by_lag_s.values,
        "o-",
        color=_BLUE,
        linewidth=1.6,
        markersize=4,
        label="fixed lag after quarter end",
    )
    ax.axhline(
        ic_asof,
        color=_GREEN,
        linestyle="--",
        linewidth=1.4,
        label="true release date (as-of)",
    )
    ax.set_xlabel("assumed lag after quarter end (days)")
    ax.set_ylabel("mean IC")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


# #############################################################################
# Cell 1.4: Restatements and vintages
# #############################################################################


def compute_ic_by_vintage_rule(
    fund_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    *,
    min_stocks: int,
) -> pd.Series:
    """
    Compute the mean IC of the signal under each rule to pick a vintage.

    The rules are:
    - `latest vintage`: restated values from the release date on, as a
      database that overwrites the old values
    - `point in time`: first value on the release date and restated value on
      the restatement date
    - `first reported`: first value only, restatements ignored

    :param fund_df: fundamentals from `simulate_fundamentals()`
    :param returns_df: returns from `simulate_monthly_returns()`
    :param min_stocks: minimum number of stocks with a signal to report a month
    :return: mean IC indexed by the rule
    """
    month_ends = pd.DatetimeIndex(returns_df.index)
    mean_ic = {}
    for label, mode in [
        ("latest vintage", "latest"),
        ("point in time", "pit"),
        ("first reported", "first"),
    ]:
        signal_df = build_signal(
            fund_df, month_ends, mode=mode, assumed_lag_days=0
        )
        ic_s = compute_monthly_ic(signal_df, returns_df, min_stocks=min_stocks)
        mean_ic[label] = ic_s.mean()
    mean_ic_s = pd.Series(mean_ic, name="mean_ic")
    return mean_ic_s


def pick_restated_record(fund_df: pd.DataFrame) -> pd.Series:
    """
    Pick a restated record with a typical release lag, to plot its vintages.

    :param fund_df: fundamentals from `simulate_fundamentals()`
    :return: the row of the record
    """
    restated_df = fund_df.loc[fund_df["restate_date"].notna()]
    median_lag = restated_df["lag_days"].median()
    # Take the record closest to the median lag and to the median revision.
    distance = (restated_df["lag_days"] - median_lag).abs() + (
        restated_df["delta"] - restated_df["delta"].median()
    ).abs()
    record = restated_df.loc[distance.idxmin()]
    return record


def cell1_4_plot_restatements(
    record: pd.Series,
    ic_by_vintage_s: pd.Series,
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot the vintages of one restated record and the IC of each vintage rule.

    :param record: restated record from `pick_restated_record()`
    :param ic_by_vintage_s: mean IC indexed by the vintage rule
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with the vintages of one record and the IC bars
    """
    fig, axes = _new_figure(
        figsize, nrows=2, gridspec_kw={"height_ratios": [1.2, 1.0]}
    )
    # Plot what was known when, against what a database shows today.
    ax = axes[0, 0]
    datadate = record["datadate"]
    rdq = record["rdq"]
    restate_date = record["restate_date"]
    end = restate_date + pd.Timedelta(days=45)
    ax.hlines(
        record["signal_restated"],
        datadate,
        end,
        color=_CORAL,
        linestyle="--",
        linewidth=1.6,
        label="latest vintage (database today)",
    )
    ax.hlines(
        record["signal_first"],
        rdq,
        restate_date,
        color=_BLUE,
        linewidth=2.4,
        label="known then",
    )
    ax.hlines(
        record["signal_restated"], restate_date, end, color=_BLUE, linewidth=2.4
    )
    # Leave room above the curves for the legend, and left of the first date for
    # its label.
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min - 0.1, y_max + 0.9)
    ax.set_xlim(datadate - pd.Timedelta(days=12), end)
    y_text = 0.5 * (y_min + y_max)
    for date, label in [
        (datadate, "quarter end"),
        (rdq, "release"),
        (restate_date, "restatement"),
    ]:
        ax.axvline(date, color=_GRAY, linestyle=":", linewidth=1.0)
        ax.text(
            date,
            y_text,
            f"{label} ",
            rotation=90,
            va="center",
            ha="right",
            fontsize=8,
            color=_GRAY,
        )
    ax.set_ylabel("reported value")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    # Plot the IC of each vintage rule.
    ax = axes[1, 0]
    colors = [_CORAL, _BLUE, _GRAY]
    ax.bar(
        ic_by_vintage_s.index,
        ic_by_vintage_s.values,
        color=colors[: len(ic_by_vintage_s)],
        width=0.6,
    )
    for i, value in enumerate(ic_by_vintage_s.values):
        ax.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("mean IC")
    fig.tight_layout()
    return fig


# #############################################################################
# Cell 1.5: A guard that fails loudly
# #############################################################################


def count_future_records(
    df: pd.DataFrame, *, decision_col: str, knowledge_col: str
) -> int:
    """
    Count the rows with a knowledge time after the decision time.

    :param df: rows joined to a decision
    :param decision_col: name of the decision time column
    :param knowledge_col: name of the knowledge time column
    :return: number of rows that use information from the future
    """
    n_future = int((df[knowledge_col] > df[decision_col]).sum())
    return n_future


def assert_point_in_time(
    df: pd.DataFrame, *, decision_col: str, knowledge_col: str
) -> None:
    """
    Fail if any row uses information that was not yet known at the decision.

    :param df: rows joined to a decision
    :param decision_col: name of the decision time column
    :param knowledge_col: name of the knowledge time column
    """
    n_future = count_future_records(
        df, decision_col=decision_col, knowledge_col=knowledge_col
    )
    hdbg.dassert_eq(
        n_future,
        0,
        msg=f"'{knowledge_col}' is after '{decision_col}' in {n_future} rows",
    )


# #############################################################################
# Cell 2.1: Delisting returns
# #############################################################################


def simulate_delisting_panel(
    *,
    n_stocks: int,
    n_months: int,
    base_hazard: float,
    hazard_slope: float,
    premium_slope: float,
    mean_delisting_return: float,
    delisting_vol: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Simulate monthly returns of stocks that delist for performance reasons.

    - Each stock has a static cheapness score `c` (high is cheap, as in a
      value ranking)
    - Before delisting, the monthly return is a drift, plus `premium_slope`
      times `c`, plus Student-t noise
    - The monthly probability of delisting is `base_hazard * exp(hazard_slope
      * c)`, so cheap stocks fail more often
    - The return of the delisting month is normal with mean
      `mean_delisting_return`, clipped to [-100%, +50%]

    :param n_stocks: number of stocks
    :param n_months: number of months
    :param base_hazard: monthly probability of delisting for `c = 0`
    :param hazard_slope: sensitivity of the log-hazard to cheapness
    :param premium_slope: monthly return per unit of cheapness
    :param mean_delisting_return: mean return of the delisting month
    :param delisting_vol: standard deviation of the delisting return
    :param seed: seed of the random number generator
    :return: returns with the true delisting returns (months x stocks, NaN
        after the delisting), flags of the delisting months, and cheapness
        scores
    """
    rng = np.random.default_rng(seed)
    cheapness = rng.standard_normal(n_stocks)
    hazard = np.minimum(base_hazard * np.exp(hazard_slope * cheapness), 0.5)
    dof = 4.0
    normal_scale = 0.09 * np.sqrt((dof - 2.0) / dof)
    returns = np.full((n_months, n_stocks), np.nan)
    is_delisting = np.zeros((n_months, n_stocks), dtype=bool)
    is_alive = np.ones(n_stocks, dtype=bool)
    for month in range(n_months):
        normal_return = (
            0.007
            + premium_slope * cheapness
            + normal_scale * rng.standard_t(dof, size=n_stocks)
        )
        delisting_return = np.clip(
            rng.normal(mean_delisting_return, delisting_vol, size=n_stocks),
            -1.0,
            0.5,
        )
        delists_now = is_alive & (rng.random(n_stocks) < hazard)
        month_return = np.where(delists_now, delisting_return, normal_return)
        returns[month] = np.where(is_alive, month_return, np.nan)
        is_delisting[month] = delists_now
        is_alive = is_alive & ~delists_now
    months = pd.date_range("2000-01-31", periods=n_months, freq="ME")
    stocks = pd.Index([f"S{i:04d}" for i in range(n_stocks)])
    returns_df = pd.DataFrame(returns, index=months, columns=stocks)
    is_delisting_df = pd.DataFrame(is_delisting, index=months, columns=stocks)
    cheapness_s = pd.Series(cheapness, index=stocks, name="cheapness")
    return returns_df, is_delisting_df, cheapness_s


def compute_decile_returns(
    returns_df: pd.DataFrame,
    is_delisting_df: pd.DataFrame,
    cheapness_s: pd.Series,
    *,
    delisting_rule: str,
    imputed_return: float,
) -> pd.Series:
    """
    Compute the annualized return of the cheapness deciles.

    The rules for the return of the delisting month are:
    - `true`: the return that the stock earned
    - `missing`: the return is missing, so the stock drops out of that month
    - `imputed`: the return is replaced by `imputed_return`

    :param returns_df: returns with the true delisting returns
    :param is_delisting_df: flags of the delisting months
    :param cheapness_s: cheapness scores, high is cheap
    :param delisting_rule: rule for the delisting month
    :param imputed_return: return used by the `imputed` rule
    :return: mean annual return of each decile, from 1 (most expensive) to 10
        (cheapest)
    """
    hdbg.dassert_in(delisting_rule, ["true", "missing", "imputed"])
    returns_df = returns_df.copy()
    if delisting_rule == "missing":
        returns_df = returns_df.mask(is_delisting_df)
    elif delisting_rule == "imputed":
        returns_df = returns_df.mask(is_delisting_df, imputed_return)
    deciles = pd.Series(pd.qcut(cheapness_s, 10, labels=False)) + 1
    deciles.index = cheapness_s.index
    # Average the returns of the stocks in each decile each month, then
    # average the months.
    monthly_df = returns_df.T.groupby(deciles).mean().T
    annual_s = pd.Series(12.0 * monthly_df.mean(), name="annual_return")
    annual_s.index.name = "decile"
    return annual_s


def cell2_1_plot_delisting_returns(
    delisting_returns: np.ndarray,
    decile_df: pd.DataFrame,
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot the delisting returns and the decile returns under each rule.

    :param delisting_returns: returns of the delisting months
    :param decile_df: annual return of each cheapness decile (rows) under the
        rules `true`, `missing`, and `imputed` (columns)
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with the histogram of delisting returns and the decile
        returns
    """
    fig, axes = _new_figure(
        figsize, nrows=2, gridspec_kw={"height_ratios": [0.8, 1.2]}
    )
    # Plot the delisting returns.
    ax = axes[0, 0]
    ax.hist(
        100.0 * delisting_returns,
        bins=np.arange(-100, 55, 5),
        color=_BLUE,
        alpha=0.85,
    )
    mean_return = 100.0 * delisting_returns.mean()
    ax.axvline(mean_return, color=_CORAL, linestyle="--", linewidth=1.2)
    ax.text(
        mean_return + 3,
        ax.get_ylim()[1] * 0.85,
        f"mean {mean_return:.0f}%",
        color=_CORAL,
        fontsize=8,
    )
    ax.set_xlabel("return of the delisting month (%)")
    ax.set_ylabel("delistings")
    # Plot the decile returns under each rule.
    ax = axes[1, 0]
    styles = {
        "true": (_BLUE, "-", "true delisting return"),
        "missing": (_CORAL, "-", "delisting return missing"),
        "imputed": (_GREEN, "--", "imputed -30%"),
    }
    for column, (color, linestyle, label) in styles.items():
        spread = 100.0 * (decile_df[column].iloc[-1] - decile_df[column].iloc[0])
        ax.plot(
            decile_df.index,
            100.0 * decile_df[column],
            marker="o",
            markersize=3.5,
            color=color,
            linestyle=linestyle,
            linewidth=1.6,
            label=f"{label}: spread {spread:.1f}%",
        )
    ax.set_xlabel("cheapness decile (10 = cheapest)")
    ax.set_ylabel("annual return (%)")
    ax.set_xticks(decile_df.index)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


# #############################################################################
# Cell 3.1: Adjusted prices and total returns
# #############################################################################


def simulate_dividend_payer(
    *,
    n_years: int,
    start_price: float,
    annual_drift: float,
    annual_vol: float,
    quarterly_yield: float,
    split_price: float,
    split_ratio: int,
    seed: int,
) -> pd.DataFrame:
    """
    Simulate the raw daily prices of a stock with dividends and splits.

    - The total return follows a geometric random walk
    - Every quarter the stock goes ex-dividend: the price falls by
      `quarterly_yield` times the previous close
    - When the close exceeds `split_price` the stock splits, and the raw price
      is divided by `split_ratio`

    :param n_years: number of years
    :param start_price: first raw close
    :param annual_drift: annual drift of the total return
    :param annual_vol: annual volatility of the total return
    :param quarterly_yield: dividend per share over the previous close
    :param split_price: raw close above which the stock splits
    :param split_ratio: number of shares after the split for each old share
    :param seed: seed of the random number generator
    :return: one row per business day with the raw `close`, the `dividend`
        per share paid on the day, and the `split_ratio` of the day (1 for no
        split)
    """
    rng = np.random.default_rng(seed)
    n_days = n_years * _DAYS_PER_YEAR
    dates = pd.bdate_range("2005-01-03", periods=n_days)
    daily_vol = annual_vol / np.sqrt(_DAYS_PER_YEAR)
    log_returns = rng.normal(
        annual_drift / _DAYS_PER_YEAR - 0.5 * daily_vol**2,
        daily_vol,
        size=n_days,
    )
    close = np.empty(n_days)
    dividend = np.zeros(n_days)
    ratio = np.ones(n_days, dtype=int)
    price = start_price
    for day in range(n_days):
        cum_dividend_price = price * np.exp(log_returns[day])
        # Pay the dividend on the ex-date, every quarter.
        if day % _DAYS_PER_QUARTER == _DAYS_PER_QUARTER - 1:
            dividend[day] = quarterly_yield * price
        price_today = cum_dividend_price - dividend[day]
        # Split the stock when the price is high.
        if price_today > split_price:
            price_today = price_today / split_ratio
            ratio[day] = split_ratio
        close[day] = price_today
        price = price_today
    price_df = pd.DataFrame(
        {"close": close, "dividend": dividend, "split_ratio": ratio},
        index=dates,
    )
    return price_df


def adjust_prices(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust the raw prices for splits and dividends.

    - An event on date `T` multiplies the prices before `T` by `1 / ratio` for
      a split, and by `(P - D) / P` for a dividend `D`, where `P` is the close
      before `T`
    - The adjustment factor of a day is the product of the factors of all
      the later events

    :param price_df: prices from `simulate_dividend_payer()`
    :return: dataframe with the raw `close`, `split_adjusted`,
        `fully_adjusted` closes, and the `split_factor` and `total_factor`
        applied to the raw close
    """
    previous_close = price_df["close"].shift(1)
    split_event = 1.0 / price_df["split_ratio"]
    dividend_ratio = (price_df["dividend"] / previous_close).fillna(0.0)
    dividend_event = 1.0 - dividend_ratio

    def _factor_after(event: pd.Series) -> pd.Series:
        # Product of the events strictly after each day.
        reverse_product = event.iloc[::-1].cumprod().iloc[::-1]
        factor = reverse_product.shift(-1).fillna(1.0)
        return factor

    split_factor = _factor_after(split_event)
    total_factor = _factor_after(split_event * dividend_event)
    adjusted_df = pd.DataFrame(
        {
            "close": price_df["close"],
            "split_adjusted": price_df["close"] * split_factor,
            "fully_adjusted": price_df["close"] * total_factor,
            "split_factor": split_factor,
            "total_factor": total_factor,
        }
    )
    return adjusted_df


def cell3_1_plot_adjusted_prices(
    adjusted_df: pd.DataFrame,
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot price-only vs total-return growth and the adjustment factor.

    :param adjusted_df: adjusted prices from `adjust_prices()`
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with the growth of $1 and the cumulative adjustment factor
    """
    fig, axes = _new_figure(figsize, nrows=2)
    # Plot the growth of $1 with and without dividends.
    ax = axes[0, 0]
    price_index = (
        adjusted_df["split_adjusted"] / adjusted_df["split_adjusted"].iloc[0]
    )
    total_index = (
        adjusted_df["fully_adjusted"] / adjusted_df["fully_adjusted"].iloc[0]
    )
    ax.plot(
        total_index.index,
        total_index,
        color=_BLUE,
        linewidth=1.6,
        label=f"total return: ${total_index.iloc[-1]:.1f}",
    )
    ax.plot(
        price_index.index,
        price_index,
        color=_CORAL,
        linewidth=1.6,
        label=f"price only: ${price_index.iloc[-1]:.1f}",
    )
    ax.set_ylabel("growth of $1")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    # Plot the factor that turns the raw close into the adjusted close.
    ax = axes[1, 0]
    ax.plot(
        adjusted_df.index,
        adjusted_df["total_factor"],
        color=_BLUE,
        linewidth=1.6,
        drawstyle="steps-post",
        label="splits and dividends",
    )
    ax.plot(
        adjusted_df.index,
        adjusted_df["split_factor"],
        color=_ORANGE,
        linewidth=1.4,
        linestyle="--",
        drawstyle="steps-post",
        label="splits only",
    )
    ax.set_yscale("log")
    ax.set_yticks([0.2, 0.3, 0.5, 1.0])
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylabel("factor on the raw close")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def make_adjustment_table(adjusted_df: pd.DataFrame, date: str) -> pd.DataFrame:
    """
    Compare the price known on a past date with the adjusted price today.

    :param adjusted_df: adjusted prices from `adjust_prices()`
    :param date: past date, e.g., `2010-06-30`
    :return: table with the raw close on the date and the adjusted closes of a
        database built today
    """
    row = adjusted_df.loc[:date].iloc[-1]
    table_df = pd.DataFrame(
        {
            "price": [
                row["close"],
                row["split_adjusted"],
                row["fully_adjusted"],
            ],
            "total_factor": [1.0, row["split_factor"], row["total_factor"]],
        },
        index=[
            f"raw close on {row.name.date()}",
            "split-adjusted close",
            "fully adjusted close",
        ],
    )
    return table_df


# #############################################################################
# Cell 3.3: Time zones and daylight saving
# #############################################################################


def compute_close_in_utc(start: str, end: str) -> pd.DataFrame:
    """
    Compute the UTC time of the New York close and the gap with London.

    - The New York close is 16:00 local time
    - The gap is the difference between noon in London and noon in New York,
      both expressed in UTC

    :param start: first date, e.g., `2024-01-01`
    :param end: last date, e.g., `2025-12-31`
    :return: one row per calendar day with the UTC hour of the New York close
        and the gap in hours between New York and London
    """
    days = pd.date_range(start, end, freq="D")
    close_utc = (
        (days + pd.Timedelta(hours=16))
        .tz_localize("America/New_York")
        .tz_convert("UTC")
    )
    noon_ny_utc = (
        (days + pd.Timedelta(hours=12))
        .tz_localize("America/New_York")
        .tz_convert("UTC")
    )
    noon_london_utc = (
        (days + pd.Timedelta(hours=12))
        .tz_localize("Europe/London")
        .tz_convert("UTC")
    )
    gap_hours = (noon_ny_utc - noon_london_utc) / pd.Timedelta(hours=1)
    close_df = pd.DataFrame(
        {
            "ny_close_utc_hour": close_utc.hour,
            "ny_minus_london_hours": gap_hours,
        },
        index=days,
    )
    return close_df


def list_non_24_hour_days(start: str, end: str, *, time_zone: str) -> pd.Series:
    """
    List the local days that do not have 24 hours.

    :param start: first date, e.g., `2024-01-01`
    :param end: last date, e.g., `2024-12-31`
    :param time_zone: time zone name, e.g., `America/New_York`
    :return: length in hours of each local day that is not 24 hours long
    """
    days = pd.date_range(start, end, freq="D")
    midnight = days.tz_localize(time_zone)
    next_midnight = (days + pd.Timedelta(days=1)).tz_localize(time_zone)
    hours = pd.Series(
        (next_midnight - midnight) / pd.Timedelta(hours=1), index=days
    )
    hours = hours[hours != 24.0]
    return hours


def cell3_3_plot_utc_close(
    close_df: pd.DataFrame,
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> mfigure.Figure:
    """
    Plot the UTC hour of the New York close and the New York / London gap.

    :param close_df: output of `compute_close_in_utc()`
    :param figsize: figure size, `None` uses `plt.rcParams["figure.figsize"]`
    :return: figure with the UTC hour of the close and the time zone gap
    """
    fig, axes = _new_figure(figsize, nrows=2, sharex=True)
    # Plot the UTC hour of the same local close.
    ax = axes[0, 0]
    ax.plot(
        close_df.index,
        close_df["ny_close_utc_hour"],
        color=_BLUE,
        linewidth=1.8,
        drawstyle="steps-post",
    )
    ax.set_ylabel("UTC hour of the\n16:00 New York close")
    ax.set_yticks([20, 21])
    ax.set_ylim(19.5, 21.5)
    # Plot the gap, shading the weeks where it is not the usual 5 hours.
    ax = axes[1, 0]
    ax.plot(
        close_df.index,
        close_df["ny_minus_london_hours"],
        color=_CORAL,
        linewidth=1.8,
        drawstyle="steps-post",
    )
    is_off = close_df["ny_minus_london_hours"] != 5.0
    ax.fill_between(
        close_df.index,
        3.5,
        5.5,
        where=is_off,
        color=_LIGHT_GRAY,
        step="post",
    )
    ax.set_ylabel("New York minus\nLondon (hours)")
    ax.set_yticks([4, 5])
    ax.set_ylim(3.5, 5.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig
