"""
Backtest a strategy that trades only in some market regimes.

Import as:

import backtest_utils as butil
"""

import numpy as np
import pandas as pd

import data_utils as dutil

# #############################################################################
# Select the regimes to trade
# #############################################################################


def select_groups_by_past_pnl(
    pos: pd.Series,
    fwd_ret: pd.Series,
    groups: pd.Series,
    test_years: list[int],
    *,
    cost: float,
    min_trades: int = 100,
) -> pd.DataFrame:
    """
    Choose the groups to trade in each year from the trades of earlier years.

    A group, e.g., a volatility regime, is traded in a year if its trades in
    all earlier years made money after costs, on average. So the choice for a
    year never uses that year's returns.

    :param pos: position for each trade, +1 or -1
    :param fwd_ret: return over the holding period of each trade
    :param groups: group of each trade
    :param test_years: years to choose for, e.g., [2022, 2023]
    :param cost: cost of a round trip, as a return (0.001 = 10 bp)
    :param min_trades: fewest earlier trades needed to trade a group
    :return: one row per test year and one column per group, True if the
        group is traded in that year
    """
    df = pd.DataFrame({"pos": pos, "fwd_ret": fwd_ret, "group": groups})
    df = df.dropna()
    df = df[df["fwd_ret"] != 0]
    df["group"] = df["group"].astype(str)
    df["net"] = df["pos"] * df["fwd_ret"] - cost
    years = df.index.year
    all_groups = sorted(df["group"].unique())
    rows = {}
    for test_year in test_years:
        past = df[years < test_year]
        stats = past.groupby("group")["net"].agg(["mean", "count"])
        stats = stats.reindex(all_groups)
        is_traded = (stats["mean"] > 0) & (stats["count"] >= min_trades)
        rows[test_year] = is_traded
    selected = pd.DataFrame(rows).T
    selected.index.name = "year"
    return selected


def keep_selected(
    pos: pd.Series, groups: pd.Series, selected: pd.DataFrame
) -> pd.Series:
    """
    Keep the positions in the groups selected for their year, else stay out.

    :param pos: position for each trade, +1 or -1
    :param groups: group of each trade
    :param selected: groups to trade per year, as returned by
        `select_groups_by_past_pnl()`
    :return: `pos` where the group is selected for the year, NaN (no trade)
        elsewhere and in the years not in `selected`
    """
    stacked = selected.stack()
    keys = pd.MultiIndex.from_arrays(
        [pos.index.year, groups.reindex(pos.index).astype(str).to_numpy()]
    )
    is_kept = stacked.reindex(keys).eq(True).to_numpy()
    kept = pos.where(is_kept)
    return kept


# #############################################################################
# Measure the strategy
# #############################################################################


def compute_pnl(pos: pd.Series, fwd_ret: pd.Series, *, cost: float) -> pd.Series:
    """
    Compute the PnL of each decision time, with 0 when there is no trade.

    Each trade holds one unit from t to t + horizon, so the PnL is
    `pos * fwd_ret - cost`. PnLs are added, not compounded.

    :param pos: position for each decision time, +1, -1, or NaN for no trade
    :param fwd_ret: return over the holding period
    :param cost: cost of a round trip, as a return (0.001 = 10 bp)
    :return: PnL of each decision time with a known return
    """
    df = pd.DataFrame({"pos": pos, "fwd_ret": fwd_ret})
    df = df[df["fwd_ret"].notna()]
    is_trade = df["pos"].notna()
    pnl = (df["pos"] * df["fwd_ret"] - cost).where(is_trade, 0.0)
    return pnl


def compute_backtest_stats(
    pos: pd.Series,
    fwd_ret: pd.Series,
    *,
    cost: float,
    horizon: str,
) -> pd.Series:
    """
    Measure a strategy over its decision times.

    :param pos: position for each decision time, +1, -1, or NaN for no trade
    :param fwd_ret: return over the holding period
    :param cost: cost of a round trip, as a return (0.001 = 10 bp)
    :param horizon: holding period, e.g., "4h", to annualize the Sharpe ratio
    :return: metrics:
        - `n_trades`: number of trades
        - `share_traded`: share of the decision times with a trade
        - `net_bps`: average PnL per trade after costs, in basis points
        - `t_stat`: `net_bps` divided by its standard error
        - `total_return`: sum of the PnLs
        - `sharpe`: annualized Sharpe ratio, with 0 PnL when out of the
          market and 365 trading days a year
        - `max_drawdown`: largest fall of the cumulative PnL from a peak
    """
    pnl = compute_pnl(pos, fwd_ret, cost=cost)
    is_trade = pos.reindex(pnl.index).notna()
    trades = pnl[is_trade]
    n_trades = len(trades)
    if n_trades > 1:
        net_bps = trades.mean() * 1e4
        t_stat = trades.mean() / (trades.std() / np.sqrt(n_trades))
    else:
        net_bps = np.nan
        t_stat = np.nan
    n_per_year = pd.Timedelta(365, "D") / dutil.to_timedelta(horizon)
    std = pnl.std()
    sharpe = pnl.mean() / std * np.sqrt(n_per_year) if std > 0 else np.nan
    equity = pnl.cumsum()
    max_drawdown = (equity - equity.cummax()).min()
    stats = {
        "n_trades": n_trades,
        "share_traded": n_trades / len(pnl),
        "net_bps": net_bps,
        "t_stat": t_stat,
        "total_return": pnl.sum(),
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }
    return pd.Series(stats)
