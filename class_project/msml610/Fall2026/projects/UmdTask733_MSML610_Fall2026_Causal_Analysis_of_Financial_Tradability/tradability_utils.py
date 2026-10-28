"""
Utilities for the Causal Analysis of Financial Tradability project.

Import as:

import tradability_utils as tu
"""

import os
import time

import numpy as np
import pandas as pd
import requests

_BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"
_DATA_DIR = "data"


def load_prices(symbol: str, interval: str, start: str, end: str) -> pd.Series:
    """
    Load close prices from Binance, using a local CSV cache.

    :param symbol: trading pair, e.g., "BTCUSDT"
    :param interval: bar size, e.g., "1h", "1d"
    :param start: start date (inclusive), e.g., "2020-01-01"
    :param end: end date (exclusive), e.g., "2026-10-01"
    :return: close prices indexed by bar open time (UTC)
    """
    file_name = f"{symbol}_{interval}_{start}_{end}.csv"
    file_path = os.path.join(_DATA_DIR, file_name)
    # Use the cached file if we already downloaded this data.
    if os.path.exists(file_path):
        prices = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return prices["close"]
    # Binance returns at most 1000 bars per request, so download in pages.
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    # End is exclusive: do not include the bar that opens exactly at `end`.
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000) - 1
    rows = []
    while start_ms < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000,
        }
        resp = requests.get(_BINANCE_URL, params=params, timeout=30)
        # Fail loudly instead of silently using fake data.
        resp.raise_for_status()
        bars = resp.json()
        if not bars:
            break
        # Each bar is [open_time_ms, open, high, low, close, volume, ...].
        rows.extend((bar[0], float(bar[4])) for bar in bars)
        start_ms = bars[-1][0] + 1
        # Stay well under the Binance rate limit.
        time.sleep(0.2)
    if not rows:
        msg = f"No data for {symbol} {interval} from {start} to {end}"
        raise ValueError(msg)
    prices = pd.DataFrame(rows, columns=["open_time", "close"])
    prices["open_time"] = pd.to_datetime(
        prices["open_time"], unit="ms", utc=True
    )
    prices = prices.drop_duplicates("open_time").set_index("open_time")
    os.makedirs(_DATA_DIR, exist_ok=True)
    prices.to_csv(file_path)
    return prices["close"]


def compute_returns(prices: pd.Series) -> pd.Series:
    """
    Compute simple returns from prices.

    :param prices: close prices
    :return: percentage returns, without the first NaN
    """
    rets = prices.pct_change().dropna()
    return rets


def simulate_pnl(
    rets: pd.Series,
    hit_rate: float,
    n_sims: int,
    *,
    mode: str = "shuffle",
    seed: int = 0,
) -> np.ndarray:
    """
    Simulate the PnL of a predictor with a given hit rate.

    A correct call earns |r| and a wrong call loses |r|, so the PnL is
    sum((2 * hit - 1) * |r|).

    :param rets: returns, one per trade
    :param hit_rate: fraction of correct calls, e.g., 0.51
    :param n_sims: number of simulations
    :param mode: how to draw the hits
        - "shuffle": exactly round(hit_rate * N) hits in random order
          (Exercise #1 in #381)
        - "bernoulli": each call is correct with probability hit_rate
          (as in the spec formalization)
    :param seed: random seed
    :return: array with one PnL value per simulation
    """
    rng = np.random.default_rng(seed)
    abs_rets = np.abs(rets.to_numpy())
    n = len(abs_rets)
    n_hits = int(round(hit_rate * n))
    base_hits = np.zeros(n)
    base_hits[:n_hits] = 1
    pnls = np.empty(n_sims)
    for i in range(n_sims):
        if mode == "shuffle":
            hits = rng.permutation(base_hits)
        elif mode == "bernoulli":
            hits = (rng.random(n) < hit_rate).astype(float)
        else:
            raise ValueError(f"Invalid mode='{mode}'")
        pnls[i] = np.sum((2 * hits - 1) * abs_rets)
    return pnls


def run_experiment(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    hit_rates: list[float],
    n_sims: int,
    *,
    mode: str = "shuffle",
) -> pd.DataFrame:
    """
    Simulate the PnL distribution for each hit rate on one asset / frequency.

    :param symbol: trading pair, e.g., "BTCUSDT"
    :param interval: bar size, e.g., "1h", "1d"
    :param start: start date (inclusive)
    :param end: end date (exclusive)
    :param hit_rates: hit rates to simulate, e.g., [0.50, 0.51]
    :param n_sims: number of simulations per hit rate
    :param mode: "shuffle" or "bernoulli" (see `simulate_pnl()`)
    :return: one row per (hit_rate, simulation) with the PnL
    """
    prices = load_prices(symbol, interval, start, end)
    rets = compute_returns(prices)
    results = []
    for hit_rate in hit_rates:
        pnls = simulate_pnl(rets, hit_rate, n_sims, mode=mode)
        df = pd.DataFrame({"pnl": pnls})
        df["hit_rate"] = hit_rate
        results.append(df)
    results = pd.concat(results, ignore_index=True)
    results["symbol"] = symbol
    results["interval"] = interval
    results["n_trades"] = len(rets)
    results["mode"] = mode
    return results


def compute_effective_n(rets: pd.Series) -> float:
    """
    Compute the effective number of trades, accounting for unequal moves.

    With returns of different sizes, a few large moves dominate the PnL, so
    the PnL is as noisy as if there were fewer equal-sized trades:
    N_eff = (sum |r|)^2 / sum r^2. N_eff = N when all |r| are equal.

    :param rets: returns, one per trade
    :return: effective number of trades
    """
    abs_rets = np.abs(rets.to_numpy())
    n_eff = abs_rets.sum() ** 2 / np.sum(abs_rets**2)
    return float(n_eff)


def compute_min_hit_rate(n: float, *, z_q: float = 1.645) -> float:
    """
    Compute the minimum hit rate for a target probability of positive PnL.

    Use the normal approximation from the spec: p* = 1/2 + z_q / (2 sqrt(N)).

    :param n: number of trades (or effective number of trades)
    :param z_q: normal quantile of the target probability (1.645 for 95%)
    :return: minimum hit rate
    """
    p_star = 0.5 + z_q / (2 * np.sqrt(n))
    return float(p_star)
