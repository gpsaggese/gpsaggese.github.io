"""
Load and clean OHLCV bars for the Causal Analysis of Financial Tradability
project.

Import as:

import data_utils as dutil
"""

import hashlib
import io
import os
import zipfile

import numpy as np
import pandas as pd
import requests

_BINANCE_URL = "https://data.binance.vision/data/spot/monthly/klines"
_DATA_DIR = "data"
_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


# #############################################################################
# Load
# #############################################################################


def _download_month(symbol: str, interval: str, month: str) -> pd.DataFrame:
    """
    Download one month of bars from the Binance bulk data files.

    :param symbol: trading pair, e.g., "BTCUSDT"
    :param interval: bar size, e.g., "5m"
    :param month: month to download, e.g., "2024-01"
    :return: OHLCV bars indexed by bar open time (UTC)
    """
    file_name = f"{symbol}-{interval}-{month}.zip"
    url = f"{_BINANCE_URL}/{symbol}/{interval}/{file_name}"
    resp = requests.get(url, timeout=60)
    # Fail loudly instead of silently skipping a month.
    resp.raise_for_status()
    # Check the file against the checksum that Binance publishes next to it.
    checksum_resp = requests.get(url + ".CHECKSUM", timeout=60)
    checksum_resp.raise_for_status()
    expected = checksum_resp.text.split()[0]
    actual = hashlib.sha256(resp.content).hexdigest()
    if actual != expected:
        raise ValueError(f"Checksum mismatch for {file_name}")
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = zf.namelist()[0]
        # Each row is [open_time, open, high, low, close, volume, ...].
        df = pd.read_csv(zf.open(csv_name), header=None, usecols=range(6))
    df.columns = ["open_time"] + _OHLCV_COLUMNS
    # Binance switched from milliseconds to microseconds in 2025.
    unit = "us" if df["open_time"].iloc[0] > 1e14 else "ms"
    df["open_time"] = pd.to_datetime(df["open_time"], unit=unit, utc=True)
    df = df.set_index("open_time")
    return df


def load_ohlcv(
    symbol: str, start: str, end: str, *, interval: str = "5m"
) -> pd.DataFrame:
    """
    Load OHLCV bars from Binance, using a local parquet cache.

    The data comes from the monthly files on `data.binance.vision`, which are
    much faster to download than the API for 5-minute bars.

    :param symbol: trading pair, e.g., "BTCUSDT"
    :param start: start date (inclusive), e.g., "2020-01-01"
    :param end: end date (exclusive), e.g., "2026-10-01"
    :param interval: bar size, e.g., "5m"
    :return: OHLCV bars indexed by bar open time (UTC), as published by
        Binance (not cleaned: may have duplicates and gaps)
    """
    file_name = f"{symbol}_{interval}_{start}_{end}.parquet"
    file_path = os.path.join(_DATA_DIR, file_name)
    # Use the cached file if we already downloaded this data.
    if os.path.exists(file_path):
        bars = pd.read_parquet(file_path)
        return bars
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    # Download every month that overlaps [start, end).
    months = pd.period_range(
        start_ts.tz_localize(None),
        (end_ts - pd.Timedelta(1, "ns")).tz_localize(None),
        freq="M",
    )
    bars = [_download_month(symbol, interval, str(month)) for month in months]
    bars = pd.concat(bars)
    bars = bars[(bars.index >= start_ts) & (bars.index < end_ts)]
    if bars.empty:
        msg = f"No data for {symbol} {interval} from {start} to {end}"
        raise ValueError(msg)
    os.makedirs(_DATA_DIR, exist_ok=True)
    bars.to_parquet(file_path)
    return bars


# #############################################################################
# Clean
# #############################################################################


def to_timedelta(interval: str) -> pd.Timedelta:
    """
    Convert a Binance interval to a timedelta.

    :param interval: bar size, e.g., "5m", "1h", "1d"
    :return: bar size as a timedelta
    """
    units = {"m": "min", "h": "h", "d": "D"}
    value, unit = int(interval[:-1]), interval[-1]
    if unit not in units:
        raise ValueError(f"Invalid interval='{interval}'")
    return pd.Timedelta(value, units[unit])


def find_gaps(bars: pd.DataFrame, *, interval: str = "5m") -> pd.DataFrame:
    """
    Find the runs of missing bars.

    :param bars: OHLCV bars indexed by bar open time
    :param interval: bar size, e.g., "5m"
    :return: one row per gap with:
        - `start`: open time of the first missing bar
        - `end`: open time of the last missing bar
        - `n_bars`: number of missing bars
    """
    step = to_timedelta(interval)
    index = bars.index.drop_duplicates().sort_values()
    # A gap is a jump of more than one bar between consecutive bars.
    diffs = index.to_series().diff()
    is_gap = diffs > step
    gaps = pd.DataFrame(
        {
            "start": index[:-1][is_gap.to_numpy()[1:]] + step,
            "end": index[is_gap.to_numpy()] - step,
        }
    )
    gaps["n_bars"] = ((gaps["end"] - gaps["start"]) / step).astype(int) + 1
    return gaps


def clean_bars(
    bars: pd.DataFrame,
    *,
    interval: str = "5m",
    max_fill_bars: int = 3,
    vol_window_bars: int = 288,
    extreme_std: float = 10.0,
) -> pd.DataFrame:
    """
    Clean OHLCV bars and flag the bars that need care.

    - Drop duplicate timestamps (keep the first)
    - Put the bars on a regular time grid, so that a shift by n rows is always
      a shift by n bars
    - Fill gaps of up to `max_fill_bars` bars with flat bars at the last close
      and zero volume
    - Leave longer gaps as NaN and start a new segment after each one, so
      features and labels can be computed per segment and never span a gap
    - Flag zero-volume bars and extreme returns, without dropping them, since
      fat tails are part of what we study

    :param bars: OHLCV bars indexed by bar open time
    :param interval: bar size, e.g., "5m"
    :param max_fill_bars: longest gap (in bars) to fill
    :param vol_window_bars: number of past bars used for the rolling std of
        returns (288 five-minute bars = 1 day)
    :param extreme_std: flag a return if |r| > extreme_std * rolling std
    :return: bars on a regular grid with the OHLCV columns and:
        - `ret`: return from the previous bar, NaN at the start of a segment
        - `is_missing`: the bar is not in the data
        - `is_filled`: the bar is missing and was filled
        - `is_zero_volume`: the bar is in the data with zero volume
        - `is_extreme`: |ret| is larger than `extreme_std` rolling stds
        - `segment_id`: id of the run of bars between long gaps, NaN inside a
          long gap
    """
    step = to_timedelta(interval)
    bars = bars[_OHLCV_COLUMNS].sort_index()
    bars = bars[~bars.index.duplicated(keep="first")]
    # Put the bars on a regular grid: missing bars become NaN rows.
    grid = pd.date_range(bars.index[0], bars.index[-1], freq=step)
    df = bars.reindex(grid)
    df.index.name = bars.index.name
    is_missing = df["close"].isna()
    # Number each run of missing bars and measure its length.
    run_id = (is_missing != is_missing.shift()).cumsum()
    run_len = is_missing.groupby(run_id).transform("size")
    is_filled = is_missing & (run_len <= max_fill_bars)
    is_long_gap = is_missing & ~is_filled
    # Fill short gaps with a flat bar at the last close and no volume.
    last_close = df["close"].ffill()
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].where(~is_filled, last_close)
    df["volume"] = df["volume"].where(~is_filled, 0.0)
    # Start a new segment after each long gap.
    gap_end = is_long_gap.shift(fill_value=False) & ~is_long_gap
    segment_id = gap_end.cumsum().astype("Int64")
    df["segment_id"] = segment_id.where(~is_long_gap)
    # Compute returns within each segment, so no return spans a long gap.
    prev_close = df.groupby("segment_id")["close"].shift()
    df["ret"] = df["close"] / prev_close - 1
    # Use only past returns for the rolling std (no look-ahead).
    past_std = df.groupby("segment_id")["ret"].transform(
        lambda r: r.shift().rolling(vol_window_bars, min_periods=12).std()
    )
    df["is_missing"] = is_missing
    df["is_filled"] = is_filled
    df["is_zero_volume"] = (df["volume"] == 0) & ~is_missing
    df["is_extreme"] = df["ret"].abs() > extreme_std * past_std
    cols = _OHLCV_COLUMNS + [
        "ret",
        "is_missing",
        "is_filled",
        "is_zero_volume",
        "is_extreme",
        "segment_id",
    ]
    return df[cols]


def compute_quality_table(
    raw_bars: pd.DataFrame, clean: pd.DataFrame, *, interval: str = "5m"
) -> pd.Series:
    """
    Summarize the data quality of one coin.

    :param raw_bars: bars as returned by `load_ohlcv()`
    :param clean: bars as returned by `clean_bars()`
    :param interval: bar size, e.g., "5m"
    :return: counts of bars, duplicates, gaps, and flagged bars
    """
    gaps = find_gaps(raw_bars, interval=interval)
    step = to_timedelta(interval)
    longest_gap = gaps["n_bars"].max() * step if len(gaps) else pd.Timedelta(0)
    n_segments = int(clean["segment_id"].nunique())
    stats = {
        "start": clean.index[0],
        "end": clean.index[-1],
        "n_grid_bars": len(clean),
        "n_raw_bars": len(raw_bars),
        "n_duplicates": int(raw_bars.index.duplicated().sum()),
        "n_gaps": len(gaps),
        "n_missing_bars": int(clean["is_missing"].sum()),
        "longest_gap": longest_gap,
        "n_filled_bars": int(clean["is_filled"].sum()),
        "n_long_gaps": n_segments - 1,
        "n_segments": n_segments,
        "n_zero_volume": int(clean["is_zero_volume"].sum()),
        "n_extreme": int(clean["is_extreme"].sum()),
        "max_abs_ret": float(np.nanmax(clean["ret"].abs())),
    }
    return pd.Series(stats)


# #############################################################################
# Features
# #############################################################################


def get_n_bars(duration: str, interval: str) -> int:
    """
    Compute how many bars of size `interval` fit in `duration`.

    :param duration: length of time, e.g., "1h", "1d"
    :param interval: bar size, e.g., "5m"
    :return: number of bars, e.g., 12 for "1h" with "5m" bars
    """
    n_bars = to_timedelta(duration) / to_timedelta(interval)
    if n_bars != int(n_bars) or n_bars < 1:
        raise ValueError(f"'{duration}' is not a multiple of '{interval}'")
    return int(n_bars)


def _compute_return_features(
    clean: pd.DataFrame, lags: list[int], *, prefix: str = ""
) -> pd.DataFrame:
    """
    Compute the return over the last `lag` bars, within each segment.

    :param clean: bars as returned by `clean_bars()`
    :param lags: numbers of bars to look back, e.g., [1, 3, 12]
    :param prefix: prefix for the column names, e.g., "eth_"
    :return: one column `{prefix}ret_{lag}` per lag
    """
    by_segment = clean.groupby("segment_id")["close"]
    features = pd.DataFrame(index=clean.index)
    for lag in lags:
        past_close = by_segment.shift(lag)
        features[f"{prefix}ret_{lag}"] = clean["close"] / past_close - 1
    return features


def compute_features(
    clean: pd.DataFrame,
    *,
    other: pd.DataFrame | None = None,
    other_name: str = "other",
    interval: str = "5m",
) -> pd.DataFrame:
    """
    Compute the features for each bar, using only data up to that bar.

    The row at time t uses the bars up to and including bar t, so it is what
    is known at the close of bar t. All rolling windows stay inside one
    segment, so no feature spans a long gap.

    :param clean: bars as returned by `clean_bars()`
    :param other: bars of another asset as returned by `clean_bars()`, for the
        cross-asset features
    :param other_name: prefix for the cross-asset columns, e.g., "eth"
    :param interval: bar size, e.g., "5m"
    :return: one row per bar with:
        - `ret_{n}`: return over the last n bars, for n in 1, 3, 6, 12, 48
        - `vol_{w}`: std of the 1-bar returns over the last 1h, 1d, 7d
        - `volume_z`: z-score of log volume against the last day
        - `range`: (high - low) / close of the bar
        - `mom_{w}`: return over the last 1d and 7d
        - `hour`, `day_of_week`: time of the bar (UTC)
        - `{other_name}_ret_{n}`: the other asset's return over the last n
          bars, for n in 1, 3, 12
    """
    lags = [1, 3, 6, 12, 48]
    features = _compute_return_features(clean, lags)
    by_segment = clean.groupby("segment_id")
    # Realized volatility over the last 1h, 1d, and 7d.
    for window in ["1h", "1d", "7d"]:
        n_bars = get_n_bars(window, interval)
        features[f"vol_{window}"] = by_segment["ret"].transform(
            lambda r, n=n_bars: r.rolling(n, min_periods=n).std()
        )
    # Volume compared with its last day, on a log scale.
    n_bars = get_n_bars("1d", interval)
    log_volume = np.log1p(clean["volume"])
    by_segment_volume = log_volume.groupby(clean["segment_id"])
    mean = by_segment_volume.transform(
        lambda v: v.rolling(n_bars, min_periods=n_bars).mean()
    )
    std = by_segment_volume.transform(
        lambda v: v.rolling(n_bars, min_periods=n_bars).std()
    )
    features["volume_z"] = (log_volume - mean) / std
    features["range"] = (clean["high"] - clean["low"]) / clean["close"]
    # Momentum over the last 1d and 7d.
    for window in ["1d", "7d"]:
        n_bars = get_n_bars(window, interval)
        past_close = by_segment["close"].shift(n_bars)
        features[f"mom_{window}"] = clean["close"] / past_close - 1
    features["hour"] = clean.index.hour
    features["day_of_week"] = clean.index.dayofweek
    # Returns of the other asset, aligned on the same bar times.
    if other is not None:
        other_features = _compute_return_features(
            other, [1, 3, 12], prefix=f"{other_name}_"
        )
        features = features.join(other_features, how="left")
    # Inside a long gap there is no data, so no features.
    features[clean["segment_id"].isna()] = np.nan
    return features


# #############################################################################
# Labels
# #############################################################################


def compute_labels(
    clean: pd.DataFrame,
    horizons: list[str],
    *,
    interval: str = "5m",
) -> pd.DataFrame:
    """
    Compute the forward return and its sign for each horizon.

    The label at time t is the return from the close of bar t to the close of
    bar t + h. It is NaN when t + h is past the end of the segment, so no
    label spans a long gap.

    :param clean: bars as returned by `clean_bars()`
    :param horizons: holding periods, e.g., ["5m", "15m", "1h", "4h", "1d"]
    :param interval: bar size, e.g., "5m"
    :return: one row per bar with, for each horizon h:
        - `fwd_ret_{h}`: return from t to t + h
        - `label_{h}`: 1 if `fwd_ret_{h}` > 0, 0 if < 0, NaN if exactly 0 or
          unknown
    """
    by_segment = clean.groupby("segment_id")["close"]
    labels = pd.DataFrame(index=clean.index)
    for horizon in horizons:
        n_bars = get_n_bars(horizon, interval)
        future_close = by_segment.shift(-n_bars)
        fwd_ret = future_close / clean["close"] - 1
        labels[f"fwd_ret_{horizon}"] = fwd_ret
        label = pd.Series(np.nan, index=clean.index)
        label[fwd_ret > 0] = 1.0
        label[fwd_ret < 0] = 0.0
        labels[f"label_{horizon}"] = label
    return labels


def sample_non_overlapping(
    df: pd.DataFrame, horizon: str, *, interval: str = "5m", offset: int = 0
) -> pd.DataFrame:
    """
    Keep one decision time every `horizon`, so that trades do not overlap.

    With 5-minute bars and a 1h horizon, the forward returns of consecutive
    bars share 11 of their 12 bars, so they are not independent trades.
    Keeping every 12th bar gives independent trades.

    :param df: rows on a regular grid of `interval` bars, e.g., from
        `compute_labels()`
    :param horizon: holding period, e.g., "1h"
    :param interval: bar size, e.g., "5m"
    :param offset: which bar of each block to keep, from 0 to n_bars - 1
    :return: every n-th row of `df`, starting at `offset`
    """
    n_bars = get_n_bars(horizon, interval)
    if not 0 <= offset < n_bars:
        raise ValueError(f"Invalid offset={offset} for {n_bars} bars")
    return df.iloc[offset::n_bars]
