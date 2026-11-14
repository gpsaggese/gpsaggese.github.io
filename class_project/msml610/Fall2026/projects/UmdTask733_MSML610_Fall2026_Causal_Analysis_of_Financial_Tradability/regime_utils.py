"""
Label the market conditions (regimes) at each bar, using only past data.

Import as:

import regime_utils as rutil
"""

import numpy as np
import pandas as pd

import data_utils as dutil

VOL_REGIMES = ["calm", "normal", "volatile"]
TREND_REGIMES = ["falling", "flat", "rising"]


def _compute_vol_thresholds(
    vol: pd.Series, *, n_days: int, min_days: int
) -> pd.DataFrame:
    """
    Compute the terciles of the daily volatility over the past `n_days` days.

    The thresholds used on day D come from the days before D only.

    :param vol: volatility at each bar
    :param n_days: number of past days for the terciles
    :param min_days: minimum number of past days with data
    :return: one row per bar with the columns `low` and `high`
    """
    # One value per day: the volatility at the end of the day.
    daily = vol.resample("1D").last()
    rolling = daily.rolling(n_days, min_periods=min_days)
    thresholds = pd.DataFrame(
        {"low": rolling.quantile(1 / 3), "high": rolling.quantile(2 / 3)}
    )
    # Use only the days before the current one.
    thresholds = thresholds.shift(1)
    thresholds = thresholds.reindex(vol.index, method="ffill")
    return thresholds


def compute_regimes(
    features: pd.DataFrame,
    *,
    interval: str = "5m",
    n_days: int = 365,
    min_days: int = 180,
    trend_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Label the volatility and the trend of the market at each bar.

    - Volatility: the 7-day volatility compared with its terciles over the
      past year: "calm" (lowest third), "normal", "volatile" (highest third)
    - Trend: the 7-day return divided by its expected size from the 7-day
      volatility, t = mom_7d / (vol_7d * sqrt(n)), with n the number of bars
      in 7 days: "falling" (t < -threshold), "flat", "rising"
      (t > threshold)

    Both use only data up to the bar, so they are known at decision time.

    :param features: features as returned by `compute_features()`, with
        `vol_7d` and `mom_7d`
    :param interval: bar size, e.g., "5m"
    :param n_days: number of past days for the volatility terciles
    :param min_days: minimum number of past days for the terciles
    :param trend_threshold: |t| above which the market is trending
    :return: one row per bar with:
        - `vol_regime`: "calm", "normal", "volatile", or NaN
        - `trend_regime`: "falling", "flat", "rising", or NaN
        - `trend_t`: the trend t-statistic
    """
    vol = features["vol_7d"]
    thresholds = _compute_vol_thresholds(vol, n_days=n_days, min_days=min_days)
    vol_regime = pd.Series(np.nan, index=features.index, dtype=object)
    has_vol = vol.notna() & thresholds["low"].notna()
    vol_regime[has_vol] = "normal"
    vol_regime[has_vol & (vol < thresholds["low"])] = "calm"
    vol_regime[has_vol & (vol > thresholds["high"])] = "volatile"
    n_bars = dutil.get_n_bars("7d", interval)
    trend_t = features["mom_7d"] / (vol * np.sqrt(n_bars))
    trend_regime = pd.Series(np.nan, index=features.index, dtype=object)
    has_trend = trend_t.notna()
    trend_regime[has_trend] = "flat"
    trend_regime[has_trend & (trend_t < -trend_threshold)] = "falling"
    trend_regime[has_trend & (trend_t > trend_threshold)] = "rising"
    regimes = pd.DataFrame(
        {
            "vol_regime": pd.Categorical(vol_regime, categories=VOL_REGIMES),
            "trend_regime": pd.Categorical(
                trend_regime, categories=TREND_REGIMES
            ),
            "trend_t": trend_t,
        },
        index=features.index,
    )
    return regimes
