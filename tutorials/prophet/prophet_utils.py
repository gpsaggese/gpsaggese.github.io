"""
Utility functions for Prophet-based time series forecasting workflows.

Import as:

import tutorials.tutorial_prophet.prophet_utils as tprpru
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import prophet
import sklearn.metrics

import helpers.hdbg as hdbg

_LOG = logging.getLogger(__name__)


# #############################################################################
# Data generation
# #############################################################################


def generate_synthetic_time_series(config: Dict) -> pd.DataFrame:
    """
    Generate a synthetic daily time series with trend, seasonality, holidays,
    and autoregression.

    The series is built as:
        y[t] = phi * y[t-1] + trend[t] + weekly_seasonality[t]
               + holiday_effect[t] + noise[t]

    :param config: configuration dictionary with keys:
        - ``train_start_date``: start of the date range (str, e.g. "2020-01-01")
        - ``test_end_date``: end of the date range (str)
        - ``data``: sub-dict with fields:
            - ``slope``: linear trend slope
            - ``intercept``: linear trend intercept
            - ``weekly_amp_sin``: amplitude for the sine weekly term
            - ``weekly_amp_cos``: amplitude for the cosine weekly term
            - ``holidays_dates``: list of date strings to apply a holiday effect
            - ``holidays_impact``: additive effect magnitude on holiday dates
            - ``ar_order``: autoregressive order (currently only 1 is supported)
            - ``phi``: AR(1) coefficient
            - ``seed``: random seed for reproducibility
            - ``noise_sigma``: standard deviation of Gaussian noise
    :return: DataFrame with columns ``ds``, ``y``, and ``y.lag1`` (the 1-step
        lag of ``y``); the first row (which has a NaN lag) is dropped
    """
    hdbg.dassert_in("train_start_date", config)
    hdbg.dassert_in("test_end_date", config)
    data_cfg = config["data"]
    # Generate date range covering training and test periods.
    dates = pd.date_range(
        start=config["train_start_date"], end=config["test_end_date"], freq="D"
    )
    time = np.arange(len(dates))
    # Linear trend.
    y_trend = data_cfg["slope"] * time + data_cfg["intercept"]
    # Weekly seasonality via Fourier terms.
    p_weekly = 7
    y_weekly = data_cfg["weekly_amp_sin"] * np.sin(
        2 * np.pi * time / p_weekly
    ) + data_cfg["weekly_amp_cos"] * np.cos(2 * np.pi * time / p_weekly)
    # Holiday additive effect.
    holiday_effect = np.zeros(len(dates))
    holiday_dates = pd.to_datetime(data_cfg["holidays_dates"]).date
    holiday_effect[np.isin(dates.date, holiday_dates)] = data_cfg[
        "holidays_impact"
    ]
    # White noise.
    np.random.seed(data_cfg["seed"])
    noise = np.random.normal(
        loc=0, scale=data_cfg["noise_sigma"], size=len(time)
    )
    # AR(1) process.
    y = np.zeros(len(time))
    y[0] = y_trend[0] + y_weekly[0] + holiday_effect[0] + noise[0]
    for i in range(1, len(time)):
        y[i] = (
            data_cfg["phi"] * y[i - 1]
            + y_trend[i]
            + y_weekly[i]
            + holiday_effect[i]
            + noise[i]
        )
    df = pd.DataFrame({"ds": dates, "y": y})
    # Add 1-step lag as an external regressor.
    df["y.lag1"] = df["y"].shift(1)
    # Drop the first row which has a NaN lag value.
    df = df.dropna().reset_index(drop=True)
    _LOG.info("Generated synthetic data: shape=%s", df.shape)
    return df


# #############################################################################
# Holiday helpers
# #############################################################################


def build_holidays_df(
    dates: List[str],
    names: List[str],
    *,
    lower_window: int = 0,
    upper_window: int = 0,
) -> pd.DataFrame:
    """
    Build a Prophet-compatible holidays DataFrame.

    :param dates: list of date strings (e.g. ["2022-12-25", "2023-12-25"])
    :param names: list of holiday name strings matching ``dates`` in length
    :param lower_window: number of days before each holiday that are affected
    :param upper_window: number of days after each holiday that are affected
    :return: DataFrame with columns ``holiday``, ``ds``, ``lower_window``,
        ``upper_window``
    """
    hdbg.dassert_eq(len(dates), len(names))
    df = pd.DataFrame(
        {
            "holiday": names,
            "ds": pd.to_datetime(dates),
            "lower_window": lower_window,
            "upper_window": upper_window,
        }
    )
    return df


# #############################################################################
# ProphetForecastModel
# #############################################################################


class ProphetForecastModel:
    """
    Thin wrapper around Facebook Prophet for fitting, predicting, and
    evaluating time series forecasts.
    """

    def __init__(
        self, config: Dict, *, holidays: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Initialize the Prophet model.

        :param config: Prophet hyperparameters passed directly to
            ``prophet.Prophet()``
        :param holidays: optional DataFrame of holidays in Prophet format
            (columns: ``holiday``, ``ds``, ``lower_window``, ``upper_window``)
        """
        self.config = config
        self.holidays = holidays
        self.model = prophet.Prophet(**config, holidays=holidays)
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the Prophet model on the given DataFrame.

        Any column in ``df`` besides ``ds`` and ``y`` is treated as an
        external regressor and registered automatically.

        :param df: training data with at least columns ``ds`` and ``y``
        """
        for col in df.columns:
            if col not in ["ds", "y"]:
                _LOG.info("Adding regressor: %s", col)
                self.model.add_regressor(col)
        self.model.fit(df)
        self.fitted = True

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate forecasts for the timestamps in ``df``.

        Note: when external regressors are used, ``df`` must contain those
        regressor columns with known values (i.e. in-sample prediction or
        a test set where lagged values are available).

        :param df: DataFrame containing at least ``ds`` and any regressor
            columns used during training
        :return: Prophet forecast DataFrame (includes ``yhat``, ``yhat_lower``,
            ``yhat_upper``, and component columns)
        """
        hdbg.dassert(self.fitted, "Model must be fitted before prediction.")
        return self.model.predict(df)

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute forecast accuracy against observed values.

        :param df: merged DataFrame containing both ``y`` (actuals) and
            ``yhat`` (predictions)
        :return: dict with keys ``mae``, ``rmse``, and ``mape``
        """
        y_true = df["y"]
        y_pred = df["yhat"]
        return {
            "mae": sklearn.metrics.mean_absolute_error(y_true, y_pred),
            "rmse": sklearn.metrics.root_mean_squared_error(y_true, y_pred),
            "mape": (abs(y_true - y_pred) / y_true).mean() * 100,
        }

    def get_model(self) -> prophet.Prophet:
        """
        Return the internal Prophet object for direct access.

        :return: the fitted (or unfitted) ``prophet.Prophet`` instance
        """
        return self.model
