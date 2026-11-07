"""
Utility functions for TensorFlow-based time series workflows.

Import as:

import tutorials.TensorFlow.tensorflow_utils as tteteuti
"""

import collections
import logging
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

_LOG = logging.getLogger(__name__)


# #############################################################################
# Data generation
# #############################################################################


def generate_time_series_data(config: dict) -> pd.DataFrame:
    """
    Generate a synthetic time series with trend, seasonality, holidays, and AR.

    The generated series combines:
    - A linear trend
    - Weekly seasonal drift
    - Holiday effects
    - First-order autoregression
    - Gaussian observation noise

    :param config: dictionary with keys:
        - ``train_start_date``: start of the training period (str)
        - ``test_end_date``: end of the test period (str)
        - ``data``: nested dict with ``slope``, ``intercept``,
          ``seasonal_drift_scale``, ``holidays_dates``, ``holidays_impact``,
          ``ar_order``, ``phi``, ``seed``, ``observational_noise_sigma``
    :return: DataFrame with columns ``ds``, ``y``, ``y.lag1``
    """
    data_cfg = config["data"]
    np.random.seed(data_cfg["seed"])
    dates = pd.date_range(
        start=config["train_start_date"],
        end=config["test_end_date"],
        freq="D",
    )
    time = np.arange(len(dates))
    # Linear trend.
    y_trend = data_cfg["slope"] * time + data_cfg["intercept"]
    # Weekly seasonal drift.
    seasonality_factor = 7
    for t in range(seasonality_factor, len(dates), seasonality_factor):
        y_trend[t] = y_trend[t - seasonality_factor] + np.random.normal(
            0, data_cfg["seasonal_drift_scale"]
        )
    # Holiday effect.
    holiday_effect = np.zeros(len(dates))
    holiday_dates = pd.to_datetime(data_cfg["holidays_dates"])
    holiday_effect[np.isin(dates.date, holiday_dates.date)] = data_cfg[
        "holidays_impact"
    ]
    # White noise.
    noise = np.random.normal(
        loc=0,
        scale=data_cfg["observational_noise_sigma"],
        size=len(time),
    )
    # AR(1) process.
    y = np.zeros(len(time))
    y[0] = y_trend[0] + holiday_effect[0] + noise[0]
    for i in range(1, len(time)):
        y[i] = (
            data_cfg["phi"] * y[i - 1]
            + y_trend[i]
            + holiday_effect[i]
            + noise[i]
        )
    df = pd.DataFrame({"ds": dates, "y": y})
    df["y.lag1"] = df["y"].shift(1)
    df = df.dropna()
    return df


def build_holiday_indicators(config: dict) -> np.ndarray:
    """
    Build a one-hot holiday indicator matrix for the full date range.

    :param config: dictionary with ``train_start_date``, ``test_end_date``,
        and ``data.holidays_dates``
    :return: float array of shape ``(n_days, n_holidays)``
    """
    all_dates = pd.date_range(
        start=config["train_start_date"], end=config["test_end_date"]
    )
    holiday_dates = pd.to_datetime(config["data"]["holidays_dates"])
    holiday_indicators = np.zeros((len(all_dates), len(holiday_dates)))
    for i, holiday in enumerate(holiday_dates):
        holiday_indicators[:, i] = (all_dates == holiday).astype(int)
    return holiday_indicators


# #############################################################################
# Model building
# #############################################################################


def build_sts_model(
    observed_time_series: np.ndarray,
    holiday_features: np.ndarray,
    config: dict,
):
    """
    Build a Structural Time Series (STS) model with trend, seasonality, AR,
    and holiday components.

    :param observed_time_series: training target values as a 1-D NumPy array
    :param holiday_features: one-hot indicator matrix of shape
        ``(n_steps, n_holidays)``
    :param config: dictionary with ``model`` sub-dict containing
        ``num_seasons`` and ``num_steps_per_season``
    :return: a ``tfp.sts.Sum`` model combining all components
    """
    import tensorflow_probability as tfp

    trend = tfp.sts.LocalLinearTrend(observed_time_series=observed_time_series)
    day_of_week_effect = tfp.sts.Seasonal(
        num_seasons=config["model"]["num_seasons"],
        num_steps_per_season=config["model"]["num_steps_per_season"],
        observed_time_series=observed_time_series,
        name="day_of_week_effect",
    )
    autoregressive = tfp.sts.Autoregressive(
        order=1,
        observed_time_series=observed_time_series,
        name="autoregressive",
    )
    holiday_effect = tfp.sts.LinearRegression(
        design_matrix=holiday_features,
        name="holiday_effect",
    )
    model = tfp.sts.Sum(
        [trend, day_of_week_effect, autoregressive, holiday_effect],
        observed_time_series=observed_time_series,
    )
    return model


# #############################################################################
# Plotting
# #############################################################################


def plot_forecast(
    x: np.ndarray,
    y: np.ndarray,
    forecast_mean: np.ndarray,
    forecast_scale: np.ndarray,
    forecast_samples: np.ndarray,
    title: str,
    x_locator: Optional[plt.Locator] = None,
    x_formatter: Optional[plt.Formatter] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a forecast distribution against the ground-truth time series.

    The plot shows the full observed series, sample forecast trajectories, the
    forecast mean, and a shaded 2-sigma confidence band.

    :param x: time points for the entire dataset
    :param y: ground-truth values for the full period
    :param forecast_mean: mean of the forecasted distribution
    :param forecast_scale: standard deviation of the forecasted distribution
    :param forecast_samples: array of sample trajectories with shape
        ``(n_samples, n_forecast_steps)``
    :param title: plot title
    :param x_locator: optional Matplotlib x-axis locator
    :param x_formatter: optional Matplotlib x-axis formatter
    :return: tuple ``(fig, ax)``
    """
    colors = sns.color_palette()
    c1, c2 = colors[0], colors[1]
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    num_steps = len(y)
    num_steps_forecast = forecast_mean.shape[-1]
    num_steps_train = num_steps - num_steps_forecast
    ax.plot(x, y, lw=2, color=c1, label="ground truth")
    forecast_steps = x[num_steps_train : num_steps_train + num_steps_forecast]
    ax.plot(forecast_steps, forecast_samples.T, lw=1, color=c2, alpha=0.1)
    ax.plot(
        forecast_steps,
        forecast_mean,
        lw=2,
        ls="--",
        color=c2,
        label="forecast",
    )
    ax.fill_between(
        forecast_steps,
        forecast_mean - 2 * forecast_scale,
        forecast_mean + 2 * forecast_scale,
        color=c2,
        alpha=0.2,
    )
    ymin = min(np.min(forecast_samples), np.min(y))
    ymax = max(np.max(forecast_samples), np.max(y))
    yrange = ymax - ymin
    ax.set_ylim([ymin - yrange * 0.1, ymax + yrange * 0.1])
    ax.set_title(title)
    ax.legend()
    if x_locator is not None:
        ax.xaxis.set_major_locator(x_locator)
        ax.xaxis.set_major_formatter(x_formatter)
        fig.autofmt_xdate()
    return fig, ax


def plot_components(
    dates: np.ndarray,
    component_means_dict: Dict[str, np.ndarray],
    component_stddevs_dict: Dict[str, np.ndarray],
    *,
    x_locator: Optional[plt.Locator] = None,
    x_formatter: Optional[plt.Formatter] = None,
) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """
    Plot posterior contributions of each STS model component.

    Each component is shown in its own subplot with a 2-sigma shaded band.

    :param dates: array of dates for the x-axis
    :param component_means_dict: dict mapping component name to mean array
    :param component_stddevs_dict: dict mapping component name to stddev array
    :param x_locator: optional Matplotlib x-axis locator
    :param x_formatter: optional Matplotlib x-axis formatter
    :return: tuple ``(fig, axes_dict)`` where ``axes_dict`` is an
        ``OrderedDict`` keyed by component name
    """

    colors = sns.color_palette()
    c2 = colors[1]
    axes_dict = collections.OrderedDict()
    num_components = len(component_means_dict)
    fig = plt.figure(figsize=(12, 2.5 * num_components))
    for i, component_name in enumerate(component_means_dict.keys()):
        component_mean = component_means_dict[component_name]
        component_stddev = component_stddevs_dict[component_name]
        ax = fig.add_subplot(num_components, 1, 1 + i)
        ax.plot(dates, component_mean, lw=2)
        ax.fill_between(
            dates,
            component_mean - 2 * component_stddev,
            component_mean + 2 * component_stddev,
            color=c2,
            alpha=0.5,
        )
        ax.set_title(component_name)
        if x_locator is not None:
            ax.xaxis.set_major_locator(x_locator)
            ax.xaxis.set_major_formatter(x_formatter)
        axes_dict[component_name] = ax
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, axes_dict


def df_to_str_simple(df: pd.DataFrame, num_rows: int = 6) -> str:
    """
    Convert a DataFrame (or Series/Index) to a string for logging.

    Shows `num_rows` from top and bottom if DataFrame is large.
    """
    if df is None:
        return ""
    # Convert Series or Index to DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame()
    elif isinstance(df, pd.Index):
        df = df.to_frame(index=False)

    # Limit rows if needed
    if num_rows is not None and len(df) > num_rows:
        top = df.head(num_rows // 2)
        bottom = df.tail(num_rows // 2)
        df_to_print = pd.concat([top, bottom])
    else:
        df_to_print = df

    return df_to_print.to_string()
