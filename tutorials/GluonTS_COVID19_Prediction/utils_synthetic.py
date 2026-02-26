"""
Synthetic Time Series Data Utilities.
Generate reproducible synthetic time series for teaching GluonTS fundamentals.
Each generator returns a pandas DataFrame with `Date` and `value` columns,
compatible with `utils_gluonts.create_gluonts_dataset`.
"""

import logging
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset

_LOG = logging.getLogger(__name__)

_DEFAULT_START = "2020-01-01"



def generate_sinusoid(
    n_points: int = 365,
    *,
    period: int = 30,
    amplitude: float = 10.0,
    baseline: float = 50.0,
    noise_std: float = 1.0,
    seed: int = 42,
    start_date: str = _DEFAULT_START,
) -> pd.DataFrame:
    """
    Pure sine wave with additive Gaussian noise.

    :param n_points: length of the series
    :param period: cycle length in days
    :param amplitude: peak deviation from baseline
    :param baseline: vertical offset so values stay positive
    :param noise_std: standard deviation of Gaussian noise
    :param seed: RNG seed for reproducibility
    :param start_date: first date in the series
    :return: DataFrame with Date and value columns
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_points)
    signal = baseline + amplitude * np.sin(2 * np.pi * t / period)
    noise = rng.normal(0, noise_std, n_points)
    dates = pd.date_range(start=start_date, periods=n_points, freq="D")
    return pd.DataFrame({"Date": dates, "value": signal + noise})


def generate_multi_frequency(
    n_points: int = 365,
    *,
    trend_slope: float = 0.02,
    seasonal_period: int = 30,
    seasonal_amplitude: float = 8.0,
    weekly_amplitude: float = 3.0,
    baseline: float = 50.0,
    noise_std: float = 1.5,
    seed: int = 42,
    start_date: str = _DEFAULT_START,
) -> pd.DataFrame:
    """
    Combination of linear trend, 30-day seasonal cycle, 7-day weekly cycle,
    and Gaussian noise.

    :param n_points: length of the series
    :param trend_slope: daily increase added by the trend component
    :param seasonal_period: dominant seasonal cycle length in days
    :param seasonal_amplitude: amplitude of the seasonal component
    :param weekly_amplitude: amplitude of the 7-day component
    :param baseline: vertical offset
    :param noise_std: standard deviation of Gaussian noise
    :param seed: RNG seed for reproducibility
    :param start_date: first date in the series
    :return: DataFrame with Date and value columns
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_points)
    trend = baseline + trend_slope * t
    seasonal = seasonal_amplitude * np.sin(2 * np.pi * t / seasonal_period)
    weekly = weekly_amplitude * np.sin(2 * np.pi * t / 7)
    noise = rng.normal(0, noise_std, n_points)
    dates = pd.date_range(start=start_date, periods=n_points, freq="D")
    return pd.DataFrame({"Date": dates, "value": trend + seasonal + weekly + noise})


def generate_regime_change(
    n_points: int = 365,
    *,
    changepoint_frac: float = 0.5,
    baseline_before: float = 50.0,
    amplitude_before: float = 5.0,
    period_before: int = 30,
    baseline_after: float = 80.0,
    amplitude_after: float = 12.0,
    period_after: int = 15,
    noise_std: float = 1.5,
    seed: int = 42,
    start_date: str = _DEFAULT_START,
) -> pd.DataFrame:
    """
    Time series that changes behavior at a configurable changepoint.

    Before the changepoint: low-amplitude sinusoid around a lower baseline.
    After the changepoint: higher amplitude, different frequency, level shift.

    :param n_points: length of the series
    :param changepoint_frac: fraction of series where the regime changes (0-1)
    :param baseline_before: baseline before changepoint
    :param amplitude_before: amplitude before changepoint
    :param period_before: cycle length before changepoint
    :param baseline_after: baseline after changepoint
    :param amplitude_after: amplitude after changepoint
    :param period_after: cycle length after changepoint
    :param noise_std: standard deviation of Gaussian noise
    :param seed: RNG seed for reproducibility
    :param start_date: first date in the series
    :return: DataFrame with Date and value columns
    """
    rng = np.random.default_rng(seed)
    cp = int(n_points * changepoint_frac)
    t_before = np.arange(cp)
    t_after = np.arange(n_points - cp)
    before = baseline_before + amplitude_before * np.sin(
        2 * np.pi * t_before / period_before
    )
    after = baseline_after + amplitude_after * np.sin(
        2 * np.pi * t_after / period_after
    )
    signal = np.concatenate([before, after])
    noise = rng.normal(0, noise_std, n_points)
    dates = pd.date_range(start=start_date, periods=n_points, freq="D")
    return pd.DataFrame({"Date": dates, "value": signal + noise})



def prepare_synthetic_dataset(
    df: pd.DataFrame,
    *,
    target_col: str = "value",
    prediction_length: int = 30,
    freq: str = "D",
) -> Dict:
    """
    Split a synthetic DataFrame into train/test and convert to GluonTS
    ListDataset format.

    The test dataset follows GluonTS convention: it contains the *full*
    series (train + test) so that `make_evaluation_predictions` can
    use the last `prediction_length` points as ground truth.

    :param df: DataFrame with Date and target columns
    :param target_col: name of the target column
    :param prediction_length: forecast horizon (also used as test size)
    :param freq: time series frequency
    :return: dict with train_ds, test_ds, train_df, test_df, and metadata
    """
    date_col = "Date" if "Date" in df.columns else "date"
    df = df.copy().sort_values(date_col).reset_index(drop=True)
    split_idx = len(df) - prediction_length
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    start = pd.to_datetime(df[date_col].iloc[0])
    train_ds = ListDataset(
        [{"start": start, "target": train_df[target_col].values}],
        freq=freq,
    )
    test_ds = ListDataset(
        [{"start": start, "target": df[target_col].values}],
        freq=freq,
    )
    info = {
        "n_points": len(df),
        "train_points": len(train_df),
        "test_points": len(test_df),
        "prediction_length": prediction_length,
        "start_date": str(start.date()),
        "train_end": str(train_df[date_col].iloc[-1].date()),
        "test_start": str(test_df[date_col].iloc[0].date()),
        "test_end": str(test_df[date_col].iloc[-1].date()),
    }
    _LOG.info("Prepared synthetic dataset:")
    _LOG.info("  Train: %s points (%s to %s)",
              info["train_points"], info["start_date"], info["train_end"])
    _LOG.info("  Test:  %s points (%s to %s)",
              info["test_points"], info["test_start"], info["test_end"])
    return {
        "train_ds": train_ds,
        "test_ds": test_ds,
        "train_df": train_df,
        "test_df": test_df,
        "target": target_col,
        "prediction_length": prediction_length,
        "info": info,
    }



def plot_synthetic_series(
    df: pd.DataFrame,
    *,
    target_col: str = "value",
    title: str = "Synthetic Time Series",
    figsize: tuple = (14, 4),
) -> None:
    """
    Quick visualization of a synthetic series.

    :param df: DataFrame with Date and target columns
    :param target_col: name of the target column
    :param title: plot title
    :param figsize: figure dimensions
    """
    date_col = "Date" if "Date" in df.columns else "date"
    plt.figure(figsize=figsize)
    plt.plot(df[date_col], df[target_col], linewidth=1.2, color="steelblue")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_train_test_split(
    data: Dict,
    *,
    title: str = "Train / Test Split",
    figsize: tuple = (14, 4),
) -> None:
    """
    Visualize the train/test split produced by `prepare_synthetic_dataset`.

    :param data: dict returned by prepare_synthetic_dataset
    :param title: plot title
    :param figsize: figure dimensions
    """
    target_col = data["target"]
    date_col = "Date" if "Date" in data["train_df"].columns else "date"
    plt.figure(figsize=figsize)
    plt.plot(
        data["train_df"][date_col],
        data["train_df"][target_col],
        label="Train",
        color="steelblue",
        linewidth=1.2,
    )
    plt.plot(
        data["test_df"][date_col],
        data["test_df"][target_col],
        label="Test",
        color="orange",
        linewidth=1.2,
    )
    plt.axvline(
        x=data["test_df"][date_col].iloc[0],
        color="red",
        linestyle="--",
        alpha=0.7,
        label="Split point",
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_forecast_result(
    data: Dict,
    forecast_entry,
    *,
    model_name: str = "Model",
    context_points: int = 60,
    figsize: tuple = (14, 5),
) -> None:
    """
    Plot forecast against actuals with confidence intervals.

    :param data: dict returned by prepare_synthetic_dataset
    :param forecast_entry: single GluonTS SampleForecast object
    :param model_name: label for the legend
    :param context_points: how many historical points to show
    :param figsize: figure dimensions
    """
    target_col = data["target"]
    date_col = "Date" if "Date" in data["train_df"].columns else "date"
    train_tail = data["train_df"].tail(context_points)
    test_dates = data["test_df"][date_col].values
    actuals = data["test_df"][target_col].values
    pred_mean = forecast_entry.mean
    plt.figure(figsize=figsize)
    plt.plot(
        train_tail[date_col], train_tail[target_col],
        label="History", color="steelblue", linewidth=1.2,
    )
    plt.plot(
        test_dates, actuals,
        label="Actual", color="orange", linewidth=2, marker="o", markersize=4,
    )
    plt.plot(
        test_dates[:len(pred_mean)], pred_mean,
        label=f"{model_name} forecast", color="red",
        linewidth=2, linestyle="--", marker="s", markersize=4,
    )
    q_low = forecast_entry.quantile(0.1)
    q_high = forecast_entry.quantile(0.9)
    plt.fill_between(
        test_dates[:len(q_low)], q_low, q_high,
        alpha=0.15, color="red", label="80% interval",
    )
    plt.title(f"{model_name} Forecast", fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()