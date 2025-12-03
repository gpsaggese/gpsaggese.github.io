"""
Utility functions for energy consumption forecasting with Darts.

This file contains reusable functions for data preprocessing, feature
engineering, model setup, evaluation, and visualization that support the
tutorial and project notebooks.

- Notebooks should call these functions instead of writing raw logic inline.
- This helps keep the notebooks clean, modular, and easier to debug.
"""

import logging
from typing import List, Dict, Tuple, Optional, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, rmse, mae, smape
from darts.utils.utils import ModelMode


# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data Loading and Preprocessing
# -----------------------------------------------------------------------------

def load_energy_data(file_path: str) -> pd.DataFrame:
    """
    Load the PJME hourly energy consumption dataset.

    :param file_path: path to the CSV file
    :return: DataFrame with datetime index and energy consumption values
    """
    _LOG.info("Loading energy data from %s", file_path)
    df = pd.read_csv(file_path)
    # Parse datetime and set as index.
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime')
    df = df.sort_index()
    # Rename column for clarity.
    df = df.rename(columns={'PJME_MW': 'energy_consumption'})
    _LOG.info("Loaded %d records from %s to %s", len(df), df.index.min(), df.index.max())
    return df


def handle_missing_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing timestamps in the time series using interpolation.

    :param df: DataFrame with datetime index
    :return: DataFrame with missing timestamps filled
    """
    _LOG.info("Checking for missing timestamps")
    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H')
    missing_dates = date_range.difference(df.index)
    df = df[~df.index.duplicated(keep='first')]
    if len(missing_dates) > 0:
        _LOG.info("Found %d missing timestamps, filling with interpolation", len(missing_dates))
        df = df.reindex(date_range)
        df = df.interpolate(method='time')
    else:
        _LOG.info("No missing timestamps found")
    return df


def create_darts_series(df: pd.DataFrame, value_col: str = 'energy_consumption') -> TimeSeries:
    """
    Convert a pandas DataFrame to a Darts TimeSeries object.

    :param df: DataFrame with datetime index
    :param value_col: name of the column containing the target values
    :return: Darts TimeSeries object
    """
    _LOG.info("Creating Darts TimeSeries from DataFrame")
    series = TimeSeries.from_dataframe(
        df.reset_index(),
        time_col='index',
        value_cols=value_col,
        freq='H'
    )
    _LOG.info("Created TimeSeries with %d observations", len(series))
    return series


# -----------------------------------------------------------------------------
# Feature Engineering
# -----------------------------------------------------------------------------

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features for energy forecasting.

    Features include: hour, day of week, month, quarter, year, day of year,
    is_weekend, is_peak_hour, and season.

    :param df: DataFrame with datetime index
    :return: DataFrame with additional temporal features
    """
    _LOG.info("Creating temporal features")
    df = df.copy()
    # Basic temporal features.
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    # Binary features.
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    df['is_peak_hour'] = ((df.index.hour >= 7) & (df.index.hour <= 22)).astype(int)
    # Season encoding.
    df['season'] = df.index.month.map({
        12: 0, 1: 0, 2: 0,    # Winter
        3: 1, 4: 1, 5: 1,     # Spring
        6: 2, 7: 2, 8: 2,     # Summer
        9: 3, 10: 3, 11: 3    # Fall
    })
    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = 'energy_consumption',
    lags: List[int] = None
) -> pd.DataFrame:
    """
    Add lagged values of the target variable as features.

    :param df: DataFrame with datetime index
    :param target_col: column to create lags for
    :param lags: list of lag periods in hours (default: [1, 2, 3, 24, 48, 168])
    :return: DataFrame with lag features added
    """
    if lags is None:
        lags = [1, 2, 3, 24, 48, 168]
    _LOG.info("Adding lag features: %s", lags)
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}h'] = df[target_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = 'energy_consumption',
    windows: List[int] = None
) -> pd.DataFrame:
    """
    Add rolling statistics as features.

    Computes rolling mean, std, min, and max for each window size.

    :param df: DataFrame with datetime index
    :param target_col: column to compute rolling stats for
    :param windows: list of window sizes in hours (default: [24, 48, 168])
    :return: DataFrame with rolling features added
    """
    if windows is None:
        windows = [24, 48, 168]
    _LOG.info("Adding rolling features with windows: %s", windows)
    df = df.copy()
    for window in windows:
        shifted = df[target_col].shift(1)
        df[f'rolling_mean_{window}h'] = shifted.rolling(window=window).mean()
        df[f'rolling_std_{window}h'] = shifted.rolling(window=window).std()
        df[f'rolling_min_{window}h'] = shifted.rolling(window=window).min()
        df[f'rolling_max_{window}h'] = shifted.rolling(window=window).max()
    return df


# -----------------------------------------------------------------------------
# Model Evaluation
# -----------------------------------------------------------------------------

def evaluate_forecast(
    actual: TimeSeries,
    predicted: TimeSeries,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Evaluate forecast performance using multiple metrics.

    :param actual: actual TimeSeries values
    :param predicted: predicted TimeSeries values
    :param model_name: name of the model for logging
    :return: dictionary containing MAPE, RMSE, MAE, SMAPE metrics
    """
    # Ensure predictions match actual length.
    pred_values = predicted[:len(actual)]
    results = {
        'MAPE': mape(actual, pred_values),
        'RMSE': rmse(actual, pred_values),
        'MAE': mae(actual, pred_values),
        'SMAPE': smape(actual, pred_values)
    }
    _LOG.info("%s Results - MAPE: %.2f%%, RMSE: %.2f, MAE: %.2f, SMAPE: %.2f%%",
              model_name, results['MAPE'], results['RMSE'],
              results['MAE'], results['SMAPE'])
    return results


def compare_models(model_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a comparison summary of all models.

    :param model_results: dictionary mapping model names to their results
    :return: DataFrame with model comparison sorted by MAPE
    """
    _LOG.info("Creating model comparison summary")
    summary_data = []
    for model_name, results in model_results.items():
        summary_data.append({
            'Model': model_name,
            'MAPE (%)': round(results['MAPE'], 2),
            'RMSE (MW)': int(round(results['RMSE'])),
            'MAE (MW)': int(round(results['MAE'])),
            'SMAPE (%)': round(results['SMAPE'], 2)
        })
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('MAPE (%)').reset_index(drop=True)
    return summary_df


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def plot_time_series(
    series: TimeSeries,
    title: str = "Time Series",
    figsize: Tuple[int, int] = (16, 6)
) -> None:
    """
    Plot a Darts TimeSeries.

    :param series: TimeSeries to plot
    :param title: plot title
    :param figsize: figure size tuple
    :return: None
    """
    fig, ax = plt.subplots(figsize=figsize)
    series.plot(ax=ax, label='Energy Consumption')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Energy (MW)')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_predictions_vs_actual(
    actual: TimeSeries,
    predictions: Dict[str, TimeSeries],
    title: str = "Model Predictions vs Actual",
    figsize: Tuple[int, int] = (16, 6)
) -> None:
    """
    Plot predictions from multiple models against actual values.

    :param actual: actual TimeSeries values
    :param predictions: dictionary mapping model names to predicted TimeSeries
    :param title: plot title
    :param figsize: figure size tuple
    :return: None
    """
    fig, ax = plt.subplots(figsize=figsize)
    actual.plot(ax=ax, label='Actual', color='black', linewidth=2)
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))
    for (model_name, pred), color in zip(predictions.items(), colors):
        pred.plot(ax=ax, label=model_name, color=color, linewidth=1.5, alpha=0.8)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Energy (MW)')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_seasonality_analysis(df: pd.DataFrame, target_col: str = 'energy_consumption') -> None:
    """
    Plot seasonality patterns in the energy data.

    Creates 4 subplots showing average consumption by hour, day of week,
    month, and year.

    :param df: DataFrame with datetime index
    :param target_col: name of the target column
    :return: None
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    # Hourly pattern.
    hourly_avg = df.groupby(df.index.hour)[target_col].mean()
    axes[0, 0].bar(hourly_avg.index, hourly_avg.values, color='steelblue', alpha=0.8)
    axes[0, 0].set_title('Average Consumption by Hour', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Hour')
    axes[0, 0].set_ylabel('Average Energy (MW)')
    # Daily pattern.
    daily_avg = df.groupby(df.index.dayofweek)[target_col].mean()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[0, 1].bar(days, daily_avg.values, color='coral', alpha=0.8)
    axes[0, 1].set_title('Average Consumption by Day of Week', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Day')
    axes[0, 1].set_ylabel('Average Energy (MW)')
    # Monthly pattern.
    monthly_avg = df.groupby(df.index.month)[target_col].mean()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    axes[1, 0].bar(months, monthly_avg.values, color='seagreen', alpha=0.8)
    axes[1, 0].set_title('Average Consumption by Month', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Average Energy (MW)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    # Yearly pattern.
    yearly_avg = df.groupby(df.index.year)[target_col].mean()
    axes[1, 1].plot(yearly_avg.index, yearly_avg.values, marker='o',
                    linewidth=2, markersize=8, color='purple')
    axes[1, 1].set_title('Average Consumption by Year', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Average Energy (MW)')
    axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_error_analysis(
    actual: TimeSeries,
    predicted: TimeSeries,
    figsize: Tuple[int, int] = (16, 12)
) -> None:
    """
    Plot error analysis for model predictions.

    Creates 4 subplots: error distribution, actual vs predicted scatter,
    error by hour, and error by day of week.

    :param actual: actual TimeSeries values
    :param predicted: predicted TimeSeries values
    :param figsize: figure size tuple
    :return: None
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    actual_values = actual.univariate_values().flatten()
    pred_values = predicted[:len(actual)].univariate_values().flatten()
    errors = actual_values - pred_values
    # Error distribution.
    axes[0, 0].hist(errors, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_title('Prediction Error Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Error (MW)')
    axes[0, 0].set_ylabel('Frequency')
    # Actual vs Predicted scatter.
    axes[0, 1].scatter(actual_values, pred_values, alpha=0.5, s=10)
    axes[0, 1].plot([actual_values.min(), actual_values.max()],
                    [actual_values.min(), actual_values.max()],
                    'r--', linewidth=2, label='Perfect Prediction')
    axes[0, 1].set_title('Actual vs Predicted', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Actual Energy (MW)')
    axes[0, 1].set_ylabel('Predicted Energy (MW)')
    axes[0, 1].legend()
    # Error by hour.
    error_by_hour = pd.DataFrame({
        'error': errors,
        'hour': actual.time_index.hour
    }).groupby('hour')['error'].agg(['mean', 'std'])
    axes[1, 0].bar(error_by_hour.index, error_by_hour['mean'],
                   yerr=error_by_hour['std'], color='coral', alpha=0.7, capsize=3)
    axes[1, 0].axhline(0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 0].set_title('Mean Error by Hour', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Hour')
    axes[1, 0].set_ylabel('Mean Error (MW)')
    # Error by day of week.
    error_by_day = pd.DataFrame({
        'error': errors,
        'day': actual.time_index.dayofweek
    }).groupby('day')['error'].agg(['mean', 'std'])
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[1, 1].bar(days, error_by_day['mean'].values,
                   yerr=error_by_day['std'].values, color='seagreen', alpha=0.7, capsize=3)
    axes[1, 1].axhline(0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 1].set_title('Mean Error by Day of Week', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Day')
    axes[1, 1].set_ylabel('Mean Error (MW)')
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Train/Test Split
# -----------------------------------------------------------------------------

def train_test_split_series(
    series: TimeSeries,
    test_size: int = 720
) -> Tuple[TimeSeries, TimeSeries]:
    """
    Split a TimeSeries into training and test sets.

    :param series: TimeSeries to split
    :param test_size: number of observations for test set (default: 720 = 30 days)
    :return: tuple of (train_series, test_series)
    """
    _LOG.info("Splitting series into train/test with test_size=%d", test_size)
    train = series[:-test_size]
    test = series[-test_size:]
    _LOG.info("Train size: %d, Test size: %d", len(train), len(test))
    return train, test


def scale_series(
    train: TimeSeries,
    test: TimeSeries
) -> Tuple[TimeSeries, TimeSeries, Scaler]:
    """
    Scale training and test series using StandardScaler.

    :param train: training TimeSeries
    :param test: test TimeSeries
    :return: tuple of (scaled_train, scaled_test, scaler)
    """
    _LOG.info("Scaling time series data")
    scaler = Scaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    return train_scaled, test_scaled, scaler

