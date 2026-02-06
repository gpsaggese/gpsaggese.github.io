"""
Evaluation Metrics Utilities.

Common evaluation metrics for time series forecasting.

Import as:

import tutorials.tutorial_GluonTS_COVID19_Prediction.GluonTS_utils_evaluation as ttgcpguev
"""

import logging
from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_LOG = logging.getLogger(__name__)


def calculate_metrics(
    forecast_values: Union[np.ndarray, pd.Series, list],
    actual_values: Union[np.ndarray, pd.Series, list],
) -> Dict[str, float]:
    """
    Calculate comprehensive forecasting metrics.

    :param forecast_values: Forecasted values
    :param actual_values: Actual observed values
    :return: Dictionary with MAE, RMSE, MAPE, ME, and max_error
    """
    # Convert to numpy arrays.
    forecast_values = np.asarray(forecast_values).flatten()
    actual_values = np.asarray(actual_values).flatten()
    # Ensure same length.
    if len(forecast_values) != len(actual_values):
        min_len = min(len(forecast_values), len(actual_values))
        forecast_values = forecast_values[:min_len]
        actual_values = actual_values[:min_len]
    errors = forecast_values - actual_values
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    mape = np.mean(np.abs(errors / actual_values)) * 100
    me = np.mean(errors)
    max_error = np.max(np.abs(errors))
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "me": me,
        "max_error": max_error,
    }


def print_metrics(
    metrics: Dict[str, float],
    *,
    model_name: str = "Model",
) -> None:
    """
    Print metrics in a formatted way.

    :param metrics: Dictionary with metric values
    :param model_name: Name of the model
    """
    _LOG.info("\n%s Performance:", model_name)
    _LOG.info("=" * 60)
    _LOG.info("MAE (Mean Absolute Error):      %10,.2f", metrics["mae"])
    _LOG.info("RMSE (Root Mean Squared Error): %10,.2f", metrics["rmse"])
    _LOG.info("MAPE (Mean Abs. %% Error):       %10.2f %%", metrics["mape"])
    _LOG.info("ME (Mean Error / Bias):         %10,.2f", metrics["me"])
    _LOG.info("Maximum Error:                   %10,.2f", metrics["max_error"])
    _LOG.info("=" * 60)
    # Interpretation.
    if metrics["mape"] < 10:
        _LOG.info("\nExcellent performance, error less than 10%%")
    elif metrics["mape"] < 20:
        _LOG.info("\nGood performance, error less than 20%%")
    else:
        _LOG.info("\nModerate performance (COVID data is highly variable)")
    if abs(metrics["me"]) < metrics["mae"] / 2:
        _LOG.info("Low bias (not systematically over or under-predicting)")
    else:
        bias_direction = "over" if metrics["me"] > 0 else "under"
        _LOG.info("Model tends to %s-predict", bias_direction)


def plot_forecast(
    train_df: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    forecast_values: np.ndarray,
    actual_values: np.ndarray,
    forecast_quantiles: Dict[float, np.ndarray],
    target_column: str,
    model_name: str,
    *,
    save_path: str = None,
    context_days: int = 60,
) -> None:
    """
    Create a comprehensive forecast visualization.

    :param train_df: Training data DataFrame
    :param forecast_dates: Dates for forecast period
    :param forecast_values: Forecasted values
    :param actual_values: Actual values
    :param forecast_quantiles: Dictionary of quantile forecasts
    :param target_column: Name of target column
    :param model_name: Name of the model
    :param save_path: Optional path to save plot
    :param context_days: Number of historical days to show
    """
    plt.figure(figsize=(16, 6))
    # Historical context.
    train_context = train_df.tail(context_days)
    plt.plot(
        train_context["Date"],
        train_context[target_column],
        label="Historical Data",
        color="steelblue",
        linewidth=2,
        alpha=0.8,
    )
    # Actual future values.
    plt.plot(
        forecast_dates,
        actual_values,
        label="Actual",
        color="orange",
        linewidth=3,
        marker="o",
        markersize=8,
        zorder=5,
    )
    # Forecast.
    plt.plot(
        forecast_dates,
        forecast_values,
        label=f"{model_name} Forecast",
        color="red",
        linewidth=3,
        marker="s",
        markersize=7,
        linestyle="--",
        zorder=4,
    )
    # Confidence intervals.
    if 0.05 in forecast_quantiles and 0.95 in forecast_quantiles:
        plt.fill_between(
            forecast_dates,
            forecast_quantiles[0.05],
            forecast_quantiles[0.95],
            alpha=0.15,
            color="red",
            label="90% Confidence",
        )
    if 0.25 in forecast_quantiles and 0.75 in forecast_quantiles:
        plt.fill_between(
            forecast_dates,
            forecast_quantiles[0.25],
            forecast_quantiles[0.75],
            alpha=0.25,
            color="red",
            label="50% Confidence",
        )
    plt.title(
        f"{model_name} Forecast Visualization",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Date", fontsize=13)
    plt.ylabel(target_column.replace("_", " "), fontsize=13)
    plt.legend(loc="best", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        _LOG.info("Plot saved as '%s'", save_path)
    plt.show()


def plot_error_analysis(
    forecast_values: np.ndarray,
    actual_values: np.ndarray,
    forecast_quantiles: Dict[float, np.ndarray],
    model_name: str,
    *,
    save_path: str = None,
) -> None:
    """
    Create detailed error analysis plots.

    :param forecast_values: Forecasted values
    :param actual_values: Actual values
    :param forecast_quantiles: Dictionary of quantile forecasts
    :param model_name: Name of the model
    :param save_path: Optional path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    forecast_period = len(forecast_values)
    errors = forecast_values - actual_values
    # Plot 1: Forecast vs Actual.
    axes[0, 0].plot(
        range(1, forecast_period + 1),
        actual_values,
        "o-",
        label="Actual",
        color="orange",
        linewidth=2,
        markersize=8,
    )
    axes[0, 0].plot(
        range(1, forecast_period + 1),
        forecast_values,
        "s--",
        label="Forecast",
        color="red",
        linewidth=2,
        markersize=7,
    )
    if 0.1 in forecast_quantiles and 0.9 in forecast_quantiles:
        axes[0, 0].fill_between(
            range(1, forecast_period + 1),
            forecast_quantiles[0.1],
            forecast_quantiles[0.9],
            alpha=0.2,
            color="red",
        )
    axes[0, 0].set_title("Forecast vs Actual", fontweight="bold")
    axes[0, 0].set_xlabel("Day")
    axes[0, 0].set_ylabel("Value")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    # Plot 2: Forecast Errors.
    colors = ["red" if e > 0 else "green" for e in errors]
    axes[0, 1].bar(range(1, forecast_period + 1), errors, color=colors)
    axes[0, 1].axhline(y=0, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Daily Forecast Errors", fontweight="bold")
    axes[0, 1].set_xlabel("Day")
    axes[0, 1].set_ylabel("Error (Forecast - Actual)")
    axes[0, 1].grid(True, alpha=0.3, axis="y")
    # Plot 3: Absolute Percentage Errors.
    ape = np.abs(errors / actual_values) * 100
    mape = np.mean(ape)
    axes[1, 0].bar(
        range(1, forecast_period + 1),
        ape,
        color="steelblue",
        alpha=0.7,
    )
    axes[1, 0].axhline(
        y=mape,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean APE: {mape:.1f}%",
    )
    axes[1, 0].set_title("Absolute Percentage Error by Day", fontweight="bold")
    axes[1, 0].set_xlabel("Day")
    axes[1, 0].set_ylabel("Absolute % Error")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis="y")
    # Plot 4: Uncertainty Width.
    if 0.1 in forecast_quantiles and 0.9 in forecast_quantiles:
        ci_width = forecast_quantiles[0.9] - forecast_quantiles[0.1]
        axes[1, 1].plot(
            range(1, forecast_period + 1),
            ci_width,
            "o-",
            color="purple",
            linewidth=2,
            markersize=8,
        )
        axes[1, 1].set_title(
            "Forecast Uncertainty (80% CI Width)",
            fontweight="bold",
        )
        axes[1, 1].set_xlabel("Day")
        axes[1, 1].set_ylabel("CI Width")
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "Quantiles not available",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].set_title("Uncertainty Analysis", fontweight="bold")
    plt.suptitle(
        f"{model_name} Error Analysis",
        fontsize=16,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        _LOG.info("Error analysis saved as '%s'", save_path)
    plt.show()


def compare_models(
    results: Dict[str, Dict[str, float]],
    *,
    save_path: str = None,
) -> None:
    """
    Compare multiple models side by side.

    :param results: Dictionary mapping model names to their metrics
    :param save_path: Optional path to save plot
    """
    metrics_to_plot = ["mae", "rmse", "mape"]
    model_names = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, metric in enumerate(metrics_to_plot):
        values = [results[model][metric] for model in model_names]
        axes[idx].bar(
            model_names,
            values,
            color=["steelblue", "green", "purple"][: len(model_names)],
        )
        axes[idx].set_title(metric.upper(), fontweight="bold")
        axes[idx].set_ylabel(metric.upper())
        axes[idx].grid(True, alpha=0.3, axis="y")
        # Add value labels.
        for i, v in enumerate(values):
            axes[idx].text(i, v, f"{v:.1f}", ha="center", va="bottom")
    plt.suptitle("Model Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        _LOG.info("Comparison saved as '%s'", save_path)
    plt.show()
    # Print table.
    _LOG.info("\nModel Comparison Table:")
    _LOG.info("=" * 70)
    _LOG.info("%-20s %12s %12s %12s", "Model", "MAE", "RMSE", "MAPE")
    _LOG.info("-" * 70)
    for model, metrics in results.items():
        _LOG.info(
            "%-20s %12,.2f %12,.2f %11.2f%%",
            model,
            metrics["mae"],
            metrics["rmse"],
            metrics["mape"],
        )
    _LOG.info("=" * 70)
    # Find best model.
    best_model = min(results.items(), key=lambda x: x[1]["mape"])
    _LOG.info(
        "\nBest Model (by MAPE): %s (%.2f%%)",
        best_model[0],
        best_model[1]["mape"],
    )
