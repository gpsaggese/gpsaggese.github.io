"""
Visualization utilities for GluonTS COVID-19 forecasting notebooks.

Centralizes matplotlib plotting to keep notebooks focused on learning content.
Import as:

import utils_visualization as viz
"""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# #############################################################################
# Data exploration visualizations
# #############################################################################


def plot_data_overview(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    *,
    date_col: str = "Date",
    title: str = "US COVID-19 Cases: 7-Day Moving Average",
    ylabel: str = "Daily Cases (7-day avg)",
) -> None:
    """
    Plot training and test data with forecast boundary line.

    :param train_df: training DataFrame
    :param test_df: test DataFrame
    :param target_col: name of target column to plot
    :param date_col: name of date column
    :param title: plot title
    :param ylabel: y-axis label
    """
    plt.figure(figsize=(14, 5))
    plt.plot(
        train_df[date_col],
        train_df[target_col],
        label="Training Data",
        color="steelblue",
        linewidth=1.5,
    )
    plt.plot(
        test_df[date_col],
        test_df[target_col],
        label="Test Data (Future)",
        color="coral",
        linewidth=1.5,
    )
    plt.axvline(
        x=train_df[date_col].iloc[-1],
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label="Today (Forecast Start)",
    )
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print("\n Note that there are multiple peaks and troughs in the case data.")
    print(
        "\n This makes sense as there were multiple rounds of vaccinations and Covid variants."
    )


# #############################################################################
# Model forecast visualizations
# #############################################################################


def plot_forecast_with_confidence_intervals(
    train_df: pd.DataFrame,
    data: Dict[str, Any],
    actual: np.ndarray,
    forecast: Any,
    model_name: str,
    *,
    color: str = "forestgreen",
    context_days: int = 90,
    prediction_length: int = 14,
    ylabel: str = "Daily Cases (7-day avg)",
) -> tuple:
    """
    Plot forecast with historical context and confidence intervals.

    :param train_df: training DataFrame
    :param data: data dict with 'target' key for target column name
    :param actual: ground truth values (full series)
    :param forecast: GluonTS forecast object (has .mean, .quantile())
    :param model_name: model name for legend (e.g. "DeepAR", "SimpleFeedForward")
    :param color: color for forecast line and bands
    :param context_days: number of historical days to show
    :param prediction_length: forecast horizon
    :param ylabel: y-axis label
    :return: (train_dates, train_values, forecast_dates, actual_values) for reuse
    """
    target_col = data["target"]
    train_dates = train_df["Date"].values[-context_days:]
    train_values = train_df[target_col].values[-context_days:]
    last_train_date = pd.Timestamp(train_dates[-1])
    forecast_dates = pd.date_range(
        start=last_train_date + pd.Timedelta(days=1),
        periods=prediction_length,
        freq="D",
    )
    actual_values = actual[-prediction_length:]
    plt.figure(figsize=(14, 6))
    # Historical data.
    plt.plot(
        train_dates, train_values, label="Historical", color="steelblue", linewidth=2
    )
    # Actual future.
    plt.plot(
        forecast_dates,
        actual_values,
        label="Actual Future",
        color="coral",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    # Model prediction.
    plt.plot(
        forecast_dates,
        forecast.mean,
        label=f"{model_name} Forecast",
        color=color,
        linewidth=2.5,
        marker="s",
        markersize=5,
        linestyle="--",
    )
    # Confidence intervals.
    plt.fill_between(
        forecast_dates,
        forecast.quantile(0.1),
        forecast.quantile(0.9),
        alpha=0.3,
        color=color,
        label="80% Confidence",
    )
    plt.fill_between(
        forecast_dates,
        forecast.quantile(0.05),
        forecast.quantile(0.95),
        alpha=0.2,
        color=color,
        label="90% Confidence",
    )
    plt.axvline(
        x=last_train_date, color="red", linestyle="--", linewidth=1.5, alpha=0.7
    )
    plt.title(f"{model_name}: COVID-19 Case Forecasting", fontsize=16, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    _print_forecast_insight(model_name)
    return train_dates, train_values, forecast_dates, actual_values


def _print_forecast_insight(model_name: str) -> None:
    """Print model-specific interpretation."""
    if "DeepAR" in model_name:
        print("\n Observe that DeepAR captures the trend and provides uncertainty bounds.")
    elif "SimpleFeedForward" in model_name:
        print("\n SimpleFeedForward gives a smooth baseline forecast!")
    elif "DeepNPTS" in model_name:
        print("\n DeepNPTS adapts to the data's natural distribution!")


# #############################################################################
# Data exploration (Example notebook)
# #############################################################################


def plot_data_exploration(merged_df: pd.DataFrame) -> None:
    """
    Plot cases, deaths, and mobility in 3-panel layout.

    :param merged_df: merged DataFrame with Date, Daily_Cases_MA7,
        Daily_Deaths_MA7, workplaces columns
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    # Plot 1: Daily Cases.
    axes[0].plot(
        merged_df["Date"],
        merged_df["Daily_Cases_MA7"],
        linewidth=2,
        color="#2E86AB",
    )
    axes[0].set_title(
        " COVID-19 Daily Cases (7-Day Moving Average)",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].set_ylabel("Cases", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    # Plot 2: Daily Deaths.
    axes[1].plot(
        merged_df["Date"],
        merged_df["Daily_Deaths_MA7"],
        linewidth=2,
        color="#A23B72",
    )
    axes[1].set_title(
        " COVID-19 Daily Deaths (7-Day Moving Average)",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].set_ylabel("Deaths", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    # Plot 3: Mobility (workplaces).
    axes[2].plot(
        merged_df["Date"],
        merged_df["workplaces"],
        linewidth=2,
        color="#F18F01",
    )
    axes[2].set_title(
        " Workplace Mobility (% change from baseline)",
        fontsize=14,
        fontweight="bold",
    )
    axes[2].set_ylabel("% Change", fontsize=12)
    axes[2].set_xlabel("Date", fontsize=12)
    axes[2].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print("\n Key Observations:")
    print(" • Multiple distinct waves of cases visible")
    print(" • Deaths follow cases with a lag")
    print(" • Mobility patterns shifted dramatically during lockdowns")
    print(" • These patterns provide valuable signals for forecasting!")


def plot_model_comparison_3panel(
    deepar_results: Any,
    feedforward_results: Any,
    deepnpts_results: Any,
) -> None:
    """
    Plot 3-panel forecast comparison for DeepAR, SimpleFeedForward, DeepNPTS.

    :param deepar_results: ModelResults from train_deepar_covid
    :param feedforward_results: ModelResults from train_feedforward_covid
    :param deepnpts_results: ModelResults from train_deepnpts_covid
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    models = [
        (deepar_results, "DeepAR", "#2E86AB"),
        (feedforward_results, "SimpleFeedForward", "#A23B72"),
        (deepnpts_results, "DeepNPTS", "#F18F01"),
    ]
    for idx, (results, name, color) in enumerate(models):
        ax = axes[idx]
        forecast = results.forecasts[0]
        actual = results.ground_truths[0]
        history_len = len(actual) - len(forecast.mean)
        ax.plot(
            range(history_len),
            actual[:history_len],
            label="Historical",
            color="gray",
            alpha=0.6,
            linewidth=2,
        )
        ax.plot(
            range(history_len, len(actual)),
            actual[history_len:],
            label="Actual",
            color="black",
            linewidth=2,
        )
        forecast_range = range(history_len, history_len + len(forecast.mean))
        ax.plot(
            forecast_range,
            forecast.mean,
            label="Forecast",
            color=color,
            linewidth=2,
            linestyle="--",
        )
        ax.fill_between(
            forecast_range,
            forecast.quantile(0.1),
            forecast.quantile(0.9),
            alpha=0.3,
            color=color,
            label="80% CI",
        )
        ax.set_title(f"{name} Forecast", fontsize=14, fontweight="bold")
        ax.set_ylabel("Daily Cases", fontsize=12)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
    axes[2].set_xlabel("Days", fontsize=12)
    plt.tight_layout()
    plt.show()
    print("\n Visual Insights:")
    print(" • All models capture the general trend")
    print(" • Confidence intervals show forecast uncertainty")
    print(" • Compare forecast accuracy against actual values (black line)")


# #############################################################################
# Model comparison visualizations
# #############################################################################


def print_model_comparison(
    deepar_metrics: Dict[str, float],
    ff_metrics: Dict[str, float],
    npts_metrics: Dict[str, float],
) -> None:
    """
    Print model comparison table and winner.

    :param deepar_metrics: DeepAR metrics dict (mae, rmse, mape)
    :param ff_metrics: SimpleFeedForward metrics dict
    :param npts_metrics: DeepNPTS metrics dict
    """
    comparison = pd.DataFrame(
        [
            {
                "Model": "DeepAR",
                "MAPE (%)": deepar_metrics["mape"],
                "MAE": deepar_metrics["mae"],
                "RMSE": deepar_metrics["rmse"],
            },
            {
                "Model": "SimpleFeedForward",
                "MAPE (%)": ff_metrics["mape"],
                "MAE": ff_metrics["mae"],
                "RMSE": ff_metrics["rmse"],
            },
            {
                "Model": "DeepNPTS",
                "MAPE (%)": npts_metrics["mape"],
                "MAE": npts_metrics["mae"],
                "RMSE": npts_metrics["rmse"],
            },
        ]
    )
    comparison = comparison.sort_values("MAPE (%)")
    comparison.insert(0, "Rank", [1, 2, 3])
    print("\n" + "=" * 70)
    print(" MODEL COMPARISON")
    print("=" * 70)
    print(comparison.to_string(index=False))
    print("=" * 70)
    winner = comparison.iloc[0]["Model"]
    print(f"\n Winner: {winner}!")
    print("\nNote: These results are from a quick training demo.")
    print("Ideally, you should train with more epochs and tune hyperparameters!")


def plot_metrics_comparison_barplot(
    results_dict: Dict[str, Dict[str, float]],
    *,
    metrics: Optional[list] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Bar chart comparing metrics across models.

    :param results_dict: {model_name: {metric_name: value}}
    :param metrics: list of metric keys to plot (default: mae, rmse, mape)
    :param save_path: optional path to save figure
    """
    if metrics is None:
        metrics = ["mae", "rmse", "mape"]
    model_names = list(results_dict.keys())
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    colors = ["steelblue", "green", "purple", "coral", "darkblue"][: len(model_names)]
    for idx, metric in enumerate(metrics):
        values = [results_dict[model].get(metric, 0) for model in model_names]
        axes[idx].bar(model_names, values, color=colors)
        axes[idx].set_title(metric.upper(), fontweight="bold")
        axes[idx].set_ylabel(metric.upper())
        axes[idx].grid(True, alpha=0.3, axis="y")
        for i, v in enumerate(values):
            axes[idx].text(i, v, f"{v:.1f}", ha="center", va="bottom")
    plt.suptitle("Model Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
