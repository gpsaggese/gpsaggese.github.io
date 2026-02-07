"""
Model Training Utilities for GluonTS COVID-19 Forecasting.

Wrapper functions for training DeepAR, SimpleFeedForward, and DeepNPTS models.
Includes scenario analysis utilities for "what-if" policy simulations.

Import as:

import tutorials.tutorial_GluonTS_COVID19_Prediction.utils_models as ttgcpgumo
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.torch.model.deep_npts import DeepNPTSEstimator
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.model.simple_feedforward import SimpleFeedForwardEstimator

_LOG = logging.getLogger(__name__)

# #############################################################################
# ModelResults
# #############################################################################

@dataclass
class ModelResults:
    """
    Container for model training and evaluation results.

    :param model_name: name of the model
    :param predictor: trained GluonTS predictor object
    :param forecasts: list of forecast objects
    :param ground_truths: list of ground truth time series
    :param metrics: dictionary of evaluation metrics
    :param training_time: training time in seconds
    """

    model_name: str
    predictor: object
    forecasts: List
    ground_truths: List
    metrics: Dict
    training_time: float = 0.0


def train_deepar_covid(
    train_ds,
    test_ds,
    *,
    prediction_length: int = 14,
    num_feat_dynamic_real: int = 0,
    epochs: int = 20,
    learning_rate: float = 0.001,
    context_length: Optional[int] = None,
    num_layers: int = 2,
    hidden_size: int = 40,
    dropout: float = 0.1,
    verbose: bool = True,
) -> ModelResults:
    """
    Train a DeepAR model on COVID-19 data.

    DeepAR uses recurrent neural networks to capture complex temporal patterns.

    :param train_ds: training dataset
    :param test_ds: test dataset
    :param prediction_length: number of time steps to predict
    :param num_feat_dynamic_real: number of dynamic real features
    :param epochs: number of training epochs
    :param learning_rate: learning rate for optimizer
    :param context_length: number of time steps to use as context
    :param num_layers: number of RNN layers
    :param hidden_size: hidden layer size
    :param dropout: dropout rate
    :param verbose: whether to print progress information
    :return: ModelResults containing trained model and evaluation metrics
    """
    if verbose:
        _LOG.info("\n" + "=" * 70)
        _LOG.info("TRAINING DeepAR MODEL")
        _LOG.info("=" * 70)
        _LOG.info("\nConfiguration:")
        _LOG.info("  Epochs: %s", epochs)
        _LOG.info(
            "  Context length: %s", context_length or prediction_length * 2
        )
        _LOG.info("  Features: %s", num_feat_dynamic_real)
        _LOG.info("  Hidden size: %s", hidden_size)
        _LOG.info("  Layers: %s", num_layers)
    # Start timer.
    start_time = time.time()
    # Create estimator.
    estimator = DeepAREstimator(
        freq="D",
        prediction_length=prediction_length,
        context_length=context_length or (prediction_length * 2),
        num_feat_dynamic_real=num_feat_dynamic_real,
        num_layers=num_layers,
        hidden_size=hidden_size,
        dropout_rate=dropout,
        lr=learning_rate,
        batch_size=32,
        num_batches_per_epoch=50,
        trainer_kwargs={"max_epochs": epochs},
    )
    # Train model.
    if verbose:
        _LOG.info("\nTraining in progress...")
    predictor = estimator.train(train_ds)
    training_time = time.time() - start_time
    if verbose:
        _LOG.info("\nTraining complete in %.1f seconds", training_time)
    # Generate forecasts.
    if verbose:
        _LOG.info("\nGenerating probabilistic forecasts...")
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds, predictor=predictor, num_samples=100
    )
    forecasts = list(forecast_it)
    ground_truths = list(ts_it)
    # Calculate metrics.
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, _ = evaluator(iter(ground_truths), iter(forecasts))
    if verbose:
        _LOG.info("\nDeepAR Performance:")
        _LOG.info("  MAPE: %.2f%%", agg_metrics.get("MAPE", 0))
        _LOG.info("  RMSE: %.2f", agg_metrics.get("RMSE", 0))
        _LOG.info("  MAE: %.2f", agg_metrics.get("MAE", 0))
        _LOG.info("=" * 70)
    # Return results.
    return ModelResults(
        model_name="DeepAR",
        predictor=predictor,
        forecasts=forecasts,
        ground_truths=ground_truths,
        metrics=agg_metrics,
        training_time=training_time,
    )


def train_feedforward_covid(
    train_ds,
    test_ds,
    *,
    prediction_length: int = 14,
    epochs: int = 100,
    learning_rate: float = 0.001,
    context_length: Optional[int] = None,
    hidden_dimensions: Optional[List[int]] = None,
    verbose: bool = True,
) -> ModelResults:
    """
    Train a SimpleFeedForward model on COVID-19 data.

    SimpleFeedForward is a fast baseline using a simple neural network.
    Note: This model doesn't support external features.

    :param train_ds: training dataset
    :param test_ds: test dataset
    :param prediction_length: number of time steps to predict
    :param epochs: number of training epochs
    :param learning_rate: learning rate for optimizer
    :param context_length: number of time steps to use as context
    :param hidden_dimensions: list of hidden layer dimensions
    :param verbose: whether to print progress information
    :return: ModelResults containing trained model and evaluation metrics
    """
    if verbose:
        _LOG.info("\n" + "=" * 70)
        _LOG.info("TRAINING SimpleFeedForward MODEL")
        _LOG.info("=" * 70)
        _LOG.info("\nNote: This model doesn't use external features.")
        _LOG.info("\nConfiguration:")
        _LOG.info("  Epochs: %s", epochs)
        _LOG.info(
            "  Context length: %s", context_length or prediction_length * 2
        )
        _LOG.info("  Hidden layers: %s", hidden_dimensions or [40, 40])
    # Start timer.
    start_time = time.time()
    # Create estimator.
    estimator = SimpleFeedForwardEstimator(
        prediction_length=prediction_length,
        context_length=context_length or (prediction_length * 2),
        hidden_dimensions=hidden_dimensions or [40, 40],
        lr=learning_rate,
        batch_size=32,
        num_batches_per_epoch=50,
        trainer_kwargs={"max_epochs": epochs},
    )
    # Train model.
    if verbose:
        _LOG.info("\nTraining in progress...")
    predictor = estimator.train(train_ds)
    training_time = time.time() - start_time
    if verbose:
        _LOG.info("\nTraining complete in %.1f seconds", training_time)
    # Generate forecasts.
    if verbose:
        _LOG.info("\nGenerating probabilistic forecasts...")
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds, predictor=predictor, num_samples=100
    )
    forecasts = list(forecast_it)
    ground_truths = list(ts_it)
    # Calculate metrics.
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, _ = evaluator(iter(ground_truths), iter(forecasts))
    if verbose:
        _LOG.info("\nSimpleFeedForward Performance:")
        _LOG.info("  MAPE: %.2f%%", agg_metrics.get("MAPE", 0))
        _LOG.info("  RMSE: %.2f", agg_metrics.get("RMSE", 0))
        _LOG.info("  MAE: %.2f", agg_metrics.get("MAE", 0))
        _LOG.info("=" * 70)
    # Return results.
    return ModelResults(
        model_name="SimpleFeedForward",
        predictor=predictor,
        forecasts=forecasts,
        ground_truths=ground_truths,
        metrics=agg_metrics,
        training_time=training_time,
    )


def train_deepnpts_covid(
    train_ds,
    test_ds,
    *,
    prediction_length: int = 14,
    num_feat_dynamic_real: int = 0,
    epochs: int = 30,
    learning_rate: float = 0.001,
    context_length: Optional[int] = None,
    num_hidden_nodes: Optional[List[int]] = None,
    dropout_rate: float = 0.1,
    verbose: bool = True,
) -> ModelResults:
    """
    Train a DeepNPTS model on COVID-19 data.

    DeepNPTS (Deep Non-Parametric Time Series) doesn't assume data
    follows a specific distribution. Great for regime changes.

    Note: DeepNPTS accepts 'epochs' as a direct parameter.

    :param train_ds: training dataset
    :param test_ds: test dataset
    :param prediction_length: number of time steps to predict
    :param num_feat_dynamic_real: number of dynamic real features
    :param epochs: number of training epochs
    :param learning_rate: learning rate for optimizer
    :param context_length: number of time steps to use as context
    :param num_hidden_nodes: list of hidden node sizes
    :param dropout_rate: dropout rate
    :param verbose: whether to print progress information
    :return: ModelResults containing trained model and evaluation metrics
    """
    if verbose:
        _LOG.info("\n" + "=" * 70)
        _LOG.info("TRAINING DeepNPTS MODEL")
        _LOG.info("=" * 70)
        _LOG.info("\nConfiguration:")
        _LOG.info("  Epochs: %s", epochs)
        _LOG.info(
            "  Context length: %s", context_length or prediction_length * 2
        )
        _LOG.info("  Features: %s", num_feat_dynamic_real)
        _LOG.info("  Hidden nodes: %s", num_hidden_nodes or [40])
        _LOG.info("  Dropout: %s", dropout_rate)
    # Start timer.
    start_time = time.time()
    # Create estimator.
    estimator = DeepNPTSEstimator(
        freq="D",
        prediction_length=prediction_length,
        context_length=context_length or (prediction_length * 2),
        num_feat_dynamic_real=num_feat_dynamic_real,
        num_hidden_nodes=num_hidden_nodes or [40],
        dropout_rate=dropout_rate,
        epochs=epochs,  # DeepNPTS accepts epochs directly.
        lr=learning_rate,
        batch_size=32,
        num_batches_per_epoch=50,
    )
    # Train model.
    if verbose:
        _LOG.info("\nTraining in progress...")
    predictor = estimator.train(train_ds)
    training_time = time.time() - start_time
    if verbose:
        _LOG.info("\nTraining complete in %.1f seconds", training_time)
    # Generate forecasts.
    if verbose:
        _LOG.info("\nGenerating probabilistic forecasts...")
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds, predictor=predictor, num_samples=100
    )
    forecasts = list(forecast_it)
    ground_truths = list(ts_it)
    # Calculate metrics.
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, _ = evaluator(iter(ground_truths), iter(forecasts))
    if verbose:
        _LOG.info("\nDeepNPTS Performance:")
        _LOG.info("  MAPE: %.2f%%", agg_metrics.get("MAPE", 0))
        _LOG.info("  RMSE: %.2f", agg_metrics.get("RMSE", 0))
        _LOG.info("  MAE: %.2f", agg_metrics.get("MAE", 0))
        _LOG.info("=" * 70)
    # Return results.
    return ModelResults(
        model_name="DeepNPTS",
        predictor=predictor,
        forecasts=forecasts,
        ground_truths=ground_truths,
        metrics=agg_metrics,
        training_time=training_time,
    )


def compare_models(results_list: List[ModelResults]) -> pd.DataFrame:
    """
    Create a comparison table of multiple trained models.

    :param results_list: list of ModelResults to compare
    :return: DataFrame with model comparison metrics
    """
    comparison_data = []
    # Build comparison data.
    for results in results_list:
        comparison_data.append(
            {
                "Model": results.model_name,
                "MAPE (%)": results.metrics.get("MAPE", np.nan),
                "RMSE": results.metrics.get("RMSE", np.nan),
                "MAE": results.metrics.get("MAE", np.nan),
                "Training Time (s)": results.training_time,
            }
        )
    # Create and sort DataFrame.
    df = pd.DataFrame(comparison_data)
    df = df.sort_values("MAPE (%)")
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def print_model_comparison(comparison_df: pd.DataFrame) -> None:
    """
    Pretty print the model comparison table.

    :param comparison_df: DataFrame with model comparison metrics
    """
    _LOG.info("\n" + "=" * 80)
    _LOG.info("MODEL COMPARISON - COVID-19 FORECASTING")
    _LOG.info("=" * 80)
    _LOG.info("\nWhich model performed best?\n")
    _LOG.info(comparison_df.to_string(index=False))
    _LOG.info("\n" + "=" * 80)
    # Print winner.
    winner = comparison_df.iloc[0]
    _LOG.info(
        "\nWinner: %s with MAPE of %.2f%%", winner["Model"], winner["MAPE (%)"]
    )
    _LOG.info("=" * 80 + "\n")


def get_forecast_dataframe(
    forecast,
    ground_truth,
    start_date: pd.Timestamp,
    *,
    freq: str = "D",
) -> pd.DataFrame:
    """
    Convert GluonTS forecast and ground truth to a convenient DataFrame.

    :param forecast: GluonTS forecast object
    :param ground_truth: ground truth time series
    :param start_date: start date for forecast period
    :param freq: frequency string for date range
    :return: DataFrame with forecast predictions and quantiles
    """
    forecast_length = len(forecast.mean)
    dates = pd.date_range(start=start_date, periods=forecast_length, freq=freq)
    # Build forecast DataFrame.
    df = pd.DataFrame(
        {
            "Date": dates,
            "Prediction": forecast.mean,
            "Actual": ground_truth[-forecast_length:],
            "Lower_10": forecast.quantile(0.1),
            "Lower_25": forecast.quantile(0.25),
            "Median": forecast.quantile(0.5),
            "Upper_75": forecast.quantile(0.75),
            "Upper_90": forecast.quantile(0.9),
        }
    )
    return df

# #############################################################################
# ScenarioResult
# #############################################################################

@dataclass
class ScenarioResult:
    """
    Container for scenario analysis results.

    :param name: scenario name
    :param description: human-readable description
    :param forecast: GluonTS forecast object
    :param mean_daily_cases: mean daily cases in forecast
    :param total_cases: total cases over forecast period
    :param lower_bound: 10th percentile average
    :param upper_bound: 90th percentile average
    :param adjustments: dictionary of applied adjustments
    """

    name: str
    description: str
    forecast: Any  # GluonTS forecast object.
    mean_daily_cases: float
    total_cases: float
    lower_bound: float  # 10th percentile average.
    upper_bound: float  # 90th percentile average.
    adjustments: Dict[str, float] = field(default_factory=dict)

    def cases_vs_baseline(self, baseline_total: float) -> Tuple[float, float]:
        """
        Calculate difference from baseline scenario.

        :param baseline_total: total cases in baseline scenario
        :return: tuple of (absolute difference, percentage difference)
        """
        diff = self.total_cases - baseline_total
        pct_diff = (diff / baseline_total) * 100 if baseline_total != 0 else 0
        return diff, pct_diff


def create_scenario_dataset(
    merged_df: pd.DataFrame,
    feature_columns: List[str],
    *,
    target_column: str = "Daily_Cases_MA7",
    mobility_adjustment: float = 1.0,
    cfr_adjustment: float = 1.0,
    deaths_adjustment: float = 1.0,
    prediction_length: int = 14,
    freq: str = "D",
) -> ListDataset:
    """
    Create a modified GluonTS dataset for scenario analysis.

    This function takes the original merged DataFrame and applies adjustments
    to specific features to simulate different scenarios (e.g., lockdowns,
    reopenings, healthcare strain).

    :param merged_df: original merged DataFrame with all features
    :param feature_columns: list of feature column names used in the model
    :param target_column: name of the target column
    :param mobility_adjustment: factor to multiply mobility features by
                                (0.7 = 30% reduction, 1.2 = 20% increase)
    :param cfr_adjustment: factor to multiply CFR by
                           (1.15 = 15% increase in case fatality)
    :param deaths_adjustment: factor to multiply deaths features by
    :param prediction_length: forecast horizon
    :param freq: data frequency
    :return: modified GluonTS ListDataset ready for forecasting
    """
    # Work with a copy to avoid modifying original data.
    df = merged_df.copy()
    # Identify which feature columns to adjust.
    mobility_cols = [
        "retail and recreation",
        "grocery and pharmacy",
        "parks",
        "transit stations",
        "workplaces",
        "residential",
    ]
    cfr_cols = ["CFR"]
    deaths_cols = ["Daily_Deaths_MA7", "Cumulative_Deaths", "Daily_Deaths"]
    # Apply adjustments only to the forecast period (last prediction_length days).
    # For training data, we keep original values.
    forecast_start_idx = len(df) - prediction_length
    for col in df.columns:
        if col in mobility_cols and mobility_adjustment != 1.0:
            df.loc[df.index[forecast_start_idx:], col] = (
                df.loc[df.index[forecast_start_idx:], col] * mobility_adjustment
            )
        elif col in cfr_cols and cfr_adjustment != 1.0:
            df.loc[df.index[forecast_start_idx:], col] = (
                df.loc[df.index[forecast_start_idx:], col] * cfr_adjustment
            )
        elif col in deaths_cols and deaths_adjustment != 1.0:
            df.loc[df.index[forecast_start_idx:], col] = (
                df.loc[df.index[forecast_start_idx:], col] * deaths_adjustment
            )
    # Drop rows with NaN in target.
    df_clean = df.dropna(subset=[target_column]).copy()
    # Build the dataset.
    date_col = "Date" if "Date" in df_clean.columns else "date"
    start_date = pd.to_datetime(df_clean[date_col].iloc[0])
    target = df_clean[target_column].values.tolist()
    data_entry = {"start": start_date, "target": target}
    # Add features if specified.
    if feature_columns:
        feat_dynamic_real = []
        for col in feature_columns:
            if col in df_clean.columns:
                feat_dynamic_real.append(df_clean[col].values.tolist())
        if feat_dynamic_real:
            data_entry["feat_dynamic_real"] = feat_dynamic_real
    return ListDataset([data_entry], freq=freq)


def run_scenario_forecast(
    predictor,
    scenario_dataset: ListDataset,
    scenario_name: str,
    scenario_description: str,
    adjustments: Dict[str, float],
    *,
    num_samples: int = 100,
) -> ScenarioResult:
    """
    Run a forecast for a specific scenario using a trained predictor.

    :param predictor: trained GluonTS predictor
    :param scenario_dataset: modified dataset for this scenario
    :param scenario_name: short name for the scenario
    :param scenario_description: human-readable description
    :param adjustments: dictionary of adjustments applied (for record-keeping)
    :param num_samples: number of forecast samples to generate
    :return: ScenarioResult with forecast statistics
    """
    # Generate forecast.
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=scenario_dataset, predictor=predictor, num_samples=num_samples
    )
    forecasts = list(forecast_it)
    forecast = forecasts[0]
    # Calculate statistics.
    mean_daily = float(forecast.mean.mean())
    total_cases = float(forecast.mean.sum())
    lower = float(forecast.quantile(0.1).mean())
    upper = float(forecast.quantile(0.9).mean())
    # Return results.
    return ScenarioResult(
        name=scenario_name,
        description=scenario_description,
        forecast=forecast,
        mean_daily_cases=mean_daily,
        total_cases=total_cases,
        lower_bound=lower,
        upper_bound=upper,
        adjustments=adjustments,
    )


def run_all_scenarios(
    predictor,
    merged_df: pd.DataFrame,
    feature_columns: List[str],
    *,
    target_column: str = "Daily_Cases_MA7",
    prediction_length: int = 14,
    verbose: bool = True,
) -> List[ScenarioResult]:
    """
    Run all predefined scenarios and return results.

    Scenarios:
    1. Baseline - No changes
    2. Moderate Intervention - 15% mobility reduction
    3. Strong Intervention - 30% mobility reduction
    4. Relaxation/Reopening - 20% mobility increase
    5. Healthcare Strain - 15% CFR increase (hospital stress indicator)

    :param predictor: trained GluonTS predictor
    :param merged_df: original merged DataFrame
    :param feature_columns: feature columns used by the model
    :param target_column: target column name
    :param prediction_length: forecast horizon
    :param verbose: print progress
    :return: list of ScenarioResult objects
    """
    scenarios_config = [
        {
            "name": "Baseline",
            "description": "No intervention - current trends continue",
            "mobility": 1.0,
            "cfr": 1.0,
            "deaths": 1.0,
        },
        {
            "name": "Moderate Intervention",
            "description": "15% mobility reduction (masks, capacity limits)",
            "mobility": 0.85,
            "cfr": 1.0,
            "deaths": 1.0,
        },
        {
            "name": "Strong Intervention",
            "description": "30% mobility reduction (lockdowns, closures)",
            "mobility": 0.70,
            "cfr": 1.0,
            "deaths": 1.0,
        },
        {
            "name": "Relaxation",
            "description": "20% mobility increase (reopening, holidays)",
            "mobility": 1.20,
            "cfr": 1.0,
            "deaths": 1.0,
        },
        {
            "name": "Healthcare Strain",
            "description": "15% higher CFR (hospital capacity stressed)",
            "mobility": 1.0,
            "cfr": 1.15,
            "deaths": 1.10,
        },
    ]
    results = []
    if verbose:
        _LOG.info("\n" + "=" * 70)
        _LOG.info("RUNNING SCENARIO ANALYSIS")
        _LOG.info("=" * 70)
    # Run each scenario.
    for i, config in enumerate(scenarios_config, 1):
        if verbose:
            _LOG.info(
                "\n[%s/5] %s: %s", i, config["name"], config["description"]
            )
        # Create modified dataset.
        scenario_ds = create_scenario_dataset(
            merged_df=merged_df,
            feature_columns=feature_columns,
            target_column=target_column,
            mobility_adjustment=config["mobility"],
            cfr_adjustment=config["cfr"],
            deaths_adjustment=config["deaths"],
            prediction_length=prediction_length,
        )
        # Run forecast.
        adjustments = {
            "mobility": config["mobility"],
            "cfr": config["cfr"],
            "deaths": config["deaths"],
        }
        result = run_scenario_forecast(
            predictor=predictor,
            scenario_dataset=scenario_ds,
            scenario_name=config["name"],
            scenario_description=config["description"],
            adjustments=adjustments,
        )
        results.append(result)
        if verbose:
            _LOG.info(
                "   Avg daily: %,.0f | Total: %,.0f",
                result.mean_daily_cases,
                result.total_cases,
            )
    if verbose:
        _LOG.info("\nScenario analysis complete.")
    return results


def print_scenario_summary(results: List[ScenarioResult]) -> pd.DataFrame:
    """
    Print a formatted summary table of all scenario results.

    :param results: list of ScenarioResult objects
    :return: DataFrame with summary statistics
    """
    # Get baseline for comparison.
    baseline = next((r for r in results if r.name == "Baseline"), results[0])
    baseline_total = baseline.total_cases
    # Build summary data.
    summary_data = []
    for result in results:
        diff, pct = result.cases_vs_baseline(baseline_total)
        summary_data.append(
            {
                "Scenario": result.name,
                "Avg Daily Cases": result.mean_daily_cases,
                "Total Cases (14d)": result.total_cases,
                "Range (10%-90%)": f"{result.lower_bound:,.0f} - {result.upper_bound:,.0f}",
                "vs Baseline": f"{pct:+.1f}%"
                if result.name != "Baseline"
                else "--",
                "Cases Δ": f"{diff:+,.0f}"
                if result.name != "Baseline"
                else "--",
            }
        )
    df = pd.DataFrame(summary_data)
    # Print formatted table.
    _LOG.info("\n" + "=" * 90)
    _LOG.info("SCENARIO COMPARISON SUMMARY")
    _LOG.info("=" * 90)
    _LOG.info("\nForecast horizon: 14 days")
    _LOG.info("Baseline total cases: %,.0f", baseline_total)
    _LOG.info("")
    # Print header.
    _LOG.info(
        "%s %s %s %s %s",
        "Scenario".ljust(25),
        "Avg Daily".ljust(12),
        "Total Cases".ljust(14),
        "vs Baseline".ljust(12),
        "Cases Δ".ljust(15),
    )
    _LOG.info("-" * 90)
    # Print rows.
    for _, row in df.iterrows():
        _LOG.info(
            "%s %10,.0f %12,.0f %10s %14s",
            row["Scenario"].ljust(25),
            row["Avg Daily Cases"],
            row["Total Cases (14d)"],
            row["vs Baseline"].rjust(10),
            row["Cases Δ"].rjust(14),
        )
    _LOG.info("=" * 90)
    return df


def plot_scenario_comparison(
    results: List[ScenarioResult],
    *,
    prediction_length: int = 14,
    save_path: str = None,
) -> None:
    """
    Create visualizations comparing all scenarios.

    :param results: list of ScenarioResult objects
    :param prediction_length: forecast horizon (for x-axis)
    :param save_path: optional path to save the figure
    """
    import matplotlib.pyplot as plt

    # Define colors for each scenario.
    colors = {
        "Baseline": "#6B7280",  # Gray.
        "Moderate Intervention": "#3B82F6",  # Blue.
        "Strong Intervention": "#10B981",  # Green.
        "Relaxation": "#F59E0B",  # Orange.
        "Healthcare Strain": "#EF4444",  # Red.
    }
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # ===== Plot 1: Forecast Trajectories =====
    ax1 = axes[0]
    days = list(range(1, prediction_length + 1))
    for result in results:
        color = colors.get(result.name, "#6B7280")
        forecast = result.forecast
        # Plot mean.
        ax1.plot(
            days, forecast.mean, label=result.name, color=color, linewidth=2.5
        )
        # Plot confidence interval (lighter).
        ax1.fill_between(
            days,
            forecast.quantile(0.1),
            forecast.quantile(0.9),
            alpha=0.15,
            color=color,
        )
    ax1.set_xlabel("Days Ahead", fontsize=12)
    ax1.set_ylabel("Daily Cases", fontsize=12)
    ax1.set_title(
        "Forecast Trajectories by Scenario", fontsize=14, fontweight="bold"
    )
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, prediction_length + 1, 2))
    # ===== Plot 2: Total Cases Bar Chart =====
    ax2 = axes[1]
    names = [r.name for r in results]
    totals = [r.total_cases for r in results]
    bar_colors = [colors.get(name, "#6B7280") for name in names]
    bars = ax2.barh(names, totals, color=bar_colors, alpha=0.8)
    # Add value labels.
    baseline_total = next(
        (r.total_cases for r in results if r.name == "Baseline"), totals[0]
    )
    for bar, result in zip(bars, results):
        width = bar.get_width()
        diff, pct = result.cases_vs_baseline(baseline_total)
        # Main value.
        ax2.text(
            width + baseline_total * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:,.0f}",
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=10,
        )
        # Percentage change (if not baseline).
        if result.name != "Baseline":
            pct_text = f"({pct:+.1f}%)"
            ax2.text(
                width + baseline_total * 0.08,
                bar.get_y() + bar.get_height() / 2,
                pct_text,
                ha="left",
                va="center",
                fontsize=9,
                color="green" if pct < 0 else "red",
            )
    ax2.set_xlabel("Total Cases (14 days)", fontsize=12)
    ax2.set_title("Total Cases by Scenario", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")
    # Add baseline reference line.
    ax2.axvline(
        x=baseline_total,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Baseline",
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        _LOG.info("Saved scenario comparison plot to: %s", save_path)
    plt.show()
