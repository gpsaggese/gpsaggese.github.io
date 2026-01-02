"""
Model Training Utilities for GluonTS COVID-19 Forecasting

Wrapper functions for training DeepAR, SimpleFeedForward, and DeepNPTS models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.torch.model.deep_npts import DeepNPTSEstimator
from gluonts.evaluation import make_evaluation_predictions, Evaluator


@dataclass
class ModelResults:
    """Container for model training and evaluation results."""
    model_name: str
    predictor: object
    forecasts: List
    ground_truths: List
    metrics: Dict
    training_time: float = 0.0


def train_deepar_covid(
    train_ds,
    test_ds,
    prediction_length: int = 14,
    num_feat_dynamic_real: int = 0,
    epochs: int = 20,
    learning_rate: float = 0.001,
    context_length: Optional[int] = None,
    num_layers: int = 2,
    hidden_size: int = 40,
    dropout: float = 0.1,
    verbose: bool = True
) -> ModelResults:
    """
    Train a DeepAR model on COVID-19 data.
    
    DeepAR uses recurrent neural networks to capture complex temporal patterns.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("TRAINING DeepAR MODEL")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Epochs: {epochs}")
        print(f"  Context length: {context_length or prediction_length * 2}")
        print(f"  Features: {num_feat_dynamic_real}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Layers: {num_layers}")
    
    import time
    start_time = time.time()
    
    estimator = DeepAREstimator(
        freq='D',
        prediction_length=prediction_length,
        context_length=context_length or (prediction_length * 2),
        num_feat_dynamic_real=num_feat_dynamic_real,
        num_layers=num_layers,
        hidden_size=hidden_size,
        dropout_rate=dropout,
        lr=learning_rate,
        batch_size=32,
        num_batches_per_epoch=50,
        trainer_kwargs={"max_epochs": epochs}
    )
    
    if verbose:
        print("\nTraining in progress...")
    
    predictor = estimator.train(train_ds)
    training_time = time.time() - start_time
    
    if verbose:
        print(f"\nTraining complete in {training_time:.1f} seconds")
    
    # Generate forecasts
    if verbose:
        print("\nGenerating probabilistic forecasts...")
    
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,
        predictor=predictor,
        num_samples=100
    )
    
    forecasts = list(forecast_it)
    ground_truths = list(ts_it)
    
    # Calculate metrics
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, _ = evaluator(iter(ground_truths), iter(forecasts))
    
    if verbose:
        print(f"\nDeepAR Performance:")
        print(f"  MAPE: {agg_metrics.get('MAPE', 0):.2f}%")
        print(f"  RMSE: {agg_metrics.get('RMSE', 0):.2f}")
        print(f"  MAE: {agg_metrics.get('MAE', 0):.2f}")
        print("=" * 70)
    
    return ModelResults(
        model_name="DeepAR",
        predictor=predictor,
        forecasts=forecasts,
        ground_truths=ground_truths,
        metrics=agg_metrics,
        training_time=training_time
    )


def train_feedforward_covid(
    train_ds,
    test_ds,
    prediction_length: int = 14,
    epochs: int = 100,
    learning_rate: float = 0.001,
    context_length: Optional[int] = None,
    hidden_dimensions: Optional[List[int]] = None,
    verbose: bool = True
) -> ModelResults:
    """
    Train a SimpleFeedForward model on COVID-19 data.
    
    SimpleFeedForward is a fast baseline using a simple neural network.
    Note: This model doesn't support external features.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("TRAINING SimpleFeedForward MODEL")
        print("=" * 70)
        print("\nNote: This model doesn't use external features.")
        print(f"\nConfiguration:")
        print(f"  Epochs: {epochs}")
        print(f"  Context length: {context_length or prediction_length * 2}")
        print(f"  Hidden layers: {hidden_dimensions or [40, 40]}")
    
    import time
    start_time = time.time()
    
    estimator = SimpleFeedForwardEstimator(
        prediction_length=prediction_length,
        context_length=context_length or (prediction_length * 2),
        hidden_dimensions=hidden_dimensions or [40, 40],
        lr=learning_rate,
        batch_size=32,
        num_batches_per_epoch=50,
        trainer_kwargs={"max_epochs": epochs}
    )
    
    if verbose:
        print("\nTraining in progress...")
    
    predictor = estimator.train(train_ds)
    training_time = time.time() - start_time
    
    if verbose:
        print(f"\nTraining complete in {training_time:.1f} seconds")
    
    # Generate forecasts
    if verbose:
        print("\nGenerating probabilistic forecasts...")
    
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,
        predictor=predictor,
        num_samples=100
    )
    
    forecasts = list(forecast_it)
    ground_truths = list(ts_it)
    
    # Calculate metrics
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, _ = evaluator(iter(ground_truths), iter(forecasts))
    
    if verbose:
        print(f"\nSimpleFeedForward Performance:")
        print(f"  MAPE: {agg_metrics.get('MAPE', 0):.2f}%")
        print(f"  RMSE: {agg_metrics.get('RMSE', 0):.2f}")
        print(f"  MAE: {agg_metrics.get('MAE', 0):.2f}")
        print("=" * 70)
    
    return ModelResults(
        model_name="SimpleFeedForward",
        predictor=predictor,
        forecasts=forecasts,
        ground_truths=ground_truths,
        metrics=agg_metrics,
        training_time=training_time
    )


def train_deepnpts_covid(
    train_ds,
    test_ds,
    prediction_length: int = 14,
    num_feat_dynamic_real: int = 0,
    epochs: int = 30,
    learning_rate: float = 0.001,
    context_length: Optional[int] = None,
    num_hidden_nodes: Optional[List[int]] = None,
    dropout_rate: float = 0.1,
    verbose: bool = True
) -> ModelResults:
    """
    Train a DeepNPTS model on COVID-19 data.
    
    DeepNPTS (Deep Non-Parametric Time Series) doesn't assume data
    follows a specific distribution. Great for regime changes.
    
    Note: DeepNPTS accepts 'epochs' as a direct parameter.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("TRAINING DeepNPTS MODEL")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Epochs: {epochs}")
        print(f"  Context length: {context_length or prediction_length * 2}")
        print(f"  Features: {num_feat_dynamic_real}")
        print(f"  Hidden nodes: {num_hidden_nodes or [40]}")
        print(f"  Dropout: {dropout_rate}")
    
    import time
    start_time = time.time()
    
    estimator = DeepNPTSEstimator(
        freq='D',
        prediction_length=prediction_length,
        context_length=context_length or (prediction_length * 2),
        num_feat_dynamic_real=num_feat_dynamic_real,
        num_hidden_nodes=num_hidden_nodes or [40],
        dropout_rate=dropout_rate,
        epochs=epochs,  # DeepNPTS accepts epochs directly
        lr=learning_rate,
        batch_size=32,
        num_batches_per_epoch=50
    )
    
    if verbose:
        print("\nTraining in progress...")
    
    predictor = estimator.train(train_ds)
    training_time = time.time() - start_time
    
    if verbose:
        print(f"\nTraining complete in {training_time:.1f} seconds")
    
    # Generate forecasts
    if verbose:
        print("\nGenerating probabilistic forecasts...")
    
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,
        predictor=predictor,
        num_samples=100
    )
    
    forecasts = list(forecast_it)
    ground_truths = list(ts_it)
    
    # Calculate metrics
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, _ = evaluator(iter(ground_truths), iter(forecasts))
    
    if verbose:
        print(f"\nDeepNPTS Performance:")
        print(f"  MAPE: {agg_metrics.get('MAPE', 0):.2f}%")
        print(f"  RMSE: {agg_metrics.get('RMSE', 0):.2f}")
        print(f"  MAE: {agg_metrics.get('MAE', 0):.2f}")
        print("=" * 70)
    
    return ModelResults(
        model_name="DeepNPTS",
        predictor=predictor,
        forecasts=forecasts,
        ground_truths=ground_truths,
        metrics=agg_metrics,
        training_time=training_time
    )


def compare_models(results_list: List[ModelResults]) -> pd.DataFrame:
    """Create a comparison table of multiple trained models."""
    comparison_data = []
    
    for results in results_list:
        comparison_data.append({
            'Model': results.model_name,
            'MAPE (%)': results.metrics.get('MAPE', np.nan),
            'RMSE': results.metrics.get('RMSE', np.nan),
            'MAE': results.metrics.get('MAE', np.nan),
            'Training Time (s)': results.training_time,
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('MAPE (%)')
    df.insert(0, 'Rank', range(1, len(df) + 1))
    
    return df


def print_model_comparison(comparison_df: pd.DataFrame) -> None:
    """Pretty print the model comparison table."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON - COVID-19 FORECASTING")
    print("=" * 80)
    print("\nWhich model performed best?\n")
    print(comparison_df.to_string(index=False))
    print("\n" + "=" * 80)
    
    winner = comparison_df.iloc[0]
    print(f"\nWinner: {winner['Model']} with MAPE of {winner['MAPE (%)']:.2f}%")
    print("=" * 80 + "\n")


def get_forecast_dataframe(
    forecast,
    ground_truth,
    start_date: pd.Timestamp,
    freq: str = 'D'
) -> pd.DataFrame:
    """Convert GluonTS forecast and ground truth to a convenient DataFrame."""
    forecast_length = len(forecast.mean)
    dates = pd.date_range(start=start_date, periods=forecast_length, freq=freq)
    
    df = pd.DataFrame({
        'Date': dates,
        'Prediction': forecast.mean,
        'Actual': ground_truth[-forecast_length:],
        'Lower_10': forecast.quantile(0.1),
        'Lower_25': forecast.quantile(0.25),
        'Median': forecast.quantile(0.5),
        'Upper_75': forecast.quantile(0.75),
        'Upper_90': forecast.quantile(0.9)
    })
    
    return df
