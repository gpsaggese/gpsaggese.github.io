"""
Energy Consumption Forecasting using Darts library.

This script implements an energy consumption forecasting system for the PJM East
region using multiple time series models. It demonstrates how to apply the Darts
API for real-world forecasting tasks.

Project Objective:
- Forecast energy consumption based on historical usage patterns
- Compare multiple models to find the best fit for multi-step forecasts
- Optimize model hyperparameters using grid search

Dataset:
- PJME Hourly Energy Consumption from Kaggle
- https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption

Citations:
- Darts Library: https://unit8co.github.io/darts/
- Herzen et al. (2022). "Darts: User-Friendly Modern Machine Learning
  for Time Series" JMLR 23(124):1−6

"""

import logging
from typing import Dict, List, Optional, Tuple

# Local utility functions.
import darts_utils as utils
import numpy as np
import pandas as pd

# Darts imports.
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae, mape, rmse, smape
from darts.models import (
    ExponentialSmoothing,
    NaiveSeasonal,
    NBEATSModel,
    Prophet,
    RNNModel,
)
from sklearn.model_selection import ParameterGrid

_LOG = logging.getLogger(__name__)


# #############################################################################
# Configuration
# #############################################################################


class ForecastConfig:
    """
    Configuration settings for the energy forecasting pipeline.
    """

    def __init__(self):
        """Initialize default configuration parameters."""
        # Data paths.
        self.data_path = "data/PJME_hourly.csv"
        # Train/test split.
        self.test_size = 24 * 30  # 30 days for testing.
        self.forecast_horizon = 24 * 7  # 7 days forecast.
        # Use subset of data for faster training.
        self.years_of_data = 3
        # Random seed.
        self.random_state = 42


# #############################################################################
# Data Pipeline
# #############################################################################


class DataPipeline:
    """
    Handle data loading, preprocessing, and feature engineering.
    """

    def __init__(self, config: ForecastConfig):
        """
        Initialize the data pipeline with configuration.

        :param config: ForecastConfig object with settings
        """
        self.config = config
        self.df = None
        self.series = None
        self.scaler = Scaler()

    def load_data(self) -> pd.DataFrame:
        """
        Load and preprocess the PJME energy consumption dataset.

        :return: preprocessed DataFrame with datetime index
        """
        _LOG.info("Loading data from %s", self.config.data_path)
        # Load raw data.
        self.df = utils.load_energy_data(self.config.data_path)
        # Handle missing timestamps.
        self.df = utils.handle_missing_timestamps(self.df)
        return self.df

    def create_time_series(self) -> TimeSeries:
        """
        Convert DataFrame to Darts TimeSeries.

        :return: Darts TimeSeries object
        """
        self.series = utils.create_darts_series(self.df)
        # Use subset of data for faster training.
        hours_to_use = 24 * 365 * self.config.years_of_data
        self.series = self.series[-hours_to_use:]
        _LOG.info("Using last %d hours of data", len(self.series))
        return self.series

    def split_data(self) -> Tuple[TimeSeries, TimeSeries]:
        """
        Split time series into training and test sets.

        :return: tuple of (train_series, test_series)
        """
        train, test = utils.train_test_split_series(
            self.series, test_size=self.config.test_size
        )
        return train, test

    def scale_data(
        self, train: TimeSeries, test: TimeSeries
    ) -> Tuple[TimeSeries, TimeSeries]:
        """
        Scale training and test data for neural network models.

        :param train: training TimeSeries
        :param test: test TimeSeries
        :return: tuple of (scaled_train, scaled_test)
        """
        train_scaled = self.scaler.fit_transform(train)
        test_scaled = self.scaler.transform(test)
        return train_scaled, test_scaled

    def inverse_transform(self, series: TimeSeries) -> TimeSeries:
        """
        Inverse transform scaled predictions.

        :param series: scaled TimeSeries
        :return: TimeSeries in original scale
        """
        return self.scaler.inverse_transform(series)


# #############################################################################
# Model Training
# #############################################################################


class ModelTrainer:
    """
    Train and evaluate multiple forecasting models.
    """

    def __init__(self, config: ForecastConfig):
        """
        Initialize the model trainer.

        :param config: ForecastConfig object with settings
        """
        self.config = config
        self.results = {}

    def train_naive_seasonal(self, train: TimeSeries, test: TimeSeries) -> Dict:
        """
        Train and evaluate the Naive Seasonal baseline model.

        :param train: training TimeSeries
        :param test: test TimeSeries
        :return: dictionary with predictions and metrics
        """
        _LOG.info("Training Naive Seasonal model")
        model = NaiveSeasonal(K=168)  # Weekly seasonality.
        model.fit(train)
        predictions = model.predict(len(test))
        metrics = utils.evaluate_forecast(test, predictions, "Naive Seasonal")
        self.results["Naive Seasonal"] = {"predictions": predictions, **metrics}
        return self.results["Naive Seasonal"]

    def train_exponential_smoothing(self, train: TimeSeries, test: TimeSeries) -> Dict:
        """
        Train and evaluate the Exponential Smoothing model.

        :param train: training TimeSeries
        :param test: test TimeSeries
        :return: dictionary with predictions and metrics
        """
        _LOG.info("Training Exponential Smoothing model")
        model = ExponentialSmoothing(seasonal_periods=24, trend=None, seasonal="add")
        model.fit(train)
        predictions = model.predict(len(test))
        metrics = utils.evaluate_forecast(test, predictions, "Exponential Smoothing")
        self.results["Exponential Smoothing"] = {"predictions": predictions, **metrics}
        return self.results["Exponential Smoothing"]

    def train_prophet(self, train: TimeSeries, test: TimeSeries) -> Dict:
        """
        Train and evaluate the Prophet model.

        :param train: training TimeSeries
        :param test: test TimeSeries
        :return: dictionary with predictions and metrics
        """
        _LOG.info("Training Prophet model")
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode="multiplicative",
        )
        model.fit(train)
        predictions = model.predict(len(test))
        metrics = utils.evaluate_forecast(test, predictions, "Prophet")
        self.results["Prophet"] = {"predictions": predictions, **metrics}
        return self.results["Prophet"]

    def train_nbeats(
        self, train_scaled: TimeSeries, test: TimeSeries, scaler: Scaler, **kwargs
    ) -> Dict:
        """
        Train and evaluate the N-BEATS model.

        :param train_scaled: scaled training TimeSeries
        :param test: test TimeSeries (original scale)
        :param scaler: fitted Scaler for inverse transform
        :param kwargs: optional model hyperparameters
        :return: dictionary with predictions and metrics
        """
        _LOG.info("Training N-BEATS model")
        model = NBEATSModel(
            input_chunk_length=kwargs.get("input_chunk_length", 168),
            output_chunk_length=kwargs.get("output_chunk_length", 24),
            generic_architecture=True,
            num_stacks=kwargs.get("num_stacks", 10),
            num_blocks=1,
            num_layers=kwargs.get("num_layers", 4),
            layer_widths=kwargs.get("layer_widths", 256),
            n_epochs=kwargs.get("n_epochs", 50),
            random_state=self.config.random_state,
            pl_trainer_kwargs={
                "enable_progress_bar": True,
                "accelerator": "auto",
            },
        )
        model.fit(train_scaled, verbose=True)
        pred_scaled = model.predict(len(test))
        predictions = scaler.inverse_transform(pred_scaled)
        metrics = utils.evaluate_forecast(test, predictions, "N-BEATS")
        self.results["N-BEATS"] = {"predictions": predictions, **metrics}
        return self.results["N-BEATS"]

    def train_lstm(
        self, train_scaled: TimeSeries, test: TimeSeries, scaler: Scaler, **kwargs
    ) -> Dict:
        """
        Train and evaluate the LSTM model.

        :param train_scaled: scaled training TimeSeries
        :param test: test TimeSeries (original scale)
        :param scaler: fitted Scaler for inverse transform
        :param kwargs: optional model hyperparameters
        :return: dictionary with predictions and metrics
        """
        _LOG.info("Training LSTM model")
        model = RNNModel(
            model="LSTM",
            input_chunk_length=kwargs.get("input_chunk_length", 168),
            output_chunk_length=kwargs.get("output_chunk_length", 24),
            hidden_dim=kwargs.get("hidden_dim", 64),
            n_rnn_layers=kwargs.get("n_rnn_layers", 2),
            dropout=kwargs.get("dropout", 0.1),
            batch_size=kwargs.get("batch_size", 32),
            n_epochs=kwargs.get("n_epochs", 50),
            random_state=self.config.random_state,
            pl_trainer_kwargs={
                "enable_progress_bar": True,
                "accelerator": "auto",
            },
        )
        model.fit(train_scaled, verbose=True)
        pred_scaled = model.predict(len(test))
        predictions = scaler.inverse_transform(pred_scaled)
        metrics = utils.evaluate_forecast(test, predictions, "LSTM")
        self.results["LSTM"] = {"predictions": predictions, **metrics}
        return self.results["LSTM"]

    def get_comparison_summary(self) -> pd.DataFrame:
        """
        Create a summary comparison of all trained models.

        :return: DataFrame with model comparison sorted by MAPE
        """
        return utils.compare_models(self.results)


# #############################################################################
# Hyperparameter Tuning
# #############################################################################


class HyperparameterTuner:
    """
    Perform hyperparameter tuning for forecasting models.
    """

    def __init__(self, config: ForecastConfig):
        """
        Initialize the hyperparameter tuner.

        :param config: ForecastConfig object with settings
        """
        self.config = config
        self.tuning_results = []
        self.best_params = None

    def tune_nbeats(
        self, train_scaled: TimeSeries, val_scaled: TimeSeries, param_grid: Dict
    ) -> Dict:
        """
        Perform grid search for N-BEATS hyperparameters.

        :param train_scaled: scaled training TimeSeries
        :param val_scaled: scaled validation TimeSeries
        :param param_grid: dictionary of parameter ranges
        :return: best parameters found
        """
        _LOG.info("Starting N-BEATS hyperparameter tuning")
        for params in ParameterGrid(param_grid):
            try:
                model = NBEATSModel(
                    input_chunk_length=params["input_chunk_length"],
                    output_chunk_length=params["output_chunk_length"],
                    generic_architecture=True,
                    num_stacks=params["num_stacks"],
                    num_blocks=1,
                    num_layers=params["num_layers"],
                    layer_widths=params["layer_widths"],
                    n_epochs=20,  # Reduced for tuning.
                    random_state=self.config.random_state,
                    pl_trainer_kwargs={
                        "enable_progress_bar": False,
                        "accelerator": "auto",
                    },
                )
                model.fit(train_scaled, verbose=False)
                pred = model.predict(len(val_scaled))
                score = mape(val_scaled, pred)
                params["mape"] = score
                self.tuning_results.append(params)
                _LOG.info("Params: %s -> MAPE: %.2f%%", params, score)
            except Exception as e:
                _LOG.error("Error with params %s: %s", params, e)
                continue
        # Find best parameters.
        tuning_df = pd.DataFrame(self.tuning_results)
        tuning_df = tuning_df.sort_values("mape")
        self.best_params = tuning_df.iloc[0].to_dict()
        del self.best_params["mape"]
        _LOG.info("Best parameters: %s", self.best_params)
        return self.best_params


# #############################################################################
# Main Pipeline
# #############################################################################


def run_forecast_pipeline(config: ForecastConfig = None) -> pd.DataFrame:
    """
    Run the complete energy forecasting pipeline.

    :param config: optional ForecastConfig (uses defaults if None)
    :return: DataFrame with model comparison results
    """
    if config is None:
        config = ForecastConfig()
    # Initialize pipeline components.
    data_pipeline = DataPipeline(config)
    trainer = ModelTrainer(config)
    # Load and prepare data.
    _LOG.info("=" * 60)
    _LOG.info("ENERGY CONSUMPTION FORECASTING PIPELINE")
    _LOG.info("=" * 60)
    data_pipeline.load_data()
    data_pipeline.create_time_series()
    train, test = data_pipeline.split_data()
    train_scaled, test_scaled = data_pipeline.scale_data(train, test)
    # Train models.
    _LOG.info("Training models...")
    trainer.train_naive_seasonal(train, test)
    trainer.train_exponential_smoothing(train, test)
    trainer.train_prophet(train, test)
    trainer.train_nbeats(train_scaled, test, data_pipeline.scaler)
    trainer.train_lstm(train_scaled, test, data_pipeline.scaler)
    # Get comparison summary.
    summary = trainer.get_comparison_summary()
    _LOG.info("\n%s", summary.to_string(index=False))
    best_model = summary.iloc[0]["Model"]
    _LOG.info("Best Model: %s", best_model)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_forecast_pipeline()
    print("\nFinal Model Comparison:")
    print(results.to_string(index=False))
