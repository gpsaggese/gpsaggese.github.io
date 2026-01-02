"""
Explore the Darts library API for time series forecasting.

This script demonstrates the core functionality of the Darts library including:
- TimeSeries creation and manipulation
- Data preprocessing and scaling
- Multiple forecasting models (Prophet, N-BEATS, LSTM)
- Model evaluation metrics

Citations:
- Darts Library: https://unit8co.github.io/darts/
- Herzen et al. (2022). "Darts: User-Friendly Modern Machine Learning
  for Time Series" JMLR 23(124):1−6

Reference documentation: darts.API.md

"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Darts core imports.
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae, mape, rmse, smape

# Darts models.
from darts.models import (
    ExponentialSmoothing,
    NaiveSeasonal,
    NBEATSModel,
    Prophet,
    RNNModel,
)
from darts.utils.statistics import check_seasonality

_LOG = logging.getLogger(__name__)


# #############################################################################
# TimeSeries Creation
# #############################################################################


class TimeSeriesBuilder:
    """
    Build and manipulate Darts TimeSeries objects from various data sources.
    """

    def __init__(self):
        """Initialize the TimeSeriesBuilder."""
        pass

    def from_dataframe(
        self, df: pd.DataFrame, time_col: str, value_cols: List[str], freq: str = "H"
    ) -> TimeSeries:
        """
        Create a TimeSeries from a pandas DataFrame.

        The DataFrame should have a datetime column and one or more value
        columns. The datetime column will be used as the time index.

        :param df: input DataFrame with datetime and value columns
        :param time_col: name of the datetime column
        :param value_cols: list of value column names
        :param freq: frequency string (default: 'H' for hourly)
        :return: Darts TimeSeries object
        """
        _LOG.info("Creating TimeSeries from DataFrame")
        series = TimeSeries.from_dataframe(
            df, time_col=time_col, value_cols=value_cols, freq=freq
        )
        return series

    def from_series(self, series: pd.Series, freq: str = "H") -> TimeSeries:
        """
        Create a TimeSeries from a pandas Series with datetime index.

        :param series: pandas Series with datetime index
        :param freq: frequency string (default: 'H' for hourly)
        :return: Darts TimeSeries object
        """
        _LOG.info("Creating TimeSeries from pandas Series")
        return TimeSeries.from_series(series, freq=freq)


# #############################################################################
# Data Preprocessing
# #############################################################################


class DataPreprocessor:
    """
    Preprocess time series data for forecasting models.
    """

    def __init__(self):
        """Initialize the DataPreprocessor with a scaler."""
        self.scaler = Scaler()

    def scale(
        self, train: TimeSeries, test: Optional[TimeSeries] = None
    ) -> Tuple[TimeSeries, Optional[TimeSeries]]:
        """
        Scale time series data using StandardScaler.

        Fits the scaler on training data and transforms both train and test.

        :param train: training TimeSeries to fit and transform
        :param test: optional test TimeSeries to transform
        :return: tuple of (scaled_train, scaled_test or None)
        """
        _LOG.info("Scaling time series data")
        train_scaled = self.scaler.fit_transform(train)
        test_scaled = self.scaler.transform(test) if test is not None else None
        return train_scaled, test_scaled

    def inverse_transform(self, series: TimeSeries) -> TimeSeries:
        """
        Inverse transform a scaled TimeSeries back to original scale.

        :param series: scaled TimeSeries
        :return: TimeSeries in original scale
        """
        return self.scaler.inverse_transform(series)

    def check_seasonality(
        self, series: TimeSeries, period: int, max_lag: int = None
    ) -> Tuple[bool, int]:
        """
        Check for seasonality in a time series.

        Uses statistical tests to detect seasonal patterns.

        :param series: TimeSeries to analyze
        :param period: expected seasonal period (e.g., 24 for daily)
        :param max_lag: maximum lag for the test (default: 2 * period)
        :return: tuple of (is_seasonal, detected_period)
        """
        if max_lag is None:
            max_lag = 2 * period
        _LOG.info("Checking seasonality with period=%d", period)
        return check_seasonality(series, m=period, max_lag=max_lag)


# #############################################################################
# Forecasting Models
# #############################################################################


class NaiveSeasonalModel:
    """
    Wrapper for the Naive Seasonal baseline model.
    """

    def __init__(self, seasonal_period: int = 168):
        """
        Initialize the Naive Seasonal model.

        :param seasonal_period: seasonality period (default: 168 for weekly)
        """
        self.model = NaiveSeasonal(K=seasonal_period)
        self.seasonal_period = seasonal_period

    def fit(self, series: TimeSeries) -> None:
        """
        Fit the model on training data.

        :param series: training TimeSeries
        :return: None
        """
        _LOG.info("Fitting NaiveSeasonal model with K=%d", self.seasonal_period)
        self.model.fit(series)

    def predict(self, n: int) -> TimeSeries:
        """
        Generate forecast for n steps ahead.

        :param n: number of steps to forecast
        :return: predicted TimeSeries
        """
        return self.model.predict(n)


class ExponentialSmoothingModel:
    """
    Wrapper for the Exponential Smoothing model.
    """

    def __init__(
        self,
        seasonal_periods: int = 24,
        trend: Optional[str] = None,
        seasonal: str = "add",
    ):
        """
        Initialize the Exponential Smoothing model.

        :param seasonal_periods: number of periods in a season (default: 24)
        :param trend: trend component ('add', 'mul', or None)
        :param seasonal: seasonal component ('add', 'mul', or None)
        """
        self.model = ExponentialSmoothing(
            seasonal_periods=seasonal_periods, trend=trend, seasonal=seasonal
        )

    def fit(self, series: TimeSeries) -> None:
        """
        Fit the model on training data.

        :param series: training TimeSeries
        :return: None
        """
        _LOG.info("Fitting ExponentialSmoothing model")
        self.model.fit(series)

    def predict(self, n: int) -> TimeSeries:
        """
        Generate forecast for n steps ahead.

        :param n: number of steps to forecast
        :return: predicted TimeSeries
        """
        return self.model.predict(n)


class ProphetModel:
    """
    Wrapper for the Prophet forecasting model.
    """

    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
        seasonality_mode: str = "multiplicative",
    ):
        """
        Initialize the Prophet model with seasonality settings.

        :param yearly_seasonality: include yearly seasonality
        :param weekly_seasonality: include weekly seasonality
        :param daily_seasonality: include daily seasonality
        :param seasonality_mode: 'additive' or 'multiplicative'
        """
        self.model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            seasonality_mode=seasonality_mode,
        )

    def fit(self, series: TimeSeries) -> None:
        """
        Fit the Prophet model on training data.

        :param series: training TimeSeries
        :return: None
        """
        _LOG.info("Fitting Prophet model")
        self.model.fit(series)

    def predict(self, n: int) -> TimeSeries:
        """
        Generate forecast for n steps ahead.

        :param n: number of steps to forecast
        :return: predicted TimeSeries
        """
        return self.model.predict(n)


class NBEATSForecastModel:
    """
    Wrapper for the N-BEATS deep learning model.
    """

    def __init__(
        self,
        input_chunk_length: int = 168,
        output_chunk_length: int = 24,
        num_stacks: int = 10,
        num_layers: int = 4,
        layer_widths: int = 256,
        n_epochs: int = 50,
        random_state: int = 42,
    ):
        """
        Initialize the N-BEATS model.

        :param input_chunk_length: length of input sequence
        :param output_chunk_length: length of output forecast
        :param num_stacks: number of stacks in the model
        :param num_layers: number of layers per block
        :param layer_widths: width of each layer
        :param n_epochs: number of training epochs
        :param random_state: random seed for reproducibility
        """
        self.model = NBEATSModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            generic_architecture=True,
            num_stacks=num_stacks,
            num_blocks=1,
            num_layers=num_layers,
            layer_widths=layer_widths,
            n_epochs=n_epochs,
            random_state=random_state,
            pl_trainer_kwargs={
                "enable_progress_bar": True,
                "accelerator": "auto",
            },
        )

    def fit(self, series: TimeSeries, verbose: bool = True) -> None:
        """
        Fit the N-BEATS model on training data.

        :param series: training TimeSeries (should be scaled)
        :param verbose: show training progress
        :return: None
        """
        _LOG.info("Fitting N-BEATS model")
        self.model.fit(series, verbose=verbose)

    def predict(self, n: int) -> TimeSeries:
        """
        Generate forecast for n steps ahead.

        :param n: number of steps to forecast
        :return: predicted TimeSeries
        """
        return self.model.predict(n)


class LSTMForecastModel:
    """
    Wrapper for the LSTM recurrent neural network model.
    """

    def __init__(
        self,
        input_chunk_length: int = 168,
        output_chunk_length: int = 24,
        hidden_dim: int = 64,
        n_rnn_layers: int = 2,
        dropout: float = 0.1,
        n_epochs: int = 50,
        batch_size: int = 32,
        random_state: int = 42,
    ):
        """
        Initialize the LSTM model.

        :param input_chunk_length: length of input sequence
        :param output_chunk_length: length of output forecast
        :param hidden_dim: number of hidden units
        :param n_rnn_layers: number of RNN layers
        :param dropout: dropout rate
        :param n_epochs: number of training epochs
        :param batch_size: training batch size
        :param random_state: random seed for reproducibility
        """
        self.model = RNNModel(
            model="LSTM",
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            hidden_dim=hidden_dim,
            n_rnn_layers=n_rnn_layers,
            dropout=dropout,
            batch_size=batch_size,
            n_epochs=n_epochs,
            random_state=random_state,
            pl_trainer_kwargs={
                "enable_progress_bar": True,
                "accelerator": "auto",
            },
        )

    def fit(self, series: TimeSeries, verbose: bool = True) -> None:
        """
        Fit the LSTM model on training data.

        :param series: training TimeSeries (should be scaled)
        :param verbose: show training progress
        :return: None
        """
        _LOG.info("Fitting LSTM model")
        self.model.fit(series, verbose=verbose)

    def predict(self, n: int) -> TimeSeries:
        """
        Generate forecast for n steps ahead.

        :param n: number of steps to forecast
        :return: predicted TimeSeries
        """
        return self.model.predict(n)


# #############################################################################
# Model Evaluation
# #############################################################################


def compute_metrics(actual: TimeSeries, predicted: TimeSeries) -> Dict[str, float]:
    """
    Compute forecasting evaluation metrics.

    :param actual: actual TimeSeries values
    :param predicted: predicted TimeSeries values
    :return: dictionary with MAPE, RMSE, MAE, SMAPE metrics
    """
    pred_trimmed = predicted[: len(actual)]
    return {
        "MAPE": mape(actual, pred_trimmed),
        "RMSE": rmse(actual, pred_trimmed),
        "MAE": mae(actual, pred_trimmed),
        "SMAPE": smape(actual, pred_trimmed),
    }


def print_metrics(metrics: Dict[str, float], model_name: str) -> None:
    """
    Print formatted evaluation metrics.

    :param metrics: dictionary of metric values
    :param model_name: name of the model
    :return: None
    """
    print(f"\n{model_name} Results:")
    print(f"  MAPE:  {metrics['MAPE']:.2f}%")
    print(f"  RMSE:  {metrics['RMSE']:,.2f} MW")
    print(f"  MAE:   {metrics['MAE']:,.2f} MW")
    print(f"  SMAPE: {metrics['SMAPE']:.2f}%")
