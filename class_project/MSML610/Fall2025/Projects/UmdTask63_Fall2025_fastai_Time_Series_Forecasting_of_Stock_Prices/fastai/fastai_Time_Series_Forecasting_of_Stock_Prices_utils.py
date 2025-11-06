"""
Time Series Forecasting Utils Module for Stock Price Prediction

This module provides utility functions for:
- Data collection from Yahoo Finance API
- Data preprocessing and feature engineering
- Model training using FastAI's time series capabilities
- Model evaluation and prediction
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from fastai.tabular.all import *
from fastai.metrics import mae

@dataclass
class StockData:
    symbol: str
    data: pd.DataFrame
    start_date: str
    end_date: str


@dataclass
class ModelConfig:
    sequence_length: int = 60
    prediction_horizon: int = 1
    train_split: float = 0.8
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-3


class StockDataCollector:
    """Collects stock data from Yahoo Finance"""

    @staticmethod
    def fetch_stock_data(symbol: str, start_date: str, end_date: str):
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date)

            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")

            return StockData(
                symbol=symbol,
                data=data,
                start_date=start_date,
                end_date=end_date
            )
        except Exception as e:
            raise Exception(f"Error fetching data for {symbol}: {str(e)}")


class DataPreprocessor:
    """Preprocesses data for time series forecasting"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.scaler = MinMaxScaler()
        self.feature_columns = ['Close', 'Volume', 'MA_5', 'MA_20', 'RSI', 'Volatility']

    def create_features(self, data: pd.DataFrame):
        df = data.copy()

        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()

        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Volatility (rolling standard deviation)
        df['Volatility'] = df['Close'].rolling(window=20).std()

        # Price change
        df['Price_Change'] = df['Close'].pct_change()

        # Lagged variables
        for lag in [1, 2, 3, 5]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)

        # Drop rows with NaN values created by rolling calculations
        df = df.dropna()

        return df

    def create_sequences(self, data: np.ndarray):
        X, y = [], []

        for i in range(len(data) - self.config.sequence_length):
            X.append(data[i:(i + self.config.sequence_length)])
            y.append(data[i + self.config.sequence_length, 0])  # Predict next close price

        return np.array(X), np.array(y)

    def preprocess_data(self, stock_data: StockData):
        # Create features
        df = self.create_features(stock_data.data)

        # Select features for modeling
        features_df = df[self.feature_columns].copy()

        # Scale the data
        scaled_data = self.scaler.fit_transform(features_df)

        # Create sequences
        X, y = self.create_sequences(scaled_data)

        return X, y, self.scaler


class TimeSeriesForecaster:
    """Time series forecasting model using FastAI"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_names = None

    def prepare_data(self, X: np.ndarray, y: np.ndarray):
        # Split data
        split_idx = int(len(X) * self.config.train_split)

        X_train, X_valid = X[:split_idx], X[split_idx:]
        y_train, y_valid = y[:split_idx], y[split_idx:]

        # Reshape for FastAI (flatten sequence for tabular approach)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_valid_flat = X_valid.reshape(X_valid.shape[0], -1)

        # Combine training and validation data for FastAI
        X_combined = np.vstack([X_train_flat, X_valid_flat])
        y_combined = np.concatenate([y_train, y_valid])

        # Create combined DataFrame
        combined_df = pd.DataFrame(X_combined)
        combined_df['target'] = y_combined

        # Store feature names for later use in prediction
        self.feature_names = list(combined_df.columns[:-1])

        # Validation indices point to the validation set portion
        valid_idx = list(range(len(X_train_flat), len(X_combined)))

        # Create FastAI DataLoaders
        dls = TabularDataLoaders.from_df(
            combined_df,
            y_names='target',
            cont_names=self.feature_names,
            valid_idx=valid_idx,
            procs=[Normalize],
            bs=self.config.batch_size
        )

        return dls

    def train_model(self, X: np.ndarray, y: np.ndarray):
        # Prepare data
        dls = self.prepare_data(X, y)

        # Create and train model
        self.model = tabular_learner(
            dls,
            layers=[200, 100],
            metrics=mae,
            cbs=EarlyStoppingCallback(patience=5)
        )

        # Train the model
        self.model.fit_one_cycle(self.config.epochs, self.config.learning_rate)

    def predict(self, X: np.ndarray, scaler: MinMaxScaler):
        # Reshape for prediction
        X_flat = X.reshape(X.shape[0], -1)


        test_df = pd.DataFrame(X_flat, columns=self.feature_names)
        test_df['target'] = 0.0  # Dummy target column

        # Create test dataloader using the same processing pipeline
        test_dl = self.model.dls.test_dl(test_df)

        # Make predictions
        predictions = self.model.get_preds(dl=test_dl)[0].numpy()

        # Inverse scale predictions (only for the target variable)
        dummy_array = np.zeros((len(predictions), len(scaler.feature_names_in_)))
        dummy_array[:, 0] = predictions.flatten()

        # Inverse transform
        inverse_scaled = scaler.inverse_transform(dummy_array)

        return inverse_scaled[:, 0]

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, scaler: MinMaxScaler):
        # Make predictions
        predictions = self.predict(X_test, scaler)

        # Inverse scale actual values
        dummy_array = np.zeros((len(y_test), len(scaler.feature_names_in_)))
        dummy_array[:, 0] = y_test
        y_test_inverse = scaler.inverse_transform(dummy_array)[:, 0]

        # Ensure both arrays are numpy arrays of floats
        predictions = np.array(predictions, dtype=float)
        y_test_inverse = np.array(y_test_inverse, dtype=float)

        # Calculate metrics
        mae = mean_absolute_error(y_test_inverse, predictions)

        # Calculate MAPE (Mean Absolute Percentage Error) with safe division
        with np.errstate(divide='ignore', invalid='ignore'):
            mape_values = np.abs((y_test_inverse - predictions) / y_test_inverse)
            mape_values = mape_values[~np.isnan(mape_values) & ~np.isinf(mape_values)]
            mape = np.mean(mape_values) * 100 if len(mape_values) > 0 else 0.0

        return {
            'MAE': float(mae),
            'MAPE': float(mape)
        }


def plot_predictions(actual: np.ndarray, predicted: np.ndarray, title: str = "Stock Price Predictions"):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual', color='blue')
    plt.plot(predicted, label='Predicted', color='red', linestyle='--')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('StockPricePredictions.png')
    plt.show()


def create_pipeline(symbol: str = "SPY", start_date: str = "2015-01-01",
                   end_date: str = "2023-12-31", config: Optional[ModelConfig] = None):
    if config is None:
        config = ModelConfig()

    # Collect data
    collector = StockDataCollector()
    stock_data = collector.fetch_stock_data(symbol, start_date, end_date)

    # Preprocess data
    preprocessor = DataPreprocessor(config)
    X, y, scaler = preprocessor.preprocess_data(stock_data)

    # Split data for evaluation
    split_idx = int(len(X) * config.train_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train model
    forecaster = TimeSeriesForecaster(config)
    forecaster.train_model(X_train, y_train)
    forecaster.scaler = scaler

    # Evaluate model
    metrics = forecaster.evaluate_model(X_test, y_test, scaler)

    # Make predictions for plotting
    predictions = forecaster.predict(X_test, scaler)

    # Inverse scale actual values for plotting
    dummy_array = np.zeros((len(y_test), len(scaler.feature_names_in_)))
    dummy_array[:, 0] = y_test
    y_test_inverse = scaler.inverse_transform(dummy_array)[:, 0]

    return {
        'model': forecaster,
        'stock_data': stock_data,
        'metrics': metrics,
        'predictions': predictions,
        'actual': y_test_inverse,
        'config': config
    }


if __name__ == "__main__":
    results = create_pipeline()
    print(f"Model MAE: {results['metrics']['MAE']:.2f}")
    print(f"Model MAPE: {results['metrics']['MAPE']:.2f}%")

    # Plot results
    plot_predictions(results['actual'], results['predictions'], "S&P 500 Stock Price Prediction")