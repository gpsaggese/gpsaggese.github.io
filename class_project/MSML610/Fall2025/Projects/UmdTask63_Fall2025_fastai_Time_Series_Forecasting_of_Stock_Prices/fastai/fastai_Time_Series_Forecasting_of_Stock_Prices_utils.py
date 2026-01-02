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
import requests

warnings.filterwarnings('ignore')

try:  # Optional dependency used for VADER-based sentiment
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover - handled gracefully at runtime
    nltk = None  # type: ignore
    SentimentIntensityAnalyzer = None  # type: ignore


def _ensure_naive_datetime(values):
    """Normalizes datetime values or indexes to be timezone naive."""
    dt_values = pd.to_datetime(values)
    if isinstance(dt_values, pd.DatetimeIndex):
        if getattr(dt_values, 'tz', None) is not None:
            dt_values = dt_values.tz_localize(None)
        return dt_values
    if isinstance(dt_values, pd.Timestamp):
        if dt_values.tzinfo is not None:
            dt_values = dt_values.tz_localize(None)
        return dt_values
    return dt_values

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
    include_sentiment: bool = True
    news_api_key: Optional[str] = None
    sentiment_window: int = 3
    news_language: str = "en"
    news_sources: Optional[List[str]] = None
    max_news_articles: int = 100


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


class NewsSentimentAnalyzer:
    """Downloads relevant news and creates sentiment features."""

    def __init__(
        self,
        symbol: str,
        api_key: Optional[str],
        language: str = "en",
        sentiment_window: int = 3,
        sources: Optional[List[str]] = None,
        max_articles: int = 100,
    ):
        self.symbol = symbol
        self.api_key = api_key
        self.language = language
        self.sentiment_window = sentiment_window
        self.sources = sources
        self.max_articles = max_articles
        self.endpoint = "https://newsapi.org/v2/everything"
        self.sentiment_model = self._init_sentiment_model()

    def _init_sentiment_model(self):
        if SentimentIntensityAnalyzer is None:
            warnings.warn(
                "nltk is not installed; skipping sentiment feature creation.",
                RuntimeWarning,
            )
            return None

        try:
            if nltk is not None:
                try:
                    nltk.data.find('sentiment/vader_lexicon')
                except LookupError:
                    nltk.download('vader_lexicon', quiet=True)
            return SentimentIntensityAnalyzer()
        except Exception as exc:  # pragma: no cover - defensive branch
            warnings.warn(
                f"Unable to initialize VADER sentiment analyzer: {exc}",
                RuntimeWarning,
            )
            return None

    def _fetch_news(self, start_date: str, end_date: str) -> pd.DataFrame:
        if not self.api_key:
            warnings.warn(
                "News API key missing; skipping news sentiment features.",
                RuntimeWarning,
            )
            return pd.DataFrame()

        params = {
            'q': self.symbol,
            'language': self.language,
            'from': start_date,
            'to': end_date,
            'sortBy': 'relevancy',
            'pageSize': min(self.max_articles, 100),
        }
        if self.sources:
            params['sources'] = ','.join(self.sources)

        headers = {'X-Api-Key': self.api_key}

        try:
            response = requests.get(self.endpoint, params=params, headers=headers, timeout=30)
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - depends on connectivity
            warnings.warn(f"News sentiment request failed: {exc}", RuntimeWarning)
            return pd.DataFrame()

        payload = response.json()
        if payload.get('status') != 'ok':
            warnings.warn(
                f"News sentiment request returned error: {payload.get('message', 'unknown error')}",
                RuntimeWarning,
            )
            return pd.DataFrame()

        articles = payload.get('articles') or []
        if not isinstance(articles, list):
            warnings.warn(
                "Unexpected News API response format; skipping sentiment features.",
                RuntimeWarning,
            )
            return pd.DataFrame()
        rows = []
        for idx, article in enumerate(articles):
            published = article.get('publishedAt')
            if not published or self.sentiment_model is None:
                continue

            try:
                published_dt = _ensure_naive_datetime(published)
            except Exception:
                continue

            text_segments = [article.get('title'), article.get('description'), article.get('content')]
            combined_text = ' '.join([segment for segment in text_segments if segment])
            if not combined_text.strip():
                continue

            sentiment = self.sentiment_model.polarity_scores(combined_text)
            rows.append(
                {
                    'date': published_dt.normalize(),
                    'compound': sentiment['compound'],
                    'positive': sentiment['pos'],
                    'negative': sentiment['neg'],
                    'neutral': sentiment['neu'],
                    'article_id': idx,
                }
            )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        return df

    def _aggregate_sentiment(self, news_df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        if news_df.empty:
            return pd.DataFrame()

        resampled = news_df.resample('D')
        aggregated = pd.DataFrame({
            'Sentiment_Mean': resampled['compound'].mean(),
            'Sentiment_Median': resampled['compound'].median(),
            'Sentiment_Max': resampled['compound'].max(),
            'Sentiment_Min': resampled['compound'].min(),
            'Sentiment_Positive': resampled['positive'].mean(),
            'Sentiment_Negative': resampled['negative'].mean(),
            'Sentiment_Neutral': resampled['neutral'].mean(),
            'Sentiment_Count': resampled['compound'].count(),
        })
        aggregated['Sentiment_Volatility'] = resampled['compound'].std().fillna(0.0)
        aggregated = aggregated.fillna(0.0)

        full_range = pd.date_range(start=start_date, end=end_date, freq='D')
        aggregated = aggregated.reindex(full_range)
        aggregated = aggregated.fillna(method='ffill').fillna(0.0)

        if self.sentiment_window > 1:
            aggregated = aggregated.rolling(window=self.sentiment_window, min_periods=1).mean()

        aggregated.index = _ensure_naive_datetime(aggregated.index)
        aggregated.index.name = 'Date'
        return aggregated

    def build_sentiment_index(self, start_date: str, end_date: str) -> pd.DataFrame:
        if self.sentiment_model is None:
            return pd.DataFrame()

        news_df = self._fetch_news(start_date, end_date)
        if news_df.empty:
            return pd.DataFrame()

        return self._aggregate_sentiment(news_df, start_date, end_date)


class DataPreprocessor:
    """Preprocesses data for time series forecasting"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.scaler = MinMaxScaler()
        self.base_feature_columns = ['Close', 'Volume', 'MA_5', 'MA_20', 'RSI', 'Volatility']
        self.feature_columns = list(self.base_feature_columns)

    def create_features(self, data: pd.DataFrame, sentiment_features: Optional[pd.DataFrame] = None):
        df = data.copy()
        df.index = _ensure_naive_datetime(df.index)

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

        if sentiment_features is not None and not sentiment_features.empty:
            sentiment_df = sentiment_features.copy()
            sentiment_df.index = _ensure_naive_datetime(sentiment_df.index)
            df = df.merge(sentiment_df, left_index=True, right_index=True, how='left')
            sentiment_cols = list(sentiment_df.columns)
            if sentiment_cols:
                df[sentiment_cols] = df[sentiment_cols].fillna(method='ffill').fillna(0.0)
                self.feature_columns = self.base_feature_columns + sentiment_cols
            else:
                self.feature_columns = list(self.base_feature_columns)
        else:
            self.feature_columns = list(self.base_feature_columns)

        # Drop rows with NaN values created by rolling calculations
        df = df.dropna()

        return df

    def create_sequences(self, data: np.ndarray):
        X, y = [], []

        for i in range(len(data) - self.config.sequence_length):
            X.append(data[i:(i + self.config.sequence_length)])
            y.append(data[i + self.config.sequence_length, 0])  # Predict next close price

        return np.array(X), np.array(y)

    def preprocess_data(self, stock_data: StockData, sentiment_features: Optional[pd.DataFrame] = None):
        # Create features
        df = self.create_features(stock_data.data, sentiment_features)

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
                   end_date: str = "2023-12-31", config: Optional[ModelConfig] = None,
                   sentiment_df: Optional[pd.DataFrame] = None):
    if config is None:
        config = ModelConfig()

    # Collect data
    collector = StockDataCollector()
    stock_data = collector.fetch_stock_data(symbol, start_date, end_date)

    # Build sentiment index when configured
    sentiment_features = sentiment_df
    if sentiment_features is None and config.include_sentiment:
        analyzer = NewsSentimentAnalyzer(
            symbol=symbol,
            api_key=config.news_api_key,
            language=config.news_language,
            sentiment_window=config.sentiment_window,
            sources=config.news_sources,
            max_articles=config.max_news_articles,
        )
        sentiment_features = analyzer.build_sentiment_index(start_date, end_date)

    # Preprocess data
    preprocessor = DataPreprocessor(config)
    X, y, scaler = preprocessor.preprocess_data(stock_data, sentiment_features)

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
        'config': config,
        'sentiment_features': sentiment_features
    }


if __name__ == "__main__":
    results = create_pipeline()
    print(f"Model MAE: {results['metrics']['MAE']:.2f}")
    print(f"Model MAPE: {results['metrics']['MAPE']:.2f}%")

    # Plot results
    plot_predictions(results['actual'], results['predictions'], "S&P 500 Stock Price Prediction")
