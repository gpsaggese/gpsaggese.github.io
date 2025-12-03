<!-- toc -->

- [Darts API Tutorial](#darts-api-tutorial)
  * [Overview](#overview)
  * [Installation](#installation)
  * [Core Concepts](#core-concepts)
    + [TimeSeries Object](#timeseries-object)
    + [Data Preprocessing](#data-preprocessing)
  * [Forecasting Models](#forecasting-models)
    + [Statistical Models](#statistical-models)
    + [Machine Learning Models](#machine-learning-models)
    + [Deep Learning Models](#deep-learning-models)
  * [Model Evaluation](#model-evaluation)
  * [API Reference](#api-reference)

<!-- tocstop -->

# Darts API Tutorial

This document provides a comprehensive guide to the Darts library API for time
series forecasting, as explored in `darts.API.py` and `darts.API.ipynb`.

## Overview

Darts is a Python library for easy manipulation and forecasting of time series.
It contains a variety of models, from classics such as ARIMA to deep neural
networks. The library aims to provide a unified interface for multiple time
series forecasting models.

**Key Features:**
- Unified TimeSeries data structure
- Multiple forecasting models (statistical, ML, deep learning)
- Support for multivariate time series
- Built-in evaluation metrics
- Seamless integration with PyTorch and scikit-learn

**Citation:**
```
Herzen et al. (2022). "Darts: User-Friendly Modern Machine Learning
for Time Series" JMLR 23(124):1−6
```

## Installation

```bash
pip install darts
```

For deep learning models (N-BEATS, LSTM):
```bash
pip install "darts[torch]"
```

For Prophet support:
```bash
pip install "darts[prophet]"
```

## Core Concepts

### TimeSeries Object

The `TimeSeries` class is the fundamental data structure in Darts. It represents
a time series with:
- A time index (datetime)
- One or more value columns (univariate or multivariate)
- Optional static covariates

**Creating TimeSeries:**

```python
from darts import TimeSeries
import pandas as pd

# From DataFrame
df = pd.read_csv('data.csv')
series = TimeSeries.from_dataframe(
    df,
    time_col='datetime',
    value_cols='value',
    freq='H'  # Hourly frequency
)

# From pandas Series
series = TimeSeries.from_series(pd_series, freq='H')

# From numpy array
series = TimeSeries.from_values(values_array)
```

**TimeSeries Operations:**

```python
# Slicing
train = series[:-100]  # All but last 100
test = series[-100:]   # Last 100

# Length
len(series)

# Time range
series.start_time()
series.end_time()
series.freq

# Get values
series.univariate_values()  # Returns numpy array
series.pd_dataframe()       # Returns DataFrame
```

### Data Preprocessing

**Scaling:**

```python
from darts.dataprocessing.transformers import Scaler

scaler = Scaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

# Inverse transform predictions
predictions = scaler.inverse_transform(pred_scaled)
```

**Missing Value Handling:**

```python
from darts.dataprocessing.transformers import MissingValuesFiller

filler = MissingValuesFiller()
series_filled = filler.transform(series)
```

**Seasonality Detection:**

```python
from darts.utils.statistics import check_seasonality

is_seasonal, period = check_seasonality(series, m=24, max_lag=48)
```

## Forecasting Models

### Statistical Models

**Naive Seasonal:**
Simple baseline that repeats the last observed seasonal pattern.

```python
from darts.models import NaiveSeasonal

model = NaiveSeasonal(K=168)  # Weekly seasonality (168 hours)
model.fit(train)
forecast = model.predict(n=24)  # Predict 24 steps
```

**Exponential Smoothing:**
Classical time series model with trend and seasonality components.

```python
from darts.models import ExponentialSmoothing

model = ExponentialSmoothing(
    seasonal_periods=24,  # Daily seasonality
    trend=None,           # No trend
    seasonal='add'        # Additive seasonality
)
model.fit(train)
forecast = model.predict(n=24)
```

### Machine Learning Models

**Prophet:**
Facebook's additive forecasting model with automatic seasonality detection.

```python
from darts.models import Prophet

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
    seasonality_mode='multiplicative'
)
model.fit(train)
forecast = model.predict(n=168)  # Predict 1 week
```

### Deep Learning Models

**N-BEATS:**
Neural Basis Expansion Analysis for interpretable time series forecasting.

```python
from darts.models import NBEATSModel

model = NBEATSModel(
    input_chunk_length=168,   # 1 week of history
    output_chunk_length=24,   # Predict 1 day
    generic_architecture=True,
    num_stacks=10,
    num_blocks=1,
    num_layers=4,
    layer_widths=256,
    n_epochs=50,
    random_state=42,
    pl_trainer_kwargs={
        "enable_progress_bar": True,
        "accelerator": "auto",
    }
)

# Train on scaled data
model.fit(train_scaled, verbose=True)
pred_scaled = model.predict(n=720)
predictions = scaler.inverse_transform(pred_scaled)
```

**LSTM (RNNModel):**
Long Short-Term Memory recurrent neural network.

```python
from darts.models import RNNModel

model = RNNModel(
    model='LSTM',
    input_chunk_length=168,
    output_chunk_length=24,
    hidden_dim=64,
    n_rnn_layers=2,
    dropout=0.1,
    batch_size=32,
    n_epochs=50,
    random_state=42,
    pl_trainer_kwargs={
        "enable_progress_bar": True,
        "accelerator": "auto",
    }
)

model.fit(train_scaled, verbose=True)
predictions = model.predict(n=720)
```

## Model Evaluation

Darts provides several metrics for evaluating forecasts:

```python
from darts.metrics import mape, rmse, mae, smape

# Compute metrics
mape_score = mape(actual, predicted)    # Mean Absolute Percentage Error
rmse_score = rmse(actual, predicted)    # Root Mean Squared Error
mae_score = mae(actual, predicted)      # Mean Absolute Error
smape_score = smape(actual, predicted)  # Symmetric MAPE
```

**Backtesting:**

```python
# Historical forecasts for backtesting
historical_forecasts = model.historical_forecasts(
    series=full_series,
    start=start_time,
    forecast_horizon=24,
    stride=168,        # Evaluate weekly
    retrain=False,
    verbose=True
)
```

## API Reference

### Classes in darts.API.py

| Class | Description |
|-------|-------------|
| `TimeSeriesBuilder` | Create TimeSeries from various data sources |
| `DataPreprocessor` | Scale and preprocess time series data |
| `NaiveSeasonalModel` | Wrapper for naive seasonal baseline |
| `ExponentialSmoothingModel` | Wrapper for exponential smoothing |
| `ProphetModel` | Wrapper for Prophet forecasting |
| `NBEATSForecastModel` | Wrapper for N-BEATS deep learning model |
| `LSTMForecastModel` | Wrapper for LSTM recurrent neural network |

### Functions

| Function | Description |
|----------|-------------|
| `compute_metrics(actual, predicted)` | Compute MAPE, RMSE, MAE, SMAPE |
| `print_metrics(metrics, model_name)` | Print formatted evaluation metrics |

### Key Parameters

**N-BEATS:**
- `input_chunk_length`: Length of lookback window
- `output_chunk_length`: Length of forecast horizon
- `num_stacks`: Number of stacks (more = more capacity)
- `num_layers`: Layers per block
- `layer_widths`: Hidden layer width

**LSTM:**
- `hidden_dim`: Number of hidden units
- `n_rnn_layers`: Number of stacked LSTM layers
- `dropout`: Regularization dropout rate

## References

- [Darts Documentation](https://unit8co.github.io/darts/)
- [Darts GitHub Repository](https://github.com/unit8co/darts)
- [N-BEATS Paper](https://arxiv.org/abs/1905.10437)
- [Prophet Documentation](https://facebook.github.io/prophet/)
