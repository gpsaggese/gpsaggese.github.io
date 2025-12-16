# Xformers Time Series API

This module provides a set of tools to implement efficient time-series forecasting using Transformer architectures, leveraging the `xformers` library where available for optimized attention mechanisms.

## Core Components

### `xformers_timeseries_utils.TimeSeriesDataset`
A PyTorch Dataset wrapper that converts a time-series array into sliding window sequences for supervised learning.

- **Args**:
  - `data` (np.ndarray): Input time series data.
  - `seq_len` (int): Lookback window size.
  - `target_len` (int): Prediction horizon size (default 1).

### `xformers_timeseries_utils.XformersTimeSeriesModel`
A PyTorch `nn.Module` that implements a Transformer Encoder architecture for regression.

- **Args**:
  - `input_dim`: Number of input features per time step.
  - `d_model`: Embedding dimension.
  - `nhead`: Number of attention heads.
  - `num_layers`: Number of encoder layers.
  - `dropout`: Dropout rate.

### `xformers_timeseries_utils.DataPreprocessor`
A helper class to handle data scaling and splitting.

- **Methods**:
  - `fit_transform(df, feature_cols)`: Scales the data using MinMaxScaler and splits it into train/test loaders.

### `utils_data_io.fetch_stock_data`
Fetches and caches historical stock data from Yahoo Finance.

## Usage Pattern

The general pattern involves:
1. Fetching data with `utils_data_io`.
2. Preprocessing with `DataPreprocessor`.
3. Instantiating `XformersTimeSeriesModel`.
4. Training with `train_model` from `xformers_timeseries_utils`.
