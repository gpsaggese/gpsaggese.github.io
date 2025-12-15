# Time Series Forecasting with Transformers — API Documentation


## Overview

This document describes the native programming interface and the lightweight
wrapper layer implemented in `transformer_utils.py`. The API provides utilities
for data preparation, model training, evaluation, and forecasting using a
Transformer architecture built with JAX and Flax.

---

## Native Model API

### `TimeSeriesTransformer`

A Flax module implementing a Transformer encoder for multivariate time-series
forecasting.

**Key Parameters**
- `seq_len`: Length of historical input window
- `d_model`: Embedding dimension
- `num_heads`: Number of attention heads
- `num_layers`: Number of Transformer blocks
- `mlp_dim`: Hidden dimension of feed-forward layers
- `out_len`: Number of future timesteps predicted
- `num_features`: Number of input features per timestep

---

## Wrapper Layer Functions

The wrapper layer simplifies usage of the native API by exposing higher-level
functions.

---

### Data Utilities

#### `prepare_dataset(path, seq_len, horizon)`

Loads a CSV file and returns windowed, scaled datasets.

**Returns**
- Training, validation, and test splits
- Feature scaler
- Mean and standard deviation of closing prices

---

### Training Utilities

#### `train_model(X_train, y_train, X_val, y_val, ...)`

Trains a Transformer model and automatically saves the best checkpoint.

**Features**
- JIT-compiled training steps
- Validation-based checkpointing
- Configurable batch size and epochs

---

### Evaluation Utilities

#### `predict(model, params, X)`

Runs inference on input windows and returns scaled predictions.

#### `compute_metrics(y_true, y_pred)`

Computes Mean Absolute Error (MAE) for each forecast horizon.

---

### Inference Utilities

#### `predict_next_5_days(model, params, last_window_scaled, scaler_y)`

Generates a real-price forecast for the next 5 timesteps using the most recent
observed window.

---

## Design Philosophy

- Core logic lives in `transformer_utils.py`
- Notebooks remain lightweight and declarative
- API is reusable across notebooks and scripts
- Clear separation between model definition, training, and application logic

---

## Related Files

- `transformer.API.ipynb`: Minimal API usage demonstration
- `transformer.example.ipynb`: End-to-end forecasting application
- `transformer_utils.py`: Reusable API and wrapper layer
