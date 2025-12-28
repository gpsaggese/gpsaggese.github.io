# GluonTS API Documentation

A reference guide for using GluonTS time series forecasting models.

---

## Overview

GluonTS is an open-source Python library for probabilistic time series forecasting. It provides a consistent API and utilities for preparing time series data, defining estimators (model + training code), producing probabilistic forecasts, and evaluating models. GluonTS supports a range of models (classical and neural), convenient dataset wrappers such as `ListDataset`, and evaluation helpers like `make_evaluation_predictions` to generate forecasts and compute metrics.

This document explains how to use three models demonstrated in `GluonTS.API.ipynb`:
- **DeepAR**: Autoregressive RNN for complex temporal patterns
- **SimpleFeedForward**: Fast baseline for stable trends
- **DeepNPTS**: Non-parametric learner for changing distributions

---

## The Three Models

### 1. DeepAR

**What it does:** Uses recurrent neural networks to learn temporal patterns and dependencies.

**When to use it:**
- Complex patterns with seasonality
- Long-term dependencies matter
- Need high accuracy
- Have sufficient training data

**Key parameters:**
- `freq`: Data frequency ('D' for daily, 'H' for hourly, etc.)
- `prediction_length`: How many steps ahead to forecast
- `context_length`: Historical window size (typically 2-4x prediction_length)
- `num_feat_dynamic_real`: Number of external features
- `num_layers`: RNN depth (1-3 typical)
- `hidden_size`: Network capacity (20-100 typical)
- `dropout_rate`: Regularization (0.1-0.3 typical)
- `lr`: Learning rate (0.001 typical)
- `trainer_kwargs`: Pass `{"max_epochs": N}` for training epochs

**Training time:** 1-5 minutes (depends on data size and epochs)

---

### 2. SimpleFeedForward

**What it does:** Direct mapping from recent history to future predictions using feedforward network.

**When to use it:**
- Need fast training
- Stable, smooth trends
- Baseline comparison
- Quick prototyping

**Key parameters:**
- `prediction_length`: How many steps ahead to forecast
- `context_length`: Historical window size
- `hidden_dimensions`: List of layer sizes (e.g., [40, 40])
- `trainer_kwargs`: Pass `{"max_epochs": N}` for training epochs

**Note:** Does NOT use `freq` or `num_feat_dynamic_real` parameters.

**Training time:** 10-30 seconds (much faster than DeepAR)

---

### 3. DeepNPTS

**What it does:** Learns data distribution non-parametrically without assuming specific distribution shape.

**When to use it:**
- Distribution changes over time
- Regime shifts expected
- Unusual, non-standard data
- Heavy tails or rare events

**Key parameters:**
- `freq`: Data frequency
- `prediction_length`: How many steps ahead to forecast
- `context_length`: Historical window size
- `num_feat_dynamic_real`: Number of external features
- `epochs`: Training epochs (passed directly, not via trainer_kwargs)
- `num_hidden_nodes`: List of layer sizes (e.g., [40, 40])

**Note:** Uses `epochs` directly and `num_hidden_nodes` (not `hidden_size`).

**Training time:** 1-3 minutes

---

## Basic Usage Pattern

### Step 1: Prepare Your Data

GluonTS expects data in `ListDataset` format:

```python
from gluonts.dataset.common import ListDataset

train_ds = ListDataset([
    {
        "start": "2020-01-01",
        "target": [100, 105, 110, ...],  # Your time series
        "feat_dynamic_real": [[1.2, 1.3, 1.4, ...],  # Optional features
                             [5.0, 5.1, 5.2, ...]]
    }
], freq="D")
```

**Key points:**
- `start`: First timestamp (string or pandas.Timestamp)
- `target`: Your time series values (list or array)
- `feat_dynamic_real`: External features (optional, 2D array)
- `freq`: Pandas frequency string

---

### Step 2: Configure the Model

```python
from gluonts.torch.model.deepar import DeepAREstimator

estimator = DeepAREstimator(
    freq='D',
    prediction_length=14,
    context_length=60,
    num_feat_dynamic_real=2,  # If using features
    num_layers=2,
    hidden_size=40,
    trainer_kwargs={"max_epochs": 10}
)
```

---

### Step 3: Train

```python
predictor = estimator.train(train_ds)
```

---

### Step 4: Generate Forecasts

```python
from gluonts.evaluation import make_evaluation_predictions

forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,
    predictor=predictor,
    num_samples=100
)

forecasts = list(forecast_it)
actuals = list(ts_it)
```

---

### Step 5: Access Predictions

```python
forecast = forecasts[0]

# Point predictions
mean_forecast = forecast.mean
median_forecast = forecast.quantile(0.5)

# Confidence intervals
lower_bound = forecast.quantile(0.1)  # 10th percentile
upper_bound = forecast.quantile(0.9)  # 90th percentile

# All samples (for uncertainty analysis)
all_samples = forecast.samples  # Shape: (num_samples, prediction_length)
```

---

## Common Parameters Explained

### Temporal Parameters

- **`freq`**: Pandas frequency string
  - 'D': Daily
  - 'H': Hourly
  - 'W': Weekly
  - 'M': Monthly

- **`prediction_length`**: Forecast horizon (e.g., 14 for 14 days ahead)

- **`context_length`**: Historical window used for prediction
  - Rule of thumb: 2-4x prediction_length
  - Longer = more context, but slower training

### Architecture Parameters

- **`num_layers`** (DeepAR): Number of RNN layers
  - 1-2: Fast, less capacity
  - 3-4: More capacity, slower

- **`hidden_size`** (DeepAR): Neurons per layer
  - 20-40: Small, fast
  - 40-60: Medium (recommended)
  - 60-100: Large, more capacity

- **`hidden_dimensions`** (SimpleFeedForward): List of layer sizes
  - `[40, 40]`: Two layers with 40 neurons each

- **`num_hidden_nodes`** (DeepNPTS): List of layer sizes

### Regularization

- **`dropout_rate`**: Prevent overfitting (0.0 to 0.5)
  - 0.1: Light regularization
  - 0.2: Medium (recommended)
  - 0.3: Heavy

### Training Parameters

- **`trainer_kwargs`**: Dictionary for PyTorch Lightning trainer
  - `{"max_epochs": 10}`: Number of training epochs
  - `{"max_epochs": 10, "batch_size": 32}`: Control epochs and batch size

- **`lr`**: Learning rate (0.001 typical)

---

## Features (External Variables)

If your time series is influenced by external factors, include them as features:

```python
# Example: Sales influenced by promotions and temperature
train_ds = ListDataset([
    {
        "start": "2020-01-01",
        "target": [100, 105, 110, ...],  # Sales
        "feat_dynamic_real": [
            [1, 0, 1, ...],      # Promotions (binary)
            [20, 21, 19, ...]    # Temperature
        ]
    }
], freq="D")

# Tell the model how many features you have
estimator = DeepAREstimator(
    freq='D',
    prediction_length=7,
    num_feat_dynamic_real=2,  # 2 features
    ...
)
```

**Important:** Test data must have features for the ENTIRE forecast horizon!

---

## Model Selection Guide

**Choose DeepAR when:**
- Complex patterns with seasonality
- Multiple correlated series
- Accuracy is critical
- You have time for training

**Choose SimpleFeedForward when:**
- Need fast results
- Stable, predictable trends
- Quick baseline
- Limited computational resources

**Choose DeepNPTS when:**
- Distribution changes over time
- Unusual, non-standard data
- Regime shifts expected
- Standard distributions don't fit

---

## Evaluation

Calculate standard metrics:

```python
from GluonTS_utils_evaluation import calculate_metrics

metrics = calculate_metrics(forecast.mean, actual_values)

print(f"MAE: {metrics['mae']:.2f}")
print(f"RMSE: {metrics['rmse']:.2f}")
print(f"MAPE: {metrics['mape']:.2f}%")
```

---

## Probabilistic Forecasts

GluonTS provides uncertainty quantification:

```python
# 50% confidence interval (interquartile range)
q25 = forecast.quantile(0.25)
q75 = forecast.quantile(0.75)

# 90% confidence interval
q05 = forecast.quantile(0.05)
q95 = forecast.quantile(0.95)

# Mean and median
mean = forecast.mean
median = forecast.quantile(0.5)
```

---

## Tips for Better Results

### Data Preprocessing
- Normalize/standardize your data
- Handle missing values before feeding to GluonTS
- Consider log transformation for exponential growth

### Training
- Start with fewer epochs (5-10) for quick experiments
- Increase epochs (20-30) for final models
- Use validation set to avoid overfitting

### Hyperparameters
- Start with defaults, tune one parameter at a time
- context_length: Try 2x, 3x, 4x prediction_length
- hidden_size: Try 20, 40, 60
- dropout_rate: Try 0.1, 0.2, 0.3

### Features
- More isn't always better - test with and without
- Ensure features are available for forecast horizon
- Scale features to similar ranges

---

## Common Issues

### "Wrong number of features"
- Check `num_feat_dynamic_real` matches your data
- Verify test data has features for entire forecast horizon

### "Training too slow"
- Reduce epochs
- Try SimpleFeedForward instead
- Reduce context_length or hidden_size

### "Poor forecast quality"
- Increase context_length
- Add relevant features
- Try different model (DeepNPTS for regime changes)
- Increase training epochs

### "Unexpected keyword argument"
- DeepAR/DeepNPTS: Use `trainer_kwargs={"max_epochs": N}`
- DeepNPTS: Use `epochs` directly
- SimpleFeedForward: No `freq` or `num_feat_dynamic_real`

---

## Further Resources

- [GluonTS Documentation](https://ts.gluon.ai/)
- [GluonTS GitHub](https://github.com/awslabs/gluonts)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)

---

## See Also

- **GluonTS.example.md**: Complete application example with real data
- **README.md**: Project setup and quick start guide
