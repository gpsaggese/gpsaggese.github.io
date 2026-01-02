# TFP Gaussian Process Forecaster API

## Overview
The `TFP_GaussianProcess_Forecaster` is a high-level wrapper around **TensorFlow Probability (TFP)** designed for time series forecasting. It simplifies the complex TFP API into a standard Scikit-Learn style interface (`train` and `predict`).

It uses a composite kernel consisting of:
1.  **Exponentiated Quadratic (RBF):** To capture smooth, long-term trends.
2.  **ExpSinSquared (Periodic):** To capture cyclic patterns (specifically 24-hour seasonality).

## Class Reference

### `TFP_GaussianProcess_Forecaster`

```python
class TFP_GaussianProcess_Forecaster(train_values, train_indices=None)
```

**Parameters:**
- `train_values` (numpy array): The target values (y) to train on. Must be 1D.
- `train_indices` (numpy array, optional): The time steps (X). If None, generates a standard range `[0, 1, 2, ...]`.

---

### Methods

#### `train(epochs=100, learning_rate=0.05)`
Optimizes the Gaussian Process kernel parameters (Amplitude, Length Scale, Period) using the Adam optimizer.

- **epochs** (int): Number of training iterations.
- **learning_rate** (float): Step size for the optimizer.
- **Returns:** None (Prints loss during training).

#### `predict(horizon)`
Generates forecasts for future time steps.

- **horizon** (int): Number of steps to forecast into the future.
- **Returns:**
    - `mean` (numpy array): The most likely predicted values.
    - `stddev` (numpy array): The standard deviation (uncertainty) for each prediction.