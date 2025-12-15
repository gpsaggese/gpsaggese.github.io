# AutoKeras API Documentation

## Introduction to AutoKeras

**AutoKeras** is an open-source AutoML (Automated Machine Learning) library built on top of Keras and TensorFlow. It automates the process of model selection, architecture search, and hyperparameter tuning, making deep learning accessible to users without extensive ML expertise.

### What is AutoML?

AutoML aims to automate the end-to-end process of applying machine learning to real-world problems. Instead of manually:
- Choosing model architectures
- Tuning hyperparameters
- Selecting activation functions
- Determining layer sizes

AutoML systems like AutoKeras **automatically search** through these choices to find the best configuration for your data.

### Why AutoKeras?

1. **Ease of Use**: Similar API to scikit-learn
2. **Automated**: Handles model selection and tuning
3. **Efficient**: Uses Neural Architecture Search (NAS)
4. **Flexible**: Supports various tasks (classification, regression, time series)
5. **Production-Ready**: Exports standard Keras models

---

## Core Architecture

### Neural Architecture Search (NAS)

AutoKeras uses NAS to automatically discover optimal neural network architectures. The process involves:

1. **Search Space**: Define possible model configurations
2. **Search Strategy**: Algorithm to explore the space (e.g., Bayesian optimization)
3. **Performance Estimation**: Evaluate candidate models
4. **Iteration**: Repeat until finding the best architecture

### Key Components

```
┌─────────────────────────────────────────┐
│         AutoKeras System                │
├─────────────────────────────────────────┤
│                                         │
│  ┌───────────────────────────────────┐ │
│  │    Task-Specific Searchers        │ │
│  │  (Classification, Regression, etc)│ │
│  └───────────────────────────────────┘ │
│              ↓                          │
│  ┌───────────────────────────────────┐ │
│  │    Architecture Search Engine     │ │
│  │    (Bayesian Optimization)        │ │
│  └───────────────────────────────────┘ │
│              ↓                          │
│  ┌───────────────────────────────────┐ │
│  │    Model Builder                  │ │
│  │    (Creates Keras Models)         │ │
│  └───────────────────────────────────┘ │
│              ↓                          │
│  ┌───────────────────────────────────┐ │
│  │    Training & Evaluation          │ │
│  └───────────────────────────────────┘ │
│                                         │
└─────────────────────────────────────────┘
```

---

## Native API Overview

### StructuredDataRegressor

The `StructuredDataRegressor` is AutoKeras' main class for regression on tabular/structured data.

**Purpose**: Predict continuous numerical values from structured input features.

**Use Cases**:
- Price prediction
- Demand forecasting
- Energy consumption prediction
- Sales forecasting

### Basic Usage Pattern

```python
import autokeras as ak

# 1. Initialize the searcher
model = ak.StructuredDataRegressor(
    max_trials=10,        # Number of models to try
    objective='val_loss', # Metric to optimize
    overwrite=True,       # Overwrite previous results
    seed=42               # Reproducibility
)

# 2. Train the model
model.fit(
    X_train,              # Training features
    y_train,              # Training targets
    validation_data=(X_val, y_val),  # Validation data
    epochs=100            # Epochs per trial
)

# 3. Make predictions
predictions = model.predict(X_test)

# 4. Export the best model
best_model = model.export_model()
```

---

## API Parameters

### StructuredDataRegressor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_trials` | int | 100 | Maximum number of different models to try |
| `objective` | str | 'val_loss' | Metric to optimize ('val_loss', 'val_mae', etc.) |
| `overwrite` | bool | False | Whether to overwrite previous search results |
| `seed` | int | None | Random seed for reproducibility |
| `max_model_size` | int | None | Maximum number of parameters in the model |
| `directory` | str | None | Path to save intermediate results |
| `project_name` | str | 'structured_data_regressor' | Name of the project directory |

### fit() Method Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | DataFrame/Array | Required | Training features |
| `y` | Series/Array | Required | Training targets |
| `validation_data` | tuple | None | (X_val, y_val) for validation |
| `validation_split` | float | 0.2 | Fraction of data for validation if no validation_data |
| `epochs` | int | 1000 | Maximum epochs per trial |
| `callbacks` | list | None | Keras callbacks |
| `verbose` | int | 2 | Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch) |
| `batch_size` | int | 32 | Batch size for training |

### predict() Method

```python
predictions = model.predict(X_test)
```

Returns: numpy array of predictions

### evaluate() Method

```python
loss = model.evaluate(X_test, y_test)
```

Returns: Loss value on the test data

### export_model() Method

```python
keras_model = model.export_model()
```

Returns: Best Keras model found during search

---

## Our Wrapper Layer

We've created a clean wrapper around AutoKeras to make it even easier to use for time series forecasting tasks.

### ElectricityDataPreprocessor

**Purpose**: Handle all data preprocessing for electricity load forecasting.

**Key Methods**:

```python
from autokeras_utils import ElectricityDataPreprocessor

preprocessor = ElectricityDataPreprocessor()

# Load data
df = preprocessor.load_and_prepare_data(
    filepath='data/PJME_hourly.csv',
    datetime_col='Datetime',
    target_col='PJME_MW'
)

# Create time-based features
df = preprocessor.create_time_features(df)

# Create lag features
df = preprocessor.create_lag_features(
    df, 
    target_col='PJME_MW',
    lags=[1, 2, 3, 24, 48, 168]  # 1h, 2h, 3h, 1d, 2d, 1w
)

# Create rolling statistics
df = preprocessor.create_rolling_features(
    df,
    target_col='PJME_MW',
    windows=[3, 6, 12, 24, 168]
)

# Complete feature engineering pipeline
df = preprocessor.prepare_features(
    df, 
    target_col='PJME_MW',
    create_lags=True,
    create_rolling=True
)

# Split data (maintains temporal order)
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
    df,
    target_col='PJME_MW',
    test_size=0.2,
    val_size=0.1
)

# Normalize features
X_train_scaled, X_val_scaled, X_test_scaled = preprocessor.normalize_features(
    X_train, X_val, X_test
)
```

**Why This Wrapper?**
- **Simplified API**: One method for complete feature engineering
- **Best Practices**: Implements time series-specific preprocessing
- **Reproducibility**: Handles scaling consistently
- **Temporal Ordering**: Maintains proper train/val/test splits

### AutoKerasForecaster

**Purpose**: Clean interface to AutoKeras for forecasting tasks.

**Key Methods**:

```python
from autokeras_utils import AutoKerasForecaster

# Initialize
forecaster = AutoKerasForecaster(
    max_trials=10,
    epochs=100,
    objective='val_loss',
    seed=42
)

# Build model
forecaster.build_model(X_train, y_train)

# Train
forecaster.train(
    X_train, y_train,
    X_val, y_val,
    verbose=1
)

# Predict
predictions = forecaster.predict(X_test)

# Evaluate
metrics, predictions = forecaster.evaluate(X_test, y_test)
# Returns: {'MAE': ..., 'RMSE': ..., 'MAPE': ...}
```

**Advantages of Our Wrapper**:
- **Cleaner Interface**: Separate build, train, predict steps
- **Automatic Metrics**: Returns MAE, RMSE, MAPE automatically
- **Progress Tracking**: Clear training progress output
- **Model Export**: Easy access to best model

---

## Feature Engineering for Time Series

Our wrapper includes comprehensive feature engineering specifically designed for time series forecasting.

### 1. Temporal Features

Extract meaningful time components:

```python
# Creates:
# - hour (0-23)
# - day_of_week (0-6)
# - day_of_month (1-31)
# - month (1-12)
# - quarter (1-4)
# - year
# - is_weekend (0 or 1)
```

### 2. Cyclical Encoding

Convert periodic features to continuous representations:

```python
# For hour (24-hour cycle):
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)

# For day (7-day cycle):
day_sin = sin(2π × day_of_week / 7)
day_cos = cos(2π × day_of_week / 7)

# For month (12-month cycle):
month_sin = sin(2π × month / 12)
month_cos = cos(2π × month / 12)
```

**Why cyclical encoding?**
- Hour 23 is close to hour 0 (not 23 units away)
- December is close to January
- Preserves the circular nature of time

### 3. Lag Features

Use past values as features:

```python
# Creates lag_1, lag_2, lag_3, lag_24, lag_48, lag_168
# lag_1 = value 1 hour ago
# lag_24 = value 24 hours ago (same time yesterday)
# lag_168 = value 168 hours ago (same time last week)
```

**Why lags matter?**
- Capture short-term trends (lag_1, lag_2)
- Capture daily patterns (lag_24)
- Capture weekly patterns (lag_168)

### 4. Rolling Window Statistics

Compute statistics over moving windows:

```python
# For each window size (3, 6, 12, 24, 168 hours):
# - rolling_mean_N: average over last N hours
# - rolling_std_N: standard deviation over last N hours
# - rolling_min_N: minimum over last N hours
# - rolling_max_N: maximum over last N hours
```

**Why rolling statistics?**
- Capture trends (rolling mean)
- Detect volatility (rolling std)
- Identify extremes (rolling min/max)

---

## Baseline Models

Our wrapper includes simple baseline models for comparison:

### 1. Naive Forecast

Predict the last known value:

```python
from autokeras_utils import BaselineModels

predictions = BaselineModels.naive_forecast(y_train, n_steps=len(y_test))
```

### 2. Seasonal Naive Forecast

Repeat the last seasonal pattern:

```python
predictions = BaselineModels.seasonal_naive_forecast(
    y_train, 
    n_steps=len(y_test),
    season_length=24  # Daily seasonality
)
```

### 3. Moving Average Forecast

Predict the average of recent values:

```python
predictions = BaselineModels.moving_average_forecast(
    y_train,
    n_steps=len(y_test),
    window=24
)
```

**Why use baselines?**
- Establish minimum performance
- Sometimes simple models work well
- Validate that complex models add value

---

## Visualization Tools

Our wrapper includes comprehensive visualization utilities:

### 1. Plot Predictions

```python
from autokeras_utils import ForecastVisualizer

viz = ForecastVisualizer()
fig = viz.plot_predictions(y_test, predictions, title="AutoKeras Forecast")
```

### 2. Plot Error Distribution

```python
fig = viz.plot_error_distribution(y_test, predictions)
```

### 3. Plot Model Comparison

```python
metrics_dict = {
    'AutoKeras': {'MAE': 100, 'RMSE': 150, 'MAPE': 5.2},
    'Naive': {'MAE': 200, 'RMSE': 250, 'MAPE': 10.1},
    'Seasonal': {'MAE': 180, 'RMSE': 230, 'MAPE': 8.5}
}

fig = viz.plot_metrics_comparison(metrics_dict)
```

---

## Best Practices

### 1. Data Preparation

- Handle missing values before feature engineering
- Maintain temporal order in train/test splits
- Scale features after creating lags (avoid data leakage)
- Use validation set for model selection

### 2. Feature Engineering

- Start with simple features (time components)
- Add lag features for autoregression
- Use cyclical encoding for periodic features
- Experiment with different window sizes

### 3. Model Training

- Start with fewer trials (3-5) for testing
- Use validation data to prevent overfitting
- Monitor training progress
- Compare with baseline models

### 4. Evaluation

- Use multiple metrics (MAE, RMSE, MAPE)
- Visualize predictions vs actuals
- Analyze residuals
- Test on unseen data

---

## Common Pitfalls


### Overfitting to Recent Data

**Problem**: Model learns recent patterns that don't generalize

**Solution**:
- Use validation set from middle period
- Test on truly unseen future data
- Cross-validate with time series splits

---

## Performance Optimization

### 1. Reduce Search Space

```python
# Fewer trials for faster results
model = ak.StructuredDataRegressor(
    max_trials=5,  # Instead of 100
    epochs=50      # Instead of 1000
)
```

### 2. Early Stopping

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model.fit(X_train, y_train, callbacks=[early_stop])
```

### 3. Use Subset for Development

```python
# Use 10% of data for quick experiments
n_samples = int(len(df) * 0.1)
df_subset = df.iloc[:n_samples]
```

---

## Integration with TensorFlow/Keras

AutoKeras exports standard Keras models, enabling full TensorFlow ecosystem:

```python
# Export best model
keras_model = model.export_model()

# Save model
keras_model.save('my_forecast_model.h5')

# Load model later
from tensorflow import keras
loaded_model = keras.models.load_model('my_forecast_model.h5')

# Use like any Keras model
predictions = loaded_model.predict(X_new)
```

---

## Conclusion

This API documentation covered:
1. AutoKeras fundamentals and architecture
2. Native StructuredDataRegressor API
3. Our simplified wrapper layer
4. Feature engineering for time series
5. Baseline models for comparison
6. Visualization utilities
7. Best practices and common pitfalls

---

## References

- AutoKeras Paper: "Auto-Keras: An Efficient Neural Architecture Search System" (KDD 2019)
- Official Documentation: https://autokeras.com/
- GitHub Repository: https://github.com/keras-team/autokeras
- TensorFlow Docs: https://www.tensorflow.org/
