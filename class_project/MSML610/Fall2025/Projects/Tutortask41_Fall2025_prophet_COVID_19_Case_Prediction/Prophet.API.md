# Prophet API & Wrapper Functions Documentation

## Overview

This document provides comprehensive API documentation for **Facebook Prophet** and the custom **`utils.py`** wrapper module that extends Prophet's functionality with additional models, metrics, and utilities.

**Author**: Ibrahim Ahmed Mohammed  
**Course**: DATA610

---

## 🐳 Installation & Docker Setup

To ensure reproducibility, this project is containerized.

### 1. Build the Image
```bash
docker build -t prophet_project .
```

### 2. Run the Container
```bash
# Mac/Linux/WSL
docker run -p 8888:8888 -v "$(pwd)":/app prophet_project
```

### 3. Access the Project
- Click the `http://127.0.0.1:8888...` link in your terminal
- Open `Prophet_API.ipynb` for tool demonstrations
- Open `Prophet_example.ipynb` for the COVID-19 project implementation

---

## What is Prophet?

**Prophet** is an open-source forecasting tool developed by Facebook (Meta) designed for:
- Business time series with **strong seasonal effects**
- Data with **multiple seasons** (daily, weekly, yearly)
- Series affected by **holidays and special events**
- **Missing data** and **outliers**

### Prophet Model Equation

Prophet uses an additive decomposition model:

```
y(t) = g(t) + s(t) + h(t) + ε(t)
```

Where:
- `g(t)` = trend (linear or logistic growth)
- `s(t)` = seasonality (Fourier series)
- `h(t)` = holiday/intervention effects
- `ε(t)` = error term

---

## What does `utils.py` provide?

Our custom utilities module extends Prophet with:

| Component | Description |
|-----------|-------------|
| **ProphetWrapper** | Simplified interface with method chaining |
| **ARIMA/SARIMA** | Statistical baseline models |
| **LSTMForecaster** | Deep learning model |
| **Evaluation metrics** | RMSE, MAE, SMAPE |
| **Visualization** | Forecast plots, model comparisons |
| **Data utilities** | Loading, preprocessing, splitting |

---


---

## Native Prophet API

### Basic Usage

```python
from prophet import Prophet

# Initialize
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    interval_width=0.95
)

# Fit (requires df with 'ds' and 'y' columns)
model.fit(df)

# Create future dataframe
future = model.make_future_dataframe(periods=30)

# Generate predictions
forecast = model.predict(future)

# Key output columns: ds, yhat, yhat_lower, yhat_upper
```

### Prophet Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `growth` | 'linear' | 'linear' or 'logistic' |
| `yearly_seasonality` | 'auto' | Yearly seasonal component |
| `weekly_seasonality` | 'auto' | Weekly seasonal component |
| `daily_seasonality` | 'auto' | Daily seasonal component |
| `changepoint_prior_scale` | 0.05 | Trend flexibility |
| `seasonality_prior_scale` | 10.0 | Seasonality strength |
| `holidays_prior_scale` | 10.0 | Holiday effect strength |
| `interval_width` | 0.80 | Confidence interval width |

### Built-in Plotting

```python
# Forecast plot
fig1 = model.plot(forecast)

# Component plot (trend, seasonality)
fig2 = model.plot_components(forecast)
```

---

## Class Reference: `ProphetWrapper`

Enhanced Prophet interface with method chaining and automatic non-negative predictions.

### Initialization

```python
from utils import ProphetWrapper

wrapper = ProphetWrapper(
    growth='linear',              # 'linear' or 'logistic'
    weekly_seasonality=True,      # Weekly patterns
    yearly_seasonality=True,      # Yearly patterns
    daily_seasonality=False,      # Usually False for daily data
    changepoint_prior_scale=0.05, # Trend flexibility
    seasonality_prior_scale=10.0, # Seasonality strength
    holidays_prior_scale=10.0,    # Holiday effect strength
    interval_width=0.95,          # 95% confidence intervals
    floor=0.0,                    # Minimum prediction value
    cap=None                      # Maximum for logistic growth
)
```

### Methods

#### `set_holidays(holidays_df)` → `ProphetWrapper`

Register holidays/interventions that affect the time series.

```python
holidays = pd.DataFrame({
    'holiday': ['event_1', 'event_2'],
    'ds': pd.to_datetime(['2022-03-15', '2022-07-04']),
    'lower_window': 0,
    'upper_window': 7
})

wrapper.set_holidays(holidays)
```

**Returns:** `self` for method chaining

---

#### `add_regressor(name, prior_scale=10.0, mode='additive')` → `ProphetWrapper`

Add an external regressor to the model.

```python
wrapper.add_regressor('temperature', prior_scale=10.0, mode='additive')
```

**Parameters:**
- `name`: Column name in the training data
- `prior_scale`: Regularization strength
- `mode`: 'additive' or 'multiplicative'

**Returns:** `self` for method chaining

---

#### `fit(df)` → `ProphetWrapper`

Fit the Prophet model to training data.

```python
wrapper.fit(df)  # df must have 'ds' and 'y' columns
```

**Returns:** `self` for method chaining

---

#### `predict(periods=28, freq='D', include_history=True)` → `pd.DataFrame`

Generate forecasts for future periods.

```python
forecast = wrapper.predict(periods=30)
# Returns: ds, yhat, yhat_lower, yhat_upper, trend, weekly, yearly
```

**Critical Feature:** All predictions are automatically clipped to be ≥ floor (default 0).

---

#### `cross_validate(initial, period, horizon)` → `pd.DataFrame`

Perform time series cross-validation.

```python
cv_results = wrapper.cross_validate(
    initial='365 days',  # Initial training period
    period='30 days',    # Spacing between cutoffs
    horizon='28 days'    # Forecast horizon
)
```

---

#### `get_performance_metrics(cv_results)` → `pd.DataFrame`

Calculate metrics from cross-validation results.

```python
metrics = wrapper.get_performance_metrics(cv_results)
# Returns: horizon, rmse, mae, smape, coverage
```

---

### Method Chaining Example

```python
from utils import ProphetWrapper, create_intervention_dataframe

# Complete workflow in one chain
wrapper = (
    ProphetWrapper(weekly_seasonality=True, yearly_seasonality=True)
    .set_holidays(holidays_df)
    .add_regressor('temperature')
    .fit(train_df)
)

forecast = wrapper.predict(periods=28)
```

---

## Data Loading Functions

### `load_jhu_timeseries(filepath, country='US')` → `pd.DataFrame`

Load Johns Hopkins COVID-19 time series data.

```python
from utils import load_jhu_timeseries

df = load_jhu_timeseries('jhu_confirmed_global.csv', country='US')
# Returns: DataFrame with 'ds' and 'y' columns (daily new cases)
```

**Behavior:**
1. Filters to specified country
2. Sums across provinces/states
3. Converts cumulative to daily new cases
4. Clips negative values to zero

---

### `get_available_countries(filepath)` → `List[str]`

List all countries available in the dataset.

```python
from utils import get_available_countries

countries = get_available_countries('jhu_confirmed_global.csv')
# Returns: ['Afghanistan', 'Albania', ..., 'Zimbabwe']
```

---

### `summarize_data(df, date_col='ds', value_col='y')` → `Dict`

Generate summary statistics for time series data.

```python
from utils import summarize_data

summary = summarize_data(df)
# Returns: {
#   'start_date': ...,
#   'end_date': ...,
#   'n_observations': ...,
#   'mean': ...,
#   'std': ...,
#   'min': ...,
#   'max': ...,
#   'missing_values': ...
# }
```

---

### `train_test_split_temporal(df, test_size=28)` → `Tuple[pd.DataFrame, pd.DataFrame]`

Split time series data chronologically (no data leakage).

```python
from utils import train_test_split_temporal

train_df, test_df = train_test_split_temporal(df, test_size=28)
# train_df: all but last 28 days
# test_df: last 28 days
```

---

## Intervention/Holiday Functions

### `create_intervention_dataframe(interventions)` → `pd.DataFrame`

Convert a dictionary of events to Prophet's holiday format.

```python
from utils import create_intervention_dataframe

interventions = {
    'product_launch': '2022-04-01',
    'marketing_campaign': '2022-06-15',
    'holiday_season': '2022-12-01'
}

holidays_df = create_intervention_dataframe(interventions)
# Returns DataFrame with: holiday, ds, lower_window, upper_window
```

**Default:** `upper_window=14` (effect lasts 2 weeks)

---

### `get_us_covid_interventions()` → `Dict[str, str]`

Get pre-defined US COVID-19 intervention dates.

```python
from utils import get_us_covid_interventions

interventions = get_us_covid_interventions()
# Returns:
# {
#   'national_emergency': '2020-03-13',
#   'lockdowns_begin': '2020-03-19',
#   'reopening_phase1': '2020-05-01',
#   'summer_surge': '2020-07-01',
#   'fall_surge': '2020-10-15',
#   'vaccine_auth': '2020-12-11',
#   'vaccine_rollout': '2021-01-15',
#   'delta_surge': '2021-07-01',
#   'omicron_surge': '2021-12-15'
# }
```

---

### `get_country_interventions(country)` → `Dict[str, str]`

Get intervention dates for specific countries.

```python
from utils import get_country_interventions

# Available: 'US', 'Germany', 'Brazil', 'India'
interventions = get_country_interventions('Germany')
```

---

## ARIMA/SARIMA Functions

### `fit_arima(df, order=(5,1,0), enforce_non_negative=True)` → `Tuple`

Fit ARIMA model for baseline comparison.

```python
from utils import fit_arima

model, fitted_values = fit_arima(df, order=(5, 1, 0))
print(f"AIC: {model.aic:.2f}")
```

**Parameters:**
- `order`: (p, d, q) - AR terms, differencing, MA terms
- `enforce_non_negative`: Clip predictions to ≥ 0

**Returns:** `(fitted_model, fitted_values)`

---

### `fit_sarima(df, order, seasonal_order)` → `Tuple`

Fit Seasonal ARIMA with specified seasonality.

```python
from utils import fit_sarima

model, fitted_values = fit_sarima(
    df,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7)  # Weekly seasonality (s=7)
)
```

**Returns:** `(fitted_model, fitted_values)`

---

### `forecast_arima(model, periods=28)` → `np.ndarray`

Generate ARIMA forecasts.

```python
from utils import forecast_arima

predictions = forecast_arima(model, periods=28)
# Returns: numpy array of non-negative predictions
```

---

### `forecast_sarima(model, periods=28)` → `np.ndarray`

Generate SARIMA forecasts.

```python
from utils import forecast_sarima

predictions = forecast_sarima(model, periods=28)
# Returns: numpy array of non-negative predictions
```

---

## Class Reference: `LSTMForecaster`

LSTM neural network for time series forecasting.

### Initialization

```python
from utils import LSTMForecaster

lstm = LSTMForecaster(
    sequence_length=14,    # Lookback window (time steps)
    n_features=1,          # Number of input features
    lstm_units=[64, 32],   # Units per LSTM layer
    dropout_rate=0.2,      # Dropout for regularization
    learning_rate=0.001    # Adam optimizer learning rate
)
```

### Methods

#### `fit(df, epochs, batch_size, validation_split, early_stopping_patience, verbose)` → `LSTMForecaster`

Train the LSTM model.

```python
lstm.fit(
    train_df,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    early_stopping_patience=10,
    verbose=0
)
```

**Returns:** `self` for method chaining

---

#### `predict(df, enforce_non_negative=True)` → `np.ndarray`

Generate predictions for given data.

```python
predictions = lstm.predict(test_df)
```

---

#### `forecast(df, periods=28)` → `np.ndarray`

Generate multi-step recursive forecasts.

```python
future_predictions = lstm.forecast(train_df, periods=28)
```

**Logic:** Each prediction is fed back as input for the next step.

---

#### `get_training_history()` → `Dict`

Get training metrics for diagnostics.

```python
history = lstm.get_training_history()
# Returns: {'loss': [...], 'val_loss': [...], 'mae': [...], 'val_mae': [...]}
```

---

## Evaluation Metrics

### `calculate_rmse(actual, predicted)` → `float`

Root Mean Squared Error.

```python
from utils import calculate_rmse

rmse = calculate_rmse(actual, predicted)
# Formula: sqrt(mean((actual - predicted)²))
```

---

### `calculate_mae(actual, predicted)` → `float`

Mean Absolute Error.

```python
from utils import calculate_mae

mae = calculate_mae(actual, predicted)
# Formula: mean(|actual - predicted|)
```

---

### `calculate_smape(actual, predicted)` → `float`

Symmetric Mean Absolute Percentage Error.

```python
from utils import calculate_smape

smape = calculate_smape(actual, predicted)
# Formula: 100 × mean(|F - A| / ((|A| + |F|) / 2))
# Range: 0% to 200%
```

---

### `evaluate_forecast(actual, predicted, model_name='Model')` → `Dict`

Calculate all metrics at once.

```python
from utils import evaluate_forecast

metrics = evaluate_forecast(actual, predicted, model_name='ARIMA')
# Returns: {'model': 'ARIMA', 'rmse': ..., 'mae': ..., 'smape': ...}

# Print metrics
for metric, value in metrics.items():
    if metric == 'model':
        print(f"  {metric}: {value}")
    else:
        print(f"  {metric}: {value:.2f}")
```

---

### `compare_models(actual, predictions_dict)` → `pd.DataFrame`

Compare multiple models side-by-side.

```python
from utils import compare_models

comparison = compare_models(actual, {
    'Prophet': prophet_pred,
    'ARIMA': arima_pred,
    'SARIMA': sarima_pred,
    'LSTM': lstm_pred
})

print(comparison)
#              rmse       mae     smape
# model
# Prophet  583947.26  526183.46  116.74
# ARIMA    338340.00  271519.00   42.60
# ...
```

---

## Visualization Functions

### `plot_forecast(df, forecast, title, ylabel, figsize, show_intervals=True)` → `plt.Figure`

Plot actual vs forecasted values with confidence intervals.

```python
from utils import plot_forecast

fig = plot_forecast(
    df=df,
    forecast=forecast,
    title='Forecast with Confidence Intervals'
)
plt.show()
```

---

### `plot_model_comparison(actual_dates, actual_values, predictions, title)` → `plt.Figure`

Overlay multiple model predictions against actuals.

```python
from utils import plot_model_comparison

fig = plot_model_comparison(
    actual_dates=test_df['ds'],
    actual_values=actual,
    predictions={
        'Prophet': prophet_pred,
        'ARIMA': arima_pred,
        'SARIMA': sarima_pred,
        'LSTM': lstm_pred
    },
    title='Model Comparison'
)
plt.show()
```

---

### `plot_training_history(history)` → `plt.Figure`

Plot LSTM training loss curves.

```python
from utils import plot_training_history

history = lstm.get_training_history()
fig = plot_training_history(history)
plt.show()
```

---

### `plot_intervention_effects(forecast, interventions)` → `plt.Figure`

Visualize how interventions affected the trend.

```python
from utils import plot_intervention_effects

fig = plot_intervention_effects(forecast, interventions)
plt.show()
```

---

### `plot_components(wrapper)` → `plt.Figure`

Plot trend and seasonal components.

```python
from utils import plot_components

fig = plot_components(wrapper)
plt.show()
```

---

## Complete API Reference Table

### Classes

| Class | Description |
|-------|-------------|
| `ProphetWrapper` | Enhanced Prophet with method chaining, non-negative predictions |
| `LSTMForecaster` | LSTM neural network for time series |

### Data Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `load_jhu_timeseries()` | `DataFrame` | Load JHU COVID-19 data |
| `get_available_countries()` | `List[str]` | List available countries |
| `prepare_prophet_data()` | `DataFrame` | Transform to Prophet format |
| `summarize_data()` | `Dict` | Summary statistics |
| `train_test_split_temporal()` | `Tuple[DataFrame, DataFrame]` | Time-based split |

### Intervention Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `create_intervention_dataframe()` | `DataFrame` | Convert dict to Prophet holidays |
| `get_us_covid_interventions()` | `Dict` | US intervention dates |
| `get_country_interventions()` | `Dict` | Country-specific interventions |

### Model Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `fit_arima()` | `Tuple` | Fit ARIMA model |
| `fit_sarima()` | `Tuple` | Fit SARIMA model |
| `forecast_arima()` | `ndarray` | ARIMA forecast |
| `forecast_sarima()` | `ndarray` | SARIMA forecast |

### Evaluation Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `calculate_rmse()` | `float` | Root Mean Squared Error |
| `calculate_mae()` | `float` | Mean Absolute Error |
| `calculate_smape()` | `float` | Symmetric MAPE |
| `evaluate_forecast()` | `Dict` | All metrics for one model |
| `compare_models()` | `DataFrame` | Compare multiple models |

### Visualization Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `plot_forecast()` | `Figure` | Forecast with confidence intervals |
| `plot_components()` | `Figure` | Trend and seasonality |
| `plot_model_comparison()` | `Figure` | Multi-model comparison |
| `plot_training_history()` | `Figure` | LSTM training curves |
| `plot_intervention_effects()` | `Figure` | Intervention impact |

---

## References

1. Taylor, S.J. and Letham, B. (2018). *Forecasting at Scale*. The American Statistician, 72(1), 37-45.
2. Prophet Documentation: https://facebook.github.io/prophet/
3. Box, G.E.P. and Jenkins, G.M. (1976). *Time Series Analysis: Forecasting and Control*.
4. Hochreiter, S. and Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735-1780.
