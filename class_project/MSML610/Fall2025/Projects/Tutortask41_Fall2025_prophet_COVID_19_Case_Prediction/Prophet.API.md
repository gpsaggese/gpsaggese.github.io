# Prophet API Documentation

## Overview

**Prophet** is an open-source forecasting library developed by Meta (Facebook) for producing high-quality forecasts of time series data. It is designed to handle the common challenges in business forecasting, including missing data, outliers, and holiday effects.

**Version**: Prophet 1.1+  
**License**: MIT  
**Documentation**: https://facebook.github.io/prophet/

---

## Part 1: Native Prophet API

### 1.1 Core Class: `Prophet`

The main interface for building forecasting models.

```python
from prophet import Prophet

model = Prophet(
    growth='linear',                    # 'linear' or 'logistic'
    changepoints=None,                  # List of dates for potential trend changes
    n_changepoints=25,                  # Number of automatic changepoints
    changepoint_range=0.8,              # Proportion of history for changepoints
    yearly_seasonality='auto',          # True, False, or 'auto'
    weekly_seasonality='auto',          # True, False, or 'auto'
    daily_seasonality='auto',           # True, False, or 'auto'
    holidays=None,                       # DataFrame of holidays
    seasonality_mode='additive',        # 'additive' or 'multiplicative'
    seasonality_prior_scale=10.0,       # Regularization for seasonality
    holidays_prior_scale=10.0,          # Regularization for holidays
    changepoint_prior_scale=0.05,       # Flexibility of trend
    interval_width=0.80,                # Width of uncertainty intervals
    uncertainty_samples=1000            # Number of samples for uncertainty
)
```

#### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `growth` | str | 'linear' | Type of trend: 'linear' or 'logistic' (saturating) |
| `changepoint_prior_scale` | float | 0.05 | Controls trend flexibility (higher = more flexible) |
| `seasonality_prior_scale` | float | 10.0 | Controls seasonality strength |
| `holidays_prior_scale` | float | 10.0 | Controls holiday effect strength |
| `seasonality_mode` | str | 'additive' | How seasonality combines with trend |
| `interval_width` | float | 0.80 | Confidence interval width (0-1) |

### 1.2 Data Format Requirements

Prophet requires a DataFrame with exactly two columns:

```python
import pandas as pd

df = pd.DataFrame({
    'ds': ['2020-01-01', '2020-01-02', ...],  # Datetime column
    'y': [100, 150, ...]                       # Target values
})
```

- **`ds`**: Datestamp column (must be `datetime` or parseable string)
- **`y`**: Numeric target variable to forecast

### 1.3 Core Methods

#### `fit(df)`
Fit the Prophet model to historical data.

```python
model.fit(df)
```

#### `make_future_dataframe(periods, freq='D', include_history=True)`
Create a DataFrame for forecasting future dates.

```python
future = model.make_future_dataframe(periods=30)  # 30 days ahead
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `periods` | int | required | Number of periods to forecast |
| `freq` | str | 'D' | Frequency: 'D', 'W', 'M', 'H', etc. |
| `include_history` | bool | True | Include historical dates |

#### `predict(df)`
Generate forecasts for the given dates.

```python
forecast = model.predict(future)
```

Returns DataFrame with columns:
- `ds`: Date
- `yhat`: Point forecast
- `yhat_lower`: Lower confidence bound
- `yhat_upper`: Upper confidence bound
- `trend`: Trend component
- `weekly`: Weekly seasonality (if enabled)
- `yearly`: Yearly seasonality (if enabled)

### 1.4 Adding Custom Components

#### Custom Seasonality
```python
model.add_seasonality(
    name='monthly',           # Unique name
    period=30.5,              # Period in days
    fourier_order=5,          # Complexity of seasonality
    prior_scale=10.0,         # Regularization
    mode='additive'           # 'additive' or 'multiplicative'
)
```

#### External Regressors
```python
model.add_regressor(
    name='temperature',       # Column name in df
    prior_scale=10.0,         # Regularization
    standardize='auto',       # Standardization method
    mode='additive'           # 'additive' or 'multiplicative'
)
```

**Note**: Regressor values must be present in both training and future DataFrames.

#### Holidays
```python
holidays = pd.DataFrame({
    'holiday': ['event_name', 'event_name'],
    'ds': pd.to_datetime(['2020-01-01', '2021-01-01']),
    'lower_window': 0,        # Days before event
    'upper_window': 1         # Days after event
})

model = Prophet(holidays=holidays)
```

### 1.5 Diagnostics Module

```python
from prophet.diagnostics import cross_validation, performance_metrics

# Time series cross-validation
cv_results = cross_validation(
    model,
    initial='365 days',      # Initial training period
    period='30 days',         # Spacing between cutoff dates
    horizon='90 days'         # Forecast horizon
)

# Calculate metrics
metrics = performance_metrics(cv_results)
```

Returned metrics include:
- `mse`: Mean Squared Error
- `rmse`: Root Mean Squared Error
- `mae`: Mean Absolute Error
- `mape`: Mean Absolute Percentage Error
- `mdape`: Median Absolute Percentage Error
- `coverage`: Prediction interval coverage

### 1.6 Visualization

```python
# Plot forecast
fig1 = model.plot(forecast)

# Plot components
fig2 = model.plot_components(forecast)

# Interactive plots (requires plotly)
from prophet.plot import plot_plotly, plot_components_plotly
fig = plot_plotly(model, forecast)
```

---

## Part 2: Wrapper Layer (`prophet_utils.py`)

The wrapper layer provides a simplified, COVID-19-focused interface on top of Prophet's native API.

### 2.1 Design Philosophy

1. **Sensible Defaults**: Pre-configured for epidemiological time series
2. **Method Chaining**: Fluent API for cleaner code
3. **Integrated Evaluation**: Built-in metrics and model comparison
4. **Scenario Analysis**: Tools for policy simulation

### 2.2 `ProphetWrapper` Class

```python
from prophet_utils import ProphetWrapper

wrapper = ProphetWrapper(
    weekly_seasonality=True,        # Capture reporting cycles
    yearly_seasonality=True,        # Seasonal disease patterns
    daily_seasonality=False,        # Not needed for daily data
    changepoint_prior_scale=0.05,   # Trend flexibility
    seasonality_prior_scale=10.0,   # Seasonality strength
    holidays_prior_scale=10.0,      # Intervention effect strength
    interval_width=0.95             # 95% confidence intervals
)
```

#### Method Chaining Example

```python
wrapper = (ProphetWrapper()
    .set_holidays(holidays_df)
    .add_regressor('stringency_index')
    .fit(train_data)
)

forecast = wrapper.predict(periods=28)
```

### 2.3 Data Preparation Functions

#### `load_jhu_timeseries(filepath, country='US')` ⭐ Recommended
Load Johns Hopkins COVID-19 time series data (Jan 2020 - March 2023).

```python
# Load US data
prophet_df = load_jhu_timeseries('jhu_confirmed_global.csv', country='US')

# Load other countries
germany_df = load_jhu_timeseries('jhu_confirmed_global.csv', country='Germany')
```

**Returns**: Prophet-formatted DataFrame with `ds` and `y` columns (daily new cases).

**Data Source**: https://github.com/CSSEGISandData/COVID-19

#### `get_available_countries(filepath)`
List all countries available in the JHU dataset.

```python
countries = get_available_countries('jhu_confirmed_global.csv')
print(countries)  # ['Afghanistan', 'Albania', ..., 'US', ...]
```

#### `load_covid_data(filepath, date_col='Date')` (Legacy)
Load COVID-19 CSV data in standard format (e.g., Kaggle `full_grouped.csv`).

#### `filter_region(df, country, country_col, province_col)`
Filter dataset to specific country/region.

```python
us_data = filter_region(df, 'US', country_col='Country/Region')
```

#### `prepare_prophet_data(df, date_col, target_col, compute_daily)`
Transform to Prophet format with optional daily case calculation.

```python
prophet_df = prepare_prophet_data(
    df, 
    date_col='Date',
    target_col='Confirmed',
    compute_daily=True  # Convert cumulative to daily
)
```

#### `create_intervention_dataframe(interventions)`
Convert intervention dictionary to Prophet holidays format.

```python
interventions = {
    'lockdown_start': '2020-03-15',
    'vaccine_rollout': '2020-12-14'
}
holidays = create_intervention_dataframe(interventions)
```

### 2.4 Comparison Models

#### ARIMA Baseline
```python
from prophet_utils import fit_arima, forecast_arima

model, fitted = fit_arima(df, order=(5, 1, 0))
predictions = forecast_arima(model, periods=28)
```

#### SARIMA (Seasonal)
```python
from prophet_utils import fit_sarima

model, fitted = fit_sarima(
    df, 
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7)  # Weekly seasonality
)
```

### 2.5 Evaluation Metrics

#### Individual Metrics
```python
from prophet_utils import calculate_rmse, calculate_mae, calculate_smape

rmse = calculate_rmse(actual, predicted)
mae = calculate_mae(actual, predicted)
smape = calculate_smape(actual, predicted)
```

#### Comprehensive Evaluation
```python
from prophet_utils import evaluate_forecast, compare_models

# Single model
metrics = evaluate_forecast(actual, predicted, model_name='Prophet')

# Multiple models
comparison = compare_models(actual, {
    'Prophet': prophet_pred,
    'ARIMA': arima_pred,
    'SARIMA': sarima_pred
})
```

### 2.6 Visualization Functions

#### `plot_forecast(df, forecast, title, ylabel, figsize, show_intervals)`
Plot actual vs. predicted with confidence intervals.

```python
fig = plot_forecast(
    df, forecast,
    title='COVID-19 Daily Cases - US',
    ylabel='Daily New Cases'
)
```

#### `plot_intervention_effects(forecast, interventions, figsize)`
Visualize intervention impact on trend.

```python
fig = plot_intervention_effects(forecast, interventions)
```

#### `plot_model_comparison(dates, actual, predictions, title)`
Compare multiple model forecasts visually.

```python
fig = plot_model_comparison(
    dates, actual,
    {'Prophet': p_pred, 'ARIMA': a_pred}
)
```

### 2.7 Scenario Analysis

#### Create Scenario Regressors
```python
from prophet_utils import create_scenario_regressors

future_strict = create_scenario_regressors(
    future_df, scenario='strict', restriction_level=0.8
)
```

#### Run Multiple Scenarios
```python
from prophet_utils import run_scenario_analysis

scenarios = run_scenario_analysis(
    wrapper, 
    periods=28,
    base_restriction=0.5
)
# Returns: {'baseline': ..., 'strict': ..., 'relaxed': ...}
```

### 2.8 Pre-defined Intervention Dates

```python
from prophet_utils import get_us_covid_interventions, get_country_interventions

us_interventions = get_us_covid_interventions()
germany_interventions = get_country_interventions('Germany')
```

---

## Part 3: Quick Reference

### Typical Workflow

```python
from prophet_utils import (
    load_covid_data, filter_region, prepare_prophet_data,
    create_intervention_dataframe, get_us_covid_interventions,
    ProphetWrapper, evaluate_forecast, plot_forecast
)

# 1. Load and prepare data
df = load_covid_data('full_grouped.csv')
us_df = filter_region(df, 'US')
prophet_df = prepare_prophet_data(us_df, compute_daily=True)

# 2. Create interventions
holidays = create_intervention_dataframe(get_us_covid_interventions())

# 3. Train/test split
train = prophet_df[prophet_df['ds'] < '2021-01-01']
test = prophet_df[prophet_df['ds'] >= '2021-01-01']

# 4. Fit model
wrapper = ProphetWrapper(interval_width=0.95)
wrapper.set_holidays(holidays).fit(train)

# 5. Forecast
forecast = wrapper.predict(periods=len(test))

# 6. Evaluate
metrics = evaluate_forecast(test['y'].values, forecast['yhat'].tail(len(test)))

# 7. Visualize
fig = plot_forecast(prophet_df, forecast)
```

### Parameter Tuning Guide

| Use Case | `changepoint_prior_scale` | `seasonality_prior_scale` |
|----------|--------------------------|---------------------------|
| Stable trends | 0.01 - 0.05 | 1.0 - 10.0 |
| Volatile data (COVID) | 0.1 - 0.5 | 10.0 - 25.0 |
| Strong seasonality | 0.05 | 15.0 - 25.0 |

---

## References

1. Taylor, S.J. and Letham, B. (2018). *Forecasting at Scale*. The American Statistician, 72(1), 37-45.
2. Prophet Documentation: https://facebook.github.io/prophet/
3. Prophet GitHub: https://github.com/facebook/prophet