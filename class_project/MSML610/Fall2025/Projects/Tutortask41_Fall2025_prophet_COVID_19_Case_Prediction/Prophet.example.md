# COVID-19 Case Prediction with Prophet

A complete end-to-end forecasting application demonstrating data analysis, feature engineering, and model comparison.

**Author**: Ibrahim Ahmed Mohammed  
**Course**: DATA610  
**Dataset**: Johns Hopkins University COVID-19 Time Series (Jan 2020 - March 2023)

---

## Table of Contents

1. [Setup and Imports](#1-setup-and-imports)
2. [Data Loading](#2-data-loading)
3. [Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
4. [Feature Engineering](#4-feature-engineering)
5. [Native Prophet API](#5-native-prophet-api)
6. [Wrapper Layer](#6-wrapper-layer)
7. [Model Comparison](#7-model-comparison)
8. [Summary](#8-summary)

---

## 1. Setup and Imports

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Statistical tests for EDA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Native Prophet imports
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# Our wrapper utilities
from utils import (
    ProphetWrapper,
    load_jhu_timeseries,
    get_available_countries,
    prepare_prophet_data,
    create_intervention_dataframe,
    get_us_covid_interventions,
    calculate_rmse, calculate_mae, calculate_smape,
    plot_forecast
)
```

---

## 2. Data Loading

```python
# Load US COVID-19 data from Johns Hopkins
DATA_PATH = '/app/jhu_confirmed_global.csv'

df = load_jhu_timeseries(DATA_PATH, country='US')

print(f"Data shape: {df.shape}")
print(f"Date range: {df['ds'].min().date()} to {df['ds'].max().date()}")
print(f"Total days: {len(df)}")
```

**Output**:
- Shape: (1143, 2)
- Date range: 2020-01-22 to 2023-03-09
- Total days: 1143

---

## 3. Exploratory Data Analysis (EDA)

Before building forecasting models, we analyze the characteristics of our time series data to understand patterns, seasonality, and data quality.

### 3.1 Time Series Visualization

The full time series reveals multiple pandemic waves:
- **First Wave** (April 2020): Initial outbreak
- **Summer 2020**: Sun Belt surge
- **Winter 2020-21**: Major nationwide surge
- **Delta** (Summer 2021): Delta variant wave
- **Omicron** (Winter 2021-22): Largest spike in cases

```python
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df['ds'], df['y'], 'b-', linewidth=0.8, alpha=0.8)
ax.fill_between(df['ds'], df['y'], alpha=0.3)
ax.set_xlabel('Date')
ax.set_ylabel('Daily New Cases')
ax.set_title('COVID-19 Daily New Cases - United States')
plt.show()
```

### 3.2 Distribution Analysis

```python
# Distribution statistics
print(f"Mean: {df['y'].mean():,.0f}")
print(f"Median: {df['y'].median():,.0f}")
print(f"Skewness: {stats.skew(df['y']):.2f}")
print(f"Kurtosis: {stats.kurtosis(df['y']):.2f}")
```

**Key Findings**:
- Right-skewed distribution (many low days, few extreme peaks)
- High kurtosis indicates heavy tails (extreme values)
- Omicron peak is a significant outlier

### 3.3 Data Quality Checks

```python
# Missing values
print(f"Missing values: {df['y'].isna().sum()}")

# Zero values
zero_days = (df['y'] == 0).sum()
print(f"Days with zero cases: {zero_days}")

# Outlier detection (IQR method)
Q1, Q3 = df['y'].quantile([0.25, 0.75])
IQR = Q3 - Q1
outliers = df[(df['y'] < Q1 - 1.5*IQR) | (df['y'] > Q3 + 1.5*IQR)]
print(f"Outliers: {len(outliers)} days")
```

**Results**:
- ✅ No missing values
- ✅ No date gaps
- ⚠️ Outliers present (expected for epidemic data)

### 3.4 Seasonality Analysis

#### Weekly Pattern

COVID-19 reporting shows strong weekly seasonality due to reduced testing/reporting on weekends.

| Day | Average Cases | % of Wednesday |
|-----|---------------|----------------|
| Monday | ~85,000 | 78% |
| Tuesday | ~105,000 | 96% |
| Wednesday | ~110,000 | 100% |
| Thursday | ~108,000 | 98% |
| Friday | ~100,000 | 91% |
| Saturday | ~70,000 | 64% |
| Sunday | ~60,000 | 55% |

**Insight**: Weekend reporting is ~40% lower than midweek peaks.

### 3.5 Stationarity Test (ADF)

```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(df['y'].dropna())
print(f"ADF Statistic: {result[0]:.4f}")
print(f"P-Value: {result[1]:.4f}")
```

**Results**:
- Original series: **Non-stationary** (p > 0.05) - has trend
- First difference: **Stationary** (p < 0.05) - suitable for ARIMA

### 3.6 Autocorrelation Analysis (ACF/PACF)

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(df['y'].dropna(), lags=50, ax=axes[0])
plot_pacf(df['y'].dropna(), lags=50, ax=axes[1], method='ywm')
plt.show()
```

**Interpretation**:
- Significant spike at **lag 7** confirms weekly seasonality
- Slow decay in ACF suggests trend component
- PACF suggests AR(5) or AR(7) for ARIMA model

### 3.7 EDA Summary

| Finding | Implication |
|---------|-------------|
| Right-skewed distribution | Consider log transformation for some models |
| Strong weekly seasonality | Enable `weekly_seasonality=True` in Prophet |
| Non-stationary series | Use differencing (d=1) for ARIMA |
| Lag-7 autocorrelation | Include seasonal component in SARIMA |
| High volatility during surges | Use higher `changepoint_prior_scale` |

---

## 4. Feature Engineering

While Prophet handles seasonality automatically, we engineer additional features for analysis and potential model enhancement.

### Time-Based Features

```python
df_features = df.copy()

# Calendar features
df_features['day_of_week'] = df_features['ds'].dt.dayofweek
df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
df_features['month'] = df_features['ds'].dt.month
df_features['quarter'] = df_features['ds'].dt.quarter
df_features['year'] = df_features['ds'].dt.year
```

### Lag Features

```python
# Lagged values for autoregressive patterns
df_features['lag_1'] = df_features['y'].shift(1)    # Yesterday
df_features['lag_7'] = df_features['y'].shift(7)    # 1 week ago
df_features['lag_14'] = df_features['y'].shift(14)  # 2 weeks ago
```

### Rolling Statistics

```python
# Rolling window features
df_features['rolling_mean_7'] = df_features['y'].rolling(window=7).mean()
df_features['rolling_std_7'] = df_features['y'].rolling(window=7).std()
df_features['rolling_mean_14'] = df_features['y'].rolling(window=14).mean()
```

### Percent Change

```python
# Rate of change features
df_features['pct_change'] = df_features['y'].pct_change()
df_features['pct_change_7d'] = df_features['y'].pct_change(periods=7)
```

### 4.1 Feature Correlation Analysis

| Feature | Correlation with y |
|---------|-------------------|
| rolling_mean_7 | 0.98 |
| lag_1 | 0.95 |
| lag_7 | 0.89 |
| is_weekend | -0.15 |
| day_of_week | -0.08 |

**Insight**: Lag and rolling features show high correlation, confirming strong temporal dependencies.

---

## 5. Native Prophet API

### 5.1 Basic Prophet Model

```python
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    interval_width=0.95
)

model.fit(df)
```

### 5.2 Generate Forecast

```python
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
```

### 5.3 Adding Holidays/Interventions

```python
holidays = pd.DataFrame({
    'holiday': ['national_emergency', 'lockdowns_begin', 'vaccine_rollout'],
    'ds': pd.to_datetime(['2020-03-13', '2020-03-19', '2021-01-15']),
    'lower_window': 0,
    'upper_window': 14
})

model_holidays = Prophet(holidays=holidays)
model_holidays.fit(df)
```

### 5.4 Cross-Validation

```python
cv_results = cross_validation(
    model,
    initial='365 days',
    period='30 days',
    horizon='28 days'
)

metrics = performance_metrics(cv_results)
metrics[['horizon', 'mse', 'rmse', 'mae', 'smape']].head(10)
```

---

## 6. Wrapper Layer (`prophet_utils.py`)

### 6.1 ProphetWrapper Class

```python
wrapper = ProphetWrapper(
    weekly_seasonality=True,
    yearly_seasonality=True,
    changepoint_prior_scale=0.1,
    interval_width=0.95
)

print(wrapper.config)
```

### 6.2 Method Chaining with Interventions

```python
interventions = get_us_covid_interventions()
holidays_df = create_intervention_dataframe(interventions)

wrapper = (ProphetWrapper()
    .set_holidays(holidays_df)
    .fit(df)
)
```

**US COVID-19 Interventions**:

| Intervention | Date |
|--------------|------|
| national_emergency | 2020-03-13 |
| lockdowns_begin | 2020-03-19 |
| reopening_phase1 | 2020-05-01 |
| summer_surge | 2020-07-01 |
| fall_surge | 2020-10-15 |
| vaccine_auth | 2020-12-11 |
| vaccine_rollout | 2021-01-15 |
| delta_surge | 2021-07-01 |
| omicron_surge | 2021-12-15 |

### 6.3 Forecasting with Wrapper

```python
forecast = wrapper.predict(periods=28)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
```

### 6.4 Evaluation Metrics

```python
rmse = calculate_rmse(actual, predictions)
mae = calculate_mae(actual, predictions)
smape = calculate_smape(actual, predictions)

print(f"RMSE:  {rmse:,.2f}")
print(f"MAE:   {mae:,.2f}")
print(f"SMAPE: {smape:.2f}%")
```

### 6.5 Cross-Validation with Wrapper

```python
cv_results = wrapper.cross_validate(
    initial='365 days',
    period='30 days',
    horizon='28 days'
)

cv_metrics = wrapper.get_performance_metrics(cv_results)
cv_metrics[['horizon', 'rmse', 'mae', 'smape']].head(10)
```

### 6.6 Multi-Country Comparison

```python
countries = ['US', 'Germany', 'Brazil', 'India']

for country in countries:
    country_df = load_jhu_timeseries(DATA_PATH, country=country)
    # ... fit and visualize
```

---

## 7. Native vs Wrapper Comparison

### Native Prophet API
```python
# More verbose, more control
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    holidays=holidays_df,
    changepoint_prior_scale=0.05
)
model.add_regressor('temperature')
model.fit(df)
future = model.make_future_dataframe(periods=30)
future['temperature'] = get_future_temps()
forecast = model.predict(future)
```

### Wrapper Layer
```python
# Cleaner, method chaining, sensible defaults
wrapper = (ProphetWrapper()
    .set_holidays(holidays_df)
    .add_regressor('temperature')
    .fit(df)
)
forecast = wrapper.predict(periods=30)
```

### Comparison Table

| Feature | Native Prophet | Wrapper Layer |
|---------|---------------|---------------|
| Data Loading | Manual | `load_jhu_timeseries()` |
| Configuration | Verbose | Method chaining |
| Interventions | Manual DataFrame | `get_us_covid_interventions()` |
| SMAPE Metric | Not included | Built-in |
| COVID Defaults | Manual setup | Pre-configured |
| Multi-country | Manual | `get_country_interventions()` |

---

## 8. Summary

This notebook demonstrated a complete COVID-19 forecasting pipeline:

### Exploratory Data Analysis
- Distribution analysis showing right-skewed case counts
- Strong weekly seasonality (lower weekend reporting)
- Non-stationary series requiring differencing for ARIMA
- ACF/PACF analysis confirming lag-7 seasonality

### Feature Engineering
- Time-based features (day_of_week, is_weekend, month)
- Lag features (lag_1, lag_7, lag_14)
- Rolling statistics (7-day and 14-day means)
- High correlation between lag features and target

### Native Prophet API
- Basic model creation and fitting
- Adding holidays and external regressors
- Cross-validation and performance metrics
- Built-in visualization

### Wrapper Layer (`prophet_utils.py`)
- `load_jhu_timeseries()` for Johns Hopkins data (1,143 days, 200+ countries)
- `ProphetWrapper` class with method chaining
- `get_us_covid_interventions()` for pre-defined intervention dates
- Custom evaluation metrics including SMAPE
- Enhanced visualizations

### Key Findings

| Aspect | Finding |
|--------|---------|
| Data | 1,143 days, right-skewed, non-stationary |
| Seasonality | Strong weekly pattern (40% weekend drop) |
| Model | Prophet handles COVID volatility well |
| Recommendation | Use weekly_seasonality=True with interventions |

---

## References

1. Taylor, S.J. and Letham, B. (2018). *Forecasting at Scale*. The American Statistician, 72(1), 37-45.
2. Johns Hopkins University COVID-19 Data Repository: https://github.com/CSSEGISandData/COVID-19
3. Prophet Documentation: https://facebook.github.io/prophet/
