# COVID-19 Case Prediction with Prophet

A complete end-to-end forecasting application demonstrating data analysis, feature engineering, and **multi-model comparison** (Prophet vs ARIMA vs SARIMA vs LSTM).

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
7. [Model Comparison (Prophet vs ARIMA vs SARIMA vs LSTM)](#7-model-comparison)
8. [Summary](#8-summary)

---
## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Time Series Forecasting Pipeline             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │ Data Prep   │───▶│   Models    │───▶│    Evaluation       │ │
│  │   Layer     │    │   Layer     │    │    & Comparison     │ │
│  └─────────────┘    └─────────────┘    └─────────────────────┘ │
│        │                  │                      │              │
│        ▼                  ▼                      ▼              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │ load_jhu_   │    │ Prophet     │    │   RMSE, MAE, SMAPE  │ │
│  │ timeseries  │    │ ARIMA       │    │   compare_models()  │ │
│  │ summarize   │    │ SARIMA      │    │   plot_forecast()   │ │
│  │ train_test  │    │ LSTM        │    │   plot_comparison() │ │
│  └─────────────┘    └─────────────┘    └─────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

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

# Our wrapper utilities (includes all models!)
from utils import (
    # Data loading
    ProphetWrapper,
    load_jhu_timeseries,
    get_available_countries,
    prepare_prophet_data,
    create_intervention_dataframe,
    get_us_covid_interventions,
    train_test_split_temporal,
    
    # Statistical models
    fit_arima, fit_sarima,
    forecast_arima, forecast_sarima,
    
    # Deep learning
    LSTMForecaster,
    
    # Metrics
    calculate_rmse, calculate_mae, calculate_smape,
    evaluate_forecast, compare_models,
    
    # Visualization
    plot_forecast, plot_model_comparison,
    plot_training_history, plot_forecast_comparison
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
- **Omicron** (Winter 2021-22): Largest spike in cases (~800k/day)

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
| Lag-7 autocorrelation | Include seasonal component in SARIMA (s=7) |
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

## 6. Wrapper Layer (`utils.py`)

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

### 6.5 Multi-Country Comparison

```python
countries = ['US', 'Germany', 'Brazil', 'India']

for country in countries:
    country_df = load_jhu_timeseries(DATA_PATH, country=country)
    # ... fit and visualize
```

---

## 7. Model Comparison (Prophet vs ARIMA vs SARIMA vs LSTM)

### 7.1 Forecasting Challenge: Predicting the Omicron Surge

Instead of predicting the end of the pandemic (declining cases), we test models on a **real forecasting challenge**: predicting the **Omicron surge** (January 2022).

```python
# Define forecast horizon and cutoff date
FORECAST_HORIZON = 28  # 4 weeks
CUTOFF_DATE = '2022-01-01'  # Predict the Omicron surge!

# Split data at cutoff date
train_df = df[df['ds'] < CUTOFF_DATE].copy()
test_df = df[(df['ds'] >= CUTOFF_DATE) & 
             (df['ds'] < pd.to_datetime(CUTOFF_DATE) + pd.Timedelta(days=FORECAST_HORIZON))].copy()

print(f"Training set: {len(train_df)} days ({train_df['ds'].min().date()} to {train_df['ds'].max().date()})")
print(f"Test set: {len(test_df)} days ({test_df['ds'].min().date()} to {test_df['ds'].max().date()})")
```

**Output**:
- Training set: 710 days (2020-01-22 to 2021-12-31)
- Test set: 28 days (2022-01-01 to 2022-01-28)
- → Predicting the **OMICRON SURGE** - a real forecasting challenge!

### 7.2 Prophet Model

```python
prophet_wrapper = (ProphetWrapper(
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    .set_holidays(holidays_df)
    .fit(train_df)
)

prophet_forecast = prophet_wrapper.predict(periods=FORECAST_HORIZON, include_history=True)
prophet_predictions = prophet_forecast[prophet_forecast['ds'].isin(test_df['ds'])]['yhat'].values

print(f"Prophet predictions: {len(prophet_predictions)} values")
print(f"Range: {prophet_predictions.min():.0f} to {prophet_predictions.max():.0f}")
```

**Output**: Range: 73,053 to 259,903

### 7.3 ARIMA Model (Statistical Baseline)

```python
# Fit ARIMA model - Order (5,1,0): AR(5) with first differencing
arima_model, arima_fitted = fit_arima(train_df, order=(5, 1, 0))
arima_predictions = forecast_arima(arima_model, periods=FORECAST_HORIZON)

print(f"ARIMA predictions: {len(arima_predictions)} values")
print(f"Range: {arima_predictions.min():.0f} to {arima_predictions.max():.0f}")
```

**Output**: Range: 431,255 to 504,669

### 7.4 SARIMA Model (Seasonal ARIMA)

```python
# Fit SARIMA with weekly seasonality (s=7)
sarima_model, sarima_fitted = fit_sarima(
    train_df, 
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7)  # Weekly seasonality
)
sarima_predictions = forecast_sarima(sarima_model, periods=FORECAST_HORIZON)

print(f"SARIMA predictions: {len(sarima_predictions)} values")
print(f"Range: {sarima_predictions.min():.0f} to {sarima_predictions.max():.0f}")
```

**Output**: Range: 394,493 to 1,329,449

### 7.5 LSTM Neural Network

```python
# Initialize LSTM forecaster
lstm = LSTMForecaster(
    sequence_length=14,      # Look back 2 weeks
    lstm_units=[64, 32],     # Two LSTM layers
    dropout_rate=0.2,
    learning_rate=0.001
)

# Fit on training data
lstm.fit(
    train_df,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    early_stopping_patience=10,
    verbose=0
)

# Generate forecast
lstm_predictions = lstm.forecast(train_df, periods=FORECAST_HORIZON)

print(f"LSTM predictions: {len(lstm_predictions)} values")
print(f"Range: {lstm_predictions.min():.0f} to {lstm_predictions.max():.0f}")
```

**Output**: Range: 361,264 to 425,073

### 7.6 Model Evaluation & Comparison

```python
# Get actual test values
actual_values = test_df['y'].values

# Store all predictions
all_predictions = {
    'Prophet': prophet_predictions,
    'ARIMA': arima_predictions,
    'SARIMA': sarima_predictions,
    'LSTM': lstm_predictions
}

# Compare models
comparison_df = compare_models(actual_values, all_predictions)
print(comparison_df.round(2))
```

### Model Comparison Results

| Model | RMSE | MAE | SMAPE |
|-------|------|-----|-------|
| **ARIMA** | **338,340** | **271,519** | 42.60% |
| LSTM | 385,830 | 311,702 | 50.58% |
| SARIMA | 387,957 | 312,909 | **40.73%** |
| Prophet | 583,947 | 526,183 | 116.74% |

### Best Model by Metric

| Metric | Best Model | Value |
|--------|------------|-------|
| **RMSE** | ARIMA | 338,340 |
| **MAE** | ARIMA | 271,519 |
| **SMAPE** | SARIMA | 40.73% |

### 7.7 Analysis of Results

**Why ARIMA outperformed Prophet on this task:**

1. **Prophet underpredicted the surge** (73K-259K vs actual 500K-800K)
   - The exponential Omicron surge was too extreme for the additive model
   - SMAPE of 116% indicates significant underprediction

2. **ARIMA captured the trend better** (431K-504K)
   - Simpler model, closer to actual values
   - Best RMSE and MAE scores

3. **SARIMA had best SMAPE** (40.73%)
   - Weekly seasonality (s=7) helped with pattern recognition
   - But higher RMSE due to wider prediction range

4. **LSTM was consistent but conservative** (361K-425K)
   - Neural network was cautious in predictions
   - Middle-of-the-road performance

**Key Insight**: No model perfectly captured the unprecedented Omicron explosion. This is realistic - predicting sudden exponential surges is extremely difficult for any forecasting method.

---

## 8. Summary

This notebook demonstrated a complete COVID-19 forecasting pipeline with **multi-model comparison**.

### Forecasting Challenge: Predicting the Omicron Surge

We trained models on data up to **January 1, 2022** and predicted the next 4 weeks - the **Omicron surge** when cases exploded to 500k-800k per day. This is a realistic and challenging test of forecasting ability.

### Exploratory Data Analysis
- Distribution analysis showing right-skewed case counts
- Strong weekly seasonality (~40% weekend reporting drop)
- Non-stationary series requiring differencing for ARIMA
- ACF/PACF analysis confirming lag-7 seasonality

### Feature Engineering
- Time-based features (day_of_week, is_weekend, month)
- Lag features (lag_1, lag_7, lag_14)
- Rolling statistics (7-day and 14-day means)
- High correlation between lag features and target

### Model Comparison Results

| Model | Type | RMSE | MAE | SMAPE | Strengths |
|-------|------|------|-----|-------|-----------|
| **ARIMA** | Statistical | **338,340** | **271,519** | 42.60% | Simple, fast, best RMSE/MAE |
| **SARIMA** | Statistical | 387,957 | 312,909 | **40.73%** | Weekly patterns, best SMAPE |
| **LSTM** | Deep Learning | 385,830 | 311,702 | 50.58% | Learns complex patterns |
| **Prophet** | Additive | 583,947 | 526,183 | 116.74% | Interpretable, handles holidays |

### Key Findings

| Aspect | Finding |
|--------|---------|
| Data | 1,143 days, right-skewed, non-stationary |
| Seasonality | Strong weekly pattern (40% weekend drop) |
| Test Scenario | Omicron surge prediction (Jan 2022) |
| Best Model (RMSE/MAE) | ARIMA - simpler model won |
| Best Model (SMAPE) | SARIMA - weekly seasonality helped |
| Prophet Performance | Underpredicted the surge significantly |
| Recommendation | Use ensemble of models for robust predictions |

### Code Deliverables

- **utils.py**: Complete utilities module with:
  - ProphetWrapper class
  - ARIMA/SARIMA functions
  - LSTMForecaster class
  - Evaluation metrics (RMSE, MAE, SMAPE)
  - Visualization functions
  
- **Prophet_example.ipynb**: End-to-end forecasting notebook with model comparison

---

## References

1. Taylor, S.J. and Letham, B. (2018). *Forecasting at Scale*. The American Statistician, 72(1), 37-45.
2. Johns Hopkins University COVID-19 Data Repository: https://github.com/CSSEGISandData/COVID-19
3. Prophet Documentation: https://facebook.github.io/prophet/
4. Box, G.E.P. and Jenkins, G.M. (1976). *Time Series Analysis: Forecasting and Control*.
5. Hochreiter, S. and Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735-1780.
