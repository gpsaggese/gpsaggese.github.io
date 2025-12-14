# COVID-19 Case Prediction with Prophet

A comprehensive time series forecasting project comparing **Prophet**, **ARIMA**, **SARIMA**, and **LSTM** models to predict daily COVID-19 cases during the Omicron surge.

**Author**: Ibrahim Ahmed Mohammed  
**Course**: DATA610  
**Date**: December 2025

---

## 📋 Project Overview

This project implements a COVID-19 case forecasting system to support healthcare planning and resource allocation. The key challenge: **predicting the Omicron surge** (January 2022) - when daily cases exploded to 500,000-800,000.

### Models Implemented

| Model | Type | Description |
|-------|------|-------------|
| **Prophet** | Additive | Facebook's decomposable model with seasonality & holidays |
| **ARIMA** | Statistical | AutoRegressive Integrated Moving Average baseline |
| **SARIMA** | Statistical | Seasonal ARIMA with weekly patterns (s=7) |
| **LSTM** | Deep Learning | Long Short-Term Memory neural network |

### Key Features

- 📈 **4-week ahead forecasting** of the Omicron surge (Jan 2022)
- 🏥 **9 policy interventions** (lockdowns, vaccine rollout, variant surges)
- 📊 **Weekly seasonality** to capture reporting cycles (~40% weekend drop)
- 🌍 **Multi-country comparison** (US, Germany, Brazil, India)
- 📉 **Model evaluation** with RMSE, MAE, and SMAPE metrics
- 🛡️ **Non-negative predictions** guaranteed for all models

---

## 📊 Results Summary

### Forecasting Challenge: Omicron Surge Prediction

- **Training**: 710 days (2020-01-22 to 2021-12-31)
- **Test**: 28 days (2022-01-01 to 2022-01-28) - **Omicron Surge!**
- **Actual cases**: ~500,000 - 800,000+ per day

### Model Comparison Results

| Model | RMSE | MAE | SMAPE | Rank |
|-------|------|-----|-------|------|
| **ARIMA** | **338,340** | **271,519** | 42.60% | 🥇 Best RMSE/MAE |
| LSTM | 385,830 | 311,702 | 50.58% | 🥈 |
| SARIMA | 387,957 | 312,909 | **40.73%** | 🥉 Best SMAPE |
| Prophet | 583,947 | 526,183 | 116.74% | 4th |

### Best Model by Metric

| Metric | Winner | Value |
|--------|--------|-------|
| **RMSE** | ARIMA | 338,340 |
| **MAE** | ARIMA | 271,519 |
| **SMAPE** | SARIMA | 40.73% |

### Key Insight

**ARIMA outperformed Prophet** on this challenging task because:
- Prophet underpredicted the surge (73K-259K vs actual 500K-800K)
- Simpler statistical models captured the explosive trend better
- No model perfectly predicted the unprecedented Omicron explosion

---

## 📁 Project Structure

```
├── utils.py                  # Complete utility module with all models
│   ├── ProphetWrapper        # Prophet model wrapper class
│   ├── fit_arima/sarima      # Statistical model functions
│   ├── LSTMForecaster        # LSTM neural network class
│   └── Evaluation metrics    # RMSE, MAE, SMAPE, compare_models
│
├── Prophet_example.ipynb     # Main notebook with full pipeline
├── Prophet_example.md        # Markdown documentation
├── Prophet.API.ipynb         # API demonstration notebook
├── Prophet.API.md            # API documentation
│
├── jhu_confirmed_global.csv  # Johns Hopkins COVID-19 data
├── Dockerfile                # Docker container configuration
└── README.md                 # This file
```

---

## 📊 Dataset

**Source**: Johns Hopkins University COVID-19 Time Series  
**Download**: https://github.com/CSSEGISandData/COVID-19

| Attribute | Value |
|-----------|-------|
| Date Range | January 22, 2020 - March 9, 2023 |
| Total Days | 1,143 |
| Countries | 200+ |
| Data Type | Daily new cases (converted from cumulative) |

### Download the Data

```bash
wget -O jhu_confirmed_global.csv "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
```

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Build the container
docker build -t covid-prophet .

# Run with your project files mounted
docker run -p 8888:8888 -v $(pwd):/app covid-prophet

# Open in browser: http://localhost:8888
```

### Option 2: Local Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install prophet pandas numpy matplotlib scikit-learn statsmodels tensorflow

# Run Jupyter
jupyter notebook Prophet_example.ipynb
```

---

## 📖 Usage

### 1. Loading Data

```python
from utils import load_jhu_timeseries

# Load US COVID-19 data
df = load_jhu_timeseries('jhu_confirmed_global.csv', country='US')

print(f"Date range: {df['ds'].min().date()} to {df['ds'].max().date()}")
print(f"Total days: {len(df)}")
```

### 2. Train/Test Split for Omicron Prediction

```python
import pandas as pd

CUTOFF_DATE = '2022-01-01'
FORECAST_HORIZON = 28

train_df = df[df['ds'] < CUTOFF_DATE].copy()
test_df = df[(df['ds'] >= CUTOFF_DATE) & 
             (df['ds'] < pd.to_datetime(CUTOFF_DATE) + pd.Timedelta(days=FORECAST_HORIZON))].copy()

print(f"Training: {len(train_df)} days")  # 710 days
print(f"Test: {len(test_df)} days")        # 28 days (Omicron surge!)
```

### 3. Prophet Model

```python
from utils import ProphetWrapper, get_us_covid_interventions, create_intervention_dataframe

# Setup interventions
interventions = get_us_covid_interventions()
holidays_df = create_intervention_dataframe(interventions)

# Fit model with method chaining
prophet = (ProphetWrapper(weekly_seasonality=True, yearly_seasonality=True)
    .set_holidays(holidays_df)
    .fit(train_df)
)

# Forecast
prophet_forecast = prophet.predict(periods=28)
```

### 4. ARIMA Model

```python
from utils import fit_arima, forecast_arima

# Fit ARIMA(5,1,0)
arima_model, arima_fitted = fit_arima(train_df, order=(5, 1, 0))

# Forecast
arima_predictions = forecast_arima(arima_model, periods=28)
```

### 5. SARIMA Model

```python
from utils import fit_sarima, forecast_sarima

# Fit SARIMA with weekly seasonality (s=7)
sarima_model, sarima_fitted = fit_sarima(
    train_df, 
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7)
)

# Forecast
sarima_predictions = forecast_sarima(sarima_model, periods=28)
```

### 6. LSTM Model

```python
from utils import LSTMForecaster

# Initialize LSTM
lstm = LSTMForecaster(
    sequence_length=14,      # 2 weeks lookback
    lstm_units=[64, 32],     # Two LSTM layers
    dropout_rate=0.2
)

# Fit
lstm.fit(train_df, epochs=100, batch_size=32, verbose=0)

# Forecast
lstm_predictions = lstm.forecast(train_df, periods=28)
```

### 7. Model Comparison

```python
from utils import compare_models

actual_values = test_df['y'].values

all_predictions = {
    'Prophet': prophet_predictions,
    'ARIMA': arima_predictions,
    'SARIMA': sarima_predictions,
    'LSTM': lstm_predictions
}

comparison_df = compare_models(actual_values, all_predictions)
print(comparison_df)
```

---

## 🔧 API Reference

### Data Loading

| Function | Description |
|----------|-------------|
| `load_jhu_timeseries(filepath, country)` | Load JHU COVID-19 data for a country |
| `get_available_countries(filepath)` | List all countries in dataset |
| `train_test_split_temporal(df, test_size)` | Time-based train/test split |

### Interventions

| Function | Description |
|----------|-------------|
| `get_us_covid_interventions()` | Pre-defined US intervention dates |
| `create_intervention_dataframe(interventions)` | Convert to Prophet holidays format |

### Models

| Class/Function | Description |
|----------------|-------------|
| `ProphetWrapper` | Prophet model with method chaining |
| `fit_arima(df, order)` | Fit ARIMA model |
| `fit_sarima(df, order, seasonal_order)` | Fit SARIMA model |
| `forecast_arima(model, periods)` | Generate ARIMA forecast |
| `forecast_sarima(model, periods)` | Generate SARIMA forecast |
| `LSTMForecaster` | LSTM neural network class |

### Evaluation

| Function | Description |
|----------|-------------|
| `calculate_rmse(actual, predicted)` | Root Mean Squared Error |
| `calculate_mae(actual, predicted)` | Mean Absolute Error |
| `calculate_smape(actual, predicted)` | Symmetric Mean Absolute Percentage Error |
| `compare_models(actual, predictions_dict)` | Compare multiple models |

### Visualization

| Function | Description |
|----------|-------------|
| `plot_forecast(df, forecast)` | Plot forecast with confidence intervals |
| `plot_model_comparison(dates, actual, predictions)` | Compare model predictions |
| `plot_training_history(history)` | Plot LSTM training loss |

---

## 🗓️ US COVID-19 Interventions

| Intervention | Date | Effect Window |
|--------------|------|---------------|
| National Emergency | 2020-03-13 | 14 days |
| Lockdowns Begin | 2020-03-19 | 14 days |
| Reopening Phase 1 | 2020-05-01 | 14 days |
| Summer Surge | 2020-07-01 | 14 days |
| Fall Surge | 2020-10-15 | 14 days |
| Vaccine Authorization | 2020-12-11 | 14 days |
| Vaccine Rollout | 2021-01-15 | 14 days |
| Delta Surge | 2021-07-01 | 14 days |
| Omicron Surge | 2021-12-15 | 14 days |

---

## 🐳 Dockerfile

```dockerfile
FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install prophet pandas numpy matplotlib scikit-learn statsmodels tensorflow jupyter

COPY . /app

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
```

---

## 📈 Exploratory Data Analysis Highlights

### Weekly Seasonality

| Day | Avg Cases | % of Peak |
|-----|-----------|-----------|
| Wednesday | ~110,000 | 100% |
| Sunday | ~60,000 | 55% |

**Finding**: Weekend reporting is ~40% lower than midweek peaks.

### Stationarity

- Original series: **Non-stationary** (has trend)
- First difference: **Stationary** (suitable for ARIMA)

### Autocorrelation

- **Lag 7** spike confirms weekly seasonality
- PACF suggests AR(5) or AR(7) for ARIMA

---

## 📚 References

1. Taylor, S.J. and Letham, B. (2018). *Forecasting at Scale*. The American Statistician, 72(1), 37-45.
2. Johns Hopkins University COVID-19 Data Repository: https://github.com/CSSEGISandData/COVID-19
3. Prophet Documentation: https://facebook.github.io/prophet/
4. Box, G.E.P. and Jenkins, G.M. (1976). *Time Series Analysis: Forecasting and Control*.
5. Hochreiter, S. and Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735-1780.

---

## 📄 License

This project is for educational purposes as part of DATA610 coursework at University of Maryland.

---

## 🤝 Acknowledgments

- **Johns Hopkins University** for the COVID-19 dataset
- **Meta (Facebook)** for the Prophet library
- **University of Maryland** DATA610 course

---

## ✅ Project Requirements Checklist

| Requirement | Status |
|-------------|--------|
| Data Preparation (JHU data) | ✅ |
| Prophet with weekly seasonality | ✅ |
| Interventions as holidays | ✅ (9 events) |
| 4-week forecast | ✅ (28 days) |
| ARIMA baseline | ✅ ARIMA(5,1,0) |
| SARIMA with seasonality | ✅ SARIMA(1,1,1)(1,1,1,7) |
| LSTM Neural Network | ✅ 64+32 units |
| RMSE, MAE, SMAPE metrics | ✅ |
| Actual vs Predicted plot | ✅ |
| Multi-region comparison | ✅ (Bonus) |
