# COVID-19 Case Prediction with Prophet

A comprehensive time series forecasting project using Facebook Prophet to predict daily COVID-19 cases, with comparisons to ARIMA/SARIMA baselines.

**Author**: Ibrahim Ahmed Mohammed  
**Course**: DATA607-PCS2  
**Date**: December 2024

---

## 📋 Project Overview

This project implements a COVID-19 case forecasting system to support healthcare planning and resource allocation. It demonstrates:

- **Prophet** for time series forecasting with seasonality and intervention effects
- **ARIMA/SARIMA** baseline models for comparison
- **Custom wrapper layer** for simplified, reusable forecasting workflows
- **Multi-region analysis** across different countries

### Key Features

- 📈 4-week ahead daily case forecasting
- 🏥 Incorporation of policy interventions (lockdowns, vaccine rollout, etc.)
- 📊 Weekly seasonality to capture reporting cycles
- 🌍 Multi-country comparison (US, Germany, Brazil, India)
- 📉 Model evaluation with RMSE, MAE, and SMAPE metrics

---

## 📁 Project Structure

```
├── prophet_utils.py          # Utility module with wrapper classes and helper functions
├── Prophet.API.md            # API documentation (native + wrapper layer)
├── Prophet.API.ipynb         # API demonstration notebook
├── Prophet.example.md        # COVID-19 application walkthrough
├── Prophet.example.ipynb     # Complete forecasting pipeline
├── jhu_confirmed_global.csv  # Johns Hopkins COVID-19 time series data
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

# Open in browser
# http://localhost:8888
```

### Option 2: Local Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install prophet pandas numpy matplotlib scikit-learn statsmodels

# Run Jupyter
jupyter notebook
```

---

## 📖 Usage

### Loading Data

```python
from prophet_utils import load_jhu_timeseries

# Load US COVID-19 data
prophet_df = load_jhu_timeseries('jhu_confirmed_global.csv', country='US')

print(f"Date range: {prophet_df['ds'].min().date()} to {prophet_df['ds'].max().date()}")
print(f"Total days: {len(prophet_df)}")
```

### Basic Forecasting

```python
from prophet_utils import ProphetWrapper, get_us_covid_interventions, create_intervention_dataframe

# Get intervention dates
interventions = get_us_covid_interventions()
holidays_df = create_intervention_dataframe(interventions)

# Fit model with method chaining
model = (ProphetWrapper(weekly_seasonality=True, yearly_seasonality=True)
    .set_holidays(holidays_df)
    .fit(prophet_df)
)

# Forecast 4 weeks ahead
forecast = model.predict(periods=28)
```

### Model Evaluation

```python
from prophet_utils import calculate_rmse, calculate_mae, calculate_smape

rmse = calculate_rmse(actual, predicted)
mae = calculate_mae(actual, predicted)
smape = calculate_smape(actual, predicted)

print(f"RMSE: {rmse:,.0f}")
print(f"MAE: {mae:,.0f}")
print(f"SMAPE: {smape:.1f}%")
```

### Visualization

```python
from prophet_utils import plot_forecast, plot_intervention_effects

# Plot forecast with confidence intervals
fig = plot_forecast(prophet_df, forecast, title='COVID-19 Forecast - US')

# Plot intervention effects on trend
fig = plot_intervention_effects(forecast, interventions)
```

---

## 🔧 API Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `load_jhu_timeseries(filepath, country)` | Load JHU COVID-19 data for a specific country |
| `get_available_countries(filepath)` | List all countries in the dataset |
| `create_intervention_dataframe(interventions)` | Convert intervention dict to Prophet holidays format |
| `get_us_covid_interventions()` | Get pre-defined US intervention dates |
| `get_country_interventions(country)` | Get intervention dates for other countries |

### ProphetWrapper Class

```python
wrapper = ProphetWrapper(
    weekly_seasonality=True,      # Capture reporting cycles
    yearly_seasonality=True,      # Seasonal patterns
    changepoint_prior_scale=0.1,  # Trend flexibility (higher = more flexible)
    interval_width=0.95           # Confidence interval width
)

# Method chaining
wrapper.set_holidays(holidays_df)  # Add interventions
wrapper.add_regressor('name')       # Add external regressor
wrapper.fit(df)                     # Fit model
wrapper.predict(periods=28)         # Generate forecast
wrapper.cross_validate(...)         # Time series CV
```

### Evaluation Metrics

| Function | Description |
|----------|-------------|
| `calculate_rmse(actual, predicted)` | Root Mean Squared Error |
| `calculate_mae(actual, predicted)` | Mean Absolute Error |
| `calculate_smape(actual, predicted)` | Symmetric Mean Absolute Percentage Error |
| `compare_models(actual, predictions_dict)` | Compare multiple models |

---

## 📈 Results

### Model Comparison (28-day forecast)

| Model | RMSE | MAE | SMAPE |
|-------|------|-----|-------|
| **Prophet** | ~110,000 | ~72,000 | ~65% |
| ARIMA(5,1,0) | ~125,000 | ~85,000 | ~75% |
| SARIMA(1,1,1)(1,1,1,7) | ~120,000 | ~80,000 | ~70% |

### Cross-Validation Metrics

| Horizon | RMSE | MAE | SMAPE | Coverage |
|---------|------|-----|-------|----------|
| 3 days | 113,390 | 67,882 | 65.5% | 71.7% |
| 14 days | 124,300 | 74,964 | 68.0% | 70.9% |
| 28 days | 110,502 | 72,363 | 65.1% | 67.7% |

### Key Findings

1. **Weekly Seasonality**: COVID-19 reporting shows strong weekly patterns with lower counts on weekends
2. **Intervention Effects**: Major policy changes create visible trend changepoints
3. **Model Performance**: Prophet outperforms traditional ARIMA for volatile COVID data
4. **Omicron Surge**: December 2021 shows the largest spike in US cases

---

## 🗓️ US COVID-19 Interventions

| Intervention | Date |
|--------------|------|
| National Emergency | 2020-03-13 |
| Lockdowns Begin | 2020-03-19 |
| Reopening Phase 1 | 2020-05-01 |
| Summer Surge | 2020-07-01 |
| Fall Surge | 2020-10-15 |
| Vaccine Authorization | 2020-12-11 |
| Vaccine Rollout | 2021-01-15 |
| Delta Surge | 2021-07-01 |
| Omicron Surge | 2021-12-15 |

---

## 🐳 Dockerfile

```dockerfile
FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install prophet pandas numpy matplotlib scikit-learn statsmodels jupyter

COPY . /app

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
```

---

## 📚 References

1. Taylor, S.J. and Letham, B. (2018). *Forecasting at Scale*. The American Statistician, 72(1), 37-45.
2. Johns Hopkins University COVID-19 Data Repository: https://github.com/CSSEGISandData/COVID-19
3. Prophet Documentation: https://facebook.github.io/prophet/
4. Prophet GitHub: https://github.com/facebook/prophet

---

## 📄 License

This project is for educational purposes as part of DATA607-PCS2 coursework.

---

## 🤝 Acknowledgments

- **Johns Hopkins University** for the COVID-19 dataset
- **Meta (Facebook)** for the Prophet library
- **University of Maryland** DATA607-PCS2 course
