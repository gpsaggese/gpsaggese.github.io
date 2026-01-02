<!-- toc -->

- [PMDARIMA API Tutorial](#pmdarima-api-tutorial)
  * [Table of Contents](#table-of-contents)
  * [General Guidelines](#general-guidelines)

<!-- tocstop -->

# PMDARIMA API Tutorial

Native API demonstration for the `pmdarima` library.

---

## Table of Contents

This document provides a structured overview of the PMDARIMA API demonstration notebook,  
including model training and forecast visualization using a synthetic univariate time series.

---

## General Guidelines

- This tutorial is based on the implementation explored in  
  [`pmdarima.API.ipynb`](./pmdarima.API.ipynb)

- The tutorial demonstrates the **native API usage of PMDARIMA** for **time-series forecasting**.

---

## 1️⃣ Overview

This tutorial introduces the **PMDARIMA library**, a Python package that automates ARIMA model selection for univariate time-series forecasting.

It demonstrates:
- Preparation of a univariate time series for forecasting
- Splitting data into training and testing sets
- Model training using `pmdarima.auto_arima()`
- Forecast generation and visualization

This document focuses only on demonstrating the PMDARIMA API; all project-specific implementation details are documented in the example notebook.

---

## 2️⃣ Create a Univariate Time Series

For demonstration purposes, a synthetic univariate time series is created directly in the notebook.

The time series is represented as a pandas Series indexed by datetime.

---

## 3️⃣ Split Dataset

The time series is split chronologically into training and testing segments.

---

## 4️⃣ Train the Auto-ARIMA Model

We train a univariate Auto-ARIMA model on the time series.
The primary entry point of the PMDARIMA API is the `auto_arima()` function.
It automatically selects the optimal ARIMA `(p, d, q)` parameters based on information criteria.

```python
import pmdarima as pm

# Assume `train` is a univariate pandas Series
model = pm.auto_arima(
    train,
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore"
)
```


**Key features:**
- Automatically determines optimal `(p, d, q)` parameters using AIC/BIC  
- Stepwise search for efficiency  
- Suppresses warnings for stationarity or convergence  

Model summary includes estimated coefficients, residuals, and fit statistics.

---

## 5️⃣ Forecast and Visualization

We forecast the next N steps (equal to the test set length) and visualize the predicted vs. actual series.

```python
# Assume test is the held-out portion of the time series.
n_periods = len(test)
forecast = model.predict(n_periods=n_periods)
forecast_index = test.index
```

### Plotting Results

The visualization clearly shows:
- Blue: Training data  
- Orange: Actual test data  
- Green: Forecasted values  

---

## 6️⃣ Summary

This tutorial demonstrates the **complete PMDARIMA workflow** for univariate time-series forecasting:

1. Preparation of a univariate time series
2. Train/test split (chronological)
3. Model training via `auto_arima()`
4. Forecasting and performance visualization  

---

## References

- [PMDARIMA Documentation](https://alkaline-ml.com/pmdarima/)
