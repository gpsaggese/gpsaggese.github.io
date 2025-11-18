<!-- toc -->

- [PMDARIMA API Tutorial](#pmdarima-api-tutorial)
  * [Table of Contents](#table-of-contents)
  * [General Guidelines](#general-guidelines)

<!-- tocstop -->

# PMDARIMA API Tutorial

- Native API demonstration for `pmdarima` used in the **Individual Household Electric Power Consumption Forecasting Project**

## Table of Contents

This document provides a structured overview of the PMDARIMA API demonstration notebook,  
including data preprocessing, model training, and forecast visualization steps.

---

## General Guidelines

- This tutorial is based on the implementation explored in  
  [`pmdarima.API.ipynb`](./pmdarima.API.ipynb)

- The tutorial demonstrates the **native API usage of PMDARIMA** for **time-series forecasting** on the **Individual Household Electric Power Consumption dataset**.

---

## 1️⃣ Overview

This tutorial introduces the **PMDARIMA library**, a Python package that automates ARIMA model selection for univariate time-series forecasting.

It demonstrates:
- Data loading and preprocessing using `pmdarima_utils.py`
- Splitting data into training and testing sets
- Model training using `pmdarima.auto_arima()`
- Forecast generation and visualization

---

## 2️⃣ Load and Preprocess Data

The dataset used is **Individual Household Electric Power Consumption** from the UCI repository.  
The helper function `load_energy_data()` performs the following preprocessing steps:

- Combines `Date` and `Time` columns into a unified datetime index  
- Handles missing values  
- Resamples the data hourly to reduce granularity  

```python
df = load_energy_data("data/household_power_consumption.txt")
```

After preprocessing:
- Shape: `(34168, 7)`
- Columns: `Global_active_power`, `Global_reactive_power`, `Voltage`, `Global_intensity`, `Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3`

---

## 3️⃣ Split Dataset

We split the processed dataset into training and testing sets using `split_train_test()`.

```python
train, test = split_train_test(df)
```

- Training size: 27,334 records (80%)  
- Testing size: 6,834 records (20%)  
- Ensures chronological continuity (no shuffling)

---

## 4️⃣ Train the Auto-ARIMA Model

We train a univariate Auto-ARIMA model on the `Global_active_power` series.

```python
model = pm.auto_arima(
    train["Global_active_power"],
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    trace=False
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
n_periods = len(test)
forecast = model.predict(n_periods=n_periods)
forecast_index = test.index
```

### Plotting Results
```python
plt.figure(figsize=(12, 5))
plt.plot(train.index, train["Global_active_power"], label="Train", color="blue")
plt.plot(test.index, test["Global_active_power"], label="Test", color="orange")
plt.plot(forecast_index, forecast, label="Forecast", color="green")
plt.title("PMDARIMA Forecast vs Actuals")
plt.xlabel("Datetime")
plt.ylabel("Global Active Power (kW)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
```

The visualization clearly shows:
- Blue: Training data  
- Orange: Actual test data  
- Green: Forecasted values  

---

## 6️⃣ Summary

This tutorial demonstrates the **complete PMDARIMA workflow** for univariate time-series forecasting:

1. Data preprocessing and cleaning (`pmdarima_utils.py`)
2. Train/test split (chronological)
3. Model training via `auto_arima()`
4. Forecasting and performance visualization  

Future extensions:
- Add exogenous regressors (`X` argument in `auto_arima`)
- Test seasonal ARIMA (`seasonal=True`)
- Evaluate metrics such as MAE and RMSE

---

## References

- [PMDARIMA Documentation](https://alkaline-ml.com/pmdarima/)
- [UCI Machine Learning Repository – Power Consumption Dataset](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)
- [pmdarima.API.ipynb](./pmdarima.API.ipynb)
