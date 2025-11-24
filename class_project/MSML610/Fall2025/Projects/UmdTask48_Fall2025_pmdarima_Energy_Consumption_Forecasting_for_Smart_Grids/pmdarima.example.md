<!-- toc -->

- [Energy Consumption Forecasting for Smart Grids](#energy-consumption-forecasting-for-smart-grids)
  * [Table of Contents](#table-of-contents)
  * [Project Overview](#project-overview)
  * [Workflow Summary](#workflow-summary)
  * [Model Development](#model-development)
  * [Forecasting and Evaluation](#forecasting-and-evaluation)
  * [Bonus Extensions (Optional)](#bonus-extensions-optional)
  * [References](#references)

<!-- tocstop -->

# Energy Consumption Forecasting for Smart Grids  
**Level:** Hard · **Project 3**

- Example notebook demonstrating an end-to-end **PMDARIMA** workflow for time-series forecasting on the *Individual Household Electric Power Consumption* dataset.

---

## Table of Contents

This file provides a detailed explanation of the `pmdarima.example.ipynb` notebook, describing each step of the workflow, design decisions, and implementation details.

---

## Project Overview

This project implements a complete, end-to-end forecasting pipeline for **hourly household energy consumption** using PMDARIMA (Auto-ARIMA).  
It includes the requirements of **Project 3 (Difficulty: Hard)**, including:

- Data preparation  
- Time-series decomposition  
- Model development  
- Model validation  
- Forecasting and analysis 

**Objective:** 
Develop a forecasting model to predict hourly energy consumption for a smart grid system over the next week. The objective is to optimize the model for accuracy while handling large-scale and noisy data.

**Dataset:**  
- Source: [UCI Machine Learning Repository – Individual Household Electric Power Consumption](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)  
- Sampling rate: 1 minute (resampled to hourly)  
- Target variable: `Global_active_power`

---

## Workflow Summary

The notebook follows a clear, modular structure:

1️⃣ **Load and Preprocess Data**  
   - Implemented in `pmdarima_utils.py`  
   - Parses datetime from `Date` + `Time`  
   - Cleans missing values (`? → NaN`)  
   - Resamples to hourly frequency  

2️⃣ **Time-Series Decomposition**  
   - Uses `seasonal_decompose()`  
   - Extracts trend, seasonal patterns, and residuals  

3️⃣ **Train Auto-ARIMA Model**  
   - Uses `pmdarima.auto_arima()` for automatic (p, d, q) selection  
   - Robust fitting with AIC-based search  

4️⃣ **Model Validation (Rolling Cross-Validation)**  
   - Uses `RollingForecastCV`  
   - Measures stability across multiple folds  

5️⃣ **Generate 7-Day Forecast**  
   - Predicts the next 168 hours  
   - Produces confidence intervals and plots  

6️⃣ **Evaluate Forecast Performance**  
   - Metrics: MAE, RMSE  
   - Implemented using `evaluate_forecast()`  

7️⃣ **Visualize Actual vs Forecast**  
   - Overlays train, test, and predicted values  
   - Adds confidence shading  

---

## Model Development

The modeling is performed using **PMDARIMA’s Auto-ARIMA**, which automates ARIMA parameter selection based on AIC/BIC. 
This approach was chosen to avoid manual hyperparameter tuning and provide a robust model selection process suitable for long, noisy time-series.

**Key Features:**

- Automatic order selection (`stepwise=True`)  
- Robust handling of long time-series  
- Integrated with helper functions in `pmdarima_utils.py`  
- Suitable for large and noisy datasets like UCI household consumption  

Best model obtained:  
`ARIMA(0,1,3)`  

---

## Forecasting and Evaluation

After training, the model forecasts hourly electricity consumption for the next 7 days.
The forecast behavior was also analyzed by comparing predicted vs actual consumption and interpreting confidence intervals.

**Metrics Used:**

- **MAE** – Mean Absolute Error  
- **RMSE** – Root Mean Square Error  

**Visualization Includes:**

- Train and test curves  
- Forecasted values  
- Confidence interval shading  

---

## Bonus Extensions 

1️⃣ **External Factors (Weather Data)**  
   - Integrated temperature data from Open-Meteo  
   - Merged with energy dataset on hourly timestamps  
   - Used as exogenous regressor in Auto-ARIMA  
   - Helps analyze whether weather improves forecasting accuracy  

2️⃣ **Hybrid ARIMA + Machine Learning Model**  
   - Learned ARIMA residuals using a Random Forest  
   - Generated a combined hybrid forecast:  
     `hybrid = arima_forecast + ml_residual_prediction`  
   - Captures non-linear structure not modeled by ARIMA  

---

## References

- [PMDARIMA Documentation](https://alkaline-ml.com/pmdarima/)  
- [UCI Power Consumption Dataset](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)  
- [pmdarima.API.ipynb](./pmdarima.API.ipynb)  
- [pmdarima.example.ipynb](./pmdarima.example.ipynb)  
