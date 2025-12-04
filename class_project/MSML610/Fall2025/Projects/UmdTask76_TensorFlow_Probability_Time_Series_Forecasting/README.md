# TFP Forecasting Project – MSML610

## Overview
This project explores **Time Series Forecasting with Probabilistic Models** using **TensorFlow Probability (TFP)**.  
The focus is on forecasting **air quality metrics** while capturing uncertainty in predictions.

## Current Progress (≈35%)
**Phase 1 – Data Preparation & EDA**
- Set up repository and Docker/Jupyter environment.
- Loaded and cleaned the [UCI Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/air+quality).
- Parsed timestamps, handled missing values, and performed initial EDA.
- Produced trend and rolling statistics plots for CO(GT).

📘 Notebook: [`notebooks/Phase1_DataPrep_EDA.ipynb`](notebooks/Phase1_DataPrep_EDA.ipynb)

## Next Steps
- Build baseline and Gaussian Process models (TFP).
- Forecast future air quality values with uncertainty intervals.
- Evaluate forecasts with RMSE and visualize results.

---

**Author:** Satyam Rai  
**Repo:** [https://github.com/satyamrai0511/tfp_forecasting](https://github.com/satyamrai0511/tfp_forecasting)
