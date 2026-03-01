# Pmdarima

## Description
- **pmdarima** is a Python library designed for easy implementation of ARIMA
  (AutoRegressive Integrated Moving Average) models for time series forecasting.
- It simplifies the process of fitting ARIMA models by automating the selection
  of the best model parameters using the **auto_arima** function.
- The library includes features for seasonal decomposition and diagnostics,
  making it easier to analyze time series data.
- It supports integration with pandas for seamless data manipulation and time
  series analysis.
- Pmdarima is built on top of statsmodels, providing a robust statistical
  foundation for time series forecasting.

## Project Objective
The goal of the project is to develop a time series forecasting model using
pmdarima to predict future values of a specified time series dataset. Students
will optimize the model to achieve the lowest possible forecast error, measured
using metrics such as Mean Absolute Error (MAE) or Mean Squared Error (MSE).

## Dataset Suggestions
1. **Air Quality Data**
   - **Source**: UCI Machine Learning Repository
   - **URL**:
     [Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality)
   - **Data Contains**: Hourly averaged responses from an array of gas sensors
     in a specific location.
   - **Access Requirements**: Publicly available without authentication.

2. **Global Temperature Time Series**
   - **Source**: Kaggle
   - **URL**:
     [Global Temperature](https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data)
   - **Data Contains**: Monthly and yearly global temperature averages from 1750
     to present.
   - **Access Requirements**: Free to use with a Kaggle account.

3. **Stock Market Data**
   - **Source**: Yahoo Finance (via yfinance library)
   - **URL**: [Yahoo Finance API](https://pypi.org/project/yfinance/)
   - **Data Contains**: Historical stock prices, including open, high, low,
     close, and volume for various stocks.
   - **Access Requirements**: Free to use, no authentication needed.

4. **COVID-19 Daily Cases**
   - **Source**: Johns Hopkins University
   - **URL**:
     [COVID-19 Data Repository](https://github.com/CSSEGISandData/COVID-19)
   - **Data Contains**: Daily confirmed cases and deaths globally, allowing for
     time series analysis.
   - **Access Requirements**: Publicly available on GitHub.

## Tasks
- **Data Collection**: Load the chosen dataset using pandas, ensuring it is in a
  time series format.
- **Exploratory Data Analysis (EDA)**: Visualize the data to identify trends,
  seasonality, and potential outliers.
- **Preprocessing**: Clean the data, handle missing values, and transform it if
  necessary (e.g., applying differencing for stationarity).
- **Model Selection**: Use the `auto_arima` function to find the best ARIMA
  model parameters (p, d, q) based on AIC/BIC criteria.
- **Model Training**: Fit the selected ARIMA model to the training data and
  generate forecasts for the test set.
- **Evaluation**: Assess the model's performance using metrics such as MAE and
  visualize the forecast against actual values.

## Bonus Ideas
- **Feature Engineering**: Explore additional features such as lagged variables
  or moving averages to improve model performance.
- **Multiple Time Series**: Extend the project to forecast multiple time series
  simultaneously and compare their performance.
- **Ensemble Methods**: Combine the ARIMA model with other forecasting
  techniques (e.g., Exponential Smoothing) and evaluate the ensemble's
  performance.
- **Hyperparameter Tuning**: Experiment with different seasonal parameters and
  model configurations to optimize forecasting accuracy.

## Useful Resources
- [pmdarima Documentation](https://pmdarima.readthedocs.io/en/latest/)
- [ARIMA Model in Python](https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Yahoo Finance API Documentation](https://pypi.org/project/yfinance/)
- [Time Series Analysis with Python](https://towardsdatascience.com/time-series-analysis-with-python-5f8f4c7a8c7e)
