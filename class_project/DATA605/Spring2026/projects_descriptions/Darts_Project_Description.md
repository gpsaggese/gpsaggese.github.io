# Darts

## Description
- Darts is an open-source Python library designed for easy manipulation and
  forecasting of time series data.
- It provides a unified framework for various forecasting models, including
  ARIMA, Exponential Smoothing, and machine learning-based approaches.
- The library supports both univariate and multivariate time series forecasting,
  allowing for flexible model selection based on data characteristics.
- Darts includes built-in evaluation metrics and tools for backtesting, making
  it easier to assess model performance on historical data.
- It allows users to create ensembles of different forecasting models to improve
  accuracy and robustness.
- The library is user-friendly and integrates seamlessly with popular data
  manipulation libraries like Pandas.

## Project Objective
The goal of this project is to develop a time series forecasting model that
predicts future values of a given dataset. Students will focus on optimizing the
model's accuracy by comparing different forecasting techniques available in
Darts.

## Dataset Suggestions
1. **Air Quality Data**
   - **Source**: UCI Machine Learning Repository
   - **URL**:
     [Air Quality Data Set](https://archive.ics.uci.edu/ml/datasets/Air+Quality)
   - **Data Contains**: Hourly averaged concentrations of various air pollutants
     and meteorological data.
   - **Access Requirements**: Publicly available for download without
     authentication.

2. **Global Temperature Time Series**
   - **Source**: NOAA National Centers for Environmental Information
   - **URL**: [Global Temperature Data](https://datahub.io/core/global-temp)
   - **Data Contains**: Monthly global land and ocean temperature anomalies from
     1880 to present.
   - **Access Requirements**: Publicly accessible CSV files.

3. **COVID-19 Time Series Data**
   - **Source**: Johns Hopkins University
   - **URL**:
     [COVID-19 Data Repository](https://github.com/CSSEGISandData/COVID-19)
   - **Data Contains**: Daily reported cases and deaths globally, along with
     demographic information.
   - **Access Requirements**: Publicly available on GitHub.

4. **Stock Prices Dataset**
   - **Source**: Yahoo Finance (via yfinance library)
   - **URL**: [Yahoo Finance API](https://pypi.org/project/yfinance/)
   - **Data Contains**: Historical stock prices for various companies, including
     open, high, low, close prices, and volume.
   - **Access Requirements**: Publicly accessible through the yfinance library
     without authentication.

## Tasks
- **Data Collection**: Use the selected dataset to gather time series data and
  prepare it for analysis.
- **Data Preprocessing**: Clean the dataset, handle missing values, and
  transform the data into a suitable format for forecasting.
- **Model Selection**: Explore different forecasting models available in Darts
  (e.g., ARIMA, Exponential Smoothing, and machine learning models).
- **Model Training**: Train selected models on historical data and optimize
  hyperparameters for better performance.
- **Evaluation**: Use built-in evaluation metrics in Darts to assess model
  accuracy and compare results across different models.
- **Visualization**: Visualize the forecasted results alongside historical data
  to illustrate model performance and trends.

## Bonus Ideas
- Implement an ensemble model that combines predictions from multiple
  forecasting models to enhance accuracy.
- Explore advanced techniques like hyperparameter tuning using grid search or
  random search for model optimization.
- Compare the performance of Darts with other forecasting libraries (e.g.,
  Facebook Prophet, Statsmodels) using the same dataset.
- Investigate the impact of external factors (e.g., holidays, special events) on
  forecasting accuracy.

## Useful Resources
- [Darts Documentation](https://github.com/unit8co/darts)
- [Darts GitHub Repository](https://github.com/unit8co/darts)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [NOAA National Centers for Environmental Information](https://www.ncdc.noaa.gov/)
- [yfinance Documentation](https://pypi.org/project/yfinance/)
