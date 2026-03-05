```
# sktime

## Description
- sktime is a Python library designed specifically for time series analysis and machine learning, providing a unified framework for handling various time series tasks.
- It supports a wide range of time series algorithms, including classification, regression, clustering, and forecasting, making it versatile for different applications.
- The library offers tools for preprocessing time series data, such as transformation, feature extraction, and time series cross-validation.
- sktime is built on top of popular libraries like scikit-learn, allowing users to leverage familiar interfaces while working with time series data.
- It includes a collection of benchmark datasets for time series tasks, enabling easy experimentation and comparison of models.

## Project Objective
The goal of the project is to build a time series forecasting model that predicts future values of a given time series dataset. Students will optimize their models for accuracy using appropriate evaluation metrics.

## Dataset Suggestions
1. **Air Quality Data**
   - Source: UCI Machine Learning Repository
   - URL: https://archive.ics.uci.edu/ml/datasets/Air+Quality
   - Data Contains: Hourly averaged concentrations of air pollutants and meteorological data.
   - Access Requirements: Publicly accessible, no authentication needed.

2. **Electricity Consumption**
   - Source: Kaggle
   - URL: https://www.kaggle.com/datasets/uciml/electricity-consumption-data
   - Data Contains: Electricity consumption readings over time for different households.
   - Access Requirements: Free to use with a Kaggle account.

3. **Stock Prices**
   - Source: Yahoo Finance (via yfinance library)
   - URL: https://pypi.org/project/yfinance/
   - Data Contains: Historical stock price data including open, high, low, close prices, and volume.
   - Access Requirements: Publicly accessible via the yfinance library, no authentication needed.

4. **COVID-19 Daily Cases**
   - Source: Johns Hopkins University
   - URL: https://github.com/CSSEGISandData/COVID-19
   - Data Contains: Daily confirmed COVID-19 cases by country and region.
   - Access Requirements: Publicly available, no authentication needed.

## Tasks
- **Data Preprocessing**: Load the selected dataset, handle missing values, and perform necessary transformations to prepare the data for analysis.
- **Feature Engineering**: Extract relevant features from the time series data, such as lagged values, rolling statistics, or seasonal indicators.
- **Model Selection**: Choose suitable forecasting models from sktime, such as ARIMA, Exponential Smoothing, or machine learning models.
- **Model Training**: Train the selected model(s) on the training dataset and fine-tune hyperparameters for optimal performance.
- **Evaluation**: Assess model performance using appropriate metrics (e.g., RMSE, MAE) and visualize the results against actual values.
- **Reporting**: Create a comprehensive report summarizing findings, model performance, and insights gained from the analysis.

## Bonus Ideas
- Implement ensemble methods by combining predictions from multiple models to improve forecasting accuracy.
- Explore the impact of external factors (e.g., economic indicators) on the time series and incorporate them into the model.
- Challenge yourself to apply anomaly detection techniques on the time series data to identify unusual patterns or outliers.

## Useful Resources
- [sktime Documentation](https://www.sktime.org/en/stable/)
- [Time Series Forecasting with sktime](https://towardsdatascience.com/time-series-forecasting-with-sktime-7e8b5f5b5c8b)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [yfinance GitHub Repository](https://github.com/ranaroussi/yfinance)
```
