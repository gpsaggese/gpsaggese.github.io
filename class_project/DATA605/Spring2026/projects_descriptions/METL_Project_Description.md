# METL

## Description
- **METL** (Machine Learning Environment for Time Series) is a specialized tool
  designed for building and evaluating machine learning models specifically for
  time series data.
- It provides a user-friendly interface that simplifies the process of model
  selection, hyperparameter tuning, and evaluation for time series forecasting
  tasks.
- Key features include automated preprocessing, built-in evaluation metrics
  tailored for time series, and the capability to visualize predictions against
  actual data.
- METL supports various forecasting models, including ARIMA, Prophet, and LSTM,
  allowing users to experiment with different approaches easily.
- It facilitates easy integration with popular data manipulation libraries like
  Pandas and NumPy, making data handling intuitive and efficient.

## Project Objective
The goal of this project is to develop a time series forecasting model that
predicts future values of a specific metric (e.g., daily temperature, stock
prices, or sales figures) based on historical data. The project will focus on
optimizing the model's accuracy and minimizing forecasting errors.

## Dataset Suggestions
1. **Global Historical Climatology Network (GHCN) Daily Weather Data**
   - **Source**: National Oceanic and Atmospheric Administration (NOAA)
   - **URL**: [NOAA GHCN](https://www.ncdc.noaa.gov/ghcn-daily-description)
   - **Data**: Daily weather records, including temperature, precipitation, and
     wind speed.
   - **Access Requirements**: Free to use, no authentication required.

2. **Yahoo Finance Stock Prices**
   - **Source**: Yahoo Finance
   - **URL**:
     [Yahoo Finance API](https://query1.finance.yahoo.com/v8/finance/chart/AAPL)
   - **Data**: Historical stock prices, including open, close, high, low, and
     volume.
   - **Access Requirements**: Free to use, no authentication required.

3. **Retail Sales Forecasting Dataset**
   - **Source**: Kaggle
   - **URL**:
     [Kaggle Retail Sales](https://www.kaggle.com/c/retail-sales-forecasting)
   - **Data**: Historical sales data from a retail store, including date, sales
     amount, and promotional events.
   - **Access Requirements**: Free to use with a Kaggle account.

4. **COVID-19 Daily Cases Dataset**
   - **Source**: Johns Hopkins University
   - **URL**:
     [COVID-19 Data Repository](https://github.com/CSSEGISandData/COVID-19)
   - **Data**: Daily reported cases and deaths by country and region.
   - **Access Requirements**: Free to use, no authentication required.

## Tasks
- **Data Acquisition**: Retrieve the selected dataset using METL's built-in
  functionalities or through direct API calls.
- **Data Preprocessing**: Clean and preprocess the time series data, handling
  missing values and formatting issues.
- **Model Selection**: Experiment with different forecasting models (e.g.,
  ARIMA, LSTM) available in METL and select the most appropriate one.
- **Hyperparameter Tuning**: Optimize model parameters using METL's automated
  tuning features to enhance forecasting accuracy.
- **Model Evaluation**: Assess model performance using relevant time series
  metrics (e.g., MAE, RMSE) and visualize predictions against actual data.
- **Reporting**: Compile a comprehensive report detailing the methodology,
  results, and insights gained from the project.

## Bonus Ideas
- Implement ensemble methods by combining predictions from multiple models to
  improve accuracy.
- Explore the impact of external variables (e.g., holidays, economic indicators)
  on the forecasting model.
- Challenge yourself to create a web app using Flask or Streamlit to visualize
  the forecasting results interactively.

## Useful Resources
- [METL Documentation](https://metl.readthedocs.io/en/latest/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [NOAA Data Access](https://www.ncdc.noaa.gov/cdo-web/)
- [Yahoo Finance API Documentation](https://www.yahoofinanceapi.com/)
- [COVID-19 Data GitHub Repository](https://github.com/CSSEGISandData/COVID-19)
