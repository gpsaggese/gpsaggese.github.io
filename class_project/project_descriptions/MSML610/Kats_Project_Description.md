# Description

Kats is a time series analysis toolkit developed by Facebook Research. It solves the
problem of using many forecasting models, anomaly and change point detectors, and
feature extractors through one `TimeSeriesData` interface. It is worth a 60-minute
tutorial because a student can forecast a series with several models and detect
anomalies on the same series with a few lines of code.

## Technologies Used

Kats

- Provides a wide range of time series analysis functionalities including
  forecasting, anomaly detection, and change point detection
- Supports multiple forecasting models like ARIMA, Prophet, Holt-Winters, and
  advanced ML-based models
- Offers utilities for data manipulation and visualization, making it easy to analyze
  and interpret results

# Tutorial

- Usual tutorial "Learn Kats in 60 mins", following
  `.claude/skills/tutorials_in_60_mins.rules.md`
- Create `tutorials/Kats/` with `.claude/skills/tutorials_in_60_mins.create/SKILL.md`
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Compare briefly with `tutorials/Prophet/` from the forecasting point of view
- Deliverables:
  - `kats_utils.py`
  - `kats.API.ipynb`
  - `kats.example.ipynb`

# Project

## Project 1: Stock Price Forecasting

- **Difficulty**: 2 (Medium)
- **Project Objective**: Develop a model to forecast future stock prices for a
  selected company from historical price data, and choose between a baseline model
  and a trend and seasonality model by forecast error
- **Dataset Suggestions**:
  [Tesla Historical Stock Price Data](https://www.kaggle.com/datasets/timoboz/tesla-stock-data-from-2010-to-2020)
- **Tasks**:
  - **Preprocess the Data**: Load the Tesla prices into a pandas DataFrame, handle
    missing values, and convert the timestamps into a `TimeSeriesData` object
  - **Define the Forecasting Problem**: Forecast the closing price for a fixed
    horizon with a chronological train and test split
  - **Forecast with Kats**: Fit `ARIMAModel` as the baseline and `ProphetModel` to
    capture trend and seasonality, and forecast the test horizon with both
  - **Evaluate the Forecasts**: Compare MAE and RMSE of both models on the test
    horizon
  - **Visualize the Forecasts**: Plot historical vs. predicted prices for both ARIMA
    and Prophet, with their prediction intervals
- **Bonus Ideas (Optional)**: Repeat the same analysis for multiple stocks and
  different forecasting intervals; add trading volume as a feature and test the Kats
  `MLARModel`

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Preprocess the Data
  - Result: `tutorials/Kats/` container running, and the Tesla prices as a clean
    `TimeSeriesData` object with a train and test split
- Milestone 2: API notebook
  - Project tasks: Forecast with Kats
  - Result: `kats.API.ipynb` covering `TimeSeriesData`, `ARIMAModel`, `ProphetModel`,
    and `HoltWintersModel` on a synthetic series, and one change point detector
- Milestone 3: Example notebook
  - Project tasks: Define the Forecasting Problem, Forecast with Kats, Evaluate the
    Forecasts, Visualize the Forecasts
  - Result: `kats.example.ipynb` running end to end

## Project 2: Anomaly Detection in Energy Consumption

- **Difficulty**: 2 (Medium)
- **Project Objective**: Identify anomalies in building energy consumption data to
  detect unusual usage patterns
- **Dataset Suggestions**:
  [Hourly Energy Consumption Dataset](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
- **Tasks**:
  - **Preprocess the Data**: Download the dataset, load it into a pandas DataFrame,
    convert the timestamps, and handle missing values
  - **Detect Anomalies**: Use `CUSUMDetector` for sudden shifts, `BOCPDetector`
    (Bayesian Online Change Point Detection) for trend changes, and `OutlierDetector`
    for seasonal outliers
  - **Visualize Anomalies**: Highlight the anomalies of each method on time-series
    plots
  - **Report Findings**: Compare the results across detectors and discuss business
    implications
- **Bonus Ideas (Optional)**: Build an ensemble anomaly detector that combines
  results from multiple models

## Project 3: Multi-Seasonal Time Series Forecasting for Retail Sales

- **Difficulty**: 3 (Hard)
- **Project Objective**: Build a forecasting model to predict future retail sales
  while accounting for multiple seasonal effects like holidays and promotions
- **Dataset Suggestions**:
  [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)
- **Tasks**:
  - **Preprocess the Data**: Load the retail sales data, clean it, create holiday and
    promotion features, and encode categorical variables
  - **Forecast Multiple Seasonalities**: Fit `ProphetModel` with holiday regressors,
    `HoltWintersModel` for multiple seasonal cycles, and `SARIMAModel` for strong
    seasonal patterns
  - **Evaluate the Models**: Compare MAE and RMSE of the models
  - **Visualize the Forecasts**: Plot the forecasts of each model alongside the
    historical sales data
- **Bonus Ideas (Optional)**: Incorporate external economic indicators (e.g.,
  inflation) and test hybrid models
