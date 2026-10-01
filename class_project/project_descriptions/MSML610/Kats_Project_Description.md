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

- Implement the tutorial "Learn Kats in 60 mins", following
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - Build it with `.claude/skills/tutorial_in_60_mins.create/SKILL.md`
  - Follow the workflow in `tutorials/README.gp.md` and the quality principles in
    `tutorials/tutorials_checklist.md`
- Check the previous tutorials and projects, listed in the section
  `Existing Tutorials and Projects` of
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - No earlier tutorial or project uses Kats, so read the closest forecasting work
  - Read `tutorials/Prophet/README.md`
  - Read the `README.md` of the Fall2025 Prophet project, and reuse what is good
    - `class_project/msml610/Fall2025/projects/Tutortask41_Fall2025_prophet_COVID_19_Case_Prediction/`
- Create the project dir following the class instructions in
  `class_project/README.md`, section `Contribution to the Repo`
  - Start from `class_project/project_template`
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Compare briefly with `tutorials/Prophet/` from the forecasting point of view
- Deliverables:
  - `kats_utils.py`
  - `kats.API.ipynb`
  - `kats.example.ipynb`

# Project

## Project 1 (Fall2026): Forecasting US Inflation and Unemployment with Rolling Evaluation

- **Project Objective**: Forecast the monthly US inflation rate and the unemployment
  rate 12 months ahead with several Kats models and an ensemble, and check if they
  beat a seasonal naive baseline in a rolling-origin evaluation
- **Dataset Suggestions**:
  - [CPI for All Urban Consumers, not seasonally adjusted (CPIAUCNS)](https://fred.stlouisfed.org/series/CPIAUCNS)
  - [Unemployment Rate, not seasonally adjusted (UNRATENSA)](https://fred.stlouisfed.org/series/UNRATENSA)
- **Tasks**:
  - **Preprocess the Data**: Download both series from FRED, compute the monthly
    inflation rate as the percent change of the CPI index, and convert each series
    into a `TimeSeriesData` object
  - **Define the Forecasting Problem**: Forecast 12 months ahead from yearly origins
    from 2015 to 2024, fitting only on the data before each origin, and compute the
    last-value and seasonal naive baselines
  - **Forecast with Kats Models**: Fit `SARIMAModel`, `HoltWintersModel`,
    `ProphetModel`, and `ThetaModel` at each origin, and forecast the next 12 months
  - **Combine the Models**: Average the forecasts with equal weights, and build a
    weighted ensemble with `KatsEnsemble`
  - **Evaluate the Forecasts**: Compare MAE, RMSE, and MASE by horizon, the coverage
    of the 80% prediction intervals, and the share of origins in which each model
    beats the seasonal naive baseline
  - **Visualize the Forecasts**: Plot the forecasts against the actual values for the
    2021 origin, when inflation surged, and plot the error by horizon
- **Bonus Ideas (Optional)**: Repeat the evaluation with the real-time vintages of
  [ALFRED](https://alfred.stlouisfed.org/) to remove the effect of data revisions;
  add the Kats `LinearModel` trend model as another baseline

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Preprocess the Data
  - Result: project dir created and container running, and the monthly US inflation
    rate and unemployment rate as clean `TimeSeriesData` objects
- Milestone 2: API notebook
  - Project tasks: Forecast with Kats Models, Combine the Models
  - Result: `kats.API.ipynb` covering `TimeSeriesData`, `SARIMAModel`,
    `HoltWintersModel`, `ProphetModel`, `ThetaModel`, and `KatsEnsemble` on a
    synthetic series, and one change point detector
- Milestone 3: Example notebook
  - Project tasks: Define the Forecasting Problem, Forecast with Kats Models, Combine
    the Models, Evaluate the Forecasts, Visualize the Forecasts
  - Result: `kats.example.ipynb` running end to end

## Project 2: Stock Price Forecasting

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

## Project 3: Change Point Detection in Market Volatility and the Yield Curve

- **Project Objective**: Detect regime shifts in the VIX and in the 10-year minus
  2-year Treasury spread with Kats detectors, and check which detector flags known
  stress events early with few false alarms, compared with a rolling z-score rule
- **Dataset Suggestions**:
  - [CBOE Volatility Index: VIX (VIXCLS)](https://fred.stlouisfed.org/series/VIXCLS)
  - [10-Year minus 2-Year Treasury Spread (T10Y2Y)](https://fred.stlouisfed.org/series/T10Y2Y)
  - [NBER Recession Indicator (USREC)](https://fred.stlouisfed.org/series/USREC)
- **Tasks**:
  - **Preprocess the Data**: Download the daily series from FRED for 2015 to 2025,
    drop the days without a value, and convert each one into a `TimeSeriesData`
    object
  - **Define the Detection Problem**: Fix a reference list of stress events, e.g.,
    Aug 2015, Feb 2018, Mar 2020, Mar 2023, and Aug 2024, and count a detection as
    correct if it falls within 10 trading days of an event
  - **Build a Baseline**: Flag a change when the 20-day rolling z-score of the series
    exceeds 3
  - **Detect Change Points**: Run `CUSUMDetector` on rolling windows, `BOCPDetector`
    online, and `RobustStatDetector`, using only the data up to each date, and use
    `OutlierDetector` for one-day spikes
  - **Evaluate the Detectors**: Report precision, recall, F1, and the median
    detection delay in days of each detector and of the baseline
  - **Visualize the Detections**: Plot each series with the detected points, the
    event dates, and the recession bands
- **Bonus Ideas (Optional)**: Apply `OutlierDetector` to the daily trading volume of
  Bitcoin (`BTC-USD` from `yfinance`); build an ensemble that flags a stress event
  when two detectors agree
