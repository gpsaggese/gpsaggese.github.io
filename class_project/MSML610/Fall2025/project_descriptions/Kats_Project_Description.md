**Description**

Kats is a powerful time series analysis toolkit developed by Facebook, designed for easy and efficient analysis of time series data. It provides a comprehensive set of features for forecasting, anomaly detection, and time series classification, enabling users to tackle various time series tasks with minimal effort.

**Technologies Used**  
Kats  
- Provides a wide range of time series analysis functionalities including forecasting, anomaly detection, and change point detection.  
- Supports multiple forecasting models like ARIMA, Prophet, Holt-Winters, and advanced ML-based models.  
- Offers utilities for data manipulation and visualization, making it easy to analyze and interpret results.  

---

**Project 1: Stock Price Forecasting**  
**Difficulty**: 1 (Easy)  

**Project Objective**:  
Develop a model to forecast future stock prices for a selected company based on historical price data.  

**Dataset Suggestions**:  
- Kaggle: [Tesla Historical Stock Price Data](https://www.kaggle.com/datasets/timoboz/tesla-stock-data-from-2010-to-2020).  

**Tasks**:  
- **Data Collection**: Load Tesla stock price data from Kaggle into a Pandas DataFrame.  
- **Data Preprocessing**: Handle missing values and format timestamps.  
- **Time Series Forecasting**:  
  - Implement Kats’ `ARIMA` model for baseline forecasting.  
  - Compare against `ProphetModel` to capture trends and seasonality.  
- **Model Evaluation**: Evaluate forecasts with MAE and RMSE across both models.  
- **Visualization**: Plot historical vs. predicted prices for both ARIMA and Prophet.  

---

**Project 2: Anomaly Detection in Energy Consumption**  
**Difficulty**: 2 (Medium)  

**Project Objective**:  
Identify anomalies in building energy consumption data to detect unusual usage patterns.  

**Dataset Suggestions**:  
- Kaggle: [Hourly Energy Consumption Dataset](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption).  

**Tasks**:  
- **Data Collection**: Download and load the dataset into a Pandas DataFrame.  
- **Data Preprocessing**: Convert timestamps, handle missing values.  
- **Anomaly Detection**:  
  - Use Kats’ `CUSUMDetector` for detecting sudden shifts.  
  - Apply Kats’ `BOCPDDetector` (Bayesian Online Change Point Detection) for trend-based anomalies.  
  - Experiment with Kats’ `SeasonalHybridESD` for seasonal anomaly detection.  
- **Visualization**: Highlight anomalies from each method on time-series plots.  
- **Report Findings**: Compare results across detectors and discuss business implications.  

---

**Project 3: Multi-Seasonal Time Series Forecasting for Retail Sales**  
**Difficulty**: 3 (Hard)  

**Project Objective**:  
Build a forecasting model to predict future retail sales while accounting for multiple seasonal effects like holidays and promotions.  

**Dataset Suggestions**:  
- Kaggle: [Store Sales - Time Series Forecasting Dataset](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data).  

**Tasks**:  
- **Data Collection**: Load retail sales data from Kaggle.  
- **Data Preprocessing**: Clean data, create holiday and promotion features, and encode categorical variables.  
- **Multi-Seasonal Forecasting**:  
  - Implement Kats’ `ProphetModel` with holiday regressors.  
  - Compare against `HoltWintersModel` to capture multiple seasonal cycles.  
  - Experiment with Kats’ `ARIMA` or `SARIMA` for strong seasonal patterns.  
- **Model Evaluation**: Compare model performance with MAE and RMSE.  
- **Visualization**: Plot forecasts from each model alongside historical sales data.  

---

**Bonus Ideas (Optional):**  
- For Project 1: Add features like trading volume or sentiment indicators and test Kats’ MLForecast model.  
- For Project 2: Build an ensemble anomaly detector that combines results from multiple models.  
- For Project 3: Incorporate external economic indicators (e.g., inflation, holidays, promotions) and test hybrid models.  
