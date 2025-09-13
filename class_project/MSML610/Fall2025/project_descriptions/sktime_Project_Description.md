**Description**

In this project, students will utilize sktime, a Python library designed for time series analysis and forecasting. This tool provides a unified interface for a variety of time series tasks, including classification, regression, and forecasting. It also supports the integration of machine learning models and offers utilities for preprocessing and feature extraction tailored specifically for time series data.

Technologies Used
sktime

- Provides a consistent interface for time series analysis, making it easy to switch between models and tasks.
- Supports a wide range of algorithms for time series classification, regression, and forecasting.
- Includes utilities for time series preprocessing, feature extraction, and transformation.

---

### Project 1: Time Series Forecasting of Air Quality (Difficulty: 1)

**Project Objective**: The goal is to predict future air quality index (AQI) levels based on historical data, optimizing for accuracy in forecasting.

**Dataset Suggestions**: 
- Use the "Air Quality Data Set" available on Kaggle ([Kaggle Air Quality Dataset](https://www.kaggle.com/datasets/uciml/air-quality-data-set)).
  
**Tasks**:
- **Data Ingestion**: Load the air quality dataset and understand its structure.
- **Preprocessing**: Handle missing values and normalize the data for better model performance.
- **Feature Engineering**: Extract relevant features such as moving averages or seasonal trends from the time series data.
- **Model Selection**: Choose a forecasting model from sktime (e.g., ARIMA, Exponential Smoothing).
- **Training and Evaluation**: Split the dataset into training and testing sets, train the model, and evaluate using metrics like MAE or RMSE.
- **Visualization**: Plot the actual vs. predicted AQI levels to visualize forecasting performance.

---

### Project 2: Stock Price Movement Classification (Difficulty: 2)

**Project Objective**: The aim is to classify the future movement of stock prices (up or down) based on historical price data, optimizing for classification accuracy.

**Dataset Suggestions**: 
- Use the "Historical Stock Prices" dataset available on Yahoo Finance API (free and active) for a specific stock (e.g., Apple Inc. - AAPL).

**Tasks**:
- **Data Collection**: Fetch historical stock price data using the Yahoo Finance API.
- **Data Preprocessing**: Clean the dataset, handle missing values, and create labels for price movement (up/down).
- **Feature Extraction**: Generate features such as daily returns, moving averages, and volatility indicators.
- **Model Development**: Implement a classification model using sktime (e.g., Time Series Forest).
- **Training and Evaluation**: Train the model and evaluate it using accuracy, precision, and recall metrics.
- **Comparison**: Compare the performance of different classification algorithms available in sktime.

**Bonus Ideas**: 
- Explore ensemble methods to improve classification performance.
- Implement a strategy to optimize hyperparameters of the classification model.

---

### Project 3: Anomaly Detection in Energy Consumption Data (Difficulty: 3)

**Project Objective**: The goal is to detect anomalies in energy consumption patterns over time, optimizing for identifying unusual spikes or drops in usage.

**Dataset Suggestions**: 
- Use the "Energy Consumption" dataset from the UCI Machine Learning Repository ([UCI Energy Consumption Dataset](https://archive.ics.uci.edu/ml/datasets/Energy+consumption+of+the+household)).

**Tasks**:
- **Data Ingestion**: Load the energy consumption dataset and explore its characteristics.
- **Preprocessing**: Clean the dataset, handle missing values, and resample the data if necessary.
- **Feature Engineering**: Create time-based features such as hour of the day, day of the week, and rolling statistics.
- **Anomaly Detection Model**: Implement a time series anomaly detection model using sktime (e.g., Seasonal Hybrid ESD).
- **Evaluation**: Use metrics such as precision and recall to evaluate the model's performance in detecting anomalies.
- **Visualization**: Visualize detected anomalies on the time series plot to analyze their context.

**Bonus Ideas**: 
- Experiment with different anomaly detection techniques available in sktime.
- Implement a dashboard to visualize energy consumption and detected anomalies in real-time.

