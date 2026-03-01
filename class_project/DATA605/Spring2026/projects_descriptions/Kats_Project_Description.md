# Kats

## Description
- Kats is a lightweight, open-source time series analysis toolkit developed by
  Facebook.
- It provides a comprehensive set of tools for time series forecasting, anomaly
  detection, and change point detection.
- Kats supports multiple models, including ARIMA, Prophet, and Exponential
  Smoothing, making it versatile for various forecasting tasks.
- The library includes utilities for data preprocessing, feature extraction, and
  visualization, facilitating a complete workflow.
- Kats is designed to be user-friendly, with a straightforward API that allows
  users to quickly implement complex time series analyses.

## Project Objective
The goal of this project is to forecast future sales of a retail store using
historical sales data. Students will optimize the accuracy of their forecasts by
selecting appropriate models and tuning their parameters, while also detecting
any anomalies in the sales data that could indicate unusual events or trends.

## Dataset Suggestions
1. **Retail Sales Data**
   - **Source**: Kaggle
   - **URL**:
     [Retail Sales Forecasting](https://www.kaggle.com/c/store-sales-time-series-forecasting/data)
   - **Data Contains**: Historical sales data for various stores, including
     sales amounts, item categories, and store locations.
   - **Access Requirements**: Free to use after creating a Kaggle account.

2. **Global Retail Sales Data**
   - **Source**: Kaggle
   - **URL**:
     [Global Superstore Dataset](https://www.kaggle.com/datasets/irfanasrullah/global-superstore-dataset)
   - **Data Contains**: A comprehensive dataset of global retail sales,
     including order details, customer information, and shipping details.
   - **Access Requirements**: Free to use after creating a Kaggle account.

3. **UCI Machine Learning Repository - Online Retail**
   - **Source**: UCI Machine Learning Repository
   - **URL**:
     [Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail)
   - **Data Contains**: Transactions from a UK-based online retailer, including
     invoice numbers, stock codes, quantities, and invoice dates.
   - **Access Requirements**: Publicly accessible without registration.

4. **M5 Forecasting - Accuracy**
   - **Source**: Kaggle
   - **URL**:
     [M5 Forecasting - Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy/data)
   - **Data Contains**: Sales data for thousands of products across multiple
     stores, including historical sales and promotional events.
   - **Access Requirements**: Free to use after creating a Kaggle account.

## Tasks
- **Data Exploration**: Load the dataset and conduct exploratory data analysis
  (EDA) to understand trends, seasonality, and anomalies.
- **Preprocessing**: Clean the data by handling missing values, converting date
  formats, and aggregating sales data as necessary.
- **Model Selection**: Choose appropriate forecasting models (e.g., ARIMA,
  Prophet) and implement them using Kats.
- **Anomaly Detection**: Utilize Kats' anomaly detection features to identify
  unusual sales patterns in the historical data.
- **Model Evaluation**: Split the data into training and testing sets, evaluate
  model performance using metrics such as RMSE and MAE, and visualize the
  results.
- **Reporting**: Summarize findings, including model performance, detected
  anomalies, and insights derived from the analysis, in a final report.

## Bonus Ideas
- **Model Comparison**: Implement additional forecasting models and compare
  their performance against the chosen models.
- **Hyperparameter Tuning**: Explore hyperparameter optimization techniques to
  improve model accuracy.
- **Seasonal Decomposition**: Apply seasonal decomposition to better understand
  trends and seasonality in the sales data.
- **Visualization Enhancements**: Create advanced visualizations to communicate
  findings effectively, using libraries like Matplotlib or Seaborn.

## Useful Resources
- [Kats Documentation](https://facebookresearch.github.io/Kats/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Time Series Forecasting with Kats](https://github.com/facebookresearch/Kats/blob/main/docs/tutorials/Forecasting%20with%20Kats.ipynb)
- [Kats GitHub Repository](https://github.com/facebookresearch/Kats)
