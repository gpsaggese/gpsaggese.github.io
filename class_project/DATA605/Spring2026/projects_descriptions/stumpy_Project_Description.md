# Stumpy

## Description
- Stumpy is a Python library designed for time series analysis, particularly for
  motif discovery and anomaly detection.
- It utilizes state-of-the-art algorithms for computing matrix profiles, which
  provide insights into the similarity of time series data.
- The library is highly optimized for performance, allowing for efficient
  processing of large datasets and real-time applications.
- Stumpy supports various functionalities, including motif discovery, discord
  detection, and subsequence matching, making it versatile for different
  analytical tasks.
- It integrates seamlessly with popular data science libraries like NumPy and
  Pandas, facilitating easy data manipulation and analysis.

## Project Objective
The goal of the project is to detect anomalies in financial time series data
using Stumpy. Students will analyze stock price movements to identify unusual
patterns that could indicate significant market events.

## Dataset Suggestions
1. **Yahoo Finance Stock Data**
   - Source: Yahoo Finance
   - URL:
     [Yahoo Finance API](https://query1.finance.yahoo.com/v8/finance/chart/AAPL)
   - Data: Historical stock prices, including open, high, low, close prices, and
     volume.
   - Access Requirements: No authentication required; can be accessed via simple
     HTTP requests.

2. **Kaggle S&P 500 Stock Prices**
   - Source: Kaggle
   - URL:
     [Kaggle Dataset](https://www.kaggle.com/datasets/cnic92/stock-price-data)
   - Data: Daily stock prices of S&P 500 companies, including adjusted close
     prices.
   - Access Requirements: Free Kaggle account needed for download.

3. **OpenWeatherMap Historical Weather Data**
   - Source: OpenWeatherMap
   - URL: [OpenWeatherMap API](https://openweathermap.org/history)
   - Data: Historical weather data that can be correlated with stock price
     movements.
   - Access Requirements: Free account required for API key.

4. **Quandl Stock Market Data**
   - Source: Quandl
   - URL: [Quandl API](https://www.quandl.com/tools/api)
   - Data: Various datasets related to stock market prices and indices.
   - Access Requirements: Free account needed for API key, but basic datasets
     are accessible without authentication.

## Tasks
- **Data Acquisition**: Use APIs to collect historical stock price data for a
  selected company over a defined period.
- **Data Preprocessing**: Clean and preprocess the data, handling missing values
  and ensuring the time series is properly formatted.
- **Matrix Profile Calculation**: Implement Stumpy to compute the matrix profile
  of the time series data, identifying motifs and discords.
- **Anomaly Detection**: Analyze the results to detect anomalies in stock price
  movements and visualize the findings using plots.
- **Report Findings**: Summarize the analysis in a report, discussing detected
  anomalies and potential implications for investors.

## Bonus Ideas
- Extend the project by comparing the anomaly detection results of Stumpy with
  another time series analysis library (e.g., tslearn).
- Explore additional features in the dataset, such as trading volume or moving
  averages, and assess their impact on anomaly detection.
- Implement a simple dashboard using Streamlit to visualize real-time stock
  price movements and detected anomalies.

## Useful Resources
- [Stumpy Documentation](https://stumpy.readthedocs.io/en/latest/)
- [Yahoo Finance API Documentation](https://www.yahoofinanceapi.com/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [OpenWeatherMap API Documentation](https://openweathermap.org/api)
- [Quandl API Documentation](https://docs.quandl.com/)
