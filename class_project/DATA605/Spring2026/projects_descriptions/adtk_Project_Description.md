# Adtk

## Description
- **ADTK (Anomaly Detection ToolKit)** is a Python library designed for
  detecting anomalies in time series data.
- It provides a variety of algorithms for anomaly detection, including
  statistical methods and machine learning approaches.
- The toolkit supports easy integration with popular data manipulation libraries
  like Pandas and NumPy, making it user-friendly for data scientists.
- ADTK allows for the visualization of detected anomalies, helping users
  understand the nature and context of the anomalies in their datasets.
- The library is modular, enabling users to customize detection pipelines by
  combining different detection methods and preprocessing steps.

## Project Objective
The goal of this project is to build an anomaly detection system that identifies
unusual patterns in time series data from a public dataset. Students will
optimize their models to accurately detect anomalies while minimizing false
positives and false negatives.

## Dataset Suggestions
1. **Air Quality Data**
   - **Source**: UCI Machine Learning Repository
   - **URL**:
     [Air Quality Data Set](https://archive.ics.uci.edu/ml/datasets/Air+Quality)
   - **Data Contains**: Time series data of air quality measurements, including
     levels of various pollutants.
   - **Access Requirements**: No authentication required; data is freely
     downloadable.

2. **Electricity Consumption Data**
   - **Source**: Kaggle
   - **URL**:
     [Electricity Consumption Dataset](https://www.kaggle.com/datasets/uciml/electricity-consumption)
   - **Data Contains**: Hourly electricity consumption data from various
     households over time.
   - **Access Requirements**: Free to use; requires a Kaggle account for
     download.

3. **Stock Prices Data**
   - **Source**: Alpha Vantage
   - **URL**: [Alpha Vantage API](https://www.alphavantage.co/documentation/)
   - **Data Contains**: Historical stock prices and trading volumes for various
     companies.
   - **Access Requirements**: Free API key required (no paid plans).

4. **Traffic Volume Data**
   - **Source**: City of Chicago Data Portal
   - **URL**:
     [Traffic Volume Counts](https://data.cityofchicago.org/Transportation/Traffic-Volume-Counts/7f9d-2m8u)
   - **Data Contains**: Time series data of traffic volume counts at various
     locations in Chicago.
   - **Access Requirements**: Publicly available without authentication.

## Tasks
- **Data Acquisition**: Download and load the selected dataset into a Pandas
  DataFrame for analysis.
- **Data Preprocessing**: Clean the dataset by handling missing values and
  converting data types as needed.
- **Exploratory Data Analysis (EDA)**: Visualize the time series data to
  understand trends, seasonality, and potential anomalies.
- **Anomaly Detection Implementation**: Utilize ADTK to apply various anomaly
  detection algorithms to the dataset.
- **Model Evaluation**: Assess the performance of the anomaly detection methods
  using appropriate metrics (e.g., precision, recall).
- **Visualization of Results**: Create visualizations to highlight detected
  anomalies and compare them with the original time series data.

## Bonus Ideas
- **Compare Different Algorithms**: Implement and compare multiple anomaly
  detection algorithms provided by ADTK to see which performs best on the
  dataset.
- **Hyperparameter Tuning**: Experiment with different hyperparameters for the
  chosen algorithms to optimize performance.
- **Real-time Anomaly Detection**: Simulate a real-time anomaly detection system
  using streaming data (e.g., updating the model with new data periodically).
- **Integration with Alert Systems**: Create a simple alert mechanism that
  notifies users when anomalies are detected.

## Useful Resources
- [ADTK Documentation](https://adtk.readthedocs.io/en/latest/)
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Alpha Vantage API Documentation](https://www.alphavantage.co/documentation/)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
