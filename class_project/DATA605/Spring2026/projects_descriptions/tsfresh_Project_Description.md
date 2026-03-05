# Tsfresh

## Description
- **Feature Extraction**: tsfresh is a Python package designed for automatic
  extraction of relevant features from time series data, making it easier to
  prepare data for machine learning models.
- **Statistical Methods**: It employs a variety of statistical methods to
  compute features, including time-domain and frequency-domain analysis, which
  helps in capturing different aspects of time series data.
- **Feature Selection**: The tool includes capabilities for feature selection,
  allowing users to identify and retain only the most informative features for
  predictive modeling.
- **Integration with Machine Learning**: tsfresh is compatible with popular
  machine learning libraries like scikit-learn, enabling seamless integration
  into existing workflows for classification, regression, or forecasting tasks.
- **Handling Large Datasets**: It is optimized for performance and can
  efficiently handle large datasets, making it suitable for real-world
  applications in various domains such as finance, healthcare, and IoT.

## Project Objective
The goal of this project is to predict the occurrence of anomalies in time
series data collected from a public dataset. Students will optimize a model to
accurately classify time series segments as either normal or anomalous based on
the features extracted using tsfresh.

## Dataset Suggestions
1. **NASA Turbofan Engine Degradation Simulation Data Set**
   - **Source**: NASA Prognostics Data Repository
   - **URL**:
     https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository
   - **Data Contains**: Time series data simulating the degradation of turbofan
     engines over time, with labeled anomalies.
   - **Access Requirements**: Publicly available, no authentication required.

2. **ECG Heartbeat Classification Dataset**
   - **Source**: PhysioNet
   - **URL**: https://physionet.org/static/published-projects/haq/
   - **Data Contains**: Time series ECG data with labeled normal and abnormal
     heartbeats.
   - **Access Requirements**: Publicly available, no authentication required.

3. **Yahoo Finance Stock Market Data**
   - **Source**: Yahoo Finance API (via yfinance Python package)
   - **URL**: https://pypi.org/project/yfinance/
   - **Data Contains**: Historical stock prices and trading volumes for various
     companies, which can be analyzed for anomalies.
   - **Access Requirements**: Publicly available, no authentication required.

## Tasks
- **Data Collection**: Use the provided APIs or datasets to gather time series
  data relevant to the project objective.
- **Feature Extraction**: Utilize tsfresh to automatically extract a
  comprehensive set of features from the time series data.
- **Feature Selection**: Implement feature selection techniques available in
  tsfresh to identify the most relevant features for anomaly detection.
- **Model Training**: Train a supervised machine learning model (e.g., Random
  Forest, SVM) using the selected features to classify time series segments.
- **Model Evaluation**: Assess the model's performance using appropriate metrics
  (e.g., accuracy, precision, recall) and visualize the results.
- **Analysis and Reporting**: Analyze the extracted features and model
  performance, and prepare a report summarizing the findings and insights.

## Bonus Ideas
- **Hyperparameter Tuning**: Experiment with hyperparameter tuning techniques to
  optimize model performance further.
- **Ensemble Methods**: Combine multiple models to improve prediction accuracy
  and robustness.
- **Real-Time Anomaly Detection**: Extend the project to implement a simple
  real-time anomaly detection system using streaming data.
- **Visualization**: Create visualizations to represent the time series data,
  extracted features, and model predictions.

## Useful Resources
- [tsfresh Documentation](https://tsfresh.readthedocs.io/en/latest/)
- [NASA Prognostics Data Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)
- [PhysioNet ECG Data](https://physionet.org/static/published-projects/haq/)
- [yfinance GitHub Repository](https://github.com/ranaroussi/yfinance)
- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
