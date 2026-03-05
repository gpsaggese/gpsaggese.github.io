# Tslearn

## Description
- **Time Series Analysis**: tslearn is a Python library specifically designed
  for machine learning on time series data, providing tools for preprocessing,
  feature extraction, and model training.
- **Flexible Algorithms**: It supports a variety of algorithms for
  classification, regression, clustering, and anomaly detection tailored for
  time series data.
- **Distance Measures**: The library includes several distance measures (e.g.,
  Dynamic Time Warping) that are crucial for time series analysis, allowing for
  better model performance.
- **Integration with Scikit-learn**: tslearn is compatible with scikit-learn,
  enabling users to leverage familiar tools and workflows while working with
  time series data.
- **Visualization Tools**: It offers utilities for visualizing time series data
  and model results, making it easier to interpret and communicate findings.

## Project Objective
The goal of the project is to build a time series classification model that
predicts the activity type of individuals based on their accelerometer data.
Students will optimize the model's accuracy and interpretability.

## Dataset Suggestions
1. **UCI Machine Learning Repository - Human Activity Recognition Using
   Smartphones**
   - **URL**:
     [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
   - **Data Contains**: Time series data from accelerometers and gyroscopes of
     smartphones, representing various activities (walking, sitting, standing,
     etc.).
   - **Access Requirements**: No authentication needed; data is freely available
     for download.

2. **Kaggle - Daily Temperature of Major Cities**
   - **URL**:
     [Kaggle Dataset](https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data)
   - **Data Contains**: Daily temperature records for major cities around the
     world, which can be used for time series forecasting and anomaly detection.
   - **Access Requirements**: Free account required to download datasets.

3. **OpenWeatherMap API - Historical Weather Data**
   - **URL**: [OpenWeatherMap API](https://openweathermap.org/history)
   - **Data Contains**: Historical weather data, including temperature,
     humidity, and wind speed, which can be used for time series analysis of
     weather patterns.
   - **Access Requirements**: Free tier available; requires sign-up for an API
     key.

## Tasks
- **Data Acquisition**: Download and preprocess the selected dataset, ensuring
  proper formatting for time series analysis.
- **Exploratory Data Analysis (EDA)**: Visualize the time series data to
  identify trends, seasonality, and anomalies using tslearn's visualization
  tools.
- **Feature Engineering**: Extract relevant features from the time series data,
  such as statistical measures and time-domain features, to enhance model
  performance.
- **Model Training**: Train a classification model using tslearn's algorithms
  (e.g., K-Nearest Neighbors with Dynamic Time Warping) on the prepared dataset.
- **Model Evaluation**: Evaluate the model's performance using appropriate
  metrics (e.g., accuracy, confusion matrix) and interpret the results.
- **Final Report**: Document the methodology, findings, and insights gained from
  the project, including visualizations and model performance metrics.

## Bonus Ideas
- **Hyperparameter Tuning**: Experiment with different hyperparameters for the
  classification model to optimize performance further.
- **Ensemble Methods**: Combine multiple models to improve prediction accuracy
  and robustness.
- **Real-time Prediction**: Implement a simple real-time prediction system that
  can classify activities based on live accelerometer data inputs.
- **Anomaly Detection**: Extend the project to identify anomalous patterns in
  the time series data, such as unusual activity types or environmental
  conditions.

## Useful Resources
- [tslearn Documentation](https://tslearn.readthedocs.io/en/stable/)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [OpenWeatherMap API Documentation](https://openweathermap.org/api)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
