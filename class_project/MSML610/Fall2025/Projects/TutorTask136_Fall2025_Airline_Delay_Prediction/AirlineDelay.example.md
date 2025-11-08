<!-- toc -->

- [Project 3: Airline Delay Prediction](#project-3-airline-delay-prediction)
  * [Project Overview](#project-overview)
  * [Problem Statement](#problem-statement)
  * [Dataset](#dataset)
  * [Methodology](#methodology)
    + [1. Data Preprocessing](#1-data-preprocessing)
    + [2. Feature Engineering](#2-feature-engineering)
    + [3. Model Training](#3-model-training)
    + [4. Evaluation](#4-evaluation)
    + [5. Visualization](#5-visualization)
  * [Architecture](#architecture)
  * [Implementation Details](#implementation-details)
  * [Results and Insights](#results-and-insights)
  * [Bonus Features](#bonus-features)
  * [Conclusion](#conclusion)

<!-- tocstop -->

# Project 3: Airline Delay Prediction

## Project Overview

This project develops a **classification model using XGBoost** to predict whether a flight will be delayed based on historical flight and weather data. Airline delays disrupt travel plans, cause financial losses, and challenge air traffic operations. By applying machine learning, we aim to assist airlines and airports in anticipating disruptions and optimizing scheduling.

**Difficulty Level**: Hard (3/5)

**Objective**: Develop a data-driven classification model that predicts flight delays by analyzing departure and arrival times, airline and airport features, weather conditions, and day-of-week and seasonal trends.

## Problem Statement

Flight delays are a common problem in the aviation industry, affecting millions of passengers annually. Predicting delays can help:
- **Airlines**: Optimize scheduling and resource allocation.
- **Airports**: Improve ground operations and passenger management.
- **Passengers**: Make informed travel decisions.

The challenge involves:
- **Class Imbalance**: Most flights are on-time, making delayed flights a minority class.
- **Multiple Data Sources**: Combining flight schedule data with weather information.
- **Feature Complexity**: Temporal patterns, weather conditions, airline characteristics, and airport-specific factors.
- **Missing Data**: Handling incomplete weather or flight records.

## Dataset

**Source**: [US Airline On-Time Performance Dataset (Kaggle)](https://www.kaggle.com/datasets/giovamata/airlinedelaycauses)

This dataset contains extensive flight and delay information across multiple US airports, including:
- **Flight Information**: Departure/arrival times, airlines, origin/destination airports, flight numbers.
- **Delay Data**: Departure delays, arrival delays, cancellation reasons.
- **Weather Data**: Temperature, precipitation, wind speed, visibility at airports.
- **Temporal Features**: Date, day of week, month, season.
- **Operational Variables**: Aircraft type, distance, scheduled vs. actual times.

## Methodology

### 1. Data Preprocessing

**Merge Flight and Weather Data**:
- Combine flight schedule data with weather information based on airport codes and dates.
- Align timestamps to match departure times with weather conditions at origin airports.

**Handle Missing Values**:
- Identify missing values in flight and weather features.
- Use XGBoost's built-in missing value handling or imputation strategies (mean, median, mode) for numerical and categorical features.
- Drop rows with critical missing information (e.g., departure time, airport codes).

**Data Cleaning**:
- Remove duplicates and outliers.
- Handle inconsistent date formats and time zones.
- Validate airport codes and airline identifiers.

### 2. Feature Engineering

Create features that capture patterns in flight delays:

**Temporal Features**:
- `departure_hour`: Hour of departure (0-23).
- `departure_day_of_week`: Day of week (Monday=0, Sunday=6).
- `departure_month`: Month of year (1-12).
- `is_weekend`: Binary indicator for weekend flights.
- `is_holiday`: Indicator for holiday periods.

**Flight Features**:
- `airline`: Airline carrier code (categorical).
- `origin_airport`: Origin airport code (categorical).
- `destination_airport`: Destination airport code (categorical).
- `flight_distance`: Distance between origin and destination.
- `scheduled_departure_time`: Scheduled departure time.

**Weather Features**:
- `temperature`: Temperature at origin airport.
- `precipitation`: Precipitation amount at origin airport.
- `wind_speed`: Wind speed at origin airport.
- `visibility`: Visibility at origin airport.
- `weather_condition`: Categorical weather condition (clear, rain, snow, etc.).

**Derived Features**:
- `departure_delay_history`: Historical average delay for the same route/airline.
- `airport_delay_rate`: Average delay rate for origin airport.
- `airline_delay_rate`: Average delay rate for airline.

**Target Variable**:
- `is_delayed`: Binary classification target (1 if departure delay > 15 minutes, 0 otherwise).

### 3. Model Training

**XGBoost Classifier Setup**:
- Initialize `XGBClassifier` with parameters optimized for class imbalance:
  - `objective='binary:logistic'`: Binary classification.
  - `scale_pos_weight`: Automatically calculated from class distribution.
  - `max_depth`: Tuned (typically 6-10).
  - `learning_rate`: Tuned (typically 0.01-0.1).
  - `n_estimators`: Tuned with early stopping.
  - `subsample`: 0.8-0.9 for regularization.
  - `colsample_bytree`: 0.8-0.9 for feature sampling.

**Training Process**:
- Split data into training (70%), validation (15%), and test (15%) sets.
- Use validation set for early stopping to prevent overfitting.
- Apply cross-validation for hyperparameter tuning.
- Train model on training set with validation monitoring.

### 4. Evaluation

Due to class imbalance, use comprehensive evaluation metrics:

**Metrics**:
- **Precision**: Proportion of predicted delays that are actually delayed.
- **Recall (Sensitivity)**: Proportion of actual delays that are correctly predicted.
- **F1-Score**: Harmonic mean of precision and recall.
- **ROC-AUC**: Area under the ROC curve, measuring model's ability to distinguish between classes.
- **Confusion Matrix**: Detailed breakdown of true positives, false positives, true negatives, and false negatives.

**Evaluation Strategy**:
- Evaluate on held-out test set.
- Report metrics for both classes (on-time vs. delayed).
- Focus on recall for delayed flights (minimize false negatives).
- Use ROC-AUC as the primary metric for class imbalance.

### 5. Visualization

**Feature Importance**:
- Plot feature importance scores (gain, weight, cover) to identify key factors.
- Compare importance of weather vs. airline vs. airport factors.
- Visualize top 10-20 most important features.

**Delay Analysis**:
- Delay rates by airline, airport, day of week, and hour of day.
- Weather impact on delays (correlation between weather conditions and delay rates).
- Temporal trends (delays by month, season).

**Model Performance**:
- ROC curve visualization.
- Precision-Recall curve.
- Confusion matrix heatmap.

## Architecture

**Project Structure**:
```
AirlineDelay.example.ipynb  → End-to-end delay prediction workflow
AirlineDelay.API.ipynb      → XGBoost API exploration and documentation
AirlineDelay_utils.py       → Helper functions for data preprocessing, model training, and evaluation
AirlineDelay.example.md     → Project documentation (this file)
AirlineDelay.API.md         → XGBoost API documentation
README.md                   → Project overview
```

**Workflow**:
1. **Data Loading**: Load flight and weather datasets from Kaggle.
2. **Data Preprocessing**: Merge datasets, handle missing values, clean data.
3. **Feature Engineering**: Create temporal, flight, weather, and derived features.
4. **Model Training**: Train XGBoost classifier with hyperparameter tuning.
5. **Evaluation**: Compute precision, recall, F1-score, and ROC-AUC.
6. **Visualization**: Plot feature importance and delay analysis.
7. **Prediction**: Make predictions on new flight data.

## Implementation Details

**Key Functions in `AirlineDelay_utils.py`**:

- `load_and_merge_data()`: Load flight and weather data, merge by airport and date.
- `preprocess_data()`: Handle missing values, encode categorical variables, scale features.
- `create_features()`: Generate temporal, flight, weather, and derived features.
- `train_xgboost_model()`: Train XGBoost classifier with automatic class balancing.
- `evaluate_model()`: Compute precision, recall, F1-score, and ROC-AUC.
- `plot_feature_importance()`: Visualize feature importance scores.
- `plot_delay_analysis()`: Create visualizations for delay patterns.

**Technologies Used**:
- **Python**: Programming language.
- **pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computations.
- **XGBoost**: Gradient boosting classifier.
- **scikit-learn**: Model evaluation and preprocessing.
- **matplotlib/seaborn**: Static visualizations.
- **Plotly**: Interactive visualizations (bonus feature).

## Results and Insights

**Expected Outcomes**:
- Model accuracy and performance metrics on test set.
- Identification of key factors influencing flight delays (weather, airline, airport, temporal).
- Insights into delay patterns by airline, airport, and time of day.
- Feature importance rankings to understand delay drivers.

**Key Insights**:
- Weather conditions (precipitation, wind speed, visibility) are likely top predictors.
- Temporal factors (hour of day, day of week, month) show strong patterns.
- Airline and airport-specific factors contribute to delay predictions.
- Historical delay patterns (route, airport, airline) provide predictive power.

## Bonus Features

**Optional Enhancements**:

1. **Model Comparison**:
   - Compare XGBoost with LightGBM and CatBoost.
   - Evaluate performance, training time, and feature importance across models.
   - Select best model based on ROC-AUC and F1-score.

2. **Interactive Dashboard**:
   - Build a delay prediction dashboard with interactive plots.
   - Include filters for airline, airport, date range, and weather conditions.
   - Display predictions, feature importance, and delay trends.
   - Use Plotly or Streamlit for interactivity.

3. **Advanced Feature Engineering**:
   - Include airport congestion metrics.
   - Add historical delay patterns for specific routes.
   - Incorporate external data (holidays, events, airport capacity).

## Conclusion

This project demonstrates the application of XGBoost for predicting flight delays using historical flight and weather data. By combining comprehensive data preprocessing, feature engineering, and model training, we build a robust classification model that can assist airlines and airports in anticipating disruptions. The focus on evaluation metrics (precision, recall, F1-score, ROC-AUC) ensures the model performs well despite class imbalance, while feature importance visualization provides actionable insights into delay factors.

The project showcases best practices in machine learning, including proper train/validation/test splits, hyperparameter tuning, and comprehensive evaluation. The integration of weather data with flight schedules demonstrates the value of combining multiple data sources for improved predictions.
