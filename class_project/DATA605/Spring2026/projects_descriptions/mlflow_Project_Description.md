# Mlflow

## Description
- MLflow is an open-source platform designed to manage the machine learning
  lifecycle, including experimentation, reproducibility, and deployment.
- It provides four main components: Tracking, Projects, Models, and Registry,
  allowing users to log experiments, package code into reproducible runs, manage
  and deploy models, and maintain a centralized model registry.
- The Tracking component enables users to log parameters, metrics, and
  artifacts, facilitating easy comparison of different machine learning
  experiments.
- MLflow Projects allows for packaging ML code in a reusable way, making it
  easier to share and reproduce experiments across different environments.
- The Models component provides a standard format for packaging models and
  supports deployment to various platforms, including cloud services and local
  environments.
- With the Model Registry, users can manage the lifecycle of their models,
  including versioning, staging, and transitioning models between production and
  development.

## Project Objective
The goal of this project is to build a machine learning model that predicts
housing prices based on various features (e.g., location, size, number of
bedrooms, etc.) using regression techniques. Students will optimize the model's
performance by tuning hyperparameters and comparing different algorithms.

## Dataset Suggestions
1. **Kaggle - House Prices: Advanced Regression Techniques**
   - **URL**:
     [Kaggle House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   - **Data Contains**: Features related to house characteristics and sale
     prices.
   - **Access Requirements**: Free account on Kaggle to download the dataset.

2. **UCI Machine Learning Repository - Boston Housing Dataset**
   - **URL**:
     [Boston Housing Dataset](https://archive.ics.uci.edu/ml/datasets/Housing)
   - **Data Contains**: Information about housing in Boston, including features
     like crime rate, number of rooms, and distance to employment centers.
   - **Access Requirements**: No authentication required; data is freely
     available for download.

3. **Kaggle - California Housing Prices**
   - **URL**:
     [Kaggle California Housing Dataset](https://www.kaggle.com/c/california-housing-prices/data)
   - **Data Contains**: Features related to California housing prices, including
     geographical and demographic information.
   - **Access Requirements**: Free account on Kaggle to download the dataset.

4. **Open Government Data - NYC Housing Data**
   - **URL**:
     [NYC Housing Data](https://data.cityofnewyork.us/Housing-Development/Housing-Data/3q8c-8y5c)
   - **Data Contains**: Information on housing developments in New York City,
     including rent prices and unit counts.
   - **Access Requirements**: No authentication required; data is freely
     available for download.

## Tasks
- **Environment Setup**: Install MLflow and set up the project environment,
  ensuring all necessary libraries are available.
- **Data Exploration**: Load and explore the dataset to understand its
  structure, features, and any missing values.
- **Model Training**: Implement a regression model (e.g., Linear Regression,
  Random Forest) to predict housing prices and log parameters/metrics using
  MLflow Tracking.
- **Hyperparameter Tuning**: Use MLflow to track different hyperparameter
  configurations and evaluate their impact on model performance.
- **Model Evaluation**: Evaluate the model's performance using appropriate
  metrics (e.g., RMSE, R²) and log results in MLflow.
- **Model Deployment**: Package the final model using MLflow Models and deploy
  it locally or to a cloud service for inference.

## Bonus Ideas
- **Feature Engineering**: Experiment with creating new features from existing
  data to improve model accuracy and log the impact on performance.
- **Model Comparison**: Compare multiple regression algorithms (e.g., Gradient
  Boosting, Support Vector Regression) and analyze their performance using
  MLflow.
- **Experimentation with Pre-trained Models**: Investigate the use of
  pre-trained models or transfer learning techniques to enhance predictive
  capabilities.
- **Visualization of Results**: Create visualizations of model performance and
  feature importance, integrating them into the MLflow UI.

## Useful Resources
- [MLflow Official Documentation](https://www.mlflow.org/docs/latest/index.html)
- [MLflow GitHub Repository](https://github.com/mlflow/mlflow)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Open Government Data Portal](https://www.data.gov/)
