# MLflow

## Description
- MLflow is an open-source platform designed for managing the machine learning
  lifecycle, including experimentation, reproducibility, and deployment.
- Key features include tracking experiments, packaging code into reproducible
  runs, sharing and deploying models, and managing the ML lifecycle with a
  user-friendly interface.
- It supports multiple ML libraries and frameworks, making it versatile for
  various data science projects.
- MLflow provides a centralized repository for storing and organizing models,
  allowing teams to collaborate efficiently.
- The tool includes built-in functionalities for logging metrics, parameters,
  and artifacts, facilitating a streamlined workflow for data scientists.

## Project Objective
The goal of this project is to build and evaluate a machine learning model that
predicts house prices based on various features such as location, size, and
amenities. Students will optimize for the lowest possible prediction error,
using metrics such as Mean Absolute Error (MAE) or Root Mean Squared Error
(RMSE).

## Dataset Suggestions
1. **Kaggle House Prices: Advanced Regression Techniques**
   - **Source**: Kaggle
   - **URL**:
     [House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   - **Data Contains**: Features of houses (e.g., size, location, year built)
     and their sale prices.
   - **Access Requirements**: Free registration on Kaggle.

2. **Ames Housing Dataset**
   - **Source**: Kaggle
   - **URL**:
     [Ames Housing Dataset](https://www.kaggle.com/datasets/prestonv1/ames-housing-data)
   - **Data Contains**: Comprehensive features of houses in Ames, Iowa,
     including prices and various characteristics.
   - **Access Requirements**: Free registration on Kaggle.

3. **Real Estate Valuation Data Set**
   - **Source**: UCI Machine Learning Repository
   - **URL**:
     [Real Estate Valuation Data](https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set)
   - **Data Contains**: Features related to real estate properties and their
     valuations.
   - **Access Requirements**: Publicly available without registration.

## Tasks
- **Data Exploration**: Load the dataset using Pandas and perform exploratory
  data analysis (EDA) to understand the features and their distributions.
- **Data Preprocessing**: Clean the data by handling missing values, encoding
  categorical variables, and scaling numerical features as necessary.
- **Model Training**: Use MLflow to track different regression models (e.g.,
  Linear Regression, Random Forest) and log their performance metrics.
- **Model Evaluation**: Evaluate the models on a validation set using MAE and
  RMSE, and compare the results using MLflow's experiment tracking features.
- **Model Deployment**: Package the best-performing model using MLflow's model
  management capabilities and create a simple API for predictions.

## Bonus Ideas
- Implement hyperparameter tuning using MLflow's integration with libraries like
  Optuna or Hyperopt.
- Compare the performance of traditional regression models with advanced models
  like XGBoost or LightGBM.
- Extend the project by creating a web dashboard using Streamlit or Flask to
  visualize predictions and model metrics.
- Explore feature importance and perform feature selection techniques to enhance
  model performance.

## Useful Resources
- [MLflow Official Documentation](https://www.mlflow.org/docs/latest/index.html)
- [MLflow GitHub Repository](https://github.com/mlflow/mlflow)
- [Kaggle Learn - Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)
- [Ames Housing Dataset Documentation](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
