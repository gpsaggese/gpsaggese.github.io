# Pyjanitor

## Description
- Pyjanitor is a Python library that extends the capabilities of pandas, making
  data cleaning and preparation easier and more intuitive.
- It provides a fluent API that allows users to chain together multiple data
  cleaning operations in a readable and concise manner.
- The library includes a variety of built-in functions for common data cleaning
  tasks, such as removing duplicates, filling missing values, and renaming
  columns.
- Pyjanitor is designed to work seamlessly with pandas DataFrames, enhancing the
  existing functionality without requiring a complete overhaul of the workflow.
- It encourages best practices in data cleaning by promoting a clear and
  methodical approach to transforming datasets.

## Project Objective
The goal of this project is to clean and preprocess a messy dataset, followed by
applying a machine learning model to predict housing prices. The project will
focus on optimizing the accuracy of the prediction model while ensuring the data
is clean and well-structured for analysis.

## Dataset Suggestions
1. **Housing Prices Dataset**
   - **Source**: Kaggle
   - **URL**:
     [Kaggle Housing Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   - **Data Contains**: Various features related to housing characteristics,
     sale prices, and neighborhood information.
   - **Access Requirements**: Free to use with a Kaggle account (no payment
     required).

2. **California Housing Prices**
   - **Source**: California Department of Housing and Community Development
   - **URL**:
     [California Housing Data](https://data.ca.gov/dataset/housing-data)
   - **Data Contains**: Housing prices, demographics, and economic indicators
     across California.
   - **Access Requirements**: Open access; no authentication required.

3. **Ames Housing Dataset**
   - **Source**: Kaggle
   - **URL**:
     [Ames Housing](https://www.kaggle.com/datasets/prestonvong/ames-housing-data)
   - **Data Contains**: Detailed information about properties in Ames, Iowa,
     including features affecting housing prices.
   - **Access Requirements**: Free to use with a Kaggle account (no payment
     required).

## Tasks
- **Data Import and Initial Exploration**: Load the dataset using pandas and
  perform initial exploratory data analysis (EDA) to understand the structure
  and identify issues.
- **Data Cleaning with Pyjanitor**: Utilize Pyjanitor to clean the dataset by
  chaining operations such as removing duplicates, filling missing values, and
  renaming columns for clarity.
- **Feature Engineering**: Create new features that may improve the prediction
  model, such as combining or transforming existing features.
- **Model Training**: Split the dataset into training and testing sets, then
  apply a regression model (e.g., Linear Regression or Random Forest) to predict
  housing prices.
- **Model Evaluation**: Evaluate the model's performance using appropriate
  metrics (e.g., RMSE, R²) and visualize the results for better interpretation.

## Bonus Ideas
- **Hyperparameter Tuning**: Implement techniques such as Grid Search or Random
  Search to optimize model performance further.
- **Feature Importance Analysis**: Investigate which features contribute most to
  the predictions and visualize their importance.
- **Anomaly Detection**: Identify outliers in housing prices and analyze their
  impact on the overall model performance.
- **Deployment**: Explore options for deploying the model using a simple web app
  framework like Flask or Streamlit.

## Useful Resources
- [Pyjanitor Documentation](https://pyjanitor.readthedocs.io/en/latest/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Flask Documentation](https://flask.palletsprojects.com/en/2.0.x/)
