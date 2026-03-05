# Papermill

## Description
- Papermill is an open-source tool designed for parameterizing and executing
  Jupyter Notebooks.
- It allows users to run notebooks with different input parameters, making it
  ideal for reproducible research and experimentation.
- Users can create templates with placeholders that can be filled with different
  values at runtime, enabling batch processing of data.
- Papermill supports logging and output management, allowing for easy tracking
  of notebook executions and results.
- It integrates seamlessly with Jupyter, making it easy to incorporate into
  existing data science workflows without needing extensive setup.

## Project Objective
The goal of this project is to build a reproducible data analysis pipeline using
Papermill, where students will parameterize a Jupyter Notebook to analyze a
dataset on housing prices. The project will focus on predicting housing prices
based on various features using a regression model.

## Dataset Suggestions
1. **Kaggle Housing Prices Dataset**
   - **Source**: Kaggle
   - **URL**:
     [Housing Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   - **Data Contains**: Various features of houses (e.g., size, number of rooms,
     location) and their sale prices.
   - **Access Requirements**: Requires a free Kaggle account to download the
     dataset.

2. **Open Government Data - NYC Housing Data**
   - **Source**: NYC Open Data
   - **URL**:
     [NYC Housing Data](https://data.cityofnewyork.us/Housing-Development/NYC-Housing-Development-Data-2019-2020/8z8g-nv5u)
   - **Data Contains**: Information on various housing developments in New York
     City, including prices, locations, and types of housing.
   - **Access Requirements**: No authentication required; data is freely
     accessible.

3. **California Housing Prices**
   - **Source**: California Housing Prices Dataset from Scikit-Learn
   - **URL**:
     [California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
   - **Data Contains**: Features describing California housing prices, including
     median income, housing age, and geographical data.
   - **Access Requirements**: Available directly through Scikit-Learn library.

## Tasks
- **Setup and Environment Configuration**: Install Papermill and set up the
  Jupyter Notebook environment for the project.
- **Data Exploration**: Create an initial notebook to explore the chosen housing
  dataset, summarizing key statistics and visualizations.
- **Parameterization**: Modify the notebook to include parameters for different
  regression model configurations (e.g., feature selection, model type).
- **Model Training**: Implement a regression model (e.g., Linear Regression or
  Random Forest) within the notebook, using the parameterized inputs to train on
  the dataset.
- **Execution with Papermill**: Use Papermill to execute the parameterized
  notebook multiple times with different configurations and collect the results.
- **Results Analysis**: Analyze the outputs from the different runs to determine
  which model configuration yields the best performance based on evaluation
  metrics (e.g., RMSE).

## Bonus Ideas
- **Hyperparameter Tuning**: Extend the project by implementing hyperparameter
  tuning using grid search or random search within the parameterized notebook.
- **Model Comparison**: Compare multiple regression models (e.g., Ridge, Lasso,
  and Decision Trees) and visualize their performance metrics.
- **Deployment**: Explore how to deploy the final model using a web app
  framework (like Flask) that allows users to input parameters and receive
  predictions.

## Useful Resources
- [Papermill GitHub Repository](https://github.com/nteract/papermill)
- [Papermill Documentation](https://papermill.readthedocs.io/en/latest/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [NYC Open Data Portal](https://opendata.cityofnewyork.us/)
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/documentation.html)
