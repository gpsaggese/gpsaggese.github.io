# Dagster

## Description
- Dagster is an open-source data orchestrator designed for building and managing
  data pipelines.
- It allows users to define, schedule, and monitor workflows, making it easier
  to manage complex data processing tasks.
- With its strong focus on data quality and observability, Dagster provides
  tools for testing and validating data at various stages of the pipeline.
- The tool supports a wide range of integrations with popular data tools,
  databases, and cloud services, enabling seamless data flow across different
  environments.
- Dagster's user interface offers a visual representation of data pipelines,
  making it easier to understand dependencies and execution flow.

## Project Objective
The goal of this project is to build a data pipeline that ingests, processes,
and analyzes a dataset to predict housing prices in a specific region. Students
will optimize the pipeline for efficiency and data quality, ultimately
delivering a predictive model for housing prices based on various features.

## Dataset Suggestions
1. **Kaggle Housing Prices Dataset**
   - **Source**: Kaggle
   - **URL**:
     [Housing Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   - **Data Contains**: Various features of houses (e.g., size, location, number
     of rooms) and their sale prices.
   - **Access Requirements**: Free account on Kaggle to download the dataset.

2. **Open Data Portal - San Francisco Housing Data**
   - **Source**: City of San Francisco
   - **URL**:
     [SF Housing Data](https://data.sfgov.org/Housing-and-Buildings/Housing-Inventory-2018-2021-Data-Update/2w7d-9g8y)
   - **Data Contains**: Housing inventory data including types, sizes, and
     prices of residential units.
   - **Access Requirements**: No authentication required; direct access to the
     dataset.

3. **UCI Machine Learning Repository - Boston Housing Dataset**
   - **Source**: UCI Machine Learning Repository
   - **URL**: [Boston Housing](https://archive.ics.uci.edu/ml/datasets/Housing)
   - **Data Contains**: Features of houses in Boston and their median values.
   - **Access Requirements**: Direct download available without authentication.

4. **Kaggle - California Housing Prices**
   - **Source**: Kaggle
   - **URL**:
     [California Housing](https://www.kaggle.com/c/california-housing-prices)
   - **Data Contains**: Features of homes in California and their prices.
   - **Access Requirements**: Free account on Kaggle to download the dataset.

## Tasks
- **Pipeline Design**: Create a data pipeline in Dagster that ingests the
  selected housing dataset and defines the necessary transformations.
- **Data Quality Checks**: Implement data validation checks to ensure the
  integrity and quality of the data being processed.
- **Feature Engineering**: Perform feature selection and transformation to
  optimize the dataset for the predictive model.
- **Model Training**: Use a regression model (e.g., Linear Regression or Random
  Forest) to predict housing prices based on the processed features.
- **Monitoring and Logging**: Set up monitoring and logging within Dagster to
  track the performance and execution of the data pipeline.
- **Evaluation and Reporting**: Evaluate the model's performance using metrics
  like RMSE or R² and generate a report summarizing the findings.

## Bonus Ideas
- **Hyperparameter Tuning**: Experiment with different regression models and
  tune hyperparameters to improve model accuracy.
- **Compare Pipelines**: Create alternative pipelines using different data
  processing techniques or models and compare their performance.
- **Deploy the Model**: Explore options for deploying the trained model as a web
  service using Flask or FastAPI for real-time predictions.
- **Data Visualization**: Create visualizations to represent the distribution of
  housing prices and the impact of various features on price predictions.

## Useful Resources
- [Dagster Documentation](https://docs.dagster.io/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Data.gov Open Data](https://www.data.gov/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
