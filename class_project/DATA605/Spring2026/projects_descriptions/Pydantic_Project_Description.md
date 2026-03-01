# Pydantic

## Description
- Pydantic is a data validation and settings management library for Python,
  primarily used for parsing and validating data structures.
- It allows developers to define data models using Python type annotations,
  ensuring that the data adheres to specified types and formats.
- Pydantic can automatically generate JSON schemas from data models, making it
  easier to document APIs and data structures.
- It provides built-in error handling and validation, allowing for clear and
  informative error messages when data does not conform to expected types.
- Pydantic is particularly useful in data science projects for creating robust
  data pipelines and ensuring data integrity before processing.

## Project Objective
The goal of this project is to build a data validation pipeline that ingests,
validates, and processes a dataset related to housing prices. Students will
optimize a machine learning model to predict housing prices based on various
features while ensuring that the incoming data meets specified validation
criteria.

## Dataset Suggestions
1. **California Housing Prices**
   - **Source**: Kaggle
   - **URL**:
     [California Housing Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   - **Data Contains**: Features related to housing such as location, size,
     number of bedrooms, and target prices.
   - **Access Requirements**: Free account on Kaggle required to download the
     dataset.

2. **Boston Housing Dataset**
   - **Source**: UCI Machine Learning Repository
   - **URL**:
     [Boston Housing Dataset](https://archive.ics.uci.edu/ml/datasets/Housing)
   - **Data Contains**: Attributes of homes in Boston, including crime rates,
     number of rooms, and median value of owner-occupied homes.
   - **Access Requirements**: Publicly available, no authentication required.

3. **Real Estate Valuation Data Set**
   - **Source**: UCI Machine Learning Repository
   - **URL**:
     [Real Estate Valuation Data Set](https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set)
   - **Data Contains**: Features including house age, distance to the nearest
     MRT station, and number of convenience stores.
   - **Access Requirements**: Publicly available, no authentication required.

## Tasks
- **Define Data Models**: Use Pydantic to create data models for the housing
  dataset, specifying types and validation rules for each feature.
- **Data Ingestion**: Implement a data ingestion pipeline that reads the dataset
  and validates incoming data against the defined Pydantic models.
- **Feature Engineering**: Perform feature engineering to create new variables
  that may improve model performance, while ensuring they conform to the data
  models.
- **Model Training**: Train a regression model (e.g., Linear Regression or
  Random Forest) to predict housing prices using validated and processed data.
- **Model Evaluation**: Evaluate the model's performance using appropriate
  metrics (e.g., RMSE, R²) and validate the results against a test dataset.

## Bonus Ideas
- Extend the project by implementing additional validation rules, such as
  checking for outliers or ensuring certain features fall within expected
  ranges.
- Compare the performance of different regression models and select the best one
  based on evaluation metrics.
- Create a simple web interface using FastAPI to allow users to input housing
  data, which is then validated using Pydantic before making predictions.

## Useful Resources
- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
