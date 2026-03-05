# Pandera

## Description
- Pandera is a Python library that provides a framework for validating and
  testing data structures, particularly pandas DataFrames.
- It allows users to define schemas for DataFrames, ensuring that the data
  adheres to specified types, shapes, and constraints.
- The library integrates seamlessly with pandas, making it easy to incorporate
  data validation into existing data processing pipelines.
- Pandera supports custom validation functions, enabling users to implement
  complex validation logic as needed.
- It is particularly useful for data scientists and engineers who need to ensure
  data integrity before performing analysis or machine learning tasks.

## Project Objective
The goal of the project is to create a robust data validation pipeline using
Pandera for a dataset related to housing prices. The project will focus on
validating the dataset's structure and contents to ensure high-quality data for
predicting housing prices using a regression model.

## Dataset Suggestions
1. **Kaggle Housing Prices Dataset**
   - **Source**: Kaggle
   - **URL**:
     [Housing Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   - **Data Contains**: Features related to house attributes (e.g., area, number
     of rooms, location) and the target variable (sale price).
   - **Access Requirements**: Free account on Kaggle required to download the
     dataset.

2. **California Housing Prices Dataset**
   - **Source**: Scikit-learn
   - **URL**:
     [California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
   - **Data Contains**: Features such as median income, housing age, and
     population, along with the target variable (median house value).
   - **Access Requirements**: No special access required; can be fetched
     directly using Scikit-learn.

3. **Ames Housing Dataset**
   - **Source**: Kaggle
   - **URL**:
     [Ames Housing Dataset](https://www.kaggle.com/datasets/prestonvong/AmesHousing)
   - **Data Contains**: A comprehensive set of features related to home sales in
     Ames, Iowa, including various attributes and sale prices.
   - **Access Requirements**: Free account on Kaggle required to download the
     dataset.

## Tasks
- **Data Loading**: Load the selected dataset into a pandas DataFrame, ensuring
  it is ready for validation.
- **Schema Definition**: Define a Pandera schema that specifies the expected
  structure, types, and constraints of the DataFrame.
- **Data Validation**: Implement data validation checks using Pandera to ensure
  the DataFrame adheres to the defined schema.
- **Model Training**: Split the validated dataset into training and testing
  sets, then train a regression model (e.g., Linear Regression) to predict
  housing prices.
- **Model Evaluation**: Evaluate the model's performance using appropriate
  metrics (e.g., RMSE, R²) and analyze the validation results.

## Bonus Ideas
- Implement additional custom validation functions in Pandera to check for
  domain-specific data quality issues (e.g., outlier detection).
- Compare the performance of different regression models (e.g., Decision Tree,
  Random Forest) on the validated dataset.
- Extend the project to include feature engineering and selection based on the
  validation results to improve model performance.

## Useful Resources
- [Pandera Documentation](https://pandera.readthedocs.io/en/stable/)
- [Kaggle Housing Prices Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- [Scikit-learn California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- [Ames Housing Dataset on Kaggle](https://www.kaggle.com/datasets/prestonvong/AmesHousing)
- [GitHub Repository for Pandera](https://github.com/pandera-dev/pandera)
