# Autofeat

## Description
- Autofeat is an automated feature engineering tool designed to enhance machine
  learning models by generating new features from existing data.
- It utilizes various mathematical transformations and combinations to create
  features that may improve model performance.
- The tool is particularly useful for practitioners who want to optimize their
  feature sets without extensive manual intervention.
- Autofeat is compatible with popular machine learning libraries like
  scikit-learn, making it easy to integrate into existing workflows.
- It provides a straightforward interface for feature generation, allowing users
  to focus on model training and evaluation.
- The tool can handle both numerical and categorical data, making it versatile
  for different types of datasets.

## Project Objective
The goal of this project is to build a predictive model that estimates house
prices based on various features of the properties. The project will focus on
optimizing the feature set using Autofeat to improve prediction accuracy.

## Dataset Suggestions
1. **Kaggle - House Prices: Advanced Regression Techniques**
   - URL:
     [House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   - Contains: Features of houses (size, location, number of rooms, etc.) and
     their corresponding sale prices.
   - Access Requirements: Free account on Kaggle to download the dataset.

2. **Kaggle - California Housing Prices**
   - URL:
     [California Housing Dataset](https://www.kaggle.com/c/california-housing-prices)
   - Contains: Features related to housing prices in California, including
     median income, housing age, and population.
   - Access Requirements: Free account on Kaggle to download the dataset.

3. **UCI Machine Learning Repository - Boston Housing Dataset**
   - URL:
     [Boston Housing Dataset](https://archive.ics.uci.edu/ml/datasets/Housing)
   - Contains: Information about housing in Boston, including crime rates,
     number of rooms, and accessibility to highways.
   - Access Requirements: Publicly available without registration.

4. **Open Government Data - City of Chicago: Housing Data**
   - URL:
     [Chicago Housing Data](https://data.cityofchicago.org/Housing-Development/Chicago-Housing-Data/9j8f-6y6t)
   - Contains: Data on housing developments, including location, type, and
     price.
   - Access Requirements: Publicly available without registration.

## Tasks
- **Data Loading**: Import the selected dataset into your Python environment and
  perform initial data exploration.
- **Data Preprocessing**: Clean the dataset by handling missing values and
  encoding categorical variables.
- **Feature Engineering with Autofeat**: Use Autofeat to generate new features
  from the existing dataset and evaluate their impact on model performance.
- **Model Selection**: Choose a regression model (e.g., Linear Regression,
  Random Forest) and train it using the original and enhanced feature sets.
- **Model Evaluation**: Assess the performance of the models using metrics such
  as RMSE and R², and compare the results of the original and Autofeat-enhanced
  models.
- **Report Findings**: Document the process, findings, and insights gained from
  using Autofeat for feature engineering.

## Bonus Ideas
- **Hyperparameter Tuning**: Implement hyperparameter optimization techniques
  (e.g., Grid Search, Random Search) to further improve model performance.
- **Feature Importance Analysis**: Analyze and visualize the importance of the
  generated features to understand their contribution to the model.
- **Model Comparison**: Compare the results of different regression algorithms
  to find the best-performing model for house price prediction.
- **Deployment**: Create a simple web application to demonstrate the model's
  predictions using Flask or Streamlit.

## Useful Resources
- [Autofeat Documentation](https://github.com/AutoFeat/AutoFeat)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Open Government Data](https://www.data.gov/)
