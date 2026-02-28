# AutoGluon

## Description
- AutoGluon is an open-source AutoML toolkit designed to simplify the process of
  building machine learning models, making it accessible for users with varying
  levels of expertise.
- It provides a user-friendly interface for training, tuning, and deploying
  models across various machine learning tasks, including classification,
  regression, and object detection.
- AutoGluon automatically handles data preprocessing, feature engineering, model
  selection, and hyperparameter tuning, allowing users to focus on
  problem-solving rather than technical complexities.
- The toolkit supports a wide range of data types, including tabular data,
  images, and text, enabling flexibility for different project domains.
- It includes built-in functionality for model ensembling, which can
  significantly improve prediction accuracy by combining multiple models.

## Project Objective
The goal of this project is to develop a predictive model that forecasts housing
prices based on various features of the properties and their locations. Students
will optimize for accuracy in their predictions, using regression techniques to
estimate the sale price of houses.

## Dataset Suggestions
1. **Kaggle Housing Prices Dataset**
   - **Source**: Kaggle
   - **URL**:
     [Housing Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   - **Data Contains**: Information on house features (e.g., square footage,
     number of bedrooms, location) and sale prices.
   - **Access Requirements**: Free account on Kaggle to download the dataset.

2. **Open Government Data: City of Chicago**
   - **Source**: City of Chicago Data Portal
   - **URL**:
     [Chicago Housing Data](https://data.cityofchicago.org/Community-Health/2020-Chicago-Home-Prices/2t5j-6j2c)
   - **Data Contains**: Historical housing prices, property characteristics, and
     neighborhood information.
   - **Access Requirements**: No authentication required, publicly available
     dataset.

3. **Zillow Home Value Index (ZHVI)**
   - **Source**: Zillow
   - **URL**: [Zillow API](https://www.zillow.com/howto/api/ZillowAPI.htm)
   - **Data Contains**: Monthly median home values for specific regions, along
     with property characteristics.
   - **Access Requirements**: Requires a free API key (no payment needed).

## Tasks
- **Data Exploration**: Use AutoGluon to perform exploratory data analysis (EDA)
  on the chosen dataset, identifying key features and relationships.
- **Data Preprocessing**: Automatically preprocess the data using AutoGluon's
  built-in functions to handle missing values, categorical variables, and
  normalization.
- **Model Training**: Train multiple regression models using AutoGluon, allowing
  it to select the best-performing algorithms based on the dataset.
- **Hyperparameter Tuning**: Utilize AutoGluon to automatically tune
  hyperparameters for the selected models to improve prediction accuracy.
- **Model Evaluation**: Evaluate model performance using metrics such as RMSE
  (Root Mean Square Error) and visualize results to compare different models.
- **Final Predictions**: Generate predictions on a test dataset and prepare a
  report summarizing the findings, model performance, and insights.

## Bonus Ideas
- **Feature Importance Analysis**: Explore which features have the most
  significant impact on housing prices and visualize this information.
- **Model Comparison**: Compare the performance of AutoGluon models against
  traditional machine learning models (e.g., linear regression, decision trees)
  to understand the advantages of AutoML.
- **Deployment**: Implement a simple web application to deploy the model using
  Flask or Streamlit, allowing users to input property features and receive
  price predictions.

## Useful Resources
- [AutoGluon Documentation](https://auto.gluon.ai/stable/index.html)
- [Kaggle Housing Prices Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- [Chicago Data Portal](https://data.cityofchicago.org/)
- [Zillow API Documentation](https://www.zillow.com/howto/api/APIOverview.htm)
- [GitHub Repository for AutoGluon](https://github.com/awslabs/autogluon)
