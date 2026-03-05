# Dataprep

## Description
- Dataprep is an open-source library designed to simplify the data preparation
  process for machine learning projects.
- It provides a user-friendly interface to clean, transform, and visualize
  datasets, making it accessible for users with varying levels of expertise.
- Key features include automated data cleaning, visualization tools to explore
  data distributions, and the ability to handle missing values effectively.
- Dataprep supports integration with popular data science libraries like Pandas
  and Scikit-learn, facilitating seamless transitions from data preparation to
  model building.
- The tool is particularly useful for exploratory data analysis (EDA), feature
  engineering, and data validation, ensuring that datasets are ready for machine
  learning tasks.

## Project Objective
The goal of this project is to develop a machine learning model that predicts
housing prices based on various features such as location, size, and amenities.
The project will focus on optimizing the model's accuracy and interpretability
using the Dataprep library for data preparation.

## Dataset Suggestions
1. **Kaggle Housing Prices Dataset**
   - **Source:** Kaggle
   - **URL:**
     [Housing Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
   - **Data Contains:** Features of houses (e.g., size, number of rooms,
     location) and their sale prices.
   - **Access Requirements:** Free account on Kaggle for download.

2. **Zillow Home Value Index**
   - **Source:** Zillow
   - **URL:** [Zillow API](https://www.zillow.com/howto/api/APIOverview.htm)
   - **Data Contains:** Historical home values, property characteristics, and
     market trends.
   - **Access Requirements:** No authentication required for basic access.

3. **OpenStreetMap Data**
   - **Source:** OpenStreetMap
   - **URL:** [Overpass API](http://overpass-api.de/)
   - **Data Contains:** Geospatial data related to housing, amenities, and
     infrastructure in specific areas.
   - **Access Requirements:** Publicly accessible API with no authentication
     needed.

4. **UCI Machine Learning Repository - California Housing**
   - **Source:** UCI
   - **URL:**
     [California Housing Data Set](https://archive.ics.uci.edu/ml/datasets/California+Housing+Prices)
   - **Data Contains:** Housing data from California, including features like
     median income, housing age, and house prices.
   - **Access Requirements:** Direct download without authentication.

## Tasks
- **Data Loading:** Use Dataprep to load the selected dataset and explore its
  structure and features.
- **Data Cleaning:** Apply automated data cleaning techniques to handle missing
  values, outliers, and data type conversions.
- **Feature Engineering:** Utilize Dataprep's visualization tools to identify
  important features and create new features that may improve model performance.
- **Model Training:** Split the dataset into training and testing sets, and
  train a regression model (e.g., Linear Regression) using Scikit-learn.
- **Model Evaluation:** Evaluate the model's performance using metrics such as
  Mean Absolute Error (MAE) and R-squared, and visualize the results.
- **Reporting:** Create a comprehensive report summarizing the data preparation
  steps, model performance, and insights gained from the analysis.

## Bonus Ideas
- Experiment with different regression algorithms (e.g., Decision Trees, Random
  Forests) and compare their performances.
- Implement hyperparameter tuning to optimize model parameters for better
  accuracy.
- Create interactive visualizations using libraries like Plotly or Streamlit to
  present findings and model predictions.
- Explore the impact of additional features from the OpenStreetMap dataset on
  housing prices.

## Useful Resources
- [Dataprep Documentation](https://dataprep.readthedocs.io/en/latest/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Kaggle API Documentation](https://www.kaggle.com/docs/api)
- [Zillow API Overview](https://www.zillow.com/howto/api/APIOverview.htm)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
