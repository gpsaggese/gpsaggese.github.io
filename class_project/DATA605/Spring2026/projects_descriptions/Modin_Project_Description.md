# Modin

## Description
- Modin is an open-source library designed to accelerate the performance of
  pandas by enabling parallel and distributed computing.
- It allows users to seamlessly scale their data analysis workflows from a
  single machine to a cluster without changing the existing pandas code.
- Modin supports a wide range of data formats, including CSV, Parquet, and SQL
  databases, making it versatile for various data sources.
- The library is designed to efficiently utilize available hardware resources,
  providing significant speed improvements for large datasets.
- Modin integrates well with Dask and Ray, allowing for easy deployment in cloud
  environments and on multi-core systems.
- It is particularly useful for data preprocessing and exploratory data analysis
  (EDA) in machine learning projects.

## Project Objective
The goal of this project is to build a predictive model that forecasts housing
prices based on various features such as location, size, and amenities. Students
will optimize the model to achieve the highest possible accuracy in predicting
home prices.

## Dataset Suggestions
1. **Kaggle Housing Prices Dataset**
   - **Source Name**: Kaggle
   - **URL**:
     [Housing Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   - **Data Contains**: Features of houses (e.g., number of bedrooms, square
     footage, location) and their sale prices.
   - **Access Requirements**: Free account on Kaggle to download the dataset.

2. **Zillow Home Value Index (ZHVI)**
   - **Source Name**: Zillow
   - **URL**:
     [Zillow Data](https://www.zillow.com/howto/api/Zillow-Data-API.htm)
   - **Data Contains**: Historical home values, property characteristics, and
     market trends.
   - **Access Requirements**: Free access with an API key (no payment plans
     required).

3. **OpenStreetMap (OSM) Data**
   - **Source Name**: OpenStreetMap
   - **URL**: [OSM Data](https://download.geofabrik.de/)
   - **Data Contains**: Geographic data that can be used to analyze housing
     density and proximity to amenities.
   - **Access Requirements**: Publicly available data, no authentication needed.

4. **UCI Machine Learning Repository: Boston Housing Dataset**
   - **Source Name**: UCI Machine Learning Repository
   - **URL**: [Boston Housing](https://archive.ics.uci.edu/ml/datasets/Housing)
   - **Data Contains**: Features related to housing in Boston, including crime
     rates, number of rooms, and property tax rates.
   - **Access Requirements**: Publicly accessible without any authentication.

## Tasks
- **Data Loading and Preprocessing**: Use Modin to load the datasets and perform
  necessary preprocessing steps such as handling missing values and encoding
  categorical variables.
- **Exploratory Data Analysis (EDA)**: Utilize Modin to explore the dataset,
  identify trends, and visualize relationships between features and housing
  prices.
- **Feature Engineering**: Create new features based on existing data that may
  improve model performance, such as interaction terms or aggregating features.
- **Model Selection and Training**: Implement various regression models (e.g.,
  Linear Regression, Decision Trees) to predict housing prices, using Modin for
  data handling.
- **Model Evaluation**: Evaluate model performance using metrics like RMSE and
  R², and compare the results of different models to identify the
  best-performing one.
- **Final Reporting**: Compile a report summarizing the findings, model
  performance, and insights gained from the analysis.

## Bonus Ideas
- **Hyperparameter Tuning**: Implement grid search or randomized search for
  hyperparameter tuning to optimize model performance further.
- **Comparison with Traditional Pandas**: Compare the performance and execution
  time of Modin versus traditional pandas for the same tasks on large datasets.
- **Deployment**: Create a simple web app using Flask or Streamlit to showcase
  the model's predictions for new housing data.
- **Incorporate External Datasets**: Enhance the project by integrating external
  datasets, such as economic indicators or neighborhood statistics, to improve
  prediction accuracy.

## Useful Resources
- [Modin Documentation](https://modin.readthedocs.io/en/latest/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Zillow API Documentation](https://www.zillow.com/howto/api/APIOverview.htm)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [OpenStreetMap API Documentation](https://wiki.openstreetmap.org/wiki/OpenStreetMap_API)
