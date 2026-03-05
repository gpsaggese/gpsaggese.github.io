# FireDucks

## Description
- FireDucks is an open-source data science tool designed to simplify the process
  of building and deploying machine learning models.
- It features an intuitive interface for data preprocessing, model training, and
  evaluation, making it accessible for users with varying levels of expertise.
- The tool supports a wide range of machine learning algorithms, including
  classification, regression, and clustering, allowing for flexibility in
  project design.
- FireDucks integrates seamlessly with popular libraries such as Pandas,
  Scikit-learn, and Matplotlib, facilitating data manipulation and
  visualization.
- It includes built-in functionalities for hyperparameter tuning and model
  evaluation metrics, enabling users to optimize their models effectively.

## Project Objective
The goal of the project is to predict housing prices based on various features
such as location, size, and amenities. Students will optimize a regression model
to achieve the lowest mean absolute error (MAE) on a test dataset.

## Dataset Suggestions
1. **Kaggle Housing Prices Dataset**
   - **Source Name**: Kaggle
   - **URL**:
     [Kaggle Housing Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   - **Data Contains**: Features related to residential homes in Ames, Iowa,
     including sale prices, square footage, number of bedrooms, and more.
   - **Access Requirements**: Free account on Kaggle to download the dataset.

2. **California Housing Prices Dataset**
   - **Source Name**: California Department of Housing and Community Development
   - **URL**:
     [California Housing Prices](https://www.dhcs.ca.gov/services/Pages/Housing-Data.aspx)
   - **Data Contains**: Housing prices and characteristics across various
     counties in California.
   - **Access Requirements**: Publicly available data; no registration required.

3. **Zillow Home Value Index (ZHVI)**
   - **Source Name**: Zillow
   - **URL**: [Zillow API](https://www.zillow.com/howto/api/APIOverview.htm)
   - **Data Contains**: Historical home values and rental prices for various
     regions across the U.S.
   - **Access Requirements**: Free access to the API with limited requests per
     day.

4. **OpenStreetMap (OSM) Data**
   - **Source Name**: OpenStreetMap
   - **URL**: [OSM API](https://wiki.openstreetmap.org/wiki/API)
   - **Data Contains**: Geospatial data that can be used to enrich housing
     datasets with location features.
   - **Access Requirements**: Free to use; no authentication needed for basic
     queries.

## Tasks
- **Data Collection**: Use FireDucks to download and load datasets from the
  suggested sources into a suitable format for analysis.
- **Data Preprocessing**: Clean and preprocess the data using FireDucks
  functionalities, including handling missing values and feature scaling.
- **Exploratory Data Analysis (EDA)**: Utilize FireDucks to visualize data
  distributions and relationships between features and the target variable
  (housing prices).
- **Model Selection**: Choose appropriate regression models (e.g., Linear
  Regression, Random Forest) available in FireDucks and train them on the
  dataset.
- **Model Evaluation**: Evaluate model performance using metrics such as MAE and
  R², and compare results across different models.
- **Hyperparameter Tuning**: Implement hyperparameter optimization techniques
  available in FireDucks to improve model accuracy.

## Bonus Ideas
- **Feature Engineering**: Create new features based on existing data (e.g.,
  interaction terms, categorical encoding) to enhance model performance.
- **Model Comparison**: Compare the performance of traditional regression models
  with more complex models (e.g., Gradient Boosting) to analyze trade-offs.
- **Deployment**: Explore options for deploying the final model using FireDucks
  capabilities to create a simple web interface for predictions.
- **Geospatial Analysis**: Integrate geospatial features from OpenStreetMap to
  see if location impacts housing prices significantly.

## Useful Resources
- [FireDucks GitHub Repository](https://github.com/fireducks/fireducks)
- [Kaggle Documentation](https://www.kaggle.com/docs)
- [California Housing Data](https://www.dhcs.ca.gov/services/Pages/Housing-Data.aspx)
- [Zillow API Documentation](https://www.zillow.com/howto/api/APIOverview.htm)
- [OpenStreetMap API Documentation](https://wiki.openstreetmap.org/wiki/API)
