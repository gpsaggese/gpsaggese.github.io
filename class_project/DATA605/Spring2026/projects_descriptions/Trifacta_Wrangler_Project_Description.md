# Trifacta Wrangler

## Description
- Trifacta Wrangler is a powerful data preparation tool designed to help users
  clean, transform, and enrich their data for analysis and machine learning.
- It provides an intuitive interface that allows users to visually explore their
  data and apply transformations through a series of guided steps.
- The tool supports a wide range of data formats, including CSV, JSON, and
  Excel, making it versatile for various data sources.
- Trifacta Wrangler employs machine learning algorithms to suggest
  transformations and detect anomalies, streamlining the data wrangling process.
- Users can easily export their cleaned datasets to various formats or directly
  to popular analytics platforms for further analysis.

## Project Objective
The goal of the project is to prepare and analyze a dataset containing
information about global air quality. Students will apply data wrangling
techniques to clean the data, then use machine learning to predict air quality
index (AQI) levels based on various environmental factors.

## Dataset Suggestions
1. **World Air Quality Index (WAQI)**
   - **Source**: World Air Quality Index Project
   - **URL**: [WAQI API](https://aqicn.org/json-api/doc/)
   - **Data**: Real-time air quality data including AQI values, pollutants, and
     meteorological data from various locations worldwide.
   - **Access Requirements**: Free to use, no authentication required.

2. **OpenAQ**
   - **Source**: OpenAQ
   - **URL**: [OpenAQ API](https://docs.openaq.org/)
   - **Data**: Historical and real-time air quality measurements from multiple
     sources, including PM2.5, PM10, NO2, and O3 levels.
   - **Access Requirements**: Free to use, no authentication required.

3. **Kaggle Air Quality Data**
   - **Source**: Kaggle
   - **URL**:
     [Air Quality Data](https://www.kaggle.com/datasets/sogunro/air-quality-data)
   - **Data**: Historical air quality data recorded from various sensors,
     including temperature, humidity, and different pollutant levels.
   - **Access Requirements**: Free to use with a Kaggle account.

## Tasks
- **Data Acquisition**: Use Trifacta Wrangler to import datasets from the
  selected sources and explore their structure and contents.
- **Data Cleaning**: Identify and handle missing values, duplicates, and
  inconsistencies in the datasets using Trifacta's guided data preparation
  features.
- **Feature Engineering**: Create new features that may improve model
  performance, such as aggregating pollutant levels or creating time-based
  features.
- **Data Transformation**: Apply necessary transformations to the data,
  including normalization and encoding categorical variables, to prepare it for
  machine learning.
- **Model Training**: Use a machine learning library (e.g., Scikit-learn) to
  train a regression model to predict AQI levels based on the prepared dataset.
- **Model Evaluation**: Evaluate the model's performance using appropriate
  metrics (e.g., RMSE, MAE) and visualize the results to assess prediction
  accuracy.

## Bonus Ideas
- Extend the project to analyze trends in air quality over time and create
  visualizations that showcase these trends.
- Compare the performance of different regression algorithms (e.g., Linear
  Regression, Random Forest) to see which yields the best predictions.
- Implement a real-time data pipeline that continuously fetches air quality data
  and updates the model predictions.

## Useful Resources
- [Trifacta Wrangler Official Documentation](https://www.trifacta.com/product/wrangler/)
- [OpenAQ API Documentation](https://docs.openaq.org/)
- [World Air Quality Index API Documentation](https://aqicn.org/json-api/doc/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
