# Snowflake

## Description
- **Cloud Data Platform**: Snowflake is a cloud-based data warehousing platform
  that allows users to store, manage, and analyze large volumes of data
  efficiently.
- **Seamless Data Sharing**: It offers capabilities for secure and easy data
  sharing across different organizations and teams without the need for complex
  data movement.
- **Scalability**: Snowflake automatically scales compute and storage resources
  independently, enabling users to handle varying workloads without performance
  degradation.
- **SQL-Based Interface**: Users can interact with Snowflake using standard SQL,
  making it accessible for those familiar with traditional database queries.
- **Support for Diverse Data Types**: It can handle structured and
  semi-structured data (like JSON, Avro, and Parquet) natively, allowing for
  flexible data modeling.

## Project Objective
The goal of this project is to build a data pipeline that ingests, processes,
and analyzes a public dataset using Snowflake. Students will optimize a machine
learning model to predict housing prices based on various features, leveraging
Snowflake's capabilities for data storage and SQL querying.

## Dataset Suggestions
1. **California Housing Prices**
   - **Source**: Kaggle
   - **URL**:
     [California Housing Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   - **Data Contains**: Features related to housing prices including location,
     number of rooms, and property age.
   - **Access Requirements**: Free to use with a Kaggle account.

2. **Airbnb Listings**
   - **Source**: Inside Airbnb
   - **URL**: [Inside Airbnb](http://insideairbnb.com/get-the-data.html)
   - **Data Contains**: Information about Airbnb listings including price,
     location, and amenities.
   - **Access Requirements**: Publicly available CSV files; no authentication
     needed.

3. **World Happiness Report**
   - **Source**: Kaggle
   - **URL**:
     [World Happiness Report Dataset](https://www.kaggle.com/datasets/unsdsn/world-happiness)
   - **Data Contains**: Happiness scores of countries alongside various
     socio-economic factors.
   - **Access Requirements**: Free to use with a Kaggle account.

4. **Wine Quality Dataset**
   - **Source**: UCI Machine Learning Repository
   - **URL**:
     [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
   - **Data Contains**: Chemical properties of wine and associated quality
     ratings.
   - **Access Requirements**: Publicly available; no authentication needed.

## Tasks
- **Data Ingestion**: Set up a Snowflake account and create a database to ingest
  the selected dataset using Snowflake's data loading features.
- **Data Exploration**: Use SQL queries to explore the dataset, identify key
  features, and perform initial data cleaning within Snowflake.
- **Feature Engineering**: Create new features based on existing data that may
  improve the model's performance, such as aggregating or transforming existing
  variables.
- **Model Training**: Utilize a pre-trained model or build a regression model
  using a selected machine learning library (e.g., Scikit-learn) to predict
  housing prices based on the features.
- **Model Evaluation**: Evaluate the model's performance using metrics such as
  RMSE or R-squared, and visualize the results using Snowflake's integration
  with BI tools (like Tableau or Power BI).

## Bonus Ideas
- **Data Visualization**: Integrate Snowflake with a visualization tool to
  create interactive dashboards showcasing the analysis and predictions.
- **Hyperparameter Tuning**: Experiment with hyperparameter tuning techniques to
  improve the model's accuracy.
- **Anomaly Detection**: Extend the project to include anomaly detection on
  housing prices to identify unusual listings.

## Useful Resources
- [Snowflake Documentation](https://docs.snowflake.com/en/)
- [Snowflake SQL Reference](https://docs.snowflake.com/en/sql-reference.html)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Inside Airbnb Data](http://insideairbnb.com/get-the-data.html)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
