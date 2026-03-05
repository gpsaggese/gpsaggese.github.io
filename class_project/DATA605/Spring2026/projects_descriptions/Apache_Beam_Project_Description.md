# Apache Beam

## Description
- Apache Beam is an open-source unified model for defining both batch and
  streaming data-parallel processing pipelines.
- It allows users to write data processing jobs that can run on various
  execution engines, such as Apache Spark, Apache Flink, and Google Cloud
  Dataflow.
- Beam provides a rich set of built-in transformations for manipulating data,
  including filtering, grouping, and windowing.
- It supports both Java and Python SDKs, making it accessible to a wide range of
  developers and data scientists.
- With its extensible architecture, users can create custom transformations and
  I/O connectors for specialized data sources and sinks.

## Project Objective
The goal of the project is to build a data processing pipeline using Apache Beam
that ingests a public dataset, performs data transformation and aggregation, and
outputs the results for further analysis. Students will focus on optimizing the
performance of the pipeline and ensuring data quality through validation checks.

## Dataset Suggestions
1. **Kaggle's Global Terrorism Database**
   - **Source**: Kaggle
   - **URL**:
     [Global Terrorism Database](https://www.kaggle.com/datasets/uciml/global-terrorism-dataset)
   - **Data Contains**: Information on terrorist incidents around the world,
     including location, date, and type of attack.
   - **Access Requirements**: Free to use; requires a Kaggle account for
     downloading.

2. **OpenWeatherMap Historical Weather Data**
   - **Source**: OpenWeatherMap
   - **URL**:
     [OpenWeatherMap Historical Data](https://openweathermap.org/history)
   - **Data Contains**: Historical weather data, including temperature,
     humidity, and precipitation for various locations.
   - **Access Requirements**: Free tier available; requires API key (sign-up is
     free).

3. **UCI Machine Learning Repository - Wine Quality Dataset**
   - **Source**: UCI Machine Learning Repository
   - **URL**:
     [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
   - **Data Contains**: Chemical properties and quality ratings of red and white
     wines.
   - **Access Requirements**: Publicly available without any authentication.

4. **COVID-19 Open Data**
   - **Source**: Google Cloud
   - **URL**: [COVID-19 Open Data](https://covid19data.com/)
   - **Data Contains**: A comprehensive dataset containing COVID-19 case counts,
     testing, hospitalizations, and vaccination data.
   - **Access Requirements**: Publicly accessible via BigQuery without
     authentication.

## Tasks
- **Pipeline Design**: Create a data processing pipeline in Apache Beam that
  ingests the chosen dataset and defines the necessary transformations.
- **Data Transformation**: Implement data cleaning and preprocessing steps,
  including filtering out irrelevant data and handling missing values.
- **Aggregation**: Use Beam's built-in transformations to aggregate data (e.g.,
  counting incidents by year or calculating average temperatures).
- **Output Results**: Write the processed data to a suitable output format (CSV,
  JSON, or a database) for further analysis.
- **Performance Optimization**: Analyze the pipeline's performance and optimize
  it by experimenting with different windowing and triggering strategies.

## Bonus Ideas
- Implement a feature to visualize the processed data using a library like
  Matplotlib or Seaborn after exporting it.
- Compare the performance of your Apache Beam pipeline with a similar pipeline
  built using a different framework (e.g., Apache Spark).
- Extend the project to include real-time data processing by integrating a
  streaming data source, such as Twitter or a public API that provides live
  updates.

## Useful Resources
- [Apache Beam Documentation](https://beam.apache.org/documentation/)
- [Apache Beam GitHub Repository](https://github.com/apache/beam)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [OpenWeatherMap API Documentation](https://openweathermap.org/api)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
