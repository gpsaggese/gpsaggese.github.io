# Singer

## Description
- **Singer** is an open-source framework designed for data integration, focusing
  on the extraction and loading of data from various sources into data
  warehouses or analytics platforms.
- It uses a simple, standardized format for defining data extraction processes,
  known as "taps" for data extraction and "targets" for data loading.
- The tool allows users to create reusable data pipelines that can be easily
  shared and modified, promoting efficient data workflows.
- Singer supports a wide variety of data sources and destinations, making it
  versatile for different data engineering tasks.
- It emphasizes modularity, enabling users to combine different taps and targets
  to build complex data integration workflows with minimal effort.

## Project Objective
The goal of this project is to build a data pipeline that extracts data from a
public API, transforms it as necessary, and loads it into a data warehouse for
analysis. Students will focus on optimizing the data extraction process and
ensuring data quality during the transformation phase.

## Dataset Suggestions
1. **OpenWeatherMap API**
   - **URL**: [OpenWeatherMap API](https://openweathermap.org/api)
   - **Data Contains**: Current weather data, forecasts, and historical weather
     data for various locations.
   - **Access Requirements**: Free tier available with registration for an API
     key.

2. **COVID-19 Open Data**
   - **URL**: [COVID-19 Open Data](https://covid19data.com/)
   - **Data Contains**: Global COVID-19 statistics, including case counts,
     vaccination rates, and demographic information.
   - **Access Requirements**: Publicly available without authentication.

3. **Kaggle's UCI ML Repository - Wine Quality**
   - **URL**:
     [Wine Quality Dataset](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009)
   - **Data Contains**: Quality ratings and chemical properties of red and white
     wines.
   - **Access Requirements**: Free to use after signing up for a Kaggle account.

4. **GitHub Repositories - Global Terrorism Database**
   - **URL**: [Global Terrorism Database](https://www.start.umd.edu/gtd/)
   - **Data Contains**: Information on terrorist attacks around the world,
     including location, date, and casualties.
   - **Access Requirements**: Open access; no authentication needed.

## Tasks
- **Setup Singer Environment**: Install Singer and configure the necessary taps
  and targets for the chosen dataset.
- **Extract Data**: Use a Singer tap to extract data from the selected API,
  ensuring to handle pagination if necessary.
- **Transform Data**: Clean and preprocess the data (e.g., filtering,
  normalization) to ensure it meets the requirements of the target schema.
- **Load Data**: Use a Singer target to load the transformed data into a chosen
  data warehouse (e.g., Google BigQuery, PostgreSQL).
- **Validate Data**: Implement checks to ensure data integrity and quality after
  loading, such as checking for missing values or duplicates.
- **Document the Pipeline**: Create comprehensive documentation of the data
  pipeline, including how to run it and any assumptions made during the process.

## Bonus Ideas
- **Add Scheduling**: Implement a scheduling mechanism to automate the data
  extraction process at regular intervals.
- **Data Visualization**: Create a dashboard using tools like Tableau or Power
  BI to visualize the loaded data.
- **Model Training**: Use the loaded data to train a machine learning model
  (e.g., regression model to predict wine quality based on chemical properties).
- **Comparison with Other Tools**: Compare the performance of Singer with
  another ETL tool (e.g., Apache Airflow) in terms of ease of use and execution
  time.

## Useful Resources
- [Singer Documentation](https://www.singer.io/docs/)
- [Singer Taps and Targets Catalog](https://github.com/singer-io/getting-started/blob/master/taps.md)
- [OpenWeatherMap API Documentation](https://openweathermap.org/api)
- [COVID-19 Open Data Documentation](https://covid19data.com/documentation)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
