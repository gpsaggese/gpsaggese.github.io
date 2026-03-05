# Bonobo

## Description
- Bonobo is a lightweight, open-source Python framework designed for building
  data pipelines and ETL (Extract, Transform, Load) processes.
- It allows users to create data workflows using a simple and intuitive API,
  making it accessible for both beginners and experienced data engineers.
- The framework supports parallel processing, enabling efficient handling of
  large datasets and improving performance for data-intensive tasks.
- Bonobo includes built-in connectors for various data sources such as CSV
  files, databases, and APIs, facilitating seamless data integration.
- It features a visual representation of data flows, helping users understand
  and debug their pipelines more easily.
- Bonobo is highly extensible, allowing users to create custom transformations
  and connectors to meet specific project needs.

## Project Objective
The goal of this project is to build a data pipeline that extracts data from a
public API, transforms it into a structured format, and loads it into a database
for further analysis. The main focus will be on optimizing the data extraction
and transformation processes to ensure efficiency and accuracy.

## Dataset Suggestions
1. **OpenWeatherMap API**
   - **Source Name**: OpenWeatherMap
   - **URL**: [OpenWeatherMap API](https://openweathermap.org/api)
   - **Data Contains**: Weather data including temperature, humidity, wind
     speed, and conditions for various cities.
   - **Access Requirements**: Free tier available with API key registration.

2. **COVID-19 Data Repository by the Center for Systems Science and Engineering
   (CSSE) at Johns Hopkins University**
   - **Source Name**: Johns Hopkins University
   - **URL**:
     [COVID-19 Data Repository](https://github.com/CSSEGISandData/COVID-19)
   - **Data Contains**: Daily COVID-19 case counts, recoveries, and deaths by
     country and region.
   - **Access Requirements**: Publicly accessible on GitHub.

3. **Kaggle's World Happiness Report**
   - **Source Name**: Kaggle
   - **URL**:
     [World Happiness Report Dataset](https://www.kaggle.com/datasets/unsdsn/world-happiness)
   - **Data Contains**: Happiness scores and rankings of countries based on
     various factors such as GDP, social support, and life expectancy.
   - **Access Requirements**: Free to download with a Kaggle account.

4. **UCI Machine Learning Repository - Wine Quality Dataset**
   - **Source Name**: UCI Machine Learning Repository
   - **URL**:
     [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
   - **Data Contains**: Chemical properties and quality ratings of red and white
     wines.
   - **Access Requirements**: Publicly accessible without authentication.

## Tasks
- **Task 1: Data Extraction**
  - Use Bonobo to create a pipeline that extracts data from the chosen API or
    dataset, ensuring proper handling of data types and formats.
- **Task 2: Data Transformation**
  - Implement transformation functions to clean and preprocess the extracted
    data, such as handling missing values, normalizing data, or converting data
    types.
- **Task 3: Data Loading**
  - Load the transformed data into a SQLite database or a CSV file for analysis,
    ensuring that the data structure is optimized for querying.
- **Task 4: Data Analysis**
  - Perform basic exploratory data analysis (EDA) on the loaded data using
    Python libraries such as Pandas and Matplotlib to visualize trends and
    insights.

## Bonus Ideas
- Extend the project by adding additional data sources and merging them into a
  single dataset for more comprehensive analysis.
- Implement anomaly detection on the dataset to identify unusual patterns or
  outliers in the data.
- Compare the performance of different data transformation techniques and
  document their impact on the final dataset quality.

## Useful Resources
- [Bonobo Documentation](https://bonobo.readthedocs.io/en/stable/)
- [OpenWeatherMap API Documentation](https://openweathermap.org/api)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
