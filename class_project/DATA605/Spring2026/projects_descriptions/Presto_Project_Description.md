# Presto

## Description
- Presto is an open-source distributed SQL query engine designed for running
  interactive analytic queries against various data sources.
- It allows querying data where it lives, including Hadoop, AWS S3, MySQL, and
  more, without needing to move or transform the data.
- Presto supports a wide range of SQL functionalities, making it suitable for
  complex queries and analytics across large datasets.
- It is designed for speed and scalability, enabling users to run queries on
  petabytes of data in seconds.
- Presto integrates well with other data processing tools and platforms, making
  it a flexible choice for data analytics workflows.

## Project Objective
The goal of this project is to analyze a large dataset of global COVID-19 cases
to identify trends and make predictions about future case surges. Students will
create a predictive model to forecast daily case counts based on historical data
and various socio-economic factors.

## Dataset Suggestions
1. **COVID-19 Data Repository by the Center for Systems Science and Engineering
   (CSSE) at Johns Hopkins University**
   - URL: [Johns Hopkins GitHub](https://github.com/CSSEGISandData/COVID-19)
   - Contains: Daily reported cases, deaths, and recoveries by country and
     region.
   - Access Requirements: Publicly accessible; no authentication needed.

2. **COVID-19 Open Data by Google Cloud**
   - URL:
     [Google Cloud COVID-19 Open Data](https://cloud.google.com/covid19-data)
   - Contains: Global COVID-19 case counts, vaccination data, and socio-economic
     indicators.
   - Access Requirements: Publicly accessible; no authentication needed.

3. **Our World in Data COVID-19 Dataset**
   - URL: [Our World in Data](https://covid.ourworldindata.org/)
   - Contains: Comprehensive data on COVID-19 cases, testing, and vaccination
     rates across countries.
   - Access Requirements: Publicly accessible; no authentication needed.

4. **COVID-19 Data from Kaggle**
   - URL:
     [Kaggle COVID-19 Dataset](https://www.kaggle.com/datasets/sudalairajkumar/covid19-in-india)
   - Contains: Detailed records of COVID-19 cases in India, including
     demographic data.
   - Access Requirements: Free registration required on Kaggle.

## Tasks
- **Data Ingestion**: Use Presto to connect to the chosen dataset and perform
  initial data loading and exploration.
- **Data Cleaning**: Clean and preprocess the data using SQL queries to handle
  missing values and outliers.
- **Feature Engineering**: Create new features based on existing data, such as
  rolling averages of case counts and population density.
- **Model Training**: Implement a time-series forecasting model (e.g., ARIMA or
  Prophet) to predict future COVID-19 case counts.
- **Model Evaluation**: Evaluate the model's performance using metrics such as
  Mean Absolute Error (MAE) and visualize the predictions against actual data.
- **Reporting**: Create a comprehensive report summarizing findings, insights,
  and recommendations based on the analysis.

## Bonus Ideas
- **Comparative Analysis**: Compare the predictive performance of different
  models (e.g., ARIMA vs. machine learning models).
- **Geographical Insights**: Analyze how socio-economic factors influence
  COVID-19 case rates in different regions using clustering techniques.
- **Interactive Dashboard**: Build a dashboard using tools like Tableau or
  Streamlit to visualize trends and predictions interactively.

## Useful Resources
- [Presto Documentation](https://prestodb.io/docs/current/)
- [Presto GitHub Repository](https://github.com/prestodb/presto)
- [Kaggle COVID-19 Dataset](https://www.kaggle.com/datasets/sudalairajkumar/covid19-in-india)
- [Google Cloud COVID-19 Open Data](https://cloud.google.com/covid19-data)
- [Our World in Data COVID-19 Dataset](https://covid.ourworldindata.org/)
