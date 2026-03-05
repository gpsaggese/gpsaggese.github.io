# Orbit

## Description
- **Orbit** is a Python library designed for probabilistic forecasting,
  specifically tailored for time series data.
- It offers a comprehensive framework for modeling and predicting time series
  data using Bayesian techniques.
- Key features include easy-to-use APIs for model fitting, forecasting, and
  evaluation, along with built-in support for uncertainty quantification.
- Orbit allows users to incorporate covariates and seasonal effects into their
  models, making it versatile for various applications.
- The library is designed to work seamlessly with popular data science tools
  like Pandas and NumPy, facilitating integration into existing workflows.

## Project Objective
The goal of this project is to forecast future sales for a retail store based on
historical sales data. Students will build a probabilistic model to predict
monthly sales figures, optimizing for accuracy and uncertainty estimation in the
forecasts.

## Dataset Suggestions
1. **M5 Forecasting - Accuracy Competition Dataset**
   - **Source**: Kaggle
   - **URL**:
     [M5 Forecasting Dataset](https://www.kaggle.com/c/m5-forecasting-accuracy/data)
   - **Data**: Contains historical sales data for various products across
     different stores, including sales per product per day.
   - **Access Requirements**: Free to use with a Kaggle account.

2. **Retail Sales Forecasting Dataset**
   - **Source**: Kaggle
   - **URL**:
     [Retail Sales Forecasting Dataset](https://www.kaggle.com/datasets/retail/retail-sales-forecasting)
   - **Data**: Includes monthly sales data from a retail store, featuring
     additional attributes like promotions and holidays.
   - **Access Requirements**: Free to use with a Kaggle account.

3. **Walmart Sales Forecasting Dataset**
   - **Source**: Kaggle
   - **URL**:
     [Walmart Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data)
   - **Data**: Contains historical store sales data, including store locations,
     promotions, and holiday indicators.
   - **Access Requirements**: Free to use with a Kaggle account.

4. **COVID-19 Retail Sales Data**
   - **Source**: Kaggle
   - **URL**:
     [COVID-19 Retail Sales Data](https://www.kaggle.com/datasets/akanksha0410/covid19-retail-sales-data)
   - **Data**: Provides retail sales data during the COVID-19 pandemic, allowing
     for analysis of sales trends influenced by external factors.
   - **Access Requirements**: Free to use with a Kaggle account.

## Tasks
- **Data Collection**: Download and preprocess the selected dataset to ensure it
  is clean and ready for analysis.
- **Exploratory Data Analysis (EDA)**: Conduct EDA to understand sales trends,
  seasonality, and any influencing factors in the dataset.
- **Model Development**: Use Orbit to build a probabilistic forecasting model
  that incorporates historical sales data and relevant covariates.
- **Model Evaluation**: Assess the model's performance using metrics like Mean
  Absolute Error (MAE) and visualize forecast uncertainty.
- **Reporting**: Prepare a report summarizing the findings, including
  visualizations of the forecasts and insights derived from the model.

## Bonus Ideas
- Implement additional models (e.g., ARIMA, Prophet) for comparison against the
  Orbit model to evaluate performance differences.
- Explore advanced features of Orbit, such as incorporating external covariates
  (e.g., economic indicators) into the forecasting model.
- Challenge students to create a dashboard using Plotly or Streamlit to
  visualize real-time sales forecasts and uncertainty estimates.

## Useful Resources
- [Orbit Documentation](https://orbit.readthedocs.io/en/stable/)
- [Kaggle - M5 Forecasting Competition](https://www.kaggle.com/c/m5-forecasting-accuracy)
- [Time Series Analysis with Orbit - GitHub Repository](https://github.com/cambridge-ml/Orbit)
- [Understanding Bayesian Forecasting](https://towardsdatascience.com/bayesian-forecasting-in-python-using-orbit-3e5c1b5f3f2e)
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/) for data
  manipulation and analysis.
