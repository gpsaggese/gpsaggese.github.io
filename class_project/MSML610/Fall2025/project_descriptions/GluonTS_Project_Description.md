**Description**

GluonTS is a powerful Python toolkit for probabilistic time series modeling, built on top of Apache MXNet. It provides a flexible framework for building, training, and evaluating models for time series forecasting. With a rich collection of pre-built models and utilities, GluonTS simplifies the process of working with temporal data.

Technologies Used
GluonTS

- Offers a variety of state-of-the-art forecasting models, including ARIMA, DeepAR, and Transformer-based models.
- Facilitates easy model training and evaluation with built-in metrics and visualization tools.
- Supports probabilistic forecasting, allowing users to quantify uncertainty in predictions.

---

### Project 1: Sales Forecasting for Retail Products
**Difficulty**: 1 (Easy)

**Project Objective**: The goal is to build a forecasting model that predicts future sales for a retail store's products based on historical sales data. The project aims to optimize inventory management by providing accurate sales predictions.

**Dataset Suggestions**: 
- Use the "Store Item Demand Forecasting Challenge" dataset available on Kaggle. 
- Link: [Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only/data)

**Tasks**:
- Data Preprocessing:
    - Clean and format the dataset to create time series for each product.
    - Handle missing values and outliers appropriately.

- Model Selection:
    - Explore and select an appropriate model from GluonTS (e.g., DeepAR).

- Model Training:
    - Split the dataset into training and validation sets.
    - Train the model on historical sales data.

- Forecasting:
    - Generate sales forecasts for the next month.
    - Evaluate the model using metrics like MAPE or RMSE.

- Visualization:
    - Plot the actual vs. predicted sales to visualize performance.

---

### Project 2: Energy Consumption Forecasting
**Difficulty**: 2 (Medium)

**Project Objective**: The objective is to predict future energy consumption for a city using historical energy usage data. The project aims to optimize energy distribution and planning for utility companies.

**Dataset Suggestions**: 
- Use the "Household Power Consumption" dataset available on the UCI Machine Learning Repository. 
- Link: [Household Power Consumption](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)

**Tasks**:
- Data Preparation:
    - Load the dataset and convert the timestamp into a time series format.
    - Aggregate energy consumption data to a daily granularity.
    - Aggregate to daily/weekly to keep training manageable.

- Feature Engineering:
    - Create additional features such as day of the week, month, and holidays.

- Model Training:
    - Select and train a model from GluonTS (e.g., ARIMA or NBEATS) on the prepared dataset.

- Forecasting:
    - Generate daily energy consumption forecasts for the next three months.
    - Evaluate model performance using appropriate metrics.

- Analysis:
    - Analyze the impact of external factors (e.g., temperature, holidays) on energy consumption.

---

### Project 3: COVID-19 Case Prediction
**Difficulty**: 3 (Hard)

**Project Objective**: The aim is to forecast the number of COVID-19 cases in a specific region using historical case data. This project focuses on understanding the dynamics of the outbreak and optimizing resource allocation in healthcare.

**Dataset Suggestions**: 
- Use the "COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University."
- Link: [COVID-19 Data Repository](https://github.com/CSSEGISandData/COVID-19)

**Tasks**:
- Data Ingestion:
    - Collect and preprocess daily reported COVID-19 cases for the selected region.
    - Focus on a single country or region to reduce complexity.

- Time Series Transformation:
    - Transform the dataset into a format suitable for time series forecasting.

- Model Exploration:
    - Experiment with multiple advanced models in GluonTS (e.g., Transformer or Gaussian Process).

- Model Training & Evaluation:
    - Train the selected models and evaluate their performance using metrics like the Continuous Ranked Probability Score (CRPS).

- Uncertainty Quantification:
    - Analyze the uncertainty in the predictions and visualize the confidence intervals.

- Scenario Analysis:
    - Conduct scenario analysis to understand potential future outbreaks under different public health interventions.

**Bonus Ideas (Optional)**:
- For Project 1, consider adding promotions or seasonal effects in the forecasting model.
- For Project 2, explore external temperature data by downloading historical weather datasets (e.g., from NOAA or Kaggle)
- For Project 3, incorporate mobility data (e.g., Google Mobility Reports) to enhance predictions.

