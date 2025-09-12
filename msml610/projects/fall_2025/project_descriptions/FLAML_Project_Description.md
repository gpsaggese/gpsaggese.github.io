**Description**

FLAML (Fast and Lightweight AutoML) is an open-source Python library designed for automating machine learning tasks efficiently and with minimal resource consumption. It focuses on quick model training and hyperparameter tuning, making it suitable for both beginners and advanced users. 

Features:
- Efficiently automates the machine learning pipeline, including model selection and hyperparameter tuning.
- Supports various machine learning tasks such as classification, regression, and time-series forecasting.
- Lightweight design allows for easy integration into existing workflows without extensive computational resources.

---

**Project 1: Predicting Housing Prices**  
**Difficulty**: 1 (Easy)  
**Project Objective**: The goal is to predict housing prices based on various attributes such as location, size, and amenities using a regression model. The project aims to optimize the model for accuracy and interpretability.

**Dataset Suggestions**: 
- Use the "California Housing Prices" dataset available on Kaggle ([California Housing Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)).

**Tasks**:
- Data Preprocessing:
    - Clean and preprocess the dataset to handle missing values and categorical variables.
- Feature Engineering:
    - Create new features based on existing ones (e.g., price per square foot).
- Model Training with FLAML:
    - Utilize FLAML to automatically select and tune regression models (e.g., Random Forest, XGBoost).
- Model Evaluation:
    - Evaluate the model's performance using appropriate metrics (RMSE, MAE).
- Visualization:
    - Visualize the predictions against actual prices using Matplotlib.

**Bonus Ideas**: 
- Compare the results of FLAML with traditional manual hyperparameter tuning.
- Explore feature importance and its impact on model predictions.

---

**Project 2: Customer Churn Prediction**  
**Difficulty**: 2 (Medium)  
**Project Objective**: The project aims to predict customer churn for a telecommunications company, optimizing the model to minimize false negatives, which represent customers who are incorrectly predicted to stay.

**Dataset Suggestions**: 
- Use the "Telco Customer Churn" dataset from Kaggle ([Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)).

**Tasks**:
- Data Exploration:
    - Conduct exploratory data analysis (EDA) to understand customer demographics and churn patterns.
- Data Preprocessing:
    - Handle missing values and encode categorical variables.
- Model Training with FLAML:
    - Utilize FLAML to automate the selection and tuning of classification models (e.g., Logistic Regression, Decision Trees).
- Model Evaluation:
    - Evaluate performance using precision, recall, and F1-score, focusing on minimizing false negatives.
- Visualization:
    - Create confusion matrices and ROC curves to interpret model performance.

**Bonus Ideas**: 
- Implement additional techniques like SMOTE for handling class imbalance.
- Compare FLAML's results with other AutoML libraries like TPOT or H2O.ai.

---

**Project 3: Time-Series Forecasting of Stock Prices**  
**Difficulty**: 3 (Hard)  
**Project Objective**: The objective is to forecast future stock prices using historical data, optimizing the model for prediction accuracy while managing the challenges of time-series data.

**Dataset Suggestions**: 
- Use the "Historical Stock Prices" dataset available on Kaggle ([Stock Prices Dataset](https://www.kaggle.com/datasets/srajanmishra/stock-prices-dataset)).

**Tasks**:
- Data Preparation:
    - Clean the dataset and handle missing values, ensuring proper formatting for time-series analysis.
- Feature Engineering:
    - Create lag features and moving averages to enhance the dataset.
- Model Training with FLAML:
    - Use FLAML to automate the selection and hyperparameter tuning of time-series forecasting models (e.g., ARIMA, Prophet).
- Model Evaluation:
    - Evaluate the model's forecasting accuracy using metrics like MAPE and RMSE.
- Visualization:
    - Plot the actual vs. predicted stock prices over time to assess model performance.

**Bonus Ideas**: 
- Implement a rolling forecast to continuously update predictions as new data comes in.
- Explore ensemble methods by combining predictions from multiple models to improve accuracy.

