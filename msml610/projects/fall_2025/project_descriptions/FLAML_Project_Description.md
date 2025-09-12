**Description**

FLAML (Fast and Lightweight AutoML) is an open-source Python library designed for automating machine learning tasks efficiently and with minimal resource consumption. It focuses on quick model training and hyperparameter tuning, making it suitable for both beginners and advanced users. 

Features:
- Efficiently automates the machine learning pipeline, including model selection and hyperparameter tuning.  
- Supports various machine learning tasks such as classification, regression, and time-series forecasting.  
- Lightweight design allows for easy integration into existing workflows without extensive computational resources.  

---

### Project 1: Predicting Airbnb Rental Prices  
**Difficulty**: 1 (Easy)  

**Project Objective**  
Predict Airbnb rental prices based on attributes such as location, number of rooms, and amenities. The project aims to build an accurate and interpretable regression model while comparing multiple FLAML-selected regressors.  

**Dataset Suggestions**  
- [New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) on Kaggle, which includes listings with price, location, and property details.  

**Tasks**  
- **Data Preprocessing**: Clean the dataset, handle missing values, and encode categorical features.  
- **Feature Engineering**: Create features such as price per bedroom and neighborhood groupings.  
- **Model Training with FLAML**: Use FLAML to automatically select and tune regression models (e.g., Linear Regression, Random Forest, XGBoost).  
- **Model Comparison**: Compare FLAML’s top 2–3 models on validation data.  
- **Evaluation**: Report RMSE and MAE for each model.  
- **Visualization**: Plot predicted vs. actual prices and feature importance.  

**Bonus Ideas (Optional)**  
- Manually train a baseline Linear Regression model to compare with FLAML’s automated choices.  
- Investigate seasonal effects on Airbnb prices.  

---

### Project 2: Employee Attrition Prediction  
**Difficulty**: 2 (Medium)  

**Project Objective**  
Predict whether an employee is likely to leave the company, focusing on minimizing false negatives. The project demonstrates how FLAML can test multiple classification models efficiently.  

**Dataset Suggestions**  
- [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) dataset on Kaggle.  

**Tasks**  
- **EDA**: Explore attrition patterns across departments, roles, and demographics.  
- **Preprocessing**: Handle missing values, encode categorical features, normalize numerical data.  
- **Model Training with FLAML**: Run FLAML to train multiple classifiers (e.g., Logistic Regression, Random Forest, LightGBM, XGBoost).  
- **Model Comparison**: Evaluate and compare models on precision, recall, and F1-score. Highlight the trade-off between accuracy and recall.  
- **Visualization**: Plot confusion matrices and ROC curves for top models.  

**Bonus Ideas (Optional)**  
- Use SMOTE to address class imbalance and rerun FLAML to see performance differences.  
- Compare FLAML’s results with another AutoML tool (e.g., TPOT).  

---

### Project 3: Time-Series Forecasting of Energy Consumption  
**Difficulty**: 3 (Hard)  

**Project Objective**  
Forecast household energy consumption using historical usage data. The project focuses on experimenting with multiple forecasting models through FLAML to handle seasonality and optimize accuracy.  

**Dataset Suggestions**  
- [Household Electric Power Consumption](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption) dataset from the UCI ML Repository.  

**Tasks**  
- **Data Preparation**: Clean and resample data (daily/weekly aggregates), handle missing values.  
- **Feature Engineering**: Create lag features, rolling averages, and holiday/weekend indicators.  
- **Model Training with FLAML**: Use FLAML to automate selection across forecasting models (e.g., ARIMA, Prophet, LightGBM, XGBoost).  
- **Model Comparison**: Compare performance of at least 2 FLAML-selected models on RMSE and MAPE.  
- **Visualization**: Plot predicted vs. actual consumption for top models.  
- **Analysis**: Discuss which model handled seasonality and volatility best.  

**Bonus Ideas (Optional)**  
- Implement a rolling forecast evaluation for robustness.  
- Build an ensemble forecast by averaging top FLAML models.  
