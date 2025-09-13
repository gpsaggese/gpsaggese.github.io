**Description**

In this project, students will utilize tsfresh, a Python package designed for time series analysis, to extract relevant features from time series data and apply machine learning for classification or regression tasks. tsfresh automates the feature extraction process, allowing students to focus on model training and evaluation. The tool is particularly useful for transforming raw time series data into a format suitable for machine learning.

Technologies Used
tsfresh

- Automates the extraction of numerous time series features.
- Offers statistical tests to select relevant features for predictive modeling.
- Supports integration with various machine learning libraries.

---

### Project 1: Time Series Classification of Human Activity Recognition (Difficulty: 1)

**Project Objective**  
The goal is to classify different human activities (walking, sitting, standing) based on accelerometer data collected from smartphones. Students will optimize the classification accuracy of their model.

**Dataset Suggestions**  
- UCI HAR Dataset: Available on [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones).

**Tasks**
- Data Preparation:
  - Load the dataset and preprocess time series signals (accelerometer data).
- Feature Extraction:
  - Use tsfresh to extract relevant features from the time series data.
- Model Training:
  - Train a classification model (e.g., Random Forest or SVM) using the extracted features.
- Model Evaluation:
  - Evaluate the model’s performance using accuracy and confusion matrix.
- Visualization:
  - Visualize the feature importance and model predictions.

---

### Project 2: Predicting Stock Prices Using Historical Data (Difficulty: 2)

**Project Objective**  
The objective is to predict future stock prices based on historical time series data. Students will optimize their models for the best predictive accuracy.

**Dataset Suggestions**  
- Historical stock prices for a specific company (e.g., Apple Inc.) can be obtained from the [Yahoo Finance API](https://pypi.org/project/yfinance/) (free and active).

**Tasks**
- Data Collection:
  - Use the Yahoo Finance API to gather historical stock price data.
- Data Preprocessing:
  - Clean the dataset and create time series sequences for modeling.
- Feature Extraction:
  - Apply tsfresh to extract time series features from the stock price data.
- Model Training:
  - Implement regression models (e.g., Linear Regression or XGBoost) to predict future prices.
- Evaluation:
  - Assess model performance using metrics like RMSE and R-squared.
- Visualization:
  - Plot actual vs. predicted stock prices over time.

---

### Project 3: Anomaly Detection in IoT Sensor Data (Difficulty: 3)

**Project Objective**  
The goal is to detect anomalies in time series data collected from IoT sensors in a smart home environment. Students will optimize their approach to identify unusual patterns effectively.

**Dataset Suggestions**  
- The NASA Turbofan Engine Degradation Simulation Dataset is available on [Kaggle](https://www.kaggle.com/datasets/behnamf/engine-degradation-dataset).

**Tasks**
- Data Acquisition:
  - Download and preprocess the NASA engine degradation dataset.
- Data Exploration:
  - Analyze the time series data for initial insights and patterns.
- Feature Extraction:
  - Use tsfresh to extract features that may indicate anomalies in the sensor readings.
- Anomaly Detection:
  - Implement anomaly detection algorithms (e.g., Isolation Forest or Autoencoders) using the extracted features.
- Evaluation:
  - Evaluate the model’s ability to detect anomalies using precision, recall, and F1 score.
- Visualization:
  - Visualize detected anomalies against the time series data to understand patterns.

**Bonus Ideas (Optional)**  
- For Project 1, students can explore using deep learning models (e.g., LSTM) for classification.
- For Project 2, consider implementing a trading strategy based on the predicted stock prices.
- For Project 3, extend the project to classify the types of anomalies detected and visualize them in real-time.

