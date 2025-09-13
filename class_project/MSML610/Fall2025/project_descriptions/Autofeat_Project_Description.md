## Description  
Autofeat is a Python library designed to automate the feature engineering process by generating new features from existing ones, optimizing their selection based on model performance. It helps data scientists enhance their models by automatically identifying relevant features, thus streamlining the modeling pipeline.  

---

### Project 1: Predicting House Prices  
**Difficulty**: 1 (Easy)  

**Project Objective**:  
Build a regression model to predict house prices based on various features such as location, size, and amenities. The goal is to optimize the feature set to improve prediction accuracy.  

**Dataset Suggestions**:  
- **Dataset**: Melbourne Housing Market Dataset  
- **Link**: [Melbourne Housing Market (Kaggle)](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot)  

**Tasks**:  
- **Data Preprocessing**: Clean the dataset by handling missing values and encoding categorical variables.  
- **Feature Generation with Autofeat**: Use Autofeat to automatically create and select new features.  
- **Model Training (Multiple Models)**:  
  - Baseline: Linear Regression and Ridge Regression.  
  - Tree-based: Random Forest Regressor.  
  - Gradient-based: XGBoost Regressor.  
- **Evaluation**: Compare models using RMSE and R², both with and without Autofeat features.  

**Bonus Ideas (Optional)**:  
- Visualize feature importance for tree-based models.  
- Explore interaction features created by Autofeat to understand housing price drivers.  

---

### Project 2: Predicting Bank Marketing Campaign Success  
**Difficulty**: 2 (Medium)  

**Project Objective**:  
Develop a classification model to predict whether a client will subscribe to a term deposit after a marketing campaign, using demographic and behavioral data.  

**Dataset Suggestions**:  
- **Dataset**: Bank Marketing Dataset  
- **Link**: [Bank Marketing (Kaggle)](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing)  

**Tasks**:  
- **Data Exploration**: Perform EDA to understand customer demographics and campaign performance.  
- **Feature Engineering with Autofeat**: Apply Autofeat to generate new features capturing customer decision patterns.  
- **Model Training (Multiple Models)**:  
  - Baseline: Logistic Regression and k-Nearest Neighbors.  
  - Tree-based: Random Forest and Gradient Boosting Classifier.  
  - Ensemble: XGBoost or LightGBM.  
- **Model Evaluation**: Compare models using accuracy, F1-score, and confusion matrices, with and without Autofeat features.  

**Bonus Ideas (Optional)**:  
- Test cost-sensitive classification to reflect the business value of correct vs. incorrect predictions.  
- Build a dashboard to visualize campaign results and predicted customer segments.  

---

### Project 3: Network Intrusion Detection  
**Difficulty**: 3 (Hard)  

**Project Objective**:  
Build an anomaly detection model to identify potential intrusions in network traffic, optimizing the feature set to enhance detection performance in a highly imbalanced dataset.  

**Dataset Suggestions**:  
- **Dataset**: NSL-KDD Intrusion Detection Dataset  
- **Link**: [NSL-KDD (Kaggle)](https://www.kaggle.com/datasets/hassan06/nslkdd)  

**Tasks**:  
- **Data Preprocessing**: Handle class imbalance using techniques like SMOTE or undersampling, and normalize features.  
- **Feature Engineering with Autofeat**: Generate new features that may better separate normal vs. anomalous traffic patterns.  
- **Model Training (Multiple Models)**:  
  - Baseline: Logistic Regression and Decision Trees.  
  - Tree-based: Random Forest and XGBoost.  
  - Anomaly detection: Isolation Forest and Autoencoder.  
- **Model Evaluation**: Compare models using precision, recall, F1-score, and ROC-AUC, both with and without Autofeat features.  

**Bonus Ideas (Optional)**:  
- Implement near real-time scoring of network packets.  
- Compare performance of classical ML vs anomaly detection approaches under Autofeat-enhanced features.  
