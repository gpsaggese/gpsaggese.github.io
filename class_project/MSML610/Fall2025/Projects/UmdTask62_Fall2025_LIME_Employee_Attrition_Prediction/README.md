# UMD TASK 62 - Employee Attrition Prediction Using LIME

## Project Overview

### 1. Objective:
The goal of this project is to develop a machine learning model that predicts employee attrition in organizations, with the additional feature of using LIME (Local Interpretable Model-Agnostic Explanations) to provide insights into why employees are likely to leave.

---

## Files and Dataset

### 2. Dataset:

#### 2.1 Dataset Description:
We use the **WA_Fn-UseC_-HR-Employee-Attrition.csv** dataset, which contains various employee attributes such as demographics, job roles, and performance metrics, as well as information on whether the employee left the company (attrition).

#### 2.2 Where to Download:
You can download the dataset from Kaggle. To access it:

1. Go to [Kaggle’s Employee Attrition and Performance dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset).
2. Sign in with your Kaggle account.
3. Search for the **WA_Fn-UseC_-HR-Employee-Attrition.csv** dataset.
4. Download the dataset to use in your project.

#### 2.3 Where to Place the File (Host & Container Paths):
On your local machine (host), place the **WA_Fn-UseC_-HR-Employee-Attrition.csv** file under the `data/` directory.

class_project/MSML610/Fall2025/Projects/UmdTask62_Fall2025_LIME_Employee_Attrition_Prediction/
└── data/
    └── WA_Fn-UseC_-HR-Employee-Attrition.csv


### 3. Install dependencies:

You will need Python 3.8 and the following libraries:

pandas

numpy

scikit-learn

xgboost

lightgbm

lime

matplotlib

seaborn

---

### 4. Files Worked On:

- **Attrition.API.ipynb**: Main notebook for model development and evaluation.
- **Attrition.API.md**: Documentation for the API used in the model.
- **Attrition.example.ipynb**: Example notebook for demonstrating the model’s capabilities.
- **Attrition.example.md**: Explanation of the example provided.
- **Attrition_utils.py**: Utility functions for preprocessing and feature engineering.

---

### 5. Tasks

#### 1. Data Exploration (EDA):
- Perform exploratory data analysis (EDA) to understand attrition trends, correlations, and feature importance.
- Visualize key patterns like attrition rates, demographics, and job-related factors.

#### 2. Feature Engineering:
- Create new features based on employee demographics, work conditions, and tenure to improve model performance.

#### 3. Model Development:
- Use Gradient Boosting (e.g., **XGBoost** and **LightGBM**) to develop a predictive model for employee attrition.
- Split the data into training and testing datasets for model evaluation.

#### 4. LIME Explanations:
- Use the **LIME** framework to explain the model predictions.
- Highlight key factors (e.g., overtime, job role, department) that drive employee attrition predictions.

#### 5. Model Evaluation:
- Evaluate the model's performance using metrics like **accuracy**, **F1-score**, and **ROC-AUC**.
- Analyze how the features impact the model's predictions.

#### 6. Reporting:
- Summarize findings from EDA, feature engineering, model performance, and LIME explanations.
- Provide actionable recommendations for HR teams based on insights from the model.

---
