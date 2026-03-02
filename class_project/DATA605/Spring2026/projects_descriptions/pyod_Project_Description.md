# Pyod

## Description
- **pyod** (Python Outlier Detection) is a comprehensive library designed for
  detecting outliers in multivariate data.
- It supports a wide range of algorithms, including both classical statistical
  methods and modern machine learning techniques.
- The library is user-friendly, with consistent interfaces for various outlier
  detection algorithms, making it easy to switch between methods.
- It provides tools for model evaluation and visualization, allowing users to
  assess the performance of different outlier detection models effectively.
- Pyod is compatible with popular data manipulation libraries such as NumPy and
  pandas, facilitating seamless integration into data science workflows.

## Project Objective
The goal of this project is to identify and analyze outliers in a dataset
related to credit card transactions. Students will optimize their models to
effectively detect fraudulent transactions, aiming to minimize false positives
while maximizing true positives.

## Dataset Suggestions
1. **Credit Card Fraud Detection Dataset**
   - **Source**: Kaggle
   - **URL**:
     [Credit Card Fraud Detection](https://www.kaggle.com/datasets/dalpozz/creditcard-fraud)
   - **Data Contains**: Transactions made by credit cards, with features
     including time, amount, and PCA-transformed features for anonymity.
   - **Access Requirements**: Free access with a Kaggle account.

2. **Synthetic Financial Dataset**
   - **Source**: OpenML
   - **URL**:
     [OpenML - Credit Card Fraud Detection](https://www.openml.org/d/1464)
   - **Data Contains**: Synthetic financial transactions with labels indicating
     whether they are fraudulent.
   - **Access Requirements**: Free to use without authentication.

3. **Bank Marketing Dataset**
   - **Source**: UCI Machine Learning Repository
   - **URL**:
     [Bank Marketing](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
   - **Data Contains**: Information about bank marketing campaigns, including
     customer responses, which can be used to detect unusual patterns in
     customer behavior.
   - **Access Requirements**: Open access, no registration needed.

## Tasks
- **Data Loading and Preprocessing**: Load the dataset and perform necessary
  preprocessing steps, including handling missing values and scaling features.
- **Exploratory Data Analysis (EDA)**: Conduct EDA to visualize the distribution
  of transaction amounts and identify potential outliers.
- **Outlier Detection Model Selection**: Choose appropriate outlier detection
  algorithms from pyod, such as Isolation Forest, AutoEncoder, or KNN.
- **Model Training and Evaluation**: Train the selected models on the dataset
  and evaluate their performance using metrics such as precision, recall, and F1
  score.
- **Visualization of Results**: Use pyod's built-in visualization tools to plot
  the detected outliers and compare the performance of different models.

## Bonus Ideas
- **Model Comparison**: Implement a baseline model (e.g., simple statistical
  method) and compare its performance with pyod models.
- **Feature Engineering**: Experiment with creating new features from existing
  data to improve outlier detection accuracy.
- **Real-time Outlier Detection**: Simulate a real-time detection system using a
  streaming data approach with the trained model.
- **Hyperparameter Tuning**: Use techniques such as Grid Search or Random Search
  to optimize hyperparameters of the selected models.

## Useful Resources
- [pyod Documentation](https://pyod.readthedocs.io/en/latest/)
- [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/dalpozz/creditcard-fraud)
- [OpenML - Credit Card Fraud Detection](https://www.openml.org/d/1464)
- [UCI Machine Learning Repository - Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- [GitHub - pyod Examples](https://github.com/yzhao062/pyod/tree/master/examples)
