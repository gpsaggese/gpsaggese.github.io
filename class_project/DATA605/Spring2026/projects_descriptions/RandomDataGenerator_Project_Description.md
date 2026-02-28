# RandomDataGenerator

## Description
- **Data Generation**: RandomDataGenerator is a Python library that allows users
  to generate synthetic data for various applications, including testing and
  training machine learning models.
- **Customizable Output**: Users can specify data types, distributions, and
  formats, making it highly versatile for different project needs.
- **Variety of Data Types**: Supports generation of numerous data types,
  including integers, floats, strings, dates, and even complex nested
  structures.
- **Integration**: Easily integrates with popular data manipulation libraries
  like Pandas, making it straightforward to use in data science workflows.
- **Reproducibility**: Enables reproducible data generation through seed
  settings, allowing users to create consistent datasets for testing and
  validation.

## Project Objective
The goal of this project is to create a synthetic dataset that mimics a
real-world scenario, allowing students to apply machine learning techniques to
predict customer churn in a subscription-based service. The project will focus
on optimizing the accuracy of the churn prediction model using various
classification algorithms.

## Dataset Suggestions
1. **Kaggle - Customer Churn Dataset**
   - **Source**: Kaggle
   - **URL**:
     [Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
   - **Data Contains**: Information about customers, including demographics,
     account information, and whether they have churned.
   - **Access Requirements**: Free account on Kaggle to download the dataset.

2. **Open Government Data - Customer Complaints**
   - **Source**: Consumer Financial Protection Bureau
   - **URL**:
     [Consumer Complaints](https://www.consumerfinance.gov/data-research/consumer-complaints/)
   - **Data Contains**: Complaints data that can be used to simulate customer
     behavior and churn.
   - **Access Requirements**: No authentication required.

3. **Hugging Face Datasets - Bank Marketing**
   - **Source**: Hugging Face
   - **URL**:
     [Bank Marketing Dataset](https://huggingface.co/datasets/uciml/creditcard-fraud)
   - **Data Contains**: Information on marketing campaigns and customer
     responses that can be adapted for churn prediction.
   - **Access Requirements**: Free access with no authentication.

4. **GitHub - Synthetic Data Generation**
   - **Source**: GitHub Repository
   - **URL**:
     [Synthetic Data Generation](https://github.com/awslabs/synthetic-data-generator)
   - **Data Contains**: Code and examples for generating synthetic customer
     data.
   - **Access Requirements**: Open-source; no authentication required.

## Tasks
- **Data Generation**: Use RandomDataGenerator to create synthetic customer data
  that includes features relevant to churn prediction (e.g., age, account
  length, service usage).
- **Data Preprocessing**: Clean and preprocess the generated data, handling any
  inconsistencies and preparing it for model training.
- **Model Selection**: Choose appropriate classification algorithms (e.g.,
  Logistic Regression, Decision Trees, Random Forest) for predicting customer
  churn.
- **Model Training**: Train the selected models on the synthetic dataset and
  optimize hyperparameters for better performance.
- **Model Evaluation**: Evaluate the models using metrics such as accuracy,
  precision, recall, and F1-score, and compare their performance.
- **Reporting**: Prepare a report summarizing the findings, including model
  performance and insights gained from the synthetic data.

## Bonus Ideas
- **Feature Engineering**: Explore additional feature engineering techniques to
  enhance model performance, such as creating interaction terms or aggregating
  features.
- **Baseline Comparison**: Compare the synthetic data model's performance
  against a model trained on a real dataset to assess the validity of synthetic
  data.
- **Anomaly Detection**: Implement anomaly detection techniques to identify
  unusual patterns in churn behavior within the synthetic dataset.

## Useful Resources
- [RandomDataGenerator Documentation](https://pypi.org/project/RandomDataGenerator/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Hugging Face Datasets Documentation](https://huggingface.co/docs/datasets/index)
- [Consumer Financial Protection Bureau - Data Resources](https://www.consumerfinance.gov/data-research/)
- [GitHub - Synthetic Data Generation](https://github.com/awslabs/synthetic-data-generator)
