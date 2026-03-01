# Synthetic Data Vault

## Description
- **Synthetic Data Generation**: The Synthetic Data Vault (SDV) is a library
  designed to generate synthetic data that mimics the statistical properties of
  real datasets, allowing for privacy-preserving data analysis.
- **Multiple Data Types**: It supports various data types including tabular,
  time series, and relational data, making it versatile for different
  applications.
- **Modeling Techniques**: SDV employs advanced modeling techniques such as GANs
  (Generative Adversarial Networks) and Bayesian networks to generate
  high-quality synthetic data.
- **Data Privacy**: By generating synthetic data, SDV helps organizations comply
  with data privacy regulations, as the synthetic datasets do not contain any
  real personal information.
- **Evaluation Metrics**: The library includes tools for evaluating the quality
  of the generated synthetic data against the original dataset, ensuring that
  the synthetic data is statistically similar.
- **Easy Integration**: SDV can easily integrate with popular data science
  libraries like Pandas, NumPy, and Scikit-learn, making it user-friendly for
  data scientists.

## Project Objective
The goal of the project is to generate synthetic datasets based on a real-world
dataset using the Synthetic Data Vault, and then use these synthetic datasets to
train a machine learning model for a classification task. The project will focus
on optimizing the accuracy of the classifier while ensuring that the synthetic
data preserves the characteristics of the original dataset.

## Dataset Suggestions
1. **Adult Income Dataset**
   - **Source**: UCI Machine Learning Repository
   - **URL**:
     [Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
   - **Data Contains**: Demographic information such as age, education,
     occupation, and income level (binary classification).
   - **Access Requirements**: No authentication required; freely available.

2. **Credit Card Fraud Detection**
   - **Source**: Kaggle
   - **URL**:
     [Credit Card Fraud Detection](https://www.kaggle.com/datasets/dalpozz/creditcard-fraud)
   - **Data Contains**: Transactions labeled as fraudulent or legitimate,
     including transaction amounts and time.
   - **Access Requirements**: Free account on Kaggle required to download the
     dataset.

3. **Heart Disease UCI**
   - **Source**: UCI Machine Learning Repository
   - **URL**:
     [Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
   - **Data Contains**: Medical attributes related to heart disease diagnosis
     (binary classification).
   - **Access Requirements**: No authentication required; freely available.

4. **Wine Quality Dataset**
   - **Source**: UCI Machine Learning Repository
   - **URL**:
     [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
   - **Data Contains**: Chemical properties of wine samples and their quality
     ratings (regression task).
   - **Access Requirements**: No authentication required; freely available.

## Tasks
- **Data Exploration**: Analyze the chosen dataset to understand its features,
  distributions, and potential correlations.
- **Synthetic Data Generation**: Use the SDV to generate synthetic data based on
  the original dataset, ensuring it captures the same statistical properties.
- **Model Training**: Train a classification model (e.g., logistic regression,
  decision tree) using both the original and synthetic datasets.
- **Model Evaluation**: Evaluate the performance of the models using metrics
  such as accuracy, precision, recall, and F1-score.
- **Comparison Analysis**: Compare the results of the models trained on
  synthetic data versus those trained on real data to assess the effectiveness
  of synthetic data generation.

## Bonus Ideas
- **Feature Engineering**: Experiment with different feature engineering
  techniques to improve model performance on synthetic data.
- **Generative Model Tuning**: Fine-tune the parameters of the SDV models to see
  how they affect the quality of the synthetic data.
- **Anomaly Detection**: Extend the project by implementing an anomaly detection
  model on the synthetic dataset to identify unusual patterns.

## Useful Resources
- [Synthetic Data Vault Documentation](https://sdv.dev/)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Towards Data Science: Generating Synthetic Data with SDV](https://towardsdatascience.com/generating-synthetic-data-with-sdv-1c8d0f7f8e5b)
