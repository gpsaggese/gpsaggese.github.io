# DVC

## Description
- DVC (Data Version Control) is an open-source tool designed for managing
  machine learning projects, enabling version control for data and models.
- It integrates seamlessly with Git, allowing data scientists to track changes
  in datasets and machine learning models alongside code.
- DVC provides a way to create reproducible workflows by managing data pipelines
  and facilitating collaboration among team members.
- It supports large datasets and model files, enabling efficient storage and
  retrieval while maintaining a lightweight project structure.
- DVC allows users to define and manage data processing pipelines, making it
  easy to automate and reproduce experiments.

## Project Objective
The goal of the project is to build a predictive model using a public dataset to
forecast future sales for a retail store. Students will optimize the model's
accuracy and interpretability, focusing on versioning their data and model
throughout the process.

## Dataset Suggestions
1. **Store Sales - Time Series Forecasting**
   - **Source**: Kaggle
   - **URL**:
     [Store Sales Dataset](https://www.kaggle.com/c/store-sales-time-series-forecasting/data)
   - **Data Contains**: Historical sales data for a retail store, including
     sales amounts, item details, and store information.
   - **Access Requirements**: Free Kaggle account to download the dataset.

2. **Retail Product Sales**
   - **Source**: Kaggle
   - **URL**:
     [Retail Product Sales Dataset](https://www.kaggle.com/datasets/irfanasrullah/retail-product-sales)
   - **Data Contains**: Transactional data from a retail store, including
     product IDs, sales amounts, and customer demographics.
   - **Access Requirements**: Free Kaggle account to download the dataset.

3. **Monthly Milk Production**
   - **Source**: UCI Machine Learning Repository
   - **URL**:
     [Monthly Milk Production Dataset](https://archive.ics.uci.edu/ml/datasets/monthly+milk+production+in+bulgaria)
   - **Data Contains**: Monthly milk production quantities over several years,
     suitable for time series forecasting.
   - **Access Requirements**: Publicly available without authentication.

## Tasks
- **Set Up DVC**: Initialize a DVC project within a Git repository, setting up
  the necessary configurations and file structure.
- **Data Ingestion**: Use DVC to track and version the chosen dataset, ensuring
  that all changes to the data are recorded.
- **Data Preprocessing**: Implement data cleaning and preprocessing steps,
  documenting these processes in a DVC pipeline.
- **Model Development**: Develop a machine learning model to predict sales using
  the processed data, applying techniques such as regression or time series
  analysis.
- **Model Evaluation**: Evaluate the model's performance using appropriate
  metrics (e.g., RMSE, MAE) and visualize the results.
- **Experiment Tracking**: Use DVC to track experiments, comparing different
  model versions and configurations to identify the best-performing model.

## Bonus Ideas
- **Hyperparameter Tuning**: Implement hyperparameter optimization techniques
  (e.g., Grid Search or Random Search) and track these experiments with DVC.
- **Data Drift Detection**: Investigate methods for detecting data drift in the
  sales data over time and adjust the model accordingly.
- **Deployment Simulation**: Create a mock deployment pipeline using DVC and
  GitHub Actions to automate model updates when new data is available.

## Useful Resources
- [DVC Official Documentation](https://dvc.org/doc)
- [DVC GitHub Repository](https://github.com/iterative/dvc)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Machine Learning Mastery - Time Series Forecasting](https://machinelearningmastery.com/time-series-forecasting/)
