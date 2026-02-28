# FLAML

## Description
- FLAML (Fast and Lightweight AutoML) is an open-source library designed for
  automating machine learning tasks with a focus on efficiency and ease of use.
- It provides a user-friendly interface for hyperparameter tuning, allowing
  users to optimize machine learning models without deep expertise in the
  underlying algorithms.
- The tool supports various machine learning tasks, including classification,
  regression, and time-series forecasting, making it versatile for different
  project needs.
- FLAML is built to be lightweight, enabling quick experimentation and model
  selection without requiring extensive computational resources.
- It integrates seamlessly with popular libraries such as Scikit-learn and
  XGBoost, allowing users to leverage existing models and frameworks easily.

## Project Objective
The goal of this project is to build a machine learning model that predicts
housing prices based on various features such as location, size, and amenities.
Students will optimize the model for accuracy and interpretability, focusing on
hyperparameter tuning using FLAML.

## Dataset Suggestions
1. **Kaggle Housing Prices Dataset**
   - **Source**: Kaggle
   - **URL**:
     [Housing Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   - **Data Contains**: Features related to residential properties in Ames,
     Iowa, including sale price, area, number of rooms, and more.
   - **Access Requirements**: Free account on Kaggle to download the dataset.

2. **California Housing Prices Dataset**
   - **Source**: California Department of Housing and Community Development
   - **URL**:
     [California Housing](https://www.kaggle.com/c/california-housing-prices)
   - **Data Contains**: Information on housing prices in California, including
     geographical, economic, and demographic features.
   - **Access Requirements**: Free account on Kaggle to access the dataset.

3. **Boston Housing Dataset**
   - **Source**: UCI Machine Learning Repository
   - **URL**: [Boston Housing](https://archive.ics.uci.edu/ml/datasets/Housing)
   - **Data Contains**: Features related to housing in Boston, including crime
     rate, number of rooms, and distance to employment centers.
   - **Access Requirements**: No authentication required; can be downloaded
     directly.

4. **Ames Housing Dataset**
   - **Source**: Kaggle
   - **URL**:
     [Ames Housing](https://www.kaggle.com/datasets/prestonvong/ames-housing-data)
   - **Data Contains**: A detailed dataset of residential properties in Ames,
     Iowa, with over 70 features describing the properties.
   - **Access Requirements**: Free account on Kaggle to download the dataset.

## Tasks
- **Data Exploration**: Load and explore the dataset to understand the features
  and their distributions.
- **Data Preprocessing**: Clean and preprocess the data, handling missing values
  and categorical variables appropriately.
- **Model Selection**: Utilize FLAML to automatically select and tune a
  regression model based on the training data.
- **Model Evaluation**: Evaluate the model's performance using metrics such as
  Mean Absolute Error (MAE) and R-squared, and visualize the results.
- **Interpretation**: Analyze the importance of different features in the model
  and present findings in a report.

## Bonus Ideas
- **Feature Engineering**: Experiment with creating new features based on
  existing data to improve model performance.
- **Comparison with Other Models**: Compare the performance of the FLAML-tuned
  model with other baseline models (e.g., linear regression, decision trees).
- **Hyperparameter Tuning**: Explore different tuning strategies within FLAML,
  such as adjusting the search space or optimization strategies.
- **Deployment**: Create a simple web application to showcase the model
  predictions using Streamlit or Flask.

## Useful Resources
- [FLAML Documentation](https://github.com/microsoft/FLAML)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
