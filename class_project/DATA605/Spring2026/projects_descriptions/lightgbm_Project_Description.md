# LightGBM

## Description
- LightGBM (Light Gradient Boosting Machine) is an open-source, high-performance
  gradient boosting framework designed for speed and efficiency.
- It handles large datasets with ease and is optimized for distributed and
  parallel computing, making it suitable for big data applications.
- The tool supports various machine learning tasks, including classification,
  regression, and ranking, and is particularly known for its speed compared to
  other boosting algorithms.
- It employs a unique histogram-based learning algorithm that reduces memory
  usage and increases training speed significantly.
- LightGBM is highly customizable, offering numerous hyperparameters for
  fine-tuning model performance and enabling advanced techniques like
  categorical feature handling.

## Project Objective
The goal of this project is to predict house prices based on various features
using a regression model built with LightGBM. The project will focus on
optimizing the model's performance to achieve the lowest possible mean absolute
error (MAE) on a validation dataset.

## Dataset Suggestions
1. **Kaggle House Prices: Advanced Regression Techniques**
   - **Source**: Kaggle
   - **URL**:
     [House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   - **Data Contains**: Features related to house attributes (e.g., size,
     location, number of rooms) and corresponding sale prices.
   - **Access Requirements**: Free account on Kaggle to download the dataset.

2. **California Housing Prices**
   - **Source**: Scikit-learn Datasets
   - **URL**:
     [California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
   - **Data Contains**: Information on housing prices in California, including
     features such as median income, house age, and average rooms.
   - **Access Requirements**: Directly accessible via Scikit-learn, no
     additional authentication needed.

3. **Ames Housing Dataset**
   - **Source**: Kaggle
   - **URL**:
     [Ames Housing Dataset](https://www.kaggle.com/datasets/prestonvong/ames-housing-data)
   - **Data Contains**: Detailed attributes related to residential properties in
     Ames, Iowa, including sale prices and numerous categorical and numerical
     features.
   - **Access Requirements**: Free account on Kaggle to download the dataset.

## Tasks
- **Data Preprocessing**: Clean and preprocess the dataset by handling missing
  values, encoding categorical variables, and scaling numerical features.
- **Exploratory Data Analysis (EDA)**: Perform EDA to visualize relationships
  between features and house prices, identifying important predictors.
- **Model Training**: Implement LightGBM to train a regression model on the
  training dataset, tuning hyperparameters for optimal performance.
- **Model Evaluation**: Evaluate the model using metrics such as MAE and R² on a
  validation set, comparing performance with a baseline model (e.g., Linear
  Regression).
- **Feature Importance Analysis**: Analyze feature importance to understand
  which attributes contribute most to the model's predictions.

## Bonus Ideas
- **Hyperparameter Tuning**: Explore advanced hyperparameter tuning techniques
  using libraries like Optuna or GridSearchCV to further improve model
  performance.
- **Ensemble Methods**: Combine LightGBM with other models (e.g., Random Forest,
  XGBoost) to create an ensemble model and assess if it enhances prediction
  accuracy.
- **Deployment**: Create a simple web application using Flask or Streamlit to
  showcase the model's predictions based on user input for house features.

## Useful Resources
- [LightGBM Official Documentation](https://lightgbm.readthedocs.io/en/latest/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Kaggle Tutorial on LightGBM](https://www.kaggle.com/code/abhishek/lightgbm-tutorial)
- [Feature Importance in LightGBM](https://lightgbm.readthedocs.io/en/latest/Advanced-Features.html#feature-importance)
- [Optuna Documentation for Hyperparameter Optimization](https://optuna.readthedocs.io/en/stable/)
