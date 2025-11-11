<!-- toc -->

- [XGBoost API Tutorial](#xgboost-api-tutorial)
  * [Overview](#overview)
  * [Problem Statement](#problem-statement)
  * [Alternatives and Comparisons](#alternatives-and-comparisons)
    + [scikit-learn Random Forest](#scikit-learn-random-forest)
    + [LightGBM](#lightgbm)
    + [CatBoost](#catboost)
    + [Recommendation](#recommendation)
  * [Native XGBoost API Overview](#native-xgboost-api-overview)
    + [Key Features](#key-features)
    + [Challenges with the Native API](#challenges-with-the-native-api)
  * [Our Integration Layer](#our-integration-layer)
    + [Goals](#goals)
  * [Conclusion](#conclusion)

<!-- tocstop -->

# XGBoost API Tutorial

## Overview

This document describes the usage of [XGBoost](https://xgboost.readthedocs.io/) (Extreme Gradient Boosting), a powerful gradient boosting framework for building classification and regression models. XGBoost is particularly effective for structured/tabular data and excels in scenarios with class imbalance, making it ideal for predicting flight delays. Our integration focuses on using XGBoost's Python API to build a classification model that predicts whether a flight will be delayed based on historical flight and weather data.

XGBoost provides several advantages:
1. **High Performance**: Optimized gradient boosting algorithm with parallel processing capabilities.
2. **Handles Missing Values**: Built-in handling of missing values in features.
3. **Feature Importance**: Provides insights into which features contribute most to predictions.
4. **Regularization**: Built-in L1 and L2 regularization to prevent overfitting.
5. **Class Imbalance**: Supports scale_pos_weight parameter for handling imbalanced datasets.

## Problem Statement

Predicting flight delays is a complex classification problem that involves:
- **Multiple Features**: Flight schedules, airline information, airport data, weather conditions, and temporal features.
- **Class Imbalance**: Delayed flights are typically less frequent than on-time flights, requiring special handling.
- **Non-linear Relationships**: Complex interactions between features (e.g., weather conditions affecting different airports differently).
- **Missing Data**: Incomplete weather or flight records that need robust handling.

XGBoost addresses these challenges by:
- Automatically handling missing values during training and prediction.
- Providing hyperparameters to balance classes (e.g., `scale_pos_weight`).
- Capturing non-linear relationships through gradient boosting and tree-based splits.
- Offering built-in cross-validation and early stopping to prevent overfitting.

## Alternatives and Comparisons

### scikit-learn Random Forest

- **Advantages**:
  - Simple API and well-documented.
  - Good baseline for ensemble methods.
  - No hyperparameter tuning required for basic usage.
- **Limitations**:
  - Less efficient than XGBoost for large datasets.
  - Limited handling of missing values.
  - No built-in support for class imbalance without additional preprocessing.

### LightGBM

- **Advantages**:
  - Faster training than XGBoost on large datasets.
  - Lower memory usage.
  - Good performance on categorical features.
- **Limitations**:
  - Less mature ecosystem compared to XGBoost.
  - May require more tuning for optimal results.
  - Slightly different API structure.

### CatBoost

- **Advantages**:
  - Excellent handling of categorical features without preprocessing.
  - Good default parameters, requires less tuning.
  - Robust to overfitting.
- **Limitations**:
  - Slower training time compared to XGBoost and LightGBM.
  - Less flexible than XGBoost for custom objectives.
  - Larger model size.

### Recommendation

For flight delay prediction, **XGBoost** is the best choice because:
- Provides excellent performance on tabular data with mixed feature types.
- Built-in support for missing values and class imbalance.
- Extensive documentation and community support.
- Feature importance visualization helps understand delay factors.
- Well-suited for the precision, recall, F1-score, and ROC-AUC evaluation metrics required for this project.

## Native XGBoost API Overview

The XGBoost Python API (`xgboost`) provides a scikit-learn compatible interface:

- **XGBClassifier**: Main class for classification tasks.
- **Key Parameters**:
  - `n_estimators`: Number of boosting rounds (default: 100).
  - `max_depth`: Maximum tree depth (default: 6).
  - `learning_rate`: Step size shrinkage (default: 0.3).
  - `scale_pos_weight`: Controls balance of positive and negative weights (important for class imbalance).
  - `subsample`: Fraction of samples used for training (default: 1.0).
  - `colsample_bytree`: Fraction of features used for each tree (default: 1.0).
  - `objective`: Loss function (default: 'binary:logistic' for binary classification).
  - `eval_metric`: Evaluation metric (e.g., 'auc', 'logloss', 'error').
  - `early_stopping_rounds`: Stops training if no improvement (requires validation set).

- **Methods**:
  - `fit(X, y)`: Train the model on training data.
  - `predict(X)`: Predict class labels.
  - `predict_proba(X)`: Predict class probabilities.
  - `get_booster().get_score()`: Get feature importance scores.

### Key Features

1. **Handling Missing Values**: XGBoost can handle missing values in features automatically by learning the best direction to go when a value is missing.
2. **Early Stopping**: Prevents overfitting by stopping training when validation performance doesn't improve.
3. **Feature Importance**: Provides multiple importance types (weight, gain, cover) to understand feature contributions.
4. **Cross-Validation**: Built-in support for k-fold cross-validation with `cv()` function.

### Challenges with the Native API

- **Hyperparameter Tuning**: Requires careful tuning of multiple parameters (learning_rate, max_depth, n_estimators, etc.) for optimal performance.
- **Class Imbalance**: Need to manually set `scale_pos_weight` based on class distribution.
- **Data Preprocessing**: Still requires feature engineering, encoding categorical variables, and handling outliers.
- **Evaluation Metrics**: Need to implement custom evaluation functions for precision, recall, F1-score, and ROC-AUC.

## Our Integration Layer

### Goals

1. **Simplified Model Setup**: High-level functions to initialize XGBoost with sensible defaults for flight delay prediction.
2. **Automatic Class Balancing**: Automatically calculate `scale_pos_weight` from training data class distribution.
3. **Comprehensive Evaluation**: Built-in functions to compute precision, recall, F1-score, and ROC-AUC metrics.
4. **Feature Importance Visualization**: Functions to extract and visualize feature importance scores.
5. **Data Preprocessing Integration**: Seamless integration with data preprocessing and feature engineering pipelines.

Our integration layer (`AirlineDelay_utils.py`) provides:
- `train_xgboost_model()`: Train XGBoost classifier with automatic hyperparameter setup.
- `evaluate_model()`: Compute comprehensive evaluation metrics including precision, recall, F1-score, and ROC-AUC.
- `plot_feature_importance()`: Visualize feature importance scores to understand delay factors.
- `predict_delays()`: Make predictions on new flight data with proper preprocessing.

## Conclusion

XGBoost is an excellent choice for flight delay prediction due to its robust handling of missing values, class imbalance, and non-linear relationships. Our integration layer simplifies the usage of XGBoost by providing high-level functions for model training, evaluation, and visualization, making it easier to build and deploy delay prediction models. The API's flexibility and performance make it well-suited for this classification task, where understanding feature importance (weather vs. airline vs. airport factors) is crucial for actionable insights.
