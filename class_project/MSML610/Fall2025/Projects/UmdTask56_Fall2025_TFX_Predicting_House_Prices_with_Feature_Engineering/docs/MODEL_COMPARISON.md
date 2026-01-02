# Model Comparison Guide

This guide explains how to use the model comparison framework for house price prediction.

## Overview

The model comparison framework allows you to:
- Compare multiple regression models (XGBoost, Random Forest, Gradient Boosting, Ridge, Lasso, ElasticNet)
- Evaluate ensemble methods (Voting Regressor, Stacking Regressor)
- Perform k-fold cross-validation for robust performance estimates
- Automatically select the best performing model
- Save trained models and comparison results

## Quick Start

### Basic Usage

Compare all available models with 5-fold cross-validation:

```bash
python scripts/compare_models.py
```

### Advanced Usage

```bash
# Compare specific models with 10-fold CV
python scripts/compare_models.py --cv-folds 10 --models XGBoost RandomForest VotingEnsemble

# Save the best model
python scripts/compare_models.py --save-best-model

# Specify output directory
python scripts/compare_models.py --output-dir models/comparison_results
```

## Available Models

### Individual Models

1. **XGBoost** - Gradient boosting with optimized hyperparameters
   - Best for: Tabular data, handling missing values
   - Hyperparameters: 1000 estimators, max_depth=7, learning_rate=0.01

2. **Random Forest** - Ensemble of decision trees
   - Best for: Robust predictions, feature importance
   - Hyperparameters: 500 estimators, max_depth=15

3. **Gradient Boosting** - Sequential ensemble method
   - Best for: High accuracy, interpretability
   - Hyperparameters: 1000 estimators, learning_rate=0.01

4. **Ridge Regression** - L2 regularized linear regression
   - Best for: Linear relationships, preventing overfitting
   - Hyperparameters: alpha=10.0

5. **Lasso Regression** - L1 regularized linear regression
   - Best for: Feature selection, sparse models
   - Hyperparameters: alpha=0.001

6. **ElasticNet** - Combined L1 and L2 regularization
   - Best for: Balance between Ridge and Lasso
   - Hyperparameters: alpha=0.001, l1_ratio=0.5

### Ensemble Methods

7. **Voting Ensemble** - Average predictions from multiple models
   - Base models: XGBoost, Random Forest, Gradient Boosting
   - Best for: Reducing variance, combining diverse models

8. **Stacking Ensemble** - Meta-learner combines base model predictions
   - Base models: XGBoost, Random Forest, Gradient Boosting, Ridge
   - Meta-learner: Ridge Regression
   - Best for: Maximum performance, leveraging model strengths

## Cross-Validation

Cross-validation provides more robust performance estimates by:
- Splitting data into k folds
- Training on k-1 folds, validating on 1 fold
- Repeating k times with different validation folds
- Averaging results across all folds

### Benefits

- **Robust estimates**: Less sensitive to data split
- **Variance estimation**: Standard deviation across folds
- **Better model selection**: More reliable comparison

### Configuration

```python
# Default: 5-fold CV
python scripts/compare_models.py --cv-folds 5

# More folds = more robust, but slower
python scripts/compare_models.py --cv-folds 10
```

## Output Files

The comparison script generates several output files:

### 1. comparison_results.json

Complete comparison results including:
- Individual model metrics (RMSE, MAE, R²)
- Cross-validation scores (mean, std, min, max)
- Model rankings
- Training times

Example structure:
```json
{
  "comparison": {
    "best_model": "XGBoost",
    "rankings": {
      "by_rmse": ["XGBoost", "RandomForest", "GradientBoosting", ...],
      "by_r2": ["XGBoost", "RandomForest", "GradientBoosting", ...]
    },
    "summary": {
      "best_rmse": 0.1234,
      "worst_rmse": 0.5678,
      "rmse_improvement": 0.4444
    }
  },
  "models": {
    "XGBoost": {
      "cv_mean_rmse": 0.1234,
      "cv_std_rmse": 0.0123,
      "train_rmse": 0.0987,
      "train_r2": 0.92,
      "training_time": 12.34
    }
  }
}
```

### 2. best_model_*.pkl (if --save-best-model is used)

Serialized best performing model that can be loaded and used for predictions:

```python
import pickle

# Load the model
with open('models/comparison/best_model_XGBoost.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict(X_test)
```

### 3. best_model_metadata.json (if --save-best-model is used)

Metadata about the best model:
```json
{
  "model_name": "XGBoost",
  "metrics": {
    "cv_mean_rmse": 0.1234,
    "cv_std_rmse": 0.0123,
    "train_r2": 0.92
  },
  "cv_folds": 5,
  "n_features": 220,
  "n_samples": 1460
}
```

## Programmatic Usage

You can also use the model comparison framework programmatically:

```python
from utils.model_comparison import compare_all_models, ModelRegistry
import numpy as np

# Prepare your data
X_train = np.array(...)  # Shape: (n_samples, n_features)
y_train = np.array(...)  # Shape: (n_samples,)

# Compare all models
results = compare_all_models(
    X_train=X_train,
    y_train=y_train,
    use_cv=True,
    cv_folds=5
)

# Get best model
best_model_name = results['comparison']['best_model']
print(f"Best model: {best_model_name}")

# Train best model on full data
models = ModelRegistry.get_all_models()
best_model = models[best_model_name]
best_model.fit(X_train, y_train)

# Make predictions
predictions = best_model.predict(X_test)
```

## Using Individual Models

```python
from utils.model_comparison import ModelRegistry

# Get specific model
xgb_model = ModelRegistry.get_xgboost_model()
rf_model = ModelRegistry.get_random_forest_model()

# Get ensemble
voting_ensemble = ModelRegistry.get_voting_ensemble()
stacking_ensemble = ModelRegistry.get_stacking_ensemble()

# Custom ensemble with specific models
custom_ensemble = ModelRegistry.get_voting_ensemble(
    base_models=[
        ('xgb', ModelRegistry.get_xgboost_model()),
        ('rf', ModelRegistry.get_random_forest_model())
    ]
)
```

## Performance Metrics

### RMSE (Root Mean Squared Error)
- **Primary metric** for model selection
- Penalizes large errors more than small errors
- In log space: RMSE ~0.12 is good
- In original scale: RMSE <$30,000 is target

### MAE (Mean Absolute Error)
- Average absolute difference between predictions and actual values
- More interpretable than RMSE
- Less sensitive to outliers

### R² (R-squared)
- Proportion of variance explained by the model
- Range: -∞ to 1.0
- Values >0.85 indicate good fit
- Values >0.90 indicate excellent fit

## Interpreting Results

### Log-Transformed Target

Since the target variable (SalePrice) is log-transformed:
- CV RMSE is in log space (e.g., 0.12)
- To convert to dollars: `exp(prediction) - 1`
- Error of 0.12 in log space ≈ 12% error in price

### Cross-Validation Scores

Example output:
```
XGBoost CV Results:
  CV RMSE: 0.1234 (+/- 0.0123)
  Train RMSE: 0.0987
  Train R²: 0.92
```

- **CV RMSE**: Average error across all folds
- **+/- value**: Standard deviation (consistency across folds)
- **Train RMSE**: Error on training data (lower = better fit)
- **Train R²**: Variance explained on training data

### Model Rankings

Models are ranked by:
1. **RMSE** (primary): Lower is better
2. **MAE**: Lower is better
3. **R²**: Higher is better

The best model minimizes RMSE while maintaining good R².

## Tips for Best Results

### 1. Data Preprocessing

Ensure your data is properly preprocessed:
- Handle missing values
- Encode categorical variables
- Scale numerical features (if needed)
- Apply log transformation to target (already configured)

### 2. Cross-Validation Folds

- **5 folds**: Good balance (default)
- **10 folds**: More robust, slower
- **3 folds**: Faster, less robust
- Use more folds for small datasets

### 3. Model Selection

For this dataset (1460 samples, 220+ features):
- **Tree-based models** (XGBoost, Random Forest) often perform best
- **Ensemble methods** may provide marginal improvements
- **Linear models** (Ridge, Lasso) are fast baselines

### 4. Ensemble Methods

- Use **Voting Ensemble** for diverse base models
- Use **Stacking Ensemble** for maximum performance
- Both are slower to train but often more robust

## Troubleshooting

### Memory Issues

If you encounter memory errors:
```bash
# Compare fewer models
python scripts/compare_models.py --models XGBoost RandomForest Ridge

# Reduce CV folds
python scripts/compare_models.py --cv-folds 3
```

### Slow Training

To speed up training:
- Reduce CV folds: `--cv-folds 3`
- Compare specific models instead of all
- Skip ensemble methods (they're slowest)

### Poor Performance

If all models have poor performance:
- Check data preprocessing (missing values, encoding)
- Verify target transformation is applied
- Inspect feature correlations
- Consider feature engineering

## Integration with TFX Pipeline

The model comparison framework can be integrated with the TFX pipeline:

1. **After Transform component**: Use transformed features
2. **Alternative to Trainer**: Compare models before deploying
3. **Model selection**: Choose best model for production

```python
# TODO: Future integration
# Load transformed data from TFX artifacts
python scripts/compare_models.py --use-transformed
```

## Next Steps

After running model comparison:

1. **Review results**: Check `comparison_results.json`
2. **Analyze best model**: Understand why it performed best
3. **Feature importance**: Examine which features matter most
4. **Hyperparameter tuning**: Fine-tune the best model
5. **Deploy**: Integrate best model into TFX pipeline

## References

- **XGBoost**: https://xgboost.readthedocs.io/
- **Scikit-learn Ensembles**: https://scikit-learn.org/stable/modules/ensemble.html
- **Cross-validation**: https://scikit-learn.org/stable/modules/cross_validation.html
- **Model evaluation**: https://scikit-learn.org/stable/modules/model_evaluation.html
