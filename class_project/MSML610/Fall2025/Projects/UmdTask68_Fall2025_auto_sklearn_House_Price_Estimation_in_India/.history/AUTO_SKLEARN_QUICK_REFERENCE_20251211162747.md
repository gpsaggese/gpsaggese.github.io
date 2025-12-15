# Auto-sklearn Quick Reference Guide

## 🚀 Essential Methods Used in This Project

### 1. Configuration & Training

#### Create & Configure

```python
from autosklearn.regression import AutoSklearnRegressor

automl = AutoSklearnRegressor(
    time_left_for_this_task=1800,      # Total time budget (seconds)
    per_run_time_limit=120,             # Max time per model (seconds)
    n_jobs=1,                           # Parallel jobs (-1 = all cores)
    seed=42,                            # Reproducibility
    memory_limit=10240,                 # RAM limit (MB)

    # Advanced ensemble settings
    ensemble_size=50,                   # Number of models in final ensemble
    ensemble_nbest=200,                 # Pool of models to build ensemble from

    # Validation strategy
    resampling_strategy='cv',           # Use cross-validation
    resampling_strategy_arguments={'folds': 5},  # 5-fold CV

    # Algorithm selection
    include_estimators=[                # Which algorithms to try
        'random_forest',
        'gradient_boosting',
        'extra_trees',
        'adaboost',
        'decision_tree',
        'k_nearest_neighbors',
        'libsvm_svr'
    ],
)
```

#### Train Model

```python
automl.fit(X_train, y_train)
```

---

### 2. Model Inspection

#### Training Summary

```python
# Print comprehensive training statistics
print(automl.sprint_statistics())
```

**Output includes**:

- Metric optimized
- Best validation score
- Number of runs (total, successful, crashed, timeout, memory exceeded)

#### Ensemble Leaderboard

```python
# Show all models ranked by performance
leaderboard_df = automl.leaderboard()
print(leaderboard_df)
```

**Shows**: rank, ensemble_weight, type, cost, duration for each model

#### Show Models in Ensemble

```python
# Display which models are in final ensemble
print(automl.show_models())
```

**Shows**: Complete pipeline for each ensemble member

#### Get Model Weights

```python
# Get models with their weights
models_with_weights = automl.get_models_with_weights()

for weight, model in models_with_weights:
    print(f"Weight: {weight:.4f}")
    print(f"Model: {model}")
```

**Returns**: List of (weight, model) tuples

---

### 3. Cross-Validation Results

#### Access CV Results

```python
# Get detailed CV results for all models tried
cv_results = automl.cv_results_

# Convert to DataFrame for analysis
import pandas as pd
cv_df = pd.DataFrame(cv_results)
```

**Key columns in cv_results**:

- `mean_test_score`: Average performance across folds
- `std_test_score`: Standard deviation of performance
- `rank_test_scores`: Ranking of models
- `param_*`: Hyperparameters for each model
- `status`: Success/Failed/Timeout/etc
- `mean_fit_time`: Training time

#### Analyze CV Results

```python
# Number of models tried
total_models = len(cv_results)

# Best performing models
top_5 = cv_df.nsmallest(5, 'rank_test_scores')

# Performance statistics
best_score = cv_df['mean_test_score'].max()
mean_score = cv_df['mean_test_score'].mean()
std_score = cv_df['mean_test_score'].std()

# Model types tried
if 'param_classifier:__choice__' in cv_df.columns:
    model_counts = cv_df['param_classifier:__choice__'].value_counts()
```

---

### 4. Predictions

#### Make Predictions

```python
# Predict on test data
y_pred = automl.predict(X_test)
```

#### Get Probability Estimates (Classification only)

```python
# For classification tasks
y_proba = automl.predict_proba(X_test)
```

#### Score Model

```python
# Calculate R² score (or accuracy for classification)
score = automl.score(X_test, y_test)
```

---

### 5. Feature Importance

#### Extract from Ensemble

```python
import numpy as np

# Get models with weights
models = automl.get_models_with_weights()

# Aggregate feature importances
feature_importances_list = []
weights_list = []

for weight, model in models:
    if hasattr(model, 'feature_importances_'):
        feature_importances_list.append(model.feature_importances_)
        weights_list.append(weight)

# Weighted average
if feature_importances_list:
    weighted_importance = np.average(
        feature_importances_list,
        axis=0,
        weights=weights_list
    )
```

---

### 6. Model Persistence

#### Save Model

```python
import joblib

# Save complete ensemble
joblib.dump(automl, 'autosklearn_model.pkl')

# Get file size
import os
size_mb = os.path.getsize('autosklearn_model.pkl') / (1024 * 1024)
```

#### Load Model

```python
# Load saved model
loaded_automl = joblib.load('autosklearn_model.pkl')

# Verify it works
predictions = loaded_automl.predict(X_test)
```

#### Save Metadata

```python
import json

metadata = {
    'training_time': 1800,
    'n_models_tried': len(automl.cv_results_),
    'ensemble_size': len(automl.get_models_with_weights()),
    'best_validation_score': automl.cv_results_['mean_test_score'].max(),
    'features_shape': X_train.shape
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

---

### 7. Advanced Features (Not in Current Notebook)

#### Warm Start (Continue Training)

```python
# Extend training with more time
automl.refit(X_train, y_train)
```

#### Custom Metrics

```python
from autosklearn.metrics import make_scorer

def custom_metric(y_true, y_pred):
    # Your custom scoring logic
    return score

my_metric = make_scorer(
    'custom_metric',
    custom_metric,
    optimum=1.0,
    greater_is_better=True,
    needs_proba=False,
    needs_threshold=False
)

automl = AutoSklearnRegressor(metric=my_metric)
```

#### Specify Feature Types

```python
# Mark categorical vs numerical features
feat_type = [
    'Categorical',  # Feature 0
    'Numerical',    # Feature 1
    'Numerical',    # Feature 2
    'Categorical',  # Feature 3
    # ... etc
]

automl = AutoSklearnRegressor(feat_type=feat_type)
```

#### Control Preprocessors

```python
automl = AutoSklearnRegressor(
    include_preprocessors=['no_preprocessing', 'pca', 'kernel_pca'],
    exclude_preprocessors=['ica']
)
```

---

## 🔍 Common Use Cases

### 1. Quick Experiment (5 minutes)

```python
automl = AutoSklearnRegressor(
    time_left_for_this_task=300,
    per_run_time_limit=30,
    ensemble_size=10
)
```

### 2. Production Model (1 hour)

```python
automl = AutoSklearnRegressor(
    time_left_for_this_task=3600,
    per_run_time_limit=180,
    ensemble_size=100,
    ensemble_nbest=500,
    resampling_strategy='cv',
    resampling_strategy_arguments={'folds': 10}
)
```

### 3. Limited Resources

```python
automl = AutoSklearnRegressor(
    time_left_for_this_task=600,
    per_run_time_limit=60,
    memory_limit=3072,  # 3GB
    n_jobs=1,
    ensemble_size=20
)
```

---

## 📊 Typical Workflow

```python
# 1. Import and prepare data
from autosklearn.regression import AutoSklearnRegressor
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 2. Configure
automl = AutoSklearnRegressor(
    time_left_for_this_task=1800,
    per_run_time_limit=120,
    ensemble_size=50
)

# 3. Train
automl.fit(X_train, y_train)

# 4. Inspect
print(automl.sprint_statistics())
print(automl.leaderboard())

# 5. Evaluate
y_pred = automl.predict(X_test)
score = automl.score(X_test, y_test)

# 6. Save
import joblib
joblib.dump(automl, 'model.pkl')
```

---

## 🎯 Performance Tips

1. **Increase ensemble diversity**: Higher `ensemble_size` (50-100)
2. **More candidates**: Higher `ensemble_nbest` (200-500)
3. **Better validation**: Use `resampling_strategy='cv'` with 5-10 folds
4. **More time per model**: Increase `per_run_time_limit` for complex datasets
5. **Parallel processing**: Set `n_jobs=-1` to use all CPU cores
6. **Memory management**: Adjust `memory_limit` based on system RAM

---

## ⚠️ Common Pitfalls

1. **Too little time**: Set `time_left_for_this_task` >= 10 \* `per_run_time_limit`
2. **Memory issues**: Reduce `memory_limit` if crashes occur
3. **No successful runs**: Check data preprocessing and increase time limits
4. **Poor performance**: Try more algorithms, increase search time
5. **Slow predictions**: Large ensembles are slower; consider reducing `ensemble_size`

---

## 📚 Where to Find Each Method

| Method                       | Section in Notebook | Cell # |
| ---------------------------- | ------------------- | ------ |
| Configuration                | Section 2           | 29     |
| `.fit()`                     | Section 2           | 29     |
| `.sprint_statistics()`       | Section 2           | 29     |
| `.leaderboard()`             | Section 2.5         | 31     |
| `.show_models()`             | Section 2.5         | 31     |
| `.get_models_with_weights()` | Section 2.5         | 31, 33 |
| `.cv_results_`               | Section 2.5         | 32     |
| `.predict()`                 | Section 4           | 37     |
| `.score()`                   | Section 7           | 46     |
| `joblib.dump()`              | Section 7           | 45     |
| `joblib.load()`              | Section 7           | 46     |

---

## 🔗 Official Documentation

- **Main Docs**: https://automl.github.io/auto-sklearn/master/
- **API Reference**: https://automl.github.io/auto-sklearn/master/api.html
- **Examples**: https://automl.github.io/auto-sklearn/master/examples/
- **GitHub**: https://github.com/automl/auto-sklearn

---

**Last Updated**: December 11, 2025
