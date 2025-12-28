# Deploy Best Model with TFX

This guide explains how to automatically identify the best regression model through cross-validation and deploy it using the TFX pipeline.

## Overview

The integrated solution:
1. **Compares 8 regression models** using k-fold cross-validation
2. **Selects the best model** based on RMSE
3. **Wraps sklearn models** in TensorFlow SavedModel format
4. **Deploys via TFX** using the standard pipeline (Transform → Trainer → Evaluator → Pusher)

## Quick Start

### One-Command Deployment

```bash
# Run in Docker container
docker exec house-price-tfx python scripts/run_pipeline_with_best_model.py
```

This single command will:
- Compare all 8 models with 5-fold CV
- Select the best performing model
- Train it on the full dataset
- Deploy to `models/serving/`

## Available Models

The comparison evaluates 8 regression models:

**Tree-Based Models:**
- XGBoost (usually best for tabular data)
- Random Forest
- Gradient Boosting

**Linear Models:**
- Ridge Regression
- Lasso Regression
- ElasticNet

**Ensemble Methods:**
- Voting Ensemble (combines XGBoost, RF, GBM)
- Stacking Ensemble (meta-learner on base models)

## Advanced Usage

### Custom Cross-Validation

```bash
# Use 10-fold CV for more robust estimates
python scripts/run_pipeline_with_best_model.py --cv-folds 10

# Force new comparison even if results exist
python scripts/run_pipeline_with_best_model.py --force-comparison
```

### Use TensorFlow DNN Instead

```bash
# Override and use TensorFlow DNN
python scripts/run_pipeline_with_best_model.py --use-tensorflow-dnn
```

### Custom Pipeline Name

```bash
# Use custom pipeline name
python scripts/run_pipeline_with_best_model.py --pipeline-name my_custom_pipeline
```

## How It Works

### Step 1: Model Comparison

The script runs all models with cross-validation:

```python
from utils.model_comparison import compare_all_models

results = compare_all_models(
    X_train=X,
    y_train=y,
    use_cv=True,
    cv_folds=5
)

best_model = results['comparison']['best_model']
# Example: "XGBoost"
```

Results are saved to `models/comparison/comparison_results.json`.

### Step 2: TFX Pipeline with Sklearn Model

The pipeline uses a custom sklearn trainer that:

1. **Loads transformed data** from the Transform component
2. **Trains the sklearn model** (e.g., XGBoost)
3. **Wraps it in TensorFlow** for serving compatibility
4. **Exports SavedModel** for deployment

```python
# Custom trainer for sklearn models
trainer = Trainer(
    module_file='utils/sklearn_trainer.py',
    custom_config={'model_name': 'XGBoost'}
)
```

### Step 3: Model Wrapping

The `SklearnModelWrapper` class converts sklearn models to TensorFlow:

```python
class SklearnModelWrapper(tf.Module):
    def __init__(self, sklearn_model, transform_output):
        self.sklearn_model = sklearn_model
        self.transform_output = transform_output

    @tf.function
    def serve(self, serialized_examples):
        # Parse raw examples
        # Apply TFX transforms
        # Predict with sklearn model
        # Convert from log space to dollars
        return predictions
```

This allows sklearn models to:
- Accept raw CSV examples (same as TensorFlow models)
- Use TFX Transform for preprocessing
- Serve via TensorFlow Serving
- Output predictions in original scale (dollars)

## Output Files

### Comparison Results

**Location:** `models/comparison/comparison_results.json`

```json
{
  "comparison": {
    "best_model": "XGBoost",
    "rankings": {
      "by_rmse": ["XGBoost", "StackingEnsemble", "VotingEnsemble", ...],
      "by_r2": ["XGBoost", "StackingEnsemble", "VotingEnsemble", ...]
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

### Deployed Model

**Location:** `models/serving/<timestamp>/`

The deployed model contains:
- `saved_model.pb` - TensorFlow SavedModel
- `variables/` - Model weights
- `sklearn_model.pkl` - Original sklearn model (backup)

### Serving Signature

```bash
# Inspect the deployed model
saved_model_cli show --dir models/serving/<timestamp> --all
```

**Input:** Raw serialized tf.Example protos (DT_STRING)
**Output:** House price predictions in dollars (DT_FLOAT)

## Testing the Deployed Model

### Option 1: Using saved_model_cli

```bash
# Get the latest model directory
MODEL_DIR=$(ls -t models/serving/ | head -1)

# Show model signature
saved_model_cli show --dir models/serving/$MODEL_DIR --tag_set serve --signature_def serving_default
```

### Option 2: Programmatic Testing

```python
import tensorflow as tf
import numpy as np

# Load the model
model = tf.saved_model.load('models/serving/<timestamp>')

# Create test example
example = tf.train.Example(
    features=tf.train.Features(
        feature={
            'LotArea': tf.train.Feature(int64_list=tf.train.Int64List(value=[8450])),
            'OverallQual': tf.train.Feature(int64_list=tf.train.Int64List(value=[7])),
            # ... other features
        }
    )
)

# Serialize and predict
serialized = example.SerializeToString()
predictions = model.signatures['serving_default'](
    examples=tf.constant([serialized])
)

print(f"Predicted price: ${predictions['output_0'].numpy()[0][0]:,.0f}")
```

## Performance Comparison

### Expected Results (Ames Housing Dataset)

| Model | CV RMSE (log) | Training Time | Notes |
|-------|---------------|---------------|-------|
| XGBoost | 0.11-0.13 | ~15s | Usually best |
| Stacking | 0.11-0.12 | ~60s | Highest accuracy, slow |
| Voting | 0.11-0.13 | ~45s | Good balance |
| Random Forest | 0.12-0.14 | ~20s | Fast, robust |
| Gradient Boosting | 0.12-0.14 | ~30s | Good performance |
| Ridge | 0.13-0.15 | ~1s | Fast baseline |
| TensorFlow DNN | 0.15-0.20 | ~120s | Needs more data |

**Note:** RMSE is in log scale. To convert to dollars: exp(rmse) ≈ percentage error.

### Why Sklearn Models Often Win

For this dataset (1,460 samples, 220+ features):
- **Tree-based models** handle tabular data well
- **XGBoost** has superior regularization
- **Ensemble methods** combine strengths
- **TensorFlow DNN** needs more data (works better with 10k+ samples)

## Integration with Existing Pipeline

The new approach is fully compatible with the existing TFX pipeline:

### Old Approach (TensorFlow DNN only)

```bash
python scripts/api.py
```

### New Approach (Best sklearn model)

```bash
python scripts/run_pipeline_with_best_model.py
```

Both use the same:
- CsvExampleGen for data ingestion
- Transform for feature engineering
- Evaluator for metrics
- Pusher for deployment

The only difference is the Trainer component:
- **Old:** Uses `utils/model_utils.py` (TensorFlow DNN)
- **New:** Uses `utils/sklearn_trainer.py` (sklearn models)

## Troubleshooting

### Issue: "Model comparison takes too long"

**Solution:** Use fewer models and fewer CV folds

```bash
python scripts/compare_models.py --cv-folds 3 --models XGBoost RandomForest Ridge
```

### Issue: "sklearn model not found in comparison results"

**Solution:** Run comparison first

```bash
python scripts/compare_models.py --save-best-model
python scripts/run_pipeline_with_best_model.py
```

### Issue: "SavedModel serving signature mismatch"

**Solution:** Ensure transform output is correctly passed to the wrapper

Check `utils/sklearn_trainer.py:SklearnModelWrapper` for correct transform loading.

### Issue: "Predictions are in log scale"

**Solution:** The wrapper should automatically convert to original scale

Verify this line in `sklearn_trainer.py:serve()`:
```python
predictions = tf.exp(predictions) - 1.0  # Convert log to original scale
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    TFX PIPELINE WITH BEST MODEL             │
└─────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Data (CSV)  │────▶│ ExampleGen   │────▶│ SchemaGen    │
└──────────────┘     └──────────────┘     └──────────────┘
                                                   │
┌───────────────────────────────────────────────  │
│                                                  ▼
│  PARALLEL EXECUTION:                    ┌──────────────┐
│  Model Comparison (offline)             │  Transform   │
│  ┌────────────────────────┐             └──────────────┘
│  │ XGBoost      │ Ridge   │                     │
│  │ RandomForest │ Lasso   │                     ▼
│  │ GradBoost    │ ElNet   │            ┌──────────────┐
│  │ Voting       │ Stack   │            │   Trainer    │◀─ custom_config
│  └────────────────────────┘            │ (sklearn)    │   {'model_name': 'XGBoost'}
│           │                             └──────────────┘
│           ▼                                     │
│  Select Best Model                              ▼
│  (e.g., XGBoost)                       ┌──────────────┐
│                                        │  Evaluator   │
└────────────────────────────────────── └──────────────┘
                                                  │
                                                  ▼
                                         ┌──────────────┐
                                         │    Pusher    │
                                         └──────────────┘
                                                  │
                                                  ▼
                                         models/serving/
```

## Best Practices

1. **Run comparison once** - Cache results for faster subsequent pipelines
2. **Use more CV folds for small datasets** - 10 folds for <2000 samples
3. **Monitor training time** - Ensemble methods are slower but often better
4. **Validate on test set** - CV estimates may be optimistic
5. **Retrain periodically** - As data grows, best model may change

## Next Steps

After deploying the best model:

1. **Test predictions** - Verify model works on sample data
2. **Compare with TensorFlow DNN** - Quantify improvement
3. **Deploy to production** - Use TensorFlow Serving
4. **Monitor performance** - Track RMSE on new data
5. **Retrain regularly** - Update model as data grows

## References

- **XGBoost Documentation:** https://xgboost.readthedocs.io/
- **TFX Custom Trainer:** https://www.tensorflow.org/tfx/guide/trainer
- **TensorFlow SavedModel:** https://www.tensorflow.org/guide/saved_model
- **Model Comparison Guide:** `docs/MODEL_COMPARISON.md`
