# TFX Pipeline Complete Example

**Project:** Predicting House Prices with Feature Engineering using TFX

This document presents a complete, end-to-end example demonstrating how to use the TFX Pipeline wrapper layer for house price prediction.

---

## Example Overview

This example demonstrates:
1. Loading and exploring house price data
2. Comparing multiple regression models
3. Running the TFX pipeline with the best model
4. Making predictions on new data
5. Visualizing results

**Total Time:** ~5 minutes (including 8-model comparison)

---

## Prerequisites

```bash
# Ensure you're in the project directory
cd /path/to/UmdTask56_Fall2025_TFX_Predicting_House_Prices_with_Feature_Engineering

# Verify data exists
ls data/train.csv data/test.csv

# Expected output:
# data/train.csv
# data/test.csv
```

---

## Step 1: Load and Explore Data

```python
from utils.tfx_pipeline_utils import DataPipelineWrapper

# Initialize data loader
data_loader = DataPipelineWrapper(data_root='./data')

# Load training data
train_df = data_loader.load_training_data()
test_df = data_loader.load_test_data()

# Explore data
print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(f"\nTarget variable (SalePrice) statistics:")
print(train_df['SalePrice'].describe())
```

**Expected Output:**
```
Training data shape: (1460, 81)
Test data shape: (1459, 80)

Target variable (SalePrice) statistics:
count      1460.000000
mean     180921.195890
std       79442.502883
min       34900.000000
25%      129975.000000
50%      163000.000000
75%      214000.000000
max      755000.000000
```

**Key Insights:**
- 1,460 training samples with 80 features + 1 target
- 1,459 test samples (no target)
- Sale prices range from $34,900 to $755,000
- Median price: $163,000

---

## Step 2: Compare Multiple Models

```python
from utils.tfx_pipeline_utils import ModelComparisonWrapper

# Initialize model comparator
comparator = ModelComparisonWrapper(output_dir='./models/comparison')

# Compare all 8 models with 5-fold cross-validation
print("Comparing 8 regression models (this takes ~3 minutes)...")
results = comparator.compare_all_models(cv_folds=5)

# Display results
print("\n" + "="*60)
print("MODEL COMPARISON RESULTS")
print("="*60)

for model_name, metrics in results['comparison']['models'].items():
    print(f"\n{model_name}:")
    print(f"  CV RMSE: {metrics['cv_mean_rmse']:.4f} ± {metrics['cv_std_rmse']:.4f}")
    print(f"  Training Time: {metrics['training_time']:.2f}s")

# Get best model
best_model = comparator.get_best_model_name()
best_rmse = results['comparison']['models'][best_model]['cv_mean_rmse']

print("\n" + "="*60)
print(f"BEST MODEL: {best_model}")
print(f"CV RMSE: {best_rmse:.4f}")
print("="*60)
```

**Expected Output:**
```
============================================================
MODEL COMPARISON RESULTS
============================================================

XGBoost:
  CV RMSE: 0.1300 ± 0.0117
  Training Time: 9.74s

RandomForest:
  CV RMSE: 0.1497 ± 0.0100
  Training Time: 2.47s

GradientBoosting:
  CV RMSE: 0.1273 ± 0.0101
  Training Time: 24.59s

Ridge:
  CV RMSE: 0.1410 ± 0.0254
  Training Time: 0.17s

Lasso:
  CV RMSE: 0.1423 ± 0.0281
  Training Time: 0.19s

ElasticNet:
  CV RMSE: 0.1411 ± 0.0267
  Training Time: 0.23s

VotingEnsemble:
  CV RMSE: 0.1301 ± 0.0106
  Training Time: 30.56s

StackingEnsemble:
  CV RMSE: 0.1271 ± 0.0135
  Training Time: 135.99s

============================================================
BEST MODEL: StackingEnsemble
CV RMSE: 0.1271
============================================================
```

**Analysis:**
- **Winner:** StackingEnsemble (RMSE: 0.1271)
- **Runner-up:** GradientBoosting (RMSE: 0.1273, 5.5x faster)
- **Fastest:** Ridge (0.17s, but 10.9% worse accuracy)
- **Trade-off:** StackingEnsemble takes 2.3 minutes but achieves best accuracy

---

## Step 3: Run TFX Pipeline with Best Model

```python
from utils.tfx_pipeline_utils import TFXPipelineWrapper

# Initialize pipeline wrapper
pipeline = TFXPipelineWrapper(
    pipeline_name='house_price_prediction',
    pipeline_root='./pipeline_outputs',
    model_dir='./models'
)

# Run complete TFX pipeline with StackingEnsemble
print("\nRunning TFX pipeline (takes ~2 minutes)...")
pipeline.run_pipeline(
    trainer_module='utils.sklearn_trainer'
)

print("\nPipeline execution complete!")
```

**Pipeline Steps Executed:**
1. **CsvExampleGen** - Ingests train.csv and test.csv
2. **SchemaGen** - Validates data schema
3. **Transform** - Engineers 77 features from 80 raw features
4. **Trainer** - Trains StackingEnsemble model
5. **Evaluator** - Evaluates model performance
6. **Pusher** - Deploys model to `models/serving/`

**Expected Output:**
```
[CsvExampleGen] Processing data...
[SchemaGen] Generating schema...
[Transform] Applying feature engineering...
[Trainer] Training StackingEnsemble...
  Progress: Training base models...
  Progress: Training meta-learner...
[Evaluator] Evaluating model...
[Pusher] Deploying model...

Pipeline execution complete!
```

---

## Step 4: Load and Inspect Deployed Model

```python
# Get path to deployed model
model_path = pipeline.get_latest_model_path()
print(f"\nModel deployed to: {model_path}")

# Load the model
model = pipeline.load_model(model_path)
print(f"Model type: {type(model).__name__}")

# Inspect StackingEnsemble structure
if hasattr(model, 'estimators'):
    print("\nBase Models:")
    for name, estimator in model.estimators:
        print(f"  - {name}: {type(estimator).__name__}")
    print(f"\nMeta-Learner: {type(model.final_estimator_).__name__}")
```

**Expected Output:**
```
Model deployed to: ./models/serving/1764715988

Model type: StackingRegressor

Base Models:
  - XGBoost: XGBRegressor
  - RandomForest: RandomForestRegressor
  - GradientBoosting: GradientBoostingRegressor
  - Ridge: Ridge

Meta-Learner: Ridge
```

**Model Architecture:**
```
Input (77 features)
         ↓
┌────────┴────────┐
│  Base Models    │
│  - XGBoost      │
│  - RandomForest │
│  - GradBoosting │
│  - Ridge        │
└────────┬────────┘
         ↓
    Predictions
         ↓
  Meta-Learner (Ridge)
         ↓
  Final Prediction
```

---

## Step 5: Make Predictions

```python
import numpy as np
import pandas as pd

# For demonstration, let's use a sample from test data
# In practice, you would need to apply the same transformations

# Load the comparison results to see metrics
results = comparator.load_results()

# Get model performance
model_metrics = results['comparison']['models']['StackingEnsemble']

print("\n" + "="*60)
print("MODEL PERFORMANCE METRICS")
print("="*60)
print(f"Cross-Validation RMSE: {model_metrics['cv_mean_rmse']:.4f}")
print(f"Cross-Validation Std:  {model_metrics['cv_std_rmse']:.4f}")
print(f"Training R² Score:     {model_metrics['train_r2']:.4f}")
print(f"Training MAE:          {model_metrics['train_mae']:.4f}")
print("="*60)

# Interpret RMSE
avg_rmse = model_metrics['cv_mean_rmse']
print(f"\nInterpretation:")
print(f"RMSE of {avg_rmse:.4f} in log-scale means:")
print(f"  - For $100,000 house: ±$13,500 prediction error (13.5%)")
print(f"  - For $200,000 house: ±$27,000 prediction error (13.5%)")
print(f"  - For $400,000 house: ±$54,000 prediction error (13.5%)")
```

**Expected Output:**
```
============================================================
MODEL PERFORMANCE METRICS
============================================================
Cross-Validation RMSE: 0.1271
Cross-Validation Std:  0.0135
Training R² Score:     0.9808
Training MAE:          0.0398
============================================================

Interpretation:
RMSE of 0.1271 in log-scale means:
  - For $100,000 house: ±$13,500 prediction error (13.5%)
  - For $200,000 house: ±$27,000 prediction error (13.5%)
  - For $400,000 house: ±$54,000 prediction error (13.5%)
```

---

## Step 6: Visualize Results

```python
from utils.tfx_pipeline_utils import visualize_results

# Generate all visualizations
print("\nGenerating visualizations...")
visualize_results(output_dir='./docs/visualizations')

print("\nVisualizations created:")
print("  1. cv_rmse_comparison.png - Model performance comparison")
print("  2. cv_score_distributions.png - Cross-validation stability")
print("  3. training_time_comparison.png - Training efficiency")
print("  4. multi_metric_comparison.png - RMSE, MAE, R² metrics")
print("  5. cv_variability.png - Model stability")
print("  6. performance_time_tradeoff.png - Speed vs accuracy")
print("  7. summary_dashboard.png - Complete overview")
```

**Generated Visualizations:**
- 7 high-resolution PNG files (300 DPI)
- Total size: ~2.2 MB
- Location: `docs/visualizations/`

---

## Complete Script

Here's the complete example in one script:

```python
#!/usr/bin/env python
"""
Complete TFX Pipeline Example for House Price Prediction
"""

from utils.tfx_pipeline_utils import (
    DataPipelineWrapper,
    ModelComparisonWrapper,
    TFXPipelineWrapper,
    run_complete_pipeline,
    visualize_results
)

def main():
    print("="*70)
    print("HOUSE PRICE PREDICTION - COMPLETE EXAMPLE")
    print("="*70)

    # Step 1: Load data
    print("\n[Step 1/6] Loading data...")
    data_loader = DataPipelineWrapper()
    train_df = data_loader.load_training_data()
    test_df = data_loader.load_test_data()
    print(f"  Training: {train_df.shape}, Test: {test_df.shape}")

    # Step 2: Compare models
    print("\n[Step 2/6] Comparing 8 models...")
    comparator = ModelComparisonWrapper()
    results = comparator.compare_all_models(cv_folds=5)
    best_model = comparator.get_best_model_name()
    print(f"  Best model: {best_model}")

    # Step 3: Run TFX pipeline
    print("\n[Step 3/6] Running TFX pipeline...")
    pipeline = TFXPipelineWrapper()
    pipeline.run_pipeline()

    # Step 4: Load model
    print("\n[Step 4/6] Loading deployed model...")
    model_path = pipeline.get_latest_model_path()
    model = pipeline.load_model(model_path)
    print(f"  Model loaded from: {model_path}")

    # Step 5: Show metrics
    print("\n[Step 5/6] Model metrics:")
    metrics = results['comparison']['models'][best_model]
    print(f"  CV RMSE: {metrics['cv_mean_rmse']:.4f}")
    print(f"  R² Score: {metrics['train_r2']:.4f}")

    # Step 6: Visualize
    print("\n[Step 6/6] Creating visualizations...")
    visualize_results()
    print("  7 plots created in docs/visualizations/")

    print("\n" + "="*70)
    print("EXAMPLE COMPLETE")
    print("="*70)
    print(f"\nBest model ({best_model}) deployed and ready for predictions!")
    print(f"Model location: {model_path}")

if __name__ == '__main__':
    main()
```

**Save as:** `scripts/complete_example.py`

**Run:**
```bash
python scripts/complete_example.py
```

---

## Alternative: One-Line Execution

For maximum simplicity, use the convenience function:

```python
from utils.tfx_pipeline_utils import run_complete_pipeline, visualize_results

# Run everything in one call
results, model_path = run_complete_pipeline(cv_folds=5)

# Generate visualizations
visualize_results()

# Done!
print(f"Best model: {results['comparison']['best_model']}")
print(f"Deployed to: {model_path}")
```

---

## Results Summary

After running this example, you will have:

1. **Model Comparison Results**
   - Location: `models/comparison/comparison_results.json`
   - Contains metrics for all 8 models

2. **Deployed Model**
   - Location: `models/serving/<version>/`
   - Files: `saved_model.pb`, `sklearn_model.pkl`

3. **Visualizations**
   - Location: `docs/visualizations/`
   - 7 PNG files showing model performance

4. **Pipeline Artifacts**
   - Location: `pipeline_outputs/house_price_prediction/`
   - Includes transform graphs, schemas, and metadata

---

## Next Steps

After completing this example:

1. **Make Predictions:**
   ```python
   model = pipeline.load_model()
   # predictions = model.predict(new_data)
   ```

2. **Experiment with Models:**
   ```python
   # Try a different model
   pipeline.run_pipeline(
       trainer_module='utils.model_utils'  # Use TensorFlow DNN
   )
   ```

3. **Adjust Parameters:**
   ```python
   # More cross-validation folds
   results = comparator.compare_all_models(cv_folds=10)
   ```

4. **Deploy to Production:**
   - Use TensorFlow Serving with the SavedModel
   - Set up CI/CD for automated retraining
   - Monitor model performance over time

---

## Troubleshooting

**Issue:** Module not found errors
```bash
# Solution: Ensure you're in project root
cd /path/to/project
python -c "import utils.tfx_pipeline_utils"  # Should work
```

**Issue:** No models found
```bash
# Solution: Run comparison first
python scripts/compare_models.py --cv-folds 5
```

**Issue:** TFX import errors
```bash
# Solution: Install in Docker
docker build -t house-price-tfx -f docker/Dockerfile .
docker run -it house-price-tfx
```

---

## Conclusion

This example demonstrated:
- Loading data with DataPipelineWrapper
- Comparing models with ModelComparisonWrapper
- Running TFX pipeline with TFXPipelineWrapper
- Using convenience functions for simplified workflows

**Total Execution Time:** ~5 minutes
**Lines of Code:** ~15 (using wrapper) vs ~150+ (using TFX directly)
**Result:** Production-ready model with 13.5% average prediction error

The wrapper layer successfully abstracts TFX complexity while maintaining full functionality!
