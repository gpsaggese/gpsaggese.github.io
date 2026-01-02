

# TFX Pipeline API Documentation

**Project:** Predicting House Prices with Feature Engineering using TFX

**Important:** "API" here refers to the TFX tool's internal programming interface—not an external data provider API. This document focuses on the TFX pipeline components and our custom wrapper layer.

---

## Overview

This document provides comprehensive documentation of:
1. **Native TFX API** - The TensorFlow Extended components and their interfaces
2. **Wrapper Layer** - Our simplified Python wrapper (`tfx_pipeline_utils.py`) built on top of TFX

The project implements a complete machine learning pipeline for house price prediction using 8 regression models with automated model selection and deployment.

---

## Part 1: Native TFX API

### TFX Pipeline Architecture

TFX (TensorFlow Extended) provides production-ready ML pipeline components. Our pipeline uses 6 core components:

```
Data (CSV) → ExampleGen → SchemaGen → Transform → Trainer → Evaluator → Pusher → Deployed Model
```

### 1. CsvExampleGen

**Purpose:** Ingest CSV data and convert to TFRecord format

**Native API Class:**
```python
from tfx.components import CsvExampleGen

example_gen = CsvExampleGen(input_base=data_root)
```

**Configuration:**
- `input_base` (str): Directory containing train.csv and test.csv
- Output: TFRecord format examples with train/eval split

**Our Implementation:**
```python
# Location: pipelines/house_price_pipeline.py
example_gen = CsvExampleGen(input_base=data_root)
```

**Outputs:**
- Training examples: 1,168 samples (80% split)
- Evaluation examples: 292 samples (20% split)
- Format: Compressed TFRecord files

---

### 2. SchemaGen

**Purpose:** Automatically infer and validate data schema

**Native API Class:**
```python
from tfx.components import SchemaGen

schema_gen = SchemaGen(
    statistics=statistics_gen.outputs['statistics'],
    infer_feature_shape=True
)
```

**Configuration:**
- `statistics`: Input statistics from StatisticsGen or ExampleGen
- `infer_feature_shape`: Boolean to infer feature shapes

**Our Implementation:**
```python
# Uses statistics from ExampleGen
schema_gen = SchemaGen(
    statistics=example_gen.outputs['statistics'],
    infer_feature_shape=True
)
```

**Generated Schema:**
- 80 features with inferred types (int64, float, string)
- Domain specifications for categorical variables
- Missing value annotations

---

### 3. Transform

**Purpose:** Feature engineering using TensorFlow Transform

**Native API Class:**
```python
from tfx.components import Transform

transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file='utils/feature_engineering.py'
)
```

**Key Interface - preprocessing_fn:**
```python
def preprocessing_fn(inputs):
    """
    TensorFlow Transform preprocessing function.

    Args:
        inputs: Dictionary of raw features

    Returns:
        Dictionary of transformed features
    """
    outputs = {}

    # Numerical scaling
    outputs['scaled_feature'] = tft.scale_to_z_score(inputs['raw_feature'])

    # Categorical encoding
    outputs['encoded_cat'] = tft.compute_and_apply_vocabulary(
        inputs['category']
    )

    # Target transformation
    outputs['SalePrice_log'] = tf.math.log(inputs['SalePrice'] + 1)

    return outputs
```

**Our Implementation:**
- Location: `utils/feature_engineering.py`
- Input: 80 raw features
- Output: 77 transformed features
- Operations:
  - StandardScaler for numerical features (mean=0, std=1)
  - One-hot encoding for nominal categories
  - Ordinal encoding for quality ratings
  - Log transformation of target variable

**Transform Graph:**
- Saved for consistent inference-time preprocessing
- Includes vocabulary tables for categorical features
- Reusable across training and serving

---

### 4. Trainer

**Purpose:** Train machine learning models

**Native API Class:**
```python
from tfx.components import Trainer

trainer = Trainer(
    module_file='utils/sklearn_trainer.py',
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(num_steps=1000),
    eval_args=trainer_pb2.EvalArgs(num_steps=100),
    custom_config={'model_name': 'StackingEnsemble'}
)
```

**Key Interface - run_fn:**
```python
def run_fn(fn_args):
    """
    Training function called by TFX Trainer.

    Args:
        fn_args: FnArgs object with:
            - train_files: List of training TFRecord files
            - eval_files: List of evaluation TFRecord files
            - transform_output: Path to transform graph
            - serving_model_dir: Output directory for model
            - custom_config: Custom configuration dict
    """
    # Load data
    train_dataset = _input_fn(fn_args.train_files, ...)
    eval_dataset = _input_fn(fn_args.eval_files, ...)

    # Train model
    model.fit(X_train, y_train)

    # Save model
    tf.saved_model.save(model, fn_args.serving_model_dir)
```

**Our Implementation:**
- Custom sklearn trainer: `utils/sklearn_trainer.py`
- Wraps sklearn models in TensorFlow SavedModel format
- Supports 8 regression models:
  - XGBoost, RandomForest, GradientBoosting
  - Ridge, Lasso, ElasticNet
  - VotingEnsemble, StackingEnsemble
- Training time: 136 seconds for StackingEnsemble (5-fold CV)

---

### 5. Evaluator

**Purpose:** Evaluate model performance

**Native API Class:**
```python
from tfx.components import Evaluator

evaluator = Evaluator(
    examples=example_gen.outputs['examples'],
    model=trainer.outputs['model'],
    baseline_model=baseline_trainer.outputs['model']
)
```

**Configuration:**
- Metrics: RMSE, MAE, R² score
- Validation thresholds for model approval

**Our Implementation:**
- Evaluates on held-out evaluation set
- Compares against baseline models
- Metrics calculated:
  - CV RMSE: 0.1271 (StackingEnsemble)
  - Train R²: 0.9808
  - Train MAE: 0.0398

---

### 6. Pusher

**Purpose:** Deploy approved models to serving location

**Native API Class:**
```python
from tfx.components import Pusher

pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
            base_directory=serving_model_dir
        )
    )
)
```

**Our Implementation:**
- Deploys to: `models/serving/<version>/`
- Includes:
  - `saved_model.pb` - TensorFlow SavedModel
  - `sklearn_model.pkl` - Original sklearn model (48.6 MB)
  - `variables/` - Model variables
  - `assets/` - Transform graph assets

---

### Pipeline Orchestration

**Native API:**
```python
from tfx.orchestration import pipeline
from tfx.orchestration.local.local_dag_runner import LocalDagRunner

# Create pipeline
tfx_pipeline = pipeline.Pipeline(
    pipeline_name='house_price_prediction',
    pipeline_root='./pipeline_outputs',
    components=[
        example_gen,
        schema_gen,
        transform,
        trainer,
        evaluator,
        pusher
    ]
)

# Run pipeline
LocalDagRunner().run(tfx_pipeline)
```

**Our Implementation:**
- Pipeline definition: `pipelines/house_price_pipeline.py`
- Execution script: `scripts/api.py`
- Metadata stored in SQLite database

---

## Part 2: Wrapper Layer

We built a simplified wrapper layer in `utils/tfx_pipeline_utils.py` that abstracts TFX complexity and integrates with existing project scripts.

### TFXPipelineWrapper Class

**Purpose:** Simplified interface for pipeline operations

**API:**
```python
from utils.tfx_pipeline_utils import TFXPipelineWrapper

# Initialize
wrapper = TFXPipelineWrapper(
    pipeline_name='house_price_prediction',
    pipeline_root='./pipeline_outputs',
    model_dir='./models'
)

# Run pipeline
wrapper.run_pipeline(trainer_module='utils.sklearn_trainer')

# Get latest model
model_path = wrapper.get_latest_model_path()

# Load model
model = wrapper.load_model(model_path)
```

**Key Methods:**

1. **run_pipeline(trainer_module)**
   - Wraps `scripts/api.py`
   - Executes complete TFX pipeline
   - Parameters:
     - `trainer_module`: Python module for training (default: 'utils.sklearn_trainer')

2. **get_latest_model_path()**
   - Returns path to most recent deployed model
   - Finds highest version number in serving directory
   - Returns: `models/serving/<version>/`

3. **load_model(model_path)**
   - Loads trained model (sklearn or TensorFlow)
   - Tries sklearn pickle first, then TensorFlow SavedModel
   - Returns: Model object ready for predictions

---

### ModelComparisonWrapper Class

**Purpose:** Simplified model comparison interface

**API:**
```python
from utils.tfx_pipeline_utils import ModelComparisonWrapper

# Initialize
comparator = ModelComparisonWrapper(output_dir='./models/comparison')

# Compare all models
results = comparator.compare_all_models(cv_folds=5)

# Get best model
best_model = comparator.get_best_model_name()  # Returns: 'StackingEnsemble'

# Load results
results = comparator.load_results()
```

**Key Methods:**

1. **compare_all_models(cv_folds)**
   - Wraps `scripts/compare_models.py`
   - Compares 8 regression models with k-fold CV
   - Returns: Dictionary with performance metrics

2. **get_best_model_name()**
   - Returns name of best performing model
   - Based on lowest CV RMSE

3. **load_results()**
   - Loads comparison results from JSON
   - Returns full results dictionary

---

### DataPipelineWrapper Class

**Purpose:** Simplified data loading interface

**API:**
```python
from utils.tfx_pipeline_utils import DataPipelineWrapper

# Initialize
data = DataPipelineWrapper(data_root='./data')

# Load data
train_df = data.load_training_data()  # Returns pandas DataFrame
test_df = data.load_test_data()
```

**Key Methods:**

1. **load_training_data()**
   - Loads `data/train.csv`
   - Returns: pandas DataFrame (1,460 rows, 81 columns)

2. **load_test_data()**
   - Loads `data/test.csv`
   - Returns: pandas DataFrame (1,459 rows, 80 columns)

---

### Convenience Functions

**run_complete_pipeline(cv_folds)**

High-level function for complete workflow:

```python
from utils.tfx_pipeline_utils import run_complete_pipeline

# Run everything: compare models, select best, deploy
results, model_path = run_complete_pipeline(cv_folds=5)

print(f"Best model: {results['comparison']['best_model']}")
print(f"Deployed to: {model_path}")
```

**Workflow:**
1. Compares all 8 models with cross-validation
2. Selects best model (lowest CV RMSE)
3. Runs TFX pipeline with selected model
4. Returns results and deployed model path

**visualize_results(output_dir)**

Generate all visualizations:

```python
from utils.tfx_pipeline_utils import visualize_results

# Create 7 visualization plots
visualize_results(output_dir='./docs/visualizations')
```

---

## Wrapper Benefits

### 1. Simplified API
- Hide TFX component complexity
- Provide intuitive method names
- Reduce boilerplate code

### 2. Integration with Existing Scripts
- Reuses `api.py`, `compare_models.py`, `run_pipeline_with_best_model.py`
- No code duplication
- Maintains single source of truth

### 3. Consistent Interface
- Uniform patterns across operations
- Easy to learn and use
- Better error handling

### 4. Production-Ready
- Wraps proven TFX components
- Maintains full TFX functionality
- Adds convenience without sacrificing power

---

## Complete Usage Example

```python
from utils.tfx_pipeline_utils import (
    TFXPipelineWrapper,
    ModelComparisonWrapper,
    run_complete_pipeline
)

# Option 1: Step-by-step
wrapper = TFXPipelineWrapper()
comparator = ModelComparisonWrapper()

# Compare models
results = comparator.compare_all_models(cv_folds=5)
best_model = comparator.get_best_model_name()

# Run pipeline with best model
wrapper.run_pipeline(
    trainer_module='utils.sklearn_trainer',
    custom_config={'model_name': best_model}
)

# Load and use model
model = wrapper.load_model()

# Option 2: All-in-one
results, model_path = run_complete_pipeline(cv_folds=5)
```

---

## Configuration

All configuration is centralized in `utils/config.py`:

```python
from utils import config

# Paths
config.DATA_DIR          # './data'
config.PIPELINE_ROOT     # './pipeline_outputs'
config.MODEL_DIR         # './models'

# Model parameters
config.XGBOOST_PARAMS    # XGBoost hyperparameters
config.TF_DNN_PARAMS     # TensorFlow DNN parameters

# Training config
config.TRAIN_STEPS       # Number of training steps
config.EVAL_STEPS        # Number of evaluation steps
config.CV_FOLDS          # Cross-validation folds (default: 5)
```

---

## References

- **TFX Documentation:** https://www.tensorflow.org/tfx
- **TensorFlow Transform:** https://www.tensorflow.org/tfx/transform
- **XGBoost:** https://xgboost.readthedocs.io/
- **Dataset:** [Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

---

## Summary

### Native TFX API
- 6 core components for production ML pipelines
- Standardized interfaces (ExampleGen, Transform, Trainer, etc.)
- Built-in model versioning and deployment

### Our Wrapper Layer
- 3 main classes: TFXPipelineWrapper, ModelComparisonWrapper, DataPipelineWrapper
- Convenience functions for common workflows
- Integration with existing project scripts

**Result:** A clean, production-ready API for house price prediction with automated model selection and deployment.
