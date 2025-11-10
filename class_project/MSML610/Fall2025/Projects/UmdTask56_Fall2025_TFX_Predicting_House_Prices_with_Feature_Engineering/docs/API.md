# TFX Pipeline API Documentation

**Project:** Predicting House Prices with Feature Engineering using TFX

**Last Updated:** Phase 1 Complete

---

## Overview

This document provides comprehensive documentation of the TFX (TensorFlow Extended) pipeline architecture, components, and design decisions for the house price prediction project.

**Note:** "API" here refers to the TFX pipeline's internal interface and architecture—not an external data provider API. This documentation focuses on the pipeline components, their interactions, and the tool itself.

---

## Pipeline Architecture

The house price prediction pipeline consists of 6 main TFX components orchestrated sequentially:

```
┌─────────────────┐
│ CsvExampleGen   │ ──> Data Ingestion
└────────┬────────┘
         │
         v
┌─────────────────┐
│   SchemaGen     │ ──> Schema Generation & Validation
└────────┬────────┘
         │
         v
┌─────────────────┐
│   Transform     │ ──> Feature Engineering
└────────┬────────┘
         │
         v
┌─────────────────┐
│    Trainer      │ ──> Model Training (XGBoost + TF DNN)
└────────┬────────┘
         │
         v
┌─────────────────┐
│   Evaluator     │ ──> Model Evaluation & Comparison
└────────┬────────┘
         │
         v
┌─────────────────┐
│     Pusher      │ ──> Model Deployment
└─────────────────┘
```

---

## Component Details

### 1. CsvExampleGen (Phase 2)

**Status:** TODO - To be implemented in Phase 2

**Purpose:** Ingest CSV data and convert to TFRecord format

**Inputs:**
- `train.csv` - Training data (1,460 samples, 81 columns)
- `test.csv` - Test data (1,459 samples, 80 columns)

**Outputs:**
- Examples in TFRecord format
- Train/eval split

**Configuration:**
- Split ratio: TBD
- Data location: `./data/`

**Design Decisions:**
- TBD in Phase 2

---

### 2. SchemaGen (Phase 2)

**Status:** TODO - To be implemented in Phase 2

**Purpose:** Automatically generate and validate data schema

**Inputs:**
- Statistics from StatisticsGen

**Outputs:**
- `schema.pbtxt` - Feature schema definition

**Key Features:**
- Feature type inference
- Domain specification for categorical features
- Missing value handling rules

**Design Decisions:**
- TBD in Phase 2

---

### 3. Transform (Phase 3)

**Status:** TODO - To be implemented in Phase 3

**Purpose:** Feature engineering and preprocessing

**Inputs:**
- Raw examples from ExampleGen
- Schema from SchemaGen

**Outputs:**
- Transformed examples
- Transform graph for serving

**Transformation Logic:**

#### Missing Value Handling
- TBD in Phase 3

#### Feature Scaling
- TBD in Phase 3

#### Categorical Encoding
- TBD in Phase 3

#### Feature Creation
- TBD in Phase 3

**Design Decisions:**
- TBD in Phase 3

---

### 4. Trainer (Phase 4)

**Status:** TODO - To be implemented in Phase 4

**Purpose:** Train regression models

**Inputs:**
- Transformed examples from Transform
- Transform graph
- Schema

**Models:**

#### Model 1: XGBoost
- **Architecture:** Gradient boosting regressor
- **Hyperparameters:** TBD in Phase 4
- **Advantages:** Fast, handles missing values, feature importance

#### Model 2: TensorFlow DNN
- **Architecture:** Deep neural network
- **Layers:** TBD in Phase 4
- **Advantages:** Better TFX integration, flexible architecture

**Training Configuration:**
- TBD in Phase 4

**Design Decisions:**
- TBD in Phase 4

---

### 5. Evaluator (Phase 5)

**Status:** TODO - To be implemented in Phase 5

**Purpose:** Evaluate and compare models

**Inputs:**
- Trained models from Trainer
- Evaluation examples

**Metrics:**
- **RMSE** (Root Mean Squared Error) - Primary metric
- **R²** (R-squared) - Goodness of fit
- **MAE** (Mean Absolute Error) - Alternative metric

**Model Approval:**
- Threshold: TBD in Phase 5
- Comparison strategy: TBD in Phase 5

**Design Decisions:**
- TBD in Phase 5

---

### 6. Pusher (Phase 6)

**Status:** TODO - To be implemented in Phase 6

**Purpose:** Deploy approved models

**Inputs:**
- Trained model from Trainer
- Model blessing from Evaluator

**Outputs:**
- Model in serving directory
- Deployment status

**Deployment Target:**
- Docker container with TensorFlow Serving

**Design Decisions:**
- TBD in Phase 6

---

## Data Schema

### Target Variable
- **SalePrice** - House sale price in dollars (continuous)

### Feature Categories

#### Numerical Features (35+)
- Square footage features: LotArea, GrLivArea, TotalBsmtSF, etc.
- Year features: YearBuilt, YearRemodAdd, GarageYrBlt
- Count features: BedroomAbvGr, FullBath, Fireplaces, etc.
- Quality ratings: OverallQual, OverallCond (1-10 scale)

#### Categorical Features (40+)
- Nominal: Neighborhood, HouseStyle, RoofStyle, etc.
- Ordinal: ExterQual, KitchenQual, BsmtQual (Ex, Gd, TA, Fa, Po)

#### Special Handling
- "NA" values often mean "not applicable" (e.g., no garage)
- Ordinal features require ordered encoding
- Many features have missing values requiring imputation

---

## Pipeline Configuration

### Directories
- **Pipeline Root:** `./pipeline_outputs/`
- **Metadata:** `./pipeline_outputs/metadata/`
- **Models:** `./models/`
- **Serving:** `./models/serving/`

### Orchestration
- **Runner:** LocalDagRunner (development)
- **Metadata Store:** SQLite database

### Execution
```bash
# Run full pipeline
python scripts/api.py --pipeline-name house_price_pipeline

# Run specific components (if needed)
python scripts/api.py --components ExampleGen,SchemaGen
```

---

## Design Philosophy

### 1. Separation of Concerns
- **Utils module:** All logic and helper functions
- **Notebooks:** Documentation and visualization only
- **Scripts:** Automated execution

### 2. Reproducibility
- TFX Transform ensures consistent preprocessing
- Version-controlled schemas and configs
- Dockerized environment

### 3. Model Comparison
- Train both XGBoost and TensorFlow DNN
- Compare on RMSE, R², training time
- Select best model objectively

### 4. Production-Ready
- TFX pipeline enables easy deployment
- Model serving via TensorFlow Serving
- Scalable to larger datasets

---

## API Usage Examples

### Loading Configuration
```python
from utils import config

# Access paths
data_path = config.get_data_path("train")
pipeline_root = config.get_pipeline_root()

# Access model parameters
xgb_params = config.XGBOOST_PARAMS
tf_params = config.TF_DNN_PARAMS
```

### Running Pipeline Components
```python
from pipelines.house_price_pipeline import create_pipeline
from tfx.orchestration.local.local_dag_runner import LocalDagRunner

# Create pipeline
pipeline = create_pipeline(
    pipeline_name=config.PIPELINE_NAME,
    pipeline_root=config.PIPELINE_ROOT_STR,
    data_path=str(config.DATA_DIR),
    # ... other parameters
)

# Run pipeline
LocalDagRunner().run(pipeline)
```

---

## Future Enhancements

- Add hyperparameter tuning with TFX Tuner
- Implement model explainability (SHAP values)
- Add data drift detection
- Deploy on Kubernetes with KubeflowDagRunner

---

## References

- [TFX Documentation](https://www.tensorflow.org/tfx)
- [TensorFlow Transform](https://www.tensorflow.org/tfx/transform/get_started)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Ames Housing Dataset](http://jse.amstat.org/v19n3/decock.pdf)

---

**TODO:** Complete this documentation as we implement each phase.
