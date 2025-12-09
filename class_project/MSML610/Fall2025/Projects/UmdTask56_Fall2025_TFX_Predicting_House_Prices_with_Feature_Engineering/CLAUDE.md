# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Project Title:** Predicting House Prices with Feature Engineering using TFX

**Objective:** Build a robust TensorFlow Extended (TFX) pipeline that predicts house prices based on 80+ features, optimizing for accuracy and generalization. The pipeline includes data ingestion, validation, feature engineering, model training (comparing XGBoost and TensorFlow DNN), evaluation, and deployment in a Dockerized environment.

**Dataset:** Ames Housing Dataset
- Training data: 1,460 samples with 80 features + SalePrice target
- Test data: 1,459 samples (no target - for predictions)
- Location: `./data/train.csv`, `./data/test.csv`, `./data/data_description.txt`
- Features include property characteristics (size, quality, location, amenities, condition)

## Project Structure

```
.
├── data/                          # Dataset files
│   ├── train.csv                  # Training data (1460 rows, 81 columns)
│   ├── test.csv                   # Test data (1459 rows, 80 columns)
│   └── data_description.txt       # Feature descriptions
├── utils/                         # Core logic and helper functions
│   ├── __init__.py
│   ├── config.py                  # Pipeline configuration
│   ├── data_utils.py              # Data processing helpers
│   ├── feature_engineering.py     # Transform preprocessing functions
│   ├── model_utils.py             # Model training utilities
│   └── evaluation_utils.py        # Evaluation and metrics
├── notebooks/                     # Jupyter notebooks (documentation focused)
│   ├── API.ipynb                  # TFX pipeline API documentation
│   └── Example.ipynb              # House price prediction walkthrough
├── scripts/                       # Python scripts (executable versions)
│   ├── api.py                     # TFX pipeline runner
│   └── example.py                 # End-to-end example execution
├── pipelines/                     # TFX pipeline definitions
│   ├── __init__.py
│   └── house_price_pipeline.py    # Main pipeline definition
├── docker/                        # Docker configuration
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
├── docs/                          # Markdown documentation
│   ├── API.md                     # TFX API architecture documentation
│   └── Example.md                 # Project walkthrough and results
├── models/                        # Saved model artifacts (generated)
├── pipeline_outputs/              # TFX pipeline outputs (generated)
└── CLAUDE.md                      # This file
```

## TFX Pipeline Architecture

The pipeline consists of 6 main TFX components, executed sequentially:

### 1. CsvExampleGen
- **Purpose:** Ingest train.csv and test.csv data
- **Location:** First component in pipeline
- **Output:** TFRecord format examples for downstream components
- **Key considerations:** Handle the 80 features properly, ensure proper train/test split

### 2. SchemaGen
- **Purpose:** Automatically generate and validate data schema
- **Location:** After ExampleGen
- **Output:** Schema.pbtxt defining feature types, domains, and constraints
- **Key considerations:**
  - Many categorical features with specific domains (see data_description.txt)
  - Numerical features with different scales
  - Missing values are common (NA often means "not applicable" not "missing")

### 3. Transform
- **Purpose:** Feature engineering and preprocessing
- **Location:** After SchemaGen
- **Key tasks:**
  - Handle missing values (numerical imputation, categorical encoding)
  - Feature scaling (standardization/normalization)
  - Create interaction terms (e.g., TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF)
  - Encode categorical variables (one-hot or ordinal encoding)
  - Create derived features (e.g., Age = YrSold - YearBuilt)
- **Implementation:** Define `preprocessing_fn` in `utils/feature_engineering.py`
- **Output:** Transformed features as TFRecord, transform graph

### 4. Trainer
- **Purpose:** Train regression models on transformed data
- **Location:** After Transform
- **Models to implement:**
  - **XGBoost:** Gradient boosting regressor (good for tabular data)
  - **TensorFlow DNN:** Deep neural network (better TFX integration)
- **Configuration:** Hyperparameters, learning rates, epochs
- **Output:** Saved models in SavedModel format
- **Key metrics:** RMSE (primary), R² score, MAE

### 5. Evaluator
- **Purpose:** Evaluate model performance and compare models
- **Location:** After Trainer
- **Metrics:**
  - RMSE (Root Mean Squared Error) - primary metric
  - R² (R-squared) - goodness of fit
  - MAE (Mean Absolute Error)
- **Bonus features:**
  - Cross-validation results
  - Model comparison (XGBoost vs TensorFlow DNN)
  - Feature importance analysis
- **Output:** Evaluation results, validation decisions

### 6. Pusher
- **Purpose:** Deploy approved models to serving location
- **Location:** Final component
- **Deployment target:** Docker container with serving infrastructure
- **Output:** Production-ready model in serving directory

## Development Workflow

**Core Principle:** Keep notebooks clean and focused on documentation. Place all logic in the `utils/` module.

1. **Write functions in utils/** - All data processing, feature engineering, model training logic
2. **Call from notebooks** - Notebooks import and execute utils functions, display results
3. **Create scripts** - Python scripts for automated pipeline execution
4. **Document in markdown** - Detailed documentation in docs/ folder

**Example:**
```python
# ❌ BAD: Logic in notebook
# Cell 1
import pandas as pd
data = pd.read_csv('data/train.csv')
data['Age'] = data['YrSold'] - data['YearBuilt']
# ... 50 more lines of feature engineering

# ✅ GOOD: Call utils from notebook
# Cell 1
from utils.feature_engineering import load_and_engineer_features
data = load_and_engineer_features('data/train.csv')
data.head()  # Display results
```

## Phased Development Plan

### Phase 1: Project Foundation & Setup
**Status:** Not Started
**Tasks:**
- Create folder structure (utils/, notebooks/, scripts/, pipelines/, docker/, docs/)
- Set up Dockerfile and docker-compose.yml
- Create requirements.txt with TFX, TensorFlow, XGBoost, pandas, scikit-learn
- Initialize utils module with __init__.py and config.py
- Create placeholder notebooks and scripts

### Phase 2: Data Ingestion & Validation
**Status:** Not Started
**Tasks:**
- Implement CsvExampleGen in pipelines/house_price_pipeline.py
- Implement SchemaGen
- Create data exploration utilities in utils/data_utils.py
- Generate schema and validate against train.csv
- Create Example.ipynb showing data ingestion

### Phase 3: Feature Engineering & Transformation
**Status:** Not Started
**Tasks:**
- Implement Transform component with preprocessing_fn
- In utils/feature_engineering.py:
  - Handle missing values (imputation strategies per feature type)
  - Feature scaling (StandardScaler or MinMaxScaler)
  - Create interaction terms (TotalSF, TotalBath, etc.)
  - Encode categorical variables (one-hot for nominal, ordinal for ordered)
  - Create derived features (Age, Remodeled, etc.)
- Document all transformations in docstrings
- Update Example.ipynb with feature engineering walkthrough

### Phase 4: Model Training
**Status:** Not Started
**Tasks:**
- Implement Trainer component supporting both models
- In utils/model_utils.py:
  - Create XGBoost training function
  - Create TensorFlow DNN training function
  - Hyperparameter configuration
  - Model persistence utilities
- Train both models on transformed data
- Compare training metrics

### Phase 5: Model Evaluation
**Status:** Not Started
**Tasks:**
- Implement Evaluator component
- In utils/evaluation_utils.py:
  - RMSE calculation
  - R² score calculation
  - Cross-validation implementation (bonus)
  - Model comparison utilities
- Generate evaluation reports
- Determine best model

### Phase 6: Model Serving & Deployment
**Status:** Not Started
**Tasks:**
- Implement Pusher component
- Finalize Docker configuration for full pipeline
- Create model serving interface
- Test end-to-end pipeline in Docker
- Generate predictions on test.csv

### Phase 7: Documentation & Polish
**Status:** Not Started
**Tasks:**
- Write docs/API.md (TFX pipeline architecture, components, design decisions)
- Write docs/Example.md (project walkthrough, results, insights)
- Clean up notebooks (remove debugging code, add markdown explanations)
- Final code review and refactoring
- Update CLAUDE.md with lessons learned

## Key Commands

### Docker Commands
```bash
# Build Docker image
docker build -t house-price-tfx -f docker/Dockerfile .

# Run Docker container
docker-compose -f docker/docker-compose.yml up

# Enter running container
docker exec -it house-price-tfx bash

# Stop containers
docker-compose -f docker/docker-compose.yml down
```

### TFX Pipeline Commands
```bash
# Run the full pipeline
python scripts/api.py --pipeline-name house_price_pipeline --pipeline-root pipeline_outputs/

# Run specific components (for debugging)
python scripts/api.py --pipeline-name house_price_pipeline --components ExampleGen,SchemaGen

# Run example end-to-end
python scripts/example.py
```

### Development Commands
```bash
# Install dependencies
pip install -r docker/requirements.txt

# Run notebooks
jupyter notebook notebooks/

# Run linting/formatting (if configured)
black utils/ scripts/
pylint utils/ scripts/
```

### Testing Commands
```bash
# Test data ingestion
python -c "from utils.data_utils import test_data_ingestion; test_data_ingestion()"

# Test feature engineering
python -c "from utils.feature_engineering import test_feature_engineering; test_feature_engineering()"

# Test model training
python -c "from utils.model_utils import test_model_training; test_model_training()"
```

## Important Data Considerations

### Missing Values
- **"NA" strings:** Often mean "not applicable" (e.g., no garage → GarageType = NA)
- **Empty values:** True missing data requiring imputation
- **Strategy:** Check data_description.txt to understand if NA is meaningful

### Feature Types
- **Ordinal features:** Many quality/condition ratings (Ex, Gd, TA, Fa, Po)
  - Should be encoded as ordered integers, not one-hot
  - Examples: ExterQual, BsmtQual, KitchenQual, OverallQual, OverallCond
- **Nominal features:** Categories without order
  - Should be one-hot encoded
  - Examples: Neighborhood, HouseStyle, RoofStyle, Exterior1st
- **Numerical features:** Continuous or discrete numbers
  - May need scaling
  - Examples: LotArea, GrLivArea, YearBuilt, SalePrice (target)

### Feature Engineering Ideas
- **Total SF:** Combine basement, 1st floor, 2nd floor square footage
- **Age features:** YrSold - YearBuilt, YrSold - YearRemodAdd
- **Boolean indicators:** HasGarage, HasPool, HasFireplace, IsRemodeled
- **Interaction terms:** OverallQual * GrLivArea, Neighborhood * YearBuilt
- **Polynomial features:** GrLivArea², LotArea²

### Target Variable
- **SalePrice:** House sale price in dollars
- **Distribution:** Likely right-skewed, consider log transformation
- **Range:** Check for outliers

## Model Strategy

### XGBoost Model
- **Advantages:** Fast training, handles missing values, feature importance
- **Hyperparameters to tune:** n_estimators, max_depth, learning_rate, subsample
- **Expected performance:** Strong baseline for tabular data

### TensorFlow DNN Model
- **Advantages:** Better TFX integration, flexible architecture, supports TFX Transform
- **Architecture:** Input → Dense(128) → Dense(64) → Dense(32) → Output
- **Hyperparameters to tune:** layers, units, dropout, learning_rate, batch_size
- **Expected performance:** May require more tuning than XGBoost

### Comparison Criteria
- **Primary:** RMSE on validation set
- **Secondary:** R² score, training time, inference speed
- **Bonus:** Cross-validation stability, feature importance insights

## TFX-Specific Notes

### Transform Component Best Practices
- Define `preprocessing_fn` using TensorFlow Transform (tf.Transform)
- Use `tft.scale_to_z_score()` for numerical scaling
- Use `tft.compute_and_apply_vocabulary()` for categorical encoding
- Save transform graph for consistent inference-time preprocessing

### Trainer Component Best Practices
- Use `tfx.components.Trainer` with custom training function
- For XGBoost: Create custom executor or use TFX's generic trainer
- For TensorFlow: Use `run_fn` with `tf.keras` model
- Save models in SavedModel format for TFX Pusher

### Evaluator Component Best Practices
- Define custom metrics using `tfma.MetricConfig`
- Set evaluation thresholds for model approval
- Compare against baseline or previous models

### Pipeline Orchestration
- Use `tfx.orchestration.LocalDagRunner` for local development
- Consider `tfx.orchestration.experimental.KubeflowDagRunner` for production
- Store metadata in SQLite (local) or MySQL (production)

## Documentation Requirements

### docs/API.md
Should contain:
- TFX pipeline architecture diagram (text-based is fine)
- Detailed explanation of each component
- Design decisions and trade-offs
- Transform preprocessing_fn logic
- Model architectures (XGBoost and TF DNN)
- Evaluation metrics and thresholds

### docs/Example.md
Should contain:
- Project objective and dataset description
- Step-by-step walkthrough of pipeline execution
- Feature engineering decisions and rationale
- Model comparison results
- Final predictions on test.csv
- Insights and lessons learned

### Notebooks
- **API.ipynb:** Interactive documentation of TFX pipeline components
- **Example.ipynb:** End-to-end project execution with visualizations and outputs

## Common Pitfalls to Avoid

1. **Don't put logic in notebooks** - Use utils module
2. **Don't ignore data_description.txt** - It explains feature meanings and NA values
3. **Don't forget to handle ordinal features** - They need ordered encoding, not one-hot
4. **Don't skip data validation** - SchemaGen catches issues early
5. **Don't overtrain on small dataset** - Use cross-validation, regularization
6. **Don't forget log transform** - SalePrice is likely right-skewed
7. **Don't mix train/test data** - Test.csv has no SalePrice, don't leak information

## Next Steps

1. **Start with Phase 1:** Set up project structure and Docker environment
2. **Implement incrementally:** Complete each phase before moving to the next
3. **Test frequently:** Validate each component before integration
4. **Document as you go:** Add docstrings, comments, and markdown explanations
5. **Compare models:** Ensure both XGBoost and TensorFlow DNN are properly evaluated

Good luck with the project!
