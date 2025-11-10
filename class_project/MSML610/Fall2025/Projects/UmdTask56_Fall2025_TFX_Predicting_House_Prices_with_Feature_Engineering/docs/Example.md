# House Price Prediction - Project Walkthrough

**Project:** Predicting House Prices with Feature Engineering using TFX

**Status:** Phase 1 Complete

---

## Project Overview

### Objective
Build a robust machine learning pipeline using TensorFlow Extended (TFX) to predict house prices based on various property features. The goal is to create an end-to-end, production-ready system that can accurately estimate house prices while demonstrating best practices in ML pipeline development.

### Dataset
**Ames Housing Dataset**
- Training data: 1,460 samples
- Test data: 1,459 samples
- Features: 80 (property characteristics)
- Target: SalePrice (house price in dollars)

### Success Metrics
- **Primary:** RMSE < $30,000
- **Secondary:** R² > 0.85
- **Bonus:** Compare XGBoost vs TensorFlow DNN performance

---

## Phase 1: Project Foundation ✓

### What We Built

#### Folder Structure
```
├── data/                    # Dataset files
├── utils/                   # Helper functions and logic
│   ├── config.py           # Configuration settings
│   ├── data_utils.py       # Data loading and exploration
│   ├── feature_engineering.py
│   ├── model_utils.py
│   └── evaluation_utils.py
├── notebooks/              # Documentation notebooks
│   ├── API.ipynb
│   └── Example.ipynb
├── scripts/                # Executable Python scripts
│   ├── api.py
│   └── example.py
├── pipelines/              # TFX pipeline definitions
├── docker/                 # Docker configuration
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
└── docs/                   # Markdown documentation
```

#### Key Files Created
- **CLAUDE.md** - Comprehensive guide for future development
- **config.py** - Centralized configuration management
- **Pipeline skeleton** - TFX components ready for implementation
- **Docker setup** - Containerized development environment

#### Configuration Highlights
```python
# Pipeline Configuration
PIPELINE_NAME = "house_price_prediction_pipeline"
TARGET_COLUMN = "SalePrice"

# Model Parameters
XGBOOST_PARAMS = {...}  # Optimized for tabular data
TF_DNN_PARAMS = {...}   # Deep learning configuration

# Feature Engineering
LOG_TRANSFORM_TARGET = True
CREATE_INTERACTIONS = True
```

### Testing Phase 1

Run the example script to verify setup:
```bash
python scripts/example.py
```

Expected output:
```
✓ Train data loaded: (1460, 81)
✓ Test data loaded: (1459, 80)
✓ Configuration loaded successfully
```

---

## Phase 2: Data Ingestion & Validation

**Status:** TODO - To be implemented next

### Goals
1. Implement CsvExampleGen to ingest train.csv and test.csv
2. Generate schema using SchemaGen
3. Explore data characteristics
4. Validate data quality

### Data Exploration (Preview)

Based on initial inspection:
- **Missing values:** Multiple columns have missing data
- **Feature types:** Mix of numerical and categorical
- **Target distribution:** SalePrice likely right-skewed
- **Special values:** "NA" often means "not applicable"

### Tasks
- [ ] Implement CsvExampleGen component
- [ ] Generate and review schema
- [ ] Create data exploration visualizations
- [ ] Document missing value patterns
- [ ] Update Example.ipynb with findings

---

## Phase 3: Feature Engineering & Transformation

**Status:** TODO

### Planned Transformations

#### Missing Value Handling
- Numerical features: Median imputation
- Categorical features: "Missing" category or mode
- Special "NA" values: Recognize as valid category

#### Feature Scaling
- Numerical features: StandardScaler (zero mean, unit variance)
- Rationale: Neural networks require scaled inputs

#### Categorical Encoding
- **Ordinal features** (quality ratings): Label encoding
  - Example: ExterQual: Po=1, Fa=2, TA=3, Gd=4, Ex=5
- **Nominal features** (no order): One-hot encoding
  - Example: Neighborhood → 25 binary columns

#### Derived Features
```python
# Age features
Age = YrSold - YearBuilt
YearsSinceRemodel = YrSold - YearRemodAdd

# Total square footage
TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF

# Total bathrooms
TotalBath = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath

# Boolean indicators
HasGarage = (GarageArea > 0)
HasPool = (PoolArea > 0)
IsRemodeled = (YearRemodAdd != YearBuilt)
```

#### Target Transformation
```python
# Log transform to normalize distribution
log_SalePrice = log(SalePrice)
```

### Tasks
- [ ] Implement preprocessing_fn in feature_engineering.py
- [ ] Create derived features
- [ ] Implement Transform component
- [ ] Visualize feature distributions
- [ ] Document transformation rationale

---

## Phase 4: Model Training

**Status:** TODO

### Model 1: XGBoost

**Why XGBoost?**
- Excellent for tabular data
- Handles missing values natively
- Fast training
- Provides feature importance

**Hyperparameters:**
```python
{
    "n_estimators": 1000,
    "max_depth": 7,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}
```

### Model 2: TensorFlow DNN

**Why TensorFlow DNN?**
- Better TFX integration
- Flexible architecture
- Handles complex patterns
- Production-ready with TF Serving

**Architecture:**
```
Input Layer (n_features)
    ↓
Dense(128, relu) + Dropout(0.2)
    ↓
Dense(64, relu) + Dropout(0.2)
    ↓
Dense(32, relu)
    ↓
Output(1) - Linear activation
```

### Training Strategy
1. Train XGBoost with early stopping
2. Train TensorFlow DNN with early stopping
3. Record training metrics for both
4. Compare training time and performance

### Tasks
- [ ] Implement XGBoost training function
- [ ] Implement TensorFlow DNN training function
- [ ] Implement Trainer component
- [ ] Track training metrics
- [ ] Save model artifacts

---

## Phase 5: Model Evaluation

**Status:** TODO

### Evaluation Metrics

#### RMSE (Root Mean Squared Error)
- **Primary metric** for model selection
- Target: RMSE < $30,000
- Penalizes large errors more heavily

#### R² (R-squared)
- Measures goodness of fit
- Target: R² > 0.85
- Range: 0 (worst) to 1 (perfect)

#### MAE (Mean Absolute Error)
- Average absolute prediction error
- More interpretable than RMSE
- Less sensitive to outliers

### Cross-Validation (Bonus)
```python
# 5-fold cross-validation
cv_scores = cross_validate(model, X, y, cv=5)
mean_rmse = cv_scores.mean()
std_rmse = cv_scores.std()
```

### Model Comparison

Create comparison table:
```
| Metric      | XGBoost | TF DNN  | Winner  |
|-------------|---------|---------|---------|
| RMSE        | TBD     | TBD     | TBD     |
| R²          | TBD     | TBD     | TBD     |
| MAE         | TBD     | TBD     | TBD     |
| Train Time  | TBD     | TBD     | TBD     |
| CV Stable   | TBD     | TBD     | TBD     |
```

### Tasks
- [ ] Implement Evaluator component
- [ ] Calculate all metrics
- [ ] Perform cross-validation
- [ ] Create comparison visualizations
- [ ] Select best model

---

## Phase 6: Model Deployment & Predictions

**Status:** TODO

### Deployment Steps
1. Approve best model via Evaluator
2. Push model to serving directory via Pusher
3. Set up TensorFlow Serving in Docker
4. Create prediction interface

### Generating Predictions
```python
# Load test data
test_df = pd.read_csv('data/test.csv')

# Apply same transformations
test_transformed = transform_fn(test_df)

# Generate predictions
predictions = model.predict(test_transformed)

# Inverse log transform if needed
final_predictions = exp(predictions)

# Create submission file
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'SalePrice': final_predictions
})
submission.to_csv('predictions.csv', index=False)
```

### Docker Deployment
```bash
# Build image
docker build -t house-price-tfx -f docker/Dockerfile .

# Run container
docker-compose up

# Access TensorFlow Serving
curl http://localhost:8501/v1/models/house_price_model
```

### Tasks
- [ ] Implement Pusher component
- [ ] Set up TensorFlow Serving
- [ ] Generate test predictions
- [ ] Create submission file
- [ ] Test end-to-end pipeline in Docker

---

## Phase 7: Documentation & Polish

**Status:** TODO

### Documentation Tasks
- [ ] Complete API.md with all component details
- [ ] Complete Example.md with results and insights
- [ ] Update notebooks with final visualizations
- [ ] Add code comments and docstrings
- [ ] Create README with setup instructions

### Code Quality
- [ ] Run linting (black, pylint)
- [ ] Remove debugging code
- [ ] Optimize performance
- [ ] Add error handling

---

## Results and Insights

**TODO:** Fill in after completing all phases

### Final Model Performance
- Best model: TBD
- RMSE: TBD
- R²: TBD
- MAE: TBD

### Key Findings
1. TBD
2. TBD
3. TBD

### Feature Importance
Top 10 most important features: TBD

### Lessons Learned
1. TBD
2. TBD
3. TBD

---

## Next Steps

**Current Phase:** Phase 1 Complete ✓
**Next Phase:** Phase 2 - Data Ingestion & Validation

To proceed:
```bash
# Run example script to test current setup
python scripts/example.py

# Start implementing Phase 2 components
# Focus on CsvExampleGen and SchemaGen
```

---

## Troubleshooting

### Common Issues

**Issue:** Data not found
```bash
# Verify data files exist
ls data/
# Should show: train.csv, test.csv, data_description.txt
```

**Issue:** Import errors
```bash
# Install dependencies
pip install -r docker/requirements.txt
```

**Issue:** Docker build fails
```bash
# Check Docker daemon is running
docker info

# Rebuild from scratch
docker-compose down
docker-compose build --no-cache
```

---

## References

- [TFX Guide](https://www.tensorflow.org/tfx/guide)
- [Ames Dataset Description](./data/data_description.txt)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/)

---

**Last Updated:** Phase 1 Complete
**Next Update:** After Phase 2 implementation
