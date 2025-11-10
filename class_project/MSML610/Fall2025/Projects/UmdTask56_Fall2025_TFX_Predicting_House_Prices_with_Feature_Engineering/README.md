# House Price Prediction with TFX

A production-ready machine learning pipeline using TensorFlow Extended (TFX) to predict house prices based on property features.

## Project Overview

**Objective:** Build a robust TFX pipeline that predicts house prices using feature engineering, comparing XGBoost and TensorFlow DNN models.

**Dataset:** Ames Housing Dataset
- Training: 1,460 samples with 80+ features
- Testing: 1,459 samples
- Target: SalePrice (house price in dollars)

**Status:** Phase 1 Complete - Foundation Established ✓

---

## Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- 4GB+ RAM

### Setup

1. **Clone/Navigate to the project:**
   ```bash
   cd MSML610/Fall2025/Projects/UmdTask56_Fall2025_TFX_Predicting_House_Prices_with_Feature_Engineering
   ```

2. **Verify data files exist:**
   ```bash
   ls data/
   # Should show: train.csv, test.csv, data_description.txt
   ```

3. **Install dependencies (local development):**
   ```bash
   pip install -r docker/requirements.txt
   ```

4. **Or use Docker (recommended):**
   ```bash
   docker-compose -f docker/docker-compose.yml build
   docker-compose -f docker/docker-compose.yml up
   ```

### Running the Pipeline

#### Test Current Setup
```bash
# Run example script to verify Phase 1
python scripts/example.py
```

#### Run TFX Pipeline (when implemented)
```bash
# Run full pipeline
python scripts/api.py

# Or in Docker
docker exec -it house-price-tfx python scripts/api.py
```

#### Launch Jupyter Notebooks
```bash
# Local
jupyter notebook notebooks/

# Or in Docker
docker exec -it house-price-tfx jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
# Then open: http://localhost:8888
```

---

## Project Structure

```
.
├── data/                          # Dataset files
│   ├── train.csv                  # Training data
│   ├── test.csv                   # Test data
│   └── data_description.txt       # Feature descriptions
├── utils/                         # Core logic and utilities
│   ├── config.py                  # Configuration settings
│   ├── data_utils.py              # Data processing helpers
│   ├── feature_engineering.py     # Transform preprocessing
│   ├── model_utils.py             # Model training utilities
│   └── evaluation_utils.py        # Evaluation metrics
├── notebooks/                     # Jupyter notebooks
│   ├── API.ipynb                  # TFX API documentation
│   └── Example.ipynb              # Project walkthrough
├── scripts/                       # Executable Python scripts
│   ├── api.py                     # TFX pipeline runner
│   └── example.py                 # End-to-end example
├── pipelines/                     # TFX pipeline definitions
│   └── house_price_pipeline.py    # Main pipeline
├── docker/                        # Docker configuration
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
├── docs/                          # Documentation
│   ├── API.md                     # TFX architecture docs
│   └── Example.md                 # Project walkthrough
├── models/                        # Saved models (generated)
├── pipeline_outputs/              # TFX outputs (generated)
├── CLAUDE.md                      # Development guide
└── README.md                      # This file
```

---

## Development Phases

### ✅ Phase 1: Project Foundation (Complete)
- Folder structure and Docker setup
- Configuration management
- Utility module scaffolding
- Documentation templates

### 🔄 Phase 2: Data Ingestion & Validation (Next)
- Implement CsvExampleGen
- Generate schema with SchemaGen
- Data exploration and validation
- Handle missing values

### ⏳ Phase 3: Feature Engineering
- Implement Transform component
- Missing value imputation
- Feature scaling and encoding
- Create derived features

### ⏳ Phase 4: Model Training
- Train XGBoost model
- Train TensorFlow DNN model
- Hyperparameter tuning
- Compare training metrics

### ⏳ Phase 5: Model Evaluation
- Calculate RMSE, R², MAE
- Cross-validation
- Model comparison
- Select best model

### ⏳ Phase 6: Model Deployment
- Implement Pusher component
- Deploy to TensorFlow Serving
- Generate test predictions
- Dockerize complete pipeline

### ⏳ Phase 7: Documentation & Polish
- Complete all documentation
- Final testing and validation
- Code cleanup and optimization

---

## TFX Pipeline Components

1. **CsvExampleGen** - Ingest CSV data → TFRecord format
2. **SchemaGen** - Generate and validate data schema
3. **Transform** - Feature engineering and preprocessing
4. **Trainer** - Train XGBoost and TensorFlow DNN models
5. **Evaluator** - Evaluate and compare model performance
6. **Pusher** - Deploy approved model to serving directory

---

## Key Features

### Feature Engineering
- Missing value imputation (numerical & categorical)
- Feature scaling (StandardScaler)
- Ordinal and one-hot encoding
- Derived features (Age, TotalSF, TotalBath, etc.)
- Log transformation of target variable

### Model Comparison
- **XGBoost:** Fast, handles missing values, feature importance
- **TensorFlow DNN:** Deep learning, flexible architecture, TFX integration
- Compare on: RMSE, R², MAE, training time, cross-validation stability

### Production Ready
- Dockerized environment
- TFX pipeline for reproducibility
- Version-controlled schemas
- TensorFlow Serving for deployment

---

## Configuration

Key settings in `utils/config.py`:

```python
# Pipeline
PIPELINE_NAME = "house_price_prediction_pipeline"
TARGET_COLUMN = "SalePrice"

# Paths
DATA_DIR = "./data"
PIPELINE_ROOT = "./pipeline_outputs"
MODELS_DIR = "./models"

# Model Parameters
XGBOOST_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 7,
    "learning_rate": 0.01,
    ...
}

TF_DNN_PARAMS = {
    "hidden_units": [128, 64, 32],
    "dropout_rate": 0.2,
    "learning_rate": 0.001,
    ...
}

# Evaluation Thresholds
RMSE_THRESHOLD = 30000  # Max acceptable RMSE
R2_THRESHOLD = 0.85     # Min acceptable R²
```

---

## Docker Usage

### Build and Run
```bash
# Build image
docker build -t house-price-tfx -f docker/Dockerfile .

# Run with docker-compose
docker-compose -f docker/docker-compose.yml up

# Enter container
docker exec -it house-price-tfx bash
```

### Container Features
- Jupyter Lab on port 8888
- TensorFlow Serving on port 8501
- Volume mounts for live code editing
- All dependencies pre-installed

---

## Documentation

- **[CLAUDE.md](./CLAUDE.md)** - Comprehensive development guide with phased plan
- **[docs/API.md](./docs/API.md)** - TFX pipeline architecture and component details
- **[docs/Example.md](./docs/Example.md)** - Project walkthrough and results
- **[notebooks/API.ipynb](./notebooks/API.ipynb)** - Interactive API documentation
- **[notebooks/Example.ipynb](./notebooks/Example.ipynb)** - Interactive project example

---

## Development Workflow

1. **Write logic in `utils/` module** - Keep all functions and logic here
2. **Call from notebooks** - Notebooks import and execute utils functions
3. **Document in markdown** - Detailed explanations in `docs/`
4. **Run scripts for automation** - Use `scripts/` for automated execution

**Example:**
```python
# In notebook cell
from utils.data_utils import load_data, explore_data

# Load data using utility function
train_df = load_data("train")

# Explore and display results
exploration = explore_data(train_df)
print(f"Dataset shape: {exploration['shape']}")
```

---

## Dataset Information

**Source:** Ames Housing Dataset (Iowa, USA)

**Features (80 total):**
- Property characteristics (square footage, rooms, bathrooms)
- Quality ratings (overall quality, kitchen quality, etc.)
- Location (neighborhood)
- Amenities (garage, pool, fireplace, deck)
- Age and condition (year built, year remodeled)
- Sale information (sale type, sale condition)

**Target:** SalePrice (continuous, likely right-skewed)

**Data Files:**
- `train.csv` - 1,460 samples with SalePrice
- `test.csv` - 1,459 samples without SalePrice (for predictions)
- `data_description.txt` - Detailed feature descriptions

---

## Success Metrics

- **Primary:** RMSE < $30,000 on validation set
- **Secondary:** R² > 0.85
- **Bonus:** Compare XGBoost vs TensorFlow DNN

---

## Troubleshooting

### Data not found
```bash
# Verify data directory
ls data/
# Should contain: train.csv, test.csv, data_description.txt
```

### Import errors
```bash
# Reinstall dependencies
pip install -r docker/requirements.txt
```

### Docker issues
```bash
# Check Docker is running
docker info

# Rebuild from scratch
docker-compose down
docker-compose build --no-cache
docker-compose up
```

### TFX errors
```bash
# Clear pipeline metadata
rm -rf pipeline_outputs/metadata/

# Clear pipeline artifacts
rm -rf pipeline_outputs/house_price_prediction_pipeline/
```

---

## Next Steps

**Current Status:** Phase 1 Complete ✓

**Next Phase:** Phase 2 - Data Ingestion & Validation

To proceed:
1. Run `python scripts/example.py` to test current setup
2. Implement CsvExampleGen in `pipelines/house_price_pipeline.py`
3. Implement SchemaGen for data validation
4. Update `Example.ipynb` with data exploration

---

## Contributing

This is a class project for MSML610 Fall 2025. Follow the phased development plan in CLAUDE.md.

---

## References

- [TFX Documentation](https://www.tensorflow.org/tfx)
- [Ames Housing Dataset Paper](http://jse.amstat.org/v19n3/decock.pdf)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)

---

## License

Educational project for MSML610 - Advanced Machine Learning (Fall 2025)

---

**Last Updated:** Phase 1 Complete
**Project Status:** Foundation Established, Ready for Phase 2
