# House Price Prediction with TFX - Project Summary

## Project Overview

**Objective:** Build a production-ready machine learning pipeline for house price prediction using TensorFlow Extended (TFX), comparing multiple regression models to achieve optimal performance.

**Dataset:** Ames Housing Dataset (1,460 training samples, 80 features)
- Source: [Kaggle House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- Training data: 1,460 samples with 80 features
- Test data: 1,459 samples for predictions

**Best Model:** StackingEnsemble (CV RMSE: 0.1271)

**Status:** Complete - Model deployed and ready for serving

---

## System Overview

```mermaid
graph TB
    subgraph Input
        D1[Raw CSV Data<br/>train.csv<br/>1460 samples, 80 features]
    end

    subgraph TFX Pipeline
        T1[Data<br/>Ingestion] --> T2[Schema<br/>Validation]
        T2 --> T3[Feature<br/>Engineering<br/>77 features]
        T3 --> T4[Model<br/>Training<br/>8 models]
        T4 --> T5[Model<br/>Evaluation<br/>5-fold CV]
        T5 --> T6[Best Model<br/>Selection]
    end

    subgraph Model Comparison
        M1[XGBoost<br/>0.1300]
        M2[RandomForest<br/>0.1497]
        M3[GradientBoosting<br/>0.1273]
        M4[Ridge<br/>0.1410]
        M5[Lasso<br/>0.1423]
        M6[ElasticNet<br/>0.1411]
        M7[VotingEnsemble<br/>0.1301]
        M8[StackingEnsemble<br/>0.1271 BEST]
    end

    subgraph Deployment
        DEP[Production<br/>Model]
        VIZ[Visualizations<br/>7 plots]
        DOC[Documentation<br/>4 guides]
    end

    D1 --> T1
    T4 --> M1 & M2 & M3 & M4 & M5 & M6 & M7 & M8
    T6 --> M8
    M8 --> DEP
    T5 --> VIZ
    T6 --> DOC

    style D1 fill:#e1f5ff
    style M8 fill:#2ecc71
    style DEP fill:#c3fdb8
    style VIZ fill:#fff5ba
    style DOC fill:#fff5ba
    style T6 fill:#ffd700
```

---

## Key Achievements

### 1. Complete TFX Pipeline
- **ExampleGen:** CSV data ingestion
- **SchemaGen:** Automatic schema validation
- **Transform:** Feature engineering with 77 transformed features
- **Trainer:** Custom sklearn trainer for ensemble models
- **Evaluator:** Performance metrics and validation
- **Pusher:** Model deployment to serving directory

### 2. Comprehensive Model Comparison
Evaluated **8 regression models** with 5-fold cross-validation:
- XGBoost
- RandomForest
- GradientBoosting
- Ridge
- Lasso
- ElasticNet
- VotingEnsemble
- StackingEnsemble (Winner!)

### 3. Production-Ready Deployment
- Model saved in TensorFlow SavedModel format
- Compatible with TF Serving
- Sklearn pickle available for direct inference
- Comprehensive documentation and visualizations

### 4. Docker Containerization
- Reproducible environment
- Based on official TFX image
- All dependencies pre-installed
- Ready for cloud deployment

---

## Model Performance

### Best Model: StackingEnsemble

| Metric | Value | Interpretation |
|--------|-------|----------------|
| CV RMSE | 0.1271 ± 0.0135 | ~12.7% average error (log scale) |
| Train R² | 0.9808 | Explains 98% of variance |
| Train MAE | 0.0398 | Mean absolute error in log scale |
| Training Time | 136 seconds | 5-fold CV on 1,460 samples |

### Model Rankings (by CV RMSE)

1. **StackingEnsemble** - 0.1271 (Best)
2. GradientBoosting - 0.1273 (0.2% worse)
3. XGBoost - 0.1300 (2.3% worse)
4. VotingEnsemble - 0.1301 (2.4% worse)
5. Ridge - 0.1410 (10.9% worse)
6. ElasticNet - 0.1411 (11.0% worse)
7. Lasso - 0.1423 (12.0% worse)
8. RandomForest - 0.1497 (17.8% worse)

**Improvement:** StackingEnsemble is 15.1% better than the worst model (RandomForest)

### Model Comparison Workflow

```mermaid
graph TB
    subgraph Data Preparation
        A[Transformed Data<br/>1168 training samples<br/>77 features] --> B[5-Fold<br/>Cross-Validation]
    end

    subgraph Model Training
        B --> C1[XGBoost]
        B --> C2[RandomForest]
        B --> C3[GradientBoosting]
        B --> C4[Ridge]
        B --> C5[Lasso]
        B --> C6[ElasticNet]
        B --> C7[VotingEnsemble]
        B --> C8[StackingEnsemble]
    end

    subgraph Evaluation
        C1 --> D[Calculate Metrics<br/>RMSE, MAE, R²]
        C2 --> D
        C3 --> D
        C4 --> D
        C5 --> D
        C6 --> D
        C7 --> D
        C8 --> D
    end

    subgraph Selection
        D --> E[Compare CV RMSE]
        E --> F{Best Model?}
        F -->|Lowest RMSE| G[StackingEnsemble<br/>RMSE: 0.1271]
    end

    subgraph Deployment
        G --> H[Save Model]
        H --> I[TFX Pipeline]
        I --> J[models/serving/]
    end

    style A fill:#e1f5ff
    style G fill:#c3fdb8
    style J fill:#c3fdb8
    style F fill:#ffd700
```

### Model Performance Ranking

```mermaid
graph LR
    subgraph Top Performers
        A[1. StackingEnsemble<br/>0.1271]
        B[2. GradientBoosting<br/>0.1273]
        C[3. XGBoost<br/>0.1300]
    end

    subgraph Mid Performers
        D[4. VotingEnsemble<br/>0.1301]
        E[5. Ridge<br/>0.1410]
        F[6. ElasticNet<br/>0.1411]
    end

    subgraph Lower Performers
        G[7. Lasso<br/>0.1423]
        H[8. RandomForest<br/>0.1497]
    end

    style A fill:#2ecc71
    style B fill:#3498db
    style C fill:#3498db
    style D fill:#95a5a6
    style H fill:#e74c3c
```

---

## Technical Architecture

### TFX Pipeline Flow

```mermaid
graph TB
    A[Raw CSV Data<br/>train.csv] --> B[ExampleGen<br/>Data Ingestion]
    B --> C[SchemaGen<br/>Schema Validation]
    C --> D[Transform<br/>Feature Engineering]
    D --> E[Trainer<br/>Model Training]
    E --> F[Evaluator<br/>Model Evaluation]
    F --> G[Pusher<br/>Model Deployment]

    D -.->|Transform Graph| H[(Transform<br/>Artifacts)]
    E -.->|SavedModel| I[(Model<br/>Artifacts)]
    G -.->|Deployed Model| J[models/serving/]

    style A fill:#e1f5ff
    style G fill:#c3fdb8
    style J fill:#c3fdb8
    style H fill:#fff5ba
    style I fill:#fff5ba
```

### StackingEnsemble Architecture

```mermaid
graph TB
    subgraph Input
        X[Training Data<br/>1460 samples<br/>77 features]
    end

    subgraph Base Models
        X --> M1[XGBoost<br/>n_estimators=1000<br/>max_depth=7]
        X --> M2[RandomForest<br/>n_estimators=500<br/>max_depth=15]
        X --> M3[GradientBoosting<br/>n_estimators=1000<br/>lr=0.01]
        X --> M4[Ridge<br/>alpha=10.0]
    end

    subgraph Meta-Learner
        M1 --> P1[Base Predictions]
        M2 --> P1
        M3 --> P1
        M4 --> P1
        P1 --> ML[Ridge Meta-Model<br/>alpha=1.0]
    end

    subgraph Output
        ML --> Y[Final Prediction<br/>House Price]
    end

    style X fill:#e1f5ff
    style M1 fill:#ffcccc
    style M2 fill:#ccffcc
    style M3 fill:#ccccff
    style M4 fill:#ffccff
    style ML fill:#ffd700
    style Y fill:#c3fdb8
```

**Why Stacking Works:**
- Combines diverse model types (boosting, bagging, linear)
- Meta-learner learns optimal weighting of base models
- Reduces overfitting through ensemble averaging
- Each base model captures different data patterns

---

## Feature Engineering

### Transformation Pipeline (77 features)

```mermaid
graph LR
    subgraph Raw Data
        A[80 Features<br/>train.csv]
    end

    subgraph Preprocessing
        A --> B[Split by Type]
        B --> C[Numerical<br/>40 features]
        B --> D[Categorical<br/>40 features]

        C --> E[Impute Missing<br/>Median]
        E --> F[StandardScaler<br/>mean=0, std=1]
        F --> G[Derived Features<br/>TotalSF, Age, etc.]

        D --> H[Impute Missing<br/>Mode/NA]
        H --> I[One-Hot Encode<br/>Nominal]
        H --> J[Ordinal Encode<br/>Quality Ratings]
    end

    subgraph Target
        A --> K[SalePrice]
        K --> L[Log Transform<br/>log price + 1]
    end

    subgraph Output
        G --> M[77 Transformed<br/>Features]
        I --> M
        J --> M
        L --> N[SalePrice_log<br/>Target]
    end

    style A fill:#e1f5ff
    style M fill:#c3fdb8
    style N fill:#ffd700
```

**Transformation Details:**

**Numerical Features (40):**
- Scaling: StandardScaler (mean=0, std=1)
- Missing value imputation: Median for numerical
- Log transformation: Target variable (SalePrice)

**Categorical Features (37):**
- One-hot encoding for nominal categories
- Ordinal encoding for ordered categories (quality ratings)
- Vocabulary generation: Handles unseen categories

**Derived Features:**
- TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
- Age = YrSold - YearBuilt
- Remodeled = YearRemodAdd != YearBuilt
- TotalBath = FullBath + 0.5 × HalfBath

---

## Directory Structure

```
.
├── data/                      # Dataset files
│   ├── train.csv              # Training data (1460 rows)
│   ├── test.csv               # Test data (1459 rows)
│   └── data_description.txt   # Feature descriptions
│
├── utils/                     # Core implementation
│   ├── config.py              # Pipeline configuration
│   ├── feature_engineering.py # Transform preprocessing_fn
│   ├── model_utils.py         # TensorFlow DNN trainer
│   ├── sklearn_trainer.py     # Sklearn model trainer (used)
│   ├── model_comparison.py    # Model registry
│   └── evaluation_utils.py    # Evaluation metrics
│
├── pipelines/                 # TFX pipeline definition
│   └── house_price_pipeline.py
│
├── scripts/                   # Executable scripts
│   ├── api.py                 # Pipeline runner
│   ├── compare_models.py      # Model comparison CLI
│   ├── run_pipeline_with_best_model.py  # Auto-deploy best model
│   ├── visualize_results.py   # Generate visualizations
│   └── use_deployed_model.py  # Model usage guide
│
├── models/                    # Model outputs
│   ├── serving/               # Deployed models
│   │   └── <version>/         # Model version directory
│   │       ├── saved_model.pb # TensorFlow format
│   │       └── sklearn_model.pkl  # Sklearn pickle (48.6 MB)
│   └── comparison/            # Model comparison results
│       ├── comparison_results.json
│       └── best_model_StackingEnsemble.pkl
│
├── docs/                      # Documentation
│   ├── VISUALIZATION_GUIDE.md # This visualization guide
│   ├── PROJECT_SUMMARY.md     # This file
│   ├── MODEL_COMPARISON.md    # Detailed model analysis
│   ├── BEST_MODEL_DEPLOYMENT.md  # Deployment guide
│   └── visualizations/        # Generated plots (7 PNG files)
│
├── docker/                    # Docker configuration
│   ├── Dockerfile             # TFX 1.14.0 base
│   └── requirements.txt       # Python dependencies
│
└── pipeline_outputs/          # TFX artifacts
    └── house_price_prediction_pipeline/
        ├── CsvExampleGen/
        ├── SchemaGen/
        ├── Transform/
        ├── Trainer/
        ├── Evaluator/
        └── Pusher/
```

---

## Visualizations

All visualizations are in `docs/visualizations/`:

1. **summary_dashboard.png** - Comprehensive overview (recommended for presentations)
2. **cv_rmse_comparison.png** - Model performance comparison
3. **cv_score_distributions.png** - Score consistency across folds
4. **training_time_comparison.png** - Computational efficiency
5. **multi_metric_comparison.png** - RMSE, MAE, R² metrics
6. **cv_variability.png** - Model stability analysis
7. **performance_time_tradeoff.png** - Efficiency vs accuracy

**See:** `docs/VISUALIZATION_GUIDE.md` for detailed interpretation

---

## How to Use This Project

### 1. Build Docker Container

```bash
docker build -t house-price-tfx -f docker/Dockerfile .
```

### 2. Run Container

```bash
docker run -it --rm \
  -v ${PWD}/models:/app/models \
  -v ${PWD}/pipeline_outputs:/app/pipeline_outputs \
  house-price-tfx
```

### 3. Run Full Pipeline with Best Model

```bash
# Inside container
python scripts/run_pipeline_with_best_model.py --cv-folds 5
```

This will:
1. Compare all 8 models with 5-fold CV
2. Select the best model (StackingEnsemble)
3. Deploy it via TFX pipeline to `models/serving/`

### 4. Generate Visualizations

```bash
# Inside container
python scripts/visualize_results.py
```

### 5. Use Deployed Model

```python
# Load sklearn model
import pickle
with open('models/serving/<timestamp>/sklearn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions (requires preprocessed features)
predictions = model.predict(X_test)
import numpy as np
predictions = np.exp(predictions) - 1.0  # Convert from log scale
```

**See:** `scripts/use_deployed_model.py` for detailed examples

### Deployment Architecture

```mermaid
graph TB
    subgraph User Interface
        U[User/Client<br/>Application]
    end

    subgraph Docker Container
        subgraph TFX Pipeline
            P1[ExampleGen] --> P2[SchemaGen]
            P2 --> P3[Transform]
            P3 --> P4[Trainer]
            P4 --> P5[Evaluator]
            P5 --> P6[Pusher]
        end

        subgraph Model Serving
            M1[SavedModel<br/>TensorFlow Format]
            M2[Pickle Model<br/>Sklearn Format]
        end

        P6 --> M1
        P6 --> M2
    end

    subgraph Storage
        S1[(models/serving/)]
        S2[(pipeline_outputs/)]
        S3[(Transform Graph)]
    end

    M1 --> S1
    M2 --> S1
    P3 --> S3
    P6 --> S2

    U -->|Prediction Request| M2
    M2 -->|House Price| U

    style U fill:#e1f5ff
    style M1 fill:#c3fdb8
    style M2 fill:#c3fdb8
    style S1 fill:#fff5ba
    style P6 fill:#ffd700
```

### Data Flow for Prediction

```mermaid
sequenceDiagram
    participant User
    participant Model as StackingEnsemble
    participant Transform as Transform Graph
    participant Base1 as XGBoost
    participant Base2 as RandomForest
    participant Base3 as GradientBoosting
    participant Base4 as Ridge
    participant Meta as Meta-Learner

    User->>Transform: Raw Features (80)
    Transform->>Transform: Preprocess & Scale
    Transform->>Model: Transformed Features (77)

    Model->>Base1: Features
    Model->>Base2: Features
    Model->>Base3: Features
    Model->>Base4: Features

    Base1->>Meta: Prediction 1
    Base2->>Meta: Prediction 2
    Base3->>Meta: Prediction 3
    Base4->>Meta: Prediction 4

    Meta->>Meta: Weighted Combination
    Meta->>Model: Final Prediction (log scale)
    Model->>Model: exp(pred) - 1
    Model->>User: House Price ($)
```

---

## Key Documentation Files

| Document | Purpose |
|----------|---------|
| `README.md` | Project overview and quick start |
| `docs/VISUALIZATION_GUIDE.md` | Visualization interpretation guide |
| `docs/PROJECT_SUMMARY.md` | This file - complete project summary |
| `docs/MODEL_COMPARISON.md` | Detailed model comparison analysis |
| `docs/BEST_MODEL_DEPLOYMENT.md` | Deployment instructions |

---

## Dependencies

### Core Frameworks
- TensorFlow 2.13.0
- TensorFlow Extended (TFX) 1.14.0
- TensorFlow Transform

### ML Libraries
- XGBoost 2.0.3
- scikit-learn
- pandas
- numpy

### Visualization
- matplotlib
- seaborn

### Development
- Docker
- Jupyter
- PyYAML

---

## Performance Interpretation

### Understanding RMSE = 0.1271

Since SalePrice is log-transformed: `log(price + 1)`

**Example Predictions:**

| Actual Price | Predicted Error Range | Relative Error |
|--------------|----------------------|----------------|
| $100,000 | ±$13,500 | 13.5% |
| $200,000 | ±$27,000 | 13.5% |
| $400,000 | ±$54,000 | 13.5% |

**Interpretation:**
- Model typically predicts within **±13.5%** of actual price
- Error percentage is consistent across price ranges (benefit of log transform)
- Competitive with industry benchmarks for housing prediction

---

## Future Improvements

### Model Enhancement
1. **Hyperparameter tuning:** Use Optuna or Hyperopt for optimal params
2. **Feature selection:** Remove low-importance features
3. **Additional features:** Neighborhood economic indicators, school ratings
4. **Time-based CV:** Account for temporal trends in housing market

### Pipeline Optimization
1. **Parallel training:** Train base models concurrently
2. **Incremental training:** Update model with new data
3. **Model monitoring:** Track prediction drift
4. **A/B testing:** Compare StackingEnsemble vs XGBoost in production

### Deployment
1. **TF Serving setup:** Enable REST/gRPC API
2. **Kubernetes deployment:** Scale horizontally
3. **CI/CD pipeline:** Automated retraining and deployment
4. **Monitoring dashboard:** Real-time performance metrics

---

## Team & Contact

**Project:** UmdTask56_Fall2025_TFX_Predicting_House_Prices_with_Feature_Engineering

**Course:** MSML610 - Advanced Machine Learning

**Semester:** Fall 2025

**Institution:** University of Maryland

---

## Reproducibility

This project is fully reproducible:

1. **Docker-based:** Consistent environment across machines
2. **Version-controlled:** All code and configs in Git
3. **Deterministic:** Random seeds set (seed=42)
4. **Documented:** Step-by-step instructions provided

**To reproduce:**
```bash
git clone <repository>
cd <project-directory>
docker build -t house-price-tfx -f docker/Dockerfile .
docker run -it house-price-tfx
python scripts/run_pipeline_with_best_model.py --cv-folds 5
python scripts/visualize_results.py
```

---

## Acknowledgments

- **Dataset:** Ames Housing Dataset (Dean De Cock, 2011) - [Kaggle Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- **Framework:** TensorFlow Extended (Google)
- **Models:** XGBoost, scikit-learn
- **Tools:** Docker, matplotlib, seaborn

---

## License

This project is for educational purposes as part of MSML610 coursework.

---

## Visual Diagrams Index

This document includes **8 Mermaid diagrams** for comprehensive visualization:

### Architecture Diagrams
1. **System Overview** - Complete end-to-end pipeline view
2. **TFX Pipeline Flow** - Detailed component-level architecture
3. **StackingEnsemble Architecture** - Model structure with base models and meta-learner
4. **Feature Engineering Pipeline** - Data transformation workflow
5. **Deployment Architecture** - Production deployment structure

### Process Diagrams
6. **Model Comparison Workflow** - 8-model evaluation process
7. **Model Performance Ranking** - Visual ranking by RMSE
8. **Data Flow for Prediction** - Sequence diagram for inference

**Viewing Instructions:**
- Mermaid diagrams render automatically in GitHub, VS Code (with Markdown Preview Mermaid Support extension), and many markdown viewers
- For other viewers, copy diagram code to [Mermaid Live Editor](https://mermaid.live/)
