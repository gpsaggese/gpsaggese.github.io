# Credit Scoring Model with SHAP for Interpretability

**Author**: Aneesh Chepuri  
**Course**: MSML610 — Fall 2025  

---

## Overview

A production-ready credit scoring system that combines the predictive power of XGBoost with the transparency of SHAP explanations. This project demonstrates how banks can make accurate lending decisions while satisfying regulatory requirements for explainability and fairness.

Built on the German Credit dataset, the pipeline handles real-world challenges including class imbalance, hyperparameter tuning, probability calibration, and cost-sensitive decision thresholds. Every prediction comes with both global feature importance and borrower-specific explanations.

**Key Achievement**: High-performance model with full explainability, business-optimized thresholds, and calibrated probabilities suitable for regulatory capital planning.

---

## Project Goals

This project builds an end-to-end credit scoring model on the German Credit dataset and uses SHAP (SHapley Additive exPlanations) to interpret model predictions.

### Data Preparation
- Load the German Credit dataset from the UCI ML Repository
- Clean and preprocess data (handle missing values, encode categoricals, scale numericals)
- Split into train/test sets with stratification to maintain class balance

### Modeling
- Train baseline models for performance benchmarking
- Optimize with systematic hyperparameter search using cross-validation
- Handle class imbalance with appropriate weighting techniques
- Apply business logic for cost-sensitive decision thresholds
- Calibrate probabilities for reliable risk estimates

### Interpretability with SHAP
- Compute SHAP values for the optimized model
- Generate global explanations showing feature importance across all predictions
- Create local explanations for individual borrower decisions
- Perform sensitivity analysis to demonstrate feature impact

### Visualization and Reporting
- Generate comprehensive diagnostic plots (confusion matrices, ROC/PR curves, calibration plots)
- Create SHAP visualizations (summary plots, decision plots, dependence plots)
- Produce sensitivity analysis curves showing "what-if" scenarios
- Save all outputs for analysis and reporting

---

## Getting Started

### Prerequisites
- Python 3.8 or higher
- 2GB RAM minimum
---

## Setup Instructions (Local Development)

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/gpsaggese/umd_classes.git
   cd umd_classes/class_project/MSML610/Fall2025/Projects/TutorTask_28_Fall2025_SHAP_Credit_Scoring_Model_with_SHAP_for_Interpretability
   ```

2. **Create and Activate a Virtual Environment (Windows)**:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Create and Activate a Virtual Environment (macOS/Linux)**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run Jupyter Notebook Locally**:
   ```bash
   jupyter notebook
   ```
   - Open `SHAP_Credit.API.ipynb` in your browser to understand the APIs
   - Open `shap_example.ipynb` for the full end-to-end pipeline
   - Use **Restart & Run All** for reproducible results

---

## Setup Instructions (Docker)

Docker provides a fully reproducible environment with all dependencies pre-installed and is the recommended way to run this project.

### 1. Install Docker
Install Docker Desktop from: https://www.docker.com/products/docker-desktop/  
Verify installation: `docker --version`

### 2. Navigate to the Project Directory
```bash
`cd ~/umd_classes/class_project/MSML610/Fall2025/Projects/TutorTask_28_Fall2025_SHAP_Credit_Scoring_Model_with_SHAP_for_Interpretability`
```
### 3. Build the Docker Image
```bash
`docker build -t final_project_shap_credit_image .`
```  
The first build may take 10–20 minutes due to scientific Python packages. Subsequent builds are faster.

### 4. Start the Container and Launch Jupyter
Run:
```bash
`docker run --rm \
-p 8888:8888 \
-v $(pwd):/workspace \
-e PORT=8888 \
final_project_shap_credit_image \
bash /workspace/run_jupyter.sh`
```
### 5. Open Jupyter Notebook
Go to: http://localhost:8888

You should see:
- SHAP_Credit.API.ipynb (API-focused tutorial)
- shap_example.ipynb (end-to-end project notebook)
- credit_scoring_shap/ (reusable project package)

---

## Project Structure

```
TutorTask_28_Fall2025_SHAP_Credit_Scoring_Model_with_SHAP_for_Interpretability/
│
├── README.md                           # This file
├── SHAP_Credit.API.md                  # API documentation
├── SHAP_Credit.example.md              # Complete walkthrough with detailed metrics
├── SHAP_Credit.API.ipynb               # API tutorial notebook
├── shap_example.ipynb                  # Main pipeline notebook
│
├── credit_scoring_shap/                # Python package
│   ├── __init__.py
│   ├── config.py                       # Configuration with random seeds
│   ├── data.py                         # Data loading and preprocessing
│   ├── modeling.py                     # Model building and training
│   ├── evaluation.py                   # Metrics and visualization
│   ├── explain.py                      # SHAP analysis
│   └── sensitivity.py                  # Sensitivity analysis
│
├── reports/                            # Generated outputs
│   ├── baseline_lr_*.png               # Logistic regression plots
│   ├── baseline_xgb_*.png              # Baseline XGBoost plots
│   ├── tuned_xgb_*.png                 # Tuned model plots
│   ├── balanced_xgb_*.png              # Balanced model plots
│   ├── calibration_*.png               # Calibration plots
│   ├── shap_*.png                      # SHAP visualizations
│   ├── sensitivity_*.png               # Sensitivity curves
│   └── metrics_*.txt                   # Metric summaries
│
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # Docker configuration
├── docker_build.sh                     # Build Docker image
├── docker_bash.sh                      # Run Docker container
├── run_jupyter.sh                      # Start Jupyter server
│
├── bashrc                              # Shell config
├── etc_sudoers                         # Sudo configuration
├── install_common_packages.sh          # System dependencies
├── install_jupyter_extensions.sh       # Jupyter extensions
├── utils.sh                            # Helper utilities
└── version.sh                          # Version management
```

---

## Pipeline Workflow

```mermaid
flowchart TB

%% =========================
%% DATA PREPARATION
%% =========================
subgraph data["DATA PREPARATION"]
    direction TB
    A1["German Credit Dataset<br/>Cleaning, encoding,<br/>stratified train/test split"]
end

%% =========================
%% BASELINE MODELS
%% =========================
subgraph baseline["BASELINE MODELS"]
    direction TB
    B1["Logistic Regression<br/>XGBoost<br/>Baseline performance metrics"]
end

%% =========================
%% HYPERPARAMETER TUNING
%% =========================
subgraph tuning["HYPERPARAMETER OPTIMIZATION"]
    direction TB
    C1["GridSearchCV<br/>Cross-validation<br/>Best model selection"]
end

%% =========================
%% CLASS IMBALANCE
%% =========================
subgraph balance["CLASS IMBALANCE & THRESHOLD"]
    direction TB
    D1["Class weighting<br/>Cost-sensitive threshold<br/>optimization"]
end

%% =========================
%% CALIBRATION
%% =========================
subgraph calibration["PROBABILITY CALIBRATION"]
    direction TB
    E1["Isotonic regression<br/>Well-calibrated<br/>probability estimates"]
end

%% =========================
%% SHAP
%% =========================
subgraph shap["SHAP EXPLAINABILITY"]
    direction TB
    F1["Global feature importance<br/>Local explanations<br/>Sensitivity analysis"]
end

%% =========================
%% OUTPUTS
%% =========================
subgraph outputs["PROJECT OUTPUTS"]
    direction TB
    G1["Trained models<br/>Evaluation metrics<br/>SHAP visualizations<br/>Documentation"]
end

%% FLOW
data --> baseline --> tuning --> balance --> calibration --> shap --> outputs

%% =========================
%% STYLES
%% =========================
classDef dataStyle fill:#e1f5ff,stroke:#0288d1,stroke-width:2px,color:#000
classDef baselineStyle fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:#000
classDef tuningStyle fill:#ffecb3,stroke:#ffa726,stroke-width:2px,color:#000
classDef balanceStyle fill:#c8e6c9,stroke:#66bb6a,stroke-width:2px,color:#000
classDef calibrationStyle fill:#b2dfdb,stroke:#26a69a,stroke-width:2px,color:#000
classDef shapStyle fill:#f3e5f5,stroke:#ab47bc,stroke-width:2px,color:#000
classDef outputStyle fill:#c5cae9,stroke:#5c6bc0,stroke-width:2px,color:#000

class A1 dataStyle
class B1 baselineStyle
class C1 tuningStyle
class D1 balanceStyle
class E1 calibrationStyle
class F1 shapStyle
class G1 outputStyle

%% =========================
%% SIZE CONTROL (CRITICAL)
%% =========================
style A1 min-width:360px,padding:12px
style B1 min-width:360px,padding:12px
style C1 min-width:380px,padding:12px
style D1 min-width:360px,padding:12px
style E1 min-width:360px,padding:12px
style F1 min-width:360px,padding:12px
style G1 min-width:380px,padding:12px

```

---

## How It Works

The pipeline follows a seven-stage workflow from raw data to interpretable, cost-sensitive credit risk predictions.

### 1. Data Preparation
- Load the German Credit dataset with `ucimlrepo`
- Clean data, encode categoricals, standardize numerics
- Create stratified train/test splits to preserve class balance

### 2. Baseline Model Training
- Train Logistic Regression and XGBoost with default settings
- Evaluate using standard metrics and basic diagnostic plots
- Use results as a baseline for later improvements

### 3. Hyperparameter Optimization
- Define a grid over key model hyperparameters
- Run stratified cross-validated GridSearchCV
- Select and test the best configuration on the held-out test set

### 4. Class Imbalance Handling
- Inspect class distribution and compute class weights
- Train cost-sensitive models with weights/sample weights
- Analyze precision–recall trade-offs, especially for the minority class

### 5. Cost-Based Threshold Optimization
- Define business costs for false positives/negatives
- Sweep decision thresholds and compute expected cost
- Choose the threshold that minimizes total business cost

### 6. Probability Calibration
- Calibrate predicted probabilities (isotonic regression)
- Validate with Brier score and calibration plots
- Ensure probabilities are suitable for downstream decisions

### 7. SHAP Explainability & Sensitivity Analysis
- Compute SHAP values for global feature importance and interactions
- Generate local explanations and decision plots for individual borrowers
- Run sensitivity analysis by varying key features to see impact on risk

---

## Key Features

### Production-Ready ML Pipeline
- Baseline models for benchmarking
- Systematic hyperparameter tuning with cross-validation
- Class imbalance handling with sample weighting
- Cost-sensitive threshold optimization
- Probability calibration for reliable estimates

### Explainability with SHAP
- Global feature importance rankings
- Individual decision explanations with reason codes
- Sensitivity analysis for "what-if" scenarios
- Regulatory compliance support with audit trails

### Comprehensive Evaluation
- Multiple performance metrics (AUC, Precision-Recall, Brier Score)
- Rich visualization suite (confusion matrices, ROC/PR curves, calibration plots)
- Business impact analysis across different thresholds
- Framework for fairness testing and disparate impact analysis

### Risk Management Integration
- Calibrated probability estimates suitable for capital planning
- Stress testing capabilities via sensitivity analysis
- Model monitoring framework for drift detection
- Deployment guidelines and operational recommendations

---

## Usage

### Understanding the APIs
```bash
jupyter notebook SHAP_Credit.API.ipynb
```
This notebook demonstrates:
- XGBoost API for gradient boosted tree models
- SHAP API for model interpretation and explainability
- Project package structure and module organization
- Best practices for using the credit scoring pipeline

### Running the Full Pipeline
```bash
jupyter notebook shap_example.ipynb
```
The complete end-to-end pipeline:
- Loads and preprocesses German Credit dataset
- Trains baseline and optimized models
- Handles class imbalance systematically
- Optimizes decision thresholds for business costs
- Calibrates probabilities for reliable risk estimates
- Generates comprehensive SHAP explanations
- Performs sensitivity analysis on key features
- Saves all outputs to reports directory

### Best Practices
- Use **Restart & Run All** in Jupyter for fully reproducible results
- Review generated outputs in `reports/` directory after execution
- Start with API notebook if new to XGBoost or SHAP frameworks
- Consult `SHAP_Credit.example.md` for detailed metrics and analysis
- Check configuration in `credit_scoring_shap/config.py` before running

---

## What You'll Learn

### Machine Learning Fundamentals
- Baseline model establishment and performance benchmarking
- Hyperparameter optimization with grid search and cross-validation
- Strategies for handling imbalanced classification problems
- Advanced model evaluation beyond simple accuracy metrics
- Probability calibration methods for reliable predictions

### shap Techniques
- SHAP value theory and practical implementation
- Difference between global and local explanations
- Interpreting decision plots and waterfall charts
- Conducting sensitivity analysis for feature impact
- Translating technical explanations to business language

### Risk Management Applications
- Cost-sensitive threshold optimization for business objectives
- Expected loss calculations using probability estimates
- Stress testing approaches using feature-based scenarios
- Model monitoring strategies for production deployment
- Frameworks for regulatory compliance and fairness testing

### Software Engineering Practices
- Modular code architecture with clear separation of concerns
- Configuration-driven development for maintainability
- Reproducible experiments using random seed management
- Docker containerization for consistent environments
- Professional documentation and visualization standards

---

## Troubleshooting

### Port Already in Use
If port 8888 is occupied:
```bash
sudo lsof -i :8888
sudo kill -9 <PID>
```

### Module Not Found Error
Ensure you are running inside the Docker container started with the docker run command.
Inside the container, the project directory is mounted at: /workspace

### Dataset Download Issues
The dataset is automatically fetched from UCI repository via `ucimlrepo`. If download fails:
- Verify firewall settings allow outbound connections
- Manually download dataset and update `config.py` to use local file

### Reproducibility Issues
All random seeds are set to 42 throughout the project for reproducibility. If results differ across runs:
- Verify you are using the Docker environment (ensures consistent library versions)
- Check that notebooks initialize seeds at the beginning
- Confirm `random_state=42` is set in all relevant function calls
- For maximum reproducibility, use `n_jobs=1` in GridSearchCV (slower but deterministic)

### Visualization Problems
If plots do not display in Jupyter notebooks:
- Add `%matplotlib inline` magic command at the start of notebooks
- Verify matplotlib and seaborn are properly installed
- Check that `reports/` directory exists and has write permissions
- Update visualization libraries: `pip install --upgrade matplotlib seaborn`

### Memory Issues
If encountering out-of-memory errors:
- Reduce parameter grid size for GridSearchCV
- Use smaller sample for initial testing
- Increase Docker memory allocation in Docker Desktop settings
- Consider using incremental learning approaches for large datasets

---

## References

### Core Libraries
- [XGBoost Documentation](https://xgboost.readthedocs.io/) - Gradient boosting framework
- [SHAP Documentation](https://shap.readthedocs.io/) - Model interpretation library
- [scikit-learn Documentation](https://scikit-learn.org/stable/) - Machine learning toolkit
- [ucimlrepo Documentation](https://pypi.org/project/ucimlrepo/) - UCI dataset access

### Dataset
- [German Credit (Statlog) – UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)

### Research Papers
- Chen, T., & Guestrin, C. (2016). [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754). Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
- Lundberg, S. M., & Lee, S. I. (2017). [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874). Advances in Neural Information Processing Systems.

### Key APIs and Methods
- [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) - Hyperparameter optimization
- [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) - Data splitting
- [CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html) - Probability calibration
- [ROC AUC Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) - Model evaluation
- [Confusion Matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) - Error analysis

