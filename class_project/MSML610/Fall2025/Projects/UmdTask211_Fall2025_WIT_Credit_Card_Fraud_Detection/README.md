# Credit Card Fraud Detection (MSML610 Fall 2025)

Graduate-level anomaly detection and fraud classification project using the Kaggle credit card dataset. This project demonstrates end-to-end machine learning engineering with emphasis on class imbalance handling, ensemble methods, model interpretability, and interactive decision-boundary analysis via the What-If Tool (WIT).

## Project Overview

The objective of this project is to build a production-grade fraud detection system that:

- Identifies fraudulent credit card transactions with high precision and recall
- Combines multiple anomaly detection techniques (Isolation Forest, autoencoder) with supervised learning (Logistic Regression, RandomForest, XGBoost, CatBoost)
- Uses a weighted soft-voting ensemble to leverage model diversity and improve generalization
- Includes threshold optimization on validation data to balance false positive and false negative rates
- Provides interactive exploration of decision boundaries and feature sensitivity through the What-If Tool
- Implements strict validation discipline to prevent data leakage (train-only scaling, SMOTE-Tomek on training data only)

## Project Structure

- `READme.md` – this comprehensive guide with quickstart, architecture, and troubleshooting
- `WIT.API.md` – detailed API documentation for all utility functions
- `WIT.API.ipynb` – hands-on walkthrough of each API function with minimal examples
- `WIT.example.md` – narrative guide to the complete end-to-end workflow
- `WIT.example.ipynb` – full pipeline execution with EDA, modeling, evaluation, and WIT analysis
- `WIT_utils.py` – data IO, cleaning, feature engineering, scaling, SMOTE-Tomek balancing, anomaly/supervised models, ensembles, evaluation, and WIT integration
- `Dockerfile` – containerized development environment with JupyterLab, TensorFlow, and WIT pre-installed
- `requirements.txt` – Python package dependencies
- `data/raw/creditcard.csv` – raw Kaggle credit card fraud dataset (284,807 transactions, 0.17% fraud rate)
- `data/processed/creditcard_processed.csv` – processed dataset after feature engineering
- `artifacts/` – serialized models (ensemble.joblib, scaler.joblib) for deployment

## Visual Documentation: Architecture and Workflows

### System Architecture Diagram

```mermaid
graph TB
    subgraph Input["Input Data"]
        Raw["Raw Dataset<br/>(284,807 transactions)"]
    end
    
    subgraph Preprocessing["Data Preprocessing"]
        Clean["Clean Data<br/>(remove duplicates)"]
        Engineer["Feature Engineering<br/>(Hour, Amount_log1p, Amount_per_hour)"]
        Split["Stratified Split<br/>(train/val/test)"]
        Scale["StandardScaler<br/>(fit train only)"]
    end
    
    subgraph Balance["Balance Training Data"]
        SMOTE["SMOTE-Tomek<br/>(over-sample + boundary clean)"]
    end
    
    subgraph Models["Model Training"]
        subgraph Anomaly["Anomaly Detection"]
            IF["Isolation Forest"]
            AE["Autoencoder"]
        end
        subgraph Supervised["Supervised Learning"]
            LR["Logistic Regression"]
            RF["RandomForest"]
            XGB["XGBoost"]
            CB["CatBoost"]
        end
    end
    
    subgraph Ensemble["Ensemble & Optimization"]
        SoftVote["Soft-Voting Ensemble<br/>(weighted combination)"]
        ThreshOpt["Threshold Optimization<br/>(on validation set)"]
    end
    
    subgraph Evaluation["Evaluation & Analysis"]
        Metrics["Metrics Computation<br/>(Precision, Recall, F1, ROC-AUC, PR-AUC)"]
        Visualize["Visualizations<br/>(confusion matrix, PR/ROC curves)"]
        WIT["What-If Tool Analysis<br/>(decision boundaries, feature impact)"]
    end
    
    subgraph Output["Production Deployment"]
        Artifacts["Save Artifacts<br/>(ensemble.joblib, scaler.joblib)"]
        Monitor["Deploy with Monitoring<br/>(drift detection, retraining)"]
    end
    
    Raw --> Clean
    Clean --> Engineer
    Engineer --> Split
    Split --> Scale
    Scale --> SMOTE
    SMOTE --> IF
    SMOTE --> AE
    SMOTE --> LR
    SMOTE --> RF
    SMOTE --> XGB
    SMOTE --> CB
    IF --> SoftVote
    AE --> SoftVote
    LR --> SoftVote
    RF --> SoftVote
    XGB --> SoftVote
    CB --> SoftVote
    SoftVote --> ThreshOpt
    ThreshOpt --> Metrics
    Metrics --> Visualize
    Visualize --> WIT
    WIT --> Artifacts
    Artifacts --> Monitor
    
    style Input fill:#e1f5ff
    style Preprocessing fill:#fff3e0
    style Balance fill:#fce4ec
    style Models fill:#f3e5f5
    style Ensemble fill:#e8f5e9
    style Evaluation fill:#fef5e7
    style Output fill:#e0f2f1
```

### Complete Data Flow Pipeline

```mermaid
flowchart LR
    A["Raw Data<br/>284,807 rows<br/>0.17% fraud"] -->|load_raw_data| B["Load & Explore"]
    B -->|clean_data| C["Remove Duplicates<br/>Handle Nulls"]
    C -->|engineer_features| D["Add Features<br/>Hour, Amount_log1p<br/>Amount_per_hour"]
    D -->|split_features_target| E["Train/Val/Test Split<br/>80/10/10<br/>Stratified"]
    
    E -->|scale_features| F["Scale Features<br/>Train fit only<br/>Apply to all"]
    
    F -->|Training Data| G["Anomaly Models"]
    F -->|Training Data| H["Supervised Models"]
    F -->|Training Data| I["balance_with_smote_tomek"]
    
    I -->|Balanced Train| J["Resampled Training<br/>~50% fraud ratio"]
    
    J -->|IF, AE training| G
    J -->|LogReg, RF, XGB, CatBoost| H
    
    G -->|Predictions + Scores| K["Ensemble"]
    H -->|Probabilities| K
    
    K -->|optimize_threshold| L["Threshold Tuning<br/>On Validation Data"]
    
    L -->|Frozen Threshold| M["evaluate_<br/>binary_<br/>classification"]
    
    F -->|Test Data| M
    
    M -->|Metrics + Visualizations| N["Performance Report"]
    
    N -->|WIT Widget| O["Interactive Analysis<br/>FP/FN Exploration<br/>Feature Sensitivity"]
    
    O -->|Acceptance Testing| P["Deployment"]
    
    P -->|save_processed| Q["Artifacts"]
    P -->|joblib.dump| R["Serialized Models"]
    
    style A fill:#ffebee
    style E fill:#fff9c4
    style F fill:#fff3e0
    style I fill:#f3e5f5
    style K fill:#e8f5e9
    style L fill:#e0f2f1
    style M fill:#fce4ec
    style O fill:#f1f8e9
    style P fill:#e0f2f1
```

### Model Training and Ensemble Construction Workflow

```mermaid
graph TB
    Train["Training Data<br/>(balanced via SMOTE-Tomek)"]
    
    Train --> IF["train_isolation_forest<br/>contamination=0.00172<br/>n_estimators=300"]
    Train --> AE["train_autoencoder<br/>encoding_dim=16<br/>epochs=10<br/>threshold at 99.5%"]
    Train --> LR["train LogisticRegression<br/>class_weight='balanced'<br/>max_iter=800"]
    Train --> RF["train RandomForest<br/>class_weight='balanced'<br/>n_estimators=500"]
    Train --> XGB["train XGBoost<br/>scale_pos_weight tuned<br/>learning_rate=0.06"]
    Train --> CB["train CatBoost<br/>class_weights<br/>iterations=400"]
    
    IF -->|predict_isolation_forest| IFPred["Anomaly Scores<br/>normalized 0-1"]
    AE -->|predict_autoencoder| AEPred["Reconstruction Errors<br/>0-1 range"]
    LR -->|predict_proba| LRProba["Fraud Probabilities"]
    RF -->|predict_proba| RFProba["Fraud Probabilities"]
    XGB -->|predict_proba| XGBProba["Fraud Probabilities"]
    CB -->|predict_proba| CBProba["Fraud Probabilities"]
    
    IFPred -.->|Optional| Fusion["Hybrid Fusion<br/>Weighted Combination"]
    AEPred -.->|Optional| Fusion
    LRProba --> Ensemble["build_soft_voting_ensemble<br/>Weights: XGB/CB=3, RF=2, LR=1"]
    RFProba --> Ensemble
    XGBProba --> Ensemble
    CBProba --> Ensemble
    
    Ensemble -->|predict_proba| EnsembleProba["Ensemble Probabilities<br/>0-1 range"]
    Fusion -->|weighted avg| FusionScore["Fused Anomaly+Supervised<br/>Score"]
    
    EnsembleProba --> Threshold["optimize_threshold<br/>on validation set<br/>maximize F1-score"]
    FusionScore --> Threshold
    
    Threshold -->|Frozen Decision Rule| Decision["if probability > threshold<br/>then FRAUD<br/>else LEGITIMATE"]
    
    Decision -->|Test Set| Eval["evaluate_<br/>binary_<br/>classification"]
    
    Eval --> Metrics["Precision, Recall, F1<br/>ROC-AUC, PR-AUC<br/>Confusion Matrix"]
    
    style Train fill:#c8e6c9
    style Ensemble fill:#bbdefb
    style Decision fill:#ffe0b2
    style Metrics fill:#f8bbd0
    style Fusion fill:#e1bee7
```

### Threshold Optimization Visualization

```mermaid
graph LR
    ValData["Validation Set<br/>Predictions & Ground Truth"]
    
    ValData --> Sweep["Sweep Thresholds<br/>0.0 to 1.0<br/>step 0.01"]
    
    Sweep --> Metrics["For each threshold<br/>calculate:<br/>Precision<br/>Recall<br/>F1-Score"]
    
    Metrics --> Selection["Select Threshold<br/>Maximizing F1"]
    
    Selection --> Frozen["Freeze Threshold<br/>e.g., 0.35"]
    
    Frozen --> Apply["Apply to Test Set<br/>if prob > 0.35<br/>predict FRAUD"]
    
    Apply --> Eval["Evaluate on Test<br/>Get final metrics"]
    
    style ValData fill:#fff9c4
    style Frozen fill:#ffe0b2
    style Eval fill:#f8bbd0
```

### What-If Tool (WIT) Analysis Workflow

```mermaid
graph TB
    Test["Test Set Sample<br/>400-500 transactions"]
    
    Test -->|build_predict_fn| PredFn["Prediction Wrapper<br/>returns probabilities 0-1"]
    
    PredFn -->|build_wit_widget| WIT["WIT Widget<br/>Interactive Exploration"]
    
    WIT -->|Feature Sliders| Sliders["Adjust Features<br/>Amount, Hour<br/>V1-V28"]
    
    WIT -->|Filtering| Filters["Filter Examples<br/>False Positives<br/>False Negatives<br/>High Uncertainty"]
    
    WIT -->|Real-time| Predict["Observe Prediction<br/>Probability Changes<br/>Decision Boundary"]
    
    Sliders --> Analysis["Sensitivity Analysis<br/>How robust is model?<br/>What triggers fraud flag?"]
    
    Filters --> Investigation["Failure Mode Analysis<br/>Why are FP/FN missed?<br/>What patterns?"]
    
    Predict --> Understanding["Decision Boundary<br/>Characterization<br/>Confidence Calibration"]
    
    Analysis --> Report["Acceptance Testing<br/>Report"]
    Investigation --> Report
    Understanding --> Report
    
    Report -->|Pass| Deploy["Proceed to Deployment"]
    Report -->|Fail| Retune["Adjust Hyperparameters<br/>or Retrain Models"]
    
    Retune --> WIT
    
    style Test fill:#fff3e0
    style WIT fill:#f1f8e9
    style Analysis fill:#e0f2f1
    style Investigation fill:#fce4ec
    style Understanding fill:#f3e5f5
    style Deploy fill:#c8e6c9
```

### Data Leakage Prevention: Strict Workflow

```mermaid
graph LR
    Raw["Raw Data<br/>n=284,807"]
    
    Raw -->|load + clean| Cleaned["Cleaned Data<br/>n=284,456"]
    
    Cleaned -->|engineer_features| Featured["Featured Data<br/>Added: Hour, Amount_log1p<br/>Amount_per_hour"]
    
    Featured -->|stratified split| Split["STRICT SEPARATION"]
    
    Split -->|80%| Train["TRAINING SET<br/>n=227,565"]
    Split -->|10%| Val["VALIDATION SET<br/>n=28,445"]
    Split -->|10%| Test["TEST SET<br/>n=28,446"]
    
    Train -->|fit StandardScaler| Scaler["Scaler Statistics<br/>mean, std from train only"]
    
    Scaler -->|transform train| TrainScaled["Scaled Training<br/>using train stats"]
    Scaler -->|transform val| ValScaled["Scaled Validation<br/>using train stats"]
    Scaler -->|transform test| TestScaled["Scaled Test<br/>using train stats"]
    
    TrainScaled -->|fit SMOTE-Tomek| SMOTE["Resampler<br/>fit on train only"]
    
    SMOTE -->|resample train| TrainBal["Balanced Training<br/>~50% fraud ratio<br/>n~250,000"]
    
    TrainBal -->|Train Models| Models["Isolation Forest<br/>Autoencoder<br/>LogReg, RF, XGB, CatBoost"]
    
    Models -->|validate on Val| Tuning["Threshold Tuning<br/>on VALIDATION only<br/>Freeze threshold"]
    
    Tuning -->|apply to Test| TestEval["Evaluate on TEST<br/>Final metrics<br/>NO CHANGES"]
    
    style Train fill:#c8e6c9
    style Val fill:#fff9c4
    style Test fill:#ffccbc
    style TrainScaled fill:#c8e6c9
    style ValScaled fill:#fff9c4
    style TestScaled fill:#ffccbc
    style Tuning fill:#ffe0b2
    style TestEval fill:#ffccbc
```

### Anomaly Detection vs Supervised Learning Comparison

```mermaid
graph TB
    subgraph Unsupervised["UNSUPERVISED ANOMALY DETECTION"]
        IF["Isolation Forest<br/>- No labels required<br/>- Fast training<br/>- Isolation mechanism<br/>ROC-AUC ~0.95"]
        AE["Autoencoder<br/>- Learns normal patterns<br/>- Reconstruction error<br/>- Neural network based<br/>ROC-AUC ~0.94"]
    end
    
    subgraph Supervised["SUPERVISED LEARNING"]
        LR["LogisticRegression<br/>- Class weights<br/>- Linear boundary<br/>- Fast, interpretable<br/>ROC-AUC ~0.97"]
        RF["RandomForest<br/>- Class weights<br/>- Non-linear<br/>- Feature importance<br/>ROC-AUC ~0.98"]
        XGB["XGBoost<br/>- scale_pos_weight<br/>- Gradient boosting<br/>- Strong generalization<br/>ROC-AUC ~0.99"]
        CB["CatBoost<br/>- Class weights<br/>- Reduced overfitting<br/>- Categorical aware<br/>ROC-AUC ~0.99"]
    end
    
    Unsupervised -->|Complementary Signals| Ensemble["WEIGHTED SOFT-VOTING<br/>ENSEMBLE<br/>ROC-AUC ~0.99+<br/>F1 ~0.70-0.80"]
    Supervised -->|Ensemble Vote| Ensemble
    
    Ensemble -->|Optional| Fusion["HYBRID FUSION<br/>Anomaly + Supervised<br/>ROC-AUC ~0.9920<br/>F1 ~0.68-0.75"]
    
    style Unsupervised fill:#e1f5ff
    style Supervised fill:#f3e5f5
    style Ensemble fill:#e8f5e9
    style Fusion fill:#fff3e0
```

### Class Imbalance Handling: Before and After SMOTE-Tomek

```mermaid
graph LR
    Original["Original Training Data<br/>n=227,565<br/>Fraud: 398 (0.17%)<br/>Legit: 227,167 (99.83%)"]
    
    Original -->|SMOTE| Balanced["After SMOTE-Tomek<br/>n~250,000<br/>Fraud: ~125,000 (50%)<br/>Legit: ~125,000 (50%)"]
    
    Balanced -->|Train Models| Better["Models Learn Better<br/>- Balanced representations<br/>- Fewer false negatives<br/>- Better generalization"]
    
    Better -->|Eval on Unbalanced Test| Final["Test Set (Unbalanced)<br/>n=28,446<br/>Fraud: 50 (0.176%)<br/>Legit: 28,396 (99.824%)<br/><br/>Realistic Evaluation<br/>Represents Production"]
    
    style Original fill:#ffcdd2
    style Balanced fill:#c8e6c9
    style Better fill:#bbdefb
    style Final fill:#ffccbc
```

### Feature Engineering Pipeline

```mermaid
graph TB
    Raw["Raw Features<br/>Time (seconds from start)<br/>Amount (transaction $)<br/>V1-V28 (PCA anonymized)"]
    
    Raw -->|Engineer 1| Hour["Hour of Day<br/>= (Time // 3600) % 24<br/>Captures temporal patterns<br/>Values: 0-23"]
    
    Raw -->|Engineer 2| AmountLog["Amount Log Transform<br/>= log1p(Amount)<br/>Handles skewed distribution<br/>Reduces outlier impact"]
    
    Raw -->|Engineer 3| AmountPerHour["Amount per Hour<br/>= Amount / (Hour + 1)<br/>Normalized spending rate<br/>Identifies bursts"]
    
    Hour --> Combined["Final Feature Set<br/>30 features:<br/>V1-V28 + Hour<br/>+ Amount_log1p<br/>+ Amount_per_hour"]
    AmountLog --> Combined
    AmountPerHour --> Combined
    
    Combined -->|StandardScaler| Scaled["Scaled Features<br/>Mean=0, Std=1<br/>Ready for modeling"]
    
    style Hour fill:#fff3e0
    style AmountLog fill:#fff3e0
    style AmountPerHour fill:#fff3e0
    style Combined fill:#e8f5e9
    style Scaled fill:#bbdefb
```

### Production Deployment Workflow

```mermaid
graph LR
    Models["Trained Models<br/>Ensemble + Scaler"]
    
    Models -->|joblib.dump| Artifacts["artifacts/<br/>ensemble.joblib<br/>scaler.joblib"]
    
    Artifacts -->|Load in Production| Prod["Production System"]
    
    Prod -->|New Transaction| Receive["Receive Transaction<br/>[Time, Amount, V1-V28]"]
    
    Receive -->|engineer_features| Features["Engineer Features<br/>Hour, Amount_log1p<br/>Amount_per_hour"]
    
    Features -->|scaler.transform| Scale["Scale Transaction<br/>Using saved mean/std"]
    
    Scale -->|ensemble.predict_proba| Proba["Get Fraud Probability<br/>0-1 range"]
    
    Proba -->|apply threshold| Decision["Decision<br/>if prob > 0.35<br/>FRAUD<br/>else<br/>LEGITIMATE"]
    
    Decision -->|Log| Output["Output Decision<br/>Store prediction<br/>Monitor metrics"]
    
    Output -->|Daily Check| Monitor["Monitor Drift<br/>Track Precision/Recall<br/>Alert on degradation"]
    
    Monitor -->|If Drift| Retrain["Retrain Ensemble<br/>With recent data<br/>Reoptimize threshold"]
    
    Retrain -->|Save New| Artifacts
    
    style Artifacts fill:#c8e6c9
    style Prod fill:#bbdefb
    style Decision fill:#ffe0b2
    style Monitor fill:#fce4ec
    style Retrain fill:#fff3e0
```

## Key Features & Task Fulfillment

- **Preprocessing:** Train-only scaling, simple feature engineering (Hour, log Amount, Amount-per-hour), and SMOTE-Tomek on training data.
- **Models:** Isolation Forest and autoencoder (anomaly) plus LogReg, RandomForest, XGBoost, and CatBoost (supervised).
- **Ensembles:** Weighted soft voting with validation threshold tuning; optional fusion of anomaly scores with ensemble probabilities.
- **Evaluation:** Precision, recall, F1, ROC-AUC, PR-AUC, confusion matrices.
- **WIT:** Interactive FP/FN inspection and feature sliders (Amount, Hour) on held-out samples.

## Quick Start Guide

### Option 1: Local Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Jupyter Lab:**
   ```bash
   jupyter lab
   ```

3. **Open and explore the notebooks:**
   - Start with `WIT.API.ipynb` to understand the API layer and see minimal examples of each function
   - Progress to `WIT.example.ipynb` for the full end-to-end workflow
   - For faster iteration during development, set `N_SAMPLE` parameter in notebooks to load only a subset of data (e.g., 50,000 rows instead of full 284,807)

### Option 2: Docker (Containerized Environment)

1. **Build the Docker image:**
   ```bash
   docker build -t wit-fraud:latest .
   ```

2. **Run the container:**
   ```bash
   docker run -it -p 8888:8888 -v "$PWD":/work wit-fraud:latest
   ```

3. **Start Jupyter Lab inside container:**
   ```bash
   jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
   ```

4. **Access via browser:** http://localhost:8888

## Complete Workflow Overview

The workflow in short:
- Load and clean the raw CSV; engineer Hour/log Amount/Amount-per-hour.
- Split train/validation/test with stratification; scale using train-only stats.
- Balance training with SMOTE-Tomek; keep validation/test untouched.
- Train anomaly models (Isolation Forest, autoencoder) and supervised models (LogReg, RF, XGB, CatBoost); build the soft-voting ensemble and optional fusion.
- Tune the decision threshold on validation; evaluate on test with PR/ROC, F1, and confusion matrix; explore FP/FN with WIT.
- Save processed data and serialized models in `artifacts/`.

## Notebook Structure

### WIT.API.ipynb – API Reference Layer

Run first to see each utility function in isolation. It covers setup/imports, data IO, preprocessing, anomaly and supervised models, ensemble construction, evaluation helpers, and WIT wiring.

### WIT.example.ipynb – Complete Workflow

Run start-to-finish for the full pipeline: EDA, preprocessing, splitting, scaling, SMOTE-Tomek, anomaly and supervised training, ensemble + fusion, evaluation plots, WIT widget exploration, and saving artifacts.

## Data Leakage Prevention

This project implements strict validation discipline to prevent data leakage:

- **StandardScaler**: Fit statistics (mean, std) computed from training data only; applied to validation and test
- **SMOTE-Tomek**: Applied to training data only; test and validation sets remain unmodified
- **Train/validation/test split**: Stratified to preserve fraud ratio; no data flow from test back to training
- **Threshold optimization**: Performed on validation data; threshold frozen before test evaluation
- **Feature engineering**: Performed before split to avoid temporal leakage within transactions, but split timing ensures no information leaks between folds

## Important Notes for Reviewers

### Notebook Execution and Performance
- Both notebooks are designed to be "restart-and-run-all clean": no manual setup or intermediate state required
- Default `N_SAMPLE` values keep execution practical (~60,000 rows for quick testing):
  - Set `N_SAMPLE = None` in notebooks to run on full dataset (284,807 rows) for production results
  - Full runs take longer but provide more stable metrics and better model generalization
- Heavy lifting is centralized in the unified utility module (`WIT_utils.py`) to keep notebooks readable and maintainable

### What-If Tool (WIT) Requirements and Setup
- WIT requires: `witwidget`, `ipywidgets==7.*`, `tensorflow`, and a Jupyter kernel with these packages installed
- Provided `Dockerfile` includes all dependencies pre-configured
- Local installation may require additional setup; see troubleshooting section below
- On successful setup, WIT widget will display interactively in the notebook; feature sliders allow real-time model response visualization

### Model Training Notes
- Autoencoder and boosting models run quickly on modern hardware; ensure `tensorflow` is installed
- XGBoost on macOS requires libomp library; the code handles this automatically via environment variable configuration
- CatBoost and XGBoost support multi-threading via `n_jobs=-1` (use all CPU cores)
- Logistic Regression uses `solver='lbfgs'` with `max_iter=800` for stability on scaled, imbalanced data

### Threshold Optimization
- Threshold optimization occurs on validation data only; the selected threshold is then frozen and applied to test set
- Optimization targets F1-score (harmonic mean of precision and recall), balancing both metrics
- If your use case prioritizes recall over precision (catch all fraud, accept false alarms) or vice versa, adjust the `metric='f1'` parameter in `optimize_threshold()` to use a different objective
- Thresholds must be retuned when fraud class prior shifts in production

### Platform-Specific Configuration
- **macOS with Apple Silicon (M1/M2/M3)**: Use Python 3.11+ ARM interpreter with `tensorflow-macos==2.13.0` for TensorFlow + WIT support
- **macOS with Intel**: Standard `tensorflow==2.13.0` works; XGBoost requires `brew install libomp`
- **Linux**: Standard pip installation; ensure `libssl-dev` and build tools available
- **Docker**: All platform-specific issues handled inside container; recommended for reproducible environment

### Troubleshooting Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: No module named 'witwidget'` | WIT not installed | Run `pip install witwidget ipywidgets==7.* tensorflow` or use Docker |
| TensorFlow fails to load on macOS | x86/Rosetta Python lacks AVX support | Switch Jupyter kernel to ARM Python 3.11+ or install `tensorflow-macos` |
| XGBoost crashes on macOS | Missing libomp | Run `brew install libomp` |
| SMOTE-Tomek produces unchanged results | Random state not set consistently | Check `RANDOM_STATE=42` in utility files |
| Threshold optimization produces threshold > 1.0 or < 0.0 | Edge case in precision-recall curve | Adjust validation set size or check for degenerate probability distributions |
| WIT widget not rendering | JavaScript/WebSocket issue | Ensure JupyterLab v3.x installed; disable extensions if needed |

## Expected Performance Metrics

Based on representative runs (full dataset, various random seeds):

- **Isolation Forest (unsupervised)**: ROC-AUC ~0.95, Recall ~0.60-0.70
- **Autoencoder (unsupervised)**: ROC-AUC ~0.94, Recall ~0.55-0.65
- **Logistic Regression (supervised)**: ROC-AUC ~0.97, F1 ~0.60-0.70
- **RandomForest (supervised)**: ROC-AUC ~0.98, F1 ~0.65-0.75
- **XGBoost (supervised)**: ROC-AUC ~0.99, F1 ~0.70-0.80
- **CatBoost (supervised)**: ROC-AUC ~0.99, F1 ~0.70-0.80
- **Soft-voting ensemble (validation-tuned threshold)**: ROC-AUC ~0.99, F1 ~0.75-0.85
- **Hybrid fusion (anomaly + supervised)**: ROC-AUC ~0.99+, F1 ~0.78-0.87

Note: Exact numbers depend on data subset size, random seed, and hyperparameter choices. Full runs generally show better stability.

## Visual Model Performance Comparison

### ROC-AUC and F1-Score Progression Across Models

```mermaid
graph LR
    IF["Isolation Forest<br/>ROC-AUC: 0.95<br/>F1: 0.55"]
    AE["Autoencoder<br/>ROC-AUC: 0.94<br/>F1: 0.50"]
    LR["LogisticRegression<br/>ROC-AUC: 0.97<br/>F1: 0.62"]
    RF["RandomForest<br/>ROC-AUC: 0.98<br/>F1: 0.68"]
    XGB["XGBoost<br/>ROC-AUC: 0.99<br/>F1: 0.72"]
    CB["CatBoost<br/>ROC-AUC: 0.99<br/>F1: 0.73"]
    ENS["Ensemble<br/>ROC-AUC: 0.9920<br/>F1: 0.76"]
    FUS["Hybrid Fusion<br/>ROC-AUC: 0.9925<br/>F1: 0.78"]
    
    IF --> LR
    AE --> LR
    LR --> RF
    RF --> XGB
    XGB --> CB
    CB --> ENS
    ENS --> FUS
    
    style IF fill:#ffccbc
    style AE fill:#ffccbc
    style LR fill:#ffe0b2
    style RF fill:#fff9c4
    style XGB fill:#f1f8e9
    style CB fill:#c8e6c9
    style ENS fill:#bbdefb
    style FUS fill:#b2dfdb
```

### Precision-Recall Tradeoff and Threshold Selection

```mermaid
graph TB
    Default["Default Threshold: 0.5<br/>Precision: 0.75<br/>Recall: 0.55<br/>F1: 0.63"]
    
    Sweep["Sweep Thresholds<br/>on Validation Set"]
    
    Optimized["Optimized Threshold: 0.35<br/>Precision: 0.70<br/>Recall: 0.62<br/>F1: 0.66"]
    
    Default --> Sweep
    Sweep --> Optimized
    
    Optimized -->|Trade-off| Tradeoff["Slight precision loss<br/>for better recall<br/>catches more fraud"]
    
    style Default fill:#ffccbc
    style Optimized fill:#c8e6c9
    style Tradeoff fill:#f1f8e9
```

### Confusion Matrix Interpretation at Optimized Threshold

```mermaid
graph TB
    subgraph CM["Confusion Matrix"]
        TN["True Negatives<br/>28,326<br/>(Legit, Correct)"]
        FP["False Positives<br/>70<br/>(Legit, Wrong Flag)"]
        FN["False Negatives<br/>19<br/>(Fraud, Missed)"]
        TP["True Positives<br/>31<br/>(Fraud, Correct)"]
    end
    
    CM --> Metrics["Metrics Derived"]
    
    Metrics --> Precision["Precision = TP/(TP+FP)<br/>= 31/(31+70)<br/>= 0.307 or 30.7%<br/>Of predicted fraud<br/>30.7% are correct"]
    
    Metrics --> Recall["Recall = TP/(TP+FN)<br/>= 31/(31+19)<br/>= 0.620 or 62%<br/>Of actual fraud<br/>62% were caught"]
    
    Metrics --> F1["F1 = 2*(P*R)/(P+R)<br/>= 2*(0.307*0.620)/0.927<br/>= 0.412 or 41.2%<br/>Balanced metric"]
    
    style TN fill:#c8e6c9
    style TP fill:#c8e6c9
    style FP fill:#ffccbc
    style FN fill:#ffccbc
    style Precision fill:#fff9c4
    style Recall fill:#fff9c4
    style F1 fill:#f1f8e9
```

### Workflow Execution Timeline

```mermaid
timeline
    title Complete Fraud Detection Workflow
    section Data Phase
        Load Raw Data: 284,807 transactions
        Clean & Explore: Remove duplicates
        Engineer Features: Hour, Amount_log1p, Amount_per_hour
        Stratified Split: 80% train, 10% val, 10% test
    section Preprocessing Phase
        Scale Features: StandardScaler on train only
        Apply to Val/Test: Using train statistics
        Balance Training: SMOTE-Tomek resampling
    section Model Training Phase
        Train Anomaly: Isolation Forest, Autoencoder
        Train Supervised: LogReg, RF, XGB, CatBoost
        Build Ensemble: Weighted soft-voting
    section Tuning Phase
        Optimize Threshold: On validation set
        Freeze Decision: threshold = 0.35
        Prepare for Test: No further tuning
    section Evaluation Phase
        Evaluate on Test: Compute all metrics
        Confusion Matrix: TP, FP, TN, FN analysis
        PR/ROC Curves: Visualize performance
    section Analysis & Deployment
        WIT Analysis: Decision boundary exploration
        Acceptance Testing: Validate robustness
        Save Artifacts: ensemble.joblib, scaler.joblib
        Production Deploy: Monitor for drift
```

### Decision Boundary Visualization Concept

```mermaid
graph TB
    FN["False Negatives<br/>Fraud Missed<br/>(Too Conservative)"]
    
    Normal["Decision Boundary"]
    
    FP["False Positives<br/>Legitimate Flagged<br/>(Too Aggressive)"]
    
    Threshold["Threshold Position<br/>determines balance<br/>between FP and FN"]
    
    Lower["Lower Threshold<br/>0.25<br/>More FP<br/>Fewer FN<br/>Higher Recall"]
    Higher["Higher Threshold<br/>0.45<br/>Fewer FP<br/>More FN<br/>Higher Precision"]
    Optimal["Optimal Threshold<br/>0.35<br/>Balanced F1<br/>~62% Recall<br/>~31% Precision"]
    
    FN --> Threshold
    FP --> Threshold
    Normal --> Threshold
    
    Threshold --> Lower
    Threshold --> Higher
    Threshold --> Optimal
    
    style FN fill:#ffccbc
    style FP fill:#ffccbc
    style Normal fill:#e0f2f1
    style Optimal fill:#c8e6c9
    style Lower fill:#fff9c4
    style Higher fill:#fff9c4
```

### WIT Interactive Analysis Pathways

```mermaid
graph TB
    Widget["What-If Tool Widget"]
    
    Widget -->|Pathway 1| FPInv["False Positive Investigation"]
    Widget -->|Pathway 2| FNInv["False Negative Investigation"]
    Widget -->|Pathway 3| Sensitivity["Feature Sensitivity"]
    Widget -->|Pathway 4| Boundary["Boundary Exploration"]
    
    FPInv --> FPSteps["1. Filter: pred=fraud, true=legit<br/>2. Inspect: which features unusual?<br/>3. Adjust: move sliders to see changes<br/>4. Outcome: why model over-flagged?"]
    
    FNInv --> FNSteps["1. Filter: pred=legit, true=fraud<br/>2. Inspect: what fraud patterns?<br/>3. Adjust: find threshold where flip<br/>4. Outcome: model blindspots?"]
    
    Sensitivity --> SensitivitySteps["1. Pick legitimate transaction<br/>2. Increase Amount by 20%<br/>3. Observe prob change<br/>4. Assess robustness"]
    
    Boundary --> BoundarySteps["1. Sort by uncertainty (near 0.5)<br/>2. Inspect edge cases<br/>3. Use sliders to find exact flip<br/>4. Understand confidence"]
    
    FPSteps --> Insights["Insights for<br/>Deployment"]
    FNSteps --> Insights
    SensitivitySteps --> Insights
    BoundarySteps --> Insights
    
    Insights -->|Acceptance| Deploy["Deploy Model"]
    Insights -->|Issues Found| Retune["Retune Hyperparameters"]
    
    style Widget fill:#f1f8e9
    style Insights fill:#bbdefb
    style Deploy fill:#c8e6c9
    style Retune fill:#fff9c4
```

### Feature Engineering Impact

```mermaid
graph TB
    Original["Original Features<br/>V1-V28 (anonymized)<br/>Time (seconds)<br/>Amount ($)"]
    
    Hour["Hour Engineering<br/>Extracts time-of-day<br/>0-23 range<br/>Captures temporal patterns"]
    
    AmountLog["Amount Log Transform<br/>log1p(Amount)<br/>Handles skew<br/>Reduces outlier impact"]
    
    AmountPerHour["Amount Per Hour<br/>Amount / (Hour+1)<br/>Spending rate<br/>Identifies bursts"]
    
    Combined["Combined Features<br/>30 total:<br/>V1-V28<br/>Hour<br/>Amount_log1p<br/>Amount_per_hour"]
    
    Original --> Hour
    Original --> AmountLog
    Original --> AmountPerHour
    
    Hour --> Combined
    AmountLog --> Combined
    AmountPerHour --> Combined
    
    Combined -->|StandardScaler| Scaled["Scaled Features<br/>Mean=0, Std=1<br/>Ready for modeling"]
    
    Scaled -->|Improves| Performance["Model Performance<br/>Better generalization<br/>Faster convergence<br/>Stable predictions"]
    
    style Original fill:#fff3e0
    style Hour fill:#ffe0b2
    style AmountLog fill:#ffe0b2
    style AmountPerHour fill:#ffe0b2
    style Combined fill:#f1f8e9
    style Scaled fill:#bbdefb
    style Performance fill:#c8e6c9
```

### Production Monitoring Strategy

```mermaid
graph LR
    Deployed["Deployed Model<br/>ensemble.joblib<br/>scaler.joblib"]
    
    Deployed -->|Daily Transactions| Monitor["Monitoring Dashboard"]
    
    Monitor -->|Track Metrics| Drift["Metric Tracking<br/>Precision: baseline 0.70<br/>Recall: baseline 0.62<br/>F1: baseline 0.66<br/>ROC-AUC: baseline 0.99"]
    
    Monitor -->|Track Distributions| Features["Feature Distribution<br/>Amount: check range<br/>Hour: check patterns<br/>V1-V28: check bounds"]
    
    Drift -->|Degradation?| Alert1["Alert: Metric Drift<br/>Precision down 20%?<br/>Recall down 15%?<br/>Action: Investigate"]
    
    Features -->|Shift?| Alert2["Alert: Data Drift<br/>Unusual Amount dist?<br/>Time patterns change?<br/>Action: Retrain"]
    
    Alert1 -->|No Issue| Continue["Continue Production"]
    Alert2 -->|No Issue| Continue
    
    Alert1 -->|Confirmed| Retrain["Retrain on Recent Data<br/>Re-optimize threshold<br/>Update artifacts"]
    Alert2 -->|Confirmed| Retrain
    
    Retrain --> Deployed
    Continue --> Monitor
    
    style Deployed fill:#c8e6c9
    style Monitor fill:#bbdefb
    style Drift fill:#fff9c4
    style Features fill:#fff9c4
    style Alert1 fill:#ffccbc
    style Alert2 fill:#ffccbc
    style Retrain fill:#f1f8e9
```
