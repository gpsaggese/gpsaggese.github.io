# Project: Financial Fraud Detection with LakeFS Version Control

## 1. Project Overview
This project implements an end-to-end MLOps pipeline for detecting fraudulent credit card transactions. Unlike traditional static notebooks, this application leverages **LakeFS** to treat data, models, and experimental results as versioned assets.

The core objective is to demonstrate **reproducibility** by linking specific dataset versions to model performance metrics using LakeFS branching strategies.

## 2. Architecture & Workflow
The application logic is contained in `LakeFS_Fraud.example.ipynb` and executes the following pipeline:

### Phase 1: Ingestion, Feature Engineering & Preprocessing
1.  **Ingestion**: The raw dataset (`creditcard.csv`) is loaded for processing.
2.  **Feature Engineering**: New predictive signals are extracted to improve model performance:
    * **Hour Extraction**: Converted raw seconds into `Hour_of_Day` (0-23) to capture circadian fraud patterns.
    * **Log Transformation**: Applied `Log(1+x)` to the `Amount` column to handle extreme skewness in transaction values.
3.  **Transformation**: The enhanced data is passed through a "Pro" pipeline:
    * **Stratified Splitting**: To maintain class ratios.
    * **SMOTE**: To handle the extreme class imbalance (0.17% fraud).
    * **Standard Scaling**: To normalize features (V1-V28, Hour, LogAmount) for algorithms like Neural Networks.
4.  **Versioning**: The resulting `train.csv` and `test.csv` are committed to the `main` branch, creating an immutable "Golden Record" for all experiments.

### Phase 2: The 8-Model Tournament
To maximize detection accuracy, the pipeline runs a comprehensive tournament comparing 8 different approaches. **Each experiment is executed on its own isolated LakeFS branch** to prevent metric pollution.

| Branch Name | Algorithm | Description |
| :--- | :--- | :--- |
| `exp-lr` | Logistic Regression | Linear baseline model. |
| `exp-rf` | Random Forest | Bagging ensemble (Parallel trees). |
| `exp-xgb` | XGBoost | Gradient Boosting (Sequential trees). |
| `exp-lgbm` | LightGBM | High-efficiency Gradient Boosting. |
| `exp-nn` | Neural Network | Deep learning MLP with Dropout. |
| `exp-ensemble` | Basic Ensemble | Soft Voting classifier combining all above models. |
| `exp-power_ensemble` | Power Ensemble | Voting classifier using only top-tier Tree models (RF, XGB, LGBM). |
| `exp-tuned_xgb` | Tuned XGBoost | XGBoost optimized via GridSearchCV (Hyperparameter Tuning). |

### Phase 3: Artifact Generation
For every experiment, the pipeline generates and uploads three visualization artifacts to the corresponding branch:
1.  **Confusion Matrix**: To visualize False Positives vs. False Negatives.
2.  **ROC Curve**: To assess trade-offs between sensitivity and specificity.
3.  **Precision-Recall Curve**: To evaluate performance on the minority (Fraud) class.

## 3. Key Results & Analysis
A dynamic leaderboard is generated at the end of the pipeline and saved to the `main` branch (`results/final_leaderboard.csv`).

**Performance Highlights:**

* **Top Performer**: The **Random Forest** model achieved the highest F1-Score (~0.85), proving highly effective at capturing the cyclical patterns in the `Hour` feature.
* **Ensemble Insight**: The **Power Ensemble** (0.82) outperformed the **Basic Ensemble** (0.78). This confirms our hypothesis that "Power Ensembles" (grouping only high-variance/high-performance models) yield better results than averaging every available model.
* **Neural Network**: The MLP achieved an F1 of ~0.59. While functional, it lagged behind tree-based methods, a common outcome in tabular data tasks where decision trees often generalize better than standard neural networks.

## 4. Conclusion
By versioning the data and the results (artifacts) in LakeFS, we ensure that every experimental result—whether a success or a failure—is traceable back to the exact code and data snapshot that produced it.