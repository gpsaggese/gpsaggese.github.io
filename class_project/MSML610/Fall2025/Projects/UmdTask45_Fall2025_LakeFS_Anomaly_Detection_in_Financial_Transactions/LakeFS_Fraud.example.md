# Project: Financial Fraud Detection with LakeFS Version Control

## 1. Project Overview
This project implements an end-to-end MLOps pipeline for detecting fraudulent credit card transactions. Unlike traditional static notebooks, this application leverages **LakeFS** to treat data, models, and experimental results as versioned assets.

The core objective is to demonstrate **reproducibility** by linking specific dataset versions to model performance metrics using LakeFS branching strategies.

## 2. Architecture & Workflow
The application logic is contained in `LakeFS_Fraud.example.ipynb` and executes the following pipeline:

### Phase 1: Ingestion & Preprocessing
1.  **Ingestion**: The raw dataset (`creditcard.csv`) is loaded and uploaded to the `main` branch of the LakeFS repository.
2.  **Transformation**: The data is processed using a "Pro" pipeline:
    * **Stratified Splitting**: To maintain class ratios.
    * **SMOTE**: To handle the extreme class imbalance (0.17% fraud).
    * **Standard Scaling**: To normalize features for algorithms like Neural Networks and Logistic Regression.
3.  **Versioning**: The resulting `train.csv` and `test.csv` are committed to the `main` branch, creating an immutable baseline.

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
* **Winner**: The standard **XGBoost** model achieved the highest F1-Score (~0.83).
* **Ensemble Insight**: 
    * The **Power Ensemble** (0.81) outperformed the **Basic Ensemble** (0.79). This demonstrates that removing "weak learners" (like Logistic Regression) improves the overall voting quality.
    * However, single tree models (XGBoost, RF) still slightly outperformed the ensemble, suggesting that the diversity of errors between the models was not sufficient to boost the score further.
* **Tuning Insight**: The **Tuned XGBoost** (0.78) performed slightly worse than the default XGBoost (0.83). This indicates that the Grid Search range may have introduced slight overfitting to the training set, whereas the default XGBoost parameters are highly robust for this specific dataset.

## 4. Conclusion
By versioning the data and the results (artifacts) in LakeFS, we ensure that every experimental result—whether a success or a failure—is traceable back to the exact code and data snapshot that produced it.