# Anomaly Detection in Network Traffic Data  
### MSML610 - Fall 2025 Final Project

This repository contains the full pipeline, code, and results for network intrusion/anomaly detection using the **UNSW-NB15 dataset**.  
We implemented both **supervised** and **unsupervised** approaches, built a reusable preprocessing API, performed feature selection, and compared multiple ML models using ROC-AUC & PR-AUC.

The project includes:
- A full machine-learning workflow  
- A preprocessing API  
- Feature selection (MI + EFS)  
- Supervised models (RandomForest, XGBoost)  
- Unsupervised models (Isolation Forest, LOF)  
- Final comparison, visualizations, and interpretation  

---

## Repository Structure
.
├── anomaly_API.ipynb          # Preprocessing & feature-selection API build
├── anomaly_API.md             # API documentation (tool-focused)
├── anomaly_example.ipynb      # Full ML pipeline, models, evaluation, plots
├── anomaly_example.md         # Project workflow, results, interpretation
├── anomaly_utils.py           # Helper functions (loading, preprocessing, EFS)
│
├── outputs/
│   ├── selected_numeric.json
│   ├── metrics_supervised_efs.csv
│   ├── metrics_unsupervised.csv
│   ├── metrics_final_comparison.csv
│   └── plots/
│
├── Dockerfile
├── docker_build.sh
├── docker_jupyter.sh
├── requirements.txt
└── README.md

---

## Project Overview

Network intrusion detection is a classic anomaly-detection problem.  
The UNSW-NB15 dataset contains millions of network-flow records with labels for normal vs malicious traffic.

Our goals:
1. Build a reusable preprocessing API  
2. Perform feature selection  
3. Train **RandomForest + XGBoost** (supervised)  
4. Train **Isolation Forest + LocalOutlierFactor** (unsupervised)  
5. Compare models with clear metrics  
6. Save all results for reuse  

---

## Step 1 - Data Loading & Initial Exploration

We loaded the UNSW-NB15 dataset, inspected datatypes, missing values, and class imbalance.  
This helped identify which features needed careful preprocessing and scaling.

---

## Step 2 - Preprocessing Pipeline (API)

We built a reusable API using:
- `SimpleImputer`
- `StandardScaler`
- `OneHotEncoder`
- `ColumnTransformer`
- Train/test splits

The API outputs:
- Cleaned training/testing data  
- Encoded numeric and categorical features  
- A reproducible preprocessing pipeline  

Defined in: **anomaly_utils.py**

---

## Step 3 - Feature Selection

We used:
- **Mutual Information** to rank numeric features  
- **Exhaustive Feature Selector (EFS)** on a sampled subset  

Final selected features are saved in: outputs/model_comparison.csv


This reduces model complexity and speeds up training.

---

## Step 4 - Supervised Models (RF & XGBoost)

We trained:
- **RandomForestClassifier** (balanced subsampling)  
- **XGBoostClassifier** (handling imbalance with scale_pos_weight)  

We computed:
- Precision, Recall, F1  
- ROC-AUC  
- PR-AUC  
- Confusion Matrices  
- ROC & PR Curves  

**Supervised Results:**  
Both models performed extremely well (ROC-AUC ~ 0.999), showing strong separability when labels are available.

---

## Step 5 - Supervised Visualization & Export

This step includes:
- Confusion matrices (RF & XGB)
- ROC curve comparison
- Saving results to: outputs/model_comparison.csv


This lets us quickly compare both supervised models side-by-side.

---

## Step 6 - Unsupervised Models (IF & LOF)

We trained:
- **Isolation Forest** → ROC-AUC ~ 0.89–0.95  
- **LocalOutlierFactor** → struggled due to data dimensionality  

Unsupervised models typically perform much worse than supervised ones, especially on complex network data.

Results saved to: outputs/metrics_unsupervised.csv


---

## Step 7 - Final Comparison & Interpretation

We merged all results:
- Supervised metrics  
- Unsupervised metrics  
- ROC/PR curves  
- Written interpretations  

Saved to: outputs/metrics_final_comparison.csv


### Key Insights:
- Supervised methods dramatically outperform unsupervised ones  
- Feature selection improved stability and training speed  
- Isolation Forest performed reasonably well  
- LOF struggled with high-dimensional data  
- Dataset contains strong signal for supervised detection  

---

## Overall Summary

This project shows a complete, end-to-end anomaly detection pipeline:
- Data cleaning & preprocessing  
- Feature ranking and selection  
- Supervised & unsupervised models  
- Metrics export  
- Visualization & interpretation  

Supervised models provide near-perfect accuracy, while unsupervised ones offer realistic baselines for unlabeled detection tasks.

---

## How to Run the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Launch Jupyter
jupyter lab

### 3. Run Notebooks in order:
1. anomaly_API.ipynb
2. anomaly_example.ipynb

Outputs will appear automatically inside /outputs.

## Possible Enhancements:
1. Autoencoder-based anomaly detection
2. LSTM-Autoencoder for sequence flows
3. SHAP explanations for interpretability
4. Real-time anomaly stream using Kafka + Spark
5. FastAPI deployment for model inference

## Acknowledgements:
Developed by Pradeep Yellapu, Roshan Syed, Rasagna Tirumani
Dataset: UNSW-NB15 (UNSW & Australian Defence Force Academy)
Course: MSML610 - Fall 2025