# Fraud Detection in Credit Card Transactions

## What Has Been Done So Far

**Phase 1 (Completed)**

1. **Environment Setup**

   * Created project folder with modular structure (`configs/`, `src/`, `data/`, `tests/`, `Makefile`).
   * Installed and configured dependencies including `pandas`, `torch`, and `torch-geometric`.
   * Resolved compatibility issues between `numpy`, `torch`, and PyG on macOS.
     

2. **Dataset Used**

This project uses the IEEE-CIS Fraud Detection dataset from Kaggle
.
It combines transactional and identity-level information to simulate real-world credit card fraud detection scenarios.

Key Files Used

train_transaction.csv: contains transaction details such as TransactionID, TransactionDT, TransactionAmt, card1–card6, addr1–addr2, and the fraud label isFraud.

train_identity.csv: provides identity features linked to transactions by TransactionID, such as DeviceInfo, DeviceType, and email domains.

Both files are joined on TransactionID to form a unified dataset stored as data/processed/merged.parquet.
3. **Data Integration**

   * Joined `train_transaction.csv` and `train_identity.csv` into a unified dataset (`merged.parquet`).
   * Implemented `src/data/load_data.py` to clean and optimize data types for memory efficiency.

4. **Feature Engineering**

   * Built a feature pipeline (`src/features/`) that generates:

     * Transaction-level features (log-transformed amount, hour-of-day, day-of-week).
     * Account-level aggregated statistics (mean, std, count of transactions).
   * Saved engineered features to `data/processed/features.parquet`.

5. **Graph Construction**

   * Implemented a bipartite heterogeneous graph (`src/graph/construct_hetero_graph.py`) linking **transaction ↔ account** nodes.
   * Saved the graph as a PyTorch Geometric `HeteroData` object (`data/artifacts/hetero_graph.pt`).

6. **Baseline Model**

   * Implemented `src/models/tabular_baselines.py` using Logistic Regression (balanced).
   * Trained and evaluated with proper data split, producing precision/recall/F1/PR-AUC.
   * Stored results in `data/artifacts/baseline_metrics.json`.

7. **Automation and Testing**

   * Created a reproducible `Makefile` for the full pipeline:

     ```
     make data → make features → make graph → make train_tabular
     ```
   * Added unit tests (`tests/`) to verify graph construction and data split integrity.
   * Confirmed the full pipeline runs successfully end-to-end.

8. **Version Control**

   * Cleaned nested `.git` repo and committed all code under the main repository branch:
     `UmdTask88_Fall2025_DGL_Fraud_Detection_in_Credit_Card_Transactions`.

---

## What’s Next (Planned Work)

**Phase 2 and Beyond**

1. **Leakage-Free Features**

   * Recompute account-level aggregates using train-only data to avoid data leakage.

2. **Graph Neural Network Modeling**

   * Implement and train a GNN (GraphSAGE or GAT) to classify transaction nodes as fraudulent or not.
   * Compare performance against the logistic regression baseline.

3. **Temporal and Structural Enhancements**

   * Incorporate temporal encoding or edge features to capture evolving transaction patterns.
   * Experiment with edge classification as an alternative fraud detection approach.

4. **Performance Benchmarking and Error Analysis**

   * Perform detailed precision–recall and confusion matrix analysis.
   * Investigate misclassified transactions to understand fraud behavior.

5. **MLOps Integration**

   * Containerize the project using Docker.
   * Integrate with ClearML or W&B for experiment tracking and reproducibility.

