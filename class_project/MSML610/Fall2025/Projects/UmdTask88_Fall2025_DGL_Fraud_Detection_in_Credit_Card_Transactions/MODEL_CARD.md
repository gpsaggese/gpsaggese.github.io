# Model Card — UmdTask88 Fraud Detection

## Overview

- **Use case**: Detect fraudulent card-not-present transactions by modeling relationships between transaction events and account proxies (`card1` aggregations) using a heterograph + GraphSAGE.
- **Audience**: MSML610 project graders and potential downstream developers who need a reproducible fraud-detection baseline with tabular + GNN components.

## Data

- **Source**: IEEE-CIS Fraud Detection dataset (Kaggle). Contains `train_transaction.csv` (transactions + `isFraud`) and `train_identity.csv` (device/email signals) joined via `TransactionID`.
- **Pre-processing**:
  - Data types down-cast for memory efficiency.
  - Missing categorical proxies (card/email/device) cast to string for factorization.
  - Dataset optionally subsampled via `configs/default.yaml:data.sample_frac`.
- **Splits**:
  - Time-aware windows controlled via `splits.val_days` (`TransactionDT` units).
  - Account-aware (proxy `card1`) — no account appears in more than one split.
  - Train / Val / Test indexes shared across tabular + GNN models.

## Graph Schema

- **Node types**: `transaction`, `account`.
- **Edge types**: `transaction -owns-> account` and its reverse.
- Graph built via `src/graph/construct_hetero_graph.py`; stored as PyG `HeteroData` or converted to DGL on load.

## Features

- **Transaction-level**:
  - Hour-of-day, day-of-week, sine/cosine encodings.
  - `txn_time_norm` (min-max normalized TransactionDT) — *temporal encoding booster*.
  - `hours_since_last_txn` per account (captures velocity / dormancy).
  - Log-transformed amount, device-change flag.
- **Account-level (leak-free)**:
  - Cumulative transaction count, rolling mean/std of transaction amounts.
  - Smoothed fraud-rate prior (beta prior with `a0=1`, `b0=20`).
- All features stored in `data/processed/features.parquet`.

## Models

### Logistic Regression Baseline
- Implementation: `src/models/tabular_baselines.py`.
- Class-weighted `LogisticRegression` with numeric features listed above.
- Auto-tuned decision threshold based on validation PR curve.
- Outputs val/test PR-AUC, precision, recall, F1 (JSON -> `baseline_metrics.json`).

### LightGBM Baseline
- Implementation: `src/models/tabular_baselines.py` (`model_type="lgbm"`).
- `scale_pos_weight` for imbalance, early stopping on validation PR-AUC, feature standardization.
- Outputs to `lgbm_metrics.json`.

### Graph Neural Network
- Implementation: `src/models/gnn_fraud.py` + `src/train/train_gnn.py`.
- Architecture:
  - Heterogeneous GraphSAGE (mean aggregator) applied to all canonical edge types.
  - Non-transaction nodes receive learned embeddings; transaction nodes consume feature matrix.
  - Two-layer GraphSAGE + MLP head → scalar logit per transaction.
- Training:
  - Loss: `BCEWithLogitsLoss` with class imbalance weight.
  - Optimizer: Adam (`lr` default 1e-3).
  - Dropout default 0.2; hidden dim 128 (overridable); features standardized using training split stats.
  - Metrics logged for train/val/test splits every epoch (ROC-AUC, PR-AUC, precision, recall, F1); validation PR curve used to tune threshold.
- Artifacts:
  - `data/artifacts/gnn_model.pt`
  - `data/artifacts/gnn_metrics.json` (includes split sizes + threshold + best epoch)
  - `data/artifacts/gnn_val_test_preds.parquet` (val/test probabilities + FP/FN tags)

## Evaluation

- **Primary metrics**: PR-AUC, Precision, Recall, F1 (per split) with validation-tuned thresholds.
- **Operational view**: `utils_post_processing.precision_recall_at_k` used to report Precision@k / Recall@k for top-k alerts once raw logits are saved.
- **Reporting**: `compare_models_table()` builds markdown-ready tables; final numbers will be copied to REPORT + slides. Error tables support qualitative FP/FN inspection.

## Ethical / Practical Considerations

- IEEE-CIS data is heavily imbalanced and anonymized; performance may not translate to production environments without recalibration.
- Sensitive attributes (device/email/address proxies) are used; ensure compliance before deploying.
- Temporal leakage is mitigated via cumulative aggregates and time-based splits, but any distribution shift (e.g., after the test window) requires re-training.

## Reproducibility

- Deterministic pipeline controlled by `configs/default.yaml`.
- `Makefile` and Docker scripts encapsulate build/run commands.
- Random seeds embedded in scikit-learn splits + PyTorch training loops (via config).
- Tests guard split correctness and graph shape; add more scenario-specific tests as the project evolves.
