# DGL Fraud Detection — API Notebook

This is the minimal, restart-and-run notebook that demonstrates the public helper API (`utils_data_io.py` / `utils_post_processing.py`) on the IEEE-CIS fraud dataset. Cells are kept short and rely only on the helper surface so graders/TA can mirror CLI behavior.

## Flow
1) Imports + minimal knobs (set once).
2) Build artifacts (or reuse existing): merged parquet → features → heterograph.
3) Train baselines (logreg + LightGBM; LightGBM may skip on tiny samples).
4) Train GraphSAGE (with pos_weight scaling for imbalanced data).
5) GNN metrics table (val/test) — printed, no fancy styling for compatibility.
6) Model comparison: styled table + baseline vs GNN bar charts.
7) Precision@k (validation) and FP/FN examples for quick inspection.

## Knobs (notebook cell)
- `USE_EXISTING_ARTIFACTS`: reuse or rebuild.
- `SAMPLE_FRAC`, `MAX_ROWS`: control sample size.
- `VAL_DAYS`, `TEST_DAYS`: temporal splits.
- `EPOCHS`, `HIDDEN_DIM`, `NUM_LAYERS`, `DROPOUT`, `LR`, `THRESHOLD`, `POS_WEIGHT_SCALE`, `DEVICE`: GraphSAGE training controls.

## Helper API used
- `load_config`, `build_dataset`, `build_features`, `build_graph`
- `train_tabular_baseline`, `train_lgbm_baseline`, `train_gnn_model`
- `compare_models_table`, `precision_recall_at_k`, `load_gnn_error_table`

## Outputs
- Baseline metrics JSONs, GNN metrics JSON + val/test predictions (for PR/precision@k/FP-FN).
- Styled comparison table and bar charts (val/test splits).
- Simple GNN metrics printout (PR-AUC, Precision, Recall, F1, ROC-AUC per split).

## How to run
1) Open `DGL_Fraud_Detection_in_Credit_Card_Transactions.API.ipynb`.
2) Adjust the **Run settings** cell (keep it small for fast demo; bump for stronger runs).
3) Run cells top-to-bottom. If LightGBM is unavailable, the notebook will skip it gracefully.***
