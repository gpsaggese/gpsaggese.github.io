# DGL Fraud Detection — Example Notebook (quick demo)

This notebook is the “lightweight demo” of the helper API. It runs a small slice of the IEEE-CIS data, builds artifacts, trains baselines + GraphSAGE, and surfaces concise visuals for storytelling.

## What it does (in order)
1) Imports + minimal run settings (set once, then run top-to-bottom).
2) Quick peek at raw data balance (shape, head, fraud rate).
3) Build artifacts: merged parquet → features → heterograph (reuses existing if toggled).
4) Train baselines: logistic regression + LightGBM (auto threshold on validation; LightGBM may skip on tiny samples).
5) Train GraphSAGE: uses the engineered features on transaction nodes and learnable account embeddings; pos_weight scaled for recall/PR-AUC.
6) GNN metrics table (val/test) and small PR-AUC/F1 bar charts.
7) Baseline vs GNN comparison: styled table + bar charts across val/test splits.
8) Visuals: per-epoch PR-AUC/F1 curves, validation PR curve + confusion summary.
9) Precision@k and quick FP/FN examples for narration.

## How to run it
1) Open `DGL_Fraud_Detection_in_Credit_Card_Transactions.example.ipynb`.
2) Set the **Run settings** cell (keep `USE_EXISTING_ARTIFACTS=True` for speed; defaults use a 10% sample, 200k max rows, 4 epochs).
3) Run all cells in order. If LightGBM is missing, it will skip gracefully; GNN metrics land in `data/artifacts/gnn_metrics.json`.

## Helper API used
- From `utils_data_io.py`: `load_config`, `build_dataset`, `build_features`, `build_graph`, `train_tabular_baseline`, `train_lgbm_baseline`, `train_gnn_model`.
- From `utils_post_processing.py`: `compare_models_table`, `precision_recall_at_k`, `load_gnn_error_table`.

## Outputs to look at
- Styled comparison table (PR-AUC, precision, recall, F1, ROC-AUC) for val/test.
- GNN per-split metrics table; PR-AUC/F1 bar charts.
- GNN vs baseline bar charts across key metrics.
- Validation PR curve + simple confusion summary at the current threshold.
- Precision@k printout and FP/FN examples for storytelling.

## Tips
- If runtime is slow, lower `SAMPLE_FRAC` or `MAX_ROWS`, or set `USE_EXISTING_ARTIFACTS=True` after one full run.
- Bump `EPOCHS`/`HIDDEN_DIM`/`POS_WEIGHT_SCALE` for stronger GNN runs; keep the demo small for live walkthroughs.
- The notebook relies solely on the public helpers, so graders can replicate via the same API or `make` targets.***
