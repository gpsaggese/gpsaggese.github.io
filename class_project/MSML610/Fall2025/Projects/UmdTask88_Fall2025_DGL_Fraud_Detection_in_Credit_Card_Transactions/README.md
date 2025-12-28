# Fraud Detection in Credit Card Transactions (IEEE-CIS)

Graph-aware fraud detection for the IEEE-CIS dataset. The project builds a leak-free tabular feature store, constructs a transaction↔account heterograph, and trains both tabular baselines and a GraphSAGE GNN. Everything is runnable via `make`/Docker or the slim demo notebooks.

## Architecture at a glance
- **Data prep**: `src/data/load_data.py` merges transaction/identity CSVs → `data/processed/merged.parquet`.
- **Features**: `src/features/make_features.py` adds temporal encodings + cumulative account aggregates (leak-free) → `data/processed/features.parquet`.
- **Graph**: `src/graph/construct_hetero_graph.py` builds a bipartite `transaction ↔ account` heterograph (DGL/PyG compatible) → `data/artifacts/hetero_graph.pt`.
- **Models**
  - Tabular: class-weighted Logistic Regression + LightGBM (time/group-aware splits, threshold tuned on validation).
  - GNN: GraphSAGE over the heterograph using engineered features on transaction nodes and learnable account embeddings; positive class weight scaled for recall/PR-AUC.
- **Outputs**: metrics JSONs + prediction/error tables under `data/artifacts/` for reporting.

## Quickstart (local)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# end-to-end pipeline
make data features graph train_tabular train_tabular_lgbm train_gnn

# tests
pytest -q
```

## Docker
```bash
./docker_build.sh      # build image
./docker_bash.sh       # shell with repo mounted
./docker_jupyter.sh    # Jupyter on http://localhost:8888 (no token)
```

## Notebooks / API surface
- `DGL_Fraud_Detection_in_Credit_Card_Transactions.API.ipynb`: minimal API demo using only helpers in `utils_data_io.py` / `utils_post_processing.py`. Shows build → baselines → GNN → metrics/plots.
- `DGL_Fraud_Detection_in_Credit_Card_Transactions.example.ipynb`: narrated walkthrough with light defaults for quick runs.
- `DGL_Fraud_Detection_in_Credit_Card_Transactions.API.md` / `...example.md`: text companions with the same flow.

## Key scripts & helpers
- `utils_data_io.py`: `load_config`, `build_dataset`, `build_features`, `build_graph`, `train_tabular_baseline`, `train_lgbm_baseline`, `train_gnn_model`.
- `utils_post_processing.py`: `compare_models_table`, `precision_recall_at_k`, `load_gnn_error_table`, `load_*_metrics`.
- `src/train/train_gnn.py`: GraphSAGE training loop (pos_weight scaling, threshold tuning, metrics/predictions saved to artifacts).
- `Makefile`: `make data`, `make features`, `make graph`, `make train_tabular`, `make train_tabular_lgbm`, `make train_gnn`.

## Artifacts
- Tabular metrics: `data/artifacts/baseline_metrics.json`, `lgbm_metrics.json`
- GNN: `gnn_metrics.json`, `gnn_model.pt`, `gnn_val_test_preds.parquet`
- Graph: `data/artifacts/hetero_graph.pt`
- Features: `data/processed/features.parquet`

## Repository layout
```
├── DGL_Fraud_Detection_in_Credit_Card_Transactions.API.ipynb / .md
├── DGL_Fraud_Detection_in_Credit_Card_Transactions.example.ipynb / .md
├── utils_data_io.py, utils_post_processing.py
├── src/ (data, features, graph, models, train)
├── configs/default.yaml
├── data/ (raw, processed, artifacts)
├── tests/
├── Dockerfile, docker_*.sh
└── MODEL_CARD.md
```

## Notes for demo
- Use the notebooks’ run-settings cell to keep runs light (`SAMPLE_FRAC`, `MAX_ROWS`, `EPOCHS`).
- Precision/Recall/PR-AUC are the primary metrics (fraud is imbalanced). Use the provided charts + Precision@k.
- If LightGBM is missing (macOS), install `libomp` or skip; the code handles that gracefully.
