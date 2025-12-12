## Overview
Time-series forecasting project for MSML610 with W&B experiment tracking.

## Quickstart (inside professor Docker image)

### Run pipeline (offline)
```bash
export WANDB_MODE=offline
python main.py train
```

### Bayesian tuning (offline)
```bash
export WANDB_MODE=offline
python main.py tune --model xgboost
python main.py tune --model lightgbm
python main.py tune --model random_forest
```

### Dashboard (local Flask UI)
```bash
python main.py serve --host 0.0.0.0 --port 8000
```

## Phase 5 (W&B Online)
```bash
export WANDB_MODE=online
# If needed:
export WANDB_ENTITY="othakur-university-of-maryland-org"
export WANDB_PROJECT="time_seires_forecasting"
python main.py train
```

## Phase 6 (Deliverables)
- Metrics: `artifacts/metrics/last_run.json`
- Best model: `artifacts/best_model/linear_regression_final.joblib`
- Plots: `artifacts/plots/*.png`

