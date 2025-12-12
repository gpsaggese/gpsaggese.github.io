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

## FastAPI + React (local deployment, no cloud)

### Backend (FastAPI)
From project root:
```bash
export WANDB_MODE=offline
uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health`
- `POST /predict` body: `{"ticker":"AAPL","lookback_days":365,"horizon_days":10}`
- `GET /metrics`
- `GET /feature_names`
- `GET /feature_template` (copy/paste starter dict for manual features)
- `POST /predict_features` body: `{"features":{...},"horizon_days":1}`

### Frontend (React)
From `frontend/`:
```bash
npm install
npm run dev
```
Then open `http://localhost:5173`.

Notes:
- The UI defaults to API Base URL `/api` (Vite proxy -> `http://127.0.0.1:8000`). You can also set it to `http://localhost:8000` directly.
- If you see **“Failed to fetch”** it means your backend is not reachable from the browser. Verify `http://localhost:8000/health` opens.
- To use your own background image, save it as: `frontend/public/background.png` (the app will automatically use it).

