## Project 3 (Hard): Time Series Forecasting for Stock Prices

### Overview (STAR)
- **Situation**: Need a reproducible forecasting project that runs in the professor Docker environment and logs experiments to Weights & Biases (W&B).
- **Task**: Fetch stock data (Yahoo Finance), engineer features, train multiple forecasting models, compare results, save the best model, and provide a simple local app (FastAPI + React) to serve predictions.
- **Action**: Built a modular pipeline (`src/`), added W&B logging, implemented multiple model families (baseline ML + tree + statistical + Prophet), added Optuna Bayesian tuning scripts, saved best model artifacts, and shipped a local API + UI.
- **Result**: Metrics/plots are saved under `artifacts/`, best model is saved under `artifacts/best_model/`, and the app supports 1–120 step iterative forecasts with a profit/loss summary.

### Repo structure (key folders)
```
TutorTask_103_Weights_and_Biases_Hard/
  config/
    params.yaml              # all pipeline + model parameters
    wandb.yaml               # W&B entity/project + logging toggles
  src/
    components/              # ingestion, features, preprocessing, training, evaluation, registry
    pipeline/                # orchestration: train all models, compare, save best
    logging/                 # WandbLogger (console/file + W&B)
    utils/                   # config loader, helpers
    exception.py             # ProjectException
  scripts/
    tune_*_bayes.py          # Optuna Bayesian tuning for RF/XGB/LGBM
    phase5_wandb_online.sh   # Phase 5: run training in W&B online mode
    e2e_smoke_test.sh        # simple backend smoke test
  backend/                   # FastAPI app (health/predict/metrics/features)
  frontend/                  # React (Vite) UI
  artifacts/                 # generated: metrics/plots/best_model
  docs/FINAL_REPORT.md       # Phase 6 report
```

## Step-by-step: what we did end-to-end

### 1) Set up W&B (offline-first, then online)
- **Why offline-first**: You can run everything in Docker without permissions/API issues and still capture logs; then switch to online when ready.
- W&B settings live in `config/wandb.yaml` (entity/project/job type).
- The logger (`src/logging/logger.py`) calls `wandb.init()` and logs:
  - configs (from `params.yaml`)
  - metrics (MAE/RMSE/MAPE/R2)
  - plots (actual vs pred, residuals)

### 2) Data collection (Yahoo Finance)
- Implemented in `src/components/data_ingestion.py` using `yfinance`.
- Pulls OHLCV for a ticker (default `AAPL`) and saves raw/processed copies under `data/`.

### 3) Feature engineering (make the problem learnable)
Implemented in `src/components/feature_engineering.py`:
- **Returns**: percent returns (captures momentum).
- **Moving averages**: short/long/very-long windows (trend).
- **RSI / MACD**: classic momentum indicators.
- **Volatility**: rolling std (risk proxy).
- **Lag features**: past returns (autoregressive signal).
- Also avoids dropping the target column and keeps inference stable.

### 4) Preprocessing for forecasting
Implemented in `src/components/preprocessor.py`:
- Time-based train/val/test split (no shuffling).
- Scaling with `StandardScaler` (fit on train only).
- True forecasting by shifting the target by `forecast_horizon`.

### 5) Model development (we trained multiple families)
Implemented in `src/components/model_trainer.py`, orchestrated by `src/pipeline/training_pipeline.py`:
- **Baseline**: `linear_regression` (strong sanity check).
- **Tree/boosting**: `random_forest`, `xgboost`, `lightgbm`.
- **Statistical**: `ma`, `ar`, `arima`, `sarimax` (statsmodels).
- **Prophet**: `prophet` (trend + seasonality).
- **Ensemble**: stacking ensemble (meta LR trained on validation predictions).

### 6) Evaluation + best model selection
Implemented in `src/components/model_evaluation.py` + `src/pipeline/training_pipeline.py`:
- Metrics: **MAE, RMSE, MAPE, R2**
- Plots saved under `artifacts/plots/`
- Metrics saved under `artifacts/metrics/last_run.json`
- Best model chosen by validation metric `training.metrics_primary` (RMSE) and saved to `artifacts/best_model/`

### 7) Best model (current result)
From `artifacts/metrics/last_run.json`, the best-by-validation model is:
- **Best model**: `linear_regression`

## Model comparison (from `artifacts/metrics/last_run.json`)
Lower is better for MAE/RMSE/MAPE; higher is better for R2.

| Model | Val RMSE | Test RMSE | Test MAE | Test MAPE | Test R2 |
|---|---:|---:|---:|---:|---:|
| linear_regression | 0.470 | 4.939 | 3.773 | 2.822 | 0.995 |
| xgboost | 3.913 | 123.119 | 101.677 | 71.260 | -2.146 |
| lightgbm | 3.854 | 122.974 | 101.491 | 71.034 | -2.138 |
| random_forest | 5.254 | 124.452 | 103.298 | 73.232 | -2.214 |
| ma | 22.974 | 141.244 | 122.996 | 97.096 | -3.140 |
| ar | 11.237 | 132.089 | 111.685 | 82.538 | -2.621 |
| arima | 10.254 | 129.697 | 109.558 | 80.844 | -2.491 |
| sarimax | 4.839 | 86.289 | 72.785 | 53.680 | -0.545 |
| prophet | 3.908 | 104.206 | 83.559 | 54.986 | -1.254 |
| ensemble_stacking (test) | - | 4.934 | 3.693 | 2.634 | 0.995 |

## Plots for Phase 6 (to show why LR is best + training curves)
Generate summary RMSE plots (bar charts across models):
```bash
python scripts/plot_rmse_summary.py
```
Outputs:
- `artifacts/plots/rmse_comparison_test.png`
- `artifacts/plots/rmse_comparison_val.png`
- `artifacts/metrics/rmse_table.csv`

Training-curve plots (only for models that have iterative training):
- XGBoost: `artifacts/plots/xgboost_val_rmse_curve.png` (saved during training)
- LightGBM: `artifacts/plots/lightgbm_val_rmse_curve.png` (saved during training)
- LSTM: `artifacts/plots/lstm_loss_curve.png` + `artifacts/metrics/lstm_history.csv` (saved during training)

If LSTM fails, the traceback is saved to: `artifacts/metrics/lstm_error.txt`

## Phase 5: run W&B online (inside Docker)
Inside the professor Docker container (with `/venv` active + `.env` containing `WANDB_API_KEY`):
```bash
bash scripts/phase5_wandb_online.sh
```
Success = a new run appears in W&B under entity `othakur-university-of-maryland-org` and project `time_seires_forecasting`.

Note: If you `git pull` inside the Docker container, run:
```bash
git pull --recurse-submodules=no
```
This avoids submodule SSH fetch errors inside the container.

### LSTM (optional) and how to run it safely
Add `"lstm"` to `training.models_to_run` in `config/params.yaml`.
Then cap resources (recommended for Docker):
```bash
export LSTM_MAX_TRIALS=2
export LSTM_EPOCHS=8
export LSTM_BATCH_SIZE=32
```

## Phase 6: deliverables
- Metrics: `artifacts/metrics/last_run.json`
- Best model: `artifacts/best_model/linear_regression_final.joblib` (+ scaler + meta)
- Plots: `artifacts/plots/*.png`
- Report: `docs/FINAL_REPORT.md`

## Backend + Frontend (local-only deployment)

### Backend (FastAPI) — in Docker (host terminal from `.../umd_classes/`)
```bash
docker run --rm --entrypoint "" -p 8000:8000 \
  --env-file class_project/MSML610/Fall2025/Projects/TutorTask_103_Weights_and_Biases_Hard/.env \
  -v "$(pwd)":/workspace \
  -w /workspace/class_project/MSML610/Fall2025/Projects/TutorTask_103_Weights_and_Biases_Hard \
  -e PYTHONPATH=/workspace/class_project/MSML610/Fall2025/Projects/TutorTask_103_Weights_and_Biases_Hard:/workspace/helpers_root \
  msml610assignmentimage /bin/bash -lc "source /venv/bin/activate && pip install -q -r requirements.txt && python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000"
```
Endpoints:
- `GET /health`
- `GET /metrics`
- `POST /predict` (`ticker`, `lookback_days`, `horizon_days`, optional `investment_usd`/`shares`)
- `GET /feature_names`, `GET /feature_template`
- `POST /predict_features` (manual feature dict; supports iterative horizons up to 120)

### Frontend (React)
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` (API Base URL can stay `/api`).

## Smoke test (backend)
```bash
BASE_URL=http://127.0.0.1:8000 bash scripts/e2e_smoke_test.sh
```

