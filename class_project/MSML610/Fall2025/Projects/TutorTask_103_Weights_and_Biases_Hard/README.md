## Overview (STAR)
**S**: Need a reproducible forecasting project in the professor Docker setup with W&B tracking + local serving.  
**T**: Build modular pipeline, compare models, save best model, ship local FastAPI+React app.  
**A**: Implemented `src/` pipeline + W&B logging + registry + FastAPI endpoints + React UI.  
**R**: Metrics+plots+best model saved under `artifacts/`; app serves 1–120 step forecasts and profit/loss summary.

## Repo structure
```
TutorTask_103_Weights_and_Biases_Hard/
  backend/        # FastAPI app + model service
  frontend/       # React (Vite) UI
  src/            # pipeline/components/eval/registry/logging
  scripts/        # tuning + phase5 + smoke tests
  config/         # params.yaml + wandb.yaml
  artifacts/      # metrics/plots/best_model (generated)
  docs/FINAL_REPORT.md
```

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

## Phase 5 (W&B online)
Inside the professor Docker container (with `/venv` active + `.env` present):
```bash
bash scripts/phase5_wandb_online.sh
```

## Phase 6 (deliverables)
- Metrics: `artifacts/metrics/last_run.json`
- Best model: `artifacts/best_model/linear_regression_final.joblib` (+ scaler/meta)
- Plots: `artifacts/plots/*.png`
- Report: `docs/FINAL_REPORT.md`

## Run app locally (FastAPI + React)
Backend in Docker (host terminal, from `.../umd_classes/`):
```bash
docker run --rm --entrypoint "" -p 8000:8000 \
  --env-file class_project/MSML610/Fall2025/Projects/TutorTask_103_Weights_and_Biases_Hard/.env \
  -v "$(pwd)":/workspace \
  -w /workspace/class_project/MSML610/Fall2025/Projects/TutorTask_103_Weights_and_Biases_Hard \
  -e PYTHONPATH=/workspace/class_project/MSML610/Fall2025/Projects/TutorTask_103_Weights_and_Biases_Hard:/workspace/helpers_root \
  msml610assignmentimage /bin/bash -lc "source /venv/bin/activate && pip install -q -r requirements.txt && python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000"
```
Frontend (host terminal):
```bash
cd frontend && npm install && npm run dev
```
Open `http://localhost:5173` (API Base URL can stay `/api`).

## Smoke test
```bash
BASE_URL=http://127.0.0.1:8000 bash scripts/e2e_smoke_test.sh
```

