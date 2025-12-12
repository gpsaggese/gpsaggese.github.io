# Phase 6 — Final Report (TutorTask103: Weights & Biases Hard)

## Situation
We need an end-to-end, reproducible stock forecasting project that runs in the professor Docker environment, tracks experiments in W&B, and serves predictions via a small full-stack app (FastAPI + React).

## Task
- Build a modular ML pipeline (ingestion → feature engineering → preprocessing → training/eval → registry).
- Compare multiple forecasting approaches and record metrics.
- Save and serve the **best** trained model.
- Provide local-only deployment with robust logging + basic UX.
- Complete **Phase 5 (W&B online run)** and **Phase 6 (documentation/report)**.

## Action
- Implemented modular pipeline under `src/`:
  - **Data ingestion**: `yfinance` OHLCV (AAPL default).
  - **Feature engineering**: returns, moving averages, RSI, MACD, volatility, lag features.
  - **Preprocessing**: time split + scaling (train-only fit), forecasting target shift.
  - **Model training**: Linear Regression baseline + tree models (RF/XGB/LGBM) + stats models (MA/AR/ARIMA/SARIMAX) + Prophet + stacking ensemble.
  - **Evaluation**: MAE/RMSE/MAPE/R2 + plots saved to `artifacts/plots/`.
  - **Registry**: save best model + scaler + metadata to `artifacts/best_model/`.
- Built a serving layer:
  - **FastAPI** endpoints: `/health`, `/metrics`, `/predict`, `/predict_features`, `/feature_names`, `/feature_template`.
  - Supports **1–120 step** iterative forecasts and returns a **profit/loss summary** in the response.
- Built a local UI:
  - React dashboard with API health indicator, ticker mode (horizon up to 120), and interactive manual feature mode.
- W&B:
  - Offline-first development for reproducibility, with a dedicated Phase 5 script for online logging.

## Result
- Produced artifacts for reporting and serving:
  - `artifacts/metrics/last_run.json` (all-model metrics + best-by-validation)
  - `artifacts/best_model/*` (best saved model artifacts)
  - `artifacts/plots/*.png` (evaluation plots)
- Best-by-validation model is typically **Linear Regression** on this configuration; stacking ensemble can match it on test metrics depending on run.
- Full-stack local app runs in Docker with port mapping and provides multi-day forecasts (up to 120 steps) with a simple profit/loss summary.

## Limitations / Notes
- Long-horizon forecasts (30/60/120) are **iterative**, so error compounds with horizon.
- “Profit/loss” output is a deterministic summary of the predicted path and does **not** account for uncertainty or transaction costs.


