from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.components.feature_engineering import FeatureEngineering


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEST_DIR = PROJECT_ROOT / "artifacts" / "best_model"
MODEL_PATH = BEST_DIR / "linear_regression_final.joblib"
SCALER_PATH = BEST_DIR / "linear_regression_final_scaler.joblib"
META_PATH = BEST_DIR / "linear_regression_final_meta.json"


class PredictRequest(BaseModel):
    ticker: str = Field(default="AAPL", description="Yahoo Finance ticker symbol, e.g. AAPL")
    lookback_days: int = Field(default=400, ge=120, le=5000, description="Days of history to compute features/rolling windows")


class PredictResponse(BaseModel):
    ticker: str
    as_of_date: str
    predicted_next_close: float
    model_name: str
    features_used: list[str]


def _load_artifacts() -> tuple[Any, Any, Dict[str, Any]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model at {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Missing scaler at {SCALER_PATH}")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing metadata at {META_PATH}")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    meta = json.loads(META_PATH.read_text())
    return model, scaler, meta


def _fetch_history(ticker: str, lookback_days: int) -> pd.DataFrame:
    # Fetch extra to be safe around holidays/weekends.
    period = f"{lookback_days}d"
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(ticker, axis=1, level=-1, drop_level=True)
        except Exception:
            df.columns = ["_".join(map(str, c)) for c in df.columns]
    df.index.name = "Date"
    return df


def _build_latest_feature_row(df: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, str]:
    # Reuse FeatureEngineering feature computations, but avoid correlation dropping and any disk side-effects.
    fe = FeatureEngineering()
    tmp = df.copy()
    tmp = fe.fill_missing(tmp)
    tmp = fe.add_returns(tmp)
    tmp = fe.add_moving_averages(tmp)
    tmp = fe.add_rsi(tmp)
    tmp = fe.add_macd(tmp)
    tmp = fe.add_volatility(tmp)
    tmp = fe.add_lags(tmp)
    tmp = tmp.dropna()
    if tmp.empty:
        raise ValueError("Not enough history to compute features. Increase lookback_days.")
    as_of = str(tmp.index[-1].date())

    row = tmp.iloc[[-1]]
    # Ensure all required features exist (fill missing with 0.0)
    for c in feature_names:
        if c not in row.columns:
            row[c] = 0.0
    X_df = row[feature_names].astype(float)
    return X_df, as_of


app = FastAPI(title="Time Series Forecasting API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model, scaler, meta = _load_artifacts()
    feature_names = list(meta.get("feature_names", []))
    if not feature_names:
        raise ValueError("Missing feature_names in model metadata.")

    df = _fetch_history(req.ticker, req.lookback_days)
    X_df, as_of = _build_latest_feature_row(df, feature_names)
    X_scaled = scaler.transform(X_df.values)
    y_pred = float(np.asarray(model.predict(X_scaled)).reshape(-1)[0])

    return PredictResponse(
        ticker=req.ticker,
        as_of_date=as_of,
        predicted_next_close=y_pred,
        model_name=str(meta.get("model_name", "linear_regression_final")),
        features_used=feature_names,
    )


