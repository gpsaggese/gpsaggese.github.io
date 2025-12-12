from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Dict

import numpy as np
import pandas as pd
import joblib

from backend.logging_utils import setup_app_logger
from src.components.feature_engineering import FeatureEngineering
from src.exception import ProjectException


@dataclass
class LoadedArtifacts:
    model: Any
    scaler: Any
    feature_names: List[str]
    metadata: Dict[str, Any]


class ModelService:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.logger = setup_app_logger("api", logs_dir=project_root / "logs")

        self.artifacts_dir = project_root / "artifacts"
        self.best_dir = self.artifacts_dir / "best_model"
        self.metrics_path = self.artifacts_dir / "metrics" / "last_run.json"
        self.plots_dir = self.artifacts_dir / "plots"

        self._loaded: LoadedArtifacts | None = None

    def load_best(self) -> LoadedArtifacts:
        if self._loaded is not None:
            return self._loaded

        meta_path = self.best_dir / "linear_regression_final_meta.json"
        model_path = self.best_dir / "linear_regression_final.joblib"
        scaler_path = self.best_dir / "linear_regression_final_scaler.joblib"

        if not meta_path.exists() or not model_path.exists() or not scaler_path.exists():
            raise ProjectException(
                "Best model artifacts not found. Run the pipeline once to generate artifacts/best_model/*."
            )

        metadata = json.loads(meta_path.read_text())
        feature_names = metadata.get("feature_names")
        if not feature_names:
            raise ProjectException("feature_names missing from best-model metadata.")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        self._loaded = LoadedArtifacts(model=model, scaler=scaler, feature_names=list(feature_names), metadata=metadata)
        return self._loaded

    def build_features_for_ticker(self, ticker: str, lookback_days: int) -> pd.DataFrame:
        import yfinance as yf

        df = yf.download(ticker, period=f"{lookback_days}d", interval="1d", auto_adjust=False)
        if df.empty:
            raise ProjectException(f"No data returned for ticker {ticker}")
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(ticker, axis=1, level=-1, drop_level=True)
            except Exception:
                df.columns = ["_".join(map(str, c)) for c in df.columns]
        df.index.name = "Date"

        fe = FeatureEngineering(config_path=str(self.project_root / "config"), wandb_logger=None)
        feats = fe.build_features(df.copy())
        # Do NOT drop columns dynamically at inference. Use training feature set.
        return feats

    def predict_next_close(self, ticker: str, lookback_days: int) -> Dict[str, Any]:
        artifacts = self.load_best()
        feats = self.build_features_for_ticker(ticker, lookback_days)

        # Align to training feature order.
        missing = [c for c in artifacts.feature_names if c not in feats.columns]
        if missing:
            raise ProjectException(f"Missing required features for inference: {missing}")

        last_row = feats.iloc[-1]
        X = last_row[artifacts.feature_names].values.reshape(1, -1)
        Xs = artifacts.scaler.transform(X)
        yhat = float(artifacts.model.predict(Xs).reshape(-1)[0])

        last_date = str(feats.index[-1].date()) if hasattr(feats.index[-1], "date") else str(feats.index[-1])
        last_close = float(last_row["Close"]) if "Close" in feats.columns else float("nan")

        return {
            "ticker": ticker,
            "last_date": last_date,
            "last_close": last_close,
            "predicted_next_close": yhat,
            "model_name": "linear_regression_final",
        }


