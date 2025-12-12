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

    def feature_template(self) -> Dict[str, float]:
        """
        Return a template dict of feature_name -> 0.0 for manual feature entry.
        """
        artifacts = self.load_best()
        return {name: 0.0 for name in artifacts.feature_names}

    def _profit_summary(
        self,
        baseline_price: float,
        predictions: List[Dict[str, Any]],
        investment_usd: float | None = None,
        shares: float | None = None,
    ) -> Dict[str, Any]:
        """
        Compute simple profit/loss summaries based on predicted closes.

        This is NOT a guarantee; it's a deterministic summary of the predicted path.
        """
        if not np.isfinite(baseline_price) or baseline_price <= 0:
            return {"note": "baseline_price missing/invalid; profit/loss not computed"}

        if shares is None and investment_usd is not None and investment_usd > 0:
            shares = float(investment_usd) / float(baseline_price)

        closes = [float(p["predicted_close"]) for p in predictions]
        best_idx = int(np.argmax(closes))
        worst_idx = int(np.argmin(closes))

        best_close = closes[best_idx]
        worst_close = closes[worst_idx]
        last_close = closes[-1]

        def _pnl(price: float) -> float:
            if shares is None:
                return float("nan")
            return (price - baseline_price) * float(shares)

        return {
            "baseline_price": float(baseline_price),
            "shares": float(shares) if shares is not None else None,
            "best_day": predictions[best_idx]["date"],
            "best_predicted_close": float(best_close),
            "best_return_pct": float((best_close - baseline_price) / baseline_price),
            "best_pnl_usd": _pnl(best_close),
            "worst_day": predictions[worst_idx]["date"],
            "worst_predicted_close": float(worst_close),
            "worst_return_pct": float((worst_close - baseline_price) / baseline_price),
            "worst_pnl_usd": _pnl(worst_close),
            "end_day": predictions[-1]["date"],
            "end_predicted_close": float(last_close),
            "end_return_pct": float((last_close - baseline_price) / baseline_price),
            "end_pnl_usd": _pnl(last_close),
            "signal": "profit_possible" if best_close > baseline_price else "loss_likely",
            "note": "Uses predicted closes only; ignores transaction costs and uncertainty.",
        }

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

    def _predict_row(self, artifacts: LoadedArtifacts, row: pd.Series) -> float:
        missing = [c for c in artifacts.feature_names if c not in row.index]
        if missing:
            raise ProjectException(f"Missing required features for inference: {missing}")
        X = row[artifacts.feature_names].values.reshape(1, -1)
        Xs = artifacts.scaler.transform(X)
        return float(artifacts.model.predict(Xs).reshape(-1)[0])

    def predict_ticker(
        self,
        ticker: str,
        lookback_days: int,
        horizon_days: int = 1,
        investment_usd: float | None = None,
        shares: float | None = None,
    ) -> Dict[str, Any]:
        artifacts = self.load_best()
        feats = self.build_features_for_ticker(ticker, lookback_days)

        if len(feats) == 0:
            raise ProjectException("Feature engineering produced empty features. Increase lookback_days.")

        last_row = feats.iloc[-1].copy()
        last_date = str(feats.index[-1].date()) if hasattr(feats.index[-1], "date") else str(feats.index[-1])
        last_close = float(last_row["Close"]) if "Close" in feats.columns else float("nan")

        # Multi-step is an iterative approximation: we update only features that depend on Close/returns/lags
        # and keep other exogenous features constant (Open/High/Low/Volume/RSI/MACD/etc).
        preds: List[Dict[str, Any]] = []
        prev_close = last_close
        if "return_pct" in last_row.index and pd.notna(last_row.get("return_pct")):
            prev_return = float(last_row["return_pct"])
        else:
            prev_return = 0.0

        # business-day dates
        try:
            start = pd.to_datetime(last_date) + pd.tseries.offsets.BDay(1)
            future_dates = pd.bdate_range(start=start, periods=horizon_days)
            future_dates_str = [d.date().isoformat() for d in future_dates]
        except Exception:
            future_dates_str = [f"t+{i}" for i in range(1, horizon_days + 1)]

        for i in range(1, horizon_days + 1):
            yhat = self._predict_row(artifacts, last_row)
            preds.append({"step": i, "date": future_dates_str[i - 1], "predicted_close": yhat})

            # Update a minimal set of features for the next step.
            if "Close" in last_row.index:
                last_row["Close"] = yhat
            if "return_pct" in last_row.index and prev_close and np.isfinite(prev_close):
                new_ret = (yhat - prev_close) / prev_close
                last_row["return_pct"] = float(new_ret)
                # shift lag_return_pct_k features if present
                lag_cols = [c for c in last_row.index if c.startswith("lag_return_pct_")]
                # shift from largest k -> 1 to avoid overwriting values we still need
                lag_pairs = []
                for c in lag_cols:
                    try:
                        k = int(c.split("_")[-1])
                    except Exception:
                        continue
                    lag_pairs.append((k, c))
                for k, c in sorted(lag_pairs, key=lambda x: x[0], reverse=True):
                    if k == 1:
                        last_row[c] = float(prev_return)
                    else:
                        prev_name = f"lag_return_pct_{k-1}"
                        if prev_name in last_row.index:
                            last_row[c] = float(last_row[prev_name])

                prev_return = float(last_row["return_pct"])

            prev_close = yhat

        return {
            "ticker": ticker,
            "last_date": last_date,
            "last_close": last_close,
            "horizon_days": int(horizon_days),
            "predictions": preds,
            "model_name": "linear_regression_final",
            "extra": {
                "note": "Multi-day forecast is iterative; only Close/return-based features are updated each step. Other features are held constant.",
                "profit_summary": self._profit_summary(last_close, preds, investment_usd=investment_usd, shares=shares),
            },
        }

    def predict_from_features(
        self,
        features: Dict[str, float],
        horizon_days: int = 1,
        current_price: float | None = None,
        investment_usd: float | None = None,
        shares: float | None = None,
    ) -> Dict[str, Any]:
        artifacts = self.load_best()
        row = pd.Series(features).copy()

        baseline = current_price
        if baseline is None and "Close" in row.index and pd.notna(row.get("Close")):
            try:
                baseline = float(row.get("Close"))
            except Exception:
                baseline = None

        # Iterative manual forecasting: update only Close/return-based features each step.
        preds: List[Dict[str, Any]] = []
        prev_close = float(baseline) if baseline is not None and np.isfinite(baseline) else float(row.get("Close", np.nan))
        prev_return = float(row.get("return_pct", 0.0)) if "return_pct" in row.index else 0.0

        for i in range(1, int(horizon_days) + 1):
            yhat = self._predict_row(artifacts, row)
            preds.append({"step": i, "date": f"t+{i}", "predicted_close": yhat})

            if "Close" in row.index:
                row["Close"] = yhat
            if "return_pct" in row.index and prev_close and np.isfinite(prev_close):
                new_ret = (yhat - prev_close) / prev_close
                row["return_pct"] = float(new_ret)

                lag_cols = [c for c in row.index if c.startswith("lag_return_pct_")]
                lag_pairs = []
                for c in lag_cols:
                    try:
                        k = int(c.split("_")[-1])
                    except Exception:
                        continue
                    lag_pairs.append((k, c))
                for k, c in sorted(lag_pairs, key=lambda x: x[0], reverse=True):
                    if k == 1:
                        row[c] = float(prev_return)
                    else:
                        prev_name = f"lag_return_pct_{k-1}"
                        if prev_name in row.index:
                            row[c] = float(row[prev_name])
                prev_return = float(row.get("return_pct", prev_return))

            prev_close = yhat

        return {
            "ticker": "MANUAL_FEATURES",
            "last_date": "N/A",
            "last_close": float(baseline) if baseline is not None and np.isfinite(baseline) else float("nan"),
            "horizon_days": int(horizon_days),
            "predictions": preds,
            "model_name": "linear_regression_final",
            "extra": {
                "note": "Manual feature mode: provide exactly the trained feature_names (see /feature_names). Multi-step is iterative; Close/return-based features are updated, others held constant.",
                "profit_summary": self._profit_summary(
                    float(baseline) if baseline is not None and np.isfinite(baseline) else float("nan"),
                    preds,
                    investment_usd=investment_usd,
                    shares=shares,
                ),
            },
        }


