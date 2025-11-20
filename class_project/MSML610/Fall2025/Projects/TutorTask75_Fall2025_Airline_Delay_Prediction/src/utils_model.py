# src/utils_model.py
# Reusable helpers for Airline Delay Prediction
# Works with artifacts produced by:
#   - src/tuning_models.py  (tuned XGBoost with Optuna)
#   - src/train_baselines.py (LightGBM + CatBoost baselines)

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# ---------------------------
# Project-wide schema
# ---------------------------

BASE_CATEGORICAL: List[str] = [
    "AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "ORIGIN_STATE", "DEST_STATE"
]

BASE_NUMERIC: List[str] = [
    "DEPARTURE_DELAY", "AIR_TIME", "DISTANCE",
    "ORIGIN_LAT", "ORIGIN_LON", "DEST_LAT", "DEST_LON",
    "temp", "rhum", "prcp", "snow", "wspd", "pres",
]

SCHEMA: List[str] = BASE_CATEGORICAL + BASE_NUMERIC  # convenience
TARGET = "is_delayed"

RANDOM_STATE = 610


# ---------------------------
# Artifact registry (paths)
# ---------------------------

def get_artifacts() -> Dict[str, Dict[str, object]]:
    """
    Return the tag->paths mapping used across the project/app.
    Tags:
      - 'xgb_tuned' : tuned XGBoost artifacts
      - 'lgbm'      : LightGBM baseline artifacts
      - 'cat'       : CatBoost baseline artifacts
    """
    return {
        "xgb_tuned": {
            "prefix": "tuned_all_features_bo",
            "model": "models/tuned_all_features_bo_model.joblib",
            "metrics": "models/tuned_all_features_bo_metrics.json",
            "plots": {
                "PR":   "models/tuned_all_features_bo_pr.png",
                "ROC":  "models/tuned_all_features_bo_roc.png",
                "Loss": "models/tuned_all_features_bo_loss_curve.png",
                "CM":   "models/tuned_all_features_bo_confusion_matrix.png",
                "FI":   "models/tuned_all_features_bo_feature_importance.png",
            },
        },
        "lgbm": {
            "prefix": "lgbm_all_features",
            "model": "models/lgbm_all_features_model.joblib",
            "metrics": "models/lgbm_all_features_metrics.json",
            "plots": {
                "PR":   "models/lgbm_all_features_pr.png",
                "ROC":  "models/lgbm_all_features_roc.png",
                "Loss": "models/lgbm_all_features_loss_curve.png",
                "CM":   "models/lgbm_all_features_confusion_matrix.png",
                "FI":   "models/lgbm_all_features_feature_importance.png",
            },
        },
        "cat": {
            "prefix": "cat_all_features",
            "model": "models/cat_all_features_model.joblib",
            "metrics": "models/cat_all_features_metrics.json",
            "plots": {
                "PR":   "models/cat_all_features_pr.png",
                "ROC":  "models/cat_all_features_roc.png",
                "Loss": "models/cat_all_features_loss_curve.png",
                "CM":   "models/cat_all_features_confusion_matrix.png",
                "FI":   "models/cat_all_features_feature_importance.png",
            },
        },
    }


# ---------------------------
# IO helpers
# ---------------------------

def safe_read_csv(path_or_buffer, **kwargs) -> pd.DataFrame:
    """CSV reader with sensible defaults for big files."""
    defaults = dict(low_memory=False)
    defaults.update(kwargs or {})
    return pd.read_csv(path_or_buffer, **defaults)


def load_model(tag: str):
    """Load a trained model by tag. Returns None if missing."""
    art = get_artifacts().get(tag, {})
    path = art.get("model")
    if not path or not os.path.exists(path):
        return None
    return joblib.load(path)


def load_metrics(tag: str) -> Dict[str, object]:
    """
    Load saved metrics JSON for a tag.
    Expected keys (some may be absent depending on trainer):
      - roc_auc, pr_auc, f1_at_threshold, precision_at_threshold, recall_at_threshold
      - threshold, best_iteration
      - log_loss or valid_logloss
    """
    art = get_artifacts().get(tag, {})
    path = art.get("metrics")
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    # normalize the log loss key for downstream consumers
    if "log_loss" not in data:
        if "valid_logloss" in data:
            data["log_loss"] = data["valid_logloss"]
        elif "train_snapshot" in data and isinstance(data["train_snapshot"], dict):
            data["log_loss"] = data["train_snapshot"].get("valid_logloss")
    return data


def load_all_metrics_table(tags: Iterable[str] = ("cat", "lgbm", "xgb_tuned")) -> pd.DataFrame:
    """Assemble a leaderboard-like table across tags."""
    rows = []
    pretty = {"cat": "CatBoost", "lgbm": "LightGBM", "xgb_tuned": "XGBoost (tuned)"}
    for t in tags:
        m = load_metrics(t)
        if not m:
            continue
        rows.append({
            "model":      pretty.get(t, t),
            "AUC":        m.get("roc_auc"),
            "AP":         m.get("pr_auc"),
            "F1":         m.get("f1_at_threshold"),
            "Precision":  m.get("precision_at_threshold"),
            "Recall":     m.get("recall_at_threshold"),
            "log_loss":   m.get("log_loss"),
            "best_iter":  m.get("best_iteration"),
            "threshold":  m.get("threshold"),
            "tag":        t,
        })
    return pd.DataFrame(rows)


# ---------------------------
# Schema / dtype utilities
# ---------------------------

def coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the training dtypes:
      - numeric features -> float32
      - categorical features -> pandas 'category'
      - optional TARGET -> int
    Unseen/missing columns are left untouched.
    """
    out = df.copy()

    for c in BASE_NUMERIC:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")

    for c in BASE_CATEGORICAL:
        if c in out.columns:
            out[c] = out[c].astype("category")

    if TARGET in out.columns:
        out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce").fillna(0).astype(int)

    return out


# ---------------------------
# Prediction utilities
# ---------------------------

def _predict_xgb(model, Xc: pd.DataFrame) -> np.ndarray:
    """
    Predict probabilities for an XGBoost Booster or sklearn wrapper,
    handling categorical support and fallback to categorical codes.
    """
    import xgboost as xgb  # local import to keep import-time light

    # Try native Booster API via DMatrix first
    try:
        dmat = xgb.DMatrix(Xc, feature_names=list(Xc.columns), enable_categorical=True)
        return model.predict(dmat)
    except Exception:
        # Fallback: convert category -> codes
        Xtmp = Xc.copy()
        for c in Xtmp.select_dtypes(include="category").columns:
            Xtmp[c] = Xtmp[c].cat.codes.astype("int32")
        dmat = xgb.DMatrix(Xtmp, feature_names=list(Xtmp.columns))
        return model.predict(dmat)


def predict_proba(tag: str, model, X: pd.DataFrame) -> np.ndarray:
    """
    Library-aware probability prediction. Returns P(delay=1) for each row.

    Supports:
      - tuned XGBoost saved Booster (via DMatrix)
      - sklearn-style `predict_proba` (LightGBM, CatBoost sklearn wrappers)
      - CatBoost native model with `predict(..., prediction_type="Probability")`
    """
    if model is None or X is None or len(X) == 0:
        return np.array([], dtype="float32")

    Xc = coerce_schema(X)

    # XGBoost path (we use tag to decide preferred route)
    if tag == "xgb_tuned":
        try:
            return _predict_xgb(model, Xc)
        except Exception:
            pass  # fallback below

    # sklearn-like API (LGBM/CatBoost wrappers)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xc)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba.ravel()

    # CatBoost native API
    if hasattr(model, "predict"):
        try:
            preds = model.predict(Xc, prediction_type="Probability")
            preds = np.asarray(preds)
            if preds.ndim == 2:
                return preds[:, -1]
            return preds.ravel()
        except Exception:
            pass

    # Worst-case: can’t score
    return np.full(len(Xc), np.nan, dtype="float32")


def pick_threshold(metrics: Mapping[str, object], fallback: float = 0.5) -> float:
    """Return the threshold stored in metrics (or a fallback)."""
    th = metrics.get("threshold")
    try:
        return float(th)
    except Exception:
        return float(fallback)


# ---------------------------
# Scoring helpers
# ---------------------------

def score_dataframe(tag: str, df: pd.DataFrame, model=None, threshold: Optional[float] = None) -> pd.DataFrame:
    """
    Score a whole dataframe.
    Returns a copy with columns: proba_delay, pred_delay.
    """
    model = model or load_model(tag)
    if model is None:
        raise FileNotFoundError(f"Model for tag '{tag}' not found. Train or place it under models/.")

    m = load_metrics(tag)
    th = pick_threshold(m, 0.5) if threshold is None else float(threshold)

    Xc = coerce_schema(df)
    p = predict_proba(tag, model, Xc)
    pred = (p >= th).astype(int)

    out = Xc.copy()
    out["proba_delay"] = p
    out["pred_delay"] = pred
    return out


def score_row(tag: str, row: Mapping[str, object], model=None, threshold: Optional[float] = None) -> Tuple[float, int]:
    """
    Score a single example.
    Returns (probability, predicted_class).
    """
    df = pd.DataFrame([row], columns=row.keys())
    scored = score_dataframe(tag, df, model=model, threshold=threshold)
    p = float(scored["proba_delay"].iloc[0])
    y = int(scored["pred_delay"].iloc[0])
    return p, y


# ---------------------------
# Convenience: comparison CSV
# ---------------------------

def write_model_comparison_csv(path: str = "models/model_comparison.csv",
                               tags: Iterable[str] = ("cat", "lgbm", "xgb_tuned")) -> pd.DataFrame:
    """
    Build and save the leaderboard table so notebooks/app can display it.
    """
    tbl = load_all_metrics_table(tags)
    if not tbl.empty:
        tbl.to_csv(path, index=False)
    return tbl
