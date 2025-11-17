"""
Reusable utilities for the Renewable Energy Production forecasting demo.

Functions:
- load_data(path)
- make_features(df)
- train_model(df_features, target_col)
- forecast(model, df_features)
- evaluate(y_true, y_pred)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# -------------------- Data --------------------
def load_data(path: str) -> pd.DataFrame:
    """Load a small CSV; caller controls path."""
    df = pd.read_csv(path)
    return df

def make_features(df: pd.DataFrame, time_col: Optional[str] = None) -> pd.DataFrame:
    """
    Create simple time-based features if a datetime column is provided.
    Otherwise returns df unchanged (caller may have already engineered features).
    """
    out = df.copy()
    if time_col and time_col in out.columns:
        out[time_col] = pd.to_datetime(out[time_col])
        out["year"] = out[time_col].dt.year
        out["month"] = out[time_col].dt.month
        out["dayofyear"] = out[time_col].dt.dayofyear
        out["week"] = out[time_col].dt.isocalendar().week.astype(int)
        # simple seasonality proxy
        out["sin_month"] = np.sin(2 * np.pi * out["month"] / 12.0)
        out["cos_month"] = np.cos(2 * np.pi * out["month"] / 12.0)
    return out

# -------------------- Modeling --------------------
@dataclass
class TrainedModel:
    model: LinearRegression
    features: list[str]
    target: str

def train_model(df: pd.DataFrame, target_col: str, drop_cols: Optional[list[str]] = None) -> TrainedModel:
    """
    Train a simple baseline LinearRegression on numeric columns (minus target and drop_cols).
    """
    drop_cols = drop_cols or []
    num = df.select_dtypes(include=[np.number]).copy()
    if target_col not in num.columns:
        raise ValueError(f"Target '{target_col}' not found or is non-numeric.")

    X = num.drop(columns=list(set([target_col] + drop_cols)), errors="ignore")
    y = num[target_col].copy()

    # Drop columns with all-NaN after selection
    X = X.dropna(axis=1, how="all")
    # Simple impute for demo
    X = X.fillna(X.median(numeric_only=True))

    lr = LinearRegression()
    lr.fit(X, y)
    return TrainedModel(model=lr, features=list(X.columns), target=target_col)

def forecast(trained: TrainedModel, df: pd.DataFrame) -> np.ndarray:
    """Generate predictions using trained features (with the same simple impute)."""
    X = df[trained.features].copy()
    X = X.fillna(X.median(numeric_only=True))
    return trained.model.predict(X)

# -------------------- Evaluation & Plot --------------------
def evaluate(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict:
    """Return MAE and R^2 to keep things simple."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

def plot_series(df: pd.DataFrame, time_col: str, value_col: str, pred_col: Optional[str] = None):
    """Quick line plot for actual vs optional predicted."""
    plt.figure()
    plt.plot(df[time_col], df[value_col], label="actual")
    if pred_col and pred_col in df.columns:
        plt.plot(df[time_col], df[pred_col], label="pred")
    plt.xlabel(time_col)
    plt.ylabel(value_col)
    plt.legend()
    plt.tight_layout()
    plt.show()
