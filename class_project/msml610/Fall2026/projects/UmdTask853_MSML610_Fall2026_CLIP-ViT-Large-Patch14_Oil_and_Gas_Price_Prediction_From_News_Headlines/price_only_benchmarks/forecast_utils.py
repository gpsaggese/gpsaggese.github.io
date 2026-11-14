"""Leakage-aware daily spot-price return prediction, with forecast date = target date.

For a row dated t, y_t = log(P_t/P_prev), where prev is the previous *observed*
FRED quote. Every price feature is computed from data no later than prev.
Date-only news on d may enter only the first valid price date strictly after d.

Observation timestamps DO NOT establish FRED publication timestamps. This is a
retrospective research benchmark, NOT a deployable live trading forecast.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FRED_SERIES = {"wti": "DCOILWTICO", "henry_hub": "DHHNGSP"}
PRICE_FEATURES = [f"return_lag_{k}" for k in range(1, 6)] + ["momentum_5", "volatility_20"]


def read_fred_csv(source: str | Path, series: str) -> pd.DataFrame:
    raw = pd.read_csv(source)
    date_col = next((c for c in ("DATE", "observation_date", "date") if c in raw.columns), None)
    if date_col is None or series not in raw.columns:
        raise ValueError(f"Expected a FRED date column and {series}, got {list(raw.columns)}")
    out = pd.DataFrame({"date": pd.to_datetime(raw[date_col], errors="raise"),
                        "price": pd.to_numeric(raw[series], errors="coerce")})
    if out.date.isna().any() or out.date.duplicated().any():
        raise ValueError("Missing or duplicate FRED observation dates")
    return out.sort_values("date").reset_index(drop=True)


def make_price_features(raw_prices: pd.DataFrame) -> pd.DataFrame:
    """Forecast r_t at first valid quote t from lagged information through prev.

    Drop absent price quotes, NOT nonpositive quotes: if P_t <= 0, returns
    touching t are invalid. No positive-to-positive bridging across the negative
    WTI observation of April 2020.
    """
    x = raw_prices[["date", "price"]].copy().sort_values("date").reset_index(drop=True)
    if x.date.duplicated().any():
        raise ValueError("Price observation dates must be unique")
    x.price = pd.to_numeric(x.price, errors="coerce")
    x = x.loc[x.price.notna()].reset_index(drop=True)
    prev = x.price.shift(1)
    x["feature_asof_date"] = x.date.shift(1)
    pairs_valid = x.price.gt(0) & prev.gt(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        x["log_return"] = np.log(x.price / prev).where(pairs_valid)
    # All predictors are shifted at least one observed quote before the label.
    for k in range(1, 6):
        x[f"return_lag_{k}"] = x.log_return.shift(k)
    x["momentum_5"] = x.log_return.shift(1).rolling(5, min_periods=5).sum()
    x["volatility_20"] = x.log_return.shift(1).rolling(20, min_periods=20).std()
    x["target_return"] = x.log_return
    out = x.dropna(subset=PRICE_FEATURES + ["target_return", "feature_asof_date"]).reset_index(drop=True)
    if not (out.feature_asof_date < out.date).all():
        raise AssertionError("All price features must be from strictly older observations")
    if not np.isfinite(out[PRICE_FEATURES + ["target_return"]].to_numpy(dtype=float)).all():
        raise AssertionError("Nonfinite features or returns")
    return out


def split_chronologically(frame: pd.DataFrame, *, train_frac: float = .65,
                          val_frac: float = .15, gap: int = 5) -> dict[str, pd.DataFrame]:
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1 and gap >= 0):
        raise ValueError("Invalid split fractions or gap")
    x = frame.sort_values("date").reset_index(drop=True)
    if x.date.duplicated().any():
        raise ValueError("Duplicate forecast dates")
    a, b = int(len(x) * train_frac), int(len(x) * (train_frac + val_frac))
    parts = {"train": x.iloc[:a].copy(), "validation": x.iloc[a + gap:b].copy(),
             "test": x.iloc[b + gap:].copy()}
    if any(len(p) < 20 for p in parts.values()):
        raise ValueError("Not enough observations for chronological folds")
    return parts


def align_date_only_news(news: pd.DataFrame, valid_price_dates: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    """Map publication d to first positive/observed price date t > d.

    valid_price_dates must refer to positive nonmissing FRED quotes, not the
    already feature-filtered forecast rows. This prevents dropping news during
    feature warmup or nonpositive-price transitions.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(valid_price_dates).unique()).sort_values()
    if len(dates) == 0:
        raise ValueError("No valid price observations")
    publication = pd.DatetimeIndex(pd.to_datetime(news.published_date))
    positions = dates.searchsorted(publication, side="right")
    keep = positions < len(dates)
    out = news.loc[keep].copy()
    out["headline_index"] = np.flatnonzero(keep)
    out["date"] = dates[positions[keep]].to_numpy()
    if not (out.date > out.published_date).all():
        raise AssertionError("Date-only headlines must map STRICTLY after publication")
    return out.reset_index(drop=True)


def attach_daily_news_features(frame: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    if daily.date.duplicated().any():
        raise ValueError("Duplicate daily-news dates")
    cols = [c for c in daily if c != "date"]
    if not cols:
        raise ValueError("Need daily news features")
    if set(cols) & set(frame):
        raise ValueError("Column collision")
    out = frame.merge(daily, how="left", on="date", validate="one_to_one")
    out[cols] = out[cols].fillna(0).astype(float)
    if not np.isfinite(out[cols].to_numpy()).all():
        raise ValueError("Nonfinite news features")
    return out


def build_ridge(clip_cols: list[str] | None = None, *, n_components: int = 20) -> Pipeline:
    transforms = [("price", StandardScaler(), PRICE_FEATURES)]
    if clip_cols:
        vectors = [c for c in clip_cols if c.startswith("clip_")]
        auxiliary = [c for c in clip_cols if c not in vectors]
        if vectors:
            transforms.append(("clip", Pipeline([
                ("standardize", StandardScaler()),
                ("pca", PCA(n_components=min(n_components, len(vectors)),
                            svd_solver="randomized", random_state=42))]), vectors))
        if auxiliary:
            transforms.append(("aux", StandardScaler(), auxiliary))
    return Pipeline([("columns", ColumnTransformer(transforms)), ("ridge", Ridge())])


def tune_ridge(train: pd.DataFrame, *, clip_cols: list[str] | None = None):
    if clip_cols and any(c.startswith("clip_") for c in clip_cols) and len(train) // 4 - 5 < 25:
        raise ValueError("At least 25 examples needed in smallest training fold for PCA")
    cv = TimeSeriesSplit(n_splits=3, gap=5)
    grid = GridSearchCV(build_ridge(clip_cols),
                        {"ridge__alpha": [.1, 1., 10., 100., 1000.]},
                        cv=cv, scoring="neg_mean_squared_error", n_jobs=1,
                        refit=True, error_score="raise")
    grid.fit(train, train.target_return)
    return grid


def metrics(actual: np.ndarray, pred: np.ndarray, *, reference_mean: float) -> dict[str, float]:
    y = np.asarray(actual, dtype=float); p = np.asarray(pred, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Expected paired finite 1D actual/pred arrays")
    mse = float(np.mean((y - p)**2))
    ref_mse = float(np.mean((y - reference_mean)**2))
    meaningful = p != 0
    return {"rmse": float(np.sqrt(mse)),
            "oos_r2": float(1 - mse / ref_mse) if ref_mse > 0 else float("nan"),
            "directional_accuracy": float(np.mean(np.sign(y[meaningful]) == np.sign(p[meaningful]))) if meaningful.any() else float("nan")}


def block_bootstrap_rmse_gain(y, price_pred, news_pred, *, block_size: int = 5,
                              n_boot: int = 1000, seed: int = 42):
    y = np.asarray(y, dtype=float); a = np.asarray(price_pred, dtype=float); b = np.asarray(news_pred, dtype=float)
    if y.ndim != 1 or len(y) < block_size or len(a) != len(y) or len(b) != len(y):
        raise ValueError("Invalid paired bootstrap samples")
    rng = np.random.default_rng(seed)
    n = len(y); nb = (n + block_size - 1) // block_size
    gains = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(n, size=nb)
        idx = ((starts[:, None] + np.arange(block_size)) % n).ravel()[:n]
        gains[i] = np.sqrt(np.mean((y[idx] - a[idx])**2)) - np.sqrt(np.mean((y[idx] - b[idx])**2))
    observed = np.sqrt(np.mean((y-a)**2)) - np.sqrt(np.mean((y-b)**2))
    lo, hi = np.quantile(gains, [.025, .975])
    return float(observed), float(lo), float(hi)
