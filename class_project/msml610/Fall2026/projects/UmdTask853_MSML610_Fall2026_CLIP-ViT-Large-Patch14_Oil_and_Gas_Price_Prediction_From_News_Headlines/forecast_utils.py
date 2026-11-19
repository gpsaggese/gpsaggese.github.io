"""Leakage-aware next-observed-quote oil/gas return benchmarking utilities.

All features for a row at date t use information observed on or before t.
For date-only news, first eligible forecast origin is strictly *after* the
headline's publication calendar date. This conservative convention does not
establish intraday/real-time publication availability for FRED spot prices.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PRICE_FEATURES = [f"return_lag_{k}" for k in range(1, 6)] + ["momentum_5", "volatility_20"]
FRED_SERIES = {"wti": "DCOILWTICO", "henry_hub": "DHHNGSP"}
NEWS_PATTERN = re.compile(
    r"\b(?:oil|crude|opec|petrol|gas|pipeline|refiner(?:y|ies)|lng|energy|petroleum)\b",
    flags=re.IGNORECASE,
)
EVENT_PROMPTS = [
    "oil supply disruption",
    "OPEC production cut",
    "weak oil demand",
    "strong oil demand",
    "natural gas supply shortage",
    "cold weather increases natural gas demand",
]


def read_fred_csv(source: str | Path, series: str) -> pd.DataFrame:
    """Read a downloaded FRED CSV (DATE or observation_date, series column)."""
    df = pd.read_csv(source)
    date_col = next((c for c in ("DATE", "observation_date", "date") if c in df), None)
    if date_col is None or series not in df:
        raise ValueError(f"FRED CSV requires a date column and {series}; got {list(df.columns)}")
    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="raise"),
        "price": pd.to_numeric(df[series], errors="coerce"),
    })
    if out["date"].duplicated().any():
        raise ValueError("FRED CSV contains duplicate observation dates")
    return out.sort_values("date").reset_index(drop=True)


def make_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Predict the next *observed* price return using history available at t.

    FRED missing-price rows are days without a published spot observation;
    exclude those rows explicitly (never forward-fill) so the target is the
    next *observed quote*, not the next calendar row. This is appropriate for
    a next-observed-session benchmark, but elapsed days should be audited.

    IMPORTANT: Do not drop nonpositive prices before computing returns: a
    negative or zero WTI quote invalidates both returns touching it. Dropping
    nonpositive quotes would silently create an invalid bridge across them.
    """
    frame = prices[["date", "price"]].sort_values("date").copy().reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise ValueError("Price dates must be unique")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    # Remove absent observations, not negative observations. No price is imputed.
    frame = frame.loc[frame["price"].notna()].reset_index(drop=True)
    p = frame["price"]
    prev = p.shift(1)
    valid_pair = p.gt(0) & prev.gt(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["log_return"] = np.log(p / prev).where(valid_pair)
    # Target r_(t+1) has to be computed AFTER rejecting invalid adjacent pairs.
    frame["target_next_return"] = frame["log_return"].shift(-1)
    frame["target_date"] = frame["date"].shift(-1)
    for k in range(1, 6):
        frame[f"return_lag_{k}"] = frame["log_return"].shift(k - 1)
    frame["momentum_5"] = frame["log_return"].rolling(5, min_periods=5).sum()
    frame["volatility_20"] = frame["log_return"].rolling(20, min_periods=20).std()
    return frame.dropna(subset=PRICE_FEATURES + ["target_next_return"]).reset_index(drop=True)


def split_chronologically(
    frame: pd.DataFrame, *, train_frac: float = 0.65, val_frac: float = 0.15, gap: int = 5
) -> dict[str, pd.DataFrame]:
    """Make ordered train/val/test partitions with >=gap omitted origin rows."""
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("Require 0 < train_frac, val_frac and their sum < 1")
    if gap < 0:
        raise ValueError("gap cannot be negative")
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise ValueError("Forecast dates must be unique")
    n = len(frame)
    boundary1, boundary2 = int(n * train_frac), int(n * (train_frac + val_frac))
    out = {
        "train": frame.iloc[:boundary1].copy(),
        "validation": frame.iloc[boundary1 + gap : boundary2].copy(),
        "test": frame.iloc[boundary2 + gap :].copy(),
    }
    if min(map(len, out.values())) < 15:
        raise ValueError(f"Insufficient rows for 5-day gaps and evaluation: { {k:len(v) for k,v in out.items()} }")
    return out


def read_energy_headlines(source: str | Path) -> pd.DataFrame:
    """Filter ABC Australia Kaggle headlines; columns publish_date, headline_text."""
    raw = pd.read_csv(source, usecols=["publish_date", "headline_text"])
    raw["headline_text"] = raw["headline_text"].astype("string").str.strip()
    # ABC dates are YYYYMMDD, without release time or timezone.
    dates = raw["publish_date"].astype("string").str.replace(r"\.0$", "", regex=True)
    raw["published_date"] = pd.to_datetime(dates, format="%Y%m%d", errors="coerce")
    mask = raw["published_date"].notna() & raw["headline_text"].str.contains(NEWS_PATTERN, na=False)
    out = raw.loc[mask, ["published_date", "headline_text"]].drop_duplicates().copy()
    return out.sort_values(["published_date", "headline_text"]).reset_index(drop=True)


def align_date_only_news(news: pd.DataFrame, price_dates: pd.Series) -> pd.DataFrame:
    """Map a headline to the FIRST price observation strictly after its date.

    Forecast uses this mapped day's price history and targets the following
    observed return. Thus same-calendar-day headlines never enter a forecast
    that uses that day's close, even when publication times are unknown.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(price_dates).sort_values().unique())
    positions = dates.searchsorted(pd.DatetimeIndex(news["published_date"]), side="right")
    keep = positions < len(dates)
    out = news.loc[keep].copy()
    # Preserve the positional reference to the original encoded headline array.
    out["headline_index"] = np.flatnonzero(keep)
    out["date"] = dates[positions[keep]].values
    if not (out["date"].to_numpy() > out["published_date"].to_numpy()).all():
        raise AssertionError("Headline availability must precede forecast origin")
    return out.reset_index(drop=True)


def attach_daily_news_features(frame: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Left-join daily CLIP features, filling absent-news days with zeros."""
    if daily["date"].duplicated().any():
        raise ValueError("Daily CLIP features contain duplicate dates")
    feature_cols = [c for c in daily if c != "date"]
    if not feature_cols:
        raise ValueError("No CLIP feature columns supplied")
    if any(c in frame for c in feature_cols):
        raise ValueError("Name collision between price and news features")
    out = frame.merge(daily, how="left", on="date", validate="one_to_one")
    out[feature_cols] = out[feature_cols].fillna(0.0)
    if not np.isfinite(out[feature_cols].to_numpy(dtype=float)).all():
        raise ValueError("News features must be finite")
    return out


def _build_model(*, clip_cols: list[str] | None = None, n_components: int = 20) -> Pipeline:
    """All transforms live inside a Pipeline to prevent cross-fold leakage."""
    transformers = [("price", StandardScaler(), PRICE_FEATURES)]
    if clip_cols:
        vectors = [c for c in clip_cols if c.startswith("clip_")]
        auxiliary = [c for c in clip_cols if not c.startswith("clip_")]
        if vectors:
            transformers.append(("clip", Pipeline([
                ("standardize", StandardScaler()),
                ("pca", PCA(n_components=min(n_components, len(vectors)), svd_solver="full")),
            ]), vectors))
        if auxiliary:
            transformers.append(("event", StandardScaler(), auxiliary))
    return Pipeline([
        ("columns", ColumnTransformer(transformers, remainder="drop")),
        ("ridge", Ridge()),
    ])


def tune_ridge(train: pd.DataFrame, *, clip_cols: list[str] | None = None) -> GridSearchCV:
    """Use 5-step-embargo expanding-window CV; choose alpha without test data."""
    cv = TimeSeriesSplit(n_splits=3, gap=5)
    # Minimum expanding training fold must support 20-component PCA.
    if clip_cols and any(c.startswith("clip_") for c in clip_cols) and len(train) // 4 - 5 < 21:
        raise ValueError("Need more training rows to fit 20-component PCA in every CV fold")
    grid = GridSearchCV(
        _build_model(clip_cols=clip_cols),
        {"ridge__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]},
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    grid.fit(train, train["target_next_return"])
    return grid


def predict_baselines(parts: dict[str, pd.DataFrame]) -> dict[str, dict[str, np.ndarray]]:
    """Return out-of-sample predictions for zero, train mean and price Ridge."""
    train = parts["train"]
    mean = float(train["target_next_return"].mean())
    model = tune_ridge(train)
    return {
        name: {
            "zero": np.zeros(len(frame)),
            "train_mean": np.full(len(frame), mean),
            "price_ridge": model.predict(frame),
        }
        for name, frame in parts.items() if name != "train"
    }


def metrics(actual: np.ndarray, pred: np.ndarray, *, reference_mean: float) -> dict[str, float]:
    """Return RMSE, OOS R^2 against TRAIN mean, and signed direction accuracy."""
    y, p = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    if y.shape != p.shape or y.size == 0 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Metrics need same-shape, non-empty, finite arrays")
    mse = float(np.mean((y - p) ** 2))
    denom = float(np.mean((y - reference_mean) ** 2))
    mask = (p != 0.0) & (y != 0.0)
    return {
        "rmse": float(np.sqrt(mse)),
        "oos_r2": float(1.0 - mse / denom) if denom > 0 else np.nan,
        "directional_accuracy": float(np.mean(np.sign(y[mask]) == np.sign(p[mask]))) if mask.any() else np.nan,
    }


def block_bootstrap_rmse_gain(
    actual: np.ndarray, price_pred: np.ndarray, clip_pred: np.ndarray,
    *, n_boot: int = 1000, block_size: int = 5, seed: int = 42
) -> tuple[float, float, float]:
    """Paired circular moving-block bootstrap of price minus CLIP RMSE."""
    y, p, c = map(lambda a: np.asarray(a, dtype=float), (actual, price_pred, clip_pred))
    if not (len(y) == len(p) == len(c)) or len(y) < block_size or block_size < 1 or n_boot < 2:
        raise ValueError("Invalid bootstrap input lengths, block size or repetitions")
    if not (np.isfinite(y).all() and np.isfinite(p).all() and np.isfinite(c).all()):
        raise ValueError("Nonfinite bootstrap inputs")
    rng = np.random.default_rng(seed)
    n, gains = len(y), np.empty(n_boot)
    count = int(np.ceil(n / block_size))
    for i in range(n_boot):
        starts = rng.integers(0, n, size=count)
        ix = ((starts[:, None] + np.arange(block_size)[None, :]) % n).ravel()[:n]
        gains[i] = np.sqrt(np.mean((y[ix] - p[ix]) ** 2)) - np.sqrt(np.mean((y[ix] - c[ix]) ** 2))
    observed = np.sqrt(np.mean((y - p) ** 2)) - np.sqrt(np.mean((y - c) ** 2))
    lo, hi = np.quantile(gains, [0.025, 0.975])
    return float(observed), float(lo), float(hi)
