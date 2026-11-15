"""
Utility functions supporting the River tutorial notebooks.

The notebooks call these functions instead of writing raw logic inline, so the
notebooks stay short and every step is testable in isolation.

The module is organized as the pipeline is executed:
1. Data acquisition: download daily OHLCV bars from Yahoo Finance and cache
   them on disk so every run after the first is offline and reproducible.
2. Data validation and cleaning.
3. Feature engineering: features for day `t` use only information available
   at the close of day `t`.
4. Labels: direction of the return from close `t` to close `t+1`.
5. Streaming: replay the panel one trading day at a time.
6. Prequential (test-then-train) evaluation with a one-day label delay.
7. Statistical tests to check whether accuracy differs from chance.

Import as:

import river_utils as rivutil
"""

import datetime
import json
import logging
import pathlib
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats

_LOG = logging.getLogger(__name__)

# Default universe: liquid large caps from different sectors plus the S&P 500
# ETF, which is also used as a market-wide feature.
DEFAULT_TICKERS = [
    "SPY",
    "AAPL",
    "MSFT",
    "JPM",
    "XOM",
    "JNJ",
    "WMT",
    "NVDA",
]
MARKET_TICKER = "SPY"
DEFAULT_START = "2010-01-01"
DEFAULT_END = "2026-10-31"
DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
OHLCV_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


# #############################################################################
# 1. Data acquisition
# #############################################################################


def _cache_path(
    tickers: List[str], start: str, end: str, cache_dir: pathlib.Path
) -> pathlib.Path:
    """
    Build a cache file name that encodes the request, so changing the
    universe or dates never silently reuses stale data.
    """
    tag = f"{'-'.join(sorted(tickers))}_{start}_{end}"
    return cache_dir / f"ohlcv_{tag}.parquet"


def download_ohlcv(
    tickers: Optional[List[str]] = None,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    *,
    cache_dir: pathlib.Path = DATA_DIR,
    force: bool = False,
) -> pd.DataFrame:
    """
    Download daily OHLCV bars and return them in long format.

    The first call downloads from Yahoo Finance and writes a parquet file plus
    a JSON sidecar recording when and how the data was obtained. Later calls
    read the parquet file, so results do not change when Yahoo revises data.

    :param tickers: ticker symbols; defaults to `DEFAULT_TICKERS`
    :param start: first date (inclusive), `YYYY-MM-DD`
    :param end: last date (exclusive, as in yfinance), `YYYY-MM-DD`
    :param cache_dir: directory for the cached files
    :param force: re-download even if a cached file exists
    :return: dataframe with columns `date, ticker` + `OHLCV_COLUMNS`, sorted
        by `date, ticker`
    """
    tickers = list(tickers or DEFAULT_TICKERS)
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(tickers, start, end, cache_dir)
    if path.exists() and not force:
        _LOG.info("Loading cached data from %s", path)
        return pd.read_parquet(path)
    # Import lazily so the rest of the module works without network access.
    import yfinance as yf

    _LOG.info("Downloading %d tickers from Yahoo Finance", len(tickers))
    # `auto_adjust=False` keeps both raw and split/dividend-adjusted closes:
    # returns are computed from `adj_close`, raw prices are kept for checks.
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    if raw.empty:
        raise RuntimeError(
            "Yahoo Finance returned no data; check the network connection."
        )
    df = _wide_to_long(raw, tickers)
    df.to_parquet(path, index=False)
    meta = {
        "tickers": tickers,
        "start": start,
        "end": end,
        "downloaded_at_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "source": "Yahoo Finance via yfinance",
        "yfinance_version": yf.__version__,
        "n_rows": int(len(df)),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    _LOG.info("Cached %d rows to %s", len(df), path)
    return df


def _wide_to_long(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Convert the yfinance multi-index (ticker, field) frame to long format.
    """
    frames = []
    for ticker in tickers:
        if ticker not in raw.columns.get_level_values(0):
            _LOG.warning("No data returned for %s", ticker)
            continue
        sub = raw[ticker].copy()
        sub.columns = [
            c.lower().replace(" ", "_") for c in sub.columns.astype(str)
        ]
        sub = sub.reset_index().rename(columns={"Date": "date"})
        sub["ticker"] = ticker
        frames.append(sub[["date", "ticker"] + OHLCV_COLUMNS])
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


# #############################################################################
# 2. Validation and cleaning
# #############################################################################


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Report data-quality problems without modifying the data.

    :param df: long OHLCV dataframe
    :return: one row per ticker with counts of each problem
    """
    report = []
    for ticker, g in df.groupby("ticker"):
        report.append(
            {
                "ticker": ticker,
                "n_rows": len(g),
                "first_date": g["date"].min().date(),
                "last_date": g["date"].max().date(),
                "n_duplicate_dates": int(g["date"].duplicated().sum()),
                "n_missing_close": int(g["adj_close"].isna().sum()),
                "n_nonpositive_price": int(
                    (g[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()
                ),
                "n_high_below_low": int((g["high"] < g["low"]).sum()),
                "n_zero_volume": int((g["volume"] == 0).sum()),
            }
        )
    return pd.DataFrame(report).set_index("ticker")


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows that cannot be used, keeping a log of what was removed.

    Gaps are not forward-filled: inventing a price for a day with no trading
    would create a fake zero return.

    :param df: long OHLCV dataframe
    :return: cleaned dataframe
    """
    n0 = len(df)
    out = df.drop_duplicates(["date", "ticker"], keep="last")
    out = out.dropna(subset=["adj_close", "close", "high", "low", "open"])
    out = out[(out[["open", "high", "low", "close"]] > 0).all(axis=1)]
    out = out[out["high"] >= out["low"]]
    _LOG.info("clean_ohlcv: dropped %d of %d rows", n0 - len(out), n0)
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


# #############################################################################
# 3. Feature engineering
# #############################################################################


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index with Wilder smoothing, scaled to [0, 1].
    """
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (1 - 1 / (1 + rs)).fillna(0.5)


def _ticker_features(g: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features for one ticker. Every rolling window ends at day `t`, so
    no feature uses prices after the close of day `t`.
    """
    g = g.sort_values("date").copy()
    logret = np.log(g["adj_close"]).diff()
    feats = pd.DataFrame({"date": g["date"], "ticker": g["ticker"]})
    # Lagged returns: `ret_lag0` is today's return (close t-1 to close t).
    for lag in range(5):
        feats[f"ret_lag{lag}"] = logret.shift(lag)
    # Multi-day momentum.
    feats["mom_5"] = logret.rolling(5).sum()
    feats["mom_20"] = logret.rolling(20).sum()
    # Realized volatility.
    feats["vol_5"] = logret.rolling(5).std()
    feats["vol_20"] = logret.rolling(20).std()
    feats["vol_ratio"] = feats["vol_5"] / feats["vol_20"]
    # Price relative to moving averages.
    for w in (10, 50):
        sma = g["adj_close"].rolling(w).mean()
        feats[f"px_to_sma{w}"] = g["adj_close"] / sma - 1
    # Oscillators.
    feats["rsi_14"] = _rsi(g["adj_close"], 14)
    ema12 = g["adj_close"].ewm(span=12, adjust=False).mean()
    ema26 = g["adj_close"].ewm(span=26, adjust=False).mean()
    feats["macd_norm"] = (ema12 - ema26) / g["adj_close"]
    # Intraday range and where the close sits inside it.
    rng = (g["high"] - g["low"]).replace(0, np.nan)
    feats["hl_range"] = rng / g["close"]
    feats["close_loc"] = ((g["close"] - g["low"]) / rng).fillna(0.5)
    # Volume surprise relative to the last 20 days.
    logvol = np.log1p(g["volume"])
    feats["volume_z"] = (logvol - logvol.rolling(20).mean()) / logvol.rolling(
        20
    ).std()
    # Target: next-day log return. This is the only forward-looking column
    # and it is used exclusively as a label.
    feats["target_ret"] = logret.shift(-1)
    return feats


def build_features(
    df: pd.DataFrame, market_ticker: str = MARKET_TICKER
) -> pd.DataFrame:
    """
    Build the feature panel for all tickers.

    :param df: cleaned long OHLCV dataframe
    :param market_ticker: ticker used for market-wide features
    :return: dataframe indexed by row with `date`, `ticker`, feature columns,
        `target_ret`; rows with incomplete feature history are dropped
    """
    feats = pd.concat(
        [_ticker_features(g) for _, g in df.groupby("ticker")],
        ignore_index=True,
    )
    # Market context: same-day market return and volatility (known at close t).
    mkt = feats.loc[
        feats["ticker"] == market_ticker, ["date", "ret_lag0", "vol_20"]
    ].rename(columns={"ret_lag0": "mkt_ret", "vol_20": "mkt_vol_20"})
    feats = feats.merge(mkt, on="date", how="left")
    feats["excess_ret"] = feats["ret_lag0"] - feats["mkt_ret"]
    # Calendar feature.
    feats["day_of_week"] = feats["date"].dt.dayofweek
    feature_cols = feature_columns(feats)
    n0 = len(feats)
    # Warm-up rows lack history; the last row per ticker lacks a label.
    feats = feats.replace([np.inf, -np.inf], np.nan)
    feats = feats.dropna(subset=feature_cols + ["target_ret"])
    _LOG.info("build_features: dropped %d warm-up/unlabeled rows", n0 - len(feats))
    return feats.sort_values(["date", "ticker"]).reset_index(drop=True)


def feature_columns(feats: pd.DataFrame) -> List[str]:
    """
    Return the model input columns (everything except ids and the target).
    """
    excluded = {"date", "ticker", "target_ret", "target_up"}
    return [c for c in feats.columns if c not in excluded]


# #############################################################################
# 4. Labels
# #############################################################################


def add_direction_label(
    feats: pd.DataFrame, dead_zone: float = 0.0
) -> pd.DataFrame:
    """
    Add the binary target `target_up` = 1 if the next-day return is positive.

    With `dead_zone > 0`, days with `|return| <= dead_zone` are removed. Those
    moves are mostly noise and cannot be traded profitably after costs.

    :param feats: feature panel with `target_ret`
    :param dead_zone: absolute log-return threshold
    :return: copy of the panel with `target_up` (bool)
    """
    out = feats.copy()
    if dead_zone > 0:
        out = out[out["target_ret"].abs() > dead_zone]
    out["target_up"] = out["target_ret"] > 0
    return out.reset_index(drop=True)


# #############################################################################
# 5. Streaming
# #############################################################################


def iter_days(
    panel: pd.DataFrame, feature_cols: List[str], target_col: str = "target_up"
) -> Iterator[Tuple[pd.Timestamp, List[Tuple[str, Dict[str, float], Any]]]]:
    """
    Replay the panel one trading day at a time.

    River models consume one `dict` per observation. Grouping by date makes
    the information timeline explicit: all tickers for day `t` are predicted
    together, and their labels become known only at the close of day `t+1`.

    :param panel: labeled feature panel sorted by date
    :param feature_cols: columns to put in each feature dict
    :param target_col: label column
    :return: iterator of `(date, [(ticker, x, y), ...])`
    """
    for date, g in panel.groupby("date", sort=True):
        rows = []
        for ticker, x, y in zip(
            g["ticker"],
            g[feature_cols].to_dict(orient="records"),
            g[target_col],
        ):
            rows.append((ticker, x, y))
        yield date, rows


# #############################################################################
# 6. Prequential evaluation
# #############################################################################


def run_prequential(
    model: Any,
    panel: pd.DataFrame,
    feature_cols: List[str],
    *,
    target_col: str = "target_up",
    warmup_days: int = 0,
    drift_detector_factory: Optional[Callable[[], Any]] = None,
) -> pd.DataFrame:
    """
    Run test-then-train evaluation of a River classifier over the panel.

    For each trading day `t`:
    1. Predict every ticker using features at close `t`.
    2. Learn from the labels of day `t-1`, which became known at close `t`.

    Learning from day `t`'s labels before predicting other tickers on day `t`
    would leak the future, because those labels depend on close `t+1`. A
    naive row-by-row loop over a multi-ticker panel makes exactly this
    mistake, so the delay is handled explicitly here.

    :param model: River classifier (or pipeline) implementing
        `predict_proba_one` and `learn_one`; it is mutated in place
    :param panel: labeled feature panel
    :param feature_cols: model input columns
    :param target_col: label column
    :param warmup_days: number of initial days used for learning only; their
        predictions are not recorded
    :param drift_detector_factory: optional callable returning a River drift
        detector; it is fed the 0/1 error stream and drift days are flagged
    :return: one row per prediction with `date, ticker, y_true, p_up, y_pred,
        drift`
    """
    records = []
    pending: List[Tuple[Dict[str, float], Any]] = []
    detector = drift_detector_factory() if drift_detector_factory else None
    for i, (date, rows) in enumerate(iter_days(panel, feature_cols, target_col)):
        # Labels of the previous day are revealed at today's close.
        for x_prev, y_prev in pending:
            model.learn_one(x_prev, y_prev)
        pending = []
        for ticker, x, y in rows:
            proba = model.predict_proba_one(x)
            p_up = float(proba.get(True, 0.5)) if proba else 0.5
            y_pred = p_up > 0.5
            drift = False
            if detector is not None and i >= warmup_days:
                detector.update(int(y_pred != y))
                drift = bool(detector.drift_detected)
            if i >= warmup_days:
                records.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "y_true": bool(y),
                        "p_up": p_up,
                        "y_pred": bool(y_pred),
                        "drift": drift,
                    }
                )
            pending.append((x, y))
    return pd.DataFrame(records)


def summarize_predictions(preds: pd.DataFrame) -> pd.Series:
    """
    Compute standard classification metrics from a predictions frame.

    :param preds: output of `run_prequential`
    :return: series of metric values
    """
    import sklearn.metrics as skm

    y, yhat, p = preds["y_true"], preds["y_pred"], preds["p_up"]
    out = {
        "n": len(preds),
        "base_rate_up": y.mean(),
        "accuracy": skm.accuracy_score(y, yhat),
        "balanced_accuracy": skm.balanced_accuracy_score(y, yhat),
        "precision": skm.precision_score(y, yhat, zero_division=0),
        "recall": skm.recall_score(y, yhat, zero_division=0),
        "f1": skm.f1_score(y, yhat, zero_division=0),
        "mcc": skm.matthews_corrcoef(y, yhat),
        "log_loss": skm.log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]),
    }
    out["roc_auc"] = (
        skm.roc_auc_score(y, p) if y.nunique() == 2 else float("nan")
    )
    return pd.Series(out)


# #############################################################################
# 7. Statistical tests
# #############################################################################


def binomial_test_vs_baseline(preds: pd.DataFrame) -> Dict[str, float]:
    """
    Test whether accuracy beats always predicting the majority class.

    The baseline rate is the majority-class frequency, not 0.5: if prices rise
    on 53% of days, a model that always says "up" already scores 53%.

    Caveat: predictions for different tickers on the same day are correlated,
    so the effective sample size is smaller than `n` and the p-value is
    optimistic. `block_bootstrap_accuracy` addresses this.

    :param preds: output of `run_prequential`
    :return: accuracy, baseline accuracy, one-sided p-value
    """
    n = len(preds)
    k = int((preds["y_true"] == preds["y_pred"]).sum())
    base = max(preds["y_true"].mean(), 1 - preds["y_true"].mean())
    res = scipy.stats.binomtest(k, n, base, alternative="greater")
    return {"accuracy": k / n, "baseline": base, "p_value": res.pvalue}


def block_bootstrap_accuracy(
    preds: pd.DataFrame,
    *,
    block_days: int = 20,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Confidence interval for accuracy using a moving-block bootstrap over days.

    Resampling whole blocks of consecutive days keeps both the cross-sectional
    correlation (same day) and short-range serial correlation in the errors.

    :param preds: output of `run_prequential`
    :param block_days: block length in trading days
    :param n_boot: number of bootstrap replicates
    :param seed: random seed
    :return: point estimate and 95% interval
    """
    daily = (
        preds.assign(correct=preds["y_true"] == preds["y_pred"])
        .groupby("date")["correct"]
        .agg(["sum", "count"])
    )
    hits, counts = daily["sum"].to_numpy(), daily["count"].to_numpy()
    n_days = len(daily)
    n_blocks = int(np.ceil(n_days / block_days))
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n_days - block_days + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block_days)).ravel()[:n_days]
        stats[b] = hits[idx].sum() / counts[idx].sum()
    return {
        "accuracy": hits.sum() / counts.sum(),
        "ci_low": float(np.quantile(stats, 0.025)),
        "ci_high": float(np.quantile(stats, 0.975)),
    }


def mcnemar_test(preds_a: pd.DataFrame, preds_b: pd.DataFrame) -> Dict[str, float]:
    """
    Compare two models evaluated on the same stream with McNemar's test.

    :param preds_a: predictions of model A
    :param preds_b: predictions of model B
    :return: discordant counts and exact two-sided p-value
    """
    m = preds_a.merge(preds_b, on=["date", "ticker"], suffixes=("_a", "_b"))
    a_ok = m["y_pred_a"] == m["y_true_a"]
    b_ok = m["y_pred_b"] == m["y_true_b"]
    n01 = int((a_ok & ~b_ok).sum())
    n10 = int((~a_ok & b_ok).sum())
    p = scipy.stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"a_right_b_wrong": n01, "a_wrong_b_right": n10, "p_value": p}


# #############################################################################
# Synthetic data (for tests and offline demos)
# #############################################################################


def make_synthetic_ohlcv(
    tickers: Optional[List[str]] = None,
    n_days: int = 600,
    *,
    seed: int = 0,
    momentum: float = 0.0,
) -> pd.DataFrame:
    """
    Generate OHLCV bars from a geometric random walk.

    With `momentum = 0` returns are unpredictable, so any model should score
    near the base rate; with `momentum > 0` returns are AR(1) and a correct
    pipeline should find signal. This is a sanity check for leakage: a model
    that "wins" on pure noise has a bug.

    :param tickers: ticker symbols
    :param n_days: number of business days
    :param seed: random seed
    :param momentum: AR(1) coefficient of daily returns
    :return: long OHLCV dataframe
    """
    tickers = tickers or ["SPY", "AAA", "BBB"]
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    frames = []
    for t in tickers:
        eps = rng.normal(0, 0.01, n_days)
        r = np.zeros(n_days)
        for i in range(1, n_days):
            r[i] = momentum * r[i - 1] + eps[i]
        close = 100 * np.exp(np.cumsum(r))
        spread = np.abs(rng.normal(0, 0.005, n_days)) * close
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": t,
                    "open": close * (1 + rng.normal(0, 0.002, n_days)),
                    "high": close + spread,
                    "low": close - spread,
                    "close": close,
                    "adj_close": close,
                    "volume": rng.integers(1_000_000, 5_000_000, n_days),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    # Make high/low consistent with open.
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


# #############################################################################
# 8. Baselines and model comparison
# #############################################################################


class PersistenceClassifier:
    """
    Predict that tomorrow moves in the same direction as today.

    It has no parameters, so `learn_one` does nothing. It only predicts a
    direction, so it returns probabilities just off 0.5: this keeps the
    up/down prediction correct while log loss stays near ln(2) instead of
    blowing up on every miss, as it would with 0/1 probabilities.
    """

    def __init__(self, feature: str = "ret_lag0", margin: float = 0.01) -> None:
        self.feature = feature
        self.margin = margin

    def learn_one(self, x: Dict[str, float], y: Any) -> None:
        pass

    def predict_proba_one(self, x: Dict[str, float]) -> Dict[bool, float]:
        p_up = 0.5 + (self.margin if x[self.feature] > 0 else -self.margin)
        return {True: p_up, False: 1 - p_up}


def run_walk_forward(
    make_model: Callable[[], Any],
    panel: pd.DataFrame,
    feature_cols: List[str],
    *,
    target_col: str = "target_up",
    warmup_days: int = 250,
    refit_period: str = "M",
) -> pd.DataFrame:
    """
    Evaluate a batch scikit-learn classifier retrained on an expanding window.

    This is the standard alternative to online learning: periodically refit on
    all data available so far, then freeze the model until the next refit. At
    a refit on day `d`, the model is trained on rows dated before `d`, whose
    labels (returns up to close `d`) are known at the close of `d`.

    The scored days are the same as `run_prequential(..., warmup_days)`, so
    the two outputs can be compared row by row.

    :param make_model: callable returning a fresh unfitted sklearn classifier
    :param panel: labeled feature panel
    :param feature_cols: model input columns
    :param target_col: label column
    :param warmup_days: initial trading days that are never scored
    :param refit_period: pandas period of refits ("M" = monthly, "Y" =
        yearly); the model is refit on the first trading day of each period
    :return: predictions in the same format as `run_prequential`
    """
    days = np.sort(panel["date"].unique())
    first_scored = days[warmup_days]
    scored = days[days >= first_scored]
    # Refit on the first trading day of each period.
    periods = pd.DatetimeIndex(scored).to_period(refit_period)
    refit_days = scored[np.r_[True, periods[1:] != periods[:-1]]]
    # Sentinel so the last block runs to the end of the data.
    refit_days = np.append(refit_days, days[-1] + np.timedelta64(1, "D"))
    out = []
    for start, end in zip(refit_days[:-1], refit_days[1:]):
        train = panel[panel["date"] < start]
        test = panel[(panel["date"] >= start) & (panel["date"] < end)]
        model = make_model()
        model.fit(train[feature_cols], train[target_col].astype(int))
        p_up = model.predict_proba(test[feature_cols])[:, 1]
        out.append(
            pd.DataFrame(
                {
                    "date": test["date"].to_numpy(),
                    "ticker": test["ticker"].to_numpy(),
                    "y_true": test[target_col].astype(bool).to_numpy(),
                    "p_up": p_up,
                    "y_pred": p_up > 0.5,
                    "drift": False,
                }
            )
        )
    return pd.concat(out, ignore_index=True)


def compare_models(
    preds: Dict[str, pd.DataFrame], reference: str, *, n_boot: int = 1000
) -> pd.DataFrame:
    """
    Summarize several models evaluated on the same predictions.

    :param preds: model name -> output of `run_prequential` or
        `run_walk_forward`
    :param reference: name of the model every other model is tested against
        with McNemar's test
    :param n_boot: bootstrap replicates for the accuracy interval
    :return: one row per model with metrics, a 95% block-bootstrap interval
        for accuracy, and the McNemar p-value against `reference`
    """
    rows = {}
    for name, p in preds.items():
        row = summarize_predictions(p)[
            ["accuracy", "balanced_accuracy", "mcc", "roc_auc", "log_loss"]
        ]
        ci = block_bootstrap_accuracy(p, n_boot=n_boot)
        row["acc_ci_low"], row["acc_ci_high"] = ci["ci_low"], ci["ci_high"]
        row["share_pred_up"] = p["y_pred"].mean()
        row[f"mcnemar_p_vs_{reference}"] = (
            np.nan
            if name == reference
            else mcnemar_test(p, preds[reference])["p_value"]
        )
        rows[name] = row
    return pd.DataFrame(rows).T
