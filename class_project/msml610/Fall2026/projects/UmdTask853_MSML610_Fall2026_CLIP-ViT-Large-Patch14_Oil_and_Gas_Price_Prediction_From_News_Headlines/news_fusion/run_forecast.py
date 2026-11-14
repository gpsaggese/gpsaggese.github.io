#!/usr/bin/env python3
"""Leakage-aware news vs price-only Ridge benchmark, common period 2010–2021.

No encoder is trained. Models use past 30 observed returns and news on
publication dates STRICTLY EARLIER than target price observation dates.
All models share each asset's exact chronological 65/15/20 splits (5-row gaps).
Source corpus membership is controlled by --manifest; output is retrospective.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
SERIES = {"wti": "DCOILWTICO", "brent": "DCOILBRENTEU", "henry_hub": "DHHNGSP"}
LAGS = 30
PRICE_COLS = [f"return_lag_{k}" for k in range(1, LAGS + 1)] + ["momentum_5", "volatility_20"]
AUX_COLS = ["news_count", "has_news"]


def read_prices(path: Path, series: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    d = next((c for c in ("observation_date", "DATE", "date") if c in raw), None)
    if d is None or series not in raw:
        raise ValueError(f"Expected date and {series} columns in {path}")
    df = pd.DataFrame({"date": pd.to_datetime(raw[d], errors="raise"),
                       "price": pd.to_numeric(raw[series], errors="coerce")})
    if df.date.isna().any() or df.date.duplicated().any():
        raise ValueError("Missing/duplicate price dates")
    return df.sort_values("date").reset_index(drop=True)


def make_price_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Exactly the v0.4 30-lag benchmark's feature construction."""
    x = raw.loc[raw.price.notna(), ["date", "price"]].copy().reset_index(drop=True)
    previous = x.price.shift(1)
    good = x.price.gt(0) & previous.gt(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        x["log_return"] = np.log(x.price / previous).where(good)
    x["feature_asof_date"] = x.date.shift(1)
    x["target_return"] = x.log_return
    for k in range(1, LAGS + 1):
        x[f"return_lag_{k}"] = x.log_return.shift(k)
    x["momentum_5"] = x.log_return.shift(1).rolling(5, min_periods=5).sum()
    x["volatility_20"] = x.log_return.shift(1).rolling(20, min_periods=20).std()
    x = x.dropna(subset=PRICE_COLS + ["target_return", "feature_asof_date"]).reset_index(drop=True)
    if x.empty or not (x.feature_asof_date < x.date).all():
        raise ValueError("No valid leak-free price features")
    if not np.isfinite(x[PRICE_COLS + ["target_return"]].to_numpy(float)).all():
        raise ValueError("Nonfinite features/targets")
    return x


def split_frame(frame: pd.DataFrame, *, gap: int = 5):
    n = len(frame)
    a, b = int(.65*n), int(.80*n)
    parts = {"train": frame.iloc[:a].copy(),
             "validation": frame.iloc[a+gap:b].copy(),
             "test": frame.iloc[b+gap:].copy()}
    if any(len(v) < 40 for v in parts.values()):
        raise ValueError(f"Too few price observations ({n}) for split")
    return parts


def load_embeddings(folder: Path, manifest: pd.DataFrame):
    desired = manifest.headline_id.astype(str).tolist()
    results = {}
    for key in ("finbert", "clip", "minilm"):
        path = folder / f"{key}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as f:
            ids = f["ids"].astype(str).tolist()
            if ids != desired:
                raise ValueError(f"{path} does not match manifest headline IDs/order. Re-encode current manifest.")
            v = f["vectors"].astype("float32")
            if v.ndim != 2 or v.shape[0] != len(desired) or not np.isfinite(v).all():
                raise ValueError(f"Bad vectors in {path}")
            results[key] = {"vectors": v}
            if key == "clip" and (folder / "clip_prompts.npz").exists():
                with np.load(folder / "clip_prompts.npz", allow_pickle=False) as pp:
                    if pp["vectors"].shape[1] != v.shape[1]:
                        raise ValueError("CLIP headline and prompt dimensions differ")
                    results[key]["prompts"] = dict(zip(pp["ids"].astype(str).tolist(),
                                                        pp["vectors"].astype("float32")))
            if key == "finbert":
                p = f["sentiment"].astype("float32")
                if p.shape != (len(desired), 3) or not np.allclose(p.sum(1), 1., atol=1e-3):
                    raise ValueError(f"Bad sentiment in {path}")
                results[key]["sentiment"] = p
    return results


def align_news_to_target(news: pd.DataFrame, positive_price_dates):
    """Use first quote date STRICTLY after headline's calendar publication date."""
    quotes = pd.DatetimeIndex(pd.to_datetime(positive_price_dates).unique()).sort_values()
    publication = pd.DatetimeIndex(pd.to_datetime(news.published_date))
    ix = quotes.searchsorted(publication, side="right")
    keep = ix < len(quotes)
    aligned = news.loc[keep].copy()
    aligned["target_date"] = quotes[ix[keep]].to_numpy()
    if not (aligned.target_date > pd.to_datetime(aligned.published_date)).all():
        raise AssertionError("News cannot affect publication-day target")
    return aligned


def build_asset_frame(raw: pd.DataFrame, manifest: pd.DataFrame, embeddings: dict,
                      *, asset: str, start: str, end: str):
    price = make_price_frame(raw)
    price = price.loc[price.date.between(pd.Timestamp(start), pd.Timestamp(end))].copy().reset_index(drop=True)
    if price.empty:
        raise ValueError("No eligible prices in selected date range")
    flag = "gas_candidate" if asset == "henry_hub" else "oil_candidate"
    news = manifest.loc[manifest[flag]].copy()
    news["idx"] = news.index
    # Positive price dates from raw observation calendar, not model-eligible rows.
    aligned = align_news_to_target(news, raw.loc[raw.price.gt(0), "date"])
    aligned = aligned.loc[aligned.target_date.isin(price.date)].copy()
    count = aligned.groupby("target_date").size().rename("news_count")
    price["news_count"] = price.date.map(count).fillna(0).astype(float)
    price["has_news"] = (price.news_count > 0).astype(float)
    texts = aligned.groupby("target_date").headline_text.agg(" ".join)
    price["daily_text"] = price.date.map(texts).fillna("").astype(str)
    map_index = pd.Index(price.date).get_indexer(aligned.target_date)
    if (map_index < 0).any():
        raise AssertionError("Unmatched target date")
    row_indices = aligned.idx.to_numpy(dtype=int)
    for encoder, data in embeddings.items():
        for name, vector in [(encoder, data["vectors"])]+(
            [("finprob", data["sentiment"])] if encoder == "finbert" else []):
            width = vector.shape[1]
            totals = np.zeros((len(price), width), dtype=np.float64)
            np.add.at(totals, map_index, vector[row_indices])
            totals = totals / price.news_count.to_numpy()[:, None].clip(min=1)
            cols = [f"{name}_{k:03d}" for k in range(width)]
            # Add in one batch instead of repeated frame inserts.
            price = pd.concat([price, pd.DataFrame(totals.astype("float32"), columns=cols)], axis=1)
        if encoder == "clip" and "prompts" in data:
            prompt_asset = "henry_hub" if asset == "henry_hub" else "wti"
            ids = [f"{prompt_asset}_{k:02d}" for k in range(6)]
            pv = np.stack([data["prompts"][k] for k in ids]).astype(float)
            pv /= np.maximum(np.linalg.norm(pv, axis=1, keepdims=True), 1e-12)
            pooled = price[[f"clip_{k:03d}" for k in range(data["vectors"].shape[1])]].to_numpy(dtype=float, copy=True)
            pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
            evt = (pooled @ pv.T).astype("float32")
            price = pd.concat([price, pd.DataFrame(evt,
                columns=[f"event_{k:02d}" for k in range(len(ids))])], axis=1)
    meta = {"asset": asset, "n_eligible_price_dates": len(price),
            "news_records_matched_to_price_dates": len(aligned),
            "price_days_with_news": int((price.has_news > 0).sum()),
            "share_days_with_news": float(price.has_news.mean()),
            "all_features_asof_prior_quote": bool((price.feature_asof_date < price.date).all()),
            "news_mapped_strictly_after_publication": True}
    return price, meta


def pipeline_for(mode: str, frame: pd.DataFrame, *, n_components: int = 20):
    transformations = [("price", StandardScaler(), PRICE_COLS)]
    if mode != "price_ridge":
        transformations.append(("news", StandardScaler(), AUX_COLS))
    if mode == "price_tfidf_ridge":
        transformations.append(("lexical", TfidfVectorizer(min_df=2, max_features=3000,
                                                            ngram_range=(1, 2), dtype=np.float64), "daily_text"))
    elif mode == "price_finbert_sentiment_ridge":
        transformations.append(("sentiment", StandardScaler(), [f"finprob_{i:03d}" for i in range(3)]))
    else:
        if mode in ("price_clip_event_ridge", "price_clip_full_ridge"):
            transformations.append(("event", StandardScaler(), [f"event_{i:02d}" for i in range(6)]))
        prefix = {"price_clip_pca_ridge": "clip_", "price_clip_full_ridge": "clip_",
                  "price_finbert_cls_pca_ridge": "finbert_",
                  "price_minilm_pca_ridge": "minilm_", "shuffled_clip_control": "clip_",
                  "shuffled_finbert_control": "finbert_"}.get(mode)
        if prefix:
            cols = [c for c in frame if c.startswith(prefix)]
            if not cols:
                raise ValueError(f"Missing embeddings for {mode}")
            transformations.append(("semantic", make_pipeline(StandardScaler(), PCA(
                n_components=min(n_components, len(cols)), random_state=SEED, svd_solver="randomized")), cols))
    return Pipeline([("features", ColumnTransformer(transformations, sparse_threshold=.3)),
                     ("ridge", Ridge(solver="lsqr"))])


def shuffle_vectors(parts, prefix, seed=SEED):
    rng = np.random.default_rng(seed)
    out = {}
    for name, p in parts.items():
        p = p.copy()
        cols = [c for c in p if c.startswith(prefix)]
        valid = np.flatnonzero(p.has_news.to_numpy() > 0)
        if len(valid) > 1:
            arr = p.iloc[valid][cols].to_numpy().copy()
            p.loc[p.index[valid], cols] = arr[rng.permutation(len(valid))]
        out[name] = p
    return out


def score(y, p, baseline_mean):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mse = np.mean((y-p)**2)
    ref = np.mean((y-baseline_mean)**2)
    sign_valid = (np.sign(y) != 0) & (np.sign(p) != 0)
    return {"rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(y-p))),
            "oos_r2": float(1-mse/ref) if ref > 0 else float("nan"),
            "directional_accuracy": float(np.mean(np.sign(y[sign_valid]) == np.sign(p[sign_valid])))
            if sign_valid.any() else float("nan")}


def bootstrap_rmse_gain(y, base_pred, news_pred, *, samples=1000, block=5, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); base_pred = np.asarray(base_pred); news_pred = np.asarray(news_pred)
    n = len(y); chunk = (n+block-1)//block
    scores = []
    for _ in range(samples):
        ix = ((rng.integers(0, n, size=chunk)[:, None] + np.arange(block)) % n).ravel()[:n]
        a = np.sqrt(np.mean((y[ix]-base_pred[ix])**2))
        b = np.sqrt(np.mean((y[ix]-news_pred[ix])**2))
        scores.append(a-b)
    delta = float(np.sqrt(np.mean((y-base_pred)**2)) - np.sqrt(np.mean((y-news_pred)**2)))
    low, high = np.quantile(scores, [.025, .975])
    return {"rmse_gain": delta, "ci95_low": float(low), "ci95_high": float(high),
            "bootstrap_type": "paired_circular_moving_block", "block": block, "samples": samples}


def fit_eval(mode, splits, *, quick=False):
    train, val, test = (splits[n] for n in ("train", "validation", "test"))
    columns = list(train.columns)
    config = pipeline_for(mode, train)
    cvs = TimeSeriesSplit(n_splits=2 if quick else 3, gap=5)
    grid = GridSearchCV(config, {"ridge__alpha": [1.] if quick else [.1, 1., 10., 100., 1000.]},
                        cv=cvs, scoring="neg_mean_squared_error", n_jobs=1, error_score="raise")
    grid.fit(train, train.target_return)
    pre_refit_val_pred = grid.predict(val)
    trainval = pd.concat([train, val], ignore_index=True)
    refit = grid.best_estimator_
    refit.fit(trainval, trainval.target_return)
    pred = refit.predict(test)
    val_rmse = float(np.sqrt(np.mean((val.target_return - pre_refit_val_pred)**2)))
    return pred, {"chosen_alpha": grid.best_params_["ridge__alpha"],
                  "train_cv_rmse": float(np.sqrt(-grid.best_score_)),
                  "validation_rmse_pre_refit": val_rmse,
                  "train_rows": len(train), "validation_rows": len(val), "test_rows": len(test)}


def benchmark(manifest: pd.DataFrame, embeddings: dict, *, prices_dir: Path,
              output: Path, start: str, end: str, quick=False, assets=None):
    output.mkdir(parents=True, exist_ok=True)
    all_rows, all_predictions, all_ci = [], [], []
    configs, coverage = {}, {}
    assets = assets or list(SERIES)
    for asset in assets:
        raw = read_prices(prices_dir / f"{SERIES[asset]}.csv", SERIES[asset])
        frame, cov = build_asset_frame(raw, manifest, embeddings, asset=asset, start=start, end=end)
        splits = split_frame(frame)
        coverage[asset] = {**cov, "split_dates": {
            k: {"n": len(p), "start": str(p.date.min().date()), "end": str(p.date.max().date())}
            for k, p in splits.items()}}
        y = splits["test"].target_return.to_numpy(float)
        reference_mean = float(pd.concat([splits["train"], splits["validation"]]).target_return.mean())
        preds = {"zero_return": np.zeros_like(y),
                 "historical_mean": np.full_like(y, reference_mean)}
        modes = ["price_ridge", "price_news_count_ridge", "price_tfidf_ridge"]
        if "finbert" in embeddings:
            modes += ["price_finbert_sentiment_ridge", "price_finbert_cls_pca_ridge"]
        if "clip" in embeddings:
            modes += ["price_clip_pca_ridge", "shuffled_clip_control"]
            if "prompts" in embeddings["clip"]:
                modes += ["price_clip_event_ridge", "price_clip_full_ridge"]
        elif "finbert" in embeddings:
            modes += ["shuffled_finbert_control"]
        if "minilm" in embeddings:
            modes += ["price_minilm_pca_ridge"]
        configs[asset] = {}
        for mode in modes:
            print(f"[{asset}] fitting {mode} (train={len(splits['train'])}, val={len(splits['validation'])}, test={len(y)})", flush=True)
            used = (shuffle_vectors(splits, "clip_" if "clip" in mode else "finbert_")
                    if mode.startswith("shuffled_") else splits)
            p, cfg = fit_eval(mode, used, quick=quick)
            preds[mode] = p
            configs[asset][mode] = cfg
        for name, predicted in preds.items():
            row = {"asset": asset, "model": name, "n_test": len(y),
                   "test_from": str(splits["test"].date.min().date()),
                   "test_to": str(splits["test"].date.max().date()),
                   "pct_news_days_test": float(splits["test"].has_news.mean()),
                   **score(y, predicted, reference_mean)}
            all_rows.append(row)
            if name not in ("price_ridge", "historical_mean", "zero_return"):
                all_ci.append({"asset": asset, "model": name,
                               **bootstrap_rmse_gain(y, preds["price_ridge"], predicted,
                                                     samples=100 if quick else 1000)})
            all_predictions.extend({"asset": asset, "date": date.date().isoformat(),
                                    "model": name, "actual_return": float(yy),
                                    "predicted_return": float(pp),
                                    "news_count": float(nc)}
                                   for date, yy, pp, nc in zip(splits["test"].date, y, predicted,
                                                               splits["test"].news_count))
    report = pd.DataFrame(all_rows)
    report.to_csv(output / "test_leaderboard.csv", index=False)
    pd.DataFrame(all_predictions).to_csv(output / "test_predictions.csv", index=False)
    pd.DataFrame(all_ci).to_csv(output / "paired_rmse_bootstrap.csv", index=False)
    (output / "model_configs.json").write_text(json.dumps(configs, indent=2))
    (output / "coverage_and_splits.json").write_text(json.dumps(coverage, indent=2))
    (output / "protocol.json").write_text(json.dumps({"start": start, "end": end,
       "quick_diagnostic_only": quick, "random_seed": SEED,
       "price_lags": LAGS, "train_fraction": .65, "validation_fraction": .15,
       "test_fraction": .20, "split_gap_rows": 5,
       "date_only_news_rule": "first positive FRED observation STRICTLY after publication calendar date",
       "model_selection": "TimeSeriesSplit on train; PCA and TF-IDF within training fold; refit on train+validation",
       "test_usage": "No hyperparameter selected using test; test is exploratory after one inspection",
       "caution": "FRED observation dates are not verified real-time release timestamps; this is NOT investable/backtested trading PnL"}, indent=2))
    print("\n" + report.sort_values(["asset", "rmse"]).to_string(index=False))
    print("\nSaved:", output.resolve())
    return report, coverage


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--prices-dir", default="data")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2021-12-31")
    ap.add_argument("--assets", default="wti,brent,henry_hub")
    ap.add_argument("--quick", action="store_true", help="Diagnostics only: smaller CV and bootstrap")
    ap.add_argument("--allow-gdelt-pilot", action="store_true")
    args = ap.parse_args()
    manifest = pd.read_csv(args.manifest)
    if manifest.headline_id.duplicated().any():
        raise ValueError("Duplicate manifest IDs")
    if manifest.source.astype(str).str.contains("gdelt").any() and not args.allow_gdelt_pilot:
        raise ValueError("GDELT GKG sample consists of four 15-minute slices; do not report a forecasting evaluation on this sparse pilot")
    embeddings = load_embeddings(Path(args.features_dir), manifest)
    assets = [x.strip() for x in args.assets.split(",")]
    if any(a not in SERIES for a in assets):
        ap.error(f"Assets must be in {list(SERIES)}")
    benchmark(manifest, embeddings, prices_dir=Path(args.prices_dir),
              output=Path(args.output_dir), start=args.start, end=args.end,
              quick=args.quick, assets=assets)

if __name__ == "__main__":
    main()
