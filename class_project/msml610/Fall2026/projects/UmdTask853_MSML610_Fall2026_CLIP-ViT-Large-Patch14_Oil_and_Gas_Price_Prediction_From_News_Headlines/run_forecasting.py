"""Run chronological price-only and optional frozen-CLIP benchmarks.

Usage examples in BENCHMARK_README.md. Writes no claims of improvement unless
actual test-set predictions and their uncertainty have been computed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from clip_features import load_or_encode_headlines, make_daily_clip_table
from forecast_utils import (
    FRED_SERIES,
    PRICE_FEATURES,
    align_date_only_news,
    attach_daily_news_features,
    block_bootstrap_rmse_gain,
    make_price_features,
    metrics,
    read_energy_headlines,
    read_fred_csv,
    split_chronologically,
    tune_ridge,
)


def _shuffled_news_parts(parts: dict[str, pd.DataFrame], news_cols: list[str], *, seed: int = 42):
    """Break date-content correspondence, separately within each split."""
    rng = np.random.default_rng(seed)
    shuffled = {}
    for split, frame in parts.items():
        clone = frame.copy()
        permutation = rng.permutation(len(frame))
        clone.loc[:, news_cols] = frame[news_cols].to_numpy()[permutation]
        shuffled[split] = clone
    return shuffled


def benchmark_asset(name: str, csv_path: str, news: pd.DataFrame | None,
                    embedded: np.ndarray | None, prompts: np.ndarray | None,
                    output_dir: Path) -> list[dict]:
    series = FRED_SERIES[name]
    raw = read_fred_csv(csv_path, series)
    price = make_price_features(raw)
    print(f"[{name}] valid forecast origins={len(price)}; raw prices={len(raw)};")
    print(f"[{name}] nonpositive price rows={int(raw.price.le(0).sum())}; missing prices={int(raw.price.isna().sum())}")

    news_cols = []
    if news is not None:
        # Restrict benchmark dates to the calendar coverage of the news source:
        # otherwise late test years would have zero news by construction.
        price = price.loc[
            (price.date >= news.published_date.min()) &
            (price.date <= news.published_date.max())
        ].copy()
        mapped = align_date_only_news(news, price.date)
        daily = make_daily_clip_table(mapped, embedded[mapped.headline_index.to_numpy()], prompts)
        price = attach_daily_news_features(price, daily)
        news_cols = [c for c in price if c.startswith(("clip_", "event_"))] + ["news_count", "has_news"]
        print(f"[{name}] filtered energy headlines={len(news)}; with-news origins={(price.has_news > 0).sum()}/{len(price)}")

    parts = split_chronologically(price, gap=5)
    train_mean = float(parts["train"].target_next_return.mean())
    price_model = tune_ridge(parts["train"])
    models = {"price_ridge": price_model}
    if news_cols:
        clip_vectors = [c for c in news_cols if c.startswith("clip_")]
        event_scores = [c for c in news_cols if c.startswith("event_")]
        availability = ["news_count", "has_news"]
        models["price_clip_pca_ridge"] = tune_ridge(parts["train"], clip_cols=clip_vectors + availability)
        models["price_event_scores_ridge"] = tune_ridge(parts["train"], clip_cols=event_scores + availability)
        models["price_clip_full_ridge"] = tune_ridge(parts["train"], clip_cols=news_cols)
        shuffled = _shuffled_news_parts(parts, news_cols)
        models["shuffled_news_control"] = tune_ridge(shuffled["train"], clip_cols=news_cols)

    results, predictions = [], []
    for split in ("validation", "test"):
        frame = parts[split]
        truth = frame.target_next_return.to_numpy()
        preds = {
            "zero_return": np.zeros(len(frame)),
            "train_mean": np.full(len(frame), train_mean),
            "price_ridge": models["price_ridge"].predict(frame),
        }
        if news_cols:
            preds["price_clip_pca_ridge"] = models["price_clip_pca_ridge"].predict(frame)
            preds["price_event_scores_ridge"] = models["price_event_scores_ridge"].predict(frame)
            preds["price_clip_full_ridge"] = models["price_clip_full_ridge"].predict(frame)
            preds["shuffled_news_control"] = models["shuffled_news_control"].predict(shuffled[split])
        for model_name, forecast in preds.items():
            stats = metrics(truth, forecast, reference_mean=train_mean)
            results.append({"asset": name, "split": split, "model": model_name,
                            "n": len(frame), "from": str(frame.date.min().date()),
                            "to": str(frame.date.max().date()), **stats})
            predictions.extend({"date": d, "target_date": td, "asset": name, "split": split,
                                "model": model_name, "actual_return": y, "predicted_return": p}
                               for d, td, y, p in zip(frame.date, frame.target_date, truth, forecast))
        if news_cols:
            gain, low, high = block_bootstrap_rmse_gain(
                truth, preds["price_ridge"], preds["price_clip_pca_ridge"])
            print(f"[{name}] {split}: CLIP gain in RMSE (price - CLIP) = {gain:+.6f}, 95% block CI [{low:+.6f}, {high:+.6f}]")
            if split == "test":
                (output_dir / f"{name}_rmse_gain_ci.json").write_text(json.dumps({
                    "rmse_gain": gain, "ci_low": low, "ci_high": high,
                    "bootstrap": "paired circular moving-block (length=5, 1000 resamples)"
                }, indent=2))
    pd.DataFrame(predictions).to_csv(output_dir / f"{name}_predictions.csv", index=False)
    pd.DataFrame(parts["test"][["date", "target_date", "target_next_return"]]).to_csv(
        output_dir / f"{name}_test_dates.csv", index=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wti-csv", required=True, help="FRED CSV for DCOILWTICO")
    parser.add_argument("--gas-csv", required=True, help="FRED CSV for DHHNGSP")
    parser.add_argument("--headlines-csv", help="Optional Kaggle ABC News headlines CSV")
    parser.add_argument("--cache-dir", default=".cache/forecast")
    parser.add_argument("--output-dir", default="results/benchmark")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    news, vectors, prompts = None, None, None
    if args.headlines_csv:
        news = read_energy_headlines(args.headlines_csv)
        if len(news) < 50:
            raise ValueError(f"Only {len(news)} filtered headlines; check data source or keywords")
        print(f"Filtered ABC energy headlines: {len(news)}; source dates {news.published_date.min().date()}..{news.published_date.max().date()}")
        vectors, prompts = load_or_encode_headlines(news, cache_dir=args.cache_dir, batch_size=args.batch_size)
    results = []
    for name, path in (("wti", args.wti_csv), ("henry_hub", args.gas_csv)):
        results.extend(benchmark_asset(name, path, news, vectors, prompts, output_dir))
    summary = pd.DataFrame(results)
    summary.to_csv(output_dir / "summary.csv", index=False)
    print("\nOUT-OF-SAMPLE BENCHMARK (real data only if real FRED / headline CSVs were provided):")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
