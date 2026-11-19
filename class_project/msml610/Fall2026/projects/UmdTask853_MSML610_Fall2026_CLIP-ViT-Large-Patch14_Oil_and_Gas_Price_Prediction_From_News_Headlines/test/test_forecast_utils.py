"""Synthetic-data checks: no external data downloads or CLIP weights needed."""
import numpy as np
import pandas as pd
import pytest

from forecast_utils import (
    PRICE_FEATURES,
    align_date_only_news,
    attach_daily_news_features,
    block_bootstrap_rmse_gain,
    make_price_features,
    metrics,
    read_energy_headlines,
    split_chronologically,
    tune_ridge,
)


def sample_prices(n=180):
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2019-11-01", periods=n)
    return pd.DataFrame({"date": dates, "price": 55 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))})


def test_lag_and_target_use_correct_dates():
    actual = sample_prices()
    frame = make_price_features(actual)
    row = frame.iloc[0]
    i = actual.index[actual.date.eq(row.date)][0]
    r_t = np.log(actual.price.iloc[i] / actual.price.iloc[i - 1])
    r_next = np.log(actual.price.iloc[i + 1] / actual.price.iloc[i])
    assert row.return_lag_1 == pytest.approx(r_t)
    assert row.target_next_return == pytest.approx(r_next)
    assert row.target_date == actual.date.iloc[i + 1]
    assert np.isfinite(frame[PRICE_FEATURES].to_numpy()).all()


def test_negative_price_invalidates_both_adjacent_returns():
    p = sample_prices()
    negative_date = p.date.iloc[90]
    day_after = p.date.iloc[91]
    preceding_date = p.date.iloc[89]
    p.loc[90, "price"] = -36.98
    frame = make_price_features(p)
    assert negative_date not in set(frame.date)
    assert day_after not in set(frame.date)
    assert preceding_date not in set(frame.date)  # invalid next-day target
    assert np.isfinite(frame.target_next_return).all()


def test_5_row_gap_and_order():
    split = split_chronologically(make_price_features(sample_prices()))
    a, b, c = split["train"], split["validation"], split["test"]
    assert a.date.max() < b.date.min() < c.date.min()
    full = make_price_features(sample_prices()).reset_index(drop=True)
    assert full.index[full.date.eq(b.date.iloc[0])][0] - full.index[full.date.eq(a.date.iloc[-1])][0] == 6
    assert full.index[full.date.eq(c.date.iloc[0])][0] - full.index[full.date.eq(b.date.iloc[-1])][0] == 6


def test_date_only_news_not_used_on_publication_date():
    trading = pd.Series(pd.to_datetime(["2025-06-13", "2025-06-16", "2025-06-17"]))
    news = pd.DataFrame({
        "published_date": pd.to_datetime(["2025-06-12", "2025-06-13", "2025-06-15"]),
        "headline_text": ["oil a", "oil b", "oil c"],
    })
    out = align_date_only_news(news, trading)
    assert out.date.dt.strftime("%Y-%m-%d").tolist() == ["2025-06-13", "2025-06-16", "2025-06-16"]
    assert (out.date > out.published_date).all()


def test_news_coverage_fills_missing_days_without_dropping_prices():
    base = make_price_features(sample_prices())
    daily = pd.DataFrame({"date": [base.date.iloc[0]], "clip_000": [0.4], "news_count": [3]})
    merged = attach_daily_news_features(base, daily)
    assert len(merged) == len(base)
    assert merged.news_count.iloc[0] == 3
    assert merged.news_count.iloc[1] == 0
    assert merged.clip_000.iloc[1] == 0


def test_expanding_cv_ridge_and_training_only_pca():
    base = make_price_features(sample_prices(n=225))
    rng = np.random.default_rng(42)
    for i in range(24):
        base[f"clip_{i:03d}"] = rng.normal(size=len(base))
    partitions = split_chronologically(base)
    model = tune_ridge(partitions["train"], clip_cols=[f"clip_{i:03d}" for i in range(24)])
    transformed = model.best_estimator_.named_steps["columns"]
    assert transformed.named_transformers_["clip"].named_steps["pca"].n_components_ == 20
    assert np.isfinite(model.predict(partitions["test"])).all()


def test_metrics_and_block_bootstrap():
    y = np.array([0.1, -0.2, 0.3, -0.4, 0.05, 0.07, -0.08, 0.03])
    p = np.zeros_like(y)
    c = y * 0.8
    m = metrics(y, p, reference_mean=0.0)
    assert np.isnan(m["directional_accuracy"])
    gain, low, high = block_bootstrap_rmse_gain(y, p, c, n_boot=150)
    assert gain > 0 and low > 0 and high > 0


def test_keyword_filter(tmp_path):
    csv = tmp_path / "headlines.csv"
    pd.DataFrame({"publish_date": [20250613, 20250613, 20250614],
                  "headline_text": ["Iran threatens oil tankers", "football tournament results", "OPEC output cut"]}).to_csv(csv, index=False)
    news = read_energy_headlines(csv)
    assert len(news) == 2


def test_missing_quote_days_do_not_reset_volatility_or_shift_target_to_nan():
    """A holiday has no quote: forecast the next observed session, without filling it."""
    original = sample_prices(n=180)
    missing_day = original.date.iloc[90]
    day_before = original.date.iloc[89]
    first_day_after = original.date.iloc[91]
    original.loc[90, "price"] = np.nan
    frame = make_price_features(original)
    assert missing_day not in set(frame.date)
    row = frame.loc[frame.date.eq(day_before)].iloc[0]
    assert row.target_date == first_day_after
    expected = np.log(original.price.iloc[91] / original.price.iloc[89])
    assert row.target_next_return == pytest.approx(expected)
    # The missing price must not delete a further 20 days of otherwise valid examples.
    assert first_day_after in set(frame.date)
    assert len(frame) > 150


def test_nonpositive_quote_still_blocks_adjacent_returns_with_missing_holidays():
    prices = sample_prices(n=180)
    prices.loc[45, "price"] = np.nan
    prices.loc[90, "price"] = -37.63
    frame = make_price_features(prices)
    for date in prices.date.iloc[[89, 90, 91]]:
        assert date not in set(frame.date)
    assert prices.date.iloc[92] not in set(frame.date)  # 20-quote volatility needs rebuilding
    assert np.isfinite(frame[PRICE_FEATURES].to_numpy()).all()
