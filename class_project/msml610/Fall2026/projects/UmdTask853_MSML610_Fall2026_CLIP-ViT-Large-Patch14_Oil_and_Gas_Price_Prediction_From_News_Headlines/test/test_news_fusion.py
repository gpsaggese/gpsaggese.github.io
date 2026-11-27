"""Synthetic unit/integration tests; NO claims of real CLIP/FinBERT performance."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from news_fusion.prepare_news import prepare
from news_fusion.run_forecast import (make_price_frame, read_prices, align_news_to_target,
                                      build_asset_frame, split_frame, benchmark)


def test_prepare_preserves_direct_flags_and_deduplicates(tmp_path):
    a = pd.DataFrame({"date": ["2020-01-01", "2020-01-01", "2020-01-03"],
                      "headline_text": ["OPEC cuts crude oil", "Natural gas demand grows", "Ukraine announces sanctions"],
                      "oil_candidate": [True, False, False],
                      "gas_candidate": [False, True, False]})
    b = pd.DataFrame({"publication_date": ["2020-01-01", "2020-01-02", "2020-01-01"],
                      "headline_text": ["OPEC cuts crude oil", "Olive oil estate", "Natural gas demand grows"],
                      "oil_candidate": [True, True, False], "gas_candidate": [False, False, True]})
    p1, p2 = tmp_path/"abc.csv", tmp_path/"fn.csv"
    a.to_csv(p1, index=False); b.to_csv(p2, index=False)
    out, meta = prepare({"abc": str(p1), "fnspid": str(p2)}, start="2020-01-01", end="2020-12-31", selection="direct")
    assert len(out) == 3
    assert out.loc[out.headline_text == "OPEC cuts crude oil", "source"].iloc[0] == "abc,fnspid"
    assert out.headline_id.is_unique
    out_strict, _ = prepare({"abc": str(p1), "fnspid": str(p2)}, start="2020-01-01", end="2020-12-31", selection="strict")
    assert len(out_strict) == 2


def test_news_strictly_after_publication_and_price_lags(tmp_path):
    days = pd.date_range("2020-01-02", periods=140, freq="B")
    raw = pd.DataFrame({"date": days, "price": np.exp(np.linspace(4, 4.3, len(days)))})
    x = make_price_frame(raw)
    assert (x.feature_asof_date < x.date).all()
    h = pd.DataFrame({"published_date": [days[40].date().isoformat(),
                                              (days[50] + pd.Timedelta(days=1)).date().isoformat()],
                      "headline_text": ["OPEC cuts output", "Natural gas storage"],
                      "oil_candidate": [True, False], "gas_candidate": [False, True]})
    result = align_news_to_target(h, raw.date)
    assert result.iloc[0].target_date == days[41]
    assert (result.target_date > pd.to_datetime(result.published_date)).all()


def test_end_to_end_synthetic_does_not_require_model_downloads(tmp_path):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2018-01-01", periods=510, freq="B")
    price = 60*np.exp(np.cumsum(rng.normal(0, .01, len(dates))))
    for asset, series in [("wti", "DCOILWTICO"), ("brent", "DCOILBRENTEU"), ("henry_hub", "DHHNGSP")]:
        pd.DataFrame({"observation_date": dates.astype(str), series: price}).to_csv(tmp_path/f"{series}.csv", index=False)
    news = pd.DataFrame({
        "headline_id": [f"id{i}" for i in range(140)],
        "published_date": [dates[i*3+35].date().isoformat() for i in range(140)],
        "headline_text": [("OPEC crude oil supply cuts" if i % 2 == 0 else "Natural gas LNG demand rises")
                          for i in range(140)],
        "oil_candidate": [i % 2 == 0 for i in range(140)],
        "gas_candidate": [i % 2 == 1 for i in range(140)],
        "source": ["abc"]*140,
    })
    vectors = rng.normal(size=(len(news), 16)).astype("float32")
    sentiment = rng.dirichlet([2, 2, 2], size=len(news)).astype("float32")
    models = {"finbert": {"vectors": vectors, "sentiment": sentiment},
              "clip": {"vectors": vectors, "prompts": {
                  **{f"wti_{k:02d}": vectors[k] for k in range(6)},
                  **{f"henry_hub_{k:02d}": vectors[k+6] for k in range(6)}
              }}}
    report, coverage = benchmark(news, models, prices_dir=tmp_path, output=tmp_path/"res",
                                 start="2018-01-01", end="2019-12-31", quick=True,
                                 assets=["wti", "henry_hub"])
    assert set(report.asset) == {"wti", "henry_hub"}
    assert "price_finbert_sentiment_ridge" in set(report.model)
    assert "price_clip_pca_ridge" in set(report.model)
    assert "price_clip_event_ridge" in set(report.model)
    assert "price_clip_full_ridge" in set(report.model)
    assert "price_tfidf_ridge" in set(report.model)
    assert "shuffled_clip_control" in set(report.model)
    assert report.rmse.notna().all()
    assert (report.n_test > 0).all()
    assert (tmp_path/"res"/"test_predictions.csv").exists()
    assert coverage["wti"]["news_records_matched_to_price_dates"] > 0
    x = pd.read_csv(tmp_path/"res"/"test_leaderboard.csv")
    assert len(set(x.loc[x.asset == "wti", "n_test"])) == 1


def test_encoder_stub_real_shapes():
    import torch
    from news_fusion.encode_news import encode_batch
    class Tok:
        def __call__(self, texts, **kwargs):
            return {"input_ids": torch.ones((len(texts), 5),dtype=torch.long),
                    "attention_mask": torch.ones((len(texts), 5),dtype=torch.long)}
    class Clip:
        def get_text_features(self, **kwargs):
            return torch.ones((kwargs["input_ids"].shape[0], 4))
    class FB:
        class Config:
            id2label = {0: "positive", 1: "negative", 2: "neutral"}
        config = Config()
        def __call__(self, **kwargs):
            n = kwargs["input_ids"].shape[0]
            return type("X", (), {"hidden_states": [torch.ones((n, 5, 8))],
                                   "logits": torch.tensor([[2.,1.,0.] for _ in range(n)])})()
    v, p = encode_batch(["news", "more news"],encoder="clip", model=Clip(),tokenizer=Tok(),device="cpu")
    assert v.shape == (2, 4) and p is None and np.allclose(np.linalg.norm(v,axis=1),1)
    v, p = encode_batch(["news"],encoder="finbert",model=FB(),tokenizer=Tok(),device="cpu")
    assert v.shape == (1, 8) and p.shape == (1,3) and p[0,0] > p[0,1] > p[0,2]
