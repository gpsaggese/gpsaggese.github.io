"""Unit tests for daily pooling; no CLIP model download required."""
import numpy as np
import pandas as pd
import pytest

from clip_features import make_daily_clip_table
from forecast_utils import EVENT_PROMPTS


def test_daily_pooling_and_similarity():
    news = pd.DataFrame({"date": pd.to_datetime(["2025-01-02", "2025-01-02", "2025-01-03"])})
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    prompts = np.tile([1., 0.], (len(EVENT_PROMPTS), 1))
    daily = make_daily_clip_table(news, vectors, prompts)
    assert len(daily) == 2
    assert daily.news_count.tolist() == [2, 1]
    assert daily.clip_000.iloc[0] == pytest.approx(0.5)
    assert daily.clip_001.iloc[0] == pytest.approx(0.5)
    assert daily.event_00.iloc[0] == pytest.approx(1 / 2**0.5)
    assert daily.event_00.iloc[1] == pytest.approx(0.0)


def test_bad_vector_shape_rejected():
    news = pd.DataFrame({"date": pd.to_datetime(["2025-01-02"])})
    with pytest.raises(ValueError):
        make_daily_clip_table(news, np.ones((2, 2)), np.ones((len(EVENT_PROMPTS), 2)))
