"""Shared test fixtures."""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_csv(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a small synthetic CSV with datetime, numeric, and categorical cols."""
    rng = np.random.default_rng(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "store_id": rng.choice(["A", "B", "C"], size=n),
        "revenue": rng.normal(1000, 200, size=n).round(2),
        "units": rng.poisson(50, size=n),
        "temperature": rng.normal(20, 5, size=n).round(1),
        "is_weekend": [1 if d.weekday() >= 5 else 0 for d in dates],
    })
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def minimal_state(synthetic_csv: pathlib.Path) -> dict:
    """Build a minimal state dict that most phase functions can consume."""
    return {
        "path": str(synthetic_csv),
        "done": [],
        "type": "multiple",
        "time_col": "date",
        "winner_formatter": {"format": "%Y-%m-%d"},
        "primary_key": "date",
        "secondary_keys": ["store_id"],
        "entity_col": "store_id",
        "cols": ["date", "store_id", "revenue", "units", "temperature", "is_weekend"],
        "numeric_val_cols": ["revenue", "units", "temperature"],
        "numeric_continuous_cols": ["revenue", "temperature"],
        "numeric_count_cols": ["units"],
        "categorical_val_cols": ["store_id"],
        "categorical_feature_cols": ["store_id"],
        "binary_flag_cols": ["is_weekend"],
        "known_exogenous_cols": [],
        "target_cols": ["revenue"],
        "covariate_cols": ["temperature", "units"],
        "temporal_cols": ["date"],
        "expected_frequency": "1D",
        "has_missing_values": False,
        "quality_dataset_path": str(synthetic_csv),
        "session_id": "test",
    }
