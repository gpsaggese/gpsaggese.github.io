"""Shared synthetic fixtures and imports for the existing script modules."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@pytest.fixture
def clean_show():
    """Two recorded rounds with one offer; EVs match the supplied boards."""
    return pd.DataFrame({
        "show_id": [1, 1],
        "series_id": [2.0, 2.0],
        "round": [1, 2],
        "special_show": [0, 0],
        "bank_offer": [np.nan, 1500.],
        "decision": [np.nan, 0],
        "board_ev": [2720., 3275.],
        "remaining_board": [
            "[100,1000,10000,2000,500]",
            "[100,1000,10000,2000]",
        ],
        "n_remaining": [5, 4],
        "offer_ev_ratio": [np.nan, 1500 / 3275],
    })


@pytest.fixture
def contaminated_show(clean_show):
    df = clean_show.copy(deep=True)
    df.loc[df["round"].eq(1), "special_show"] = 1
    return df


@pytest.fixture
def multi_show_df(clean_show):
    second_show = clean_show.assign(show_id=2, series_id=3.0)
    return pd.concat([clean_show, second_show], ignore_index=True)


@pytest.fixture
def RAW_ROUND_FIXTURE():
    """Raw eliminations, kept separate from reconstructed-board fixtures."""
    from build_dataset import CASE_COLUMNS

    df = pd.DataFrame({
        "show_id": [1, 1], "series_id": [2.0, 2.0], "round": [1, 2],
        "bank_offer": [1000., 1500.],
    })
    for column in CASE_COLUMNS:
        df[column] = np.nan
    df["case_1"] = [200000., 100000.]
    return df


@pytest.fixture
def model_sample():
    """Independent controls and both outcomes for clustered logit tests."""
    rng = np.random.default_rng(42)
    n = 600
    df = pd.DataFrame({
        "show_id": np.repeat(np.arange(150), 4),
        "series_id": np.repeat(np.tile([2., 3., 3.1], 50), 4),
        "round": np.tile([1, 2, 3, 4], 150),
        "normalized_luck": rng.normal(0, .4, n),
        "offer_ev_ratio": rng.uniform(.1, 1.5, n),
        "log_board_ev": rng.normal(9, 1, n),
        "n_remaining": rng.integers(2, 20, n),
    })
    eta = -2 - df["normalized_luck"] + 1.5 * df["offer_ev_ratio"]
    df["decision"] = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    df.loc[df["round"].eq(1), "normalized_luck"] = np.nan
    df["luck_undefined"] = df["normalized_luck"].isna()
    return df
