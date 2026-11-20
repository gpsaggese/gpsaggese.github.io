"""Decision-sample selection must preserve no-Deal shows and prior states."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import os
import subprocess
import sys

from features import build_features


@pytest.fixture
def recorded_games():
    df = pd.DataFrame({
        "show_id": [1, 1, 1, 1, 2, 2, 2],
        "round": [1, 2, 3, 4, 1, 2, 3],
        "bank_offer": [np.nan, 100, 120, 130, 100, 120, 130],
        "decision": [np.nan, 1, np.nan, 1, 0, 0, 0],
        "board_ev": [300., 200., 150., 100., 300., 200., 150.],
    })
    df["remaining_board"] = df["board_ev"].apply(
        lambda ev: str([ev / 2, ev * 1.5])
    )
    df["offer_ev_ratio"] = df["bank_offer"] / df["board_ev"]
    return df


def test_keep_first_deal_and_all_no_deal_offers(recorded_games):
    # Scrambled input must still produce correctly ordered, show-local lags.
    df = recorded_games.sample(frac=1, random_state=42)
    original = df.copy(deep=True)
    result = build_features(df)

    assert list(zip(result["show_id"], result["round"])) == [
        (1, 2), (2, 1), (2, 2), (2, 3),
    ]
    assert result.loc[result["show_id"].eq(1), "decision"].tolist() == [1]
    deal = result.loc[result["show_id"].eq(1)].iloc[0]
    assert deal["board_ev_prev"] == 300
    assert deal["normalized_luck"] == pytest.approx(-1 / 3)
    assert not deal["luck_undefined"]
    first_no_deal_offer = result.loc[result["show_id"].eq(2)].iloc[0]
    assert first_no_deal_offer["luck_undefined"]
    assert result.loc[result["show_id"].eq(2), "offers_declined"].tolist() == [0, 1, 2]
    assert "first_deal_round" not in result.columns
    pd.testing.assert_frame_equal(df, original)


def test_missing_decision_before_deal_still_fails(recorded_games):
    recorded_games.loc[0, "bank_offer"] = 100
    recorded_games.loc[0, "offer_ev_ratio"] = 1 / 3
    with pytest.raises(ValueError, match="missing decisions"):
        build_features(recorded_games)


def test_corrected_board_evs_and_luck_flag(clean_show):
    result = build_features(clean_show)
    assert len(result) == 1
    offer = result.iloc[0]
    assert offer["board_ev_prev"] == 2720
    assert offer["board_ev"] == np.mean(offer["board_list"]) == 3275
    assert offer["offer_ev_ratio"] == pytest.approx(1500 / 3275)
    assert offer["normalized_luck"] == pytest.approx((3275 - 2720) / 2720)
    # The first observed offer has a prior non-offer round, so luck is defined.
    assert not offer["luck_undefined"]
    assert "is_first_offer" not in result.columns


@pytest.mark.parametrize("empty", [pd.DataFrame(), pd.DataFrame(columns=["show_id"])])
def test_empty_input_has_feature_schema(empty):
    result = build_features(empty)
    assert result.empty
    assert {"normalized_luck", "luck_undefined", "board_list", "offers_declined"}.issubset(result)
    assert result["luck_undefined"].dtype == bool


def test_no_offers_returns_empty_features(clean_show):
    result = build_features(clean_show.assign(bank_offer=np.nan))
    assert result.empty
    assert result["luck_undefined"].dtype == bool


def test_zero_prior_ev_is_undefined_luck(clean_show):
    clean_show.loc[0, "board_ev"] = 0
    clean_show.loc[0, "remaining_board"] = "[0,0,0,0,0]"
    row = build_features(clean_show).iloc[0]
    assert pd.isna(row["normalized_luck"])
    assert row["luck_undefined"]


@pytest.mark.parametrize("column, value, message", [
    ("board_ev", -1, "Negative board_ev"),
    ("board_ev", np.inf, "Nonfinite board_ev"),
    ("decision", 2, "Invalid decision"),
    ("offer_ev_ratio", np.inf, "Infinite offer_ev_ratio"),
    ("offer_ev_ratio", np.nan, "missing offer/EV"),
    ("bank_offer", np.inf, "bank_offer contains nonfinite"),
    ("round", np.nan, "round must not be missing"),
    ("show_id", np.nan, "show_id and round"),
    ("board_ev", 999, "disagrees with remaining_board"),
    ("n_remaining", 999, "n_remaining disagrees"),
])
def test_invalid_offer_values(clean_show, column, value, message):
    clean_show.loc[1, column] = value
    with pytest.raises(ValueError, match=message):
        build_features(clean_show)


@pytest.mark.parametrize("value", [None, "bad", "42", "[1, -2]", "[1, 1e999]"])
def test_invalid_board_values_have_context(clean_show, value):
    clean_show.loc[1, "remaining_board"] = value
    with pytest.raises(ValueError, match="Invalid remaining_board"):
        build_features(clean_show)


def test_duplicate_non_offer_round_raises(clean_show):
    duplicated = pd.concat([clean_show, clean_show.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        build_features(duplicated)


def test_string_round_order_is_numeric(clean_show):
    df = clean_show.assign(round=["2", "10"])
    row = build_features(df).iloc[0]
    assert row["round"] == 10
    assert row["board_ev_prev"] == 2720
    assert row["luck_undefined"]


def test_csv_boards_match_in_memory_lists(clean_show):
    from ast import literal_eval

    in_memory = clean_show.copy()
    in_memory["remaining_board"] = in_memory["remaining_board"].map(literal_eval)
    expected = build_features(clean_show).drop(columns="remaining_board")
    actual = build_features(in_memory).drop(columns="remaining_board")
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("prizes", [[2], [1, 2], [1, 1, 1]])
def test_small_or_constant_board_skew_is_missing(clean_show, prizes):
    clean_show.loc[1, "remaining_board"] = str(prizes)
    clean_show.loc[1, "board_ev"] = np.mean(prizes)
    clean_show.loc[1, "n_remaining"] = len(prizes)
    clean_show.loc[1, "offer_ev_ratio"] = 1500 / np.mean(prizes)
    row = build_features(clean_show).iloc[0]
    assert pd.isna(row["board_skew"])
    assert row["prob_above_ev"] == sum(p > np.mean(prizes) for p in prizes) / len(prizes)


def test_terminal_board_with_offer_is_excluded(clean_show):
    clean_show.loc[1, "remaining_board"] = "[]"
    clean_show.loc[1, "board_ev"] = np.nan
    clean_show.loc[1, "n_remaining"] = 0
    assert build_features(clean_show).empty


def test_duplicate_validation_survives_optimized_python():
    root = Path(__file__).resolve().parents[2]
    code = """
import pandas as pd
from features import build_features
df = pd.DataFrame({
    'show_id': [1, 1], 'round': [1, 1], 'board_ev': [2., 2.],
    'remaining_board': ['[1,3]', '[1,3]'], 'bank_offer': [1., 1.],
    'decision': [0, 0], 'offer_ev_ratio': [.5, .5],
})
try:
    build_features(df)
except ValueError as error:
    if 'Duplicate' not in str(error):
        raise
else:
    raise RuntimeError('Duplicate validation disabled by python -O')
"""
    result = subprocess.run(
        [sys.executable, "-O", "-c", code], capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("rounds", [[2], [1, 3]])
def test_round_gap_nulls_luck(clean_show, rounds):
    df = clean_show.iloc[-len(rounds):].assign(round=rounds)
    result = build_features(df)
    assert result["normalized_luck"].isna().all()
    assert result["luck_undefined"].all()


def test_corrupted_offer_ratio_raises(clean_show):
    clean_show.loc[clean_show["bank_offer"].notna(), "offer_ev_ratio"] = 99
    with pytest.raises(ValueError, match="offer_ev_ratio disagrees"):
        build_features(clean_show)
