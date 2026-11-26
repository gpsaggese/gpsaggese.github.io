"""Board reconstruction uses raw rounds, not precomputed board features."""

import numpy as np
import pytest
import pandas as pd

from build_dataset import DERIVED_COLUMNS, MISMATCH_COLUMNS, STANDARD_BOARD, build_dataset
from features import build_features


def test_reconstruct_raw_rounds(RAW_ROUND_FIXTURE):
    reconstructed, mismatches = build_dataset(RAW_ROUND_FIXTURE)
    assert mismatches.empty
    assert reconstructed["n_remaining"].tolist() == [25, 24]
    expected = STANDARD_BOARD.copy()
    for row, removed in zip(reconstructed.itertuples(), [200000, 100000]):
        expected.remove(removed)
        assert row.remaining_board == expected
        assert row.board_ev == pytest.approx(np.mean(expected))
        assert row.offer_ev_ratio == pytest.approx(row.bank_offer / row.board_ev)


def test_mismatch_excludes_entire_show(RAW_ROUND_FIXTURE):
    RAW_ROUND_FIXTURE.loc[1, "case_1"] = 999999
    reconstructed, mismatches = build_dataset(RAW_ROUND_FIXTURE)
    assert reconstructed.empty
    assert mismatches["show_id"].tolist() == [1]
    assert reconstructed["show_id"].empty
    assert build_features(reconstructed).empty
    error = mismatches.iloc[0]["error"]
    for detail in ["show=", "series=", "round=", "999999", "remaining board"]:
        assert detail in error


def test_empty_input_preserves_schema(RAW_ROUND_FIXTURE):
    reconstructed, mismatches = build_dataset(RAW_ROUND_FIXTURE.iloc[:0])
    assert reconstructed.empty
    assert set(RAW_ROUND_FIXTURE.columns).issubset(reconstructed.columns)
    assert set(DERIVED_COLUMNS).issubset(reconstructed.columns)
    assert list(mismatches.columns) == MISMATCH_COLUMNS
    assert build_features(reconstructed).empty


@pytest.mark.parametrize("column, values, message", [
    ("series_id", [2., 3.], "mixed series_id"),
    ("series_id", [9., 9.], "Unknown series_id"),
    ("series_id", [pd.NA, pd.NA], "series_id contains"),
    ("series_id", ["bad", "bad"], "series_id contains"),
    ("show_id", [np.nan, 1], "missing show_id"),
    ("round", [1, 1], "Duplicate"),
    ("round", [1, np.nan], "round contains"),
    ("round", [1, 1.5], "positive integers"),
])
def test_invalid_round_metadata_raises(RAW_ROUND_FIXTURE, column, values, message):
    df = RAW_ROUND_FIXTURE.assign(**{column: values})
    with pytest.raises(ValueError, match=message):
        build_dataset(df)


def test_sorted_boards_and_no_input_mutation(RAW_ROUND_FIXTURE):
    df = RAW_ROUND_FIXTURE.iloc[::-1].copy()
    df["round"] = df["round"].astype(str)
    original = df.copy(deep=True)
    reconstructed, _ = build_dataset(df)
    assert reconstructed["round"].tolist() == [1, 2]
    for values in reconstructed.groupby("show_id")["n_remaining"].apply(list):
        assert values == sorted(values, reverse=True)
    pd.testing.assert_frame_equal(df, original)


def test_valid_show_survives_another_show_mismatch(RAW_ROUND_FIXTURE):
    bad = RAW_ROUND_FIXTURE.assign(show_id=99, case_1=999999.)
    reconstructed, mismatches = build_dataset(pd.concat([RAW_ROUND_FIXTURE, bad]))
    assert set(reconstructed["show_id"]) == {1}
    assert set(mismatches["show_id"]) == {99}


def test_duplicate_prizes_removed_one_at_a_time(RAW_ROUND_FIXTURE):
    df = RAW_ROUND_FIXTURE.assign(series_id=2.11, case_1=200000.)
    reconstructed, mismatches = build_dataset(df)
    assert mismatches.empty
    assert reconstructed.iloc[0]["remaining_board"].count(200000) == 1
    assert reconstructed.iloc[1]["remaining_board"].count(200000) == 0


def test_snapshots_are_independent(RAW_ROUND_FIXTURE):
    reconstructed, _ = build_dataset(RAW_ROUND_FIXTURE)
    before = reconstructed.iloc[1]["remaining_board"].copy()
    reconstructed.iloc[0]["remaining_board"].append(123)
    assert reconstructed.iloc[1]["remaining_board"] == before
    assert 123 not in STANDARD_BOARD


@pytest.mark.parametrize("bad_prize", [False, True])
def test_stale_csvs_overwritten(tmp_path, RAW_ROUND_FIXTURE, bad_prize):
    from pathlib import Path
    import subprocess
    import sys
    from load_zbw import ZBW_COLUMNS

    source = tmp_path / "data/raw/zbw/deal-data.txt"
    source.parent.mkdir(parents=True)
    raw = RAW_ROUND_FIXTURE.reindex(columns=ZBW_COLUMNS).assign(
        year=2004, month=2, day=2,
    )
    if bad_prize:
        raw.loc[0, "case_1"] = 999999
    raw.to_csv(source, sep="\t", index=False, header=False, na_rep="-999")

    output_dir = tmp_path / "data/processed"
    output_dir.mkdir(parents=True)
    reconstructed_path = output_dir / "zbw_reconstructed.csv"
    mismatches_path = output_dir / "zbw_board_mismatches.csv"
    reconstructed_path.write_text("show_id,remaining_board\n999,[]\n")
    mismatches_path.write_text("show_id,series_id,error\n999,2.0,stale\n")
    script = Path(__file__).resolve().parents[2] / "src/build_dataset.py"
    result = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    reconstructed = pd.read_csv(reconstructed_path)
    mismatches = pd.read_csv(mismatches_path)
    assert 999 not in reconstructed["show_id"].values
    assert 999 not in mismatches["show_id"].values
    assert len(reconstructed) == (0 if bad_prize else 2)
    assert len(mismatches) == (1 if bad_prize else 0)
