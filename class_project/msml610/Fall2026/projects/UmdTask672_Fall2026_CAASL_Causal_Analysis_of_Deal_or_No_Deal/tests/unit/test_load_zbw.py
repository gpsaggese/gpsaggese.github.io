"""Synthetic text inputs exercise loader validation without real data."""

import pandas as pd
import pytest

from load_zbw import ZBW_COLUMNS, load_zbw


@pytest.fixture
def write_record(tmp_path):
    def write(**changes):
        values = dict.fromkeys(ZBW_COLUMNS, -999)
        values.update(year=2004, month=2, day=2, show_id=1, series_id=2.0, round=1)
        values.update(changes)
        path = tmp_path / "deal-data.txt"
        path.write_text("\t".join(str(values[c]) for c in ZBW_COLUMNS) + "\n")
        return path
    return write


def test_round_and_series_casts(write_record):
    df = load_zbw(write_record(round="10", series_id="3.10"))
    assert str(df["round"].dtype) == "Int64"
    assert df.iloc[0]["round"] == 10
    assert df.iloc[0]["series_id"] == 3.1
    assert pd.isna(df.iloc[0]["bank_offer"])


@pytest.mark.parametrize("column, value, message", [
    ("show_id", -999, "missing show_id"),
    ("series_id", "bad", "series_id contains"),
    ("series_id", -999, "series_id contains"),
    ("series_id", "inf", "series_id contains"),
    ("round", "bad", "invalid round"),
    ("round", -999, "invalid round"),
    ("round", 1.5, "invalid round"),
    ("round", 0, "invalid round"),
])
def test_invalid_identifiers_and_rounds(write_record, column, value, message):
    with pytest.raises(ValueError, match=message):
        load_zbw(write_record(**{column: value}))


@pytest.mark.parametrize("n_fields", [19, 21])
def test_incomplete_raw_record_raises(write_record, n_fields):
    path = write_record(case_3=5000)
    fields = path.read_text().split()
    if n_fields == 19:
        fields.pop(ZBW_COLUMNS.index("case_3"))
    else:
        fields.append("5000")
    path.write_text("\t".join(fields) + "\n")
    with pytest.raises(ValueError, match=f"incomplete record.*expected 20 fields, got {n_fields}"):
        load_zbw(path)
