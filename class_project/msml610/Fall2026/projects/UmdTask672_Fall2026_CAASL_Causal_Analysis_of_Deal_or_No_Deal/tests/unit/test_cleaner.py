"""Filtering tests use synthetic games without dataset-size assertions."""

import pandas as pd
import numpy as np
import pytest

from cleaner import clean


def test_series_filter_excludes_210(clean_show):
    assert clean(clean_show.assign(series_id=2.1)).empty


@pytest.mark.parametrize("series_id", [2.0, 3.0, 3.1])
def test_series_filter_keeps_200_300_310(clean_show, series_id):
    assert len(clean(clean_show.assign(series_id=series_id))) == 2


def test_show_level_contamination(contaminated_show):
    assert clean(contaminated_show).empty


def test_contamination_does_not_spill(contaminated_show, multi_show_df):
    df = pd.concat([
        contaminated_show,
        multi_show_df.loc[multi_show_df["show_id"].eq(2)],
    ], ignore_index=True)
    assert set(clean(df)["show_id"]) == {2}


def test_no_mutation_of_input(multi_show_df):
    original = multi_show_df.copy(deep=True)
    clean(multi_show_df)
    pd.testing.assert_frame_equal(multi_show_df, original)


@pytest.mark.parametrize("flag, dtype", [(np.nan, "float64"), (pd.NA, "Int64")])
def test_unknown_special_flags_are_clean(clean_show, flag, dtype):
    df = clean_show.copy()
    df["special_show"] = pd.Series([flag, 0], dtype=dtype)
    assert clean(df)["show_id"].nunique() == 1


@pytest.mark.parametrize("flag", [1, 2, 3])
def test_all_known_special_flags_exclude_show(clean_show, flag):
    df = clean_show.copy()
    df.loc[0, "special_show"] = flag
    assert clean(df).empty


def test_cleaner_missing_show_id_fails(clean_show):
    with pytest.raises(ValueError, match="missing show_id"):
        clean(clean_show.assign(show_id=np.nan))


def test_mixed_series_detected_before_filter(multi_show_df):
    extra = multi_show_df.iloc[[0]].assign(series_id=2.10)
    df = pd.concat([multi_show_df, extra])
    with pytest.raises(ValueError, match="inconsistent series_id"):
        clean(df)
