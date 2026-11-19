
"""Tests for housing dataset validation and preprocessing."""

import numpy as np
import pandas as pd
import pytest

from housing_data import (
    check_duplicates,
    check_missing_values,
    preprocess_housing_data,
    validate_housing_data,
)


@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        "X1": [1.0, 2.0, 3.0, 4.0],
        "X2": [10.0, 20.0, 30.0, 40.0],
    })
    y = pd.Series([100.0, 200.0, 300.0, 400.0], name="price")
    return X, y


def test_valid_data(sample_data):
    X, y = sample_data

    report = validate_housing_data(X, y)

    assert report["valid"] is True
    assert report["rows"] == 4
    assert report["features"] == 2


def test_missing_required_column(sample_data):
    X, y = sample_data

    with pytest.raises(ValueError, match="Missing required features"):
        validate_housing_data(
            X, y, required_features=["X1", "X2", "X3"]
        )


def test_missing_values(sample_data):
    X, y = sample_data
    X.loc[1, "X1"] = np.nan

    report = check_missing_values(X, y)

    assert report["total_feature_missing"] == 1

    with pytest.raises(ValueError, match="Missing values"):
        validate_housing_data(X, y)


def test_duplicate_detection():
    X = pd.DataFrame({
        "X1": [1.0, 1.0, 2.0],
    })
    y = pd.Series([100.0, 100.0, 200.0])

    report = check_duplicates(X, y)

    assert report["duplicate_feature_rows"] == 1
    assert report["duplicate_records"] == 1


def test_preprocessing_drops_missing_rows(sample_data):
    X, y = sample_data
    X.loc[1, "X1"] = np.nan

    clean_X, clean_y = preprocess_housing_data(
        X, y, missing_strategy="drop"
    )

    assert len(clean_X) == 3
    assert len(clean_y) == 3
    assert not clean_X.isna().any().any()


def test_preprocessing_drops_duplicates():
    X = pd.DataFrame({
        "X1": [1.0, 1.0, 2.0],
    })
    y = pd.Series([100.0, 100.0, 200.0])

    clean_X, clean_y = preprocess_housing_data(
        X, y, duplicate_strategy="drop"
    )

    assert len(clean_X) == 2
    assert len(clean_y) == 2


def test_invalid_feature_values(sample_data):
    X, y = sample_data
    X.loc[0, "X1"] = np.inf

    with pytest.raises(ValueError, match="infinite"):
        validate_housing_data(X, y)


def test_invalid_target_type(sample_data):
    X, y = sample_data
    y = y.astype(str)

    with pytest.raises(ValueError, match="Target must be numeric"):
        validate_housing_data(X, y)


def test_mismatched_rows(sample_data):
    X, y = sample_data

    with pytest.raises(ValueError, match="row counts"):
        validate_housing_data(X, y.iloc[:-1])


def test_invalid_preprocessing_strategy(sample_data):
    X, y = sample_data

    with pytest.raises(ValueError, match="Invalid missing_strategy"):
        preprocess_housing_data(
            X, y, missing_strategy="impute"
        )
