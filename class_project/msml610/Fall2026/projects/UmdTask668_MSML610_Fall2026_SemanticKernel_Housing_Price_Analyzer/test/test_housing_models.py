
"""Tests for the Housing Price Analyzer models."""

import numpy as np
import pandas as pd

from housing_models import train_models


def test_train_models():
    """Both models should train and return valid scores."""
    rng = np.random.default_rng(42)

    X = pd.DataFrame(
        rng.normal(size=(100, 6)),
        columns=[f"X{i}" for i in range(1, 7)],
    )

    y = pd.Series(
        3 * X["X1"] - 2 * X["X2"]
        + rng.normal(scale=0.1, size=100),
        name="price",
    )

    models, comparison = train_models(X, y)

    assert set(models) == {
        "Linear Regression",
        "Random Forest",
    }
    assert list(comparison.columns) == ["MAE", "RMSE", "R2"]
    assert comparison.shape == (2, 3)
    assert np.isfinite(comparison.to_numpy()).all()
    assert (comparison["MAE"] >= 0).all()
    assert (comparison["RMSE"] >= 0).all()


def test_reproducible_results():
    """A fixed random seed should produce repeatable scores."""
    rng = np.random.default_rng(42)

    X = pd.DataFrame(
        rng.normal(size=(100, 6)),
        columns=[f"X{i}" for i in range(1, 7)],
    )
    y = pd.Series(2 * X["X1"] + X["X2"], name="price")

    _, first = train_models(X, y)
    _, second = train_models(X, y)

    pd.testing.assert_frame_equal(first, second)
