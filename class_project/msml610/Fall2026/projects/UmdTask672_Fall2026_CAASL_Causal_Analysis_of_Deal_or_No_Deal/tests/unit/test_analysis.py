"""Regression validation, clustered inference, and failed-fit safeguards."""

import numpy as np
import pandas as pd
import pytest

from analysis import _fit_clustered, run_analysis


def test_clustered_fit_and_probability_scale_report(model_sample, capsys):
    original = model_sample.copy(deep=True)
    result = run_analysis(model_sample)
    output = capsys.readouterr().out
    assert result.mle_retvals["converged"]
    assert result.cov_type == "cluster"
    assert result.nobs == 450
    assert "log_odds_coef" in output and "dy/dx" in output
    assert "Series 3.10" in output
    assert np.isfinite(result.get_margeff(at="mean").margeff).all()
    pd.testing.assert_frame_equal(model_sample, original)


def test_inconsistent_luck_flag_rejected(model_sample):
    model_sample.loc[1, "luck_undefined"] = True
    with pytest.raises(ValueError, match="disagrees"):
        run_analysis(model_sample)


def test_string_flags_rejected(model_sample):
    model_sample["luck_undefined"] = "False"
    with pytest.raises(ValueError, match="boolean flags"):
        run_analysis(model_sample)


@pytest.mark.parametrize("kind, message", [
    ("one_class", "both Deal and No Deal"),
    ("one_cluster", "at least two shows"),
    ("collinear", "rank deficient"),
])
def test_invalid_model_samples(model_sample, kind, message):
    df = model_sample.loc[~model_sample["luck_undefined"]].copy()
    if kind == "one_class":
        df["decision"] = 0
    elif kind == "one_cluster":
        df["show_id"] = 1
    else:
        df["n_remaining"] = 11 - df["round"]
    with pytest.raises(ValueError, match=message):
        _fit_clustered(df)


@pytest.mark.filterwarnings("ignore:overflow encountered in exp:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Maximum Likelihood optimization failed")
def test_separation_never_returns_an_unreliable_fit(model_sample, capsys):
    df = model_sample.loc[~model_sample["luck_undefined"]].copy()
    df["decision"] = df["normalized_luck"].gt(0).astype(int)
    with pytest.raises(RuntimeError, match="converge|separation"):
        run_analysis(df)
    assert "Baseline logit (standard errors" not in capsys.readouterr().out


def test_empty_model_sample_is_rejected(model_sample):
    with pytest.raises(ValueError, match="No offer rows"):
        run_analysis(model_sample.iloc[:0])
