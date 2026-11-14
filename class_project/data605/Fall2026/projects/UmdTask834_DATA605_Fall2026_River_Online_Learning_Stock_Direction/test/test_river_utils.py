"""
Fast unit tests for `river_utils`, run on synthetic data (no network).

Import as:

import test.test_river_utils as ttrivut
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest
from river import compose, linear_model, preprocessing

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import river_utils as rivutil  # noqa: E402


def _make_model():
    return compose.Pipeline(
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression(),
    )


def _panel(**kwargs) -> pd.DataFrame:
    raw = rivutil.make_synthetic_ohlcv(**kwargs)
    feats = rivutil.build_features(rivutil.clean_ohlcv(raw))
    return rivutil.add_direction_label(feats)


# #############################################################################
# Test_features
# #############################################################################


class Test_features:
    def test_no_lookahead(self) -> None:
        """
        Features up to day `t` must not change when data after `t` is deleted.
        """
        raw = rivutil.make_synthetic_ohlcv(n_days=300, seed=1)
        cutoff = raw["date"].unique()[200]
        full = rivutil.build_features(raw)
        trunc = rivutil.build_features(raw[raw["date"] <= cutoff])
        cols = rivutil.feature_columns(full)
        # The truncated panel loses its last row per ticker (no label), so
        # compare on the dates both panels share.
        merged = full.merge(trunc, on=["date", "ticker"], suffixes=("", "_t"))
        assert len(merged) > 100
        for c in cols:
            np.testing.assert_allclose(merged[c], merged[f"{c}_t"], err_msg=c)

    def test_target_is_next_day_return(self) -> None:
        raw = rivutil.make_synthetic_ohlcv(n_days=120, seed=2)
        feats = rivutil.build_features(raw)
        g = feats[feats["ticker"] == "AAA"].reset_index(drop=True)
        # Tomorrow's `ret_lag0` is today's `target_ret`.
        np.testing.assert_allclose(
            g["target_ret"].iloc[:-1].to_numpy(), g["ret_lag0"].iloc[1:].to_numpy()
        )

    def test_dead_zone_removes_small_moves(self) -> None:
        feats = rivutil.build_features(rivutil.make_synthetic_ohlcv(seed=3))
        out = rivutil.add_direction_label(feats, dead_zone=0.005)
        assert (out["target_ret"].abs() > 0.005).all()


# #############################################################################
# Test_prequential
# #############################################################################


class Test_prequential:
    def test_noise_is_unpredictable(self) -> None:
        """
        On a pure random walk, accuracy must stay near chance. A large edge
        here would indicate leakage.
        """
        panel = _panel(n_days=1500, seed=4, momentum=0.0)
        preds = rivutil.run_prequential(
            _make_model(), panel, rivutil.feature_columns(panel), warmup_days=50
        )
        ci = rivutil.block_bootstrap_accuracy(preds, n_boot=300)
        assert ci["ci_low"] < 0.5 + 0.03
        assert abs(ci["accuracy"] - 0.5) < 0.04

    def test_momentum_is_learned(self) -> None:
        """
        With strong AR(1) returns the model must find the signal.
        """
        panel = _panel(n_days=1500, seed=5, momentum=0.4)
        preds = rivutil.run_prequential(
            _make_model(), panel, rivutil.feature_columns(panel), warmup_days=50
        )
        assert rivutil.summarize_predictions(preds)["accuracy"] > 0.58

    def test_labels_are_delayed_one_day(self) -> None:
        """
        On each day, the model must not have seen that day's labels.
        """

        class Spy:
            def __init__(self) -> None:
                self.learned = 0
                self.seen_at_predict = []

            def learn_one(self, x, y) -> None:
                self.learned += 1

            def predict_proba_one(self, x):
                self.seen_at_predict.append(self.learned)
                return {True: 0.5, False: 0.5}

        panel = _panel(n_days=200, seed=6)
        spy = Spy()
        rivutil.run_prequential(spy, panel, rivutil.feature_columns(panel))
        n_tickers = panel["ticker"].nunique()
        # Day k: k * n_tickers labels learned, for every ticker on that day.
        n_days = len(spy.seen_at_predict) // n_tickers
        expected = np.repeat(np.arange(n_days) * n_tickers, n_tickers)
        np.testing.assert_array_equal(spy.seen_at_predict, expected)


# #############################################################################
# Test_stats
# #############################################################################


class Test_stats:
    def test_mcnemar_identical_models(self) -> None:
        panel = _panel(n_days=300, seed=7)
        cols = rivutil.feature_columns(panel)
        a = rivutil.run_prequential(_make_model(), panel, cols)
        b = rivutil.run_prequential(_make_model(), panel, cols)
        res = rivutil.mcnemar_test(a, b)
        assert res["p_value"] == pytest.approx(1.0)


# #############################################################################
# Test_baselines
# #############################################################################


class Test_baselines:
    def test_walk_forward_trains_only_on_past(self) -> None:
        """
        At every refit, all training dates must precede all scored dates.
        """
        from sklearn.linear_model import LogisticRegression

        fits = []

        class SpyModel(LogisticRegression):
            def fit(self, X, y):
                fits.append(len(X))
                return super().fit(X, y)

        panel = _panel(n_days=400, seed=8)
        cols = rivutil.feature_columns(panel)
        preds = rivutil.run_walk_forward(
            SpyModel, panel, cols, warmup_days=100
        )
        # Training sets only grow (expanding window), one fit per month.
        assert fits == sorted(fits)
        n_months = preds["date"].dt.to_period("M").nunique()
        assert len(fits) == n_months
        # The first fit uses exactly the warm-up days.
        n_tickers = panel["ticker"].nunique()
        assert fits[0] == 100 * n_tickers

    def test_walk_forward_scores_same_rows_as_prequential(self) -> None:
        from sklearn.linear_model import LogisticRegression

        panel = _panel(n_days=400, seed=9)
        cols = rivutil.feature_columns(panel)
        wf = rivutil.run_walk_forward(
            LogisticRegression, panel, cols, warmup_days=100
        )
        pq = rivutil.run_prequential(
            _make_model(), panel, cols, warmup_days=100
        )
        key = ["date", "ticker"]
        pd.testing.assert_frame_equal(
            wf[key].reset_index(drop=True), pq[key].reset_index(drop=True)
        )

    def test_persistence_repeats_today(self) -> None:
        model = rivutil.PersistenceClassifier()
        assert model.predict_proba_one({"ret_lag0": 0.01})[True] > 0.5
        assert model.predict_proba_one({"ret_lag0": -0.01})[True] < 0.5
