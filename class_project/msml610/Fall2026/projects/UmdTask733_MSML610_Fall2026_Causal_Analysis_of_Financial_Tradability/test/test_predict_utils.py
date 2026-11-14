"""
Import as:

import test.test_predict_utils as ttprut
"""

import logging

import helpers.hunit_test as hunitest
import numpy as np
import pandas as pd
import predict_utils as putil

_LOG = logging.getLogger(__name__)


def _get_index(start: str, end: str) -> pd.DatetimeIndex:
    """
    Build a 5-minute grid of bar times.

    :param start: first bar time (UTC)
    :param end: last bar time (UTC)
    :return: bar times
    """
    index = pd.date_range(start, end, freq="5min", tz="UTC", name="open_time")
    return index


# #############################################################################
# Test_split_walk_forward
# #############################################################################


class Test_split_walk_forward(hunitest.TestCase):
    """
    Test `predict_utils.split_walk_forward()`.
    """

    def test1(self) -> None:
        """
        Test that training stops one horizon before the test year.
        """
        # Prepare inputs.
        index = _get_index("2022-12-31 20:00", "2023-01-01 04:00")
        # Run test.
        is_train, is_test = putil.split_walk_forward(index, 2023, "1h")
        # Check outputs.
        # The last training label goes from 22:55 to 23:55 on 2022-12-31,
        # which is before the test year starts.
        self.assertEqual(
            index[is_train][-1], pd.Timestamp("2022-12-31 22:55", tz="UTC")
        )
        self.assertEqual(
            index[is_test][0], pd.Timestamp("2023-01-01 00:00", tz="UTC")
        )
        # The hour before the test year is in neither set.
        self.assertEqual(int((~is_train & ~is_test).sum()), 12)

    def test2(self) -> None:
        """
        Test that the test set stops at the end of the year.
        """
        # Prepare inputs.
        index = _get_index("2023-12-31 23:00", "2024-01-01 01:00")
        # Run test.
        _, is_test = putil.split_walk_forward(index, 2023, "5m")
        # Check outputs.
        self.assertEqual(
            index[is_test][-1], pd.Timestamp("2023-12-31 23:55", tz="UTC")
        )


# #############################################################################
# Test_predict_momentum
# #############################################################################


class Test_predict_momentum(hunitest.TestCase):
    """
    Test `predict_utils.predict_momentum()`.
    """

    def test1(self) -> None:
        """
        Test that the position follows the sign of the past return.
        """
        # Prepare inputs.
        features = pd.DataFrame({"ret_12": [0.01, -0.02, 0.0, np.nan]})
        # Run test.
        pos = putil.predict_momentum(features, "ret_12")
        # Check outputs.
        self.assertEqual(pos.astype(str).tolist(), ["1.0", "-1.0", "nan", "nan"])


# #############################################################################
# Test_predict_logistic
# #############################################################################


class Test_predict_logistic(hunitest.TestCase):
    """
    Test `predict_utils.predict_logistic()`.
    """

    def test1(self) -> None:
        """
        Test that the labels of the test year do not change its predictions
        (no look-ahead).
        """
        # Prepare inputs.
        rng = np.random.default_rng(0)
        index = _get_index("2022-12-25", "2023-01-03")
        features = pd.DataFrame(
            {
                "x1": rng.normal(size=len(index)),
                "x2": rng.normal(size=len(index)),
            },
            index=index,
        )
        up = (features["x1"] + rng.normal(size=len(index)) > 0).astype(float)
        labels = pd.DataFrame({"label_1h": up})
        changed = labels.copy()
        changed.loc[changed.index >= "2023-01-01", "label_1h"] = 1.0
        # Run test.
        pos = putil.predict_logistic(features, labels, "1h", [2023])
        pos_changed = putil.predict_logistic(features, changed, "1h", [2023])
        # Check outputs.
        pd.testing.assert_series_equal(pos, pos_changed)
        # Only the test year gets a position.
        self.assertTrue(pos[pos.index < "2023-01-01"].isna().all())
        self.assertFalse(pos[pos.index >= "2023-01-01"].isna().any())

    def test2(self) -> None:
        """
        Test that the model learns a feature that predicts the label.
        """
        # Prepare inputs.
        rng = np.random.default_rng(0)
        index = _get_index("2022-12-25", "2023-01-03")
        features = pd.DataFrame({"x1": rng.normal(size=len(index))}, index=index)
        labels = pd.DataFrame({"label_1h": (features["x1"] > 0).astype(float)})
        # Run test.
        pos = putil.predict_logistic(features, labels, "1h", [2023])
        # Check outputs.
        is_test = pos.notna()
        expected = np.where(features["x1"][is_test] > 0, 1.0, -1.0)
        self.assertGreater((pos[is_test].to_numpy() == expected).mean(), 0.99)


def _get_skewed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build data where the frequent move and the big move go opposite ways.

    When `x1` > 0, the price goes up 10 bp 70% of the time and down 100 bp
    30% of the time, and the reverse when `x1` < 0. So `x1` > 0 predicts
    more up moves, but a loss on average.

    :return: features and labels for a 1h horizon
    """
    rng = np.random.default_rng(0)
    index = _get_index("2022-12-25", "2023-01-03")
    x1 = rng.normal(size=len(index))
    is_frequent = rng.uniform(size=len(index)) < 0.7
    direction = np.where(x1 > 0, 1.0, -1.0)
    fwd_ret = np.where(is_frequent, 0.001 * direction, -0.01 * direction)
    features = pd.DataFrame({"x1": x1}, index=index)
    labels = pd.DataFrame(
        {"fwd_ret_1h": fwd_ret, "label_1h": (fwd_ret > 0).astype(float)},
        index=index,
    )
    return features, labels


# #############################################################################
# Test_predict_with_target
# #############################################################################


class Test_predict_with_target(hunitest.TestCase):
    """
    Test `predict_utils.predict_with_target()`.
    """

    def test1(self) -> None:
        """
        Test that the returns of the test year do not change its predictions,
        for every target (no look-ahead).
        """
        # Prepare inputs.
        features, labels = _get_skewed_data()
        changed = labels.copy()
        is_test_year = changed.index >= "2023-01-01"
        changed.loc[is_test_year, "fwd_ret_1h"] = 0.05
        changed.loc[is_test_year, "label_1h"] = 1.0
        for target in putil.TARGETS:
            # Run test.
            pos = putil.predict_with_target(
                features, labels, "1h", [2023], target=target
            )
            pos_changed = putil.predict_with_target(
                features, changed, "1h", [2023], target=target
            )
            # Check outputs.
            pd.testing.assert_series_equal(pos, pos_changed)
            self.assertTrue(pos[~is_test_year].isna().all())
            self.assertFalse(pos[is_test_year].isna().any())

    def test2(self) -> None:
        """
        Test that only the sign target follows the frequent move.
        """
        # Prepare inputs.
        features, labels = _get_skewed_data()
        is_up = features["x1"] > 0
        for target in putil.TARGETS:
            # Run test.
            pos = putil.predict_with_target(
                features, labels, "1h", [2023], target=target, min_move=0.005
            )
            # Check outputs.
            is_test = pos.notna()
            share_long = (pos[is_test & is_up] > 0).mean()
            if target == "sign":
                # Long when x1 > 0, where the price goes up 70% of the time.
                self.assertGreater(share_long, 0.9, msg=target)
            else:
                # Short when x1 > 0, where the big moves are down.
                self.assertLess(share_long, 0.1, msg=target)

    def test3(self) -> None:
        """
        Test that an unknown target raises an error.
        """
        # Prepare inputs.
        features, labels = _get_skewed_data()
        # Run test and check outputs.
        with self.assertRaises(ValueError):
            putil.predict_with_target(
                features, labels, "1h", [2023], target="volume"
            )


# #############################################################################
# Test_evaluate_positions
# #############################################################################


class Test_evaluate_positions(hunitest.TestCase):
    """
    Test `predict_utils.evaluate_positions()`.
    """

    def test1(self) -> None:
        """
        Test the hit rates and PnL on a small example.
        """
        # Prepare inputs.
        pos = pd.Series([1.0, -1.0, 1.0, 1.0, np.nan])
        fwd_ret = pd.Series([0.01, 0.02, -0.03, 0.0, 0.05])
        # Run test.
        stats = putil.evaluate_positions(pos, fwd_ret, cost=0.001)
        # Check outputs.
        # The zero return and the missing position are not trades.
        self.assertEqual(stats["n"], 3)
        self.assertAlmostEqual(stats["hit_rate"], 1 / 3)
        self.assertAlmostEqual(stats["weighted_hit_rate"], 0.01 / 0.06)
        # PnL per trade: 0.01, -0.02, -0.03.
        self.assertAlmostEqual(stats["gross_bps"], -0.04 / 3 * 1e4)
        self.assertAlmostEqual(stats["net_bps"], (-0.04 / 3 - 0.001) * 1e4)
        self.assertAlmostEqual(stats["total_net"], -0.043)
        # The cost raises p* by cost / (2 mean |r|).
        self.assertAlmostEqual(
            stats["p_star_cost"] - stats["p_star"], 0.001 / (2 * 0.02)
        )


# #############################################################################
# Test_sweep_costs
# #############################################################################


class Test_sweep_costs(hunitest.TestCase):
    """
    Test `predict_utils.sweep_costs()`.
    """

    def test1(self) -> None:
        """
        Test that each cost lowers the PnL per trade by the cost.
        """
        # Prepare inputs.
        pos = pd.Series([1.0, -1.0, 1.0])
        fwd_ret = pd.Series([0.01, -0.02, 0.03])
        costs = [0.0, 0.0005, 0.001]
        # Run test.
        df = putil.sweep_costs(pos, fwd_ret, costs)
        # Check outputs.
        self.assertEqual(df["cost_bps"].tolist(), [0.0, 5.0, 10.0])
        # The PnL before costs is 0.02 per trade at every cost.
        self.assertEqual(df["gross_bps"].round(6).tolist(), [200.0] * 3)
        self.assertEqual(df["net_bps"].round(6).tolist(), [200.0, 195.0, 190.0])


# #############################################################################
# Test_evaluate_by_group
# #############################################################################


class Test_evaluate_by_group(hunitest.TestCase):
    """
    Test `predict_utils.evaluate_by_group()`.
    """

    def test1(self) -> None:
        """
        Test that each group is measured on its own trades only.
        """
        # Prepare inputs.
        pos = pd.Series([1.0, 1.0, -1.0, 1.0, 1.0])
        fwd_ret = pd.Series([0.01, 0.03, 0.02, -0.01, 0.05])
        # The last trade has no group, so it is left out.
        groups = pd.Series(["a", "a", "b", "b", np.nan])
        # Run test.
        df = putil.evaluate_by_group(pos, fwd_ret, groups)
        # Check outputs.
        self.assertEqual(df.index.tolist(), ["a", "b"])
        self.assertEqual(df["n"].tolist(), [2, 2])
        self.assertEqual(df["share"].tolist(), [0.5, 0.5])
        # Group a: PnL 0.01 and 0.03. Group b: PnL -0.02 and -0.01.
        np.testing.assert_allclose(df["gross_bps"], [200.0, -150.0])
        np.testing.assert_allclose(df["hit_rate"], [1.0, 0.0])
        np.testing.assert_allclose(df["mean_abs_bps"], [200.0, 150.0])
        expected_se = np.std([0.01, 0.03], ddof=1) / np.sqrt(2) * 1e4
        self.assertAlmostEqual(df.loc["a", "se_bps"], expected_se)
