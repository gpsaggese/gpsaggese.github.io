"""
Import as:

import test.test_data_utils as ttdaut
"""

import logging

import numpy as np
import pandas as pd

import helpers.hunit_test as hunitest

import data_utils as dutil

_LOG = logging.getLogger(__name__)


def _get_bars(times: list[str], closes: list[float]) -> pd.DataFrame:
    """
    Build flat 5-minute bars with unit volume.

    :param times: bar open times (UTC), e.g., "2024-01-01 00:05"
    :param closes: close price of each bar
    :return: OHLCV bars indexed by bar open time
    """
    index = pd.DatetimeIndex(times, tz="UTC", name="open_time")
    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": 1.0,
        },
        index=index,
    )
    return df


def _get_random_bars(n_bars: int, *, seed: int = 0) -> pd.DataFrame:
    """
    Build `n_bars` 5-minute bars following a random walk.

    :param n_bars: number of bars
    :param seed: random seed
    :return: OHLCV bars indexed by bar open time
    """
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-01-01", periods=n_bars, freq="5min")
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n_bars)))
    df = _get_bars(times.strftime("%Y-%m-%d %H:%M").tolist(), closes.tolist())
    return df


# #############################################################################
# Test_find_gaps
# #############################################################################


class Test_find_gaps(hunitest.TestCase):
    """
    Test `data_utils.find_gaps()`.
    """

    def test1(self) -> None:
        """
        Test that two missing bars are reported as one gap.
        """
        # Prepare inputs.
        times = [
            "2024-01-01 00:00",
            "2024-01-01 00:05",
            "2024-01-01 00:20",
            "2024-01-01 00:25",
        ]
        bars = _get_bars(times, [1.0, 2.0, 3.0, 4.0])
        # Run test.
        gaps = dutil.find_gaps(bars)
        # Check outputs.
        self.assertEqual(len(gaps), 1)
        self.assertEqual(
            gaps["start"].iloc[0], pd.Timestamp("2024-01-01 00:10Z")
        )
        self.assertEqual(gaps["end"].iloc[0], pd.Timestamp("2024-01-01 00:15Z"))
        self.assertEqual(gaps["n_bars"].iloc[0], 2)

    def test2(self) -> None:
        """
        Test that bars without missing times have no gaps.
        """
        # Prepare inputs.
        bars = _get_random_bars(10)
        # Run test.
        gaps = dutil.find_gaps(bars)
        # Check outputs.
        self.assertEqual(len(gaps), 0)


# #############################################################################
# Test_clean_bars
# #############################################################################


class Test_clean_bars(hunitest.TestCase):
    """
    Test `data_utils.clean_bars()`.
    """

    def test1(self) -> None:
        """
        Test that a short gap is filled with flat bars at the last close.
        """
        # Prepare inputs.
        times = [
            "2024-01-01 00:00",
            "2024-01-01 00:05",
            "2024-01-01 00:20",
            "2024-01-01 00:25",
        ]
        bars = _get_bars(times, [1.0, 2.0, 3.0, 4.0])
        # Run test.
        clean = dutil.clean_bars(bars, max_fill_bars=3)
        # Check outputs.
        self.assertEqual(clean["close"].tolist(), [1.0, 2.0, 2.0, 2.0, 3.0, 4.0])
        self.assertEqual(
            clean["volume"].tolist(), [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
        )
        self.assertEqual(
            clean["is_filled"].tolist(),
            [False, False, True, True, False, False],
        )
        # A filled gap does not start a new segment.
        self.assertEqual(clean["segment_id"].tolist(), [0] * 6)

    def test2(self) -> None:
        """
        Test that a long gap stays empty and starts a new segment.
        """
        # Prepare inputs.
        times = [
            "2024-01-01 00:00",
            "2024-01-01 00:05",
            "2024-01-01 00:20",
            "2024-01-01 00:25",
        ]
        bars = _get_bars(times, [1.0, 2.0, 3.0, 4.0])
        # Run test.
        clean = dutil.clean_bars(bars, max_fill_bars=1)
        # Check outputs.
        self.assertEqual(clean["close"].isna().tolist()[2:4], [True, True])
        self.assertEqual(
            clean["segment_id"].astype(str).tolist(),
            ["0", "0", "<NA>", "<NA>", "1", "1"],
        )
        # The first return after the gap is NaN, not 3 / 2 - 1.
        self.assertTrue(np.isnan(clean["ret"].iloc[4]))
        self.assertAlmostEqual(clean["ret"].iloc[5], 4.0 / 3.0 - 1)

    def test3(self) -> None:
        """
        Test that duplicate timestamps keep the first bar.
        """
        # Prepare inputs.
        times = ["2024-01-01 00:00", "2024-01-01 00:05", "2024-01-01 00:05"]
        bars = _get_bars(times, [1.0, 2.0, 5.0])
        # Run test.
        clean = dutil.clean_bars(bars)
        # Check outputs.
        self.assertEqual(clean["close"].tolist(), [1.0, 2.0])

    def test4(self) -> None:
        """
        Test that a large jump is flagged as extreme but kept.
        """
        # Prepare inputs.
        bars = _get_random_bars(100)
        bars.iloc[80:, :4] *= 1.05
        # Run test.
        clean = dutil.clean_bars(bars, extreme_std=10.0)
        # Check outputs.
        self.assertEqual(
            clean.index[clean["is_extreme"]].tolist(), [bars.index[80]]
        )
        self.assertEqual(len(clean), 100)

    def test5(self) -> None:
        """
        Test that a zero-volume bar is flagged.
        """
        # Prepare inputs.
        bars = _get_random_bars(5)
        bars.iloc[2, bars.columns.get_loc("volume")] = 0.0
        # Run test.
        clean = dutil.clean_bars(bars)
        # Check outputs.
        self.assertEqual(
            clean["is_zero_volume"].tolist(), [False, False, True, False, False]
        )

    def test6(self) -> None:
        """
        Test that changing the data after time t does not change the output
        at or before t (no look-ahead).
        """
        # Prepare inputs.
        bars = _get_random_bars(600)
        t = bars.index[400]
        changed = bars.copy()
        changed.loc[changed.index > t, ["open", "high", "low", "close"]] *= 1.5
        # Run test.
        clean = dutil.clean_bars(bars)
        clean_changed = dutil.clean_bars(changed)
        # Check outputs.
        pd.testing.assert_frame_equal(clean.loc[:t], clean_changed.loc[:t])


# #############################################################################
# Test_compute_quality_table
# #############################################################################


class Test_compute_quality_table(hunitest.TestCase):
    """
    Test `data_utils.compute_quality_table()`.
    """

    def test1(self) -> None:
        """
        Test the counts on bars with one duplicate and one long gap.
        """
        # Prepare inputs.
        times = [
            "2024-01-01 00:00",
            "2024-01-01 00:05",
            "2024-01-01 00:05",
            "2024-01-01 00:20",
            "2024-01-01 00:25",
        ]
        raw = _get_bars(times, [1.0, 2.0, 2.0, 3.0, 4.0])
        clean = dutil.clean_bars(raw, max_fill_bars=1)
        # Run test.
        stats = dutil.compute_quality_table(raw, clean)
        # Check outputs.
        self.assertEqual(stats["n_grid_bars"], 6)
        self.assertEqual(stats["n_raw_bars"], 5)
        self.assertEqual(stats["n_duplicates"], 1)
        self.assertEqual(stats["n_gaps"], 1)
        self.assertEqual(stats["n_missing_bars"], 2)
        self.assertEqual(stats["n_long_gaps"], 1)
        self.assertEqual(stats["n_segments"], 2)


def _get_gap_bars() -> pd.DataFrame:
    """
    Build 40 random 5-minute bars with a 1-hour gap after the first 20.

    :return: bars as returned by `clean_bars()`, with 2 segments
    """
    bars = _get_random_bars(52)
    # Drop bars 20 to 31, so that the gap is too long to fill.
    bars = pd.concat([bars.iloc[:20], bars.iloc[32:]])
    clean = dutil.clean_bars(bars)
    return clean


# #############################################################################
# Test_compute_features
# #############################################################################


class Test_compute_features(hunitest.TestCase):
    """
    Test `data_utils.compute_features()`.
    """

    def test1(self) -> None:
        """
        Test that changing the data after time t does not change the features
        at or before t (no look-ahead).
        """
        # Prepare inputs.
        rng = np.random.default_rng(1)
        bars = _get_random_bars(2500)
        bars["volume"] = rng.uniform(1, 10, len(bars))
        other = _get_random_bars(2500, seed=2)
        t = bars.index[2200]
        changed = bars.copy()
        changed.loc[changed.index > t, ["open", "high", "low", "close"]] *= 1.5
        changed.loc[changed.index > t, "volume"] *= 3
        other_changed = other.copy()
        other_changed.loc[other_changed.index > t, "close"] *= 0.5
        # Run test.
        features = dutil.compute_features(
            dutil.clean_bars(bars), other=dutil.clean_bars(other)
        )
        features_changed = dutil.compute_features(
            dutil.clean_bars(changed), other=dutil.clean_bars(other_changed)
        )
        # Check outputs.
        pd.testing.assert_frame_equal(features.loc[:t], features_changed.loc[:t])
        # The 7-day features need 2016 bars, so they exist at t.
        self.assertFalse(features.loc[t].isna().any())

    def test2(self) -> None:
        """
        Test that the lagged returns use the right past bars.
        """
        # Prepare inputs.
        clean = dutil.clean_bars(_get_random_bars(100))
        # Run test.
        features = dutil.compute_features(clean)
        # Check outputs.
        close = clean["close"]
        self.assertAlmostEqual(
            features["ret_12"].iloc[50], close.iloc[50] / close.iloc[38] - 1
        )
        self.assertAlmostEqual(features["ret_1"].iloc[50], clean["ret"].iloc[50])
        self.assertTrue(np.isnan(features["ret_48"].iloc[47]))

    def test3(self) -> None:
        """
        Test that no feature spans a long gap.
        """
        # Prepare inputs.
        clean = _get_gap_bars()
        # Run test.
        features = dutil.compute_features(clean)
        # Check outputs.
        # The second segment starts at row 32.
        self.assertTrue(features["ret_3"].iloc[32:35].isna().all())
        self.assertFalse(np.isnan(features["ret_3"].iloc[35]))
        # The 1h volatility needs 12 returns, so 13 bars of the segment.
        self.assertTrue(features["vol_1h"].iloc[32:44].isna().all())
        self.assertFalse(np.isnan(features["vol_1h"].iloc[44]))
        # Inside the gap there are no features.
        self.assertTrue(features.iloc[20:32].isna().all().all())


# #############################################################################
# Test_compute_labels
# #############################################################################


class Test_compute_labels(hunitest.TestCase):
    """
    Test `data_utils.compute_labels()`.
    """

    def test1(self) -> None:
        """
        Test that the labels line up with the right future bars.
        """
        # Prepare inputs.
        clean = dutil.clean_bars(_get_random_bars(100))
        # Run test.
        labels = dutil.compute_labels(clean, ["5m", "1h"])
        # Check outputs.
        close = clean["close"]
        expected = close.iloc[62] / close.iloc[50] - 1
        self.assertAlmostEqual(labels["fwd_ret_1h"].iloc[50], expected)
        self.assertEqual(labels["label_1h"].iloc[50], float(expected > 0))
        expected = close.iloc[51] / close.iloc[50] - 1
        self.assertAlmostEqual(labels["fwd_ret_5m"].iloc[50], expected)
        # The last 12 bars have no 1h label.
        self.assertTrue(labels["fwd_ret_1h"].iloc[-12:].isna().all())
        self.assertFalse(np.isnan(labels["fwd_ret_1h"].iloc[-13]))

    def test2(self) -> None:
        """
        Test that no label spans a long gap.
        """
        # Prepare inputs.
        clean = _get_gap_bars()
        # Run test.
        labels = dutil.compute_labels(clean, ["15m"])
        # Check outputs.
        # The first segment ends at row 19, so rows 17-19 have no 15m label.
        self.assertFalse(np.isnan(labels["fwd_ret_15m"].iloc[16]))
        self.assertTrue(labels["fwd_ret_15m"].iloc[17:20].isna().all())
        self.assertFalse(np.isnan(labels["fwd_ret_15m"].iloc[32]))

    def test3(self) -> None:
        """
        Test that a zero forward return has no label.
        """
        # Prepare inputs.
        times = ["2024-01-01 00:00", "2024-01-01 00:05", "2024-01-01 00:10"]
        clean = dutil.clean_bars(_get_bars(times, [1.0, 1.0, 2.0]))
        # Run test.
        labels = dutil.compute_labels(clean, ["5m"])
        # Check outputs.
        self.assertEqual(
            labels["label_5m"].astype(str).tolist(), ["nan", "1.0", "nan"]
        )


# #############################################################################
# Test_sample_non_overlapping
# #############################################################################


class Test_sample_non_overlapping(hunitest.TestCase):
    """
    Test `data_utils.sample_non_overlapping()`.
    """

    def test1(self) -> None:
        """
        Test that the decision times are one horizon apart.
        """
        # Prepare inputs.
        clean = dutil.clean_bars(_get_random_bars(100))
        labels = dutil.compute_labels(clean, ["1h"])
        # Run test.
        sample = dutil.sample_non_overlapping(labels, "1h", offset=2)
        # Check outputs.
        # Rows 2, 14, ..., 98.
        self.assertEqual(len(sample), 9)
        self.assertEqual(sample.index[0], clean.index[2])
        diffs = sample.index.to_series().diff().dropna().unique()
        self.assertEqual(diffs.tolist(), [pd.Timedelta("1h")])

    def test2(self) -> None:
        """
        Test that an offset outside the horizon is rejected.
        """
        # Prepare inputs.
        clean = dutil.clean_bars(_get_random_bars(20))
        # Run test and check outputs.
        with self.assertRaises(ValueError):
            dutil.sample_non_overlapping(clean, "15m", offset=3)
