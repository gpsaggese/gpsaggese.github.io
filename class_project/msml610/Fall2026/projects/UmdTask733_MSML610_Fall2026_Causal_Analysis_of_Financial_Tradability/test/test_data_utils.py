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
