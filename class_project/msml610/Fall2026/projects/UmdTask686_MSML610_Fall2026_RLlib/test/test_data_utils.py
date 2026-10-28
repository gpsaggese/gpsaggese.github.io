"""
Import as:

import class_project.msml610.Fall2026.projects.UmdTask686_MSML610_Fall2026_RLlib.test.test_data_utils as ttdaut
"""

import logging
from typing import List

import numpy as np
import pandas as pd

import class_project.msml610.Fall2026.projects.UmdTask686_MSML610_Fall2026_RLlib.data_utils as rldautil
import helpers.hunit_test as hunitest

_LOG = logging.getLogger(__name__)


def _get_bars(
    timestamps: List[str], opens: List[float], closes: List[float]
) -> pd.DataFrame:
    """
    Build minute bars with unit volume at the given New York times.

    :param timestamps: New York local times
    :param opens: open price of each bar
    :param closes: close price of each bar
    :return: bars indexed by UTC timestamp
    """
    # Localize the timestamps to New York, since they are local exchange times.
    index = pd.DatetimeIndex(timestamps).tz_localize("America/New_York")
    df = pd.DataFrame(
        {"open": opens, "close": closes, "volume": 1.0},
        index=index.tz_convert("UTC"),
    )
    return df


# #############################################################################
# Test_filter_regular_hours
# #############################################################################


class Test_filter_regular_hours(hunitest.TestCase):
    """
    Test `data_utils.filter_regular_hours()` function.
    """

    def test1(self) -> None:
        """
        Check that only bars in [9:30, 16:00) New York time are kept.
        """
        # Prepare inputs.
        # Bars just before and at each boundary of the regular session.
        timestamps = [
            "2020-03-02 09:29",
            "2020-03-02 09:30",
            "2020-03-02 15:59",
            "2020-03-02 16:00",
        ]
        prices = [1.0, 1.0, 1.0, 1.0]
        bars = _get_bars(timestamps, prices, prices)
        # Prepare outputs.
        expected = "['09:30', '15:59']"
        # Run test.
        df = rldautil.filter_regular_hours(bars)
        # Check outputs.
        ny_time = df.index.tz_convert("America/New_York")
        actual = str(list(ny_time.strftime("%H:%M")))
        self.assert_equal(actual, expected)


# #############################################################################
# Test_calibrate
# #############################################################################


class Test_calibrate(hunitest.TestCase):
    """
    Test `data_utils.calibrate()` function.
    """

    def test1(self) -> None:
        """
        Check the volatility, volume, and impact on two days of bars.
        """
        # Prepare inputs.
        # Day 1 goes from 100 to 101, day 2 from 100 to 99, with 2 bars each.
        timestamps = [
            "2020-03-02 09:30",
            "2020-03-02 15:59",
            "2020-03-03 09:30",
            "2020-03-03 15:59",
        ]
        opens = [100.0, 100.5, 100.0, 99.5]
        closes = [100.5, 101.0, 99.5, 99.0]
        bars = _get_bars(timestamps, opens, closes)
        n_steps = 20
        total_shares = 1
        # Prepare outputs.
        daily_vol = np.std([np.log(1.01), np.log(0.99)], ddof=1)
        expected = (
            f"daily_vol={daily_vol:.6f} adv=2.0 "
            f"gamma={0.314 * daily_vol:.6f} "
            f"eta={0.142 * daily_vol * n_steps:.6f}"
        )
        # Run test.
        params = rldautil.calibrate(bars, n_steps, total_shares)
        # Check outputs.
        actual = (
            f"daily_vol={params['daily_vol']:.6f} adv={params['adv']} "
            f"gamma={params['gamma']:.6f} eta={params['eta']:.6f}"
        )
        self.assert_equal(actual, expected)


# #############################################################################
# Test_fit_permanent_impact
# #############################################################################


class Test_fit_permanent_impact(hunitest.TestCase):
    """
    Test `data_utils.fit_permanent_impact()` function.
    """

    def test1(self) -> None:
        """
        Check that `gamma` is recovered when the midpoint follows the flow.
        """
        # Prepare inputs.
        # Two days of 4 bars, one bar per bucket. Each bucket's log return is
        # `0.5 * flow / adv`, so the fit is exact.
        timestamps = [
            "2020-03-02 09:30",
            "2020-03-02 09:31",
            "2020-03-02 09:32",
            "2020-03-02 09:33",
            "2020-03-03 09:30",
            "2020-03-03 09:31",
            "2020-03-03 09:32",
            "2020-03-03 09:33",
        ]
        prices = [100.0] * len(timestamps)
        bars = _get_bars(timestamps, prices, prices)
        adv = 1000.0
        flows = np.array([0.0, 100.0, -50.0, 20.0, 0.0, -80.0, 10.0, 60.0])
        bars["signed_volume"] = flows
        # Build the midpoints from the returns, restarting at 100 each day.
        log_mid = np.concatenate(
            [np.cumsum(0.5 * flows[:4] / adv), np.cumsum(0.5 * flows[4:] / adv)]
        )
        bars["close_mid"] = 100.0 * np.exp(log_mid)
        n_steps = 4
        # Prepare outputs.
        expected = "gamma=0.500000 r2=1.000000 n_obs=6"
        # Run test.
        fit = rldautil.fit_permanent_impact(bars, n_steps, adv)
        # Check outputs.
        actual = (
            f"gamma={fit['gamma']:.6f} r2={fit['r2']:.6f} "
            f"n_obs={fit['n_obs']:.0f}"
        )
        self.assert_equal(actual, expected)
