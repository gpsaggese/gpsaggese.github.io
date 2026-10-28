"""
Import as:

import class_project.msml610.Fall2026.projects.UmdTask686_MSML610_Fall2026_RLlib.test.test_taq_utils as tttaut
"""

import decimal
import logging

import pandas as pd

import class_project.msml610.Fall2026.projects.UmdTask686_MSML610_Fall2026_RLlib.taq_utils as rltautil
import helpers.hunit_test as hunitest

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_get_trading_days
# #############################################################################


class Test_get_trading_days(hunitest.TestCase):
    """
    Test `taq_utils.get_trading_days()` function.
    """

    def test1(self) -> None:
        """
        Check that only the daily `wct_` tables of the year are kept.
        """
        # Prepare inputs.
        # Yearly view, other table types, other years, and unsorted days.
        tables = [
            "wct_2015",
            "wct_20150105",
            "ctm_20150102",
            "wct_20150102",
            "wct_20160104",
            "wct_20151231",
        ]
        year = 2015
        # Prepare outputs.
        expected = "['20150102', '20150105', '20151231']"
        # Run test.
        days = rltautil.get_trading_days(tables, year)
        # Check outputs.
        actual = str(days)
        self.assert_equal(actual, expected)


# #############################################################################
# Test_finalize_bars
# #############################################################################


class Test_finalize_bars(hunitest.TestCase):
    """
    Test `taq_utils.finalize_bars()` function.
    """

    def test1(self) -> None:
        """
        Check the derived columns and the UTC index of two minutes.
        """
        # Prepare inputs.
        # Minute 1: 300 shares at 100.02 bought and 100 shares at 99.98 sold,
        # around a 100.00 midpoint with a 0.04 spread.
        # Minute 2: 200 shares at the midpoint, so no side.
        d = decimal.Decimal
        raw = pd.DataFrame(
            {
                "minute": pd.to_datetime(
                    ["2015-01-02 09:30", "2015-01-02 09:31"]
                ),
                "open": [d("100.02"), d("100.00")],
                "high": [d("100.02"), d("100.00")],
                "low": [d("99.98"), d("100.00")],
                "close": [d("99.98"), d("100.00")],
                # A 50-share block trade counts only in `volume`.
                "volume": [450, 200],
                "reg_volume": [400, 200],
                "n_trades": [2, 1],
                "notional": [d("40004.00"), d("20000.00")],
                "buy_volume": [300, None],
                "sell_volume": [100, None],
                "abs_dev_size": [d("0.08"), d("0")],
                "rel_spread_mean": [d("0.0004"), d("0.0004")],
                "close_mid": [d("100.00"), d("100.00")],
            }
        )
        # Prepare outputs.
        # 9:30 New York is 14:30 UTC in winter.
        # VWAP: 40004 / 400 = 100.01; effective spread: 2 * 0.08 / 400.
        expected = r"""
                                   vwap  signed_volume  quoted_spread_bps  effective_spread_bps
        timestamp
        2015-01-02 14:30:00+00:00  100.01          200.0                4.0                   4.0
        2015-01-02 14:31:00+00:00  100.00            0.0                4.0                   0.0
        """
        # Run test.
        df = rltautil.finalize_bars(raw)
        # Check outputs.
        columns = [
            "vwap",
            "signed_volume",
            "quoted_spread_bps",
            "effective_spread_bps",
        ]
        actual = df[columns].round(6).to_string()
        self.assert_equal(actual, expected, dedent=True, fuzzy_match=True)
