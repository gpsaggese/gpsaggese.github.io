"""
Import as:

import test.test_backtest_utils as ttbaut
"""

import logging

import helpers.hunit_test as hunitest
import numpy as np
import pandas as pd

import backtest_utils as butil

_LOG = logging.getLogger(__name__)


def _get_trades() -> pd.DataFrame:
    """
    Build daily trades in 2022 and 2023, in two groups.

    Group "a" makes 1% per trade in 2022 and group "b" loses 1%. In 2023 the
    two swap.

    :return: `pos`, `fwd_ret`, and `group` of each trade
    """
    index = pd.date_range("2022-01-01", "2023-12-31", freq="1D", tz="UTC")
    group = np.where(np.arange(len(index)) % 2 == 0, "a", "b")
    is_2022 = index.year == 2022
    is_winner = (group == "a") == is_2022
    fwd_ret = np.where(is_winner, 0.01, -0.01)
    df = pd.DataFrame(
        {"pos": 1.0, "fwd_ret": fwd_ret, "group": group}, index=index
    )
    return df


# #############################################################################
# Test_select_groups_by_past_pnl
# #############################################################################


class Test_select_groups_by_past_pnl(hunitest.TestCase):
    """
    Test `backtest_utils.select_groups_by_past_pnl()`.
    """

    def test1(self) -> None:
        """
        Test that each year uses only the earlier years.
        """
        # Prepare inputs.
        df = _get_trades()
        # Run test.
        selected = butil.select_groups_by_past_pnl(
            df["pos"], df["fwd_ret"], df["group"], [2023, 2024], cost=0.001
        )
        # Check outputs.
        # 2023 is chosen from 2022 only, where "a" made money. 2024 is chosen
        # from 2022 and 2023, where both groups break even before costs.
        self.assertEqual(selected.loc[2023].tolist(), [True, False])
        self.assertEqual(selected.loc[2024].tolist(), [False, False])

    def test2(self) -> None:
        """
        Test that a group with too few earlier trades is not traded.
        """
        # Prepare inputs.
        df = _get_trades()
        # Run test.
        selected = butil.select_groups_by_past_pnl(
            df["pos"],
            df["fwd_ret"],
            df["group"],
            [2023],
            cost=0.0,
            min_trades=200,
        )
        # Check outputs.
        # Each group has 182 or 183 trades in 2022.
        self.assertEqual(selected.loc[2023].tolist(), [False, False])

    def test3(self) -> None:
        """
        Test that the first year, with no earlier trades, is not traded.
        """
        # Prepare inputs.
        df = _get_trades()
        # Run test.
        selected = butil.select_groups_by_past_pnl(
            df["pos"], df["fwd_ret"], df["group"], [2022], cost=0.0
        )
        # Check outputs.
        self.assertFalse(selected.loc[2022].any())


# #############################################################################
# Test_keep_selected
# #############################################################################


class Test_keep_selected(hunitest.TestCase):
    """
    Test `backtest_utils.keep_selected()`.
    """

    def test1(self) -> None:
        """
        Test that only the selected groups of each year keep a position.
        """
        # Prepare inputs.
        df = _get_trades()
        selected = pd.DataFrame(
            {"a": [True], "b": [False]}, index=pd.Index([2023], name="year")
        )
        # Run test.
        kept = butil.keep_selected(df["pos"], df["group"], selected)
        # Check outputs.
        is_2023 = df.index.year == 2023
        is_a = df["group"] == "a"
        self.assertTrue(kept[is_2023 & is_a].eq(1.0).all())
        self.assertTrue(kept[is_2023 & ~is_a].isna().all())
        # 2022 is not in `selected`, so it is not traded.
        self.assertTrue(kept[~is_2023].isna().all())


# #############################################################################
# Test_compute_backtest_stats
# #############################################################################


class Test_compute_backtest_stats(hunitest.TestCase):
    """
    Test `backtest_utils.compute_backtest_stats()`.
    """

    def test1(self) -> None:
        """
        Test the metrics on a small example.
        """
        # Prepare inputs.
        index = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
        pos = pd.Series([1.0, -1.0, np.nan, 1.0, 1.0], index=index)
        fwd_ret = pd.Series([0.02, 0.01, 0.05, -0.03, 0.01], index=index)
        # Run test.
        stats = butil.compute_backtest_stats(
            pos, fwd_ret, cost=0.001, horizon="1d"
        )
        # Check outputs.
        # PnL: 0.019, -0.011, 0 (no trade), -0.031, 0.009.
        pnl = np.array([0.019, -0.011, 0.0, -0.031, 0.009])
        self.assertEqual(stats["n_trades"], 4)
        self.assertAlmostEqual(stats["share_traded"], 0.8)
        self.assertAlmostEqual(stats["net_bps"], -0.014 / 4 * 1e4)
        self.assertAlmostEqual(stats["total_return"], -0.014)
        # Cumulative PnL: 0.019, 0.008, 0.008, -0.023, -0.014.
        self.assertAlmostEqual(stats["max_drawdown"], -0.042)
        expected_sharpe = pnl.mean() / pnl.std(ddof=1) * np.sqrt(365)
        self.assertAlmostEqual(stats["sharpe"], expected_sharpe)

    def test2(self) -> None:
        """
        Test that a strategy that never trades has no PnL per trade.
        """
        # Prepare inputs.
        index = pd.date_range("2024-01-01", periods=3, freq="4h", tz="UTC")
        pos = pd.Series(np.nan, index=index)
        fwd_ret = pd.Series([0.01, -0.02, 0.03], index=index)
        # Run test.
        stats = butil.compute_backtest_stats(
            pos, fwd_ret, cost=0.001, horizon="4h"
        )
        # Check outputs.
        self.assertEqual(stats["n_trades"], 0)
        self.assertTrue(np.isnan(stats["net_bps"]))
        self.assertEqual(stats["total_return"], 0.0)
        self.assertTrue(np.isnan(stats["sharpe"]))
