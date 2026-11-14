"""
Import as:

import test.test_regime_utils as ttreut
"""

import logging

import helpers.hunit_test as hunitest
import numpy as np
import pandas as pd

import regime_utils as rutil

_LOG = logging.getLogger(__name__)


def _get_features(vol: list[float], mom: list[float]) -> pd.DataFrame:
    """
    Build daily features with the given 7-day volatility and momentum.

    :param vol: `vol_7d` of each day
    :param mom: `mom_7d` of each day
    :return: features indexed by day (UTC)
    """
    index = pd.date_range("2024-01-01", periods=len(vol), freq="1D", tz="UTC")
    features = pd.DataFrame({"vol_7d": vol, "mom_7d": mom}, index=index)
    return features


# #############################################################################
# Test_compute_regimes
# #############################################################################


class Test_compute_regimes(hunitest.TestCase):
    """
    Test `regime_utils.compute_regimes()`.
    """

    def test1(self) -> None:
        """
        Test the volatility regime against the terciles of the past days.
        """
        # Prepare inputs.
        # The volatility goes 1, 2, ..., 9, then a low and a high day.
        vol = [float(v) for v in range(1, 10)] + [1.5, 8.5]
        features = _get_features(vol, [0.0] * len(vol))
        # Run test.
        regimes = rutil.compute_regimes(
            features, interval="1d", n_days=9, min_days=9
        )
        # Check outputs.
        # The first 9 days have no thresholds yet.
        self.assertTrue(regimes["vol_regime"].iloc[:9].isna().all())
        self.assertEqual(
            regimes["vol_regime"].iloc[9:].tolist(), ["calm", "volatile"]
        )

    def test2(self) -> None:
        """
        Test the trend regime from the 7-day t-statistic.
        """
        # Prepare inputs.
        # With daily bars, t = mom_7d / (vol_7d * sqrt(7)).
        scale = np.sqrt(7)
        mom = [-2 * scale, 0.5 * scale, 2 * scale]
        features = _get_features([1.0, 1.0, 1.0], mom)
        # Run test.
        regimes = rutil.compute_regimes(features, interval="1d")
        # Check outputs.
        self.assertEqual(
            regimes["trend_regime"].tolist(), ["falling", "flat", "rising"]
        )
        np.testing.assert_allclose(regimes["trend_t"], [-2.0, 0.5, 2.0])

    def test3(self) -> None:
        """
        Test that changing the data after day t does not change the regimes
        at or before t (no look-ahead).
        """
        # Prepare inputs.
        rng = np.random.default_rng(0)
        n_days = 60
        vol = rng.uniform(0.5, 2.0, n_days).tolist()
        mom = rng.normal(0, 1, n_days).tolist()
        features = _get_features(vol, mom)
        t = features.index[40]
        changed = features.copy()
        changed.loc[changed.index > t] *= 3.0
        # Run test.
        regimes = rutil.compute_regimes(
            features, interval="1d", n_days=20, min_days=20
        )
        regimes_changed = rutil.compute_regimes(
            changed, interval="1d", n_days=20, min_days=20
        )
        # Check outputs.
        pd.testing.assert_frame_equal(regimes.loc[:t], regimes_changed.loc[:t])
