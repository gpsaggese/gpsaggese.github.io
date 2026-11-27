"""
Import as:

import test.test_tradability_utils as tttrut
"""

import logging

import helpers.hunit_test as hunitest
import pandas as pd
import tradability_utils as tu

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_drop_overnight
# #############################################################################


class Test_drop_overnight(hunitest.TestCase):
    """
    Test `tradability_utils.drop_overnight()`.
    """

    def test1(self) -> None:
        """
        Test that the first hourly bar of each New York day is dropped.
        """
        # Prepare inputs.
        # 14:30 UTC is 9:30 in New York in January.
        times = [
            "2024-01-02 14:30",
            "2024-01-02 15:30",
            "2024-01-02 16:30",
            "2024-01-03 14:30",
            "2024-01-03 15:30",
        ]
        index = pd.DatetimeIndex(times, tz="UTC")
        rets = pd.Series([0.05, 0.01, 0.02, -0.04, 0.03], index=index)
        # Run test.
        out = tu.drop_overnight(rets)
        # Check outputs.
        self.assertEqual(out.tolist(), [0.01, 0.02, 0.03])
        self.assertEqual(out.index.tolist(), index[[1, 2, 4]].tolist())
