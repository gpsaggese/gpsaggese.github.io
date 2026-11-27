"""
Import as:

import test.test_fetch_evalplus_data as ttfeevda
"""

import logging

import helpers.hunit_test as hunitest

import fetch_evalplus_data as feevda

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_get_leaderboard
# #############################################################################


class Test_get_leaderboard(hunitest.TestCase):
    """
    Test `fetch_evalplus_data.get_leaderboard()`.
    """

    def test1(self) -> None:
        """
        Test that the models are sorted and that a missing score stays empty.
        """
        # Prepare inputs.
        results = {
            "Model-B": {
                "pass@1": {"humaneval": 80.0, "humaneval+": 75.0, "mbpp": None},
                "size": 7,
                "prompted": True,
                "open-data": "Full",
            },
            "Model-A": {
                "pass@1": {
                    "humaneval": 60.0,
                    "humaneval+": 55.5,
                    "mbpp": 70.0,
                    "mbpp+": 61.2,
                },
                "size": None,
                "prompted": False,
                "open-data": "NONE",
            },
        }
        # Prepare outputs.
        expected = """
             model  size  prompted open_data  humaneval  humaneval_plus  mbpp  mbpp_plus
        0  Model-A   NaN     False      NONE       60.0            55.5  70.0       61.2
        1  Model-B   7.0      True      Full       80.0            75.0   NaN        NaN
        """
        # Run test.
        leaderboard = feevda.get_leaderboard(results)
        # Check outputs.
        actual = leaderboard.to_string()
        self.assert_equal(actual, expected, dedent=True)
