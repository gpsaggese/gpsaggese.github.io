"""
Import as:

import research.Noesis.scripts.test.test_demo_credit_ledger as rnsttdcl
"""

import logging

import helpers.hunit_test as hunitest
import research.Noesis.scripts.demo_credit_ledger as rnscdecl

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_build_report
# #############################################################################


class Test_build_report(hunitest.TestCase):
    """
    Test `demo_credit_ledger.build_report()`.
    """

    def test1(self) -> None:
        """
        Test the report with the script's defaults.
        """
        # Prepare inputs.
        free_credit = 100.0
        bid_n_tasks = 4
        bid_p_max = 20.0
        ask_n_tasks = 3
        ask_p_min = 16.0
        # Prepare outputs.
        expected = """
        1) Account 'acct_demo' starts with balance=100.0
        2) Bid 4 tasks at p_max=20.0 (cost 80.0) -> 201 accepted
           balance after the bid: 100.0 (nothing is held)
        3) Bid 6 tasks at p_max=20.0 (cost 120.0) -> 400 rejected: Insufficient credit for 'acct_demo': bid costs up to 120.0, balance is 100.0
        4) Ask 3 tasks at p_min=16.0 -> clear_round():
           tier=standard clearing_price=18.0 matched_volume=3
        5) Balance after clearing: 46.0
           balance after a second clear: 46.0
        6) Bid 2 tasks at p_max=20.0 (cost 40.0) -> 201 accepted
        6) Bid 3 tasks at p_max=20.0 (cost 60.0) -> 400 rejected: Insufficient credit for 'acct_demo': bid costs up to 60.0, balance is 46.0
        """
        # Run test.
        actual = rnscdecl.build_report(
            free_credit=free_credit,
            bid_n_tasks=bid_n_tasks,
            bid_p_max=bid_p_max,
            ask_n_tasks=ask_n_tasks,
            ask_p_min=ask_p_min,
        )
        # Check outputs.
        self.assert_equal(actual, expected, fuzzy_match=True)
