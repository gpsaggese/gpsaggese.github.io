"""
Import as:

import research.Noesis.market.test.test_credit_ledger as rnmtcrle
"""

import logging

import helpers.hunit_test as hunitest
import research.Noesis.market.batch_call_auction as rnbacaau
import research.Noesis.market.credit_ledger as rnmacrle

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_CreditLedger
# #############################################################################


class Test_CreditLedger(hunitest.TestCase):
    """
    Test `credit_ledger.CreditLedger`.
    """

    def test1(self) -> None:
        """
        Test that a new account starts with the free grant.
        """
        # Prepare inputs.
        ledger = rnmacrle.CreditLedger(free_credit=1.0)
        account_id = "acct_1"
        # Prepare outputs.
        expected = 1.0
        # Run test.
        actual = ledger.get_balance(account_id)
        # Check outputs.
        self.assert_equal(str(actual), str(expected))

    def test2(self) -> None:
        """
        Test that a bid within the balance is affordable and one over it is
        not.
        """
        # Prepare inputs.
        ledger = rnmacrle.CreditLedger(free_credit=1.0)
        account_id = "acct_1"
        affordable = rnbacaau.Bid("buyer_1", 50, "standard", 2.0, 0.9, 0.001)
        too_expensive = rnbacaau.Bid("buyer_1", 50, "standard", 2.0, 0.9, 0.03)
        # Prepare outputs.
        expected = "affordable=True too_expensive=False"
        # Run test.
        actual = (
            f"affordable={ledger.can_afford(account_id, rnmacrle.bid_cost(affordable))}"
            f" too_expensive="
            f"{ledger.can_afford(account_id, rnmacrle.bid_cost(too_expensive))}"
        )
        # Check outputs.
        self.assert_equal(actual, expected)

    def test3(self) -> None:
        """
        Test that a debit lowers the balance once, and a repeat for the same
        contract changes nothing.
        """
        # Prepare inputs.
        ledger = rnmacrle.CreditLedger(free_credit=1.0)
        account_id = "acct_1"
        contract_id = 7
        amount = rnmacrle.fill_cost(50, 0.0006)
        # Prepare outputs.
        expected = "first=True second=False balance=0.97"
        # Run test.
        first = ledger.debit(account_id, contract_id, amount)
        second = ledger.debit(account_id, contract_id, amount)
        balance = round(ledger.get_balance(account_id), 9)
        # Check outputs.
        actual = f"first={first} second={second} balance={balance}"
        self.assert_equal(actual, expected)

    def test4(self) -> None:
        """
        Test that a negative amount is rejected.
        """
        # Prepare inputs.
        ledger = rnmacrle.CreditLedger(free_credit=1.0)
        account_id = "acct_1"
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            ledger.debit(account_id, 1, -0.5)
