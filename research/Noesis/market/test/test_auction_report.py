"""
Test fixed expected tables for the five named M11 regimes.

Import as:

import research.Noesis.market.test.test_auction_report as rnmaterep
"""

import json
import os

import helpers.hunit_test as hunitest

import research.Noesis.market.auction_report as rnmaaurp
import research.Noesis.market.batch_call_auction as rnbacaau
import research.Noesis.market.round_runner as rnmaroru
import research.Noesis.market.scenarios as rnmascen


def _check_table(test_case: hunitest.TestCase, scenario: str, kind: str) -> None:
    """
    Compare an entire report with a checked-in golden, including column types.

    :param test_case: assertion owner
    :param scenario: named M11 family
    :param kind: table under test
    """
    # Prepare inputs: small books keep the golden tables manually reviewable.
    factory = getattr(rnmascen, f"make_{scenario}_scenario")
    bids, asks = factory(seed=0, n_bids=1, n_asks=1)
    book = rnbacaau.OrderBook()
    for bid in bids:
        book.submit_bid(bid)
    for ask in asks:
        book.submit_ask(ask)
    result = rnmaroru.run_round(
        book, rnmaroru.InMemoryContractStore(), lambda c: True
    )
    # Prepare outputs: expected values are static, not recomputed by the helper.
    path = os.path.join(
        os.path.dirname(__file__), "input", "auction_report.json"
    )
    with open(path) as stream:
        golden = json.load(stream)
    # Run test: create only the requested report.
    tables = {
        "orders": lambda: rnmaaurp.orders_to_df(bids, asks),
        "contracts": lambda: rnmaaurp.contracts_to_df(result.contracts),
    }
    actual = tables[kind]()
    # Check outputs: CSV checks all fields; types also cover nullable values.
    test_case.assert_equal(actual.to_csv(index=False), golden[scenario][kind])
    test_case.assert_equal(
        str(actual.dtypes.astype(str).tolist()), str(golden["types"][kind])
    )


# #############################################################################
# Test_orders_to_df
# #############################################################################


class Test_orders_to_df(hunitest.TestCase):
    """
    Test orders tables across the five named regimes.
    """

    def test1(self) -> None:
        """
        Check the balanced golden table.
        """
        # Prepare inputs.
        scenario, kind = "balanced", "orders"
        # Run test and check outputs.
        _check_table(self, scenario, kind)

    def test2(self) -> None:
        """
        Check the excess_demand golden table.
        """
        # Prepare inputs.
        scenario, kind = "excess_demand", "orders"
        # Run test and check outputs.
        _check_table(self, scenario, kind)

    def test3(self) -> None:
        """
        Check the excess_supply golden table.
        """
        # Prepare inputs.
        scenario, kind = "excess_supply", "orders"
        # Run test and check outputs.
        _check_table(self, scenario, kind)

    def test4(self) -> None:
        """
        Check the multi_tier golden table.
        """
        # Prepare inputs.
        scenario, kind = "multi_tier", "orders"
        # Run test and check outputs.
        _check_table(self, scenario, kind)

    def test5(self) -> None:
        """
        Check the no_cross golden table.
        """
        # Prepare inputs.
        scenario, kind = "no_cross", "orders"
        # Run test and check outputs.
        _check_table(self, scenario, kind)


# #############################################################################
# Test_contracts_to_df
# #############################################################################


class Test_contracts_to_df(hunitest.TestCase):
    """
    Test contracts tables across the five named regimes.
    """

    def test1(self) -> None:
        """
        Check the balanced golden table.
        """
        # Prepare inputs.
        scenario, kind = "balanced", "contracts"
        # Run test and check outputs.
        _check_table(self, scenario, kind)

    def test2(self) -> None:
        """
        Check the excess_demand golden table.
        """
        # Prepare inputs.
        scenario, kind = "excess_demand", "contracts"
        # Run test and check outputs.
        _check_table(self, scenario, kind)

    def test3(self) -> None:
        """
        Check the excess_supply golden table.
        """
        # Prepare inputs.
        scenario, kind = "excess_supply", "contracts"
        # Run test and check outputs.
        _check_table(self, scenario, kind)

    def test4(self) -> None:
        """
        Check the multi_tier golden table.
        """
        # Prepare inputs.
        scenario, kind = "multi_tier", "contracts"
        # Run test and check outputs.
        _check_table(self, scenario, kind)

    def test5(self) -> None:
        """
        Check the no_cross golden table.
        """
        # Prepare inputs.
        scenario, kind = "no_cross", "contracts"
        # Run test and check outputs.
        _check_table(self, scenario, kind)
