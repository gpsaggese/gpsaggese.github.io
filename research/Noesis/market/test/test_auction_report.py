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
        "summary": lambda: rnmaaurp.tier_summary_df(result),
        "unmatched": lambda: rnmaaurp.unmatched_to_df(result),
    }
    actual = tables[kind]()
    if kind == "summary":
        for row in actual.itertuples(index=False):
            expected = (
                sum(b.n_tasks for b in bids if b.c_level_min == row.tier),
                sum(a.n_tasks for a in asks if a.c_level == row.tier),
            )
            test_case.assertEqual(
                (row.total_bid_volume, row.total_ask_volume), expected
            )

    # Check outputs: CSV checks all fields; types also cover nullable values.
    test_case.assert_equal(actual.to_csv(index=False), golden[scenario][kind])
    test_case.assert_equal(
        str(actual.dtypes.astype(str).tolist()), str(golden["types"][kind])
    )


def _check_empty(test_case: hunitest.TestCase, kind: str) -> None:
    """
    Check empty results against the fixed schema rather than inferred columns.

    :param test_case: assertion owner
    :param kind: report to check
    """
    # Prepare inputs and expected outputs.
    result = rnmaroru.RoundResult(0, {}, [], [], [])
    path = os.path.join(
        os.path.dirname(__file__), "input", "auction_report.json"
    )
    with open(path) as stream:
        golden = json.load(stream)
    tables = {
        "orders": lambda: rnmaaurp.orders_to_df([], []),
        "contracts": lambda: rnmaaurp.contracts_to_df([]),
        "summary": lambda: rnmaaurp.tier_summary_df(result),
        "unmatched": lambda: rnmaaurp.unmatched_to_df(result),
    }
    # Run test and check outputs.
    actual = tables[kind]()
    expected_header = golden["balanced"][kind].splitlines()[0]
    test_case.assertEqual(actual.to_csv(index=False).strip(), expected_header)
    test_case.assertEqual(
        actual.dtypes.astype(str).tolist(), golden["types"][kind]
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

    def test6(self) -> None:
        """
        Preserve columns and types for an empty table.
        """
        # Prepare inputs.
        kind = "orders"
        # Run test and check outputs.
        _check_empty(self, kind)


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

    def test6(self) -> None:
        """
        Preserve columns and types for an empty table.
        """
        # Prepare inputs.
        kind = "contracts"
        # Run test and check outputs.
        _check_empty(self, kind)


# #############################################################################
# Test_tier_summary_df
# #############################################################################


class Test_tier_summary_df(hunitest.TestCase):
    """
    Test summary tables across the named regimes.
    """

    def test1(self) -> None:
        """
        Check the balanced golden table.
        """
        # Prepare inputs.
        scenario, kind = "balanced", "summary"
        # Run test and check outputs.
        _check_table(self, scenario, kind)

    def test2(self) -> None:
        """
        Check the excess_demand golden table.
        """
        # Prepare inputs.
        scenario, kind = "excess_demand", "summary"
        # Run test and check outputs.
        _check_table(self, scenario, kind)

    def test3(self) -> None:
        """
        Check the excess_supply golden table.
        """
        # Prepare inputs.
        scenario, kind = "excess_supply", "summary"
        # Run test and check outputs.
        _check_table(self, scenario, kind)

    def test4(self) -> None:
        """
        Check the multi_tier golden table.
        """
        # Prepare inputs.
        scenario, kind = "multi_tier", "summary"
        # Run test and check outputs.
        _check_table(self, scenario, kind)

    def test5(self) -> None:
        """
        Check the no_cross golden table.
        """
        # Prepare inputs.
        scenario, kind = "no_cross", "summary"
        # Run test and check outputs.
        _check_table(self, scenario, kind)

    def test6(self) -> None:
        """
        Preserve columns and types for an empty table.
        """
        # Prepare inputs.
        kind = "summary"
        # Run test and check outputs.
        _check_empty(self, kind)


# #############################################################################
# Test_unmatched_to_df
# #############################################################################


class Test_unmatched_to_df(hunitest.TestCase):
    """
    Test unmatched tables across the named regimes.
    """

    def test1(self) -> None:
        """
        Check the balanced golden table.
        """
        # Prepare inputs.
        scenario, kind = "balanced", "unmatched"
        # Run test and check outputs.
        _check_table(self, scenario, kind)

    def test2(self) -> None:
        """
        Check the excess_demand golden table.
        """
        # Prepare inputs.
        scenario, kind = "excess_demand", "unmatched"
        # Run test and check outputs.
        _check_table(self, scenario, kind)

    def test3(self) -> None:
        """
        Check the excess_supply golden table.
        """
        # Prepare inputs.
        scenario, kind = "excess_supply", "unmatched"
        # Run test and check outputs.
        _check_table(self, scenario, kind)

    def test4(self) -> None:
        """
        Check the multi_tier golden table.
        """
        # Prepare inputs.
        scenario, kind = "multi_tier", "unmatched"
        # Run test and check outputs.
        _check_table(self, scenario, kind)

    def test5(self) -> None:
        """
        Check the no_cross golden table.
        """
        # Prepare inputs.
        scenario, kind = "no_cross", "unmatched"
        # Run test and check outputs.
        _check_table(self, scenario, kind)

    def test6(self) -> None:
        """
        Preserve columns and types for an empty table.
        """
        # Prepare inputs.
        kind = "unmatched"
        # Run test and check outputs.
        _check_empty(self, kind)
