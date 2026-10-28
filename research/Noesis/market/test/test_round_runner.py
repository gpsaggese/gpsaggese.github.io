"""
Test library round execution, residual orders, storage, and HTTP independence.

Import as:

import research.Noesis.market.test.test_round_runner as rnmateroru
"""

import dataclasses
import logging
import shlex
import sys
from typing import Callable, List

import helpers.hprint as hprint
import helpers.hsystem as hsystem
import helpers.hunit_test as hunitest
import research.Noesis.market.batch_call_auction as rnbacaau
import research.Noesis.market.contract_dispatch as rnocodis
import research.Noesis.market.round_runner as rnmaroru
import research.Noesis.market.scenarios as rnmascen

_LOG = logging.getLogger(__name__)


def _make_book(
    bids: List[rnbacaau.Bid], asks: List[rnbacaau.Ask]
) -> rnbacaau.OrderBook:
    """
    Submit a fixture without duplicating setup in each round test.

    :param bids: buyer orders to submit in order
    :param asks: seller orders to submit in order
    :return: a fresh populated order book
    """
    book = rnbacaau.OrderBook()
    for bid in bids:
        book.submit_bid(bid)
    for ask in asks:
        book.submit_ask(ask)
    return book


# #############################################################################
# Test_run_round
# #############################################################################


class Test_run_round(hunitest.TestCase):
    """
    Test unchanged clearing, residual accounting, and persistence from Python.
    """

    def helper(
        self,
        factory: Callable[..., rnmascen.Scenario],
        regime: str,
    ) -> None:
        """
        Exercise one M11 scenario through the complete market workflow.

        :param factory: scenario generator supplying the fixture
        :param regime: expected balance or non-crossing behavior
        """
        # Prepare inputs: use varied orders, not a second handwritten fixture bank.
        seed = 7
        bids, asks = factory(seed=seed)
        book = _make_book(bids, asks)
        store = rnmaroru.InMemoryContractStore()
        fulfill = lambda contract: True
        # Run test.
        result = rnmaroru.run_round(book, store, fulfill)
        # Check outputs: no orders remain queued after any regime.
        self.assert_equal(
            str((book.get_pending_bids(), book.get_pending_asks())),
            str(([], [])),
        )
        # Check conservation separately in each tier, including partial orders.
        for tier, summary in result.tier_results.items():
            demand = sum(b.n_tasks for b in bids if b.c_level_min == tier)
            supply = sum(a.n_tasks for a in asks if a.c_level == tier)
            residual_bids = sum(
                b.n_tasks for b in result.unmatched_bids if b.c_level_min == tier
            )
            residual_asks = sum(
                a.n_tasks for a in result.unmatched_asks if a.c_level == tier
            )
            actual = (
                summary.matched_volume + residual_bids,
                summary.matched_volume + residual_asks,
            )
            expected = (demand, supply)
            self.assert_equal(str(actual), str(expected))
            self.assertEqual(summary.round_id, result.round_id)
            self.assertEqual(store.get_latest_round(tier), summary)
            # The family name dictates volume and the side left unmatched.
            if regime == "no_cross":
                self.assertEqual(summary.matched_volume, 0)
                self.assertIsNone(summary.clearing_price)
            else:
                self.assertEqual(summary.matched_volume, min(demand, supply))
                if regime == "demand":
                    self.assertGreater(residual_bids, 0)
                    self.assertEqual(residual_asks, 0)
                elif regime == "supply":
                    self.assertGreater(residual_asks, 0)
                    self.assertEqual(residual_bids, 0)
                else:
                    self.assert_equal(
                        str((residual_bids, residual_asks)), str((0, 0))
                    )
        # Stored contracts must carry the injected fulfillment outcome.
        self.assertEqual(
            sum(c.n_tasks for c in result.contracts),
            sum(t.matched_volume for t in result.tier_results.values()),
        )
        for contract_id, contract in enumerate(result.contracts):
            self.assertEqual(store.get_contract(contract_id), contract)
            self.assertIs(contract.fulfilled, True)
        if regime == "multi_tier":
            prices = [t.clearing_price for t in result.tier_results.values()]
            self.assertGreaterEqual(len(prices), 2)
            self.assertEqual(len(prices), len(set(prices)))

    def test1(self) -> None:
        """
        Match the original API fixture's exact price, volume, and contract.
        """
        # Prepare inputs.
        bids = [rnbacaau.Bid("buyer_1", 100, "frontier", 2.0, 0.999, 20.0)]
        asks = [rnbacaau.Ask("seller_1", 100, "frontier", 2.5, 0.98, 16.0)]
        book = _make_book(bids, asks)
        store = rnmaroru.InMemoryContractStore()
        fulfill = lambda contract: True
        # Prepare outputs: fixed historical API values, not a second calculation.
        expected = rnmaroru.RoundResult(
            0,
            {"frontier": rnmaroru.RoundSummary("frontier", 0, 18.0, 100)},
            [
                rnocodis.Contract(
                    "buyer_1",
                    "seller_1",
                    100,
                    "frontier",
                    2.0,
                    0.999,
                    18.0,
                    True,
                )
            ],
            [],
            [],
        )
        # Run test.
        actual = rnmaroru.run_round(book, store, fulfill)
        # Check outputs.
        self.assert_equal(
            str(dataclasses.asdict(actual)), str(dataclasses.asdict(expected))
        )
        self.assertEqual(store.get_contract(0), expected.contracts[0])

    def test2(self) -> None:
        """
        Fully match a seeded balanced book.
        """
        # Prepare inputs.
        factory = rnmascen.make_balanced_scenario
        regime = "balanced"
        # Run test and check outputs.
        self.helper(factory, regime)

    def test3(self) -> None:
        """
        Report unmatched buyer quantity under excess demand.
        """
        # Prepare inputs.
        factory = rnmascen.make_excess_demand_scenario
        regime = "demand"
        # Run test and check outputs.
        self.helper(factory, regime)

    def test4(self) -> None:
        """
        Report unmatched seller quantity under excess supply.
        """
        # Prepare inputs.
        factory = rnmascen.make_excess_supply_scenario
        regime = "supply"
        # Run test and check outputs.
        self.helper(factory, regime)

    def test5(self) -> None:
        """
        Share one round id across tiers with distinct clearing prices.
        """
        # Prepare inputs.
        factory = rnmascen.make_multi_tier_scenario
        regime = "multi_tier"
        # Run test and check outputs.
        self.helper(factory, regime)

    def test6(self) -> None:
        """
        Publish zero-volume results and retain all non-crossing quantities.
        """
        # Prepare inputs.
        factory = rnmascen.make_no_cross_scenario
        regime = "no_cross"
        # Run test and check outputs.
        self.helper(factory, regime)

    def test7(self) -> None:
        """
        Give empty rounds ids without dispatching contracts or erasing history.
        """
        # Prepare inputs.
        seed = 0
        bids, asks = rnmascen.make_balanced_scenario(seed=seed)
        book = _make_book(bids, asks)
        store = rnmaroru.InMemoryContractStore()
        calls = []

        def fulfill(contract: rnocodis.Contract) -> bool:
            calls.append(contract)
            return False

        # Run test: one populated round, followed by two empty rounds.
        first = rnmaroru.run_round(book, store, fulfill)
        first_call_count = len(calls)
        second = rnmaroru.run_round(book, store, fulfill)
        third = rnmaroru.run_round(book, store, fulfill)
        # Check outputs: the empty rounds still consume exactly one id each.
        actual = (
            first.round_id,
            dataclasses.asdict(second),
            third.round_id,
            len(calls),
            store.get_latest_round("standard"),
        )
        expected = (
            0,
            dataclasses.asdict(rnmaroru.RoundResult(1, {}, [], [], [])),
            2,
            first_call_count,
            first.tier_results["standard"],
        )
        self.assert_equal(str(actual), str(expected))
        self.assertTrue(all(c.fulfilled is False for c in first.contracts))
        self.assertEqual(store.get_contract(0), first.contracts[0])

    def test8(self) -> None:
        """
        Preserve partial-order identity, priority, and original input quantities.
        """
        # Prepare inputs: repeated ids and out-of-priority submission order.
        bids = [
            rnbacaau.Bid("buyer", 8, "x", 2, 0.9, 6),
            rnbacaau.Bid("buyer", 5, "x", 2, 0.9, 10),
            rnbacaau.Bid("buyer", 4, "y", 2, 0.9, 1),
        ]
        asks = [
            rnbacaau.Ask("seller", 3, "x", 1, 0.95, 5),
            rnbacaau.Ask("seller", 4, "x", 1, 0.95, 2),
            rnbacaau.Ask("seller", 9, "y", 1, 0.95, 8),
        ]
        book = _make_book(bids, asks)
        store = rnmaroru.InMemoryContractStore()
        fulfill = lambda contract: True
        original = str((bids, asks))
        # Prepare outputs: five expensive tasks fill first, then two cheaper ones.
        expected_bids = [dataclasses.replace(bids[0], n_tasks=6), bids[2]]
        expected_asks = [asks[2]]
        # Run test.
        result = rnmaroru.run_round(book, store, fulfill)
        # Check outputs: residuals are copies, including entirely unmatched orders.
        self.assert_equal(str(result.unmatched_bids), str(expected_bids))
        self.assert_equal(str(result.unmatched_asks), str(expected_asks))
        self.assert_equal(str((bids, asks)), original)
        self.assertIsNot(result.unmatched_bids[0], bids[0])
        self.assertIsNot(result.unmatched_asks[0], asks[2])

    def test9(self) -> None:
        """
        Publish one-sided tiers without inventing matches or dropping residuals.
        """
        # Prepare inputs.
        bids = [rnbacaau.Bid("buyer", 5, "buy_only", 2, 0.9, 10)]
        asks = [rnbacaau.Ask("seller", 7, "sell_only", 1, 0.95, 2)]
        book = _make_book(bids, asks)
        store = rnmaroru.InMemoryContractStore()
        fulfill = lambda contract: True
        # Prepare outputs.
        expected = rnmaroru.RoundResult(
            0,
            {
                "buy_only": rnmaroru.RoundSummary("buy_only", 0, None, 0),
                "sell_only": rnmaroru.RoundSummary("sell_only", 0, None, 0),
            },
            [],
            bids,
            asks,
        )
        # Run test.
        actual = rnmaroru.run_round(book, store, fulfill)
        # Check outputs.
        self.assert_equal(
            str(dataclasses.asdict(actual)), str(dataclasses.asdict(expected))
        )

    def test10(self) -> None:
        """
        Import and execute the market in a process that rejects HTTP imports.
        """
        # Prepare inputs: a fresh process prevents prior API imports hiding a leak.
        script = hprint.dedent("""
            import importlib.abc
            import sys
            class RejectHttp(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in ('fastapi', 'pydantic'):
                        raise RuntimeError('Unexpected HTTP import: ' + fullname)
                    if fullname.startswith('research.Noesis.noesis_platform'):
                        raise RuntimeError('Unexpected platform import: ' + fullname)
            sys.meta_path.insert(0, RejectHttp())
            import research.Noesis.market.round_runner as runner
            import research.Noesis.market.batch_call_auction as auction
            result = runner.run_round(auction.OrderBook(),
                                     runner.InMemoryContractStore(), lambda c: True)
            print(result.round_id)
        """)
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"
        expected = (0, "0")
        # Run test.
        return_code, output = hsystem.system_to_string(
            command, abort_on_error=False
        )
        # Check outputs.
        actual = (return_code, output.strip())
        self.assert_equal(str(actual), str(expected))

    def test11(self) -> None:
        """
        Allocate partial seller fills by price priority for a repeated seller id.
        """
        # Prepare inputs: submit the more expensive ask before the cheaper ask.
        bids = [rnbacaau.Bid("buyer", 7, "x", 2, 0.9, 10)]
        asks = [
            rnbacaau.Ask("seller", 8, "x", 1, 0.95, 5),
            rnbacaau.Ask("seller", 5, "x", 1, 0.95, 2),
        ]
        book = _make_book(bids, asks)
        store = rnmaroru.InMemoryContractStore()
        fulfill = lambda contract: True
        original = str(asks)
        # Prepare outputs: all five cheap tasks and two expensive tasks fill.
        expected = [dataclasses.replace(asks[0], n_tasks=6)]
        # Run test.
        result = rnmaroru.run_round(book, store, fulfill)
        # Check outputs.
        self.assert_equal(str(result.unmatched_asks), str(expected))
        self.assert_equal(str(asks), original)
        self.assert_equal(str(result.unmatched_bids), str([]))


# #############################################################################
# TestInMemoryContractStore
# #############################################################################


class TestInMemoryContractStore(hunitest.TestCase):
    """
    Test independent counters, missing lookups, and replacement of tier summaries.
    """

    def test1(self) -> None:
        """
        Keep independent store counters and retain the latest summary per tier.
        """
        # Prepare inputs.
        first = rnmaroru.InMemoryContractStore()
        second = rnmaroru.InMemoryContractStore()
        summaries = [
            rnmaroru.RoundSummary("x", 0, 2.0, 3),
            rnmaroru.RoundSummary("x", 1, None, 0),
        ]
        # Run test.
        ids = (
            first.next_round_id(),
            first.next_round_id(),
            second.next_round_id(),
        )
        for summary in summaries:
            first.save_round(summary)
        # Check outputs.
        self.assert_equal(str(ids), str((0, 1, 0)))
        self.assertEqual(first.get_latest_round("x"), summaries[-1])
        with self.assertRaises(AssertionError):
            second.get_latest_round("x")
        with self.assertRaises(AssertionError):
            first.get_contract(0)
