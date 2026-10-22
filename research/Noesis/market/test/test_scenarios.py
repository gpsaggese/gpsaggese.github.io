"""
Test reproducible varied scenario families; M10 runner integration remains pending.

Import as:

import research.Noesis.market.test.test_scenarios as rnmatestsce
"""

import logging
import random
from typing import Callable, Dict, List

import helpers.hunit_test as hunitest
import research.Noesis.market.batch_call_auction as rnbacaau
import research.Noesis.market.contract_dispatch as rnocodis
import research.Noesis.market.scenarios as rnmascen

_LOG = logging.getLogger(__name__)


# #############################################################################
# Shared assertions
# #############################################################################


def _check_counts(
    test_case: hunitest.TestCase,
    bids: List[rnbacaau.Bid],
    asks: List[rnbacaau.Ask],
    n_bids: int,
    n_asks: int,
) -> None:
    """
    Check per-tier order counts and uniqueness of participant identifiers.

    :param test_case: test case providing the assertion methods
    :param bids: submitted buyer orders
    :param asks: submitted seller orders
    :param n_bids: buyer order count per tier
    :param n_asks: seller order count per tier
    """
    # Each tier must contribute exactly the requested number of distinct orders.
    tier_names = {b.c_level_min for b in bids}
    actual_counts = {
        "bids": len(bids),
        "asks": len(asks),
        "buyers": len({b.buyer_id for b in bids}),
        "sellers": len({a.seller_id for a in asks}),
    }
    expected_counts = {
        "bids": n_bids * len(tier_names),
        "asks": n_asks * len(tier_names),
        "buyers": n_bids * len(tier_names),
        "sellers": n_asks * len(tier_names),
    }
    test_case.assert_equal(str(actual_counts), str(expected_counts))


def _clear_book(
    test_case: hunitest.TestCase,
    bids: List[rnbacaau.Bid],
    asks: List[rnbacaau.Ask],
) -> Dict[str, rnbacaau.TierClearResult]:
    """
    Submit fresh scenario orders and clear them through the auction API.

    :param test_case: test case providing the assertion methods
    :param bids: submitted buyer orders
    :param asks: submitted seller orders
    """
    # Prepare inputs: submit both sides before clearing any tier.
    book = rnbacaau.OrderBook()
    for bid in bids:
        book.submit_bid(bid)
    for ask in asks:
        book.submit_ask(ask)
    # Run test.
    results = book.clear_round()
    return results


def _check_results(
    test_case: hunitest.TestCase,
    bids: List[rnbacaau.Bid],
    asks: List[rnbacaau.Ask],
    results: Dict[str, rnbacaau.TierClearResult],
    regime: str,
) -> None:
    """
    Check volume conservation, the promised regime, and valid fill prices.

    :param test_case: test case providing the assertion methods
    :param bids: submitted buyer orders
    :param asks: submitted seller orders
    :param results: auction clearing results keyed by tier
    :param regime: matching behavior promised by the named family
    """
    prices = []
    # Check outputs.
    for tier, result in results.items():
        # Account for every submitted task, including quantities left unmatched.
        tier_bids = [b for b in bids if b.c_level_min == tier]
        demand = sum(b.n_tasks for b in tier_bids)
        supply = sum(a.n_tasks for a in asks if a.c_level == tier)
        matched = sum(f.n_tasks for f in result.fills)
        actual_volumes = (
            matched + result.unfilled_bid_n_tasks,
            matched + result.unfilled_ask_n_tasks,
        )
        expected_volumes = (demand, supply)
        actual_text = str(actual_volumes)
        expected_text = str(expected_volumes)
        test_case.assert_equal(actual_text, expected_text)
        # Check the named regime separately from the common auction invariants.
        if regime == "no_cross":
            test_case.assertEqual(matched, 0)
            test_case.assertIsNone(result.clearing_price)
            test_case.assertLess(
                max(b.p_max for b in bids),
                min(a.p_min for a in asks),
            )
        else:
            test_case.assertEqual(matched, min(demand, supply))
            test_case.assertIsNotNone(result.clearing_price)
            prices.append(result.clearing_price)
            if regime == "demand":
                test_case.assertGreater(demand, supply)
            elif regime == "supply":
                test_case.assertGreater(supply, demand)
            else:
                test_case.assertEqual(demand, supply)
            # Every trade must respect both participants' price limits.
            for fill in result.fills:
                bid = next(b for b in bids if b.buyer_id == fill.buyer_id)
                ask = next(a for a in asks if a.seller_id == fill.seller_id)
                test_case.assertLessEqual(ask.p_min, fill.price)
                test_case.assertLessEqual(fill.price, bid.p_max)
        # All fills within a tier share its clearing price and have positive volume.
        test_case.assertTrue(
            all(
                f.c_level == tier
                and f.price == result.clearing_price
                and f.n_tasks > 0
                for f in result.fills
            )
        )
    # Distinct tier price bands must remain distinct after clearing.
    if regime == "multi_tier":
        test_case.assertGreaterEqual(len(prices), 2)
        test_case.assertEqual(len(set(prices)), len(prices))


def _check_contracts(
    test_case: hunitest.TestCase,
    bids: List[rnbacaau.Bid],
    results: Dict[str, rnbacaau.TierClearResult],
) -> None:
    """
    Check that every matched task becomes a successfully dispatched contract.

    :param test_case: test case providing the assertion methods
    :param bids: submitted buyer orders
    :param results: auction clearing results keyed by tier
    """
    # Exercise the existing contract path until M10 supplies the round runner.
    contracts = rnocodis.build_contracts(bids, results)
    fulfillment_func = lambda contract: True
    rnocodis.dispatch_contracts(contracts, fulfillment_func=fulfillment_func)
    test_case.assertEqual(
        sum(c.n_tasks for c in contracts),
        sum(f.n_tasks for r in results.values() for f in r.fills),
    )
    test_case.assertTrue(all(c.fulfilled for c in contracts))


def _check_regime(
    test_case: hunitest.TestCase,
    factory: Callable[..., rnmascen.Scenario],
    regime: str,
    seed: int,
    n_bids: int,
    n_asks: int,
    scale: int,
) -> None:
    """
    Check one reproducible scenario case, independently runnable by its test.

    :param test_case: test case providing the assertion methods
    :param factory: named scenario generator under test
    :param regime: matching behavior promised by the named family
    :param seed: seed identifying one reproducible input book
    :param n_bids: buyer order count per tier
    :param n_asks: seller order count per tier
    :param scale: positive quantity and price multiplier
    """
    # Prepare inputs: retain global state to detect accidental random.seed use.
    global_state = random.getstate()
    _LOG.debug(
        "factory='%s', seed='%s', counts='%s', scale='%s'",
        factory.__name__,
        seed,
        (n_bids, n_asks),
        scale,
    )
    kwargs = dict(
        n_bids=n_bids,
        n_asks=n_asks,
        quantity_scale=scale,
        price_scale=scale,
    )
    # Run test.
    bids, asks = factory(seed=seed, **kwargs)
    second_bids, second_asks = factory(seed=seed, **kwargs)
    # Check outputs.
    test_case.assert_equal(str((bids, asks)), str((second_bids, second_asks)))
    for first, second in ((bids, second_bids), (asks, second_asks)):
        test_case.assertIsNot(first, second)
        for a, b in zip(first, second):
            test_case.assertIsNot(a, b)
    # Check outputs: counts, auction results, and contracts are independent checks.
    _check_counts(test_case, bids, asks, n_bids, n_asks)
    results = _clear_book(test_case, bids, asks)
    _check_results(test_case, bids, asks, results, regime)
    _check_contracts(test_case, bids, results)
    # Mutating one result must not affect a second call's orders.
    bids[0].n_tasks = -1
    test_case.assertGreater(second_bids[0].n_tasks, 0)
    test_case.assertEqual(random.getstate(), global_state)


def _check_variation(
    test_case: hunitest.TestCase, factory: Callable[..., rnmascen.Scenario]
) -> None:
    """
    Verify that changing seeds produces varied scenario data.

    :param test_case: test case providing the assertion methods
    :param factory: named scenario generator under test
    """
    # Prepare inputs: preserve the original 30-seed, three-configuration matrix.
    fingerprints = set()
    for seed in range(30):
        for n_bids, n_asks, scale in ((1, 1, 1), (3, 2, 1), (2, 4, 10000)):
            scenario = factory(
                seed=seed,
                n_bids=n_bids,
                n_asks=n_asks,
                quantity_scale=scale,
                price_scale=scale,
            )
            fingerprints.add(str(scenario))
    # Check outputs: seeds must change values, not merely reproduce one fixed book.
    test_case.assertGreater(len(fingerprints), 30)


# #############################################################################
# Test_make_balanced_scenario
# #############################################################################


class Test_make_balanced_scenario(hunitest.TestCase):
    """
    Test balanced across seeds, sizes, scales, and invalid inputs.
    """

    def helper(self, seed: int, n_bids: int, n_asks: int, scale: int) -> None:
        """
        Run one explicit case using the shared scenario assertions.

        :param seed: seed identifying one reproducible input book
        :param n_bids: buyer order count per tier
        :param n_asks: seller order count per tier
        :param scale: positive quantity and price multiplier
        """
        # Prepare inputs.
        factory = rnmascen.make_balanced_scenario
        regime = "balanced"
        # Run test and check outputs.
        _check_regime(self, factory, regime, seed, n_bids, n_asks, scale)

    def test1(self) -> None:
        """
        Check seed 0, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 0, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test2(self) -> None:
        """
        Check seed 0, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 0, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test3(self) -> None:
        """
        Check seed 0, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 0, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test4(self) -> None:
        """
        Check seed 1, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 1, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test5(self) -> None:
        """
        Check seed 1, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 1, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test6(self) -> None:
        """
        Check seed 1, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 1, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test7(self) -> None:
        """
        Check seed 2, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 2, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test8(self) -> None:
        """
        Check seed 2, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 2, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test9(self) -> None:
        """
        Check seed 2, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 2, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test10(self) -> None:
        """
        Check seed 3, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 3, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test11(self) -> None:
        """
        Check seed 3, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 3, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test12(self) -> None:
        """
        Check seed 3, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 3, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test13(self) -> None:
        """
        Check seed 4, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 4, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test14(self) -> None:
        """
        Check seed 4, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 4, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test15(self) -> None:
        """
        Check seed 4, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 4, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test16(self) -> None:
        """
        Check seed 5, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 5, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test17(self) -> None:
        """
        Check seed 5, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 5, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test18(self) -> None:
        """
        Check seed 5, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 5, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test19(self) -> None:
        """
        Check seed 6, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 6, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test20(self) -> None:
        """
        Check seed 6, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 6, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test21(self) -> None:
        """
        Check seed 6, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 6, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test22(self) -> None:
        """
        Check seed 7, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 7, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test23(self) -> None:
        """
        Check seed 7, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 7, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test24(self) -> None:
        """
        Check seed 7, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 7, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test25(self) -> None:
        """
        Check seed 8, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 8, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test26(self) -> None:
        """
        Check seed 8, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 8, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test27(self) -> None:
        """
        Check seed 8, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 8, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test28(self) -> None:
        """
        Check seed 9, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 9, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test29(self) -> None:
        """
        Check seed 9, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 9, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test30(self) -> None:
        """
        Check seed 9, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 9, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test31(self) -> None:
        """
        Check seed 10, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 10, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test32(self) -> None:
        """
        Check seed 10, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 10, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test33(self) -> None:
        """
        Check seed 10, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 10, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test34(self) -> None:
        """
        Check seed 11, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 11, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test35(self) -> None:
        """
        Check seed 11, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 11, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test36(self) -> None:
        """
        Check seed 11, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 11, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test37(self) -> None:
        """
        Check seed 12, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 12, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test38(self) -> None:
        """
        Check seed 12, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 12, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test39(self) -> None:
        """
        Check seed 12, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 12, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test40(self) -> None:
        """
        Check seed 13, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 13, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test41(self) -> None:
        """
        Check seed 13, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 13, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test42(self) -> None:
        """
        Check seed 13, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 13, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test43(self) -> None:
        """
        Check seed 14, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 14, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test44(self) -> None:
        """
        Check seed 14, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 14, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test45(self) -> None:
        """
        Check seed 14, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 14, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test46(self) -> None:
        """
        Check seed 15, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 15, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test47(self) -> None:
        """
        Check seed 15, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 15, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test48(self) -> None:
        """
        Check seed 15, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 15, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test49(self) -> None:
        """
        Check seed 16, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 16, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test50(self) -> None:
        """
        Check seed 16, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 16, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test51(self) -> None:
        """
        Check seed 16, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 16, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test52(self) -> None:
        """
        Check seed 17, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 17, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test53(self) -> None:
        """
        Check seed 17, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 17, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test54(self) -> None:
        """
        Check seed 17, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 17, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test55(self) -> None:
        """
        Check seed 18, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 18, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test56(self) -> None:
        """
        Check seed 18, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 18, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test57(self) -> None:
        """
        Check seed 18, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 18, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test58(self) -> None:
        """
        Check seed 19, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 19, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test59(self) -> None:
        """
        Check seed 19, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 19, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test60(self) -> None:
        """
        Check seed 19, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 19, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test61(self) -> None:
        """
        Check seed 20, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 20, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test62(self) -> None:
        """
        Check seed 20, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 20, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test63(self) -> None:
        """
        Check seed 20, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 20, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test64(self) -> None:
        """
        Check seed 21, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 21, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test65(self) -> None:
        """
        Check seed 21, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 21, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test66(self) -> None:
        """
        Check seed 21, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 21, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test67(self) -> None:
        """
        Check seed 22, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 22, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test68(self) -> None:
        """
        Check seed 22, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 22, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test69(self) -> None:
        """
        Check seed 22, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 22, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test70(self) -> None:
        """
        Check seed 23, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 23, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test71(self) -> None:
        """
        Check seed 23, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 23, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test72(self) -> None:
        """
        Check seed 23, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 23, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test73(self) -> None:
        """
        Check seed 24, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 24, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test74(self) -> None:
        """
        Check seed 24, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 24, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test75(self) -> None:
        """
        Check seed 24, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 24, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test76(self) -> None:
        """
        Check seed 25, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 25, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test77(self) -> None:
        """
        Check seed 25, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 25, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test78(self) -> None:
        """
        Check seed 25, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 25, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test79(self) -> None:
        """
        Check seed 26, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 26, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test80(self) -> None:
        """
        Check seed 26, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 26, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test81(self) -> None:
        """
        Check seed 26, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 26, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test82(self) -> None:
        """
        Check seed 27, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 27, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test83(self) -> None:
        """
        Check seed 27, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 27, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test84(self) -> None:
        """
        Check seed 27, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 27, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test85(self) -> None:
        """
        Check seed 28, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 28, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test86(self) -> None:
        """
        Check seed 28, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 28, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test87(self) -> None:
        """
        Check seed 28, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 28, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test88(self) -> None:
        """
        Check seed 29, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 29, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test89(self) -> None:
        """
        Check seed 29, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 29, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test90(self) -> None:
        """
        Check seed 29, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 29, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test91(self) -> None:
        """
        Check that omitted optional arguments produce a nonempty scenario.
        """
        # Prepare inputs.
        factory = rnmascen.make_balanced_scenario
        # Run test.
        bids, asks = factory()
        # Check outputs.
        self.assertTrue(bids and asks)

    def test92(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"seed": True}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_balanced_scenario(**kwargs)

    def test93(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"n_bids": 0}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_balanced_scenario(**kwargs)

    def test94(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"n_asks": -1}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_balanced_scenario(**kwargs)

    def test95(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"quantity_scale": 0}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_balanced_scenario(**kwargs)

    def test96(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"price_scale": False}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_balanced_scenario(**kwargs)

    def test97(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"tier": ""}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_balanced_scenario(**kwargs)

    def test98(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"tier": " "}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_balanced_scenario(**kwargs)

    def test99(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"tier": None}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_balanced_scenario(**kwargs)

    def test100(self) -> None:
        """
        Reject a positional seed for the keyword-only family interface.
        """
        # Prepare inputs.
        seed = 0
        # Run test and check outputs.
        with self.assertRaises(TypeError):
            rnmascen.make_balanced_scenario(seed)

    def test101(self) -> None:
        """
        Check that the seed changes the generated book.
        """
        # Prepare inputs.
        factory = rnmascen.make_balanced_scenario
        # Run test and check outputs.
        _check_variation(self, factory)


# #############################################################################
# Test_make_excess_demand_scenario
# #############################################################################


class Test_make_excess_demand_scenario(hunitest.TestCase):
    """
    Test excess demand across seeds, sizes, scales, and invalid inputs.
    """

    def helper(self, seed: int, n_bids: int, n_asks: int, scale: int) -> None:
        """
        Run one explicit case using the shared scenario assertions.

        :param seed: seed identifying one reproducible input book
        :param n_bids: buyer order count per tier
        :param n_asks: seller order count per tier
        :param scale: positive quantity and price multiplier
        """
        # Prepare inputs.
        factory = rnmascen.make_excess_demand_scenario
        regime = "demand"
        # Run test and check outputs.
        _check_regime(self, factory, regime, seed, n_bids, n_asks, scale)

    def test1(self) -> None:
        """
        Check seed 0, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 0, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test2(self) -> None:
        """
        Check seed 0, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 0, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test3(self) -> None:
        """
        Check seed 0, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 0, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test4(self) -> None:
        """
        Check seed 1, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 1, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test5(self) -> None:
        """
        Check seed 1, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 1, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test6(self) -> None:
        """
        Check seed 1, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 1, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test7(self) -> None:
        """
        Check seed 2, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 2, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test8(self) -> None:
        """
        Check seed 2, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 2, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test9(self) -> None:
        """
        Check seed 2, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 2, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test10(self) -> None:
        """
        Check seed 3, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 3, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test11(self) -> None:
        """
        Check seed 3, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 3, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test12(self) -> None:
        """
        Check seed 3, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 3, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test13(self) -> None:
        """
        Check seed 4, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 4, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test14(self) -> None:
        """
        Check seed 4, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 4, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test15(self) -> None:
        """
        Check seed 4, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 4, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test16(self) -> None:
        """
        Check seed 5, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 5, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test17(self) -> None:
        """
        Check seed 5, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 5, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test18(self) -> None:
        """
        Check seed 5, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 5, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test19(self) -> None:
        """
        Check seed 6, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 6, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test20(self) -> None:
        """
        Check seed 6, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 6, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test21(self) -> None:
        """
        Check seed 6, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 6, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test22(self) -> None:
        """
        Check seed 7, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 7, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test23(self) -> None:
        """
        Check seed 7, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 7, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test24(self) -> None:
        """
        Check seed 7, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 7, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test25(self) -> None:
        """
        Check seed 8, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 8, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test26(self) -> None:
        """
        Check seed 8, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 8, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test27(self) -> None:
        """
        Check seed 8, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 8, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test28(self) -> None:
        """
        Check seed 9, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 9, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test29(self) -> None:
        """
        Check seed 9, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 9, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test30(self) -> None:
        """
        Check seed 9, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 9, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test31(self) -> None:
        """
        Check seed 10, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 10, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test32(self) -> None:
        """
        Check seed 10, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 10, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test33(self) -> None:
        """
        Check seed 10, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 10, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test34(self) -> None:
        """
        Check seed 11, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 11, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test35(self) -> None:
        """
        Check seed 11, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 11, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test36(self) -> None:
        """
        Check seed 11, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 11, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test37(self) -> None:
        """
        Check seed 12, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 12, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test38(self) -> None:
        """
        Check seed 12, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 12, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test39(self) -> None:
        """
        Check seed 12, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 12, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test40(self) -> None:
        """
        Check seed 13, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 13, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test41(self) -> None:
        """
        Check seed 13, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 13, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test42(self) -> None:
        """
        Check seed 13, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 13, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test43(self) -> None:
        """
        Check seed 14, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 14, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test44(self) -> None:
        """
        Check seed 14, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 14, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test45(self) -> None:
        """
        Check seed 14, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 14, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test46(self) -> None:
        """
        Check seed 15, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 15, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test47(self) -> None:
        """
        Check seed 15, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 15, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test48(self) -> None:
        """
        Check seed 15, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 15, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test49(self) -> None:
        """
        Check seed 16, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 16, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test50(self) -> None:
        """
        Check seed 16, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 16, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test51(self) -> None:
        """
        Check seed 16, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 16, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test52(self) -> None:
        """
        Check seed 17, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 17, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test53(self) -> None:
        """
        Check seed 17, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 17, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test54(self) -> None:
        """
        Check seed 17, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 17, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test55(self) -> None:
        """
        Check seed 18, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 18, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test56(self) -> None:
        """
        Check seed 18, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 18, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test57(self) -> None:
        """
        Check seed 18, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 18, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test58(self) -> None:
        """
        Check seed 19, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 19, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test59(self) -> None:
        """
        Check seed 19, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 19, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test60(self) -> None:
        """
        Check seed 19, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 19, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test61(self) -> None:
        """
        Check seed 20, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 20, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test62(self) -> None:
        """
        Check seed 20, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 20, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test63(self) -> None:
        """
        Check seed 20, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 20, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test64(self) -> None:
        """
        Check seed 21, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 21, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test65(self) -> None:
        """
        Check seed 21, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 21, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test66(self) -> None:
        """
        Check seed 21, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 21, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test67(self) -> None:
        """
        Check seed 22, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 22, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test68(self) -> None:
        """
        Check seed 22, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 22, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test69(self) -> None:
        """
        Check seed 22, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 22, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test70(self) -> None:
        """
        Check seed 23, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 23, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test71(self) -> None:
        """
        Check seed 23, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 23, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test72(self) -> None:
        """
        Check seed 23, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 23, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test73(self) -> None:
        """
        Check seed 24, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 24, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test74(self) -> None:
        """
        Check seed 24, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 24, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test75(self) -> None:
        """
        Check seed 24, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 24, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test76(self) -> None:
        """
        Check seed 25, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 25, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test77(self) -> None:
        """
        Check seed 25, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 25, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test78(self) -> None:
        """
        Check seed 25, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 25, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test79(self) -> None:
        """
        Check seed 26, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 26, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test80(self) -> None:
        """
        Check seed 26, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 26, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test81(self) -> None:
        """
        Check seed 26, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 26, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test82(self) -> None:
        """
        Check seed 27, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 27, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test83(self) -> None:
        """
        Check seed 27, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 27, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test84(self) -> None:
        """
        Check seed 27, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 27, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test85(self) -> None:
        """
        Check seed 28, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 28, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test86(self) -> None:
        """
        Check seed 28, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 28, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test87(self) -> None:
        """
        Check seed 28, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 28, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test88(self) -> None:
        """
        Check seed 29, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 29, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test89(self) -> None:
        """
        Check seed 29, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 29, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test90(self) -> None:
        """
        Check seed 29, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 29, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test91(self) -> None:
        """
        Check that omitted optional arguments produce a nonempty scenario.
        """
        # Prepare inputs.
        factory = rnmascen.make_excess_demand_scenario
        # Run test.
        bids, asks = factory()
        # Check outputs.
        self.assertTrue(bids and asks)

    def test92(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"seed": True}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_excess_demand_scenario(**kwargs)

    def test93(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"n_bids": 0}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_excess_demand_scenario(**kwargs)

    def test94(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"n_asks": -1}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_excess_demand_scenario(**kwargs)

    def test95(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"quantity_scale": 0}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_excess_demand_scenario(**kwargs)

    def test96(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"price_scale": False}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_excess_demand_scenario(**kwargs)

    def test97(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"tier": ""}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_excess_demand_scenario(**kwargs)

    def test98(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"tier": " "}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_excess_demand_scenario(**kwargs)

    def test99(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"tier": None}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_excess_demand_scenario(**kwargs)

    def test100(self) -> None:
        """
        Reject a positional seed for the keyword-only family interface.
        """
        # Prepare inputs.
        seed = 0
        # Run test and check outputs.
        with self.assertRaises(TypeError):
            rnmascen.make_excess_demand_scenario(seed)

    def test101(self) -> None:
        """
        Check that the seed changes the generated book.
        """
        # Prepare inputs.
        factory = rnmascen.make_excess_demand_scenario
        # Run test and check outputs.
        _check_variation(self, factory)


# #############################################################################
# Test_make_excess_supply_scenario
# #############################################################################


class Test_make_excess_supply_scenario(hunitest.TestCase):
    """
    Test excess supply across seeds, sizes, scales, and invalid inputs.
    """

    def helper(self, seed: int, n_bids: int, n_asks: int, scale: int) -> None:
        """
        Run one explicit case using the shared scenario assertions.

        :param seed: seed identifying one reproducible input book
        :param n_bids: buyer order count per tier
        :param n_asks: seller order count per tier
        :param scale: positive quantity and price multiplier
        """
        # Prepare inputs.
        factory = rnmascen.make_excess_supply_scenario
        regime = "supply"
        # Run test and check outputs.
        _check_regime(self, factory, regime, seed, n_bids, n_asks, scale)

    def test1(self) -> None:
        """
        Check seed 0, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 0, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test2(self) -> None:
        """
        Check seed 0, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 0, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test3(self) -> None:
        """
        Check seed 0, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 0, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test4(self) -> None:
        """
        Check seed 1, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 1, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test5(self) -> None:
        """
        Check seed 1, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 1, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test6(self) -> None:
        """
        Check seed 1, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 1, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test7(self) -> None:
        """
        Check seed 2, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 2, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test8(self) -> None:
        """
        Check seed 2, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 2, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test9(self) -> None:
        """
        Check seed 2, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 2, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test10(self) -> None:
        """
        Check seed 3, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 3, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test11(self) -> None:
        """
        Check seed 3, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 3, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test12(self) -> None:
        """
        Check seed 3, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 3, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test13(self) -> None:
        """
        Check seed 4, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 4, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test14(self) -> None:
        """
        Check seed 4, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 4, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test15(self) -> None:
        """
        Check seed 4, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 4, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test16(self) -> None:
        """
        Check seed 5, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 5, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test17(self) -> None:
        """
        Check seed 5, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 5, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test18(self) -> None:
        """
        Check seed 5, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 5, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test19(self) -> None:
        """
        Check seed 6, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 6, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test20(self) -> None:
        """
        Check seed 6, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 6, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test21(self) -> None:
        """
        Check seed 6, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 6, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test22(self) -> None:
        """
        Check seed 7, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 7, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test23(self) -> None:
        """
        Check seed 7, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 7, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test24(self) -> None:
        """
        Check seed 7, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 7, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test25(self) -> None:
        """
        Check seed 8, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 8, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test26(self) -> None:
        """
        Check seed 8, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 8, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test27(self) -> None:
        """
        Check seed 8, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 8, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test28(self) -> None:
        """
        Check seed 9, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 9, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test29(self) -> None:
        """
        Check seed 9, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 9, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test30(self) -> None:
        """
        Check seed 9, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 9, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test31(self) -> None:
        """
        Check seed 10, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 10, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test32(self) -> None:
        """
        Check seed 10, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 10, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test33(self) -> None:
        """
        Check seed 10, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 10, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test34(self) -> None:
        """
        Check seed 11, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 11, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test35(self) -> None:
        """
        Check seed 11, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 11, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test36(self) -> None:
        """
        Check seed 11, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 11, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test37(self) -> None:
        """
        Check seed 12, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 12, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test38(self) -> None:
        """
        Check seed 12, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 12, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test39(self) -> None:
        """
        Check seed 12, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 12, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test40(self) -> None:
        """
        Check seed 13, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 13, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test41(self) -> None:
        """
        Check seed 13, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 13, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test42(self) -> None:
        """
        Check seed 13, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 13, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test43(self) -> None:
        """
        Check seed 14, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 14, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test44(self) -> None:
        """
        Check seed 14, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 14, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test45(self) -> None:
        """
        Check seed 14, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 14, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test46(self) -> None:
        """
        Check seed 15, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 15, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test47(self) -> None:
        """
        Check seed 15, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 15, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test48(self) -> None:
        """
        Check seed 15, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 15, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test49(self) -> None:
        """
        Check seed 16, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 16, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test50(self) -> None:
        """
        Check seed 16, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 16, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test51(self) -> None:
        """
        Check seed 16, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 16, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test52(self) -> None:
        """
        Check seed 17, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 17, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test53(self) -> None:
        """
        Check seed 17, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 17, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test54(self) -> None:
        """
        Check seed 17, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 17, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test55(self) -> None:
        """
        Check seed 18, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 18, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test56(self) -> None:
        """
        Check seed 18, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 18, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test57(self) -> None:
        """
        Check seed 18, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 18, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test58(self) -> None:
        """
        Check seed 19, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 19, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test59(self) -> None:
        """
        Check seed 19, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 19, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test60(self) -> None:
        """
        Check seed 19, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 19, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test61(self) -> None:
        """
        Check seed 20, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 20, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test62(self) -> None:
        """
        Check seed 20, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 20, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test63(self) -> None:
        """
        Check seed 20, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 20, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test64(self) -> None:
        """
        Check seed 21, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 21, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test65(self) -> None:
        """
        Check seed 21, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 21, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test66(self) -> None:
        """
        Check seed 21, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 21, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test67(self) -> None:
        """
        Check seed 22, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 22, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test68(self) -> None:
        """
        Check seed 22, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 22, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test69(self) -> None:
        """
        Check seed 22, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 22, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test70(self) -> None:
        """
        Check seed 23, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 23, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test71(self) -> None:
        """
        Check seed 23, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 23, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test72(self) -> None:
        """
        Check seed 23, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 23, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test73(self) -> None:
        """
        Check seed 24, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 24, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test74(self) -> None:
        """
        Check seed 24, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 24, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test75(self) -> None:
        """
        Check seed 24, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 24, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test76(self) -> None:
        """
        Check seed 25, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 25, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test77(self) -> None:
        """
        Check seed 25, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 25, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test78(self) -> None:
        """
        Check seed 25, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 25, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test79(self) -> None:
        """
        Check seed 26, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 26, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test80(self) -> None:
        """
        Check seed 26, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 26, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test81(self) -> None:
        """
        Check seed 26, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 26, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test82(self) -> None:
        """
        Check seed 27, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 27, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test83(self) -> None:
        """
        Check seed 27, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 27, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test84(self) -> None:
        """
        Check seed 27, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 27, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test85(self) -> None:
        """
        Check seed 28, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 28, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test86(self) -> None:
        """
        Check seed 28, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 28, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test87(self) -> None:
        """
        Check seed 28, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 28, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test88(self) -> None:
        """
        Check seed 29, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 29, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test89(self) -> None:
        """
        Check seed 29, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 29, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test90(self) -> None:
        """
        Check seed 29, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 29, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test91(self) -> None:
        """
        Check that omitted optional arguments produce a nonempty scenario.
        """
        # Prepare inputs.
        factory = rnmascen.make_excess_supply_scenario
        # Run test.
        bids, asks = factory()
        # Check outputs.
        self.assertTrue(bids and asks)

    def test92(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"seed": True}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_excess_supply_scenario(**kwargs)

    def test93(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"n_bids": 0}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_excess_supply_scenario(**kwargs)

    def test94(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"n_asks": -1}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_excess_supply_scenario(**kwargs)

    def test95(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"quantity_scale": 0}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_excess_supply_scenario(**kwargs)

    def test96(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"price_scale": False}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_excess_supply_scenario(**kwargs)

    def test97(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"tier": ""}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_excess_supply_scenario(**kwargs)

    def test98(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"tier": " "}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_excess_supply_scenario(**kwargs)

    def test99(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"tier": None}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_excess_supply_scenario(**kwargs)

    def test100(self) -> None:
        """
        Reject a positional seed for the keyword-only family interface.
        """
        # Prepare inputs.
        seed = 0
        # Run test and check outputs.
        with self.assertRaises(TypeError):
            rnmascen.make_excess_supply_scenario(seed)

    def test101(self) -> None:
        """
        Check that the seed changes the generated book.
        """
        # Prepare inputs.
        factory = rnmascen.make_excess_supply_scenario
        # Run test and check outputs.
        _check_variation(self, factory)


# #############################################################################
# Test_make_multi_tier_scenario
# #############################################################################


class Test_make_multi_tier_scenario(hunitest.TestCase):
    """
    Test multi tier across seeds, sizes, scales, and invalid inputs.
    """

    def helper(self, seed: int, n_bids: int, n_asks: int, scale: int) -> None:
        """
        Run one explicit case using the shared scenario assertions.

        :param seed: seed identifying one reproducible input book
        :param n_bids: buyer order count per tier
        :param n_asks: seller order count per tier
        :param scale: positive quantity and price multiplier
        """
        # Prepare inputs.
        factory = rnmascen.make_multi_tier_scenario
        regime = "multi_tier"
        # Run test and check outputs.
        _check_regime(self, factory, regime, seed, n_bids, n_asks, scale)

    def test1(self) -> None:
        """
        Check seed 0, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 0, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test2(self) -> None:
        """
        Check seed 0, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 0, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test3(self) -> None:
        """
        Check seed 0, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 0, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test4(self) -> None:
        """
        Check seed 1, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 1, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test5(self) -> None:
        """
        Check seed 1, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 1, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test6(self) -> None:
        """
        Check seed 1, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 1, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test7(self) -> None:
        """
        Check seed 2, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 2, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test8(self) -> None:
        """
        Check seed 2, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 2, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test9(self) -> None:
        """
        Check seed 2, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 2, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test10(self) -> None:
        """
        Check seed 3, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 3, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test11(self) -> None:
        """
        Check seed 3, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 3, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test12(self) -> None:
        """
        Check seed 3, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 3, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test13(self) -> None:
        """
        Check seed 4, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 4, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test14(self) -> None:
        """
        Check seed 4, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 4, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test15(self) -> None:
        """
        Check seed 4, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 4, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test16(self) -> None:
        """
        Check seed 5, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 5, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test17(self) -> None:
        """
        Check seed 5, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 5, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test18(self) -> None:
        """
        Check seed 5, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 5, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test19(self) -> None:
        """
        Check seed 6, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 6, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test20(self) -> None:
        """
        Check seed 6, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 6, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test21(self) -> None:
        """
        Check seed 6, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 6, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test22(self) -> None:
        """
        Check seed 7, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 7, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test23(self) -> None:
        """
        Check seed 7, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 7, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test24(self) -> None:
        """
        Check seed 7, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 7, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test25(self) -> None:
        """
        Check seed 8, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 8, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test26(self) -> None:
        """
        Check seed 8, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 8, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test27(self) -> None:
        """
        Check seed 8, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 8, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test28(self) -> None:
        """
        Check seed 9, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 9, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test29(self) -> None:
        """
        Check seed 9, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 9, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test30(self) -> None:
        """
        Check seed 9, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 9, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test31(self) -> None:
        """
        Check seed 10, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 10, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test32(self) -> None:
        """
        Check seed 10, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 10, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test33(self) -> None:
        """
        Check seed 10, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 10, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test34(self) -> None:
        """
        Check seed 11, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 11, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test35(self) -> None:
        """
        Check seed 11, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 11, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test36(self) -> None:
        """
        Check seed 11, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 11, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test37(self) -> None:
        """
        Check seed 12, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 12, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test38(self) -> None:
        """
        Check seed 12, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 12, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test39(self) -> None:
        """
        Check seed 12, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 12, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test40(self) -> None:
        """
        Check seed 13, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 13, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test41(self) -> None:
        """
        Check seed 13, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 13, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test42(self) -> None:
        """
        Check seed 13, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 13, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test43(self) -> None:
        """
        Check seed 14, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 14, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test44(self) -> None:
        """
        Check seed 14, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 14, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test45(self) -> None:
        """
        Check seed 14, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 14, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test46(self) -> None:
        """
        Check seed 15, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 15, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test47(self) -> None:
        """
        Check seed 15, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 15, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test48(self) -> None:
        """
        Check seed 15, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 15, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test49(self) -> None:
        """
        Check seed 16, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 16, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test50(self) -> None:
        """
        Check seed 16, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 16, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test51(self) -> None:
        """
        Check seed 16, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 16, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test52(self) -> None:
        """
        Check seed 17, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 17, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test53(self) -> None:
        """
        Check seed 17, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 17, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test54(self) -> None:
        """
        Check seed 17, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 17, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test55(self) -> None:
        """
        Check seed 18, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 18, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test56(self) -> None:
        """
        Check seed 18, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 18, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test57(self) -> None:
        """
        Check seed 18, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 18, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test58(self) -> None:
        """
        Check seed 19, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 19, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test59(self) -> None:
        """
        Check seed 19, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 19, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test60(self) -> None:
        """
        Check seed 19, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 19, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test61(self) -> None:
        """
        Check seed 20, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 20, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test62(self) -> None:
        """
        Check seed 20, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 20, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test63(self) -> None:
        """
        Check seed 20, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 20, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test64(self) -> None:
        """
        Check seed 21, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 21, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test65(self) -> None:
        """
        Check seed 21, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 21, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test66(self) -> None:
        """
        Check seed 21, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 21, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test67(self) -> None:
        """
        Check seed 22, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 22, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test68(self) -> None:
        """
        Check seed 22, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 22, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test69(self) -> None:
        """
        Check seed 22, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 22, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test70(self) -> None:
        """
        Check seed 23, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 23, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test71(self) -> None:
        """
        Check seed 23, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 23, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test72(self) -> None:
        """
        Check seed 23, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 23, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test73(self) -> None:
        """
        Check seed 24, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 24, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test74(self) -> None:
        """
        Check seed 24, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 24, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test75(self) -> None:
        """
        Check seed 24, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 24, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test76(self) -> None:
        """
        Check seed 25, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 25, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test77(self) -> None:
        """
        Check seed 25, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 25, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test78(self) -> None:
        """
        Check seed 25, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 25, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test79(self) -> None:
        """
        Check seed 26, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 26, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test80(self) -> None:
        """
        Check seed 26, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 26, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test81(self) -> None:
        """
        Check seed 26, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 26, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test82(self) -> None:
        """
        Check seed 27, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 27, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test83(self) -> None:
        """
        Check seed 27, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 27, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test84(self) -> None:
        """
        Check seed 27, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 27, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test85(self) -> None:
        """
        Check seed 28, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 28, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test86(self) -> None:
        """
        Check seed 28, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 28, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test87(self) -> None:
        """
        Check seed 28, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 28, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test88(self) -> None:
        """
        Check seed 29, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 29, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test89(self) -> None:
        """
        Check seed 29, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 29, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test90(self) -> None:
        """
        Check seed 29, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 29, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test91(self) -> None:
        """
        Check that omitted optional arguments produce a nonempty scenario.
        """
        # Prepare inputs.
        factory = rnmascen.make_multi_tier_scenario
        # Run test.
        bids, asks = factory()
        # Check outputs.
        self.assertTrue(bids and asks)

    def test92(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"seed": True}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_multi_tier_scenario(**kwargs)

    def test93(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"n_bids": 0}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_multi_tier_scenario(**kwargs)

    def test94(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"n_asks": -1}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_multi_tier_scenario(**kwargs)

    def test95(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"quantity_scale": 0}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_multi_tier_scenario(**kwargs)

    def test96(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"price_scale": False}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_multi_tier_scenario(**kwargs)

    def test97(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"tiers": ("only",)}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_multi_tier_scenario(**kwargs)

    def test98(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"tiers": ("x", "x")}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_multi_tier_scenario(**kwargs)

    def test99(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"tiers": "ab"}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_multi_tier_scenario(**kwargs)

    def test100(self) -> None:
        """
        Reject a positional seed for the keyword-only family interface.
        """
        # Prepare inputs.
        seed = 0
        # Run test and check outputs.
        with self.assertRaises(TypeError):
            rnmascen.make_multi_tier_scenario(seed)

    def test101(self) -> None:
        """
        Check that the seed changes the generated book.
        """
        # Prepare inputs.
        factory = rnmascen.make_multi_tier_scenario
        # Run test and check outputs.
        _check_variation(self, factory)


# #############################################################################
# Test_make_no_cross_scenario
# #############################################################################


class Test_make_no_cross_scenario(hunitest.TestCase):
    """
    Test no cross across seeds, sizes, scales, and invalid inputs.
    """

    def helper(self, seed: int, n_bids: int, n_asks: int, scale: int) -> None:
        """
        Run one explicit case using the shared scenario assertions.

        :param seed: seed identifying one reproducible input book
        :param n_bids: buyer order count per tier
        :param n_asks: seller order count per tier
        :param scale: positive quantity and price multiplier
        """
        # Prepare inputs.
        factory = rnmascen.make_no_cross_scenario
        regime = "no_cross"
        # Run test and check outputs.
        _check_regime(self, factory, regime, seed, n_bids, n_asks, scale)

    def test1(self) -> None:
        """
        Check seed 0, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 0, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test2(self) -> None:
        """
        Check seed 0, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 0, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test3(self) -> None:
        """
        Check seed 0, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 0, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test4(self) -> None:
        """
        Check seed 1, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 1, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test5(self) -> None:
        """
        Check seed 1, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 1, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test6(self) -> None:
        """
        Check seed 1, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 1, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test7(self) -> None:
        """
        Check seed 2, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 2, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test8(self) -> None:
        """
        Check seed 2, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 2, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test9(self) -> None:
        """
        Check seed 2, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 2, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test10(self) -> None:
        """
        Check seed 3, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 3, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test11(self) -> None:
        """
        Check seed 3, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 3, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test12(self) -> None:
        """
        Check seed 3, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 3, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test13(self) -> None:
        """
        Check seed 4, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 4, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test14(self) -> None:
        """
        Check seed 4, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 4, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test15(self) -> None:
        """
        Check seed 4, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 4, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test16(self) -> None:
        """
        Check seed 5, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 5, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test17(self) -> None:
        """
        Check seed 5, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 5, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test18(self) -> None:
        """
        Check seed 5, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 5, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test19(self) -> None:
        """
        Check seed 6, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 6, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test20(self) -> None:
        """
        Check seed 6, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 6, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test21(self) -> None:
        """
        Check seed 6, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 6, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test22(self) -> None:
        """
        Check seed 7, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 7, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test23(self) -> None:
        """
        Check seed 7, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 7, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test24(self) -> None:
        """
        Check seed 7, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 7, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test25(self) -> None:
        """
        Check seed 8, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 8, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test26(self) -> None:
        """
        Check seed 8, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 8, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test27(self) -> None:
        """
        Check seed 8, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 8, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test28(self) -> None:
        """
        Check seed 9, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 9, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test29(self) -> None:
        """
        Check seed 9, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 9, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test30(self) -> None:
        """
        Check seed 9, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 9, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test31(self) -> None:
        """
        Check seed 10, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 10, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test32(self) -> None:
        """
        Check seed 10, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 10, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test33(self) -> None:
        """
        Check seed 10, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 10, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test34(self) -> None:
        """
        Check seed 11, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 11, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test35(self) -> None:
        """
        Check seed 11, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 11, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test36(self) -> None:
        """
        Check seed 11, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 11, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test37(self) -> None:
        """
        Check seed 12, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 12, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test38(self) -> None:
        """
        Check seed 12, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 12, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test39(self) -> None:
        """
        Check seed 12, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 12, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test40(self) -> None:
        """
        Check seed 13, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 13, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test41(self) -> None:
        """
        Check seed 13, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 13, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test42(self) -> None:
        """
        Check seed 13, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 13, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test43(self) -> None:
        """
        Check seed 14, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 14, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test44(self) -> None:
        """
        Check seed 14, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 14, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test45(self) -> None:
        """
        Check seed 14, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 14, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test46(self) -> None:
        """
        Check seed 15, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 15, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test47(self) -> None:
        """
        Check seed 15, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 15, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test48(self) -> None:
        """
        Check seed 15, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 15, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test49(self) -> None:
        """
        Check seed 16, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 16, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test50(self) -> None:
        """
        Check seed 16, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 16, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test51(self) -> None:
        """
        Check seed 16, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 16, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test52(self) -> None:
        """
        Check seed 17, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 17, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test53(self) -> None:
        """
        Check seed 17, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 17, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test54(self) -> None:
        """
        Check seed 17, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 17, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test55(self) -> None:
        """
        Check seed 18, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 18, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test56(self) -> None:
        """
        Check seed 18, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 18, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test57(self) -> None:
        """
        Check seed 18, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 18, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test58(self) -> None:
        """
        Check seed 19, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 19, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test59(self) -> None:
        """
        Check seed 19, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 19, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test60(self) -> None:
        """
        Check seed 19, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 19, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test61(self) -> None:
        """
        Check seed 20, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 20, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test62(self) -> None:
        """
        Check seed 20, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 20, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test63(self) -> None:
        """
        Check seed 20, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 20, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test64(self) -> None:
        """
        Check seed 21, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 21, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test65(self) -> None:
        """
        Check seed 21, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 21, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test66(self) -> None:
        """
        Check seed 21, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 21, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test67(self) -> None:
        """
        Check seed 22, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 22, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test68(self) -> None:
        """
        Check seed 22, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 22, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test69(self) -> None:
        """
        Check seed 22, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 22, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test70(self) -> None:
        """
        Check seed 23, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 23, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test71(self) -> None:
        """
        Check seed 23, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 23, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test72(self) -> None:
        """
        Check seed 23, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 23, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test73(self) -> None:
        """
        Check seed 24, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 24, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test74(self) -> None:
        """
        Check seed 24, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 24, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test75(self) -> None:
        """
        Check seed 24, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 24, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test76(self) -> None:
        """
        Check seed 25, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 25, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test77(self) -> None:
        """
        Check seed 25, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 25, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test78(self) -> None:
        """
        Check seed 25, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 25, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test79(self) -> None:
        """
        Check seed 26, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 26, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test80(self) -> None:
        """
        Check seed 26, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 26, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test81(self) -> None:
        """
        Check seed 26, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 26, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test82(self) -> None:
        """
        Check seed 27, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 27, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test83(self) -> None:
        """
        Check seed 27, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 27, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test84(self) -> None:
        """
        Check seed 27, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 27, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test85(self) -> None:
        """
        Check seed 28, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 28, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test86(self) -> None:
        """
        Check seed 28, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 28, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test87(self) -> None:
        """
        Check seed 28, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 28, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test88(self) -> None:
        """
        Check seed 29, 1 bids, 1 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 29, 1, 1, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test89(self) -> None:
        """
        Check seed 29, 3 bids, 2 asks, and scale 1.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 29, 3, 2, 1
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test90(self) -> None:
        """
        Check seed 29, 2 bids, 4 asks, and scale 10000.
        """
        # Prepare inputs.
        seed, n_bids, n_asks, scale = 29, 2, 4, 10000
        # Run test and check outputs.
        self.helper(seed, n_bids, n_asks, scale)

    def test91(self) -> None:
        """
        Check that omitted optional arguments produce a nonempty scenario.
        """
        # Prepare inputs.
        factory = rnmascen.make_no_cross_scenario
        # Run test.
        bids, asks = factory()
        # Check outputs.
        self.assertTrue(bids and asks)

    def test92(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"seed": True}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_no_cross_scenario(**kwargs)

    def test93(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"n_bids": 0}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_no_cross_scenario(**kwargs)

    def test94(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"n_asks": -1}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_no_cross_scenario(**kwargs)

    def test95(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"quantity_scale": 0}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_no_cross_scenario(**kwargs)

    def test96(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"price_scale": False}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_no_cross_scenario(**kwargs)

    def test97(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"tier": ""}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_no_cross_scenario(**kwargs)

    def test98(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"tier": " "}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_no_cross_scenario(**kwargs)

    def test99(self) -> None:
        """
        Reject an invalid family argument.
        """
        # Prepare inputs.
        kwargs = {"tier": None}
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_no_cross_scenario(**kwargs)

    def test100(self) -> None:
        """
        Reject a positional seed for the keyword-only family interface.
        """
        # Prepare inputs.
        seed = 0
        # Run test and check outputs.
        with self.assertRaises(TypeError):
            rnmascen.make_no_cross_scenario(seed)

    def test101(self) -> None:
        """
        Check that the seed changes the generated book.
        """
        # Prepare inputs.
        factory = rnmascen.make_no_cross_scenario
        # Run test and check outputs.
        _check_variation(self, factory)


# #############################################################################
# Test_make_random_scenario
# #############################################################################


class Test_make_random_scenario(hunitest.TestCase):
    """
    Test unrestricted generation with separately runnable boundary and seed cases.
    """

    def test1(self) -> None:
        """
        Verify seeded reproducibility and unchanged global random state.
        """
        # Prepare inputs and expected outputs.
        seed = 42
        n_bids = 30
        n_asks = 25
        tiers = ["standard", "frontier"]
        before = random.getstate()
        # Run test.
        first = rnmascen.make_random_scenario(seed, n_bids, n_asks, tiers)
        second = rnmascen.make_random_scenario(seed, n_bids, n_asks, tiers)
        # Check outputs.
        self.assert_equal(str(first), str(second))
        self.assert_equal(str(random.getstate()), str(before))
        for a, b in zip(first, second):
            self.assertIsNot(a, b)
            for x, y in zip(a, b):
                self.assertIsNot(x, y)
        first[0][0].n_tasks = 999
        self.assertNotEqual(first[0][0].n_tasks, second[0][0].n_tasks)

    def helper(self, n_bids: int, n_asks: int) -> None:
        """
        Check one count boundary and the unrestricted generator field ranges.

        :param n_bids: buyer order count per tier
        :param n_asks: seller order count per tier
        """
        # Prepare inputs.
        seed = -3
        tiers = ("custom",)
        # Run test.
        bids, asks = rnmascen.make_random_scenario(seed, n_bids, n_asks, tiers)
        # Check outputs.
        actual_orders = (
            len(bids),
            len(asks),
            [b.buyer_id for b in bids],
            [a.seller_id for a in asks],
        )
        expected_orders = (
            n_bids,
            n_asks,
            [f"buyer_{i}" for i in range(n_bids)],
            [f"seller_{i}" for i in range(n_asks)],
        )
        self.assert_equal(str(actual_orders), str(expected_orders))
        for order in bids + asks:
            self.assertGreaterEqual(order.n_tasks, 1)
            self.assertLessEqual(order.n_tasks, 100)
        for bid in bids:
            self.assert_equal(
                str((bid.c_level_min, bid.l_max, bid.r_min)),
                str(("custom", 2.0, 0.90)),
            )
            self.assertIsInstance(bid.p_max, float)
            self.assertTrue(1 <= bid.p_max <= 20)
        for ask in asks:
            self.assert_equal(
                str((ask.c_level, ask.l_typical, ask.r_typical)),
                str(("custom", 1.0, 0.95)),
            )
            self.assertIsInstance(ask.p_min, float)
            self.assertTrue(1 <= ask.p_min <= 20)

    def test2(self) -> None:
        """
        Check unrestricted generation with 0 bids and 0 asks.
        """
        # Prepare inputs.
        n_bids, n_asks = 0, 0
        # Run test and check outputs.
        self.helper(n_bids, n_asks)

    def test3(self) -> None:
        """
        Check unrestricted generation with 0 bids and 4 asks.
        """
        # Prepare inputs.
        n_bids, n_asks = 0, 4
        # Run test and check outputs.
        self.helper(n_bids, n_asks)

    def test4(self) -> None:
        """
        Check unrestricted generation with 4 bids and 0 asks.
        """
        # Prepare inputs.
        n_bids, n_asks = 4, 0
        # Run test and check outputs.
        self.helper(n_bids, n_asks)

    def test5(self) -> None:
        """
        Check unrestricted generation with 20 bids and 15 asks.
        """
        # Prepare inputs.
        n_bids, n_asks = 20, 15
        # Run test and check outputs.
        self.helper(n_bids, n_asks)

    def test6(self) -> None:
        """
        Reject an invalid unrestricted-generator argument.
        """
        # Prepare inputs.
        args = (True, 1, 1, ["x"])
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_random_scenario(*args)

    def test7(self) -> None:
        """
        Reject an invalid unrestricted-generator argument.
        """
        # Prepare inputs.
        args = (1.5, 1, 1, ["x"])
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_random_scenario(*args)

    def test8(self) -> None:
        """
        Reject an invalid unrestricted-generator argument.
        """
        # Prepare inputs.
        args = (1, -1, 1, ["x"])
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_random_scenario(*args)

    def test9(self) -> None:
        """
        Reject an invalid unrestricted-generator argument.
        """
        # Prepare inputs.
        args = (1, 1, -1, ["x"])
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_random_scenario(*args)

    def test10(self) -> None:
        """
        Reject an invalid unrestricted-generator argument.
        """
        # Prepare inputs.
        args = (1, True, 1, ["x"])
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_random_scenario(*args)

    def test11(self) -> None:
        """
        Reject an invalid unrestricted-generator argument.
        """
        # Prepare inputs.
        args = (1, 1, False, ["x"])
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_random_scenario(*args)

    def test12(self) -> None:
        """
        Reject an invalid unrestricted-generator argument.
        """
        # Prepare inputs.
        args = (1, 1.5, 1, ["x"])
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_random_scenario(*args)

    def test13(self) -> None:
        """
        Reject an invalid unrestricted-generator argument.
        """
        # Prepare inputs.
        args = (1, 1, "2", ["x"])
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_random_scenario(*args)

    def test14(self) -> None:
        """
        Reject an invalid unrestricted-generator argument.
        """
        # Prepare inputs.
        args = (1, 0, 0, [])
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_random_scenario(*args)

    def test15(self) -> None:
        """
        Reject an invalid unrestricted-generator argument.
        """
        # Prepare inputs.
        args = (1, 1, 1, "standard")
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_random_scenario(*args)

    def test16(self) -> None:
        """
        Reject an invalid unrestricted-generator argument.
        """
        # Prepare inputs.
        args = (1, 1, 1, b"x")
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_random_scenario(*args)

    def test17(self) -> None:
        """
        Reject an invalid unrestricted-generator argument.
        """
        # Prepare inputs.
        args = (1, 1, 1, {"x"})
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_random_scenario(*args)

    def test18(self) -> None:
        """
        Reject an invalid unrestricted-generator argument.
        """
        # Prepare inputs.
        args = (1, 1, 1, None)
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_random_scenario(*args)

    def test19(self) -> None:
        """
        Reject an invalid unrestricted-generator argument.
        """
        # Prepare inputs.
        args = (1, 1, 1, [""])
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_random_scenario(*args)

    def test20(self) -> None:
        """
        Reject an invalid unrestricted-generator argument.
        """
        # Prepare inputs.
        args = (1, 1, 1, [" "])
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_random_scenario(*args)

    def test21(self) -> None:
        """
        Reject an invalid unrestricted-generator argument.
        """
        # Prepare inputs.
        args = (1, 1, 1, ["x", "x"])
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_random_scenario(*args)

    def test22(self) -> None:
        """
        Reject an invalid unrestricted-generator argument.
        """
        # Prepare inputs.
        args = (1, 1, 1, [1])
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_random_scenario(*args)

    def test23(self) -> None:
        """
        Reject an invalid unrestricted-generator argument.
        """
        # Prepare inputs.
        args = (1, 1, 1, [["x"]])
        # Run test and check outputs.
        with self.assertRaises(AssertionError):
            rnmascen.make_random_scenario(*args)

    def _check_random_volume(self, seed: int) -> None:
        """
        Conserve volume and preserve valid fill fields for one seeded random book.

        :param seed: seed identifying one reproducible input book
        """
        # Prepare inputs.
        tiers = ["standard", "frontier"]
        n_bids = seed % 13
        n_asks = seed % 17
        # Run test.
        bids, asks = rnmascen.make_random_scenario(seed, n_bids, n_asks, tiers)
        book = rnbacaau.OrderBook()
        for bid in bids:
            book.submit_bid(bid)
        for ask in asks:
            book.submit_ask(ask)
        results = book.clear_round()
        # Check outputs.
        for tier, result in results.items():
            matched = sum(f.n_tasks for f in result.fills)
            self.assert_equal(
                str(matched + result.unfilled_bid_n_tasks),
                str(sum(b.n_tasks for b in bids if b.c_level_min == tier)),
            )
            self.assert_equal(
                str(matched + result.unfilled_ask_n_tasks),
                str(sum(a.n_tasks for a in asks if a.c_level == tier)),
            )
            for fill in result.fills:
                self.assertGreater(fill.n_tasks, 0)
                self.assert_equal(str(fill.c_level), str(tier))
                self.assert_equal(str(fill.price), str(result.clearing_price))

    def test24(self) -> None:
        """
        Conserve unrestricted auction volume for seed 0.
        """
        # Prepare inputs.
        seed = 0
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test25(self) -> None:
        """
        Conserve unrestricted auction volume for seed 1.
        """
        # Prepare inputs.
        seed = 1
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test26(self) -> None:
        """
        Conserve unrestricted auction volume for seed 2.
        """
        # Prepare inputs.
        seed = 2
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test27(self) -> None:
        """
        Conserve unrestricted auction volume for seed 3.
        """
        # Prepare inputs.
        seed = 3
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test28(self) -> None:
        """
        Conserve unrestricted auction volume for seed 4.
        """
        # Prepare inputs.
        seed = 4
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test29(self) -> None:
        """
        Conserve unrestricted auction volume for seed 5.
        """
        # Prepare inputs.
        seed = 5
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test30(self) -> None:
        """
        Conserve unrestricted auction volume for seed 6.
        """
        # Prepare inputs.
        seed = 6
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test31(self) -> None:
        """
        Conserve unrestricted auction volume for seed 7.
        """
        # Prepare inputs.
        seed = 7
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test32(self) -> None:
        """
        Conserve unrestricted auction volume for seed 8.
        """
        # Prepare inputs.
        seed = 8
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test33(self) -> None:
        """
        Conserve unrestricted auction volume for seed 9.
        """
        # Prepare inputs.
        seed = 9
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test34(self) -> None:
        """
        Conserve unrestricted auction volume for seed 10.
        """
        # Prepare inputs.
        seed = 10
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test35(self) -> None:
        """
        Conserve unrestricted auction volume for seed 11.
        """
        # Prepare inputs.
        seed = 11
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test36(self) -> None:
        """
        Conserve unrestricted auction volume for seed 12.
        """
        # Prepare inputs.
        seed = 12
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test37(self) -> None:
        """
        Conserve unrestricted auction volume for seed 13.
        """
        # Prepare inputs.
        seed = 13
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test38(self) -> None:
        """
        Conserve unrestricted auction volume for seed 14.
        """
        # Prepare inputs.
        seed = 14
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test39(self) -> None:
        """
        Conserve unrestricted auction volume for seed 15.
        """
        # Prepare inputs.
        seed = 15
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test40(self) -> None:
        """
        Conserve unrestricted auction volume for seed 16.
        """
        # Prepare inputs.
        seed = 16
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test41(self) -> None:
        """
        Conserve unrestricted auction volume for seed 17.
        """
        # Prepare inputs.
        seed = 17
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test42(self) -> None:
        """
        Conserve unrestricted auction volume for seed 18.
        """
        # Prepare inputs.
        seed = 18
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test43(self) -> None:
        """
        Conserve unrestricted auction volume for seed 19.
        """
        # Prepare inputs.
        seed = 19
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test44(self) -> None:
        """
        Conserve unrestricted auction volume for seed 20.
        """
        # Prepare inputs.
        seed = 20
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test45(self) -> None:
        """
        Conserve unrestricted auction volume for seed 21.
        """
        # Prepare inputs.
        seed = 21
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test46(self) -> None:
        """
        Conserve unrestricted auction volume for seed 22.
        """
        # Prepare inputs.
        seed = 22
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test47(self) -> None:
        """
        Conserve unrestricted auction volume for seed 23.
        """
        # Prepare inputs.
        seed = 23
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test48(self) -> None:
        """
        Conserve unrestricted auction volume for seed 24.
        """
        # Prepare inputs.
        seed = 24
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test49(self) -> None:
        """
        Conserve unrestricted auction volume for seed 25.
        """
        # Prepare inputs.
        seed = 25
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test50(self) -> None:
        """
        Conserve unrestricted auction volume for seed 26.
        """
        # Prepare inputs.
        seed = 26
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test51(self) -> None:
        """
        Conserve unrestricted auction volume for seed 27.
        """
        # Prepare inputs.
        seed = 27
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test52(self) -> None:
        """
        Conserve unrestricted auction volume for seed 28.
        """
        # Prepare inputs.
        seed = 28
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test53(self) -> None:
        """
        Conserve unrestricted auction volume for seed 29.
        """
        # Prepare inputs.
        seed = 29
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test54(self) -> None:
        """
        Conserve unrestricted auction volume for seed 30.
        """
        # Prepare inputs.
        seed = 30
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test55(self) -> None:
        """
        Conserve unrestricted auction volume for seed 31.
        """
        # Prepare inputs.
        seed = 31
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test56(self) -> None:
        """
        Conserve unrestricted auction volume for seed 32.
        """
        # Prepare inputs.
        seed = 32
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test57(self) -> None:
        """
        Conserve unrestricted auction volume for seed 33.
        """
        # Prepare inputs.
        seed = 33
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test58(self) -> None:
        """
        Conserve unrestricted auction volume for seed 34.
        """
        # Prepare inputs.
        seed = 34
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test59(self) -> None:
        """
        Conserve unrestricted auction volume for seed 35.
        """
        # Prepare inputs.
        seed = 35
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test60(self) -> None:
        """
        Conserve unrestricted auction volume for seed 36.
        """
        # Prepare inputs.
        seed = 36
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test61(self) -> None:
        """
        Conserve unrestricted auction volume for seed 37.
        """
        # Prepare inputs.
        seed = 37
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test62(self) -> None:
        """
        Conserve unrestricted auction volume for seed 38.
        """
        # Prepare inputs.
        seed = 38
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test63(self) -> None:
        """
        Conserve unrestricted auction volume for seed 39.
        """
        # Prepare inputs.
        seed = 39
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test64(self) -> None:
        """
        Conserve unrestricted auction volume for seed 40.
        """
        # Prepare inputs.
        seed = 40
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test65(self) -> None:
        """
        Conserve unrestricted auction volume for seed 41.
        """
        # Prepare inputs.
        seed = 41
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test66(self) -> None:
        """
        Conserve unrestricted auction volume for seed 42.
        """
        # Prepare inputs.
        seed = 42
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test67(self) -> None:
        """
        Conserve unrestricted auction volume for seed 43.
        """
        # Prepare inputs.
        seed = 43
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test68(self) -> None:
        """
        Conserve unrestricted auction volume for seed 44.
        """
        # Prepare inputs.
        seed = 44
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test69(self) -> None:
        """
        Conserve unrestricted auction volume for seed 45.
        """
        # Prepare inputs.
        seed = 45
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test70(self) -> None:
        """
        Conserve unrestricted auction volume for seed 46.
        """
        # Prepare inputs.
        seed = 46
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test71(self) -> None:
        """
        Conserve unrestricted auction volume for seed 47.
        """
        # Prepare inputs.
        seed = 47
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test72(self) -> None:
        """
        Conserve unrestricted auction volume for seed 48.
        """
        # Prepare inputs.
        seed = 48
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test73(self) -> None:
        """
        Conserve unrestricted auction volume for seed 49.
        """
        # Prepare inputs.
        seed = 49
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test74(self) -> None:
        """
        Conserve unrestricted auction volume for seed 50.
        """
        # Prepare inputs.
        seed = 50
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test75(self) -> None:
        """
        Conserve unrestricted auction volume for seed 51.
        """
        # Prepare inputs.
        seed = 51
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test76(self) -> None:
        """
        Conserve unrestricted auction volume for seed 52.
        """
        # Prepare inputs.
        seed = 52
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test77(self) -> None:
        """
        Conserve unrestricted auction volume for seed 53.
        """
        # Prepare inputs.
        seed = 53
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test78(self) -> None:
        """
        Conserve unrestricted auction volume for seed 54.
        """
        # Prepare inputs.
        seed = 54
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test79(self) -> None:
        """
        Conserve unrestricted auction volume for seed 55.
        """
        # Prepare inputs.
        seed = 55
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test80(self) -> None:
        """
        Conserve unrestricted auction volume for seed 56.
        """
        # Prepare inputs.
        seed = 56
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test81(self) -> None:
        """
        Conserve unrestricted auction volume for seed 57.
        """
        # Prepare inputs.
        seed = 57
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test82(self) -> None:
        """
        Conserve unrestricted auction volume for seed 58.
        """
        # Prepare inputs.
        seed = 58
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test83(self) -> None:
        """
        Conserve unrestricted auction volume for seed 59.
        """
        # Prepare inputs.
        seed = 59
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test84(self) -> None:
        """
        Conserve unrestricted auction volume for seed 60.
        """
        # Prepare inputs.
        seed = 60
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test85(self) -> None:
        """
        Conserve unrestricted auction volume for seed 61.
        """
        # Prepare inputs.
        seed = 61
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test86(self) -> None:
        """
        Conserve unrestricted auction volume for seed 62.
        """
        # Prepare inputs.
        seed = 62
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test87(self) -> None:
        """
        Conserve unrestricted auction volume for seed 63.
        """
        # Prepare inputs.
        seed = 63
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test88(self) -> None:
        """
        Conserve unrestricted auction volume for seed 64.
        """
        # Prepare inputs.
        seed = 64
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test89(self) -> None:
        """
        Conserve unrestricted auction volume for seed 65.
        """
        # Prepare inputs.
        seed = 65
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test90(self) -> None:
        """
        Conserve unrestricted auction volume for seed 66.
        """
        # Prepare inputs.
        seed = 66
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test91(self) -> None:
        """
        Conserve unrestricted auction volume for seed 67.
        """
        # Prepare inputs.
        seed = 67
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test92(self) -> None:
        """
        Conserve unrestricted auction volume for seed 68.
        """
        # Prepare inputs.
        seed = 68
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test93(self) -> None:
        """
        Conserve unrestricted auction volume for seed 69.
        """
        # Prepare inputs.
        seed = 69
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test94(self) -> None:
        """
        Conserve unrestricted auction volume for seed 70.
        """
        # Prepare inputs.
        seed = 70
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test95(self) -> None:
        """
        Conserve unrestricted auction volume for seed 71.
        """
        # Prepare inputs.
        seed = 71
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test96(self) -> None:
        """
        Conserve unrestricted auction volume for seed 72.
        """
        # Prepare inputs.
        seed = 72
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test97(self) -> None:
        """
        Conserve unrestricted auction volume for seed 73.
        """
        # Prepare inputs.
        seed = 73
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test98(self) -> None:
        """
        Conserve unrestricted auction volume for seed 74.
        """
        # Prepare inputs.
        seed = 74
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test99(self) -> None:
        """
        Conserve unrestricted auction volume for seed 75.
        """
        # Prepare inputs.
        seed = 75
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test100(self) -> None:
        """
        Conserve unrestricted auction volume for seed 76.
        """
        # Prepare inputs.
        seed = 76
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test101(self) -> None:
        """
        Conserve unrestricted auction volume for seed 77.
        """
        # Prepare inputs.
        seed = 77
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test102(self) -> None:
        """
        Conserve unrestricted auction volume for seed 78.
        """
        # Prepare inputs.
        seed = 78
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test103(self) -> None:
        """
        Conserve unrestricted auction volume for seed 79.
        """
        # Prepare inputs.
        seed = 79
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test104(self) -> None:
        """
        Conserve unrestricted auction volume for seed 80.
        """
        # Prepare inputs.
        seed = 80
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test105(self) -> None:
        """
        Conserve unrestricted auction volume for seed 81.
        """
        # Prepare inputs.
        seed = 81
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test106(self) -> None:
        """
        Conserve unrestricted auction volume for seed 82.
        """
        # Prepare inputs.
        seed = 82
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test107(self) -> None:
        """
        Conserve unrestricted auction volume for seed 83.
        """
        # Prepare inputs.
        seed = 83
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test108(self) -> None:
        """
        Conserve unrestricted auction volume for seed 84.
        """
        # Prepare inputs.
        seed = 84
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test109(self) -> None:
        """
        Conserve unrestricted auction volume for seed 85.
        """
        # Prepare inputs.
        seed = 85
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test110(self) -> None:
        """
        Conserve unrestricted auction volume for seed 86.
        """
        # Prepare inputs.
        seed = 86
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test111(self) -> None:
        """
        Conserve unrestricted auction volume for seed 87.
        """
        # Prepare inputs.
        seed = 87
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test112(self) -> None:
        """
        Conserve unrestricted auction volume for seed 88.
        """
        # Prepare inputs.
        seed = 88
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test113(self) -> None:
        """
        Conserve unrestricted auction volume for seed 89.
        """
        # Prepare inputs.
        seed = 89
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test114(self) -> None:
        """
        Conserve unrestricted auction volume for seed 90.
        """
        # Prepare inputs.
        seed = 90
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test115(self) -> None:
        """
        Conserve unrestricted auction volume for seed 91.
        """
        # Prepare inputs.
        seed = 91
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test116(self) -> None:
        """
        Conserve unrestricted auction volume for seed 92.
        """
        # Prepare inputs.
        seed = 92
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test117(self) -> None:
        """
        Conserve unrestricted auction volume for seed 93.
        """
        # Prepare inputs.
        seed = 93
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test118(self) -> None:
        """
        Conserve unrestricted auction volume for seed 94.
        """
        # Prepare inputs.
        seed = 94
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test119(self) -> None:
        """
        Conserve unrestricted auction volume for seed 95.
        """
        # Prepare inputs.
        seed = 95
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test120(self) -> None:
        """
        Conserve unrestricted auction volume for seed 96.
        """
        # Prepare inputs.
        seed = 96
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test121(self) -> None:
        """
        Conserve unrestricted auction volume for seed 97.
        """
        # Prepare inputs.
        seed = 97
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test122(self) -> None:
        """
        Conserve unrestricted auction volume for seed 98.
        """
        # Prepare inputs.
        seed = 98
        # Run test and check outputs.
        self._check_random_volume(seed)

    def test123(self) -> None:
        """
        Conserve unrestricted auction volume for seed 99.
        """
        # Prepare inputs.
        seed = 99
        # Run test and check outputs.
        self._check_random_volume(seed)
