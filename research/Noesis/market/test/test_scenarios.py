"""
Test reproducible varied scenario families; M10 runner integration remains pending.

Import as:

import research.Noesis.market.test.test_scenarios as rnmatestsce
"""

import logging
import random
from typing import Any, Callable, Dict, List

import helpers.hprint as hprint
import helpers.hunit_test as hunitest
import research.Noesis.market.batch_call_auction as rnbacaau
import research.Noesis.market.contract_dispatch as rnocodis
import research.Noesis.market.scenarios as rnmascen

_LOG = logging.getLogger(__name__)


# TODO(giles): You can use /testing.fix_unit_tests and /coding.factor_common_code


# TODO(giles): A bit weird since some of this stuff can be 
def _check_family_arguments(
    self_: hunitest.TestCase,
    factory: Callable[..., rnmascen.Scenario],
    invalid: List[Dict[str, Any]],
) -> None:
    """
    Check defaults and invalid named-family arguments without shared test state.

    :param self_: test case used for assertions
    :param factory: named family generator
    :param invalid: invalid keyword argument sets
    """
    _LOG.debug("Checking family '%s'", factory.__name__)
    # Run test.
    bids, asks = factory()
    # Check outputs.
    self_.assertTrue(bids and asks)
    for kwargs in invalid:
        with self_.subTest(kwargs=kwargs):
            with self_.assertRaises(AssertionError):
                factory(**kwargs)
    # Check that optional seed cannot be passed positionally.
    positional_seed = 0
    with self_.assertRaises(TypeError):
        factory(positional_seed)


# TODO(giles): Way more comments! Use the skill.
# The goal is to let user read the high level comments, instead of reading the
# entire code.
def _check_regime(
    self_: hunitest.TestCase,
    factory: Callable[..., rnmascen.Scenario],
    regime: str,
) -> None:
    """
    Exercise a named generator across seeds, sizes, and large numeric scales.

    :param self_: assertion owner
    :param factory: scenario family generator
    :param regime: promised auction behavior
    """
    _LOG.debug(hprint.to_str("regime"))
    # Prepare inputs.
    global_state = random.getstate()
    fingerprints = set()
    for seed in range(30):
        for n_bids, n_asks, scale in [(1, 1, 1), (3, 2, 1), (2, 4, 10000)]:
            # TODO(giles): Not a fan of pytest subtests, since they make it
            # different to repro. I would factor out the code and then 
            # have one test method to invocation. Maybe it's a pet peeve.
            # But we should specify it in the skills.
            with self_.subTest(seed=seed, counts=(n_bids, n_asks), scale=scale):
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
                self_.assert_equal(
                    str((bids, asks)), str((second_bids, second_asks))
                )
                fingerprints.add(str((bids, asks)))
                for first, second in ((bids, second_bids), (asks, second_asks)):
                    self_.assertIsNot(first, second)
                    for a, b in zip(first, second):
                        self_.assertIsNot(a, b)
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
                self_.assert_equal(str(actual_counts), str(expected_counts))
                book = rnbacaau.OrderBook()
                for bid in bids:
                    book.submit_bid(bid)
                for ask in asks:
                    book.submit_ask(ask)
                # Run test.
                results = book.clear_round()
                prices = []
                # Check outputs.
                for tier, result in results.items():
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
                    self_.assert_equal(actual_text, expected_text)
                    if regime == "no_cross":
                        self_.assertEqual(matched, 0)
                        self_.assertIsNone(result.clearing_price)
                        self_.assertLess(
                            max(b.p_max for b in bids),
                            min(a.p_min for a in asks),
                        )
                    else:
                        self_.assertEqual(matched, min(demand, supply))
                        self_.assertIsNotNone(result.clearing_price)
                        prices.append(result.clearing_price)
                        if regime == "demand":
                            self_.assertGreater(demand, supply)
                        elif regime == "supply":
                            self_.assertGreater(supply, demand)
                        else:
                            self_.assertEqual(demand, supply)
                        for fill in result.fills:
                            bid = next(
                                b for b in bids if b.buyer_id == fill.buyer_id
                            )
                            ask = next(
                                a for a in asks if a.seller_id == fill.seller_id
                            )
                            self_.assertLessEqual(ask.p_min, fill.price)
                            self_.assertLessEqual(fill.price, bid.p_max)
                    self_.assertTrue(
                        all(
                            f.c_level == tier
                            and f.price == result.clearing_price
                            and f.n_tasks > 0
                            for f in result.fills
                        )
                    )
                if regime == "multi_tier":
                    self_.assertGreaterEqual(len(prices), 2)
                    self_.assertEqual(len(set(prices)), len(prices))
                contracts = rnocodis.build_contracts(bids, results)
                fulfillment_func = lambda contract: True
                rnocodis.dispatch_contracts(
                    contracts, fulfillment_func=fulfillment_func
                )
                self_.assertEqual(
                    sum(c.n_tasks for c in contracts),
                    sum(f.n_tasks for r in results.values() for f in r.fills),
                )
                self_.assertTrue(all(c.fulfilled for c in contracts))
                bids[0].n_tasks = -1
                self_.assertGreater(second_bids[0].n_tasks, 0)
    self_.assertGreater(len(fingerprints), 30)
    self_.assertEqual(random.getstate(), global_state)


# #############################################################################
# Test_make_balanced_scenario
# #############################################################################


class Test_make_balanced_scenario(hunitest.TestCase):
    """
    Test varied balanced books.
    """

    def test1(self) -> None:
        """
        Preserve the regime across seeds, counts, and numeric scales.
        """
        # Prepare inputs.
        factory = rnmascen.make_balanced_scenario
        regime = "balanced"
        # Run test and check outputs.
        _check_regime(self, factory, regime)

    def test2(self) -> None:
        """
        Reject invalid family parameters and support default calls.
        """
        # Prepare inputs.
        factory = rnmascen.make_balanced_scenario
        invalid = [
            dict(seed=True),
            dict(n_bids=0),
            dict(n_asks=-1),
            dict(quantity_scale=0),
            dict(price_scale=False),
        ]
        invalid += [dict(tier=""), dict(tier=" "), dict(tier=None)]
        # Run test and check outputs.
        _check_family_arguments(self, factory, invalid)


# #############################################################################
# Test_make_excess_demand_scenario
# #############################################################################


class Test_make_excess_demand_scenario(hunitest.TestCase):
    """
    Test varied excess demand books.
    """

    def test1(self) -> None:
        """
        Preserve the regime across seeds, counts, and numeric scales.
        """
        # Prepare inputs.
        factory = rnmascen.make_excess_demand_scenario
        regime = "demand"
        # Run test and check outputs.
        _check_regime(self, factory, regime)

    def test2(self) -> None:
        """
        Reject invalid family parameters and support default calls.
        """
        # Prepare inputs.
        factory = rnmascen.make_excess_demand_scenario
        invalid = [
            dict(seed=True),
            dict(n_bids=0),
            dict(n_asks=-1),
            dict(quantity_scale=0),
            dict(price_scale=False),
        ]
        invalid += [dict(tier=""), dict(tier=" "), dict(tier=None)]
        # Run test and check outputs.
        _check_family_arguments(self, factory, invalid)


# #############################################################################
# Test_make_excess_supply_scenario
# #############################################################################


class Test_make_excess_supply_scenario(hunitest.TestCase):
    """
    Test varied excess supply books.
    """

    def test1(self) -> None:
        """
        Preserve the regime across seeds, counts, and numeric scales.
        """
        # Prepare inputs.
        factory = rnmascen.make_excess_supply_scenario
        regime = "supply"
        # Run test and check outputs.
        _check_regime(self, factory, regime)

    def test2(self) -> None:
        """
        Reject invalid family parameters and support default calls.
        """
        # Prepare inputs.
        factory = rnmascen.make_excess_supply_scenario
        invalid = [
            dict(seed=True),
            dict(n_bids=0),
            dict(n_asks=-1),
            dict(quantity_scale=0),
            dict(price_scale=False),
        ]
        invalid += [dict(tier=""), dict(tier=" "), dict(tier=None)]
        # Run test and check outputs.
        _check_family_arguments(self, factory, invalid)


# #############################################################################
# Test_make_multi_tier_scenario
# #############################################################################


class Test_make_multi_tier_scenario(hunitest.TestCase):
    """
    Test varied multi tier books.
    """

    def test1(self) -> None:
        """
        Preserve the regime across seeds, counts, and numeric scales.
        """
        # Prepare inputs.
        factory = rnmascen.make_multi_tier_scenario
        regime = "multi_tier"
        # Run test and check outputs.
        _check_regime(self, factory, regime)

    def test2(self) -> None:
        """
        Reject invalid family parameters and support default calls.
        """
        # Prepare inputs.
        factory = rnmascen.make_multi_tier_scenario
        invalid = [
            dict(seed=True),
            dict(n_bids=0),
            dict(n_asks=-1),
            dict(quantity_scale=0),
            dict(price_scale=False),
        ]
        invalid += [
            dict(tiers=("only",)),
            dict(tiers=("x", "x")),
            dict(tiers="ab"),
        ]
        # Run test and check outputs.
        _check_family_arguments(self, factory, invalid)


# #############################################################################
# Test_make_no_cross_scenario
# #############################################################################


class Test_make_no_cross_scenario(hunitest.TestCase):
    """
    Test varied no cross books.
    """

    def test1(self) -> None:
        """
        Preserve the regime across seeds, counts, and numeric scales.
        """
        # Prepare inputs.
        factory = rnmascen.make_no_cross_scenario
        regime = "no_cross"
        # Run test and check outputs.
        _check_regime(self, factory, regime)

    def test2(self) -> None:
        """
        Reject invalid family parameters and support default calls.
        """
        # Prepare inputs.
        factory = rnmascen.make_no_cross_scenario
        invalid = [
            dict(seed=True),
            dict(n_bids=0),
            dict(n_asks=-1),
            dict(quantity_scale=0),
            dict(price_scale=False),
        ]
        invalid += [dict(tier=""), dict(tier=" "), dict(tier=None)]
        # Run test and check outputs.
        _check_family_arguments(self, factory, invalid)


# #############################################################################
# Test_make_random_scenario
# #############################################################################


class Test_make_random_scenario(hunitest.TestCase):
    """
    Check reproducibility, validation, and generated auction invariants.
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

    def test2(self) -> None:
        """
        Verify order counts, custom tiers, and generated field ranges.
        """
        # Prepare inputs and expected outputs.
        seed = -3
        tiers = ("custom",)
        counts = [(0, 0), (0, 4), (4, 0), (20, 15)]
        for n_bids, n_asks in counts:
            with self.subTest(counts=(n_bids, n_asks)):
                # Run test.
                bids, asks = rnmascen.make_random_scenario(
                    seed, n_bids, n_asks, tiers
                )
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

    def test3(self) -> None:
        """
        Reject invalid seed, count, and tier arguments.
        """
        # Prepare inputs and expected outputs.
        cases = [
            (True, 1, 1, ["x"]),
            (1.5, 1, 1, ["x"]),
            (1, -1, 1, ["x"]),
            (1, 1, -1, ["x"]),
            (1, True, 1, ["x"]),
            (1, 1, False, ["x"]),
            (1, 1.5, 1, ["x"]),
            (1, 1, "2", ["x"]),
            (1, 0, 0, []),
            (1, 1, 1, "standard"),
            (1, 1, 1, b"x"),
            (1, 1, 1, {"x"}),
            (1, 1, 1, None),
            (1, 1, 1, [""]),
            (1, 1, 1, [" "]),
            (1, 1, 1, ["x", "x"]),
            (1, 1, 1, [1]),
            (1, 1, 1, [["x"]]),
        ]
        # Run test and check outputs.
        for args in cases:
            with self.subTest(args=args):
                with self.assertRaises(AssertionError):
                    rnmascen.make_random_scenario(*args)

    def test4(self) -> None:
        """
        Conserve submitted volume across 100 generated order books.
        """
        # Prepare inputs and expected outputs.
        tiers = ["standard", "frontier"]
        seeds = range(100)
        for seed in seeds:
            with self.subTest(seed=seed):
                n_bids = seed % 13
                n_asks = seed % 17
                # Run test.
                bids, asks = rnmascen.make_random_scenario(
                    seed, n_bids, n_asks, tiers
                )
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
                        str(
                            sum(b.n_tasks for b in bids if b.c_level_min == tier)
                        ),
                    )
                    self.assert_equal(
                        str(matched + result.unfilled_ask_n_tasks),
                        str(sum(a.n_tasks for a in asks if a.c_level == tier)),
                    )
                    for fill in result.fills:
                        self.assertGreater(fill.n_tasks, 0)
                        self.assert_equal(str(fill.c_level), str(tier))
                        self.assert_equal(
                            str(fill.price), str(result.clearing_price)
                        )
