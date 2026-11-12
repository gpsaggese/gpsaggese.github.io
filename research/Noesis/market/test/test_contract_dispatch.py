"""
Import as:

import research.Noesis.market.test.test_contract_dispatch as rnttcd
"""

import logging
import pprint
import random
from typing import List

import helpers.hunit_test as hunitest
import research.Noesis.market.batch_call_auction as rnbacaau
import research.Noesis.market.contract_dispatch as rnocodis

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_build_contracts
# #############################################################################


class Test_build_contracts(hunitest.TestCase):
    """
    Test `contract_dispatch.build_contracts()`.
    """

    def test1(self) -> None:
        """
        Test one contract built from a single-tier, single-fill round.
        """
        # Prepare inputs.
        bids = [rnbacaau.Bid("buyer_1", 100, "frontier", 2.0, 0.999, 20.0)]
        asks = [rnbacaau.Ask("seller_1", 100, "frontier", 2.5, 0.98, 16.0)]
        order_book = rnbacaau.OrderBook()
        order_book.submit_bid(bids[0])
        order_book.submit_ask(asks[0])
        tier_results = order_book.clear_round()
        # Prepare outputs.
        expected = """
        [Contract(buyer_id='buyer_1',
                  seller_id='seller_1',
                  n_tasks=100,
                  c_level='frontier',
                  l_max=2.0,
                  r_min=0.999,
                  price=18.0,
                  fulfilled=None)]
        """
        # Run test.
        actual = pprint.pformat(rnocodis.build_contracts(bids, tier_results))
        # Check outputs.
        self.assert_equal(actual, expected, dedent=True)

    def test2(self) -> None:
        """
        Test that a bid partially filled across two asks yields one contract
        per fill, both carrying the same bid's `l_max`/`r_min`.
        """
        # Prepare inputs.
        bids = [rnbacaau.Bid("buyer_1", 100, "frontier", 2.0, 0.999, 20.0)]
        asks = [
            rnbacaau.Ask("seller_1", 60, "frontier", 2.5, 0.98, 15.0),
            rnbacaau.Ask("seller_2", 60, "frontier", 2.5, 0.98, 18.0),
        ]
        order_book = rnbacaau.OrderBook()
        order_book.submit_bid(bids[0])
        for ask in asks:
            order_book.submit_ask(ask)
        tier_results = order_book.clear_round()
        # Prepare outputs.
        expected = """
        [Contract(buyer_id='buyer_1',
                  seller_id='seller_1',
                  n_tasks=60,
                  c_level='frontier',
                  l_max=2.0,
                  r_min=0.999,
                  price=19.0,
                  fulfilled=None),
         Contract(buyer_id='buyer_1',
                  seller_id='seller_2',
                  n_tasks=40,
                  c_level='frontier',
                  l_max=2.0,
                  r_min=0.999,
                  price=19.0,
                  fulfilled=None)]
        """
        # Run test.
        actual = pprint.pformat(rnocodis.build_contracts(bids, tier_results))
        # Check outputs.
        self.assert_equal(actual, expected, dedent=True)

    def test3(self) -> None:
        """
        Test that a tier with no crossing bid/ask yields no contracts.
        """
        # Prepare inputs.
        bids = [rnbacaau.Bid("buyer_1", 100, "frontier", 2.0, 0.99, 10.0)]
        asks = [rnbacaau.Ask("seller_1", 80, "frontier", 2.5, 0.98, 12.0)]
        order_book = rnbacaau.OrderBook()
        order_book.submit_bid(bids[0])
        order_book.submit_ask(asks[0])
        tier_results = order_book.clear_round()
        # Prepare outputs.
        expected: List[rnocodis.Contract] = []
        # Run test.
        actual = rnocodis.build_contracts(bids, tier_results)
        # Check outputs.
        self.assert_equal(str(actual), str(expected))

    def test4(self) -> None:
        """
        Test that a two-tier round yields one contract per tier, each
        carrying that tier's own bid `l_max`/`r_min` and clearing price.
        """
        # Prepare inputs.
        bids = [
            rnbacaau.Bid("buyer_c", 50, "cheap", 5.0, 0.9, 5.0),
            rnbacaau.Bid("buyer_f", 100, "frontier", 2.0, 0.99, 20.0),
        ]
        asks = [
            rnbacaau.Ask("seller_c", 50, "cheap", 4.0, 0.92, 3.0),
            rnbacaau.Ask("seller_f", 100, "frontier", 2.5, 0.98, 16.0),
        ]
        order_book = rnbacaau.OrderBook()
        for bid in bids:
            order_book.submit_bid(bid)
        for ask in asks:
            order_book.submit_ask(ask)
        tier_results = order_book.clear_round()
        # Prepare outputs.
        expected = """
        [Contract(buyer_id='buyer_c',
                  seller_id='seller_c',
                  n_tasks=50,
                  c_level='cheap',
                  l_max=5.0,
                  r_min=0.9,
                  price=4.0,
                  fulfilled=None),
         Contract(buyer_id='buyer_f',
                  seller_id='seller_f',
                  n_tasks=100,
                  c_level='frontier',
                  l_max=2.0,
                  r_min=0.99,
                  price=18.0,
                  fulfilled=None)]
        """
        # Run test.
        actual = pprint.pformat(rnocodis.build_contracts(bids, tier_results))
        # Check outputs.
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_mock_fulfill
# #############################################################################


class Test_mock_fulfill(hunitest.TestCase):
    """
    Test `contract_dispatch.mock_fulfill()`.
    """

    def test1(self) -> None:
        """
        Test that a `success_rate` of 1.0 always reports success.
        """
        # Prepare inputs.
        contract = rnocodis.Contract(
            "buyer_1", "seller_1", 100, "frontier", 2.0, 0.999, 18.0
        )
        # Run test.
        actual = rnocodis.mock_fulfill(
            contract, success_rate=1.0, rng=random.Random(0)
        )
        # Check outputs.
        self.assertTrue(actual)

    def test2(self) -> None:
        """
        Test that a `success_rate` of 0.0 always reports failure.
        """
        # Prepare inputs.
        contract = rnocodis.Contract(
            "buyer_1", "seller_1", 100, "frontier", 2.0, 0.999, 18.0
        )
        # Run test.
        actual = rnocodis.mock_fulfill(
            contract, success_rate=0.0, rng=random.Random(0)
        )
        # Check outputs.
        self.assertFalse(actual)

    def test3(self) -> None:
        """
        Test that a fixed seed makes the outcome reproducible.
        """
        # Prepare inputs.
        contract = rnocodis.Contract(
            "buyer_1", "seller_1", 100, "frontier", 2.0, 0.999, 18.0
        )
        # Run test.
        actual1 = rnocodis.mock_fulfill(
            contract, success_rate=0.5, rng=random.Random(42)
        )
        actual2 = rnocodis.mock_fulfill(
            contract, success_rate=0.5, rng=random.Random(42)
        )
        # Check outputs.
        self.assert_equal(str(actual1), str(actual2))

    def test4(self) -> None:
        """
        Test that a `success_rate` outside `[0, 1]` raises `AssertionError`.
        """
        # Prepare inputs.
        contract = rnocodis.Contract(
            "buyer_1", "seller_1", 100, "frontier", 2.0, 0.999, 18.0
        )
        success_rate = 1.5
        # Run test and check output.
        with self.assertRaises(AssertionError):
            rnocodis.mock_fulfill(
                contract, success_rate=success_rate, rng=random.Random(0)
            )


# #############################################################################
# Test_dispatch_contract
# #############################################################################


class Test_dispatch_contract(hunitest.TestCase):
    """
    Test `contract_dispatch.dispatch_contract()`.
    """

    def test1(self) -> None:
        """
        Test that dispatching with a fixed always-succeed callable logs
        `fulfilled=True` onto the contract.
        """
        # Prepare inputs.
        contract = rnocodis.Contract(
            "buyer_1", "seller_1", 100, "frontier", 2.0, 0.999, 18.0
        )
        # Run test.
        actual = rnocodis.dispatch_contract(
            contract, fulfillment_func=lambda _contract: True
        )
        # Check outputs.
        self.assertTrue(actual.fulfilled)
        # The same object is mutated and returned.
        self.assertIs(actual, contract)

    def test2(self) -> None:
        """
        Test that dispatching with a fixed always-fail callable logs
        `fulfilled=False` onto the contract.
        """
        # Prepare inputs.
        contract = rnocodis.Contract(
            "buyer_1", "seller_1", 100, "frontier", 2.0, 0.999, 18.0
        )
        # Run test.
        actual = rnocodis.dispatch_contract(
            contract, fulfillment_func=lambda _contract: False
        )
        # Check outputs.
        self.assertFalse(actual.fulfilled)


# #############################################################################
# Test_dispatch_contracts
# #############################################################################


class Test_dispatch_contracts(hunitest.TestCase):
    """
    Test `contract_dispatch.dispatch_contracts()`.
    """

    def test1(self) -> None:
        """
        Test the full match -> contract -> dispatch -> logged outcome loop
        for a two-contract round with a fixed fulfillment outcome.
        """
        # Prepare inputs.
        bids = [
            rnbacaau.Bid("buyer_c", 50, "cheap", 5.0, 0.9, 5.0),
            rnbacaau.Bid("buyer_f", 100, "frontier", 2.0, 0.99, 20.0),
        ]
        asks = [
            rnbacaau.Ask("seller_c", 50, "cheap", 4.0, 0.92, 3.0),
            rnbacaau.Ask("seller_f", 100, "frontier", 2.5, 0.98, 16.0),
        ]
        order_book = rnbacaau.OrderBook()
        for bid in bids:
            order_book.submit_bid(bid)
        for ask in asks:
            order_book.submit_ask(ask)
        tier_results = order_book.clear_round()
        contracts = rnocodis.build_contracts(bids, tier_results)
        # Prepare outputs.
        expected = """
        [Contract(buyer_id='buyer_c',
                  seller_id='seller_c',
                  n_tasks=50,
                  c_level='cheap',
                  l_max=5.0,
                  r_min=0.9,
                  price=4.0,
                  fulfilled=True),
         Contract(buyer_id='buyer_f',
                  seller_id='seller_f',
                  n_tasks=100,
                  c_level='frontier',
                  l_max=2.0,
                  r_min=0.99,
                  price=18.0,
                  fulfilled=True)]
        """
        # Run test.
        actual = rnocodis.dispatch_contracts(
            contracts, fulfillment_func=lambda _contract: True
        )
        # Check outputs.
        self.assert_equal(pprint.pformat(actual), expected, dedent=True)

    def test2(self) -> None:
        """
        Test that dispatching an empty contract list returns an empty list.
        """
        # Prepare inputs.
        contracts: List[rnocodis.Contract] = []
        # Prepare outputs.
        expected: List[rnocodis.Contract] = []
        # Run test.
        actual = rnocodis.dispatch_contracts(
            contracts, fulfillment_func=lambda _contract: True
        )
        # Check outputs.
        self.assert_equal(str(actual), str(expected))
