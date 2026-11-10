"""
Import as:

import research.Noesis.market.test.test_batch_call_auction as rnttbca
"""

import logging
import pprint
from typing import List

import helpers.hunit_test as hunitest
import research.Noesis.market.batch_call_auction as rnbacaau

_LOG = logging.getLogger(__name__)


# #############################################################################
# TestOrderBook
# #############################################################################


class TestOrderBook(hunitest.TestCase):
    """
    Test `batch_call_auction.OrderBook.clear_round()`.
    """

    def helper(
        self,
        bids: List[rnbacaau.Bid],
        asks: List[rnbacaau.Ask],
        expected: str,
    ) -> None:
        """
        Submit `bids`/`asks` to a fresh `OrderBook`, clear one round, and
        compare the pretty-printed result to `expected`.

        :param bids: buy orders to submit before clearing
        :param asks: sell orders to submit before clearing
        :param expected: expected pretty-printed `clear_round()` result
        """
        # Prepare inputs.
        order_book = rnbacaau.OrderBook()
        for bid in bids:
            order_book.submit_bid(bid)
        for ask in asks:
            order_book.submit_ask(ask)
        # Run test.
        results = order_book.clear_round()
        # Check outputs.
        actual = pprint.pformat(results)
        self.assert_equal(actual, expected, dedent=True)

    def test1(self) -> None:
        """
        Test a single tier where one bid exactly fills one ask.
        """
        # Prepare inputs.
        bids = [rnbacaau.Bid("buyer_1", 100, "frontier", 2.0, 0.99, 20.0)]
        asks = [rnbacaau.Ask("seller_1", 100, "frontier", 2.5, 0.98, 16.0)]
        # Prepare outputs.
        expected = """
        {'frontier': TierClearResult(c_level='frontier',
                                     clearing_price=18.0,
                                     fills=[Fill(buyer_id='buyer_1',
                                                 seller_id='seller_1',
                                                 c_level='frontier',
                                                 n_tasks=100,
                                                 price=18.0)],
                                     unfilled_bid_n_tasks=0,
                                     unfilled_ask_n_tasks=0)}
        """
        # Run test.
        self.helper(bids, asks, expected)

    def test2(self) -> None:
        """
        Test two tiers clearing independently at different prices.
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
        # Prepare outputs.
        expected = """
        {'cheap': TierClearResult(c_level='cheap',
                                  clearing_price=4.0,
                                  fills=[Fill(buyer_id='buyer_c',
                                              seller_id='seller_c',
                                              c_level='cheap',
                                              n_tasks=50,
                                              price=4.0)],
                                  unfilled_bid_n_tasks=0,
                                  unfilled_ask_n_tasks=0),
         'frontier': TierClearResult(c_level='frontier',
                                     clearing_price=18.0,
                                     fills=[Fill(buyer_id='buyer_f',
                                                 seller_id='seller_f',
                                                 c_level='frontier',
                                                 n_tasks=100,
                                                 price=18.0)],
                                     unfilled_bid_n_tasks=0,
                                     unfilled_ask_n_tasks=0)}
        """
        # Run test.
        self.helper(bids, asks, expected)

    def test3(self) -> None:
        """
        Test a tier where the best bid never crosses the best ask.
        """
        # Prepare inputs.
        bids = [rnbacaau.Bid("buyer_1", 100, "frontier", 2.0, 0.99, 10.0)]
        asks = [rnbacaau.Ask("seller_1", 80, "frontier", 2.5, 0.98, 12.0)]
        # Prepare outputs.
        expected = """
        {'frontier': TierClearResult(c_level='frontier',
                                     clearing_price=None,
                                     fills=[],
                                     unfilled_bid_n_tasks=100,
                                     unfilled_ask_n_tasks=80)}
        """
        # Run test.
        self.helper(bids, asks, expected)

    def test4(self) -> None:
        """
        Test a bid partially filled across two asks, leaving one ask with
        unfilled quantity.
        """
        # Prepare inputs.
        bids = [rnbacaau.Bid("buyer_1", 100, "frontier", 2.0, 0.99, 20.0)]
        asks = [
            rnbacaau.Ask("seller_1", 60, "frontier", 2.5, 0.98, 15.0),
            rnbacaau.Ask("seller_2", 60, "frontier", 2.5, 0.98, 18.0),
        ]
        # Prepare outputs.
        expected = """
        {'frontier': TierClearResult(c_level='frontier',
                                     clearing_price=19.0,
                                     fills=[Fill(buyer_id='buyer_1',
                                                 seller_id='seller_1',
                                                 c_level='frontier',
                                                 n_tasks=60,
                                                 price=19.0),
                                            Fill(buyer_id='buyer_1',
                                                 seller_id='seller_2',
                                                 c_level='frontier',
                                                 n_tasks=40,
                                                 price=19.0)],
                                     unfilled_bid_n_tasks=0,
                                     unfilled_ask_n_tasks=20)}
        """
        # Run test.
        self.helper(bids, asks, expected)

    def test5(self) -> None:
        """
        Test that `clear_round()` empties the book, matched or not.
        """
        # Prepare inputs.
        bid = rnbacaau.Bid("buyer_1", 100, "frontier", 2.0, 0.99, 20.0)
        ask = rnbacaau.Ask("seller_1", 100, "frontier", 2.5, 0.98, 16.0)
        order_book = rnbacaau.OrderBook()
        order_book.submit_bid(bid)
        order_book.submit_ask(ask)
        # Prepare outputs.
        expected_pending_bids: List[rnbacaau.Bid] = []
        expected_pending_asks: List[rnbacaau.Ask] = []
        # Run test.
        order_book.clear_round()
        # Check outputs.
        actual_pending_bids = order_book.get_pending_bids()
        actual_pending_asks = order_book.get_pending_asks()
        self.assert_equal(str(actual_pending_bids), str(expected_pending_bids))
        self.assert_equal(str(actual_pending_asks), str(expected_pending_asks))

    def test6(self) -> None:
        """
        Test that clearing an empty book returns no tiers.
        """
        # Prepare inputs.
        bids: List[rnbacaau.Bid] = []
        asks: List[rnbacaau.Ask] = []
        # Prepare outputs.
        expected = "{}"
        # Run test.
        self.helper(bids, asks, expected)

    def test7(self) -> None:
        """
        Test a tier with a bid but no ask: nothing crosses, and the ask
        side has no unfilled quantity because it has no orders at all.
        """
        # Prepare inputs.
        bids = [rnbacaau.Bid("buyer_1", 100, "frontier", 2.0, 0.99, 20.0)]
        asks: List[rnbacaau.Ask] = []
        # Prepare outputs.
        expected = """
        {'frontier': TierClearResult(c_level='frontier',
                                     clearing_price=None,
                                     fills=[],
                                     unfilled_bid_n_tasks=100,
                                     unfilled_ask_n_tasks=0)}
        """
        # Run test.
        self.helper(bids, asks, expected)

    def test8(self) -> None:
        """
        Test that a bid and an ask in different tiers never cross, even
        though their prices alone would cross.
        """
        # Prepare inputs.
        bids = [rnbacaau.Bid("buyer_1", 100, "frontier", 2.0, 0.99, 100.0)]
        asks = [rnbacaau.Ask("seller_1", 80, "cheap", 2.5, 0.98, 1.0)]
        # Prepare outputs.
        expected = """
        {'cheap': TierClearResult(c_level='cheap',
                                  clearing_price=None,
                                  fills=[],
                                  unfilled_bid_n_tasks=0,
                                  unfilled_ask_n_tasks=80),
         'frontier': TierClearResult(c_level='frontier',
                                     clearing_price=None,
                                     fills=[],
                                     unfilled_bid_n_tasks=100,
                                     unfilled_ask_n_tasks=0)}
        """
        # Run test.
        self.helper(bids, asks, expected)

    def test9(self) -> None:
        """
        Test that two bids tied at the same price are matched in
        submission order: the first bid is filled, the second is not.
        """
        # Prepare inputs.
        bids = [
            rnbacaau.Bid("buyer_1", 50, "frontier", 2.0, 0.99, 20.0),
            rnbacaau.Bid("buyer_2", 50, "frontier", 2.0, 0.99, 20.0),
        ]
        asks = [rnbacaau.Ask("seller_1", 50, "frontier", 2.5, 0.98, 15.0)]
        # Prepare outputs.
        expected = """
        {'frontier': TierClearResult(c_level='frontier',
                                     clearing_price=17.5,
                                     fills=[Fill(buyer_id='buyer_1',
                                                 seller_id='seller_1',
                                                 c_level='frontier',
                                                 n_tasks=50,
                                                 price=17.5)],
                                     unfilled_bid_n_tasks=50,
                                     unfilled_ask_n_tasks=0)}
        """
        # Run test.
        self.helper(bids, asks, expected)


# #############################################################################
# TestBid
# #############################################################################


class TestBid(hunitest.TestCase):
    """
    Test `batch_call_auction.Bid` field validation.
    """

    def test1(self) -> None:
        """
        Test that a non-positive `n_tasks` raises `AssertionError`.
        """
        # Prepare inputs.
        n_tasks = 0
        # Run test and check output.
        with self.assertRaises(AssertionError):
            rnbacaau.Bid("buyer_1", n_tasks, "frontier", 2.0, 0.99, 20.0)


# #############################################################################
# TestAsk
# #############################################################################


class TestAsk(hunitest.TestCase):
    """
    Test `batch_call_auction.Ask` field validation.
    """

    def test1(self) -> None:
        """
        Test that an `r_typical` outside `[0, 1]` raises `AssertionError`.
        """
        # Prepare inputs.
        r_typical = 1.5
        # Run test and check output.
        with self.assertRaises(AssertionError):
            rnbacaau.Ask("seller_1", 100, "frontier", 2.5, r_typical, 15.0)
