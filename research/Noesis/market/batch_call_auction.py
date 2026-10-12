"""
In-memory batch call-auction order book for the Noesis Market simulator.

Import as:

import research.Noesis.market.batch_call_auction as rnbacaau
"""

import abc
import dataclasses
import logging
from typing import Dict, List, Optional, Tuple

import helpers.hdbg as hdbg
import helpers.hprint as hprint

_LOG = logging.getLogger(__name__)


# #############################################################################
# Constants
# #############################################################################


# Default batch cadence from the plan (not enforced by this module, which
# clears one round per `OrderBook.clear_round()` call and leaves scheduling
# to the caller).
DEFAULT_BATCH_INTERVAL_MINUTES = 5


# #############################################################################
# Bid
# #############################################################################


# TODO(ai_gp): There should also info about the submitter in terms of an id
# and a submission timestamp, same thing for Ask
@dataclasses.dataclass
class Bid:
    """
    Store one buyer order for the batch auction.

    E.g., `Bid("buyer_1", 10_000, "frontier", 2.0, 0.999, 0.02)` requests
    - 10,000 tasks at the "frontier" tier
    - under 2.0 latency
    - at least 0.999 reliability
    - for up to 0.02 price per task.
    """

    buyer_id: str
    n_tasks: int
    c_level_min: str
    l_max: float
    r_min: float
    p_max: float

    def __init__(
        self,
        buyer_id: str,
        n_tasks: int,
        c_level_min: str,
        l_max: float,
        r_min: float,
        p_max: float,
    ) -> None:
        hdbg.dassert_ne(buyer_id, "", "Bid needs a non-empty buyer_id")
        self.buyer_id = buyer_id
        #
        hdbg.dassert_lt(0, n_tasks, "Bid.n_tasks must be positive: %s", n_tasks)
        self.n_tasks = n_tasks
        #
        hdbg.dassert_ne(
            c_level_min, "", "Bid needs a non-empty c_level_min tier"
        )
        self.c_level_min = c_level_min
        #
        hdbg.dassert_lt(0, l_max, "Bid.l_max must be positive: %s", l_max)
        self.l_max = l_max
        #
        hdbg.dassert_lte(0.0, r_min, "Bid.r_min must be in [0, 1]: %s", r_min)
        hdbg.dassert_lte(r_min, 1.0, "Bid.r_min must be in [0, 1]: %s", r_min)
        self.r_min = r_min
        #
        hdbg.dassert_lte(0.0, p_max, "Bid.p_max must be non-negative: %s", p_max)
        self.p_max = p_max


# #############################################################################
# Ask
# #############################################################################


@dataclasses.dataclass
class Ask:
    """
    Store one seller order for the batch auction.

    E.g., `Ask("seller_1", 10_000, "frontier", 1.5, 0.995, 0.015)` offers
    - 10,000 tasks at the "frontier" tier
    - with typical 1.5 latency
    - 0.995 reliability
    - for at least 0.015 price per task.
    """

    # Unique identifier of the seller submitting this order.
    seller_id: str
    # Number of tasks offered.
    n_tasks: int
    # Capability tier bucket this ask belongs to (matched exactly against a
    # bid's `c_level_min`).
    c_level: str
    # Typical latency the seller expects to deliver at.
    l_typical: float
    # Typical reliability the seller expects to deliver at, in [0, 1].
    r_typical: float
    # Minimum price per task the seller will accept.
    p_min: float

    def __init__(
        self,
        seller_id: str,
        n_tasks: int,
        c_level: str,
        l_typical: float,
        r_typical: float,
        p_min: float,
    ) -> None:
        hdbg.dassert_ne(seller_id, "", "Ask needs a non-empty seller_id")
        self.seller_id = seller_id
        #
        hdbg.dassert_lt(0, n_tasks, "Ask.n_tasks must be positive: %s", n_tasks)
        self.n_tasks = n_tasks
        #
        hdbg.dassert_ne(c_level, "", "Ask needs a non-empty c_level tier")
        self.c_level = c_level
        #
        hdbg.dassert_lt(
            0, l_typical, "Ask.l_typical must be positive: %s", l_typical
        )
        self.l_typical = l_typical
        #
        hdbg.dassert_lte(
            0.0, r_typical, "Ask.r_typical must be in [0, 1]: %s", r_typical
        )
        hdbg.dassert_lte(
            r_typical, 1.0, "Ask.r_typical must be in [0, 1]: %s", r_typical
        )
        self.r_typical = r_typical
        #
        hdbg.dassert_lte(0.0, p_min, "Ask.p_min must be non-negative: %s", p_min)
        self.p_min = p_min


# #############################################################################
# Fill
# #############################################################################


@dataclasses.dataclass
class Fill:
    """
    Record one matched (buyer, seller) trade produced by clearing a tier.

    # TODO(ai_gp): Add example of match inputs and outputs

    All fills within a tier's clearing round share the same uniform
    `price`.
    """

    # Identifier of the buyer whose bid was matched.
    buyer_id: str
    # Identifier of the seller whose ask was matched.
    seller_id: str
    # Capability tier the fill belongs to.
    c_level: str
    # Number of tasks matched in this fill.
    n_tasks: int
    # Uniform clearing price applied to this fill.
    price: float


# #############################################################################
# TierClearResult
# #############################################################################


@dataclasses.dataclass
class TierClearResult:
    """
    Store the outcome of clearing one capability tier for one batch round.

    # TODO(ai_gp): Add example of match inputs and outputs

    `clearing_price` is `None` when the tier's best bid never crossed its
    best ask, e.g. `TierClearResult("frontier", None, [], 500, 300)` means
    no trade happened and both sides' full quantity went unfilled.
    """

    # Capability tier this result belongs to.
    c_level: str
    # Uniform clearing price for the tier, or `None` if the best bid never
    # crossed the best ask.
    clearing_price: Optional[float]
    # Fills produced by clearing this tier.
    fills: List[Fill]
    # Total bid quantity left unmatched after clearing.
    unfilled_bid_n_tasks: int
    # Total ask quantity left unmatched after clearing.
    unfilled_ask_n_tasks: int


# #############################################################################
# Matching engine
# #############################################################################


# TODO(ai_gp): Make it static if it belongs to OrderBook
def _match_orders_in_tier(
    c_level: str, bids: List[Bid], asks: List[Ask]
) -> TierClearResult:
    """
    Clear one capability tier's bids against its asks.

    Sort bids highest-price-first and asks lowest-price-first (a standard
    call auction), then walk both queues matching quantity while the best
    remaining bid price is still at or above the best remaining ask price.
    The uniform clearing price is the midpoint of the two marginal (last
    matched) orders, applied to every fill in the tier.

    :param c_level: capability tier bucket that `bids` and `asks` belong to
    :param bids: buy orders for this tier, in submission order
    :param asks: sell orders for this tier, in submission order
    :return: `TierClearResult` with fills, clearing price (`None` if the
        tier never crossed), and unfilled quantity on each side
    """
    _LOG.debug(hprint.to_str("c_level bids asks"))
    # `sorted()` is stable, so ties fall back to submission order.
    sorted_bids = sorted(bids, key=lambda bid: bid.p_max, reverse=True)
    sorted_asks = sorted(asks, key=lambda ask: ask.p_min)
    # Walk both queues, matching the in-progress bid against the in-progress
    # ask until either side runs out or the price no longer crosses.
    raw_matches: List[Tuple[str, str, int]] = []
    bid_idx = 0
    ask_idx = 0
    remaining_bid_n_tasks = sorted_bids[0].n_tasks if sorted_bids else 0
    remaining_ask_n_tasks = sorted_asks[0].n_tasks if sorted_asks else 0
    # Prices of the last matched (marginal) orders, used for the uniform
    # clearing price once matching stops.
    marginal_bid_price = 0.0
    marginal_ask_price = 0.0
    while (
        bid_idx < len(sorted_bids)
        and ask_idx < len(sorted_asks)
        and sorted_bids[bid_idx].p_max >= sorted_asks[ask_idx].p_min
    ):
        current_bid = sorted_bids[bid_idx]
        current_ask = sorted_asks[ask_idx]
        marginal_bid_price = current_bid.p_max
        marginal_ask_price = current_ask.p_min
        matched_n_tasks = min(remaining_bid_n_tasks, remaining_ask_n_tasks)
        raw_matches.append(
            (current_bid.buyer_id, current_ask.seller_id, matched_n_tasks)
        )
        remaining_bid_n_tasks -= matched_n_tasks
        remaining_ask_n_tasks -= matched_n_tasks
        # Advance to the next order on whichever side was just exhausted.
        if remaining_bid_n_tasks == 0:
            bid_idx += 1
            if bid_idx < len(sorted_bids):
                remaining_bid_n_tasks = sorted_bids[bid_idx].n_tasks
        if remaining_ask_n_tasks == 0:
            ask_idx += 1
            if ask_idx < len(sorted_asks):
                remaining_ask_n_tasks = sorted_asks[ask_idx].n_tasks
    if raw_matches:
        clearing_price = (marginal_bid_price + marginal_ask_price) / 2.0
        fills = [
            Fill(buyer_id, seller_id, c_level, matched_n_tasks, clearing_price)
            for buyer_id, seller_id, matched_n_tasks in raw_matches
        ]
    else:
        clearing_price = None
        fills = []
    # Unfilled = whatever is left of the in-progress order plus every order
    # behind it that was never touched this round.
    unfilled_bid_n_tasks = remaining_bid_n_tasks + sum(
        bid.n_tasks for bid in sorted_bids[bid_idx + 1 :]
    )
    unfilled_ask_n_tasks = remaining_ask_n_tasks + sum(
        ask.n_tasks for ask in sorted_asks[ask_idx + 1 :]
    )
    result = TierClearResult(
        c_level,
        clearing_price,
        fills,
        unfilled_bid_n_tasks,
        unfilled_ask_n_tasks,
    )
    _LOG.debug("return=%s", result)
    return result


# #############################################################################
# OrderBookStore
# #############################################################################


class OrderBookStore(abc.ABC):
    """
    Pluggable storage backend for `OrderBook`'s pending `Bid`/`Ask` queues.
    """

    @abc.abstractmethod
    def add_bid(self, bid: Bid) -> None: ...

    @abc.abstractmethod
    def add_ask(self, ask: Ask) -> None: ...

    @abc.abstractmethod
    def get_bids(self) -> List[Bid]:
        """
        :return: pending bids, in submission order
        """
        ...

    @abc.abstractmethod
    def get_asks(self) -> List[Ask]:
        """
        :return: pending asks, in submission order
        """
        ...

    @abc.abstractmethod
    def clear(self) -> None:
        """
        Drop every stored bid/ask.
        """
        ...


# #############################################################################
# _InMemoryOrderBookStore
# #############################################################################


class _InMemoryOrderBookStore(OrderBookStore):
    """
    In-memory implementation of `OrderBookStore` using `List[Bid]`/`List[Ask]`.
    """

    def __init__(self) -> None:
        self._bids: List[Bid] = []
        self._asks: List[Ask] = []

    def add_bid(self, bid: Bid) -> None:
        self._bids.append(bid)

    def add_ask(self, ask: Ask) -> None:
        self._asks.append(ask)

    def get_bids(self) -> List[Bid]:
        return list(self._bids)

    def get_asks(self) -> List[Ask]:
        return list(self._asks)

    def clear(self) -> None:
        self._bids = []
        self._asks = []


# #############################################################################
# OrderBook
# #############################################################################


class OrderBook:
    """
    Batch call-auction order book for one Noesis Market round.

    - Buyers submit `Bid`s and sellers submit `Ask`s via `submit_bid()` /
      `submit_ask()`
    - `clear_round()` buckets every pending order by capability tier and clears
      each tier independently
    - Pending orders live in a pluggable `OrderBookStore`.

    Simplifications:
    - A bid's `c_level_min` is treated as an exact tier bucket, matched only
      against asks with the same `c_level` string; there is no cross-tier
      substitution.
    - A cleared round drops every order it processed, matched or not
    """

    def __init__(self, *, store: Optional[OrderBookStore] = None) -> None:
        """
        :param store: pluggable storage backend for pending bids/asks
        """
        # TODO(ai_gp): Add a dassert_isinstance(OrderBookStore)
        # TODO(ai_gp): Force it to specify it.
        if store is None:
            store = _InMemoryOrderBookStore()
        self._store = store

    def submit_bid(self, bid: Bid) -> None:
        """
        Queue a buyer order for the next `clear_round()`.

        :param bid: buy order to add to the book
        """
        _LOG.debug(hprint.to_str("bid"))
        self._store.add_bid(bid)

    def submit_ask(self, ask: Ask) -> None:
        """
        Queue a seller order for the next `clear_round()`.

        :param ask: sell order to add to the book
        """
        _LOG.debug(hprint.to_str("ask"))
        self._store.add_ask(ask)

    def get_pending_bids(self) -> List[Bid]:
        """
        :return: buy orders queued for the next `clear_round()`
        """
        return self._store.get_bids()

    def get_pending_asks(self) -> List[Ask]:
        """
        :return: sell orders queued for the next `clear_round()`
        """
        return self._store.get_asks()

    # TODO(gp): is there a better word for clear? Match?
    def clear_round(self) -> Dict[str, TierClearResult]:
        """
        Clear every capability tier present in the book and empty it.

        :return: map from `c_level` to that tier's `TierClearResult`
        """
        # Fetch pending orders once (not once per tier below) to avoid repeated
        # round trips to a remote `OrderBookStore`.
        bids = self._store.get_bids()
        asks = self._store.get_asks()
        _LOG.debug("num_bids=%d num_asks=%d", len(bids), len(asks))
        # Bucket by tier.
        c_levels = sorted(
            {bid.c_level_min for bid in bids} | {ask.c_level for ask in asks}
        )
        results: Dict[str, TierClearResult] = {}
        for c_level in c_levels:
            tier_bids = [bid for bid in bids if bid.c_level_min == c_level]
            tier_asks = [ask for ask in asks if ask.c_level == c_level]
            results[c_level] = _match_orders_in_tier(
                c_level, tier_bids, tier_asks
            )
        # A batch round clears the whole book, per the class docstring.
        self._store.clear()
        # TODO(gp): What happens to the clients that have submit a bid and are
        # not matched? Do they get a notification?
        _LOG.debug("return=%s", results)
        return results
