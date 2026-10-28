"""
Run and store complete market rounds without an HTTP dependency.

Import as:

import research.Noesis.market.round_runner as rnmaroru
"""

import abc
import dataclasses
import logging
from typing import Callable, Dict, List, Optional, Tuple, TypeVar

import helpers.hdbg as hdbg
import helpers.hprint as hprint
import research.Noesis.market.batch_call_auction as rnbacaau
import research.Noesis.market.contract_dispatch as rnocodis

_LOG = logging.getLogger(__name__)


# #############################################################################
# Round results
# #############################################################################


@dataclasses.dataclass
class RoundSummary:
    """
    Record one tier's published price and matched volume for a shared round id.
    """

    tier: str
    round_id: int
    # No trade means no price; volume counts tasks, not orders or contracts.
    clearing_price: Optional[float]
    matched_volume: int


@dataclasses.dataclass
class RoundResult:
    """
    Return a complete round, including copies of orders with unfilled quantities.

    `contracts` follows tier-then-fill order. Unmatched orders follow stable price
    priority on each side; fully filled orders are omitted. The book is emptied,
    so these residual orders are reports, not automatically requeued orders.
    """

    round_id: int
    tier_results: Dict[str, RoundSummary]
    contracts: List[rnocodis.Contract]
    unmatched_bids: List[rnbacaau.Bid]
    unmatched_asks: List[rnbacaau.Ask]


# #############################################################################
# ContractStore
# #############################################################################


class ContractStore(abc.ABC):
    """
    Storage interface for market contracts, round ids, and per-tier summaries.
    """

    @abc.abstractmethod
    def save_contract(self, contract: rnocodis.Contract) -> int:
        """
        Save a dispatched contract and assign its lookup id.

        :param contract: contract with a recorded fulfillment outcome
        :return: the `contract_id` assigned to `contract`
        """
        ...

    @abc.abstractmethod
    def get_contract(self, contract_id: int) -> rnocodis.Contract:
        """
        Read a saved contract or raise `AssertionError` for an unknown id.

        :param contract_id: id returned by `save_contract()`
        :return: the stored contract, including its fulfillment outcome
        """
        ...

    @abc.abstractmethod
    def next_round_id(self) -> int:
        """
        Assign one new `round_id`, shared by every tier cleared in the same
        `run_round()` call, including an empty round.
        """
        ...

    @abc.abstractmethod
    def save_round(self, round_response: RoundSummary) -> None:
        """
        Publish one tier's plain round summary.

        :param round_response: summary associated with a shared round id
        """
        ...

    @abc.abstractmethod
    def get_latest_round(self, tier: str) -> RoundSummary:
        """
        Read a tier's latest summary or raise `AssertionError` if none exists.

        :param tier: capability tier to look up
        :return: the last summary saved for this tier
        """
        ...


# #############################################################################
# InMemoryContractStore
# #############################################################################


class InMemoryContractStore(ContractStore):
    """
    Keep contract ids and latest tier summaries in memory for one store lifetime.
    """

    def __init__(self) -> None:
        # Store-local counters preserve the existing API's zero-based ids.
        self._contracts_by_id: Dict[int, rnocodis.Contract] = {}
        self._next_contract_id = 0
        self._latest_round_by_tier: Dict[str, RoundSummary] = {}
        self._next_round_id = 0

    def save_contract(self, contract: rnocodis.Contract) -> int:
        _LOG.debug("buyer_id='%s'", contract.buyer_id)
        contract_id = self._next_contract_id
        self._contracts_by_id[contract_id] = contract
        self._next_contract_id += 1
        _LOG.debug("return='%s'", contract_id)
        return contract_id

    def get_contract(self, contract_id: int) -> rnocodis.Contract:
        _LOG.debug(hprint.to_str("contract_id"))
        hdbg.dassert_in(
            contract_id,
            self._contracts_by_id,
            "Unknown contract_id '%s'",
            contract_id,
        )
        contract = self._contracts_by_id[contract_id]
        _LOG.debug("return='%s'", contract)
        return contract

    def next_round_id(self) -> int:
        round_id = self._next_round_id
        self._next_round_id += 1
        _LOG.debug("return='%s'", round_id)
        return round_id

    def save_round(self, round_response: RoundSummary) -> None:
        _LOG.debug(hprint.to_str("round_response"))
        self._latest_round_by_tier[round_response.tier] = round_response

    def get_latest_round(self, tier: str) -> RoundSummary:
        _LOG.debug(hprint.to_str("tier"))
        hdbg.dassert_in(
            tier,
            self._latest_round_by_tier,
            "No cleared round yet for tier '%s'",
            tier,
        )
        summary = self._latest_round_by_tier[tier]
        _LOG.debug("return='%s'", summary)
        return summary


# #############################################################################
# Round execution
# #############################################################################


# Apply the same quantity accounting to bids and asks without changing their type.
_Order = TypeVar("_Order", rnbacaau.Bid, rnbacaau.Ask)


def _get_unmatched_orders(
    orders: List[_Order],
    matched: Dict[Tuple[str, str], int],
    order_key: Callable[[_Order], Tuple[str, str]],
) -> List[_Order]:
    """
    Subtract filled quantities from orders already sorted in auction priority.

    :param orders: submitted orders in stable price priority
    :param matched: filled quantity keyed by tier and participant id
    :param order_key: extract the matching tier and participant identifier
    :return: independent residual order copies with positive quantities
    """
    _LOG.debug("num_orders='%s'", len(orders))
    # Fills identify participants, not individual orders. Consume their totals in
    # the matcher's stable priority order, including repeated participant ids.
    remaining_matches = dict(matched)
    unmatched = []
    for order in orders:
        key = order_key(order)
        filled = min(order.n_tasks, remaining_matches.get(key, 0))
        remaining_matches[key] = remaining_matches.get(key, 0) - filled
        remaining = order.n_tasks - filled
        if remaining:
            # Preserve the original terms while reporting only the unfilled part.
            unmatched.append(dataclasses.replace(order, n_tasks=remaining))
    _LOG.debug("num_unmatched='%s'", len(unmatched))
    return unmatched


def run_round(
    order_book: rnbacaau.OrderBook,
    contract_store: ContractStore,
    fulfillment_func: rnocodis.FulfillmentFunc,
) -> RoundResult:
    """
    Clear all tiers, dispatch contracts, and persist contracts and tier summaries.

    An empty round consumes one round id and leaves previous tier summaries
    unchanged. Exceptions propagate, as in the original API orchestration; the
    clear, dispatch, and individual saves are not an atomic transaction. Callers
    must serialize round execution and submissions; this runner adds no locking.

    :param order_book: pending orders to clear and empty
    :param contract_store: backend assigning ids and storing round output
    :param fulfillment_func: explicit callable supplying each contract's outcome
    :return: plain round data with one id shared by all cleared tiers
    """
    # Capture both sides before clearing drops all pending orders.
    bids = order_book.get_pending_bids()
    asks = order_book.get_pending_asks()
    _LOG.debug("num_bids='%s' num_asks='%s'", len(bids), len(asks))
    tier_results = order_book.clear_round()
    # Allocate once even when no tiers cleared, preserving empty-round behavior.
    round_id = contract_store.next_round_id()
    # Preserve the API's dispatch-before-save order and store-generated ids.
    contracts = rnocodis.build_contracts(bids, tier_results)
    rnocodis.dispatch_contracts(contracts, fulfillment_func=fulfillment_func)
    for contract in contracts:
        contract_store.save_contract(contract)
    # Publish even non-crossing and one-sided tiers, sharing one round id.
    summaries = {}
    matched_bids: Dict[Tuple[str, str], int] = {}
    matched_asks: Dict[Tuple[str, str], int] = {}
    for tier, tier_result in tier_results.items():
        summary = RoundSummary(
            tier,
            round_id,
            tier_result.clearing_price,
            sum(fill.n_tasks for fill in tier_result.fills),
        )
        contract_store.save_round(summary)
        summaries[tier] = summary
        # Include the tier in the key so equal ids in different tiers cannot mix.
        for fill in tier_result.fills:
            bid_key = (tier, fill.buyer_id)
            ask_key = (tier, fill.seller_id)
            matched_bids[bid_key] = matched_bids.get(bid_key, 0) + fill.n_tasks
            matched_asks[ask_key] = matched_asks.get(ask_key, 0) + fill.n_tasks
    # Mirror auction priority; never mutate or requeue the submitted orders.
    unmatched_bids = _get_unmatched_orders(
        sorted(bids, key=lambda bid: bid.p_max, reverse=True),
        matched_bids,
        lambda bid: (bid.c_level_min, bid.buyer_id),
    )
    unmatched_asks = _get_unmatched_orders(
        sorted(asks, key=lambda ask: ask.p_min),
        matched_asks,
        lambda ask: (ask.c_level, ask.seller_id),
    )
    result = RoundResult(
        round_id, summaries, contracts, unmatched_bids, unmatched_asks
    )
    _LOG.debug("round_id='%s' num_contracts='%s'", round_id, len(contracts))
    return result
