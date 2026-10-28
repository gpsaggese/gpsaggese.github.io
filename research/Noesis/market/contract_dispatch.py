"""
Contract schema and stubbed fulfillment dispatch for the Noesis Market batch
call-auction `NoesisMarket` component.

Import as:

import research.Noesis.market.contract_dispatch as rnocodis
"""

import dataclasses
import logging
import random
from typing import Callable, Dict, List, Optional

import helpers.hdbg as hdbg
import helpers.hprint as hprint
import research.Noesis.market.batch_call_auction as rnbacaau

_LOG = logging.getLogger(__name__)


# #############################################################################
# Contract
# #############################################################################


@dataclasses.dataclass
class Contract:
    """
    Store one contract produced from a cleared `Fill`.

    E.g.,
        `Contract("buyer_1", "seller_1", 100, "frontier", 2.0, 0.999, 18.0)`
    means
        "100-task frontier-tier contract guaranteeing under 2.0 latency and at
        least 0.999 reliability, cleared at price 18.0"
    """

    # Identifier of the buyer this contract was matched for.
    buyer_id: str
    # Identifier of the seller this contract was matched with.
    seller_id: str
    # Number of tasks covered by this contract.
    n_tasks: int
    # Capability tier the contract cleared at.
    c_level: str
    # Maximum latency guaranteed to the buyer (from the winning bid).
    l_max: float
    # Minimum reliability guaranteed to the buyer (from the winning bid).
    r_min: float
    # Uniform clearing price the contract settled at.
    price: float
    # Fulfillment outcome logged by `dispatch_contract()`; `None` until
    # dispatched.
    fulfilled: Optional[bool] = None


# #############################################################################
# Contract building
# #############################################################################


def build_contracts(
    bids: List[rnbacaau.Bid],
    tier_results: Dict[str, rnbacaau.TierClearResult],
) -> List[Contract]:
    """
    Build one `Contract` per fill across every cleared tier.

    Simplification: at most one active bid per `buyer_id` is assumed; if a
    buyer submitted more than one bid in the round, the last one in `bids`
    supplies `l_max`/`r_min` for all of that buyer's fills.

    :param bids: bids submitted for the round that produced `tier_results`,
        i.e., what `OrderBook.get_pending_bids()` returned before
        `clear_round()` was called
    :param tier_results: per-tier clearing results from `clear_round()`
    :return: one `Contract` per fill, in tier-then-fill order
    """
    _LOG.debug("num_bids=%d num_tiers=%d", len(bids), len(tier_results))
    bid_by_buyer_id = {bid.buyer_id: bid for bid in bids}
    contracts: List[Contract] = []
    for tier_result in tier_results.values():
        for fill in tier_result.fills:
            hdbg.dassert_in(
                fill.buyer_id,
                bid_by_buyer_id,
                "Fill references buyer_id '%s' not present in `bids`",
                fill.buyer_id,
            )
            bid = bid_by_buyer_id[fill.buyer_id]
            contract = Contract(
                fill.buyer_id,
                fill.seller_id,
                fill.n_tasks,
                fill.c_level,
                bid.l_max,
                bid.r_min,
                fill.price,
            )
            contracts.append(contract)
    _LOG.debug("return=%s", contracts)
    return contracts


# #############################################################################
# Mock fulfillment layer
# #############################################################################


# A fulfillment call stands in for reporting a matched contract to
# `NoesisServer` and getting back a pass/fail outcome. Tests inject a
# deterministic stand-in (e.g., `lambda contract: True`) so no randomness
# leaks into assertions; `mock_fulfill()` below is the randomized default.
FulfillmentFunc = Callable[[Contract], bool]


# TODO(gp): Does it belong to testing?
def mock_fulfill(
    contract: Contract,
    *,
    success_rate: float = 0.9,
    rng: Optional[random.Random] = None,
) -> bool:
    """
    Stand in for `NoesisServer` and report a pass/fail outcome.

    Placeholder for the real fulfillment/monitoring layer (`NoesisServer`
    section of `plan.md`), which does not exist yet; swap this for
    the real interface once it does. For a fixed (non-randomized) outcome,
    pass a plain callable instead, e.g. `lambda contract: True`.

    :param contract: contract being (mock) fulfilled; unused by this stub,
        kept in the signature to match `FulfillmentFunc`
    :param success_rate: probability the mocked fulfillment succeeds
    :param rng: random source to draw from
        - Default: a fresh `random.Random()`
        - Tests should pass a seeded `random.Random` for determinism
    :return: `True` if the mocked fulfillment succeeded
    """
    _ = contract
    _LOG.debug(hprint.to_str("success_rate"))
    hdbg.dassert_lte(
        0.0, success_rate, "success_rate must be in [0, 1]: %s", success_rate
    )
    hdbg.dassert_lte(
        success_rate, 1.0, "success_rate must be in [0, 1]: %s", success_rate
    )
    if rng is None:
        rng = random.Random()
    outcome = rng.random() < success_rate
    _LOG.debug("return=%s", outcome)
    return outcome


# #############################################################################
# Dispatch
# #############################################################################


# TODO(ai_gp): Do not use a default for fulfillment_func but pass it explicitly
def dispatch_contract(
    contract: Contract, *, fulfillment_func: FulfillmentFunc = mock_fulfill
) -> Contract:
    """
    Dispatch `contract` to a fulfillment layer and log the outcome.

    :param contract: cleared contract to dispatch
    :param fulfillment_func: callable that reports the fulfillment outcome for
        `contract`
        - Default: `mock_fulfill()`
    :return: `contract`, mutated in place with `fulfilled` set
    """
    _LOG.debug(hprint.to_str("contract"))
    contract.fulfilled = fulfillment_func(contract)
    _LOG.debug("return=%s", contract)
    return contract


# TODO(ai_gp): Do not use a default for fulfillment_func but pass it explicitly
def dispatch_contracts(
    contracts: List[Contract],
    *,
    fulfillment_func: FulfillmentFunc = mock_fulfill,
) -> List[Contract]:
    """
    Dispatch every contract in `contracts` and log its outcome.

    :param contracts: cleared contracts to dispatch
    :param fulfillment_func: passed through to `dispatch_contract()` for each
        contract
        - Default: `mock_fulfill()`
    :return: `contracts`, each mutated in place with `fulfilled` set
    """
    _LOG.debug("num_contracts=%d", len(contracts))
    for contract in contracts:
        dispatch_contract(contract, fulfillment_func=fulfillment_func)
    _LOG.debug("return=%s", contracts)
    return contracts
