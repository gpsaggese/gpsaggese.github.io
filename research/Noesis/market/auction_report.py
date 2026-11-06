"""
Convert auction inputs and outcomes to tables without external I/O.

Import as:

import research.Noesis.market.auction_report as rnmaaurp
"""

import dataclasses
import logging
from typing import List

import pandas as pd

import research.Noesis.market.batch_call_auction as rnbacaau
import research.Noesis.market.contract_dispatch as rnocodis

_LOG = logging.getLogger(__name__)

# One schema serves submitted and residual orders, including empty books.
_ORDER_TYPES = {
    "side": "string",
    "participant_id": "string",
    "tier": "string",
    "n_tasks": "Int64",
    "l_max": "Float64",
    "r_min": "Float64",
    "l_typical": "Float64",
    "r_typical": "Float64",
    "limit_price": "Float64",
}


# #############################################################################
# Order and contract tables
# #############################################################################


def orders_to_df(
    bids: List[rnbacaau.Bid], asks: List[rnbacaau.Ask]
) -> pd.DataFrame:
    """
    Return bids followed by asks in their supplied order, with explicit terms.

    :param bids: submitted buyer orders
    :param asks: submitted seller orders
    :return: typed table; opposite-side performance terms are missing, not zero
    """
    _LOG.debug("num_bids='%s' num_asks='%s'", len(bids), len(asks))
    # Distinguish buyer limits from seller expectations rather than merging them.
    rows = [
        {
            "side": "bid",
            "participant_id": b.buyer_id,
            "tier": b.c_level_min,
            "n_tasks": b.n_tasks,
            "l_max": b.l_max,
            "r_min": b.r_min,
            "limit_price": b.p_max,
        }
        for b in bids
    ]
    rows += [
        {
            "side": "ask",
            "participant_id": a.seller_id,
            "tier": a.c_level,
            "n_tasks": a.n_tasks,
            "l_typical": a.l_typical,
            "r_typical": a.r_typical,
            "limit_price": a.p_min,
        }
        for a in asks
    ]
    # Explicit columns and nullable types keep empty and populated tables aligned.
    result = pd.DataFrame(rows, columns=list(_ORDER_TYPES)).astype(_ORDER_TYPES)
    _LOG.debug("num_rows='%s'", len(result))
    return result


def contracts_to_df(contracts: List[rnocodis.Contract]) -> pd.DataFrame:
    """
    Return one row per contract, preserving dispatch order and pending outcomes.

    :param contracts: contracts from a market round
    :return: typed contract table; `fulfilled` describes the injected mock outcome
    """
    _LOG.debug("num_contracts='%s'", len(contracts))
    types = {
        "buyer_id": "string",
        "seller_id": "string",
        "n_tasks": "Int64",
        "tier": "string",
        "l_max": "Float64",
        "r_min": "Float64",
        "price": "Float64",
        "fulfilled": "boolean",
    }
    # Read copies so renaming the tier cannot alter the caller's contracts.
    rows = []
    for contract in contracts:
        row = dataclasses.asdict(contract)
        row["tier"] = row.pop("c_level")
        rows.append(row)
    result = pd.DataFrame(rows, columns=list(types)).astype(types)
    _LOG.debug("num_rows='%s'", len(result))
    return result
