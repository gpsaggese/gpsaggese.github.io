"""
Temporary operator tools until the auction (milestone 3) creates contracts.

Import as:

import research.Noesis.gateway_admin as rnogaadm
"""

import dataclasses
import logging

import psycopg_pool

import helpers.hprint as hprint
import research.Noesis.gateway_routing as rnogarou

_LOG = logging.getLogger(__name__)

DEV_TIER = "standard"


@dataclasses.dataclass(frozen=True)
class DevContract:
    """
    A contract created by hand through `/admin/dev-contract`.
    """

    contract_id: int
    buyer_id: str
    seller_id: str
    n_tasks: int
    l_max: float
    r_min: float
    price: float


async def create_dev_contract(
    pool: psycopg_pool.AsyncConnectionPool,
    buyer_id: str,
    seller_id: str,
    *,
    n_tasks: int = 50,
    l_max: float = 2.0,
    r_min: float = 0.9,
    price: float = 0.0,
) -> DevContract:
    """
    Create an ACTIVE contract by hand, with the round/bid/ask rows it needs.

    :param pool: open connection pool
    :param buyer_id: existing buyer account
    :param seller_id: existing seller
    :return: the new contract; raises `RoutingError` 404 if the buyer or
        seller doesn't exist
    """
    _LOG.debug(hprint.to_str("buyer_id seller_id n_tasks l_max r_min price"))
    async with pool.connection() as conn, conn.transaction():
        cursor = await conn.execute(
            "SELECT 1 FROM accounts WHERE account_id = %s AND kind = 'buyer'",
            (buyer_id,),
        )
        if await cursor.fetchone() is None:
            raise rnogarou.RoutingError(
                404, "buyer_not_found", f"Buyer '{buyer_id}' not found"
            )
        cursor = await conn.execute(
            "SELECT 1 FROM sellers WHERE seller_id = %s", (seller_id,)
        )
        if await cursor.fetchone() is None:
            raise rnogarou.RoutingError(
                404, "seller_not_found", f"Seller '{seller_id}' not found"
            )
        # A contract references the round, bid, and ask that produced it.
        cursor = await conn.execute(
            "INSERT INTO rounds"
            " (tier, cleared_at, clearing_price, matched_n_tasks) "
            "VALUES (%s, now(), %s, %s) RETURNING round_id",
            (DEV_TIER, price, n_tasks),
        )
        (round_id,) = await cursor.fetchone()
        cursor = await conn.execute(
            "INSERT INTO bids"
            " (account_id, n_tasks, c_level_min, l_max, r_min, p_max, round_id) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING order_id",
            (buyer_id, n_tasks, DEV_TIER, l_max, r_min, price, round_id),
        )
        (bid_id,) = await cursor.fetchone()
        cursor = await conn.execute(
            "INSERT INTO asks"
            " (seller_id, n_tasks, c_level, l_typical, r_typical, p_min,"
            " standing) "
            "VALUES (%s, %s, %s, %s, %s, %s, FALSE) RETURNING order_id",
            (seller_id, n_tasks, DEV_TIER, l_max, r_min, price),
        )
        (ask_id,) = await cursor.fetchone()
        cursor = await conn.execute(
            "INSERT INTO contracts (round_id, bid_order_id, ask_order_id,"
            " buyer_account_id, seller_id, n_tasks, c_level, l_max, r_min,"
            " price, state, window_start) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'ACTIVE', now()) "
            "RETURNING contract_id",
            (
                round_id,
                bid_id,
                ask_id,
                buyer_id,
                seller_id,
                n_tasks,
                DEV_TIER,
                l_max,
                r_min,
                price,
            ),
        )
        (contract_id,) = await cursor.fetchone()
    contract = DevContract(
        contract_id=contract_id,
        buyer_id=buyer_id,
        seller_id=seller_id,
        n_tasks=n_tasks,
        l_max=l_max,
        r_min=r_min,
        price=price,
    )
    _LOG.debug("return=%s", contract)
    return contract
