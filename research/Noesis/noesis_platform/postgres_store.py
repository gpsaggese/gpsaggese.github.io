"""
Postgres-backed storage for `NoesisMarket`/`NoesisServer` state (order book,
contract log, request log).

Implements the three storage `abc.ABC`s owned by the modules that use them:
- `batch_call_auction.OrderBookStore` -> `PostgresOrderBookStore`
- `platform_api.ContractStore` -> `PostgresContractStore`
- `passthrough_proxy.RequestLogStore` -> `PostgresRequestLogStore`

This module imports the three owning modules to implement their `ABC`s; none
of them import this module back, so there is no import cycle, and no caller
that never touches the Postgres backend picks up a `psycopg2` dependency
transitively (matches `helpers.hsql`'s own optional-import gating: `main.py`
only imports this module inside its `NOESIS_DB_BACKEND == "postgres"`
branch).

Import as:

import research.Noesis.noesis_platform.postgres_store as rnoposto
"""

import dataclasses
import logging
from typing import List

import pandas as pd

import helpers.hdbg as hdbg
import helpers.hprint as hprint
import helpers.hsql_implementation as hsqlimpl
import research.Noesis.gateway.passthrough_proxy as rnopapro
import research.Noesis.market.batch_call_auction as rnbacaau
import research.Noesis.market.contract_dispatch as rnocodis
import research.Noesis.noesis_platform.platform_api as rnoplapi

_LOG = logging.getLogger(__name__)


# #############################################################################
# Schema
# #############################################################################


# `noesis_` prefix on every table, since this may run against a shared Postgres
# instance alongside other projects' tables.
# One statement per list entry: `init_schema()` runs them one at a time.
_SCHEMA_DDL = [
    # noesis_bids.
    """
    CREATE TABLE IF NOT EXISTS noesis_bids (
        id BIGSERIAL PRIMARY KEY,
        buyer_id TEXT NOT NULL,
        n_tasks INTEGER NOT NULL,
        c_level_min TEXT NOT NULL,
        l_max DOUBLE PRECISION NOT NULL,
        r_min DOUBLE PRECISION NOT NULL,
        p_max DOUBLE PRECISION NOT NULL
    )
    """,
    # noesis_asks.
    """
    CREATE TABLE IF NOT EXISTS noesis_asks (
        id BIGSERIAL PRIMARY KEY,
        seller_id TEXT NOT NULL,
        n_tasks INTEGER NOT NULL,
        c_level TEXT NOT NULL,
        l_typical DOUBLE PRECISION NOT NULL,
        r_typical DOUBLE PRECISION NOT NULL,
        p_min DOUBLE PRECISION NOT NULL
    )
    """,
    # noesis_contracts.
    """
    CREATE TABLE IF NOT EXISTS noesis_contracts (
        contract_id BIGSERIAL PRIMARY KEY,
        buyer_id TEXT NOT NULL,
        seller_id TEXT NOT NULL,
        n_tasks INTEGER NOT NULL,
        c_level TEXT NOT NULL,
        l_max DOUBLE PRECISION NOT NULL,
        r_min DOUBLE PRECISION NOT NULL,
        price DOUBLE PRECISION NOT NULL,
        fulfilled BOOLEAN
    )
    """,
    # noesis_round_id_seq
    # `round_id` is NOT `SERIAL` on `noesis_tier_rounds`: one `clear_round()`
    # call clears every tier under the SAME `round_id` (see
    # `PostgresContractStore.next_round_id()` below), so the id is generated
    # once per round, not once per row.
    """
    CREATE SEQUENCE IF NOT EXISTS noesis_round_id_seq
    """,
    # noesis_tier_rounds.
    """
    CREATE TABLE IF NOT EXISTS noesis_tier_rounds (
        tier TEXT NOT NULL,
        round_id BIGINT NOT NULL,
        clearing_price DOUBLE PRECISION,
        matched_volume INTEGER NOT NULL,
        cleared_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY (tier, round_id)
    )
    """,
    # noesis_request_log.
    """
    CREATE TABLE IF NOT EXISTS noesis_request_log (
        request_id BIGSERIAL PRIMARY KEY,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        prompt TEXT NOT NULL,
        response TEXT NOT NULL,
        latency_in_secs DOUBLE PRECISION NOT NULL,
        cost DOUBLE PRECISION NOT NULL
    )
    """,
]


def init_schema(connection: hsqlimpl.DbConnection) -> None:
    """
    Create every `noesis_*` table/sequence that doesn't exist yet.

    Idempotent (every statement is `CREATE TABLE/SEQUENCE IF NOT EXISTS`),
    safe to call on every `main.py` startup.

    :param connection: Postgres connection to run the DDL on
    """
    cursor = connection.cursor()
    for statement in _SCHEMA_DDL:
        cursor.execute(statement)
    # `hsql.get_connection*()` connections default to `autocommit=True`, but
    # commit explicitly so `init_schema()` is safe to call on any connection.
    connection.commit()
    _LOG.debug("Initialized noesis_* schema")


# #############################################################################
# PostgresOrderBookStore
# #############################################################################


class PostgresOrderBookStore(rnbacaau.OrderBookStore):
    """
    Postgres-backed `OrderBookStore`: persists pending bids/asks in
    `noesis_bids`/`noesis_asks`.
    """

    def __init__(self, connection: hsqlimpl.DbConnection) -> None:
        """
        :param connection: Postgres connection, shared for the process
            lifetime (see `spec.PR_P2b.md`'s "Out of Scope": no connection
            pooling)
        """
        self._connection = connection

    def add_bid(self, bid: rnbacaau.Bid) -> None:
        _LOG.debug(hprint.to_str("bid"))
        df = pd.DataFrame([dataclasses.asdict(bid)])
        hsqlimpl.execute_insert_query(self._connection, df, "noesis_bids")

    def add_ask(self, ask: rnbacaau.Ask) -> None:
        _LOG.debug(hprint.to_str("ask"))
        df = pd.DataFrame([dataclasses.asdict(ask)])
        hsqlimpl.execute_insert_query(self._connection, df, "noesis_asks")

    def get_bids(self) -> List[rnbacaau.Bid]:
        # `ORDER BY id` is required, not cosmetic: `_match_orders_in_tier()`'s
        # docstring states "`sorted()` is stable, so ties fall back to
        # submission order"; without an explicit `ORDER BY`, Postgres does
        # not guarantee row order, which would make tie-breaking within a
        # price-tied tier nondeterministic.
        query = (
            "SELECT buyer_id, n_tasks, c_level_min, l_max, r_min, p_max "
            "FROM noesis_bids ORDER BY id"
        )
        df = hsqlimpl.execute_query_to_df(self._connection, query)
        bids = [rnbacaau.Bid(**row) for row in df.to_dict("records")]
        return bids

    def get_asks(self) -> List[rnbacaau.Ask]:
        # Same "`ORDER BY` is required" reasoning as `get_bids()` above.
        query = (
            "SELECT seller_id, n_tasks, c_level, l_typical, r_typical, p_min "
            "FROM noesis_asks ORDER BY id"
        )
        df = hsqlimpl.execute_query_to_df(self._connection, query)
        asks = [rnbacaau.Ask(**row) for row in df.to_dict("records")]
        return asks

    def clear(self) -> None:
        # Matches `OrderBook.clear_round()`'s "drop everything" semantics
        # exactly. A `TRUNCATE` would work too, but `DELETE` keeps the
        # `BIGSERIAL` sequence monotonically increasing across rounds, which
        # is harmless here since bid/ask ids are never surfaced.
        cursor = self._connection.cursor()
        cursor.execute("DELETE FROM noesis_bids")
        cursor.execute("DELETE FROM noesis_asks")
        self._connection.commit()


# #############################################################################
# PostgresContractStore
# #############################################################################


class PostgresContractStore(rnoplapi.ContractStore):
    """
    Postgres-backed `ContractStore`: persists the contract log in
    `noesis_contracts` and the per-tier "latest cleared round" cache in
    `noesis_tier_rounds`.
    """

    def __init__(self, connection: hsqlimpl.DbConnection) -> None:
        """
        :param connection: Postgres connection, shared for the process
            lifetime
        """
        self._connection = connection

    def save_contract(self, contract: rnocodis.Contract) -> int:
        _LOG.debug(hprint.to_str("contract"))
        # `hsqlimpl.execute_insert_query()` is a bulk multi-row insert built
        # on `execute_values()` with no `RETURNING` support; this call
        # specifically needs the generated `contract_id` back, so it uses a
        # raw parameterized `INSERT ... RETURNING` instead.
        query = """
            INSERT INTO noesis_contracts(
                buyer_id, seller_id, n_tasks, c_level, l_max, r_min, price,
                fulfilled
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING contract_id
        """
        cursor = self._connection.cursor()
        cursor.execute(
            query,
            (
                contract.buyer_id,
                contract.seller_id,
                contract.n_tasks,
                contract.c_level,
                contract.l_max,
                contract.r_min,
                contract.price,
                contract.fulfilled,
            ),
        )
        contract_id = cursor.fetchone()[0]
        self._connection.commit()
        _LOG.debug("return=%s", contract_id)
        return contract_id

    def get_contract(self, contract_id: int) -> rnocodis.Contract:
        # `contract_id` is an `int` (already type-coerced by FastAPI's path
        # parameter before reaching here), so interpolating it is safe.
        query = (
            "SELECT buyer_id, seller_id, n_tasks, c_level, l_max, r_min, "
            f"price, fulfilled FROM noesis_contracts "
            f"WHERE contract_id = {contract_id}"
        )
        df = hsqlimpl.execute_query_to_df(self._connection, query)
        hdbg.dassert_lt(0, len(df), "Unknown contract_id '%s'", contract_id)
        contract = rnocodis.Contract(**df.iloc[0].to_dict())
        return contract

    def next_round_id(self) -> int:
        cursor = self._connection.cursor()
        cursor.execute("SELECT nextval('noesis_round_id_seq')")
        round_id = cursor.fetchone()[0]
        _LOG.debug("return=%s", round_id)
        return round_id

    def save_round(self, round_response: rnoplapi.RoundClearResponse) -> None:
        # No `RETURNING` needed: `round_id` is already known from
        # `next_round_id()`.
        query = """
            INSERT INTO noesis_tier_rounds(
                tier, round_id, clearing_price, matched_volume
            ) VALUES (%s, %s, %s, %s)
        """
        cursor = self._connection.cursor()
        cursor.execute(
            query,
            (
                round_response.tier,
                round_response.round_id,
                round_response.clearing_price,
                round_response.matched_volume,
            ),
        )
        self._connection.commit()

    def get_latest_round(self, tier: str) -> rnoplapi.RoundClearResponse:
        _LOG.debug(hprint.to_str("tier"))
        # `tier` is caller-controlled (`GET /rounds/{tier}/latest`'s path
        # parameter): bind it as a query parameter via a raw `cursor.execute`
        # instead of interpolating it into the SQL string, since
        # `hsqlimpl.execute_query_to_df()` has no parameter-binding support.
        query = """
            SELECT tier, round_id, clearing_price, matched_volume
            FROM noesis_tier_rounds
            WHERE tier = %s
            ORDER BY round_id DESC
            LIMIT 1
        """
        cursor = self._connection.cursor()
        cursor.execute(query, (tier,))
        row = cursor.fetchone()
        hdbg.dassert_is_not(
            row, None, "No cleared round yet for tier '%s'", tier
        )
        round_response = rnoplapi.RoundClearResponse(
            tier=row[0],
            round_id=row[1],
            clearing_price=row[2],
            matched_volume=row[3],
        )
        return round_response


# #############################################################################
# PostgresRequestLogStore
# #############################################################################


class PostgresRequestLogStore(rnopapro.RequestLogStore):
    """
    Postgres-backed `RequestLogStore`: persists the request/response log in
    `noesis_request_log`.
    """

    def __init__(self, connection: hsqlimpl.DbConnection) -> None:
        """
        :param connection: Postgres connection, shared for the process
            lifetime
        """
        self._connection = connection

    def append(
        self,
        provider: str,
        model: str,
        prompt: str,
        response: str,
        latency_in_secs: float,
        cost: float,
    ) -> rnopapro.RequestLogEntry:
        _LOG.debug(hprint.to_str("provider model"))
        # Same "needs `RETURNING`, so raw SQL instead of
        # `hsqlimpl.execute_insert_query()`" reasoning as
        # `PostgresContractStore.save_contract()`.
        query = """
            INSERT INTO noesis_request_log(
                provider, model, prompt, response, latency_in_secs, cost
            ) VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING request_id
        """
        cursor = self._connection.cursor()
        cursor.execute(
            query, (provider, model, prompt, response, latency_in_secs, cost)
        )
        request_id = cursor.fetchone()[0]
        self._connection.commit()
        entry = rnopapro.RequestLogEntry(
            request_id,
            provider,
            model,
            prompt,
            response,
            latency_in_secs,
            cost,
        )
        _LOG.debug("return=%s", entry)
        return entry

    def get_all(self) -> List[rnopapro.RequestLogEntry]:
        # Same "in call order" invariant as `PostgresOrderBookStore`'s
        # `get_bids()`/`get_asks()`, and for the same reason:
        # `Gateway.get_log()`'s docstring promises call order.
        query = (
            "SELECT request_id, provider, model, prompt, response, "
            "latency_in_secs, cost FROM noesis_request_log ORDER BY request_id"
        )
        df = hsqlimpl.execute_query_to_df(self._connection, query)
        entries = [
            rnopapro.RequestLogEntry(**row) for row in df.to_dict("records")
        ]
        return entries

    def query(
        self, *, provider: str = "", model: str = ""
    ) -> List[rnopapro.RequestLogEntry]:
        _LOG.debug(hprint.to_str("provider model"))
        # `provider`/`model` are caller-controlled (`GET /logs`'s query
        # parameters): build the `WHERE` clause with bound parameters
        # instead of interpolating them, skipping a clause for each empty
        # filter, matching `Gateway.query_log()`'s existing semantics.
        query_str = (
            "SELECT request_id, provider, model, prompt, response, "
            "latency_in_secs, cost FROM noesis_request_log"
        )
        conditions = []
        params: List[str] = []
        if provider:
            conditions.append("provider = %s")
            params.append(provider)
        if model:
            conditions.append("model = %s")
            params.append(model)
        if conditions:
            query_str += " WHERE " + " AND ".join(conditions)
        query_str += " ORDER BY request_id"
        cursor = self._connection.cursor()
        cursor.execute(query_str, tuple(params))
        rows = cursor.fetchall()
        entries = [
            rnopapro.RequestLogEntry(
                request_id=row[0],
                provider=row[1],
                model=row[2],
                prompt=row[3],
                response=row[4],
                latency_in_secs=row[5],
                cost=row[6],
            )
            for row in rows
        ]
        return entries
