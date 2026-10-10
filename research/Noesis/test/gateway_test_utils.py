"""
Shared helpers for the Noesis MVP tests that need Postgres.

Tests connect to `NOESIS_TEST_DATABASE_URL` (default: the local test DB on
port 5433) and skip cleanly when it isn't reachable, so they never fail on a
machine or CI runner without a database. See `docs/onboarding.md` for starting
the test DB.

Import as:

import research.Noesis.test.gateway_test_utils as rnteteut
"""

import asyncio
import contextlib
import dataclasses
import logging
import os
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List

import httpx
import psycopg
import psycopg_pool

import helpers.hdbg as hdbg
import helpers.hunit_test as hunitest
import research.Noesis.gateway_app as rnogaapp
import research.Noesis.gateway_fake_provider as rnogafapr
import research.Noesis.noesis_db as rnonodb
import research.Noesis.noesis_seed as rnonosee
import research.Noesis.noesis_settings as rnonoset

_LOG = logging.getLogger(__name__)

TEST_DB_URL = os.environ.get(
    "NOESIS_TEST_DATABASE_URL",
    "postgresql://noesis:noesis@localhost:5433/noesis_test",
)

BUYER_KEY = "test-buyer-key"
OPERATOR_KEY = "test-operator-key"
SEED_ENV = {"NOESIS_BUYER_1_KEY": BUYER_KEY, "NOESIS_OPERATOR_KEY": OPERATOR_KEY}


# #############################################################################
# Database
# #############################################################################


def get_test_settings(**overrides: Any) -> rnonoset.Settings:
    """
    Return default settings pointing at the test DB, ignoring any `.env`.
    """
    settings = rnonoset.load_settings(
        environ={"NOESIS_DATABASE_URL": TEST_DB_URL},
        dotenv_path="/nonexistent/.env",
    )
    settings = dataclasses.replace(settings, **overrides)
    return settings


def run_with_clean_db(
    self_: hunitest.TestCase,
    body: Callable[[psycopg_pool.AsyncConnectionPool], Awaitable[Any]],
) -> Any:
    """
    Connect, apply migrations, empty every MVP table, and run `body(pool)`.

    Skips the calling test if the test DB isn't reachable.

    :param self_: the calling test case (used to skip)
    :param body: async function receiving the open pool
    :return: whatever `body` returns
    """

    async def _go() -> Any:
        # A missing test DB is an environment condition, not a test failure.
        try:
            pool = await rnonodb.connect(TEST_DB_URL, max_size=2)
        except (psycopg.OperationalError, psycopg_pool.PoolTimeout) as e:
            self_.skipTest(f"Test Postgres not reachable ({e})")
        try:
            await rnonodb.apply_migrations(pool)
            async with pool.connection() as conn:
                await conn.execute(
                    "TRUNCATE accounts, sellers, rounds, canaries CASCADE"
                )
            return await body(pool)
        finally:
            await pool.close()

    return asyncio.run(_go())


async def fetch_all(
    pool: psycopg_pool.AsyncConnectionPool, query: str, *params: Any
) -> List[tuple]:
    """
    Run `query` and return all rows.
    """
    async with pool.connection() as conn:
        cursor = await conn.execute(query, params or None)
        rows = await cursor.fetchall()
    return rows


async def add_account(
    pool: psycopg_pool.AsyncConnectionPool,
    account_id: str,
    kind: str,
    api_key: str,
) -> None:
    """
    Insert an account whose key is `api_key`.
    """
    async with pool.connection() as conn:
        await conn.execute(
            "INSERT INTO accounts (account_id, kind, api_key_hash) "
            "VALUES (%s, %s, %s)",
            (account_id, kind, rnonosee.hash_api_key(api_key)),
        )


async def add_seller(
    pool: psycopg_pool.AsyncConnectionPool,
    seller_id: str,
    provider_slug: str,
    provider_name: str,
) -> None:
    """
    Insert a seller (and its account).
    """
    await add_account(
        pool, f"seller_{seller_id}", "seller", f"unused-{seller_id}"
    )
    async with pool.connection() as conn:
        await conn.execute(
            "INSERT INTO sellers"
            " (seller_id, account_id, provider_slug, provider_name, model_id) "
            "VALUES (%s, %s, %s, %s, 'test-model')",
            (seller_id, f"seller_{seller_id}", provider_slug, provider_name),
        )


async def add_contract(
    pool: psycopg_pool.AsyncConnectionPool,
    buyer_id: str,
    seller_id: str,
    *,
    state: str = "ACTIVE",
    l_max: float = 2.0,
) -> int:
    """
    Create the round, bid, and ask a contract needs, then the contract.

    :return: the new `contract_id`
    """
    async with pool.connection() as conn, conn.transaction():
        cursor = await conn.execute(
            "INSERT INTO rounds (tier) VALUES ('standard') RETURNING round_id"
        )
        (round_id,) = await cursor.fetchone()
        cursor = await conn.execute(
            "INSERT INTO bids (account_id, n_tasks, c_level_min, l_max, r_min,"
            " p_max) VALUES (%s, 50, 'standard', %s, 0.9, 0.05) "
            "RETURNING order_id",
            (buyer_id, l_max),
        )
        (bid_id,) = await cursor.fetchone()
        cursor = await conn.execute(
            "INSERT INTO asks (seller_id, n_tasks, c_level, l_typical, r_typical,"
            " p_min) VALUES (%s, 50, 'standard', 1.0, 0.95, 0.01) "
            "RETURNING order_id",
            (seller_id,),
        )
        (ask_id,) = await cursor.fetchone()
        cursor = await conn.execute(
            "INSERT INTO contracts (round_id, bid_order_id, ask_order_id,"
            " buyer_account_id, seller_id, n_tasks, c_level, l_max, r_min,"
            " price, state) VALUES (%s, %s, %s, %s, %s, 50, 'standard', %s,"
            " 0.9, 0.03, %s) RETURNING contract_id",
            (round_id, bid_id, ask_id, buyer_id, seller_id, l_max, state),
        )
        (contract_id,) = await cursor.fetchone()
    return contract_id


# #############################################################################
# Gateway app
# #############################################################################


@dataclasses.dataclass
class Gateway:
    """
    A running gateway app plus clients to talk to it in-process.
    """

    app: Any
    http: httpx.AsyncClient
    transport: httpx.ASGITransport
    fake: rnogafapr.FakeProvider
    # ACTIVE contract `buyer_1` -> `groq`, created through the admin endpoint.
    contract_id: int

    @property
    def pool(self) -> psycopg_pool.AsyncConnectionPool:
        return self.app.state.pool


@contextlib.asynccontextmanager
async def running_gateway(
    behaviors: Dict[str, rnogafapr.FakeBehavior],
    *,
    l_max: float = 2.0,
) -> AsyncIterator[Gateway]:
    """
    Start the app against the test DB with a `FakeProvider`, seeded from
    `config/seed.yaml`, and create one ACTIVE contract.

    :param behaviors: scripts for the fake provider, keyed by provider slug
    :param l_max: latency limit of the created contract
    """
    fake = rnogafapr.FakeProvider(behaviors)
    app = rnogaapp.create_app(get_test_settings(), fake, environ=SEED_ENV)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as http:
            resp = await http.post(
                "/admin/dev-contract",
                json={"buyer_id": "buyer_1", "seller_id": "groq", "l_max": l_max},
                headers={"Authorization": f"Bearer {OPERATOR_KEY}"},
            )
            hdbg.dassert_eq(
                resp.status_code, 201, "Dev contract failed: %s", resp.text
            )
            yield Gateway(app, http, transport, fake, resp.json()["contract_id"])
