"""
Shared helpers for the Noesis MVP gateway tests: a running gateway app.

The database helpers live in `common/test/noesis_test_utils.py`.

Import as:

import research.Noesis.gateway.test.gateway_test_utils as rnteteut
"""

import contextlib
import dataclasses
import logging
from typing import Any, AsyncIterator, Dict

import httpx
import psycopg_pool

import helpers.hdbg as hdbg
import research.Noesis.common.test.noesis_test_utils as rnctnteut
import research.Noesis.gateway.gateway_app as rnogaapp
import research.Noesis.gateway.gateway_fake_provider as rnogafapr

_LOG = logging.getLogger(__name__)


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
    `common/seed.yaml`, and create one ACTIVE contract.

    :param behaviors: scripts for the fake provider, keyed by provider slug
    :param l_max: latency limit of the created contract
    """
    fake = rnogafapr.FakeProvider(behaviors)
    settings = rnctnteut.get_test_settings()
    app = rnogaapp.create_app(settings, fake, environ=rnctnteut.SEED_ENV)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as http:
            resp = await http.post(
                "/admin/dev-contract",
                json={
                    "buyer_id": "buyer_1",
                    "seller_id": "groq",
                    "l_max": l_max,
                },
                headers={"Authorization": f"Bearer {rnctnteut.OPERATOR_KEY}"},
            )
            hdbg.dassert_eq(
                resp.status_code, 201, "Dev contract failed: %s", resp.text
            )
            yield Gateway(app, http, transport, fake, resp.json()["contract_id"])
