"""
Who is calling, and which provider serves them.

`authenticate()` turns an API key into an account; `resolve_route()` turns a
contract id into the seller/provider that must serve it. Both raise
`RoutingError` carrying the HTTP status the API should return.

Import as:

import research.Noesis.gateway_routing as rnogarou
"""

import dataclasses
import logging
from typing import Optional

import psycopg_pool

import helpers.hprint as hprint
import research.Noesis.noesis_seed as rnonosee

_LOG = logging.getLogger(__name__)

ACTIVE = "ACTIVE"


# #############################################################################
# RoutingError
# #############################################################################


class RoutingError(Exception):
    """
    Request can't be routed; carries the HTTP status and an OpenAI-style code.

    This is a client error to return to the caller (e.g., bad API key), not a
    broken invariant, so it is an exception rather than a `dassert`.
    """

    def __init__(self, http_status: int, code: str, message: str) -> None:
        """
        Initialize the error.

        :param http_status: e.g., 401
        :param code: short machine-readable code, e.g., "invalid_api_key"
        :param message: human-readable explanation
        """
        super().__init__(message)
        self.http_status = http_status
        self.code = code
        self.message = message


# #############################################################################
# Account, Route
# #############################################################################


@dataclasses.dataclass(frozen=True)
class Account:
    """
    An authenticated caller.
    """

    account_id: str
    # One of "buyer", "seller", "operator".
    kind: str


@dataclasses.dataclass(frozen=True)
class Route:
    """
    Everything the gateway needs to serve one request under a contract.

    E.g., `Route("buyer_1", 42, "groq", "groq", "Groq", 2.0)` means contract 42
    of `buyer_1` is served by Groq, pinned via tag "groq", within 2 s.
    """

    buyer_id: str
    contract_id: int
    seller_id: str
    # What we pin at OpenRouter, e.g., "groq".
    provider_slug: str
    # Who should answer, e.g., "Groq".
    provider_name: str
    # Contract latency limit, in seconds.
    l_max: float


# #############################################################################
# Header parsing
# #############################################################################


def parse_bearer(authorization: Optional[str]) -> str:
    """
    Extract the API key from an `Authorization` header.

    E.g., "Bearer abc123" -> "abc123".

    :param authorization: raw header value, `None` if absent
    :return: the key; raises `RoutingError` 401 if missing or malformed
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise RoutingError(
            401, "invalid_api_key", "Missing 'Authorization: Bearer <key>' header"
        )
    key = authorization[len("Bearer ") :].strip()
    if not key:
        raise RoutingError(401, "invalid_api_key", "Empty API key")
    return key


def parse_contract_id(header: Optional[str]) -> int:
    """
    Parse the `X-Noesis-Contract` header.

    E.g., "42" -> 42.

    :param header: raw header value, `None` if absent
    :return: the contract id; raises `RoutingError` 400 if missing or not a
        positive integer
    """
    if header is None or not header.strip():
        raise RoutingError(
            400, "missing_contract", "Missing 'X-Noesis-Contract' header"
        )
    value = header.strip()
    if not value.isdigit() or int(value) == 0:
        raise RoutingError(
            400,
            "invalid_contract",
            f"Contract id must be a positive integer, got '{header}'",
        )
    return int(value)


# #############################################################################
# Database lookups
# #############################################################################


async def authenticate(
    pool: psycopg_pool.AsyncConnectionPool, api_key: str, kind: str
) -> Account:
    """
    Look up the account by the hash of its key and require the given kind.

    :param pool: open connection pool
    :param api_key: raw key from the request
    :param kind: "buyer" for completions, "operator" for admin endpoints
    :return: the account; raises `RoutingError` 401 (unknown key) or 403
        (wrong kind)
    """
    _LOG.debug(hprint.to_str("kind"))
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT account_id, kind FROM accounts WHERE api_key_hash = %s",
            (rnonosee.hash_api_key(api_key),),
        )
        row = await cursor.fetchone()
    if row is None:
        raise RoutingError(401, "invalid_api_key", "Invalid API key")
    account = Account(account_id=row[0], kind=row[1])
    if account.kind != kind:
        raise RoutingError(
            403, "wrong_account_kind", f"This endpoint needs a {kind} key"
        )
    _LOG.debug("return=%s", account)
    return account


async def resolve_route(
    pool: psycopg_pool.AsyncConnectionPool, buyer_id: str, contract_id: int
) -> Route:
    """
    Resolve contract -> seller -> provider, checking it's this buyer's and ACTIVE.

    Someone else's contract returns 404 (same as a missing one), so callers
    can't discover which contract ids exist.

    :param pool: open connection pool
    :param buyer_id: authenticated buyer
    :param contract_id: from `X-Noesis-Contract`
    :return: the route; raises `RoutingError` 404 (unknown / not yours) or
        409 (not ACTIVE)
    """
    _LOG.debug(hprint.to_str("buyer_id contract_id"))
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT c.buyer_account_id, c.state, c.l_max,"
            "       s.seller_id, s.provider_slug, s.provider_name "
            "FROM contracts c JOIN sellers s ON s.seller_id = c.seller_id "
            "WHERE c.contract_id = %s",
            (contract_id,),
        )
        row = await cursor.fetchone()
    if row is None or row[0] != buyer_id:
        raise RoutingError(
            404, "contract_not_found", f"Contract {contract_id} not found"
        )
    _, state, l_max, seller_id, slug, name = row
    if state != ACTIVE:
        raise RoutingError(
            409,
            "contract_not_active",
            f"Contract {contract_id} is {state}, not ACTIVE",
        )
    route = Route(
        buyer_id=buyer_id,
        contract_id=contract_id,
        seller_id=seller_id,
        provider_slug=slug,
        provider_name=name,
        l_max=float(l_max),
    )
    _LOG.debug("return=%s", route)
    return route
