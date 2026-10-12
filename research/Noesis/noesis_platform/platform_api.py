"""
Thin HTTP API surface wrapping NoesisMarket and NoesisServer for an external
caller (`NoesisPlatform`)

- `create_app()` builds a `fastapi.FastAPI` app that round-trips every request
  to the underlying library call, adding no new business logic beyond:
    - Assigning a `contract_id`/`round_id`, since neither `batch_call_auction.py`
      nor `contract_dispatch.py` track one
    - A per-tier "latest cleared round" cache, standing in for `NoesisMarket`'s
      pricing-dissemination feed (not implemented yet);
      - `GET /rounds/{tier}/latest` reads this cache instead of that pub/sub
    - A `POST /rounds/clear` endpoint not in `plan.Noesis.md`'s illustrative
      endpoint list: without it, `GET /contracts/{contract_id}` and `GET
      /rounds/{tier}/latest` could never be populated by an external caller,
      since `OrderBook.clear_round()` is the only thing that produces a
      `Contract`/`TierClearResult` and the plan's list has no endpoint that
      calls it

Endpoints:
- NoesisMarket:
    - `POST /bids`
    - `POST /asks`
    - `POST /rounds/clear`
    - `GET /contracts/{contract_id}`
    - `GET /rounds/{tier}/latest`
- NoesisServer:
    - `POST /completions`
    - `GET /logs`
- Health:
    - `GET /health`, an unauthenticated liveness check that does not read
      `order_book`/`gateway` state

Auth:
- `POST /bids`, `POST /asks`, and `POST /completions` require a valid
  `X-API-Key` header, checked before the underlying library call runs
- Read-only `GET` endpoints and `POST /rounds/clear` are unauthenticated here

Import as:

import research.Noesis.noesis_platform.platform_api as rnoplapi
"""

import abc
import dataclasses
import logging
from typing import Callable, Dict, List, Optional

import fastapi
import pydantic

import helpers.hdbg as hdbg
import helpers.hprint as hprint
import research.Noesis.gateway.passthrough_proxy as rnopapro
import research.Noesis.market.batch_call_auction as rnbacaau
import research.Noesis.market.contract_dispatch as rnocodis

_LOG = logging.getLogger(__name__)


# #############################################################################
# Constants
# #############################################################################


# Header carrying the caller's API key on `POST /bids`, `POST /asks`, and
# `POST /completions`.
API_KEY_HEADER = "X-API-Key"


# #############################################################################
# BidRequest
# #############################################################################


class BidRequest(pydantic.BaseModel):
    """
    Request/response body for `POST /bids`, mirrors `batch_call_auction.Bid`.
    """

    buyer_id: str
    n_tasks: int
    c_level_min: str
    l_max: float
    r_min: float
    p_max: float


# #############################################################################
# AskRequest
# #############################################################################


class AskRequest(pydantic.BaseModel):
    """
    Request/response body for `POST /asks`, mirrors `batch_call_auction.Ask`.
    """

    seller_id: str
    n_tasks: int
    c_level: str
    l_typical: float
    r_typical: float
    p_min: float


# #############################################################################
# ContractResponse
# #############################################################################


class ContractResponse(pydantic.BaseModel):
    """
    Response body for `GET /contracts/{contract_id}`.

    Mirrors `contract_dispatch.Contract` plus the `contract_id` this API
    assigns (the dataclass itself has no id field).
    """

    contract_id: int
    buyer_id: str
    seller_id: str
    n_tasks: int
    c_level: str
    l_max: float
    r_min: float
    price: float
    fulfilled: Optional[bool]


# #############################################################################
# RoundClearResponse
# #############################################################################


class RoundClearResponse(pydantic.BaseModel):
    """
    One cleared tier's outcome, returned by
    - `POST /rounds/clear`
    - `GET /rounds/{tier}/latest`

    Field names match `NoesisMarket`'s pricing-dissemination event shape
    `(tier, round_id, clearing_price, matched_volume)`.
    """

    tier: str
    round_id: int
    clearing_price: Optional[float]
    matched_volume: int


# #############################################################################
# CompletionRequest
# #############################################################################


class CompletionRequest(pydantic.BaseModel):
    """
    Request body for `POST /completions`, wraps
    `passthrough_proxy.Gateway.call()`.
    """

    provider: str
    model: str
    prompt: str


# #############################################################################
# CompletionResponse
# #############################################################################


class CompletionResponse(pydantic.BaseModel):
    """
    Response body for `POST /completions`, mirrors the
    `passthrough_proxy.RequestLogEntry` the call logs.
    """

    request_id: int
    provider: str
    model: str
    response: str
    latency_in_secs: float
    cost: float


# #############################################################################
# LogEntryResponse
# #############################################################################


class LogEntryResponse(pydantic.BaseModel):
    """
    One logged request/response pair, returned by `GET /logs`, mirrors
    `passthrough_proxy.RequestLogEntry`.
    """

    request_id: int
    provider: str
    model: str
    prompt: str
    response: str
    latency_in_secs: float
    cost: float


# #############################################################################
# ContractStore
# #############################################################################


class ContractStore(abc.ABC):
    """
    Pluggable storage backend for `_MarketState`'s contract log and per-tier
    "latest cleared round" cache.
    """

    @abc.abstractmethod
    def save_contract(self, contract: rnocodis.Contract) -> int:
        """
        :return: the `contract_id` assigned to `contract`
        """
        ...

    @abc.abstractmethod
    def get_contract(self, contract_id: int) -> rnocodis.Contract: ...

    @abc.abstractmethod
    def next_round_id(self) -> int:
        """
        Assign one new `round_id`, shared by every tier cleared in the same
        `clear_round()` call (see `_MarketState.clear_round()`).
        """
        ...

    @abc.abstractmethod
    def save_round(self, round_response: "RoundClearResponse") -> None: ...

    @abc.abstractmethod
    def get_latest_round(self, tier: str) -> "RoundClearResponse": ...


# #############################################################################
# _InMemoryContractStore
# #############################################################################


class _InMemoryContractStore(ContractStore):
    """
    Default `ContractStore`: today's `_contracts_by_id`/`_next_contract_id`/
    `_latest_round_by_tier`/`_next_round_id`, extracted unchanged.
    """

    def __init__(self) -> None:
        self._contracts_by_id: Dict[int, rnocodis.Contract] = {}
        self._next_contract_id = 0
        self._latest_round_by_tier: Dict[str, RoundClearResponse] = {}
        self._next_round_id = 0

    def save_contract(self, contract: rnocodis.Contract) -> int:
        contract_id = self._next_contract_id
        self._contracts_by_id[contract_id] = contract
        self._next_contract_id += 1
        return contract_id

    def get_contract(self, contract_id: int) -> rnocodis.Contract:
        hdbg.dassert_in(
            contract_id,
            self._contracts_by_id,
            "Unknown contract_id '%s'",
            contract_id,
        )
        return self._contracts_by_id[contract_id]

    def next_round_id(self) -> int:
        round_id = self._next_round_id
        self._next_round_id += 1
        return round_id

    def save_round(self, round_response: RoundClearResponse) -> None:
        self._latest_round_by_tier[round_response.tier] = round_response

    def get_latest_round(self, tier: str) -> RoundClearResponse:
        hdbg.dassert_in(
            tier,
            self._latest_round_by_tier,
            "No cleared round yet for tier '%s'",
            tier,
        )
        return self._latest_round_by_tier[tier]


# #############################################################################
# _MarketState
# #############################################################################


class _MarketState:
    """
    State this API layer adds on top of `OrderBook`.

    - Tracks what `batch_call_auction.py`/`contract_dispatch.py` don't:
        - a `contract_id`/`round_id` counter
        - a per-tier "latest cleared round" cache
    - Lives in a pluggable `ContractStore` (in-memory by default,
      `postgres_store.PostgresContractStore` for persistence), not directly on
      this class
    """

    def __init__(
        self,
        order_book: rnbacaau.OrderBook,
        *,
        fulfillment_func: rnocodis.FulfillmentFunc = rnocodis.mock_fulfill,
        store: Optional[ContractStore] = None,
    ) -> None:
        """
        :param order_book: order book backing `submit_bid()`/`submit_ask()`/
            `clear_round()`
        :param fulfillment_func: passed through to
            `contract_dispatch.dispatch_contracts()` on every `clear_round()`
            - Default: `contract_dispatch.mock_fulfill`
        :param store: pluggable storage backend for the contract log and
            per-tier "latest cleared round" cache
            - Default: `_InMemoryContractStore()`; see
              `OrderBook.__init__()`'s `store` parameter for why this stays
              an `Optional[...] = None` default rather than a stateful
              class-level one
        """
        self._order_book = order_book
        self._fulfillment_func = fulfillment_func
        if store is None:
            store = _InMemoryContractStore()
        self._store = store

    def submit_bid(self, bid: rnbacaau.Bid) -> None:
        """
        Queue `bid` on the underlying order book.

        :param bid: buy order to queue
        """
        self._order_book.submit_bid(bid)

    def submit_ask(self, ask: rnbacaau.Ask) -> None:
        """
        Queue `ask` on the underlying order book.

        :param ask: sell order to queue
        """
        self._order_book.submit_ask(ask)

    def clear_round(self) -> List[RoundClearResponse]:
        """
        Clear the whole book, dispatch every cleared contract, and update
        the per-tier "latest cleared round" cache.

        :return: one `RoundClearResponse` per tier cleared this round
        """
        bids = self._order_book.get_pending_bids()
        tier_results = self._order_book.clear_round()
        # One `round_id` for the whole round, shared by every tier cleared
        # below (not one per tier, and not one per row on a persistent
        # store; see `spec.PR_P2b.md`'s "Risks and Limitations to Call
        # Out").
        round_id = self._store.next_round_id()
        # Build every contract for the round in one pass, dispatch each to
        # the (mocked) fulfillment layer, then persist it.
        contracts = rnocodis.build_contracts(bids, tier_results)
        rnocodis.dispatch_contracts(
            contracts, fulfillment_func=self._fulfillment_func
        )
        for contract in contracts:
            self._store.save_contract(contract)
        # Publish one `RoundClearResponse` per cleared tier, including tiers
        # that matched no volume, standing in for `NoesisMarket`'s pricing
        # feed until it exists.
        round_responses = []
        for c_level, tier_result in tier_results.items():
            matched_volume = sum(fill.n_tasks for fill in tier_result.fills)
            round_response = RoundClearResponse(
                tier=c_level,
                round_id=round_id,
                clearing_price=tier_result.clearing_price,
                matched_volume=matched_volume,
            )
            self._store.save_round(round_response)
            round_responses.append(round_response)
        return round_responses

    def get_contract(self, contract_id: int) -> ContractResponse:
        """
        Look up a contract cleared by an earlier `clear_round()`.

        :param contract_id: id assigned to the contract by `clear_round()`
        :return: the matching contract
        :raise AssertionError: if `contract_id` is unknown, caught by
            `create_app()`'s exception handler and surfaced as an HTTP 400
        """
        contract = self._store.get_contract(contract_id)
        return ContractResponse(
            contract_id=contract_id, **dataclasses.asdict(contract)
        )

    def get_latest_round(self, tier: str) -> RoundClearResponse:
        """
        Look up the most recently cleared round for `tier`.

        :param tier: capability tier to look up
        :return: `tier`'s latest `RoundClearResponse`
        :raise AssertionError: if no round has cleared `tier` yet, caught by
            `create_app()`'s exception handler and surfaced as an HTTP 400
        """
        return self._store.get_latest_round(tier)


# #############################################################################
# Auth
# #############################################################################


def _make_require_api_key(api_keys: Dict[str, str]) -> Callable[[str], str]:
    """
    Build a FastAPI dependency that checks `API_KEY_HEADER` against
    `api_keys`.

    :param api_keys: map from API key to account id, fixed at
        `create_app()` time
    :return: dependency function for `fastapi.Depends()`
    """

    def require_api_key(
        x_api_key: str = fastapi.Header(default="", alias=API_KEY_HEADER),
    ) -> str:
        """
        :return: account id owning `x_api_key`
        :raise fastapi.HTTPException: 401 if `x_api_key` is missing or not
            in `api_keys`
        """
        if x_api_key not in api_keys:
            raise fastapi.HTTPException(
                status_code=401, detail="Missing or invalid API key"
            )
        account_id = api_keys[x_api_key]
        return account_id

    return require_api_key


# #############################################################################
# App factory
# #############################################################################


def create_app(
    order_book: rnbacaau.OrderBook,
    gateway: rnopapro.Gateway,
    api_keys: Dict[str, str],
    *,
    fulfillment_func: rnocodis.FulfillmentFunc = rnocodis.mock_fulfill,
    contract_store: Optional[ContractStore] = None,
) -> fastapi.FastAPI:
    """
    Build the `NoesisPlatform` HTTP API over `order_book` and `gateway`.

    :param order_book: `NoesisMarket` order book backing `/bids`, `/asks`,
        `/rounds/*`, `/contracts/*`
    :param gateway: `NoesisServer` gateway backing `/completions`, `/logs`
    :param api_keys: map from API key to account id; `POST /bids`, `POST
        /asks`, and `POST /completions` reject the request with an HTTP 401
        unless `API_KEY_HEADER` is a key in this map
    :param fulfillment_func: passed through to
        `contract_dispatch.dispatch_contracts()` for every contract cleared
        by `POST /rounds/clear`
        - Default: `contract_dispatch.mock_fulfill`
    :param contract_store: pluggable storage backend for `_MarketState`'s
        contract log and per-tier "latest cleared round" cache
        - Default: `None`, threaded through to `_MarketState()` which
          resolves it to `_InMemoryContractStore()`; `main.py` passes
          `postgres_store.PostgresContractStore(connection)` here when
          `NOESIS_DB_BACKEND=postgres` (see `spec.PR_P2b.md`)
    :return: configured `fastapi.FastAPI` app, ready to serve or wrap in a
        `fastapi.testclient.TestClient`
    """
    _LOG.debug(hprint.to_str("api_keys"))
    market_state = _MarketState(
        order_book, fulfillment_func=fulfillment_func, store=contract_store
    )
    require_api_key = _make_require_api_key(api_keys)
    app = fastapi.FastAPI(title="NoesisPlatform")

    @app.exception_handler(AssertionError)
    async def _assertion_error_handler(
        _request: fastapi.Request, exc: AssertionError
    ) -> fastapi.responses.JSONResponse:
        """
        Surface an `hdbg.dassert_*` failure as an HTTP 400 instead of a
        stack trace.
        """
        return fastapi.responses.JSONResponse(
            status_code=400, content={"detail": str(exc)}
        )

    # #########################################################################
    # Health check.
    # #########################################################################

    @app.get("/health")
    def health() -> Dict[str, str]:
        # Does not read `order_book`/`gateway` state, so it stays healthy
        # before the first bid/ask is ever submitted (see the module
        # docstring's "Health" endpoint entry).
        return {"status": "ok"}

    # #########################################################################
    # NoesisMarket endpoints.
    # #########################################################################

    @app.post("/bids", status_code=201)
    def submit_bid(
        bid_request: BidRequest,
        _account_id: str = fastapi.Depends(require_api_key),
    ) -> BidRequest:
        bid = rnbacaau.Bid(
            bid_request.buyer_id,
            bid_request.n_tasks,
            bid_request.c_level_min,
            bid_request.l_max,
            bid_request.r_min,
            bid_request.p_max,
        )
        market_state.submit_bid(bid)
        return bid_request

    @app.post("/asks", status_code=201)
    def submit_ask(
        ask_request: AskRequest,
        _account_id: str = fastapi.Depends(require_api_key),
    ) -> AskRequest:
        ask = rnbacaau.Ask(
            ask_request.seller_id,
            ask_request.n_tasks,
            ask_request.c_level,
            ask_request.l_typical,
            ask_request.r_typical,
            ask_request.p_min,
        )
        market_state.submit_ask(ask)
        return ask_request

    @app.post("/rounds/clear")
    def clear_round() -> List[RoundClearResponse]:
        round_responses = market_state.clear_round()
        return round_responses

    @app.get("/contracts/{contract_id}")
    def get_contract(contract_id: int) -> ContractResponse:
        contract_response = market_state.get_contract(contract_id)
        return contract_response

    @app.get("/rounds/{tier}/latest")
    def get_latest_round(tier: str) -> RoundClearResponse:
        round_response = market_state.get_latest_round(tier)
        return round_response

    # #########################################################################
    # NoesisServer endpoints.
    # #########################################################################

    @app.post("/completions")
    def create_completion(
        completion_request: CompletionRequest,
        _account_id: str = fastapi.Depends(require_api_key),
    ) -> CompletionResponse:
        gateway.call(
            completion_request.provider,
            completion_request.model,
            completion_request.prompt,
        )
        # `call()` above always appends exactly one entry, so the log's last
        # entry is this request's; `Gateway` has no per-call return value
        # beyond the raw response text.
        entry = gateway.get_log()[-1]
        completion_response = CompletionResponse(
            request_id=entry.request_id,
            provider=entry.provider,
            model=entry.model,
            response=entry.response,
            latency_in_secs=entry.latency_in_secs,
            cost=entry.cost,
        )
        return completion_response

    @app.get("/logs")
    def get_logs(provider: str = "", model: str = "") -> List[LogEntryResponse]:
        entries = gateway.query_log(provider=provider, model=model)
        log_responses = [
            LogEntryResponse(**dataclasses.asdict(entry)) for entry in entries
        ]
        return log_responses

    return app
