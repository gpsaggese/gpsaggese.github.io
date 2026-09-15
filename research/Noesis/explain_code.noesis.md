# File Description

research/Noesis/batch_call_auction.py research/Noesis/contract_dispatch.py
research/Noesis/main.py research/Noesis/passthrough_proxy.py
research/Noesis/platform_api.py research/Noesis/postgres_store.py
- `batch_call_auction.py`
  - In-memory order book for a batch call auction
  - Queues buy/sell orders, buckets by capability tier
  - Clears each tier at a uniform price once per round
- `passthrough_proxy.py`
  - LLM provider gateway
  - Routes a prompt to a registered provider
  - Times the call, estimates cost, and logs every request/response pair
- `contract_dispatch.py`
  - Turns a cleared auction round's fills into contracts
  - Then dispatches each contract to a (mocked) fulfillment layer
  - Records pass/fail
- `platform_api.py`
  - HTTP API surface over the auction and gateway
  - Adds API-key auth, contract/round id tracking
  - Adds a thin FastAPI layer with no extra business logic beyond that bookkeeping
- `postgres_store.py`
  - Persistence layer swapping the in-memory stores for Postgres-backed ones, so
    order book, contract log, and request log survive process restarts
- `main.py`
  - Process entrypoint
  - Reads env config
  - Picks memory or Postgres backend
  - Wires everything together
  - Exposes the app `uvicorn` serves

# Class Description

## `research/Noesis/batch_call_auction.py`

- `Bid`
  - Buyer order: (tasks, min tier, max latency, min reliability, max price)
- `Ask`
  - Seller order: (tasks, tier, typical latency/reliability, min price)
- `Fill`
  - One matched buyer/seller trade at a tier's uniform clearing price
- `TierClearResult`
  - Outcome of clearing one tier (price, fills, unfilled quantities)
- `OrderBookStore`
  - Abstract pluggable storage backend for pending bids/asks
- `_InMemoryOrderBookStore`
  - Default in-memory list-based `OrderBookStore`
- `OrderBook`
  - Batch call-auction book: queues orders and clears tiers per round

## `research/Noesis/contract_dispatch.py`

- `Contract`
  - One contract produced from a cleared `Fill`, with fulfillment outcome

## `research/Noesis/passthrough_proxy.py`

- `ProviderConfig`
  - Registers one LLM provider behind the gateway's `call()` API
- `RequestLogEntry`
  - Storage schema for one logged prompt/response pair
- `RequestLogStore`
  - Abstract pluggable storage backend for `Gateway`'s request log
- `_InMemoryRequestLogStore`
  - Default in-memory `RequestLogStore`
- `Gateway`
  - Passthrough proxy: dispatches prompts to providers and logs exchanges

## `research/Noesis/platform_api.py`

- `BidRequest`
  - Request/response body for `POST /bids`, mirrors `Bid`
- `AskRequest`
  - Request/response body for `POST /asks`, mirrors `Ask`
- `ContractResponse`
  - Response body for `GET /contracts/{contract_id}`
- `RoundClearResponse`
  - One cleared tier's outcome, returned by round-clearing endpoints
- `CompletionRequest`
  - Request body for `POST /completions`
- `CompletionResponse`
  - Response body for `POST /completions`
- `LogEntryResponse`
  - One logged request/response pair, returned by `GET /logs`
- `ContractStore`
  - Abstract pluggable storage for the contract log and round cache
- `_InMemoryContractStore`
  - Default in-memory `ContractStore`
- `_MarketState`
  - State layer over `OrderBook`: contract/round ids, latest-round cache

## `research/Noesis/postgres_store.py`

- `PostgresOrderBookStore`
  - Postgres-backed `OrderBookStore` persisting bids/asks
- `PostgresContractStore`
  - Postgres-backed `ContractStore` persisting contracts and round cache
- `PostgresRequestLogStore`
  - Postgres-backed `RequestLogStore` persisting the request log

# Class Interface

## `research/Noesis/batch_call_auction.py`

- `Bid(dataclass)`
  - `__init__(self, buyer_id: str, n_tasks: int, c_level_min: str, l_max: float, r_min: float, p_max: float) -> None`
    - Validates and stores one buy order
- `Ask(dataclass)`
  - `__init__(self, seller_id: str, n_tasks: int, c_level: str, l_typical: float, r_typical: float, p_min: float) -> None`
    - Validates and stores one sell order
- `Fill(dataclass)`
  - Plain data holder, no custom methods
- `TierClearResult(dataclass)`
  - Plain data holder, no custom methods
- `OrderBookStore(abc.ABC)`
  - `add_bid(self, bid: Bid) -> None`
    - Abstract, queue a bid
  - `add_ask(self, ask: Ask) -> None`
    - Abstract, queue an ask
  - `get_bids(self) -> List[Bid]`
    - Abstract, return pending bids in submission order
  - `get_asks(self) -> List[Ask]`
    - Abstract, return pending asks in submission order
  - `clear(self) -> None`
    - Abstract, drop every stored bid/ask
- `_InMemoryOrderBookStore(OrderBookStore)`
  - `__init__(self) -> None`
    - Init empty in-memory bid/ask lists
  - `add_bid(self, bid: Bid) -> None`
    - Append bid to internal list
  - `add_ask(self, ask: Ask) -> None`
    - Append ask to internal list
  - `get_bids(self) -> List[Bid]`
    - Return copy of internal bid list
  - `get_asks(self) -> List[Ask]`
    - Return copy of internal ask list
  - `clear(self) -> None`
    - Reset internal bid/ask lists to empty
- `OrderBook`
  - `__init__(self, *, store: Optional[OrderBookStore] = None) -> None`
    - Init book, defaulting to in-memory store
  - `submit_bid(self, bid: Bid) -> None`
    - Queue buyer order for next `clear_round()`
  - `submit_ask(self, ask: Ask) -> None`
    - Queue seller order for next `clear_round()`
  - `get_pending_bids(self) -> List[Bid]`
    - Return buy orders queued for next `clear_round()`
  - `get_pending_asks(self) -> List[Ask]`
    - Return sell orders queued for next `clear_round()`
  - `clear_round(self) -> Dict[str, TierClearResult]`
    - Clear every tier present in the book and empty it

## `research/Noesis/contract_dispatch.py`

- `Contract(dataclass)`
  - Plain data holder, no custom methods

## `research/Noesis/passthrough_proxy.py`

- `ProviderConfig(dataclass)`
  - `__init__(self, name: str, call_func: ProviderCallFunc, cost_per_char: float) -> None`
    - Validates and stores one provider registration
- `RequestLogEntry(dataclass)`
  - Plain data holder, no custom methods
- `RequestLogStore(abc.ABC)`
  - `append(self, provider: str, model: str, prompt: str, response: str, latency_in_secs: float, cost: float) -> RequestLogEntry`
    - Abstract, persist one logged request/response pair
  - `get_all(self) -> List[RequestLogEntry]`
    - Abstract, return every logged entry in call order
  - `query(self, *, provider: str = "", model: str = "") -> List[RequestLogEntry]`
    - Abstract, return entries filtered by provider/model
- `_InMemoryRequestLogStore(RequestLogStore)`
  - `__init__(self) -> None`
    - Init empty log list and request id counter
  - `append(self, provider: str, model: str, prompt: str, response: str, latency_in_secs: float, cost: float) -> RequestLogEntry`
    - Build entry, assign id, append to internal list
  - `get_all(self) -> List[RequestLogEntry]`
    - Return copy of internal log list
  - `query(self, *, provider: str = "", model: str = "") -> List[RequestLogEntry]`
    - Filter internal log by provider and/or model
- `Gateway`
  - `__init__(self, *, clock_func: Callable[[], float] = time.perf_counter, store: Optional[RequestLogStore] = None) -> None`
    - Init providers map and clock, defaulting to in-memory store
  - `register_provider(self, provider_config: ProviderConfig) -> None`
    - Register provider so `call()` can route to it
  - `call(self, provider_name: str, model: str, prompt: str) -> str`
    - Route prompt to provider/model and log the exchange
  - `get_log(self) -> List[RequestLogEntry]`
    - Return every logged request/response pair in call order
  - `query_log(self, *, provider: str = "", model: str = "") -> List[RequestLogEntry]`
    - Return logged entries filtered by provider and/or model

## `research/Noesis/platform_api.py`

- `BidRequest(pydantic.BaseModel)`
  - Plain data holder, no custom methods
- `AskRequest(pydantic.BaseModel)`
  - Plain data holder, no custom methods
- `ContractResponse(pydantic.BaseModel)`
  - Plain data holder, no custom methods
- `RoundClearResponse(pydantic.BaseModel)`
  - Plain data holder, no custom methods
- `CompletionRequest(pydantic.BaseModel)`
  - Plain data holder, no custom methods
- `CompletionResponse(pydantic.BaseModel)`
  - Plain data holder, no custom methods
- `LogEntryResponse(pydantic.BaseModel)`
  - Plain data holder, no custom methods
- `ContractStore(abc.ABC)`
  - `save_contract(self, contract: rnocodis.Contract) -> int`
    - Abstract, persist contract and return its assigned `contract_id`
  - `get_contract(self, contract_id: int) -> rnocodis.Contract`
    - Abstract, look up a contract by id
  - `next_round_id(self) -> int`
    - Abstract, assign one new `round_id` shared across a round's tiers
  - `save_round(self, round_response: "RoundClearResponse") -> None`
    - Abstract, persist one tier's latest cleared round
  - `get_latest_round(self, tier: str) -> "RoundClearResponse"`
    - Abstract, look up a tier's latest cleared round
- `_InMemoryContractStore(ContractStore)`
  - `__init__(self) -> None`
    - Init empty contract map, round cache, and id counters
  - `save_contract(self, contract: rnocodis.Contract) -> int`
    - Store contract under a new id, increment counter
  - `get_contract(self, contract_id: int) -> rnocodis.Contract`
    - Look up contract by id, assert it exists
  - `next_round_id(self) -> int`
    - Return and increment the round id counter
  - `save_round(self, round_response: RoundClearResponse) -> None`
    - Cache `round_response` under its tier
  - `get_latest_round(self, tier: str) -> RoundClearResponse`
    - Look up tier's cached round response, assert it exists
- `_MarketState`
  - `__init__(self, order_book: rnbacaau.OrderBook, *, fulfillment_func: rnocodis.FulfillmentFunc = rnocodis.mock_fulfill, store: Optional[ContractStore] = None) -> None`
    - Init state over `order_book`, defaulting to in-memory contract store
  - `submit_bid(self, bid: rnbacaau.Bid) -> None`
    - Queue bid on the underlying order book
  - `submit_ask(self, ask: rnbacaau.Ask) -> None`
    - Queue ask on the underlying order book
  - `clear_round(self) -> List[RoundClearResponse]`
    - Clear book, dispatch contracts, update the latest-round cache
  - `get_contract(self, contract_id: int) -> ContractResponse`
    - Look up a contract cleared by an earlier `clear_round()`
  - `get_latest_round(self, tier: str) -> RoundClearResponse`
    - Look up a tier's most recently cleared round

## `research/Noesis/postgres_store.py`

- `PostgresOrderBookStore(rnbacaau.OrderBookStore)`
  - `__init__(self, connection: hsqlimpl.DbConnection) -> None`
    - Store shared Postgres connection
  - `add_bid(self, bid: rnbacaau.Bid) -> None`
    - Insert a bid row into `noesis_bids`
  - `add_ask(self, ask: rnbacaau.Ask) -> None`
    - Insert an ask row into `noesis_asks`
  - `get_bids(self) -> List[rnbacaau.Bid]`
    - Query `noesis_bids` ordered by id, return as `Bid` list
  - `get_asks(self) -> List[rnbacaau.Ask]`
    - Query `noesis_asks` ordered by id, return as `Ask` list
  - `clear(self) -> None`
    - Delete every row from `noesis_bids` and `noesis_asks`
- `PostgresContractStore(rnoplapi.ContractStore)`
  - `__init__(self, connection: hsqlimpl.DbConnection) -> None`
    - Store shared Postgres connection
  - `save_contract(self, contract: rnocodis.Contract) -> int`
    - Insert contract row, return generated `contract_id`
  - `get_contract(self, contract_id: int) -> rnocodis.Contract`
    - Query `noesis_contracts` by `contract_id`, return as `Contract`
  - `next_round_id(self) -> int`
    - Draw next value from `noesis_round_id_seq`
  - `save_round(self, round_response: rnoplapi.RoundClearResponse) -> None`
    - Insert a round row into `noesis_tier_rounds`
  - `get_latest_round(self, tier: str) -> rnoplapi.RoundClearResponse`
    - Query the latest `noesis_tier_rounds` row for `tier`
- `PostgresRequestLogStore(rnopapro.RequestLogStore)`
  - `__init__(self, connection: hsqlimpl.DbConnection) -> None`
    - Store shared Postgres connection
  - `append(self, provider: str, model: str, prompt: str, response: str, latency_in_secs: float, cost: float) -> rnopapro.RequestLogEntry`
    - Insert log row, return entry with generated `request_id`
  - `get_all(self) -> List[rnopapro.RequestLogEntry]`
    - Query `noesis_request_log` ordered by `request_id`
  - `query(self, *, provider: str = "", model: str = "") -> List[rnopapro.RequestLogEntry]`
    - Query `noesis_request_log` filtered by provider/model

# Function Interface

## `research/Noesis/batch_call_auction.py`

- `_match_orders_in_tier(c_level: str, bids: List[Bid], asks: List[Ask]) -> TierClearResult`
  - Matches one tier's bids/asks and computes uniform clearing price

## `research/Noesis/contract_dispatch.py`

- `build_contracts(bids: List[rnbacaau.Bid], tier_results: Dict[str, rnbacaau.TierClearResult]) -> List[Contract]`
  - Builds one `Contract` per fill across every cleared tier
- `mock_fulfill(contract: Contract, *, success_rate: float = DEFAULT_FULFILLMENT_SUCCESS_RATE, rng: Optional[random.Random] = None) -> bool`
  - Stub randomized pass/fail outcome standing in for the real fulfillment layer
- `dispatch_contract(contract: Contract, *, fulfillment_func: FulfillmentFunc = mock_fulfill) -> Contract`
  - Dispatches one contract to a fulfillment layer and logs the outcome
- `dispatch_contracts(contracts: List[Contract], *, fulfillment_func: FulfillmentFunc = mock_fulfill) -> List[Contract]`
  - Dispatches every contract in the list and logs each outcome

## `research/Noesis/main.py`

- `_parse_api_keys(raw: str) -> Dict[str, str]`
  - Parses `NOESIS_API_KEYS` comma-separated `key:account` pairs into a dict
- `_get_db_backend() -> str`
  - Resolves `NOESIS_DB_BACKEND` env var, defaulting to `"memory"`

## `research/Noesis/platform_api.py`

- `_make_require_api_key(api_keys: Dict[str, str]) -> Callable[[str], str]`
  - Builds a FastAPI dependency checking `X-API-Key` header against `api_keys`
- `create_app(order_book: rnbacaau.OrderBook, gateway: rnopapro.Gateway, api_keys: Dict[str, str], *, fulfillment_func: rnocodis.FulfillmentFunc = rnocodis.mock_fulfill, contract_store: Optional[ContractStore] = None) -> fastapi.FastAPI`
  - Builds the `NoesisPlatform` HTTP API wiring endpoints to `order_book`/`gateway`

## `research/Noesis/postgres_store.py`

- `init_schema(connection: hsqlimpl.DbConnection) -> None`
  - Creates every `noesis_*` table/sequence that doesn't exist yet

# Function Relationship

## `research/Noesis/batch_call_auction.py`

- `_match_orders_in_tier()`
  - Called by: `OrderBook.clear_round()`, once per capability tier
  - Calls: no other functions in this file

## `research/Noesis/contract_dispatch.py`

- `build_contracts()`
  - Called by: `_MarketState.clear_round()` (`platform_api.py`)
  - Calls: no other functions in this file
- `mock_fulfill()`
  - Called by: `dispatch_contract()` and `_MarketState` (`platform_api.py`) as the
    default `fulfillment_func`
  - Calls: no other functions in this file
- `dispatch_contract()`
  - Called by: `dispatch_contracts()`, once per contract
  - Calls: `fulfillment_func` (`mock_fulfill()` by default)
- `dispatch_contracts()`
  - Called by: `_MarketState.clear_round()` (`platform_api.py`)
  - Calls: `dispatch_contract()`, once per contract

## `research/Noesis/main.py`

- `_parse_api_keys()`
  - Called by: module-level app construction code building `api_keys`
  - Calls: no other functions in this file
- `_get_db_backend()`
  - Called by: module-level app construction code setting `_DB_BACKEND`
  - Calls: no other functions in this file

## `research/Noesis/platform_api.py`

- `_make_require_api_key()`
  - Called by: `create_app()`
  - Calls: no other functions in this file
- `create_app()`
  - Called by: `main.py`'s module-level app construction code
  - Calls: `_make_require_api_key()`, `_MarketState()`

## `research/Noesis/postgres_store.py`

- `init_schema()`
  - Called by: `main.py`'s postgres branch, before building the Postgres stores
  - Calls: no other functions in this file

# Class Relationships

## Inheritance

- `OrderBookStore`
  - `_InMemoryOrderBookStore` from `batch_call_auction.py`
  - `PostgresOrderBookStore` from `postgres_store.py`
- `RequestLogStore`
  - `_InMemoryRequestLogStore` from `passthrough_proxy.py`
  - `PostgresRequestLogStore` from `postgres_store.py`
- `ContractStore`
  - `_InMemoryContractStore` from `platform_api.py`
  - `PostgresContractStore` from `postgres_store.py`

## Composition

- `OrderBook`
  - Holds one `OrderBookStore`
- `Gateway`
  - Holds one `RequestLogStore` and a map of registered `ProviderConfig`
- `_MarketState`
  - Holds one `OrderBook`, one `ContractStore`, and one `FulfillmentFunc`

## Uses

- `OrderBook.clear_round()` calls `_match_orders_in_tier()` per capability tier and
  returns `Dict[str, TierClearResult]`
- `TierClearResult` aggregates `List[Fill]`
- `build_contracts()` consumes `List[Bid]` and `Dict[str, TierClearResult]` to
  produce `List[Contract]`
- `dispatch_contracts()` mutates each `Contract.fulfilled` via `mock_fulfill()` or an
  injected `fulfillment_func`
- `_MarketState.clear_round()` calls `OrderBook.clear_round()`, `build_contracts()`,
  and `dispatch_contracts()` in sequence
- `create_app()` wires `_MarketState` and `Gateway` into the FastAPI endpoints
- `main.py` builds `OrderBook`/`Gateway` and, for the postgres backend,
  `PostgresOrderBookStore`/`PostgresRequestLogStore`/`PostgresContractStore`, then
  calls `create_app()`
- `postgres_store.py`'s `Postgres*` classes wrap `hsqlimpl.DbConnection` and
  implement the `ABC`s owned by `batch_call_auction.py`, `passthrough_proxy.py`, and
  `platform_api.py`

## Mirrors

- `BidRequest` (`platform_api.py`) <-> `Bid` (`batch_call_auction.py`): same fields,
  no shared code
- `AskRequest` (`platform_api.py`) <-> `Ask` (`batch_call_auction.py`): same fields,
  no shared code
- `ContractResponse` (`platform_api.py`) <-> `Contract` (`contract_dispatch.py`):
  same fields plus an assigned `contract_id`
- `CompletionResponse`/`LogEntryResponse` (`platform_api.py`) <-> `RequestLogEntry`
  (`passthrough_proxy.py`): same fields
- `RoundClearResponse` (`platform_api.py`) <-> the
  `(tier, round_id, clearing_price, matched_volume)` pricing-dissemination event
  shape