// > open_md.py -i research/Noesis/docs/design.architecture.md

# Rules

- This document describe the code as it is, without making reference to intermediate
  PRs and how the code evolved
- It refers to:
  - The \Noesis{} protocol overview in `papers/Noesis/*.tex`
  - The implementation plan `research/Noesis/docs/plan.md`

# Overview

- `research/Noesis` contains:
  - `NoesisMarket` (`market/`)
  - `NoesisServer` (`gateway/`, `checker/`, `prober/`, and `monitor/`)
  - Shared infrastructure for the MVP components (`common/`)
  - A thin HTTP surface over the market and the proxy (`noesis_platform/`,
    `NoesisPlatform` section)
  - A runnable deployment around that HTTP surface
- The HTTP surface
  - `batch_call_auction.py` and `contract_dispatch.py`: a call-auction that matches
    buyer/seller orders for LLM inference capacity and dispatches cleared contracts
    to a fulfillment layer
  - `passthrough_proxy.py`: a minimal LLM API gateway that routes prompts to
    registered providers and logs every request/response pair
  - `platform_api.py`: a `fastapi.FastAPI` app factory that wraps both of the above
    behind HTTP endpoints, so an external caller can reach them without importing the
    Python modules directly
  - `postgres_store.py`: Postgres-backed implementations of the storage interfaces
    the three modules above define, so their in-memory state can be swapped for a
    persistent backend
  - `main.py`: process entry point that builds a module-level `app`, wired to either
    the in-memory or the Postgres backend depending on env vars, meant to be run via
    `uvicorn research.Noesis.noesis_platform.main:app`
  - `devops/`: Docker Compose deployment of `main.py`'s `app` plus a Postgres sidecar
    for local dev
- Problem solved: implements a two-sided market for LLM inference capacity
  (capability tier, latency, reliability, price), a mock fulfillment/dispatch layer
  for matched contracts, and a logging/proxy layer for LLM calls, reachable over HTTP
  by a caller outside the Python process, with an optional persistent (Postgres)
  backend and a containerized deployment path
- Key design decisions visible from the code:
  - Every stateful store (`OrderBook`'s pending orders, `platform_api`'s
    contract/round log, `Gateway`'s request log) sits behind a pluggable `abc.ABC`
    (`OrderBookStore`, `ContractStore`, `RequestLogStore`), each with an in-memory
    default and a `postgres_store.py` implementation
  - Every side effect that would be non-deterministic in a test (fulfillment outcome,
    provider network call, wall-clock time) is injected as a callable
    (`FulfillmentFunc`, `ProviderCallFunc`, `clock_func`, `rng`), so tests control it
    directly
  - Dataclasses encode the schema and validate every field with `hdbg.dassert_*` at
    construction time
  - `platform_api.py` adds no new validation logic: it reuses the same
    `hdbg.dassert_*` checks already in the three lower modules, catching the
    resulting `AssertionError` once at the app level and returning an HTTP 400
  - `main.py` selects the storage backend once, at import time, from
    `NOESIS_DB_BACKEND` (`"memory"` default or `"postgres"`); every other module is
    unaware which backend is active
- Who uses it: the unit test suites under each component's `test/` dir; an HTTP client
  wrapped in a `fastapi.testclient.TestClient` in tests, or a real HTTP client
  against `main.py`'s `app` when run via `uvicorn` or the
  `devops/docker_run/run_docker_noesis.sh` container

# Architecture (C4 Model)

## C1 (Context)

- Describes how the Noesis prototype fits with its (simulated) users and the external
  systems it integrates with, some of which are stubbed or optional
- The buyer/seller side is a test harness or an HTTP caller of
  `POST /bids`/`POST /asks`
- `External caller` is any HTTP client that reaches `NoesisMarket` and
  `NoesisServer`, gated by an `X-API-Key` header on the write endpoints
- `NoesisServer` is the real fulfillment layer described in `plan.md`'s
  `NoesisServer` section; `contract_dispatch.mock_fulfill()` is its placeholder in
  this codebase
- `LLM Providers` are the real backends `passthrough_proxy.Gateway` calls through
  `ProviderConfig.call_func`; tests inject a stand-in `ProviderCallFunc` instead of a
  network call
- `Postgres` is entirely optional: `main.py` only connects to it, and
  `postgres_store.py` is only imported, when `NOESIS_DB_BACKEND=postgres`; the
  default `NOESIS_DB_BACKEND=memory` path never touches this system

```mermaid
%%{init: {"c4": {"c4ShapeInRow": 3, "c4ShapeMargin": 60, "c4ShapePadding": 20, "diagramMarginX": 40, "diagramMarginY": 50, "personFontSize": 16, "personFontWeight": "bold", "external_personFontSize": 16, "systemFontSize": 16, "systemFontWeight": "bold", "external_systemFontSize": 16, "boundaryFontSize": 16, "messageFontSize": 14}}}%%
C4Context
  Person(buyer, "Buyer", "Submits Bids")
  Person(seller, "Seller", "Submits Asks")
  Person(caller, "External caller", "HTTP client with an API key")
  System(spacer_l, " ", " ")
  System(noesis, "Noesis", "Batch call-auction, contract dispatch, and LLM proxy")
  System(spacer_r, " ", " ")
  System(gap_a, " ", " ")
  System(gap_b, " ", " ")
  System(gap_c, " ", " ")
  System_Ext(llm_providers, "LLM Providers", "OpenAI, Anthropic, etc.")
  System_Ext(postgres, "Postgres (optional)", "Persistent backend")
  System_Ext(noesis_server, "NoesisServer", "Real fulfillment/monitoring layer")
  Rel(buyer, noesis, "Submits Bid", "OrderBook.submit_bid() / POST /bids")
  Rel(seller, noesis, "Submits Ask", "OrderBook.submit_ask() / POST /asks")
  Rel(caller, noesis, "Reads contracts, calls a model", "GET / POST")
  Rel(noesis, llm_providers, "Proxies prompt, logs response", "Gateway.call()")
  Rel(noesis, postgres, "Persists orders, contracts, rounds", "postgres_store.py")
  Rel(noesis, noesis_server, "Dispatches cleared Contract", "dispatch_contract()")
  UpdateElementStyle(spacer_l, $bgColor="transparent", $borderColor="transparent", $fontColor="transparent")
  UpdateElementStyle(spacer_r, $bgColor="transparent", $borderColor="transparent", $fontColor="transparent")
  UpdateElementStyle(gap_a, $bgColor="transparent", $borderColor="transparent", $fontColor="transparent")
  UpdateElementStyle(gap_b, $bgColor="transparent", $borderColor="transparent", $fontColor="transparent")
  UpdateElementStyle(gap_c, $bgColor="transparent", $borderColor="transparent", $fontColor="transparent")
  UpdateRelStyle(buyer, noesis, $offsetX="-40", $offsetY="-10")
  UpdateRelStyle(caller, noesis, $offsetX="40", $offsetY="-10")
  UpdateRelStyle(noesis, llm_providers, $offsetX="-60", $offsetY="30")
  UpdateRelStyle(noesis, noesis_server, $offsetX="60", $offsetY="30")
  UpdateRelStyle(noesis, postgres, $offsetY="40")
```

## C2 (Container)

- Describes the components (one dir each) of `research/Noesis`, their modules, and
  the dependencies between them
- Allowed imports between components, checked by `test/test_component_imports.py`:
  `gateway/` imports `common/`; `checker/` imports `common/` and
  `gateway/gateway_providers.py`; `prober/` coordinates `checker/`, `common/`, and
  `gateway/`; `monitor/` coordinates `checker/`, `common/`, and the provider
  modules in `gateway/`; `noesis_platform/` imports `market/` and
  `gateway/passthrough_proxy.py`; `common/` and `market/` import no other component

```mermaid
%%{init: {"c4": {"c4ShapeMargin": 90, "c4ShapePadding": 20, "diagramMarginX": 40, "diagramMarginY": 50, "personFontSize": 16, "personFontWeight": "bold", "systemFontSize": 16, "external_systemFontSize": 16, "containerFontSize": 16, "containerFontWeight": "bold", "boundaryFontSize": 16, "messageFontSize": 15}}}%%
C4Container
  title research/Noesis - Container Diagram

  Person(caller, "Caller", "Unit test harness or an HTTP client (fastapi.testclient.TestClient, or real HTTP against main.py's app)")

  Container_Boundary(market, "market/") {
    Container(auction, "batch_call_auction.py", "Python module", "OrderBook: pending Bid/Ask queue (via OrderBookStore), per-tier uniform-price clearing")
    Container(dispatch, "contract_dispatch.py", "Python module", "Contract schema, build_contracts(), dispatch to a mocked fulfillment layer")
  }

  Container_Boundary(gateway, "gateway/") {
    Container(proxy, "passthrough_proxy.py", "Python module", "Gateway: routes prompts to registered providers, logs request/response (via RequestLogStore)")
    Container(mvp_gateway, "gateway_*.py", "fastapi app", "MVP gateway: OpenAI-compatible, routes to the pinned provider, logs every call (see gateway/README.md)")
  }

  Container_Boundary(platform, "noesis_platform/") {
    Container(api, "platform_api.py", "fastapi.FastAPI app factory", "create_app(): HTTP surface over auction/dispatch and proxy, plus contract_id/round_id bookkeeping (via ContractStore) and API-key auth")
    Container(store, "postgres_store.py", "Python module", "Postgres implementations of OrderBookStore, ContractStore, RequestLogStore, plus init_schema()")
    Container(entry, "main.py", "uvicorn ASGI entry point", "Builds module-level app: selects memory/postgres backend from NOESIS_DB_BACKEND, calls create_app()")
  }

  Container_Boundary(checker, "checker/") {
    Container(checker_mod, "checker_*.py", "Python modules", "Calibrates and grades deterministic canaries; computes contract verdicts with pure logic and a Postgres store")
  }

  Container_Boundary(prober, "prober/") {
    Container(prober_mod, "prober_{runtime,worker,cli}.py", "Python modules", "Claims scheduled canary slots, calls the contracted provider, grades responses, and persists hidden evidence")
  }

  Container_Boundary(monitor, "monitor/") {
    Container(monitor_mod, "monitor_*.py", "Python modules", "Runs a bounded provider/model check without a contract and persists monitoring-only observations")
  }

  Container_Boundary(common, "common/") {
    Container(infra, "noesis_{settings,db,seed}.py", "Python modules", "Settings, async Postgres pool, migrations, demo seed")
  }

  System_Ext(noesis_server, "NoesisServer (not implemented)", "mock_fulfill() stands in for it")
  System_Ext(postgres, "Postgres (optional)", "noesis_* tables")

  Rel(caller, auction, "submit_bid() / submit_ask() / clear_round()", "direct Python")
  Rel(caller, proxy, "register_provider() / call()", "direct Python")
  Rel(caller, api, "POST /bids,/asks,/rounds/clear,/completions", "GET /health,/contracts/{id},/rounds/{tier}/latest,/logs")
  Rel(mvp_gateway, infra, "imports settings, DB pool, seed")
  Rel(checker_mod, infra, "uses schema and DB pool")
  Rel(checker_mod, mvp_gateway, "imports provider status constants")
  Rel(prober_mod, checker_mod, "uses schedules, measurement, and persistence")
  Rel(prober_mod, mvp_gateway, "uses routing, provider calls, faults, and logging")
  Rel(prober_mod, infra, "uses settings, migrations, and DB pool")
  Rel(monitor_mod, checker_mod, "uses active canaries and capability measurement")
  Rel(monitor_mod, mvp_gateway, "uses provider interface and OpenRouter adapter")
  Rel(monitor_mod, infra, "uses settings, migrations, and DB pool")
  Rel(dispatch, auction, "imports Bid, Ask, TierClearResult")
  Rel(dispatch, noesis_server, "dispatch_contract() (mocked)")
  Rel(api, auction, "imports OrderBook, Bid, Ask")
  Rel(api, dispatch, "imports build_contracts(), dispatch_contracts(), mock_fulfill()")
  Rel(api, proxy, "imports Gateway, ProviderConfig")
  Rel(store, auction, "implements OrderBookStore")
  Rel(store, api, "implements ContractStore")
  Rel(store, proxy, "implements RequestLogStore")
  Rel(store, postgres, "SELECT/INSERT/DELETE via helpers.hsql_implementation")
  Rel(entry, auction, "OrderBook(store=...)")
  Rel(entry, proxy, "Gateway(store=...)")
  Rel(entry, store, "PostgresOrderBookStore/PostgresContractStore/PostgresRequestLogStore, init_schema()")
  Rel(entry, api, "create_app(order_book, gateway, api_keys, contract_store=...)")

  UpdateRelStyle(caller, proxy, $offsetY="640", $offsetX="330")
  UpdateRelStyle(caller, api, $offsetY="980", $offsetX="-30")
  UpdateRelStyle(dispatch, noesis_server, $offsetY="-20", $offsetX="-200")
  UpdateRelStyle(store, postgres, $offsetY="20", $offsetX="60")
```

- `contract_dispatch.py` is the only lower-layer module with an internal dependency
  on another lower-layer module: it imports `batch_call_auction` (`Bid`, `Ask`,
  `TierClearResult`) to build `Contract`s from a cleared round
- `passthrough_proxy.py` has no import relationship with `batch_call_auction.py`
  /`contract_dispatch.py`; it is a standalone container with no call path to or from
  the market side
- `platform_api.py` imports all three lower modules but adds no import between them:
  it is a fourth, higher container, not a rewire of the three existing ones (see the
  `## Weaknesses and Assumptions` entry on this)
- `postgres_store.py` is the only module that imports all three of
  `batch_call_auction.py`/`contract_dispatch.py`/`platform_api.py` (to subclass their
  `*Store` `abc.ABC`s and reference their dataclasses); none of those three import it
  back, so there is no import cycle and no transitive `psycopg2` dependency for a
  caller that never selects the Postgres backend
- `main.py` is the only module that imports `postgres_store.py`, and only inside its
  `NOESIS_DB_BACKEND == "postgres"` branch (a deferred import); it is the sole place
  that decides which storage backend every other module runs against for the life of
  the process
- `monitor_runtime.run_monitoring()` is a separate orchestration path: it selects
  active calibrated canaries, calls one explicit provider/model target sequentially,
  grades each successful response, and commits each observation before continuing
- Monitoring tables have no `contract_id`, and `monitor/` cannot import `market/`;
  standalone measurements therefore cannot change contract verdicts or reputation

## C3 (Component)

- Describes the runtime call chain from order submission through logged fulfillment
  outcome, the primary multi-module flow in this codebase
- `OrderBook.submit_bid()`/`submit_ask()` and `clear_round()` no longer hold
  `List[Bid]`/`List[Ask]` state directly: they delegate to an injected
  `OrderBookStore` (`_InMemoryOrderBookStore` by default), so `clear_round()` fetches
  pending orders once via `store.get_bids()`/`get_asks()`, buckets them by `c_level`
  in Python, and calls `_match_orders_in_tier()` once per tier, then calls
  `store.clear()` unconditionally (matched orders and unmatched orders alike)
- `build_contracts()` looks up each `Fill`'s buyer in `bids` (by `buyer_id`) to pull
  `l_max`/`r_min` onto the resulting `Contract`, since a `Fill` only carries
  `n_tasks`/`c_level`/`price`
- `dispatch_contract()` mutates `contract.fulfilled` in place using whatever
  `fulfillment_func` is passed (`mock_fulfill()` by default), and returns the same
  object
- `platform_api._MarketState.clear_round()` is the orchestration point that ties the
  market side together for the HTTP API: it reads pending bids, calls
  `OrderBook.clear_round()`, mints one shared `round_id` via
  `ContractStore.next_round_id()`, builds and dispatches every contract for the
  round, persists each via `ContractStore.save_contract()`, then persists and returns
  one `RoundClearResponse` per tier via `ContractStore.save_round()`
- `passthrough_proxy.Gateway.call()` runs a parallel, currently disconnected flow:
  `call()` looks up the `ProviderConfig` for `provider_name`, times
  `provider_config.call_func(model, prompt)` with the injected `clock_func`, computes
  `cost` from `cost_per_char`, and persists a `RequestLogEntry` via the injected
  `RequestLogStore.append()`, queried later via `get_log()` / `query_log()`
- `main.py` runs once, at import time, before any request-handling flow above: it
  resolves `NOESIS_DB_BACKEND`, builds the concrete `OrderBook` / `Gateway` /
  `ContractStore` (each wired to the same store backend), and passes them to
  `platform_api.create_app()` to produce the module-level `app` object `uvicorn`
  serves

```mermaid
%%{init: {"themeVariables": {"fontSize": "18px"}, "flowchart": {"nodeSpacing": 45, "rankSpacing": 55}}}%%
flowchart LR
    submit_bid["OrderBook.submit_bid()"] --> store[("OrderBookStore<br/>(in-memory or Postgres)")]
    submit_ask["OrderBook.submit_ask()"] --> store
    store --> clear_round["OrderBook.clear_round()"]
    clear_round --> match["_match_orders_in_tier()<br/>(per c_level)"]
    match --> tier_result["TierClearResult<br/>(fills, clearing_price, unfilled)"]
    tier_result --> build["build_contracts(bids, tier_results)"]
    build --> contract["Contract<br/>(fulfilled=None)"]
    contract --> dispatch["dispatch_contract() /<br/>dispatch_contracts()"]
    dispatch --> fulfill["fulfillment_func<br/>(default: mock_fulfill())"]
    fulfill --> save["ContractStore.save_contract()<br/>(_MarketState.clear_round())"]
```

### Supply refresh flow (PR_S9)

- Before a round, `supply_feed.refresh_market_supply()` turns the OpenRouter
  endpoint catalog into one `Ask` per seller of `common/seed.yaml`:
  `parse_endpoint_quotes()` reads the per-token prices, `build_asks()` matches
  sellers to endpoints by tag and prices each ask at the worst-case D1 task cost,
  and `submit_supply()` queues them on the `OrderBook`, skipping sellers that
  already have a pending ask in the round
- `market/supply_adapter.py` is pure; the HTTP fetch and the seed file live in
  `noesis_platform/supply_feed.py`, so `market/` keeps importing nothing else
- `clear_round()` drops every order, so the refresh is re-run each round; the
  scheduler (`PR_M7`) owns catch-and-retry when OpenRouter is unavailable

```mermaid
flowchart LR
    openrouter["OpenRouter<br/>GET /models/{id}/endpoints"] --> fetch["fetch_endpoint_catalog()"]
    seed[("common/seed.yaml<br/>sellers, l_typical, r_typical")] --> load["load_seller_supply()"]
    fetch --> parse["parse_endpoint_quotes()<br/>EndpointQuote per tag"]
    parse --> build["build_asks()<br/>p_min = D1 task cost x (1 + margin)"]
    load --> build
    build --> submit["submit_supply()<br/>one Ask per seller per round"]
    submit --> store[("OrderBookStore")]
    store --> clear["OrderBook.clear_round()"]
```

### Scheduled capability-measurement flow

- `prober_cli calibrate` validates the candidate bank before any provider call,
  executes reviewed live calibration only with `--execute`, and activates a bank
  only after the minimum-survivor gate passes.
- `prober_worker.run_probe_worker()` repeatedly performs a non-blocking scheduler
  pass. PostgreSQL atomically claims `(contract_id, probe_slot)`, making concurrent
  workers safe.
- The model receives an ordinary user message. Canary identity, expected answer,
  grade, and grader version remain inside Noesis and are added only to persisted
  evidence.

```mermaid
flowchart LR
    candidate["Versioned candidate bank"] --> calibration["Live calibration<br/>3 attempts per target"]
    calibration --> gate{"Healthy = 100%<br/>Weak <= 60%<br/>Survivors >= 150?"}
    gate -->|yes| active["ACTIVE canary bank"]
    gate -->|no| draft["DRAFT / not used"]
    contract["ACTIVE contract window"] --> schedule["Secret balanced schedule"]
    active --> schedule
    schedule --> claim["Atomic probe-slot claim"]
    claim --> route["Internal contract route"]
    route --> provider["Pinned provider call"]
    provider --> measure["CapabilityMeasurer"]
    measure --> evidence["Server-only graded request evidence"]
```

### Standalone monitoring flow

- `monitor_cli` validates the bank, target, limits, and planned paid-call count by
  default; `--execute` is required before PostgreSQL or a live provider is touched
- `monitor_runtime.run_monitoring()` selects a secret family-balanced subset for
  one run and calls the target sequentially with a strict timeout
- Each observation is committed before the next provider call; provider and grader
  failures remain observation outcomes, while an orchestration failure marks the
  run `FAILED`
- The flow ends in monitoring-only tables and has no path to contract verdicts or
  seller reputation

```mermaid
flowchart LR
    active["ACTIVE canary bank"] --> select["Secret balanced selection"]
    target["Explicit provider/model target"] --> runtime["Sequential monitor runtime"]
    select --> runtime
    runtime --> provider["Provider.call() with timeout"]
    provider --> measure["CapabilityMeasurer"]
    measure --> observation[("Monitoring observation")]
    observation --> run[("Monitoring run summary")]
```
