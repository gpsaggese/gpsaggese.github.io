# Noesis Implementation Plan

## Goal

- Build a working prototype of the \Noesis{} protocol
- The \Noesis{} protocol overview is in `papers/Noesis/*.tex`
- The architecture document `research/Noesis/docs/design.architecture.md`

## Plan hierarchy

- This file is the canonical Noesis roadmap and defines the component-level PRs.
- The `# MVP Execution Plan` section at the end of this file decomposes those PRs
  into smaller implementation tasks with owners, dates, and acceptance criteria.
- `progress.md` records the current implementation status.
- `decisions.md` records the decisions and the alternatives considered.
- A checked PR is complete. An unchecked PR may be either not started or partially
  implemented; partial status is stated in that PR's section below.

## Roadmap

- **v0.1**: NoesisMarket fully in-memory and controlled from Python and a notebook,
  with no database, no API, and no cloud; the notebook shows the input requests
  (bids and asks from buyers and sellers of compute) and the result of the auction
  - [x] `PR_M1`
  - [x] `PR_M2`
  - [ ] `PR_M10`
  - [ ] `PR_M11`
  - [ ] `PR_M12`
  - [ ] `PR_M13`
- **v0.2**: NoesisMarket on a Postgres database, still controlled from Python and a
  notebook without APIs
  - [ ] `PR_M14`
  - [ ] `PR_M15`
- **v0.3**: NoesisMarket receives orders through the API and runs an auction every
  10 seconds, so an allocation is available after each round; fulfillment and
  providers are still mocked
  - [x] `PR_P1`
  - [ ] `PR_M7`
  - [ ] `PR_P6`
- **v0.4**: The `v0.3` system (API, 10-second auction, mock providers) deployed on
  the cloud
  - [ ] `PR_P2`
- **v0.5**: Create an adapter from OpenRouter to the API as demand to provide
  capacity (real providers replace the mock ones on the supply side)
  - [x] `PR_S7`
  - [ ] `PR_S9`
- **v0.6**: Implement NoesisServer where people can pay for capacity and get served
  through the API of the providers (without extra charging), with delivery measured
  against the contract and fed back into seller reputation
  - [x] `PR_S1`
  - [ ] `PR_S11`
  - [ ] `PR_S4`
  - [ ] `PR_S6`
  - [ ] `PR_M8`
  - [ ] `PR_M3`
  - [ ] `PR_P4`
- **v0.7**: Buyers can see whether a model was quietly downgraded (model nerfing)
  and what a correct task costs: a per-model quality and price-per-task time series,
  with a nerfing flag fed to the market
  - [ ] `PR_S13`
  - [ ] `PR_S14`
  - [ ] `PR_S15`
  - [ ] `PR_S16`
- Releases are cumulative: each version also requires every PR listed under the
  versions before it (e.g. `v0.4` also needs `v0.1`-`v0.3`'s PRs)

# Component Specs

## NoesisMarket

### Goal

- Build a working intelligence market: a batch call-auction that:
  - Matches buyer/seller contracts for LLM inference bundling capability tier,
    latency, and reliability guarantees
  - Dispatches cleared contracts for fulfillment
  - Feeds delivery performance back into pricing
- Background: `papers/Noesis/04_noesis_market.tex`

### Open Questions

1. Task unit for cross-provider comparability: tokens vs. wall-clock compute vs
   benchmark-normalized task-equivalent -> Project
2. Auction mechanism/frequency: uniform-price batch call auction vs. continuous
   double auction, and whether 5 min is the right cadence
3. Anti-gaming: how to stop capability misrepresentation or bid shading without a
   heavy onboarding/reputation system
4. Pricing denomination: real currency vs. synthetic task-credit?
   - Maybe supporting both a credit-card and a crypto funding rail rather than
     forcing a single choice at the protocol level
5. Batch vs. hybrid: does a spot market need to sit alongside the batch auction for
   latency-sensitive buyers -> Project
6. P2: Collusion: `papers/Noesis/04_noesis_market.tex` sec:mechanism_design_risks
   flags thin, few-seller tier buckets, notably `frontier`, as vulnerable to
   coordinated bidding/bidding rings; no mitigation is designed or planned above

## NoesisServer

### Goal

- Build a lightweight LLM API gateway (modeled on OpenRouter) that:
  - Proxies requests to multiple providers
  - Logs every prompt/response pair
  - Adds difficulty-aware routing
  - Serves as the fulfillment/monitoring layer for `NoesisMarket`
- Background: `papers/Noesis/05_noesis_server.tex`

### Open Questions

1. Is a simple pass-through logger already useful for dataset building, or does
   routing quality matter for data diversity?
2. Can request difficulty be estimated cheaply enough that difficulty-aware routing
   nets a real cost saving? -> project
3. Routing vs. fusion under a fixed budget: for a fixed per-request budget, is it
   better to route to one well-chosen model or to query several cheaper models and
   combine their answers? -> project
4. **PII/data-handling safeguards** before logging real (non-synthetic)
   prompt/response pairs: `PR_S1` defers scrubbing and stays on synthetic/test
   traffic only until this is resolved; must be answered before enabling real traffic
   logging
5. Can a distilled model match routed "best model per task" quality at a fraction of
   the cost?
6. Attribution reliability: how reliably can the server attribute a quality/latency
   shortfall to a specific provider vs. noise, so fulfillment reporting to
   `NoesisMarket` is trustworthy enough to affect pricing?
7. OpenRouter dependency risk: `PR_S7` makes `Gateway`'s real liquidity depend on one
   third party's uptime, pricing, and model catalog; is a single- upstream dependency
   acceptable for the prototype, or does it need a fallback provider before real
   (non-synthetic) traffic relies on it?
8. Fidelity vs. scope of `PR_S8`'s compatibility: targeting chat completions and
   model listing only; does divergence from OpenRouter's exact error codes/streaming
   semantics break real OpenRouter clients in practice, before advertising
   `NoesisServer` as a drop-in replacement?
9. Nerfing vs. legitimate model updates (`PR_S13`): a provider that silently swaps
   the snapshot behind a stable alias looks identical to nerfing; should a
   detected quality shift be reported as "downgraded" or as "new model, re-baseline",
   and can the server tell the two apart without the provider disclosing snapshot
   ids? -> Project

## Architecture

### Goal

- Pin down the exact inputs, outputs, and invariants of the five pluggable components
  introduced in `papers/Noesis/01_introduction.tex`
  - Matching engine
  - Capability measurement
  - Reputation and feedback
  - Router / Answer fusion
  - Pricing dissemination
- A substitute implementation, e.g., a continuous double auction in place of
  `NoesisMarket`s call auction, can then be swapped in for one component without
  touching the others
- Background: `papers/Noesis/01_introduction.tex`

## NoesisPlatform

### Goal

- Turn the `NoesisMarket`/`NoesisServer` prototypes into a service an external caller
  can actually reach:
  - A public API over both components
  - A cloud deployment target
  - A way for a buyer to fund an account with a credit card or crypto before bidding
- Unlike `NoesisMarket` and `NoesisServer`, this section is not grounded in a
  specific mechanism from `papers/Noesis/*.tex`; it is the productization layer both
  component plans assume but neither scopes

### Open Questions

1. Custody: does `NoesisMarket` hold buyer funds in escrow between funding and
   settlement, or only check a balance and settle out-of-band? (affects `PR_P3`'s
   debit timing and regulatory exposure)
2. Refunds and chargebacks: how does a credit-card chargeback interact with credit
   already spent on a matched contract?
3. Cloud target for `PR_P2`: which provider to standardize on; the roadmap lands the
   Postgres-backed market (`v0.2`) before the first public deployment (`v0.4`)
4. KYC/compliance: does accepting real-currency payments (credit card or crypto)
   trigger money-transmitter obligations that a synthetic task-credit avoids?

# PRs

- PRs are listed in the order of the roadmap milestones above, `v0.1` first
- PRs not assigned to a milestone follow, grouped by component and ordered by
  number

## v0.1: In-memory NoesisMarket with a Notebook

### `PR_M1`: [x] Minimal Batch Call-auction Simulator

- In-memory order book: bid `(N_tasks, C_level_min, L_max, R_min, P_max)` and ask
  `(N_tasks, C_level, L_typical, R_typical, P_min)` submission
- Bucket bids/asks by capability tier `C_level`; every `T = 5` min, clear each tier
  at a single uniform price (highest-bid-first vs. lowest-ask-first) until supply and
  demand cross
- Defaults:
  - Task unit = tokens
  - No anti-gaming checks
  - Single currency
  - Fixed 5-min batch cadence

### `PR_M2`: [x] Contract Schema + Dispatch to a Stubbed Fulfillment Layer

- Define the contract schema `(N_tasks, C_level, L_max, R_min, P)` from a cleared
  match
- Dispatch each cleared contract to a **mock** fulfillment interface (fixed or
  randomized pass/fail outcomes), standing in for `NoesisServer`, which has no
  implementation yet
- Log the (mocked) fulfillment result back onto the contract record
- Result: closed loop match -> contract -> dispatch -> logged outcome, mocked past
  the auction boundary; swap-in point for the real `NoesisServer` once it exists,
  done by `PR_M8` below

### `PR_M10`: [ ] Library-level Round Runner (no HTTP)

- Background: the only code that turns a cleared auction into stored contracts is
  `_MarketState.clear_round()` in `research/Noesis/noesis_platform/platform_api.py`:
  it calls `OrderBook.clear_round()`, `build_contracts()`, and
  `dispatch_contracts()`, then saves each contract and one `RoundClearResponse` per
  tier; `ContractStore` and its in-memory implementation live in the same HTTP
  module, so a caller that wants only the market must import the FastAPI layer
- Move the orchestration into `market/` as
  `run_round(order_book, contract_store, fulfillment_func) -> RoundResult`:
  - `RoundResult` is a plain dataclass: round id, per-tier clearing price and
    matched volume, contracts, and the unmatched bids and asks
  - `ContractStore` and its in-memory implementation move next to it
  - One `round_id` per round, shared by every tier cleared in it (as today)
- `platform_api.py` delegates to `run_round()` and converts `RoundResult` into its
  response models, so the HTTP behavior of `PR_P1` does not change
- `market/` still imports no other Noesis component
- Unit tests: `run_round()` on a fixture book returns the same clearing prices,
  matched volumes, and contracts that `_MarketState.clear_round()` returned before
  the move; an empty book returns a `RoundResult` with no contracts and does not
  raise; `noesis_platform/test/test_platform_api.py` passes unmodified;
  `test/test_component_imports.py` still passes
- Result: the market is driven from Python alone: submit orders, run a round, read
  contracts and prices, with no HTTP layer in the import path

### `PR_M11`: [ ] Deterministic Scenario Fixtures for Bids and Asks

- Background: `v0.1`'s notebook needs input requests that a reader can scan and
  that are the same on every run; `PR_M1`'s tests build orders inline, one case at a
  time
- Add named scenarios as plain functions returning `(bids, asks)`, each small
  enough (e.g., at most 10 orders per tier) to read as a table:
  - `balanced`: demand and supply cross and every order fills
  - `excess_demand`: some bids stay unmatched
  - `excess_supply`: some asks stay unmatched
  - `multi_tier`: two or more `C_level` tiers clear at different prices
  - `no_cross`: every bid limit price is below every ask limit price, so nothing
    matches
- Add a seeded generator `make_random_scenario(seed, n_bids, n_asks, tiers)` for
  larger demos; the same seed returns the same orders
- Scenarios are data only: no store, database, network, or clock access, so the
  in-memory (`v0.1`) and Postgres (`v0.2`) notebooks reuse them unchanged
- Unit tests: the same seed returns identical orders; each named scenario, run
  through `PR_M10`'s `run_round()`, shows the regime in its name (`balanced` fully
  matches, `excess_demand` leaves bids, `excess_supply` leaves asks, `no_cross`
  matches zero volume, `multi_tier` yields distinct clearing prices per tier)
- Result: a fixed set of readable input requests, each tied to one auction regime,
  that the notebooks and later tests share

### `PR_M12`: [ ] Auction Result Report and Plot Helpers

- Background: `TierClearResult`, `Fill`, and `Contract` objects are not readable in
  a notebook cell; the input requests and the auction result need tables and a plot
- Add pure helpers in `market/` (no I/O) that return `pandas.DataFrame`:
  - `orders_to_df(bids, asks)`: one row per order with side, tier, task count,
    limits, and limit price
  - `tier_summary_df(round_result)`: one row per tier with clearing price, matched
    volume, total bid volume, total ask volume, and unmatched volume
  - `contracts_to_df(contracts)`: one row per contract with tier, price, task
    count, buyer, seller, and the mocked fulfillment outcome
  - `unmatched_to_df(round_result)`: unmatched bids and asks with their tier
- Add `plot_supply_demand(tier, bids, asks, clearing_price)`: step-function demand
  and supply curves for one tier, with the clearing point marked
- Unit tests: golden DataFrames for each `PR_M11` scenario; per tier, matched plus
  unmatched volume equals submitted volume; the plot function returns a figure with
  the expected number of curves (structure checked, no pixel comparison)
- Result: any `RoundResult` becomes a set of tables and one plot, so the notebook
  shows the auction without custom code in its cells

### `PR_M13`: [ ] Notebook: Input Requests and Auction Result (In-memory)

- Add a Jupyter notebook for `NoesisMarket`, paired with a `.py` and with logic in a
  `*_utils.py`, following the repo's notebook conventions; one section per
  `PR_M11` scenario:
  - Build the input requests and show them as a table (`orders_to_df`)
  - Submit them to an `OrderBook` with the default in-memory store
  - Run one round with `PR_M10`'s `run_round()`
  - Show the result: tier summary, supply/demand plot per tier, contracts, and
    unmatched orders (`PR_M12`)
- Print each variable with its name next to its value (repo notebook convention)
- Label the fulfillment outcomes as mocked (`PR_M2`) so the notebook does not imply
  real delivery
- The notebook uses no database, HTTP client, environment variable, or network
- Tests: the notebook executes top to bottom, headless, with
  `helpers_root/dev_scripts_helpers/notebooks/run_nbconvert.py`, and writes the HTML
  output; the helpers it calls are covered by `PR_M11` and `PR_M12`
- Result: a reader opens one notebook and sees the input requests, the supply and
  demand per tier, the clearing price, and the contracts: the `v0.1` deliverable

## v0.2: NoesisMarket on Postgres, Controlled Without APIs

### `PR_M14`: [ ] Postgres Market State Without the API Layer

- Background: Postgres stores for the order book and for the contract and round log
  already exist in `research/Noesis/noesis_platform/postgres_store.py`, selected by
  `main.py` through `NOESIS_DB_BACKEND`; that module imports `platform_api.py` and
  `passthrough_proxy.py`, so a market-only caller using Postgres imports the HTTP
  layer and the gateway too
- Move the market stores into `market/`: `PostgresOrderBookStore` (implements
  `OrderBookStore`) and `PostgresContractStore` (implements `PR_M10`'s
  `ContractStore`), plus `init_market_schema(connection)` for the bid, ask,
  contract, and round tables; reuse the current DDL, do not redesign it
- Keep the `helpers.hsql` import gated, so an in-memory caller does not need a
  Postgres driver installed
- `noesis_platform/postgres_store.py` keeps the request-log store and imports the
  moved classes, so `main.py` and `PR_P1`'s API behave as before
- Local Postgres comes from
  `research/Noesis/devops/compose/docker-compose.noesis.yml`
- Unit tests (gated like `noesis_platform/test/test_postgres_store.py`):
  - One store-contract test suite runs against both the in-memory and the Postgres
    implementation of each store
  - `run_round()` on Postgres returns the same clearing prices, matched volumes, and
    contracts as in-memory for every `PR_M11` scenario
  - `market/` still imports no other Noesis component, and importing it works
    without a Postgres driver installed
- Result: `OrderBook(store=PostgresOrderBookStore(connection))` plus `run_round()`
  runs the market on Postgres from Python alone

### `PR_M15`: [ ] Notebook: Same Scenario on Postgres, With Persistence Across Restart

- Reuse `PR_M13`'s notebook structure and `PR_M11`/`PR_M12` helpers; add a cell
  that connects to the compose Postgres and calls `init_market_schema()`
- Parity section: run one scenario in memory and on Postgres, show both
  `tier_summary_df` tables side by side, and assert they are equal
- Persistence section: close the connection and open a new one (standing in for a
  process restart), then read back the pending orders, the contracts, and the latest
  round per tier; submit more orders, run a second round, and show that `round_id`
  increases
- Inspect section: `SELECT` the stored rows of each table, so a reader sees what
  the market wrote
- Tests: the notebook executes headless with `run_nbconvert.py` against the compose
  Postgres, marked like the other Docker-dependent tests
  (`requires_docker_in_docker`)
- Result: the same market, with its state in Postgres and surviving a restart,
  controlled without any API: the `v0.2` deliverable

## v0.3: API Orders and a 10-second Auction, Mock Providers

### `PR_P1`: [x] Public API Surface for NoesisMarket and NoesisServer

- Background:
  - `NoesisMarket` and `NoesisServer` expose Python library calls only
    (`OrderBook.submit_bid()`, `Gateway.call()`, etc.)
  - An external caller cannot reach either component today
- Wrap the existing modules behind a thin HTTP API (e.g., FastAPI):
  - `NoesisMarket`: `POST /bids`, `POST /asks`, `GET /contracts/{id}`,
    `GET /rounds/{tier}/latest` (`NoesisMarket` `PR_M4`'s pricing-dissemination feed)
  - `NoesisServer`: a passthrough completion endpoint wrapping `Gateway.call()`,
    `GET /logs`
  - Auth: a per-account API key, checked before accepting a bid/ask or a gateway call
- Result: `NoesisMarket` and `NoesisServer` are callable over HTTP by an external
  client, not just from within the same Python process
- Implementation note (`research/Noesis/noesis_platform/platform_api.py`): added an unauthenticated
  `POST /rounds/clear` beyond the list above, since `OrderBook.clear_round()` is the
  only thing that produces a `Contract`/`TierClearResult` and none of the listed
  endpoints call it; without it, `GET /contracts/{id}` and
  `GET /rounds/{tier}/latest` could never be populated by an external caller
  `GET /rounds/{tier}/latest` reads from an in-process per-tier cache populated by
  `POST /rounds/clear`, standing in for `PR_M4`'s pub/sub feed until `PR_M4` lands

### `PR_M7`: [ ] Automatic Periodic Auction Clearing (Configurable Cadence)

- Background: `batch_call_auction.py`'s `DEFAULT_BATCH_INTERVAL_MINUTES = 5` constant
  is documented as "not enforced by this module, which clears one round per
  `OrderBook.clear_round()` call and leaves scheduling to the caller"; today the only
  caller is `NoesisPlatform` `PR_P1`'s `POST /rounds/clear`, triggered
  manually/externally, not on a timer
- Add a background scheduler (e.g. an `asyncio` task or a thread loop) that calls
  `clear_round()` every configurable interval `T`, replacing manual triggering as the
  default path; keep `POST /rounds/clear` available for tests and manual/debug
  triggering
- Make `T` a per-deployment configuration value rather than the hardcoded
  `DEFAULT_BATCH_INTERVAL_MINUTES` constant, so a fast dev/demo cadence (e.g. 10 s)
  and a production cadence can both be set without a code change
- Unit tests: with an injectable/fake clock, the scheduler calls `clear_round()` at
  the expected cadence; changing `T` changes the observed cadence; a round with no
  matches does not stop the scheduler from clearing the next round
- The scheduler calls `PR_M10`'s `run_round()` (through the same code path as
  `POST /rounds/clear`), so contracts and rounds are stored exactly as for a manual
  round; fulfillment stays the `PR_M2` mock until `PR_M8` swaps it in
- Result: `NoesisMarket` clears rounds autonomously at a configured cadence instead
  of only on demand; the roadmap's `v0.3` target ("every 10 seconds an auction is run
  and an allocation is available") is `T = 10 s` on this scheduler

### `PR_P6`: [ ] Notebook: Push Orders Through the API and Watch 10-second Rounds

- Background: `v0.3` is the first release where orders arrive over HTTP and the
  auction runs on a timer (`PR_P1` plus `NoesisMarket` `PR_M7`), but nothing yet
  produces order flow or shows the rounds; fulfillment and providers are the
  `PR_M2` mock
- Add a mock order-flow client in `research/Noesis/noesis_platform/`: given a
  `NoesisMarket` `PR_M11` scenario or a seed, it posts the bids and asks with
  `POST /bids` and `POST /asks` (with an API key, as `PR_P1` requires), either all
  at once or spread over a configurable interval
- Add a Jupyter notebook, paired with a `.py` and with logic in a `*_utils.py`:
  - Start the API locally with `T = 10 s` and the in-memory backend (or Postgres,
    through `NOESIS_DB_BACKEND`)
  - Push a scenario through the API and show the input requests (`PR_M12`'s
    `orders_to_df`)
  - Poll `GET /rounds/{tier}/latest` and `GET /contracts/{id}` and show at least
    three consecutive rounds: round id, clearing price, matched volume, contracts
  - Show an order submitted between two rounds that is matched only in the next
    round
- Label the fulfillment outcomes as mocked, so the notebook does not imply real
  providers
- Unit tests: against `fastapi.testclient.TestClient` with `PR_M7`'s fake clock,
  the client's posted orders produce the same contracts as the library-level
  `run_round()` on the same scenario; successive timer ticks produce increasing
  `round_id`s; a request without a valid API key is rejected and posts no order
- Notebook test: executes headless with
  `helpers_root/dev_scripts_helpers/notebooks/run_nbconvert.py`
- Result: a reader sees orders pushed over the API and an allocation after every
  10-second round, with mocked providers: the `v0.3` deliverable

## v0.4: Cloud Deployment

### `PR_P2`: [ ] Cloud Deployment

- Containerize `PR_P1`'s API into a single Docker image, following this repo's
  existing Docker template conventions (`class_project/project_template/Dockerfile*`)
- Deploy to a cloud target:
  - A single-node container service, e.g. AWS ECS, Fly.io, or Render, to start
  - Kubernetes deferred until there is more than one process to orchestrate
- Result: a `NoesisMarket`/`NoesisServer` instance reachable at a public URL
- TODO(gp): Read the documentation about how to release a product container
  ./docs/tools/dev_system/all.devops_docker_auto_release.explanation.md
  ./docs/tools/dev_system/all.devops_docker.how_to_guide.md
  ./docs/tools/dev_system/all.devops_docker.reference.md
  ./docs/tools/dev_system/all.docker_optimizer_container.how_to_guide.md
  ./docs/tools/docker/all.docker.tutorial.md
  ./docs/tools/docker/all.dockerized_flow.explanation.md

## v0.5: OpenRouter Adapter, Real Providers

### `PR_S7`: [x] Real Provider Liquidity Via an OpenRouter-backed `ProviderConfig`

- Implemented by PR #597. The gateway pins an OpenRouter provider, disables
  fallbacks, verifies the served provider, and records usage, cost, and latency.

- Background: `05_noesis_server.tex` models `NoesisServer` "on multi-provider routers
  such as OpenRouter" precisely because a router like OpenRouter already pools dozens
  of providers and models behind one API key
  - `ProviderConfig.call_fn` today wraps only test stand-ins, so `Gateway` has no
    real liquidity yet
- Add an `OpenRouterProviderConfig` (or a `call_fn` factory) whose `call_fn`:
  - Forwards `(model, prompt)` to OpenRouter's real chat-completions endpoint
    (`model` formatted as `"<upstream_provider>/<model>"`, e.g `"openai/gpt-4o"`)
  - Parses the response back into the raw text `Gateway.call()` returns today
  - Replaces `PR_S1`'s placeholder `cost_per_char` with the real per-token cost
    OpenRouter reports in `usage`
- One registered `OpenRouterProviderConfig` covers every model OpenRouter lists
  (`GET /api/v1/models`), so `Gateway` gets real multi-provider liquidity from a
  single integration instead of one bespoke `ProviderConfig` per upstream provider
- Result: `Gateway` is backed by real, multi-provider liquidity instead of test
  stand-ins; once wired to `NoesisMarket`, a cleared ask can be fulfilled by an
  actual OpenRouter call in place of `mock_fulfill()`

### `PR_S9`: [ ] OpenRouter Capacity as Market Supply (Auto-ask Adapter)

- Background:
  - `PR_S7` gives `Gateway` real OpenRouter-backed capacity and pricing
    (`GET /api/v1/models`), but nothing in the plan turns that capacity into
    `NoesisMarket` asks
  - `POST /asks` (`NoesisPlatform` `PR_P1`) still requires a human/manual submission
    per seller, so OpenRouter's capacity is invisible to the auction as supply
- Add an adapter that reads OpenRouter's model catalog and `usage`-reported pricing
  from `PR_S7`'s `OpenRouterProviderConfig` and periodically submits/refreshes one
  ask per `(model, tier)` into `NoesisMarket`'s order book (via
  `OrderBook.submit_ask()` directly, or `POST /asks` once deployed), instead of a
  manual ask per model
- Unit tests: given a fixture OpenRouter model catalog, the adapter submits one ask
  per listed model with tier/price derived from the catalog; a model that drops out
  of a later catalog refresh has its stale ask withdrawn or not renewed
- Result:
  - OpenRouter's capacity participates in `NoesisMarket` auctions as supply without a
    manual ask per model

## v0.6: NoesisServer: Pay for Capacity, Measured Fulfillment

### `PR_S1`: [x] Minimal Passthrough Proxy with Logging

- Support 2-3 providers behind one API
- Define the storage schema for prompt, response, and metadata (provider, model,
  latency, cost)
- Log raw, unscrubbed prompt/response pairs for now, scoped to synthetic/test traffic
  only; add PII scrubbing as a follow-up before any real non-synthetic traffic is
  logged (tracked as an open question below)
- Result: every request/response pair is logged and queryable

### `PR_S11`: [ ] Capability-measurement Estimator (hat-c(x))

- Status (2026-10-17): PR #604 established versioned normalizers, exact-match
  grading, bank validation, and secret balanced selection. The follow-up adds the
  300-question candidate bank, live calibration/activation command, generated
  findings, and scheduled probing. The reviewed paid calibration run remains open.
- Background: `papers/Noesis/01_introduction.tex` Table~tab:components lists
  capability measurement as one of the five pluggable components, instantiated in the
  paper as "a verifier model, benchmark probe, or reference-model agreement," with a
  hosted evaluation provider (e.g., artificialanalysis.ai) as the pluggable
  alternative; `05_noesis_server.tex` sec:fulfillment_monitoring's success indicator
  (Equation~eq:success_indicator) needs `hat c(x)` as an input, which no PR above
  builds: `PR_S4`/`PR_S6` currently take `measured_reliability` as already computed
- Define a pluggable capability-measurement interface. For the MVP, instantiate it
  with calibrated deterministic benchmark probes whose expected answers can be
  graded locally. Reference-model agreement remains a later interchangeable
  implementation when deterministic grading is not suitable.
- Unit tests: deterministic answer normalization and grading cover supported answer
  formats; invalid bank records are rejected; balanced selection is reproducible
  for the checker while remaining secret from sellers.
- Result: `NoesisServer` `PR_S4`/`PR_S6` gain a real (not stubbed) source for
  `capability_ok(x)`; should land before `PR_S4`/`PR_S6` are considered complete

### `PR_S4`: [ ] Fulfillment Monitoring Wired to the Market (stubbed Both Sides)

- Status (2026-10-12): partially implemented. PR #597 added contract binding,
  request logging, and provider attribution. PR #604 added seller-attributable
  evidence, separate latency/quality assessments, verdicts, and persistence. The
  prober, window-closing worker, and market-facing `VerdictSink` remain open.
- Background: `papers/Noesis/05_noesis_server.tex` sec:fulfillment_monitoring
  Equation~eq:success_indicator defines per-request success as the conjunction
  `capability_ok(x) and latency_ok(x)`, and `measured_reliability`
  (Definition~measured_reliability) is the mean of that joint indicator over the
  fulfillment window
- Background: the storage schema needs the `contract` field from
  sec:gateway_architecture (`PR_S1` shipped without it); add it, nullable for
  pre-market traffic, so requests can be grouped by `kappa`
- MVP specialization: assess latency reliability and deterministic canary quality
  separately. Combining them hides sparse quality failures inside a larger volume
  of normal buyer traffic.
  - Buyer requests provide latency evidence; canary latency is excluded by default
    because short synthetic prompts would bias buyer-experienced latency.
  - Deterministic canaries provide quality evidence through `PR_S11`.
  - Seller timeouts and upstream errors count as failures. Buyer errors, gateway
    errors, spend-cap refusals, and attribution mismatches are excluded.
  - Either sufficiently sampled metric can fail the contract. Passing requires
    sufficient evidence for both metrics; otherwise the verdict is `INSUFFICIENT`.
  - Checker/configuration failures produce `INSUFFICIENT` and never penalize the
    seller.
- Report violations via a **mocked** market-facing interface, matching the mock
  fulfillment interface on the `NoesisMarket` side (its `PR_M2`): real integration
  is deferred until both plans reach this point together
- Unit tests: violation correctly flagged/not-flagged against fixture contracts and
  measured outcomes; mocked report call invoked with the right payload
- Result: closed loop between a matched contract and its measured fulfillment, real
  once both sides swap the mock for the real interface

### `PR_S6`: [ ] Statistical Fulfillment Violation Test

- Status (2026-10-12): the pure statistical and verdict logic landed in PR #604.
  Runtime window orchestration and offline replay validation remain open.
- Background: `papers/Noesis/05_noesis_server.tex`'s fulfillment monitoring section
  notes that `PR_S4`'s naive rule, flagging $\kappa$ whenever
  `measured_reliability < R_min`, conflates genuine under-delivery with sampling
  noise on a small fulfillment window; the paper proposes a one-sided hypothesis test
  instead
- Use a one-sided **upper** confidence bound for each of `PR_S4`'s latency and
  quality rates. Flag a metric only when even its optimistic upper bound is below
  the promised rate. Use an exact Clopper-Pearson upper bound for fewer than 10
  observations and Wilson otherwise.
- Unit tests: small or borderline samples are not failed while their upper bound
  still reaches the promise; clear under-delivery with enough evidence is failed;
  all-success evidence is never failed.
- Result: fewer false-positive violation reports feeding `NoesisMarket` `PR_M3`'s
  reputation update, addressing the "attribution reliability" open question below

### `PR_M8`: [ ] Real Fulfillment Via NoesisServer's Gateway (swap the `PR_M2` Mock)

- Background: `contract_dispatch.py`'s module docstring says its mocked fulfillment
  layer "stands in for `NoesisServer` ... which does not exist yet"; that is no
  longer true once `NoesisServer` `PR_S1` (passthrough proxy) and `PR_S7` (real
  OpenRouter-backed liquidity) exist, but no PR currently swaps `mock_fulfill()` for
  a real call: `NoesisServer` `PR_S7`'s own Result line defers this to
  "`NoesisMarket` (`PR_M3`/`PR_M4` above)", and neither of those PRs performs a
  fulfillment call
- Replace `dispatch_contract()`'s call to `mock_fulfill()` with a real call through
  `NoesisServer`'s `Gateway.call()`, translating each `Contract`'s
  `(n_tasks, c_level, ...)` into one or more Gateway calls at the contract's tier,
  and log the real measured outcome (success/failure, latency) back onto the contract
  in place of the mocked pass/fail; keep `mock_fulfill()` available behind a flag for
  tests that should not depend on a live/fixture `Gateway`
- Unit tests: with a fixture `Gateway` in place of the real one, a dispatched
  contract's `fulfilled`/latency fields are set from the fixture's returned outcome,
  not from `random.random()` per `DEFAULT_FULFILLMENT_SUCCESS_RATE`
- Result: a cleared `NoesisMarket` contract is actually served by `NoesisServer` end
  to end, required before the roadmap's `v0.6` target ("get served through the API of
  the providers") is true rather than simulated

### `PR_M3`: [ ] Reputation and Pricing Feedback Loop

- Background: `papers/Noesis/04_noesis_market.tex` sec:reputation
- Feed logged fulfillment outcomes (from `PR_M2`, mocked or real) into per-seller
  eligibility and a pricing adjustment for future auction rounds
- Sellers whose measured reliability drops below `R_min` on repeated contracts face
  one of two remediations:
  - Priced out or excluded from subsequent rounds
  - Per sec:reputation's alternative remediation, required to submit asks at a lower
    tier reflecting the downgraded assessment of their actual capability, instead of
    outright exclusion
- Result: sellers that under-deliver lose eligibility/priority, or are downgraded a
  tier, in later auctions

### `PR_P4`: [ ] No-charge Credit Ledger for Bid Gating

- Background: the roadmap's `v0.6` explicitly scopes "No charge", but the only PR
  that gates `POST /bids` on a balance is `PR_P3`, which requires real Stripe/crypto
  payment rails; `v0.6` needs the balance-gating _mechanism_ without collecting real
  payment yet
- Add a per-account credit ledger (in-memory, or datastore-backed once `PR_P2` lands)
  seeded with a fixed/free grant of task-credit and no real payment collected; gate
  `POST /bids` (`PR_P1`) on `account_balance >= n_beta * p_beta` exactly as `PR_P3`
  describes, debiting only once a bid is matched into a contract
- Unit tests: a new account starts with the seeded free-credit balance; a bid within
  balance is accepted and debits on match; a bid exceeding balance is rejected before
  reaching the auction
- Result: `POST /bids` is balance-gated end to end with zero real payment collected,
  satisfying the roadmap's `v0.6` "No charge" scope; `PR_P3` upgrades the same gate
  to a real-money funding rail when charging for real is in scope

## v0.7: Model-nerfing Detection and Cost per Task

### `PR_S13`: [ ] Standalone Monitoring Design and Boundaries

- Specify the model-quality monitor before implementation
- Map the existing `PR_S11` bank, grader, capability measurer, provider interface,
  and calibration runtime to the reusable monitoring path
- Keep the monitoring path independent of market orders, contracts, verdicts, and
  reputation writes
- Split implementation into `PR_S14`-`PR_S16`, with acceptance criteria and
  non-overlapping ownership in `spec.PR_S13.md`
- Result: an approved, reviewable implementation plan for issue #684 with no runtime
  behavior change

### `PR_S14`: [ ] Standalone Model Monitor MVP

- Add an explicit one-shot command that probes a configured `(provider, model_id)`
  without creating a market contract
- Reuse `PR_S11`'s versioned canary bank, deterministic grader,
  `CapabilityMeasurer`, and `Provider` interface instead of duplicating them
- Store monitoring runs and observations separately from contract evidence
- Default to a no-spend validation mode; require an explicit flag for live calls
- Unit tests: a fake provider produces reproducible persisted observations, and no
  market, contract, verdict, or reputation API is called
- Result: one model can be checked on demand, independently of `NoesisMarket`

### `PR_S15`: [ ] Recurring Monitoring Windows and Metrics

- Schedule repeated standalone runs into non-overlapping windows for each
  `(provider, model_id, probe_set_version)` series
- Add bounded retries, a per-window call budget, idempotent claims, and graceful
  worker shutdown
- Aggregate quality, availability, p50/p95 latency, token use, `cost_per_task`, and
  `cost_per_correct_task` for each closed window
- Unit tests: duplicate workers cannot execute one slot twice; insufficient windows
  remain explicit; provider errors do not stop later windows
- Result: a queryable longitudinal record of model performance and cost

### `PR_S16`: [ ] Nerfing Detection, Reports, and Paper Evidence

- Compare each closed window with a baseline from the same
  `(provider, model_id, probe_set_version)` series
- Require statistical significance, a minimum practical quality drop, and
  consecutive flagged windows before returning `NERFED`
- Report `OK`, `NERFED`, or `INSUFFICIENT` through a CLI table and reproducible
  plots
- Run healthy, latency-fault, price-change, and weaker-model-swap experiments with
  fixed seeds; add measured results and limitations to the probing paper
- Keep market-facing events, reputation changes, dashboards, and alert delivery out
  of this PR until monitoring behavior is validated
- Result: evidence that the standalone monitor detects an injected model downgrade
  without flagging a healthy provider

## Unscheduled: NoesisMarket

### `PR_M4`: [ ] Pricing Dissemination Feed

- Background: `papers/Noesis/02_market_design.tex` lists pricing dissemination as one
  of the protocol's five pluggable components: each round's cleared price $p^*(c, t)$
  per tier is a useful signal beyond the bidders and sellers of the round that
  produced it
- Publish each round's per-tier `(tier, round_id, clearing_price, matched_volume)`
  outcome (from `PR_M1`'s `clear_round()`) to subscribers through a simple pub/sub
  interface (e.g., an in-memory event bus or callback registry), so a buyer timing a
  future bid, another market maker, or a monitoring dashboard does not need to poll
  `NoesisMarket` or wait for the next round to see the previous one's price
- Unit tests: a subscriber registered before a round receives exactly one event per
  cleared tier with the correct fields; a subscriber registered after a round does
  not receive past events by default, but can retrieve them from a bounded history
  buffer
- Result: cleared prices are queryable/subscribable independent of winning or losing
  a match; a real-time push feed is the default implementation here, with an on-chain
  event log flagged in `papers/Noesis/08_decentralized_extension.tex` as the
  pluggable alternative

### `PR_M5`: [ ] Cross-tier Compatibility (generalized Bucketing)

- Background: `papers/Noesis/04_noesis_market.tex`'s Remark on tier generalization
  notes that a bid's compatibility definition already allows a higher-tier ask to
  satisfy a lower-tier bid ($c_\alpha \succeq
  c_\beta$), so "a full implementation
  would ... let a tier-$c$ bucket draw on asks from tier $c$ and above"
- Extend `PR_M1`'s per-tier compatibility-graph construction, today an exact-string
  `C_level` match per `design.architecture.md`'s Weakness 2, to build each tier's bucket
  from bids at tier $c$ against asks at tier $c$ and every tier above it, while
  keeping the existing single uniform-price-per-bucket clearing rule
- Unit tests: a bid at tier `cheap` is filled by an ask at tier `frontier` when no
  `cheap`-tier ask is available and the frontier seller's limit price clears the
  `cheap` bucket; a rational `frontier`-tier seller with a marginal cost above the
  `cheap` clearing price is not drawn into serving `cheap` demand
- Result: matched volume increases relative to `PR_M1`'s exact-tier-only baseline
  without introducing a new price axis, closing `design.architecture.md`'s Weakness 2

### `PR_M6`: [ ] Commit-reveal Blind-bid Auction Simulation (exploratory)

- Exploratory/research PR, lighter bar than `PR_M1`-`PR_M5`: no real cryptography or
  on-chain settlement, tests cover protocol plumbing (commit is rejected without a
  later matching reveal, front-running is or is not blocked), not a security audit
- Background: `papers/Noesis/08_decentralized_extension.tex`'s blind-bid definition
  wraps a bid/ask in a commit-reveal scheme, commit a hash
  $h_\beta = H(\beta \,\|\, \nu_\beta)$ during a commit window, disclose
  $(\beta, \nu_\beta)$ during a reveal window, to prevent a party that sees a pending
  bid from front-running it (e.g., submitting a marginally better ask to snipe a
  favorable price)
- Wrap `PR_M1`'s `OrderBook` with a commit window, in which bidders/askers post a
  hash commitment, followed by a reveal window, in which they disclose the order
  tuple and salt; only revealed orders matching a posted commitment are passed into
  `clear_round()`
- Unit tests: an order revealed without a matching prior commitment is rejected; a
  simulated "sniper" that observes revealed orders and submits a new order before the
  reveal window closes is measurably blocked when the reveal window is a single tick,
  and not blocked when it spans multiple ticks, reproducing the mempool-monitoring
  risk the paper flags as still open
- Result: a measured front-running-resistance comparison between the plain `PR_M1`
  order book and commit-reveal; stake-backed slashing (the paper's staked-ask
  definition) and real on-chain deployment are out of scope for this PR

### `PR_M9`: [ ] Scored Compatibility (exploratory)

- Background: `papers/Noesis/09_open_questions.tex` sec:open_questions_market item 3,
  "Compatibility versus scoring," asks whether a scored compatibility measure,
  admitting partial matches at a discount, could recover volume that `PR_M1`'s hard
  eligibility filter (`papers/Noesis/04_noesis_market.tex` Definition~compatibility)
  leaves unmatched, without requiring an explicit exchange rate between quality and
  price
- Exploratory/research PR, lighter bar than `PR_M1`-`PR_M5`: add an alternative,
  opt-in compatibility function that returns a graded score in `[0, 1]` instead of
  `PR_M1`'s boolean `beta ~ alpha` test, and a price discount proportional to the
  shortfall on latency or reliability, gated behind a flag so `PR_M1`'s
  hard-constraint behavior stays the default
- Unit tests: a bid/ask pair that fails `PR_M1`'s hard test (e.g., ask latency
  narrowly above the bid's bound) is matched under the scored variant at a discounted
  price; the discount is zero, and matched volume matches `PR_M1`, when every ask
  exactly meets or exceeds the bid's terms
- Result: a measured comparison of matched volume and average discount between
  `PR_M1`'s hard-constraint baseline and the scored variant, answering
  `09_open_questions.tex` item 3 empirically rather than leaving it open

### `PR_M16`: [ ] Economic Surplus: Noesis vs a Posted-price Baseline

- Background: theory says a double-sided market allocates better than posted
  prices; we want the number for serving tokens through `NoesisMarket` versus a
  posted-price router in the style of OpenRouter, as proof that Noesis is better
  and as input to marketing (`PR_M17`)
- Add pure functions in `market/` (no I/O), specified in
  `research/Noesis/docs/spec.PR_M16.md`:
  - `compute_surplus(bids, asks, fills)`: consumer, producer, and total surplus,
    maximum surplus, and deadweight loss, with `Bid.p_max` as willingness to pay
    and `Ask.p_min` as seller cost
  - `run_posted_price_round(bids, asks, markup)`: each seller posts
    `p_min * (1 + markup)` and each bid goes to the cheapest eligible ask
  - `make_surplus_scenario(seed, ...)`: seeded books with overlapping bid and ask
    prices, so some pairs cross and some do not
  - `compare_mechanisms(bids, asks, markup)`: both mechanisms on the same book
- Unit tests: a single crossing pair gives total surplus `n_tasks * (p_max - p_min)`;
  Noesis total equals maximum surplus over 30 seeds; baseline total at `markup=0`
  is at most Noesis total and does not increase with `markup`; every fill satisfies
  `p_min <= price <= p_max`
- Result: one table of surplus per mechanism on a reproducible book; the result
  assumes truthful bids and depends on `markup`, so `PR_M17` reports a range

### `PR_M17`: [ ] Notebook: Surplus Gain and Marketing Summary

- Background: `PR_M16` gives the numbers for one book; the claim needs a sweep and a
  story for people who do not know market theory
- Add a notebook, paired with a `.py` and with logic in a `*_utils.py`, following
  the repo's notebook conventions:
  - Sweep `markup`, demand-to-supply ratio, price spread, and tier count with
    several seeds each, and report the mean gain with a confidence interval
  - Plot surplus gain versus `markup` and versus the demand-to-supply ratio
  - State every assumption (willingness to pay and cost distributions, baseline
    routing rule, truthful bids) next to the result
  - Write a plain-language summary for marketing: buyers and sellers together get
    X% more value through Noesis; do not describe OpenRouter as centrally planned
- TODO(gp): anchor the seller cost distribution and the default `markup` on real
  OpenRouter price data, and check OpenRouter's actual routing rule
- Tests: the notebook executes top to bottom, headless, with
  `helpers_root/dev_scripts_helpers/notebooks/run_nbconvert.py`
- Result: a reproducible, quantified surplus gain with its assumptions, and a short
  text that `marketing_plan.md` can reuse

## Unscheduled: NoesisServer

### `PR_S2`: [ ] Routing Policy + Cost/quality Measurement

- Add a simple routing policy (e.g., a task classifier): route to a cheap or fast
  model vs. a stronger model
- Measure cost and quality against an always-call-the-strong-model baseline, using
  logged data from `PR_S1`
- Unit tests: routing policy selects the expected model per test-case classification;
  cost/quality metrics computed correctly on a fixture log
- Result: a measured cost/quality frontier for at least one routing policy

### `PR_S3`: [ ] Difficulty Estimator + Distillation Experiment

- Exploratory/research PR, lighter bar than `PR_S1`/`PR_S2`: tests cover pipeline
  plumbing (data loads, estimator trains, metrics compute), not a model quality bar
- Train `q(prompt) -> required capability level` from logged agreement between cheap
  and strong models (`PR_S2`'s routing data)
- Explore distilling a small model from the collected dataset
- Result: a difficulty-aware router with a measured saving, and a first distillation
  experiment with reported (not necessarily strong) results -> Project

### `PR_S5`: [ ] Answer Fusion Prototype

- Background: `papers/Noesis/05_noesis_server.tex`'s answer-fusion section describes
  fusion as the complementary lever to routing: instead of picking one tier per
  request, fan a request out to a set of providers $S(x)$ and combine their answers
  through an aggregator $g$ (majority vote, a verifier model, or a learned combiner);
  a high-`R_min` contract may only be economically served through fusion, while a
  low-price, low-`R_min` contract favors `PR_S2`'s routing instead
- Extend the `PR_S1` gateway's `Gateway.call()` with a `call_fused()` that:
  - Dispatches the same prompt to every provider in `S(x)`
  - Collects responses
  - Applies a pluggable aggregator (majority vote / exact match to start; a
    verifier-model or learned combiner deferred)
- Unit tests: the majority-vote aggregator picks the modal response on a fixture with
  a clear majority; ties are broken deterministically; cost accounting sums the cost
  of every fanned-out call, not just one
- Result: a working fan-out/fan-in path measured against a single-provider baseline
  on cost and on a simple correctness proxy (agreement with a fixed reference
  answer), giving a first cost/ reliability trade-off for fusion, analogous to
  `PR_S2`'s cost/quality frontier for routing -> Project

### `PR_S8`: [ ] OpenRouter-compatible API Interface for `NoesisServer`

- Status (2026-10-12): partially implemented by PR #597. The gateway supports
  non-streaming `/v1/chat/completions`, `/v1/models`, Bearer authentication, and an
  unmodified OpenAI Python SDK client. Exact `/api/v1` route, streaming, and error
  parity remain open before this can be called a complete OpenRouter replacement.
- Background:
  - `NoesisPlatform` `PR_P1` sketches a bespoke HTTP wrapper around `Gateway.call()`
    without specifying its wire format
  - Adopting OpenRouter's own REST contract instead means any existing OpenRouter
    client (SDK, LangChain provider config, curl script) can point its `base_url` at
    a `NoesisServer` deployment and work unmodified
- Add `POST /api/v1/chat/completions` matching OpenRouter's
  - Request shape

    ```
    (`model: "<provider>/<model>"`, `messages`, `stream`)
    ```

  - Response shape

    ```
    (`id`, `choices[].message`, `usage.{prompt,completion,total}_tokens`)
    ```

    translating to/from `Gateway.call()`'s `(provider_name, model, prompt)` signature
    internally
- Add `GET /api/v1/models` mirroring OpenRouter's model-listing endpoint, sourced
  from `Gateway`'s registered `ProviderConfig`s (name and, once `PR_S7` lands, a real
  per-token price instead of `PR_S1`'s placeholder)
- Auth: `Authorization: Bearer <api_key>` header, OpenRouter's own convention,
  checked the same way as `NoesisPlatform` `PR_P1`'s per-account API key
- Unit tests: a request built with an OpenRouter client's request shape round-trips
  through `Gateway.call()` and returns an OpenRouter-shaped response on a local test
  server; an unknown `model` string is rejected with OpenRouter's error-response
  shape, not a bare 500
- Result:
  - `NoesisServer` is a drop-in replacement for OpenRouter from the caller's point of
    view
  - Supersedes the wire format of `NoesisPlatform` `PR_P1`'s passthrough-completion
    endpoint once this lands

### `PR_S10`: [ ] Request Caching for Identical/Near-duplicate Requests

- Background: `papers/Noesis/01_introduction.tex` describes `NoesisServer` as caching
  "responses to identical or near-duplicate requests to avoid paying a provider twice
  for the same answer"
- Add an exact-match cache keyed on `(provider, model, prompt)` in front of
  `Gateway.call()`, with a pluggable near-duplicate match (e.g., an
  embedding-similarity threshold) as a stretch goal behind the same interface
- Result:
  - Repeated/duplicate traffic no longer pays a provider twice
  - Measured cache-hit rate on a synthetic traffic fixture with a controlled
    duplication rate -> Project

### `PR_S12`: [ ] Opt-in Distillation Consent and Anonymization Gate

- Background: `05_noesis_server.tex` sec:distillation distinguishes the distillation
  corpus `D` from `PR_S1`'s passive logging: inclusion requires the customer to opt
  in and requires prompts/responses to be anonymized to remove customer- and
  end-user-identifying content before being added, in exchange for a preferential
  rate; this is a stricter, separate gate from `PR_S1`'s raw-log PII-scrubbing open
  question below
- Add a per-account opt-in flag checked before a logged prompt/response pair is added
  to `D` (`PR_S3`'s training corpus), and an anonymization step (strip account/user
  identifiers) applied on the way in
- Unit tests: a non-opted-in account's traffic never appears in `D` even though it is
  logged by `PR_S1`; an opted-in account's traffic appears only after anonymization
  strips its identifying fields
- Result: `PR_S3`'s distillation experiment can be pointed at `D` built under real
  consent, ahead of the PII open question being fully resolved for raw logging

## Unscheduled: NoesisPlatform

### `PR_P2b`: [ ] Use Postgres Backend

- Status (2026-10-12): partially implemented. PR #597 persisted gateway contracts,
  request logs, attribution, and fault metadata. PR #604 added checker policies,
  windows, probe claims, immutable evidence, and verdicts. Persistent market order
  books and remaining component stores are still open.
- Roadmap (2026-10-13): `NoesisMarket`'s share of this PR (order book, contracts,
  rounds) is specified as `NoesisMarket` `PR_M14`/`PR_M15` in `v0.2`, driven
  without the API; what stays here is selecting those stores, plus the request log,
  for the deployed API
- Externalize the in-memory state of `NoesisMarket` and of `NoesisServer` assume
  (order book, contract log, request log) to a real datastore (e.g Postgres or Redis)
  - A cloud deployment cannot rely on a single long-lived process the way the current
    test suites do
- Result: a `NoesisMarket`/`NoesisServer` backed by persistent storage instead of the
  current in-process `List`/`Dict` state
- TODO(gp): Read how to inject a Postgress instance in the container
  - /Users/saggese/src/csfy1/datapull/im_lib_tasks.py
  - /Users/saggese/src/csfy1/datapull/test/test_im_lib_tasks.py
  - Data605/tutorials/tutorial_postgres/tutorial_postgres.md

### `PR_P3`: [ ] Buying Credits (credit Card and Crypto)

- Background: resolves `NoesisMarket` open question 4 (pricing denomination)
  operationally, by supporting both a real-currency and a crypto funding rail instead
  of forcing a single choice
- Add a funding flow so a buyer can pre-purchase task-credit before submitting a bid:
  - Credit card: a hosted checkout (e.g. Stripe Checkout) that credits the buyer's
    account balance on a successful payment webhook
  - Crypto: a deposit address per account, or the escrow contract sketched in
    `08_decentralized_extension.tex`'s staked-ask design, that credits the account
    once a deposit transaction confirms
- Gate `POST /bids` (`PR_P1`) on `account_balance >= n_beta * p_beta`, debiting the
  balance only once a bid is matched into a contract (Definition~contract), not at
  submission time
- Unit tests: a mocked card-webhook credits the right account for the right amount
  and is idempotent on webhook retry; a mocked on-chain deposit event credits the
  account once, not once per confirmation; a bid exceeding the account's balance is
  rejected before reaching the auction
- Result: a buyer can fund an account through either rail and see the balance gate
  bid submission

### `PR_P5`: [ ] Written Interface Contracts for the Five Pluggable Components

- Background: each pluggable component today is only sketched in prose (the paper's
  own admission in sec:open_questions_cross); concretely, each of the following fixed
  one instantiation without a written contract for what a substitute must satisfy:
  - `NoesisMarket` `PR_M1`/`PR_M5` (matching engine)
  - `NoesisServer` `PR_S11` above (capability measurement)
  - `NoesisMarket` `PR_M3` (reputation and feedback)
  - `NoesisServer` `PR_S5` (answer fusion)
  - `NoesisMarket` `PR_M4` (pricing dissemination)
- Write one interface per component (e.g., a Python `Protocol` or abstract base
  class) specifying required inputs, outputs, and invariants:
  - Matching engine (`bids, asks -> contracts`)
  - Capability measurement (`prompt, response -> tier`, per `NoesisServer` `PR_S11`)
  - Reputation and feedback (`fulfillment history -> score`)
  - Answer fusion (`responses -> answer`, per `NoesisServer` `PR_S5`)
  - Pricing dissemination (`cleared round -> published event`, per `NoesisMarket`
    `PR_M4`)
- Retrofit the existing concrete implementations to satisfy the written interface, as
  a regression check that the interface is not just aspirational:
  - `NoesisMarket` `PR_M1`'s call auction
  - `NoesisServer` `PR_S11`'s reference-model-agreement estimator
  - `NoesisMarket` `PR_M3`'s exponential-decay update
  - `NoesisServer` `PR_S5`'s majority-vote aggregator
  - `NoesisMarket` `PR_M4`'s pub/sub feed
- Unit tests: a second, deliberately trivial implementation of each interface (e.g.,
  a random-price matching engine, a constant-tier capability estimator) can be
  substituted in the existing test suite for each component without modifying tests
  that exercise the other components
- Result: the pluggability claimed in `01_introduction.tex` sec:modularity is
  enforced by tests, not only documented in prose

## Conventions

- Code: `.claude/skills/coding.rules.md`
- Tests: `.claude/skills/testing.rules.md`

# MVP Execution Plan

<!-- Public, technical version. Keep business/funding/IP material out of this repo. -->

- **Date**: 2026-09-29
- **Canonical roadmap**: the Roadmap and PRs sections above
- **Purpose**: detailed implementation tracker and acceptance criteria for the
  canonical component PRs
- **Requirements**: the MVP Claim, Users and Scope, Market and Reputation Rules,
  Success Criteria, and Risks sections below
- **Target**: recorded demo that meets all Success Criteria below by **Fri 2026-12-11**
  (~11 weeks, three people part-time)
- **Owners (proposed)**:
  - **J** = Javin Ahuja: gateway + metering math (critical path)
  - **G** = Giles Greene: market, reputation
  - **L** = Lucas: canaries, calibration, prober, simulator, fault injector, tooling,
    dashboard
  - New contributors pick tasks from the Task Breakdown below (see
    `onboarding.md`)
- **Cadence**: GP's rule is to make progress every day. One commit or one written
  finding per person per day. Short async standup (done / next / blocked). Friday
  demo of whatever runs.

## MVP Claim

- **Problem**: buyers purchase LLM inference by the token at a price the provider
  sets. Nothing checks that the provider delivered the promised speed and quality
  - Models can be quietly quantized or swapped, and endpoints can be slow or flaky,
    with nobody held accountable
  - Routers (OpenRouter and similar) choose providers by price and uptime. They
    don't check delivered quality against a contract
- **The one thing to prove**: we can detect when an AI provider fails to deliver the
  speed or quality it promised, with a known false-alarm rate, and feed that back so
  under-delivering providers lose future deals
  - Matching buyers to sellers is a solved problem
  - **Verified delivery** is the new part and the part reviewers will probe. If the
    checker fails, nothing else matters
- **Result to state after the MVP**: "We ran N contracts across K real providers
  over H hours. We planted a model swap and a slowdown and caught both within X
  minutes. The false-alarm rate on healthy providers was Y%. The market
  automatically routed volume away from the bad provider"

## Users and Scope

| Role | Who, in the MVP | What they do |
|---|---|---|
| Buyer | `simulator` (our script) | Posts bids on a schedule; sends ordinary prompts under its active contracts |
| Seller | 3 to 5 real inference providers serving the **same open model**, reached via OpenRouter with the provider pinned | We post one standing ask per provider on its behalf |
| Prober | Part of Noesis (new) | Sends calibrated canary questions to each seller under each active contract |
| Operator | Javin, Lucas, Giles; GP reviews | Runs the stack, watches the dashboard, triggers planted failures, demos |

- Buyers are simulated: we control traffic, can repeat experiments, and hold no
  customer data. This avoids PII and ToS exposure before there are privacy rules
  (NoesisServer Open Questions, item 4)
- **Out of scope** (see also After the MVP below):
  - Payments and credits (`PR_P3`, `PR_P4`)
  - Smart routing, fusion, caching, and distillation (`PR_S2`, `PR_S3`, `PR_S5`,
    `PR_S10`, `PR_S12`)
  - Multiple tiers and cross-tier matching (`PR_M5`)
  - Public price feed (`PR_M4`), blind bids (`PR_M6`), scored compatibility (`PR_M9`)
  - Cloud deploy (`PR_P2`)
  - LLM-judge grading, streaming responses, and real customers

## Market and Reputation Rules

- **Matching and pricing** (task `T3.2`; decision `decisions.md` D-22)
  - Compatibility filter: bid `beta` and ask `alpha` can match only if
    - `alpha.c_level == beta.c_level_min`
    - `alpha.l_typical <= beta.l_max`
    - `alpha.r_typical >= beta.r_min`
    - the seller is not excluded by reputation (see Reputation below)
  - Order: bids by `p_max` descending, asks by `p_min` ascending (stable). For each
    bid, fill from the compatible asks in that order
  - Accept a pair only if the price interval `[max p_min of accepted asks, min p_max
    of accepted bids]` stays non-empty
  - Clearing price `p*` is the midpoint of the final interval, the same for all
    fills in the tier. No party trades at a price it refused
- **Reputation** (task `T3.4`; decision `decisions.md` D-23)
  - One score `rho` in [0, 1] per seller. New sellers start at `rho_0 = 0.8`
  - On each `PASSED` or `FAILED` verdict: `rho <- (1 - lambda) * rho + lambda *
    1[PASSED]` with `lambda = 0.3` (paper equation `reputation_update`)
  - `INSUFFICIENT` verdicts don't update `rho`
  - If `rho < rho_min = 0.5`, the seller is `blocked` for `cooldown_rounds = 6`
    rounds (1 minute at T = 10 s) and excluded from matching
  - After the cooldown the seller is on `probation`: `rho` resets to `rho_min`, and
    the seller gets at most 1 contract per round until 3 consecutive `PASSED`, then
    it is `eligible` again
  - Every change is stored in `reputation_events` for the dashboard
  - With these values a healthy seller survives one false flag (0.8 -> 0.56). Two
    failed contracts in a row block it (0.8 -> 0.56 -> 0.39). The planted-failure
    seller is blocked within about 2 to 4 contracts (under 1 minute at demo cadence)
- **Seller asks**: one standing ask per provider
  - `l_typical` and `r_typical` come from calibration
  - `p_min` = the provider's real per-token price x task size x (1 + margin)
- **Simulator** (task `T4.1`): config-driven (YAML) buyers
  - Each round posts bids with randomized `n_tasks` (20 to 60) and `p_max`, and with
    fixed `l_max` and `r_min` for the tier
  - For each `ACTIVE` contract it owns, it sends prompts from a prompt pool at a
    configurable rate through the OpenAI SDK pointed at the gateway
- **Fault injector CLI** (planned): `noesis fault --seller X --latency +3000ms`,
  `--model-swap <smaller-model>`, and `--clear`
  - Scheduled faults come from the simulator config
  - Faults can already be set through the gateway admin endpoints
    (`design.gateway.md`, Demo Faults)

## Where GP's Code Stands (Upstream `3f1a5db`)

| Module | What works | Gaps that matter for the MVP |
|---|---|---|
| `batch_call_auction.py` | Bid/Ask validation, per-tier uniform-price call auction, pluggable `OrderBookStore` | Matching **ignores `l_max`/`r_min`** (price + tier only). Unmatched orders are **dropped** every round. No order ids/timestamps. No scheduler |
| `contract_dispatch.py` | `Contract` schema, `build_contracts()` | Fulfillment is `random() < 0.9`. If a buyer submits 2 bids in one round, `build_contracts` uses the last bid's terms for all of that buyer's fills (`Fill` has no order id). No contract id, window, or state |
| `passthrough_proxy.py` | `Gateway.call()`, provider registry, request log, pluggable store | Fake providers only. Cost is a placeholder `cost_per_char`. **No contract id in the log.** A provider exception bubbles up and **the failed request is never logged**. Sync, no timeout |
| `platform_api.py` | FastAPI: `/bids`, `/asks`, `/rounds/clear`, `/contracts/{id}`, `/rounds/{tier}/latest`, `/completions`, `/logs`, API-key auth | `/completions` isn't OpenAI-shaped. It also reads its result as `get_log()[-1]`, so under concurrent requests it can return **another request's** answer, and on Postgres that call loads the whole log each time. `/rounds/clear` and `/logs` (raw prompts) are unauthenticated. `GET /contracts/{id}` for an unknown id returns 400 with an error message that dumps **every stored contract** (it should return a 404 with no data). The API key is checked but never tied to `buyer_id`/`seller_id`, so any key can bid as any buyer |
| `postgres_store.py`, `main.py`, `devops/` | Postgres-backed stores, env-selected backend, Compose file | Depends on GP's `helpers` monorepo (`hdbg`, `hprint`, `hunitest`, `hsql_implementation`) and his Docker base image. Symlinked files (`conftest.py`, `tasks.py`, `pytest.ini`, several Dockerfiles) weren't vendored |
| Tests | ~2,000 lines across 6 files | Need `helpers` to run |
| Paper | Full protocol, reputation eq., violation test | Violation test uses the lower bound. It flags healthy sellers 44–100% of the time (`scripts/detection_math.py`) |

Roadmap mapping: GP's v0.1 is done except cloud deploy (`PR_P2`). Our MVP covers his
v0.2–v0.4 **minus payments**, plus metering and reputation, which his roadmap put later.
It is server-first; his roadmap was market-first.

## Milestones and Dates

Live status and history: `progress.md`.

| # | Milestone | Dates | Owner | Exit test | Status |
|---|---|---|---|---|---|
| M0 | Setup + facts | Tue 9/29 – Fri 10/9 | J+G+L | GP's tests pass locally and in CI with no `helpers` monorepo. OpenRouter spike confirms ≥ 3 pinnable providers for the model. Schema frozen. Decisions signed | ✅ 9/30 (GP email pending) |
| M1 | Real server | Mon 10/12 – Fri 10/30 | J (+L: T1.5, T2.1 starts) | Unmodified `openai` Python SDK → our gateway → 3 pinned providers. Every log row has `contract_id` + verified `actual_provider` | ✅ 10/4 (spend cap deferred) |
| M2 | Checker | Mon 11/2 – Fri 11/20 | J + L | Calibrated canary set ≥ 150. Offline replay: injected slowdown and model swap flagged, healthy traffic not flagged, rates within ±5 pts of `detection_math.py` | 🟨 foundation merged 10/11; runtime and data remain |
| M3 | Loop closed (parallel with M1–M2) | Mon 10/12 – Fri 11/20 | G | With fake gateway: auction on timer, filter works, contracts go ACTIVE→verdict, a failing seller is blocked, cooled down, put on probation, reinstated | ⬜ |
| M4 | Integrated demo | Mon 11/23 – Fri 12/11 | J+G+L | Success Criteria 1–8 in one recorded 60-min run | ⬜ |

Buffer: Thanksgiving week (11/23) is counted as half capacity. If M2 slips more than
1 week, cut the dashboard to a read-only DB view plus a CLI report. Don't cut the
planted-failure test.

## Success Criteria

- All of these must hold in one recorded live run:
  1. **Autonomy**: the loop runs at least 60 min with at least 3 real providers, with
     zero manual steps after `make demo`
  2. **Attribution**: 100% of requests have `contract_id` and `actual_provider`.
     Attribution mismatches are under 1% and all are excluded from scoring
  3. **Planted slowdown caught**: at minute 15, add `+3000 ms` to seller A
     - The first `FAILED` verdict comes within 2 contract windows
     - A is blocked within 5 min
     - The share of matched volume that A wins drops to 0 while A is blocked
  4. **Planted model swap caught**: at minute 35, seller B serves the 8B model
     - The first `FAILED` (quality) verdict comes within 3 contracts
     - B is blocked within 5 min
  5. **No false alarms**: healthy sellers have at most 1 `FAILED` verdict each over
     the hour and are never blocked
  6. **Recovery**: after the fault clears, the seller goes through cooldown, then
     probation, then `eligible`
  7. **Legibility**: someone who has never seen the code can read all of the above
     off the dashboard in under 5 min. Test with GP or another reviewer
  8. **Cost**: total real spend for the run is at most $10
- Deliverables: the recorded run (video and exported DB), a one-page results memo
  with the numbers above, and a reproducible `make demo`

## Task Breakdown

Every task: branch → PR → tests green in CI → one reviewer (another teammate).
"GP-PR" names the item from the PRs section above that it closes or replaces.

### M0 — Setup + facts (all three)

| ID | Task | Owner | Acceptance | GP-PR |
|---|---|---|---|---|
| T0.1 | ✅ 9/29 Team backup repo created (private); this monorepo is primary since 10/6 | J | — | — |
| T0.2 | ✅ 9/29 Run GP's code without the monorepo: done in the team's backup repo with a small `helpers` stand-in. **In this repo the real `helpers` is used** | J | 63/63 upstream tests pass | resolves D8 |
| T0.3 | ✅ Tooling: deps added to `devops/docker_build/pyproject.toml` (+ `poetry.lock`); local test Postgres on :5433 (`onboarding.md`). GP's CI only runs course dirs, so Noesis tests run locally / in the dev container | L | Tests green locally | — |
| T0.4 | ✅ Superseded: new code lives as flat modules next to GP's (`noesis_*.py`, `gateway_*.py`, tests in `test/`), matching this directory's layout; later regrouped into component dirs | J | — | — |
| T0.5 | ✅ 9/30 run: 8/11 endpoints usable, 100% attribution (`findings/openrouter_spike.md`) — **OpenRouter spike** `scripts/spike_openrouter.py`: list providers for Llama 3.3 70B. For each: 20 calls with `provider.only=[slug], allow_fallbacks=false`. Record served provider (from response), latency p50/p95, error rate, `usage` cost | J | `docs/findings/openrouter_spike.md` with table. Decide D3/D4 from data | de-risks PR_S7 |
| T0.6 | ✅ `common/migrations/001_init.sql` (applied automatically by `noesis_db.apply_migrations()`) — Freeze DB schema. All three sign | J+G+L | Merged; all agree | — |
| T0.7 | Sign off the decisions marked proposed in `decisions.md`. Send this plan to GP with the Open Questions for GP | J | Written replies logged in `decisions.md` | — |

### M1 — Real server (J; L owns T1.5)

| ID | Task | Acceptance | GP-PR |
|---|---|---|---|
| T1.1 | ✅ `gateway_openrouter.py`: `OpenRouterProvider` behind the `Provider` interface (`gateway_providers.py`) (step 2, 10/4); async (`httpx`): pinned provider, no fallbacks, timeout = 2·l_max, parse text, served provider, `usage` tokens + cost | Unit tests with recorded fixtures (no network in CI). One live smoke test marked `@live` | PR_S7 |
| T1.2 | ✅ 10/4 `gateway_api.py` + `gateway_app.py`; live-verified via OpenAI SDK → Groq/CoreWeave — `POST /v1/chat/completions` + `GET /v1/models`, OpenAI request/response shape, Bearer auth, OpenAI-style error JSON | `openai` SDK with `base_url=http://localhost:8000/v1` works unmodified | PR_S8 |
| T1.3 | ✅ 10/4 `gateway_routing.py` (+ `/admin/dev-contract` until the auction exists) — Contract binding (`X-Noesis-Contract`), contract → seller → provider routing, 409 on inactive/foreign contract. Derive buyer/seller identity from the API key, not the request body. Put auth on `/rounds/clear` + `/logs` (operator only). Unknown contract → 404 without leaking the store. Return the request's own log entry, not `get_log()[-1]` | Tests for each rejection path + a concurrency test | PR_S4 (schema), PR_P1 fixes |
| T1.4 | ✅ logging + attribution check done 10/4 (`gateway_request_log.py`); token clamp in T1.2. **Spend cap deferred** (10/4): OpenRouter key credit limit covers the MVP; add our own cap at Tier 2 / first paying buyer — `requests` table logging for **every** outcome (ok / timeout / upstream_error / buyer_error / gateway_error), attribution check, token clamp | Failure paths produce log rows. Cap test with fake cost | PR_S1 fix |
| T1.5 | ✅ 10/4 `gateway_faults.py` + `/admin/faults`; live-verified (8B swap on Groq, +1.5 s slowdown). Note: 8B answered an easy probe correctly, so calibration must keep only questions 8B fails — Fault injection hooks: per-seller `extra_latency_ms`, `model_override`. Admin endpoint + CLI. Every affected row marked `fault_injected=true` | Tests | new |

### M2 — Checker (L owns T2.1–T2.3, J owns T2.4–T2.6; T2.1 starts 10/12)

| ID | Task | Acceptance | GP-PR |
|---|---|---|---|
| T2.1 | ✅ Versioned 300-question production candidate bank, validation, secret balanced selection, normalizers, and exact-match grader | Grader unit tests incl. number formats ("1,000" = "1000"); bank has ≥ 300 records | PR_S11 (simplified) |
| T2.2 | 🟨 Live calibration runner, persistence, survivor gate, activation, and findings generator are implemented. The reviewed paid provider run remains | `docs/findings/calibration.md`: survivors ≥ 150, weak-model accuracy on survivors, per-provider p95, cost | PR_S11 |
| T2.3 | ✅ For each ACTIVE contract, secret deterministic canaries are spread over its window, atomically claimed, sent as ordinary traffic, graded, and persisted server-side | Integration test with fake provider | new |
| T2.4 | 🟨 PR #604: immutable evidence, seller-attribution rules, separate `r_lat`/`q` upper bounds, verdicts, and atomic persistence. Window-closing orchestration (n_tasks served or 60 s) remains | Unit tests reproduce `scripts/detection_math.py` cases. Property test: never FAILED when all requests pass | PR_S4, PR_S6 (corrected) |
| T2.5 | `VerdictSink` interface → reputation (in-process) | Contract test shared with T3.4 | PR_S4 |
| T2.6 | **Offline replay harness**: generate synthetic request streams (healthy / slowed / weak-model at measured accuracies), run metering, report flag rates | Rates within ±5 pts of analytic numbers over 1,000 simulated contracts | new |

### M3 — Loop closed (G, starts 10/12 in parallel)

| ID | Task | Acceptance | GP-PR |
|---|---|---|---|
| T3.1 | Orders get `order_id`, `account_id`, `submitted_at`. `Fill` carries bid/ask order ids (fixes the multi-bid bug). Standing asks persist, bids expire | GP's auction tests still pass. New tests for carry-over + multi-bid buyer | PR_M1 fix |
| T3.2 | Compatibility filter (tier, latency, reliability, seller eligible) + IR-safe uniform pricing (Market and Reputation Rules) | Tests: incompatible cheap ask skipped. No fill priced outside any matched party's limit (property test over random books) | new |
| T3.3 | Contract lifecycle state machine PENDING→ACTIVE→CLOSED→PASSED/FAILED/INSUFFICIENT. Remove coin-flip from the runtime path | Tests. `mock_fulfill` used only in tests | PR_M8 (redesigned) |
| T3.4 | Reputation: decay update, block, cooldown, probation, events table | Tests: 2 consecutive fails → blocked. One fail among passes → not blocked. Cooldown → probation → eligible after 3 passes | PR_M3 |
| T3.5 | Scheduler: asyncio loop clears every `T` s, survives empty rounds and exceptions, injectable clock | Fake-clock tests (GP's PR_M7 test list) | PR_M7 |
| T3.6 | Postgres stores for new tables (reuse GP's `*Store` ABC pattern) | Store tests against the CI Postgres service | PR_P2b ext. |

### M4 — Integrated demo (all three)

| ID | Task | Owner | Acceptance |
|---|---|---|---|
| T4.1 | Simulator: YAML buyers, bids per round, prompt pool, sends via `openai` SDK under its contracts, scheduled faults | L | 10-min run with no errors |
| T4.2 | Dashboard (Streamlit): rounds and prices, contracts + verdicts + reasons, reputation timelines with block/probation markers, request log + attribution, spend vs cap, calibration stats | L (G defines what the dashboard must tell a non-technical viewer) | Legibility test with a non-coder |
| T4.3 | One-command demo start: Compose (api, postgres, simulator, dashboard), `.env.example`, seeded sellers from calibration | L | Fresh clone → running demo in < 10 min |
| T4.4 | Three rehearsal 60-min runs. Tune `l_max`, window, canaries/contract if false alarms or misses | J+G+L | Run 3 meets Sec. 8 |
| T4.5 | **Recorded run** + exported DB + 1-page results memo (numbers for the result line in MVP Claim) | J+G+L | Memo in `docs/results/` |
| T4.6 | Walk GP through it. Collect his feedback on the paper's violation rule change | J | Notes in `docs/decisions.md` |

## Critical Path

```
T0.5 spike ─► T1.1 ─► T1.3/T1.4 ─► T2.2 calibration ─► T2.4 metering ─► T4.4 rehearsals ─► T4.5
T3.1 ─► T3.2 ─► T3.3 ─► T3.4 ───────────┘
```

The OpenRouter spike (T0.5) and calibration (T2.2) are the two points where the data
might force a change of plan: model, provider count, canary design. Do them early.

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Canary check too noisy | Can't tell good from slightly worse | Calibrated set (T2.2), clear-cut questions, publish the checker's own error rate. MVP targets gross failures (model swap, slowdown). Subtle quantization detection is a post-MVP research result |
| Canaries distinguishable from real traffic | Provider games the prober | MVP: use buyer-style formatting. Later: generate canaries from the real prompt distribution, rotate the bank |
| OpenRouter attribution breaks | Wrong seller blamed | `only` + `allow_fallbacks:false`, verify the reported provider, exclude mismatches |
| Too few providers for the model | Fewer than 3 sellers | Week-1 check (done: `decisions.md` D-20). Otherwise pick the open model with the most providers |
| Top-tier supply is thin | OpenAI, Anthropic, and Google won't bid, so the frontier tier is empty or collusion-prone | MVP uses one open-weight model with many hosts. The frontier tier waits for real supply |
| Gateway intermediary hides who served a request | Reputation assigned to the wrong seller | OpenRouter pinning now. Own the gateway (LiteLLM or direct APIs) before any real customers |
| Competitors add metering | Wedge commoditized | Moat is the longitudinal, request-level verified-quality dataset. Study OpenRouter, Martian, Not Diamond, Artificial Analysis, Concentrate AI |
| High reliability hard to verify | 99.9% needs about thousands of requests | Demo at 0.90. The minimum-detectable-shortfall table (GP draft `research/ideas/draft.Noesis_Reputation_Detection_Bound.md`) is a paper result |
| Provider ToS on resale and logging | Legal exposure | Read each provider's and OpenRouter's terms before any public claim. MVP has no resale (we're the only buyer) |
| Part-time bandwidth | Slips | The daily-progress rule and Friday demo in the Cadence bullet above |

## Open Questions for GP

1. Is the server-first order (this section) fine, or should it follow the
   market-first Roadmap?
2. Should `PR_S9` (auto-asks from the OpenRouter catalog, `v0.5`) and `PR_P4`
   (free-credit ledger, `v0.6`) be in the MVP? Both are currently deferred
3. Is Noesis its own gateway, or a metering and market layer on top of existing
   gateways?

## Side Track: Provider Independence (Not Blocking the MVP)

Decision 10/4: OpenRouter stays for the MVP. In parallel, low priority, work out how
Noesis reaches providers without depending on OpenRouter (Tier 2):
- Direct accounts + adapters for the demo sellers (most are OpenAI-compatible, so one
  generic adapter + per-provider config behind the existing `Provider` interface)
- Cross-check: same probes via OpenRouter vs direct, compare answers and latency
- Triggers to switch: pinned≠served >1%, OpenRouter outages distorting scores, policy
  or pricing changes, first paying customer

## After the MVP (Not Scheduled)

First productize the checker as the **wedge**: a verified-delivery monitoring gateway for teams spending $5k–$50k/mo. Then, roughly in value order: real design-partner buyers (PII
scrubbing first, GP S-4), direct provider APIs alongside OpenRouter, subtle-degradation
detection (quantization) with a larger canary bank or reference-model agreement,
public price/reputation feed (`PR_M4`), credits ledger (`PR_P4`), cloud deploy
(`PR_P2`), multiple tiers (`PR_M5`), routing/fusion savings (`PR_S2`, `PR_S5`).
