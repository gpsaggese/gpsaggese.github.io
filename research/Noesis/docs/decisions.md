# Noesis — Decisions

- Every non-obvious choice in the MVP, why it was made, and what else was
  considered. Newest at the bottom of each section
- Each PR adds its decisions here and links them in its description
- Requirements and plan: `plan.md`

## Measurement and Grading

### D-1. Flag a violation only when the upper confidence bound misses the promise

- **Decision**: a contract measure (latency reliability, canary accuracy) is a
  violation only if the one-sided 95% Wilson **upper** bound is below the promised
  level (exact Clopper–Pearson for small samples)
- **Why**: the null hypothesis is "seller is compliant"; rejecting it requires even
  the optimistic estimate to miss the promise. Exact binomial rates
  (`scripts/detection_math.py`):

  | Scenario | Flag if lower bound < promise (paper, `PR_S6`) | Flag if upper bound < promise |
  |---|---|---|
  | Healthy latency 0.97 vs promise 0.90, n=50 | 44% of contracts flagged | 0.002% |
  | Healthy canaries 0.98 vs promise 0.85, n=10 | 100% flagged | 0.003% |
  | Slowed seller, latency 0.20, n=50 | caught | caught 100% |
  | Weaker model, canary accuracy 0.5, n=10 | caught | caught 83% per contract |

- **Across contracts**: the same weaker model is caught in at least 97% of cases
  within two contracts
- **Alternatives**: the paper's lower-bound rule (punishes healthy sellers for
  noise); a raw threshold `r_hat < R_min` (`PR_S4`; same problem on small windows)
- **Status**: proposed to GP (changes paper Sec. 5 eq. `reliability_lower_bound`)

### D-2. Score latency and quality separately

- **Decision**: per contract, compute latency reliability over all requests and
  canary accuracy over canaries only; fail the contract if either is violated
- **Why**: with ~10% canaries, a single merged success rate dilutes quality
  failures. A swapped weaker model is caught on **2.6%** of contracts merged vs
  **83%** separated (`scripts/detection_math.py`)
- **Alternatives**: `PR_S4`'s merged indicator `capability_ok AND latency_ok`

### D-3. Canaries come from Noesis's own prober, and are calibrated

- **Decision**: Noesis injects known-answer questions under each contract; only
  questions every healthy provider gets right and the weaker 8B model mostly misses
  are kept. Calibration runs every candidate three times per target, requires 100%
  healthy-target accuracy and at most 60% weak-target accuracy, and activates a bank
  only when at least 150 candidates survive
- **Why**: real buyers won't label their traffic; uncalibrated questions make the
  checker's own error rate unknown. Evidence: on 10/4 the 8B model answered an easy
  probe ("9.9 vs 9.11") correctly, so easy questions can't expose a model swap
- **Alternatives**: buyer-mixed canaries (PRD v0.1); LLM judge (later)

### D-28. Canary schedules are secret, deterministic, and server-owned

- **Decision**: derive balanced canary choices and times from an HMAC secret plus
  contract identity. Send only the ordinary prompt to the provider; keep canary ID,
  expected answer, grade, and grader version in Noesis-owned request evidence
- **Why**: deterministic schedules are reproducible and testable, while the secret
  prevents sellers from predicting which requests are probes. Server-only metadata
  prevents a provider from recognizing a canary from the request envelope
- **Alternatives**: fresh process-local randomness (hard to replay); public seeded
  schedules (predictable); marking probes in provider requests (easy to game)

### D-29. Standalone monitoring cannot mutate market state

- **Decision**: run longitudinal provider/model monitoring without market orders or
  contracts, and store its observations separately from contract evidence
- **Why**: monitoring must run without buyer traffic, while experimental thresholds
  and monitor failures must not affect fulfillment, verdicts, or seller reputation
- **Boundary**: the monitor can reuse the canary bank, capability measurer, and
  provider interface; it cannot write contracts, contract verdicts, or reputation
- **Alternatives**: synthetic contracts for monitoring (couples research traffic to
  market state); reuse contract evidence tables (risks accidental seller penalties)
- **Status**: proposed in `spec.PR_S13.md`; implementation starts in `PR_S14`

## Gateway

### D-4. OpenRouter with the provider pinned, fallbacks off

- **Decision**: reach providers through OpenRouter (`PR_S7`) with
  `provider.only=[tag]`, `allow_fallbacks=false`, and verify the reported provider
- **Why**: one integration gives 10 providers of the same model. Verified 9/30:
  8/11 endpoints usable, 100% of served calls came from the pinned provider
  (`findings/openrouter_spike.md`)
- **Alternatives**:
  - Concentrate AI: fallback can't be disabled per request per its docs, and only one
    host serves our model, so no competing sellers to compare
  - Direct provider APIs: full control and verifiable attribution; planned as the
    side track (`plan.md`, Side Track: Provider Independence) behind the same `Provider` interface
- **Known limit**: attribution is OpenRouter's report until direct adapters exist

### D-5. Blame a seller only for failures OpenRouter attributes to the provider

- **Decision**: `classify_error()` marks a failure `upstream_error` (seller's fault)
  only if OpenRouter says the provider failed; OpenRouter's own 5xx/429, our
  key/credits/pin errors, and unreachable OpenRouter are `gateway_error` (not the
  seller's fault)
- **Why**: otherwise an OpenRouter outage lowers every seller's reputation. Evidence:
  the real Parasail 429 carried `"Provider returned error"`; a bad pin (Google)
  returned 404 "no endpoints", our configuration error
- **Alternatives**: any 5xx counts against the seller (the first version did this)
- **Trade-off**: timeouts always count against the seller, even though part of the
  delay may be OpenRouter's; latency limits are set from latency measured through
  OpenRouter, so healthy providers rarely time out

### D-6. OpenAI-compatible API with the contract in a header

- **Decision**: `POST /v1/chat/completions` in OpenAI's shape (`PR_S8`); the
  contract id travels in `X-Noesis-Contract`
- **Why**: standard SDKs work unchanged (tests drive the app with the real `openai`
  SDK); SDKs can add headers without touching the body
- **Alternatives**: a body field (non-standard); `PR_P1`'s bespoke `/completions`

### D-7. Someone else's contract is a 404, like a missing one

- **Decision**: `resolve_route()` returns the same 404 for "doesn't exist" and
  "belongs to another buyer"
- **Why**: a 403 would let anyone probe which contract ids exist
- **Alternatives**: 403 (the first plan)

### D-8. Log every provider call; flag unverifiable attribution

- **Decision**: one `requests` row per provider call, including failures; an answer
  from a different or unnamed provider is `attribution_mismatch = true` and excluded
  from scoring
- **Why**: the checker grades only from these rows; `PR_S1`'s `Gateway.call()` lost
  failed calls (an exception skipped the log)
- **Alternatives**: log only successes; trust unnamed answers

### D-9. Spend cap deferred

- **Decision**: no gateway-side spend cap yet; the OpenRouter key's credit limit is
  the cap
- **Why**: costs are tiny (220-call spike: $0.0026); the key limit is a hard stop
- **Revisit**: with direct provider keys (no single limit) or the first paying buyer

### D-10. Demo faults: real delay, same-provider model swap

- **Decision**: per-seller `extra_latency_s` (a real delay the buyer feels) and
  `model_override` on the same provider
- **Why**: a swap to a different provider would be flagged as an attribution
  mismatch and excluded, hiding the failure the demo must show

### D-19. Task unit: one capped chat completion (formerly PRD D1)

- **Decision**: a task is one chat completion with at most 256 output tokens and a
  prompt of at most 1,000 input tokens; the gateway clamps larger requests
- **Alternatives**: tokens, wall-clock compute, or a benchmark-normalized
  task-equivalent (`plan.md`, NoesisMarket Open Questions 1)
- **Status**: proposed

### D-20. One open model: Llama 3.3 70B, weak model Llama 3.1 8B (formerly PRD D3)

- **Decision**: every seller serves Llama 3.3 70B Instruct; Llama 3.1 8B Instruct is
  the weaker model for calibration and for the model-swap fault
- **Why**: many hosts serve one open-weight model, so sellers compete. The check
  required at least 4 providers: 8 of 11 endpoints were usable
  (`findings/openrouter_spike.md`)
- **Alternatives**: frontier models, whose vendors (OpenAI, Anthropic, Google) won't
  bid, so the tier would be empty or collusion-prone (`plan.md`, Risks)
- **Status**: proposed

## Implementation

### D-11. Async `psycopg` 3 for the gateway, not `helpers.hsql`

- **Decision**: `noesis_db.py` uses `psycopg` 3's async pool
- **Why**: the gateway serves concurrent requests on one event loop; `helpers.hsql`
  (`psycopg2`) blocks it. GP's `postgres_store.py` is unchanged and still uses
  `helpers.hsql`
- **Alternatives**: `helpers.hsql` in a thread pool; an ORM

### D-12. Flat modules next to GP's, original code untouched

- **Superseded**: one dir per component (`common/`, `gateway/`, `checker/`,
  `market/`, `noesis_platform/`). The file names stay unchanged

- **Decision**: `noesis_*.py` and `gateway_*.py` beside GP's modules; tests in
  `test/` named `test_<module>.py`
- **Why**: matches this directory's layout; prefixes avoid test-file name clashes
  in the monorepo; GP's prototype stays as the reference for the market milestone
- **Alternatives**: a `noesis/` subpackage (the team's backup repo layout)

### D-13. Test DB: connect to a running Postgres and skip if absent

- **Decision**: DB tests connect to `NOESIS_TEST_DATABASE_URL` and skip when it's
  unreachable, instead of `helpers.hsql_test`'s per-class docker-compose
- **Why**: works on laptops and in the dev container without Docker-in-Docker;
  tests never fail just because no DB is running
- **Alternatives**: `hsql_test.TestImOmsDbHelper` (sync `psycopg2`, needs
  `requires_docker_in_docker`)

### D-14. Some `try/except` kept, each where failure is an outcome to report

- **Decision**: `try/except` only where a failure is data, not a bug: network errors
  in `gateway_openrouter.py` become statuses; `/health` reports an unreachable DB as
  503; non-JSON provider bodies are classified. Request errors use one app-level
  handler (`RoutingError`), like `platform_api.create_app()` does for
  `AssertionError`
- **Why**: the gateway must never crash or skip logging because a provider failed

### D-21. Stack: Python 3.11, FastAPI, Postgres, Streamlit (formerly PRD D12)

- **Decision**: Python 3.11, FastAPI, Postgres, `httpx`, the `openai` SDK (in the
  simulator), and a Streamlit dashboard
- **Why**: Streamlit is the fastest path to a dashboard; replace it later
- **Status**: proposed

## Market Scenario Families (PR_M11)

### D-15. Seeded Named Families With Guaranteed Regimes

- **Decision**: implement five scenario types as parameterized, seeded generators:
  balanced, excess demand, excess supply, multiple tiers, and non-crossing prices.
  Each accepts order counts; seeds vary quantities and prices reproducibly.
- **Why**: the ticket requests coverage of five regimes, not one hardcoded book
  per regime. Families let tests and notebooks explore many examples of each type.
- **Alternatives**: one fixed example per type was the initial implementation;
  it was replaced because it limited variation. Unrestricted random books alone
  cannot guarantee which regime a caller receives.
- **Boundary**: generation returns fresh data only. It does not run an auction,
  touch a database, or establish parity between persistence backends.

### D-16. Construct Quantities and Price Bands to Preserve Each Regime

- **Decision**: draw positive quantities, balance totals by increasing the final
  order on the smaller side, and add a random positive excess for demand/supply
  regimes. Use crossing bid/ask price bands, reversing them for no-cross books.
  Offset multi-tier bands so clearing prices cannot coincide between tiers.
- **Why**: equal totals alone do not guarantee matching; compatible prices are
  necessary. Construction guarantees the promised outcome without rejection loops.
- **Performance terms**: retain compatible buyer terms of 2 seconds and 90%
  reliability and seller terms of 1 second and 95%. These isolate the price,
  quantity, and tier behavior. Participant ids remain unique within each side.
- **Scale**: positive integer quantity and price multipliers support larger values
  without changing the regime. Prices are illustrative per-task units. Adjusted
  quantities can exceed the initial random draw range.
- **Alternatives**: reject randomly generated books until the desired regime appears,
  or relax the regime guarantee. Construction is predictable and avoids unbounded
  retries. Performance-aware matching and real provider prices remain out of scope.

### D-17. Local Randomness and Explicit Parameter Contracts

- **Decision**: use a local `random.Random(seed)`; identical inputs and ordered
  tiers produce equal values without changing global randomness. Named families
  have positive counts per tier and default to two orders per side and seed 0.
  All optional named-family arguments are keyword-only, following coding rules.
  Multi-tier requires at least two distinct tiers. Default examples are readable;
  counts may be increased for larger demonstrations.
- **Validation**: reject booleans and noninteger seeds/counts/scales. Named counts
  and scales must be positive; tier names must be nonblank. The unrestricted
  random generator retains nonnegative counts, ordered unique tiers, independent
  quantity draws of 1–100 and price draws of 1–20, and no regime guarantee.
- **Alternatives**: global random seeding or unordered tier inputs undermine
  isolation or reproducibility. Fixed prices/quantities across calls do not offer
  the desired range of examples.
- **Boundary**: scales support representative large-value tests, not all numerical
  extremes. Arbitrarily large floats and fractional-price precision are not proven.

### D-18. Validate Regimes Across Seeds Until M10 Is Available

- **Decision**: validate each named family across multiple seeds, counts, and
  numeric scales using `OrderBook.clear_round()` and deterministic mock dispatch.
  Check conservation, promised regimes, price limits, fresh objects, and distinct
  tier prices. Retain generated invariant checks for unrestricted books.
- **Test organization**: expose each seed/count/scale case as a numbered test
  method calling shared assertion helpers instead of `subTest()`. Keep the full
  seed matrix and describe each configuration in its method docstring so a failing
  case can be rerun by its pytest node id.
- **Why**: exact outcomes vary with seed, but the scenario's defining properties
  must hold for every supported configuration. These checks exercise the existing
  auction without duplicating M10 orchestration.
- **Status**: integration through M10's `run_round()` remains pending. This does
  not waive that original acceptance criterion or establish live-provider delivery.

## Market Mechanism (MVP)

### D-22. Clearing price is a price no matched party refused

- **Decision**: a bid/ask pair is accepted only if the interval of prices that all
  accepted parties allow stays non-empty; the clearing price is the midpoint of the
  final interval (rule: `plan.md`, Market and Reputation Rules)
- **Why**: the latency and reliability filter breaks GP's rule "midpoint of the last
  matched pair", so a matched party could trade at a price it refused
- **Alternatives**: midpoint of the last matched pair

### D-23. Reputation adds cooldown and probation

- **Decision**: a blocked seller sits out a fixed number of rounds, then returns on
  probation with its score reset to the block threshold (numbers: `plan.md`, Market
  and Reputation Rules)
- **Why**: blocked sellers get no contracts, so under the paper's update formula
  their score can never recover; without this, one block lasts forever
- **Alternatives**: the paper's update formula alone

### D-24. Contract result comes from a lifecycle, not a synchronous `fulfill()` call

- **Decision**: a contract moves `PENDING` -> `ACTIVE` -> `CLOSED` -> `PASSED`,
  `FAILED`, or `INSUFFICIENT`; the result is the checker's verdict after the window
  closes. `mock_fulfill()` stays for unit tests only
- **Why**: the buyer sends requests over time, so a contract's result exists only
  after its window closes. GP's `dispatch_contract()` returns a bool right away,
  which doesn't fit
- **Alternatives**: the `random() < 0.9` coin flip of `mock_fulfill()`
- **Status**: planned as `plan.md` T3.3 (`PR_M8`, redesigned); not built yet

## OpenRouter Supply Adapter (PR_S9)

### D-25. Price an Ask at the Worst-Case Task Cost With Zero Margin

- **Decision**: `p_min` = `prompt_price * 1000 + completion_price * 256`, i.e., the
  OpenRouter per-token prices times the PRD D1 task caps (≤1,000 prompt and ≤256
  completion tokens), times `1 + margin` with `margin = 0` by default. Prices are
  rounded to 1e-9 USD.
- **Why**: PRD S2 says a seller's `p_min` is the real per-token price times the
  task size plus a margin; the task size is the D1 definition. Pricing the cap
  means a seller never sells a task below what it can cost to serve. No margin is
  specified anywhere in the repo, so the default adds none and the value is a
  parameter.
- **Alternatives**: price the measured average token usage of the spike (cheaper
  asks, but a long prompt would be served at a loss); keep a hardcoded price per
  seller (the status quo this PR removes).

### D-26. Sellers, Latency, and Reliability Come From the Seed File

- **Decision**: the adapter builds asks only for the sellers in
  `common/seed.yaml`, matched to OpenRouter endpoints by `provider_slug` (the
  endpoint `tag`). Each seller's `l_typical` and `r_typical` are stored in the
  seed, seeded with the p95 latency and ok-rate of
  `docs/findings/openrouter_spike.md`, until PR_S11's calibration replaces them.
  Endpoints that belong to no seller are ignored; a seller whose endpoint has left
  the catalog gets no ask.
- **Why**: the seed is already the one list of sellers the gateway and checker
  use; OpenRouter's `latency_last_30m` is time-to-first-token, not the end-to-end
  latency the contract measures, and its `status` codes are undocumented, so
  neither is used as a contract term.
- **Alternatives**: create a seller for every endpoint in the catalog (would add
  providers the gateway cannot pin or blame); take latency from OpenRouter's
  30-minute figures (wrong quantity).

### D-27. Re-post One Ask per Seller Before Each Round

- **Decision**: each refresh posts one ask per seller for `n_tasks = 60` in the
  single `standard` tier; a seller that already has a pending ask in the current
  round is skipped. The refresh is a function the round scheduler (PR_M7) calls
  before `clear_round()`; it is not a background loop.
- **Why**: `OrderBook.clear_round()` drops every order it processes (no standing
  asks yet, K4), so supply must be re-posted each round; skipping pending sellers
  keeps a second refresh from doubling supply. 60 is the largest bid the PRD S1
  simulator posts in one round, so one seller can fill any single bid; OpenRouter
  publishes no capacity figure to derive it from.
- **Alternatives**: standing asks that survive clearing (K4, not built);
  a thread that polls OpenRouter on a timer (adds a second clock next to the
  scheduler's).
