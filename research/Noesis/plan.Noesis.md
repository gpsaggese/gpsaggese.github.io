# Noesis Implementation Plan

## Goal

- Build a working prototype of the \Noesis{} protocol
- The \Noesis{} protocol overview is in `papers/Noesis/*.tex`
- The architecture document `research/Noesis/architecture.md`

## Roadmap

- **v0.1**: NoesisMarket accepting inputing requests from suppliers and demands of
  compute, with API deployed on a laptop and on the cloud
  - [x] `PR_M1`
  - [x] `PR_P1`
  - [ ] `PR_P2`
- **v0.2**: Create an adapter from OpenRouter to the API as demand to provide
  capacity
  - [ ] `PR_S7`
  - [ ] `PR_S9`
- **v0.3**: Every 10 seconds an auction is run and an allocation is available
  - [ ] `PR_M7`
- **v0.4**: Implement NoesisServer where people can pay for capacity and get served
  through the API of the providers (without extra charging)
  - [x] `PR_S1`
  - [ ] `PR_M8`
  - [ ] `PR_P4`
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
3. Cloud target for `PR_P2`: which provider to standardize on, and whether the
   in-memory-to-datastore migration should land before or after the first public
   deployment
4. KYC/compliance: does accepting real-currency payments (credit card or crypto)
   trigger money-transmitter obligations that a synthetic task-credit avoids?

# PRs

## NoesisMarket

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
  `C_level` match per `architecture.md`'s Weakness 2, to build each tier's bucket
  from bids at tier $c$ against asks at tier $c$ and every tier above it, while
  keeping the existing single uniform-price-per-bucket clearing rule
- Unit tests: a bid at tier `cheap` is filled by an ask at tier `frontier` when no
  `cheap`-tier ask is available and the frontier seller's limit price clears the
  `cheap` bucket; a rational `frontier`-tier seller with a marginal cost above the
  `cheap` clearing price is not drawn into serving `cheap` demand
- Result: matched volume increases relative to `PR_M1`'s exact-tier-only baseline
  without introducing a new price axis, closing `architecture.md`'s Weakness 2

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
- Result: `NoesisMarket` clears rounds autonomously at a configured cadence instead
  of only on demand; the roadmap's `v0.3` target ("every 10 seconds an auction is run
  and an allocation is available") is `T = 10 s` on this scheduler

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
  to end, required before the roadmap's `v0.4` target ("get served through the API of
  the providers") is true rather than simulated

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

## NoesisServer

### `PR_S1`: [x] Minimal Passthrough Proxy with Logging

- Support 2-3 providers behind one API
- Define the storage schema for prompt, response, and metadata (provider, model,
  latency, cost)
- Log raw, unscrubbed prompt/response pairs for now, scoped to synthetic/test traffic
  only; add PII scrubbing as a follow-up before any real non-synthetic traffic is
  logged (tracked as an open question below)
- Result: every request/response pair is logged and queryable

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

### `PR_S4`: [ ] Fulfillment Monitoring Wired to the Market (stubbed Both Sides)

- Background: `papers/Noesis/05_noesis_server.tex` sec:fulfillment_monitoring
  Equation~eq:success_indicator defines per-request success as the conjunction
  `capability_ok(x) and latency_ok(x)`, and `measured_reliability`
  (Definition~measured_reliability) is the mean of that joint indicator over the
  fulfillment window, not two independently thresholded metrics
- Background: the storage schema needs the `contract` field from
  sec:gateway_architecture (`PR_S1` shipped without it); add it, nullable for
  pre-market traffic, so requests can be grouped by `kappa`
- Compute a per-request success indicator
  `s(x) = capability_ok(x) and latency_ok(x)`, then
  `measured_reliability = mean(s(x) for x in window)`, and flag a violation when
  `measured_reliability < R_min`; `capability_ok` depends on `NoesisServer` `PR_S11`
  below and is stubbed here if `PR_S11` has not landed yet
- Report violations via a **mocked** market-facing interface, matching the mock
  fulfillment interface on the `NoesisMarket` side (its `PR_M2`): real integration
  is deferred until both plans reach this point together
- Unit tests: violation correctly flagged/not-flagged against fixture contracts and
  measured outcomes; mocked report call invoked with the right payload
- Result: closed loop between a matched contract and its measured fulfillment, real
  once both sides swap the mock for the real interface

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

### `PR_S6`: [ ] Statistical Fulfillment Violation Test

- Background: `papers/Noesis/05_noesis_server.tex`'s fulfillment monitoring section
  notes that `PR_S4`'s naive rule, flagging $\kappa$ whenever
  `measured_reliability < R_min`, conflates genuine under-delivery with sampling
  noise on a small fulfillment window; the paper proposes a one-sided hypothesis test
  instead
- Replace `PR_S4`'s threshold with a normal-approximation lower confidence bound
  $\hat r_{\text{lower}}(\kappa, W)$, computed from `measured_reliability` and the
  window size `|W|` at a configurable confidence level $1 - \delta$; flag $\kappa$
  only when $\hat r_{\text{lower}}(\kappa, W) < R_{\min}(\kappa)$; fall back to an
  exact Clopper-Pearson interval when `|W|` is small
- Unit tests: a contract with a small window and a borderline sample reliability is
  NOT flagged when the confidence interval still clears `R_min` (a case that
  `PR_S4`'s naive rule would have flagged as a false positive); a contract with a
  large window and the same sample reliability IS flagged
- Result: fewer false-positive violation reports feeding `NoesisMarket` `PR_M3`'s
  reputation update, addressing the "attribution reliability" open question below

### `PR_S7`: [ ] Real Provider Liquidity Via an OpenRouter-backed `ProviderConfig`

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

### `PR_S8`: [ ] OpenRouter-compatible API Interface for `NoesisServer`

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

### `PR_S11`: [ ] Capability-measurement Estimator (hat-c(x))

- Background: `papers/Noesis/01_introduction.tex` Table~tab:components lists
  capability measurement as one of the five pluggable components, instantiated in the
  paper as "a verifier model, benchmark probe, or reference-model agreement," with a
  hosted evaluation provider (e.g., artificialanalysis.ai) as the pluggable
  alternative; `05_noesis_server.tex` sec:fulfillment_monitoring's success indicator
  (Equation~eq:success_indicator) needs `hat c(x)` as an input, which no PR above
  builds: `PR_S4`/`PR_S6` currently take `measured_reliability` as already computed
- Define a pluggable interface `estimate_capability(prompt, response) -> tier`;
  implement a first instantiation via reference-model agreement, reusing `PR_S3`'s
  agreement-label machinery in the other direction (compare the delivered response
  against a stronger reference model instead of comparing two routing candidates)
- Unit tests: a response matching the reference model's answer is assigned a tier at
  or above the contract's `C_level`; a response that disagrees is assigned a lower
  tier; the interface accepts a second, artificialanalysis -style stub implementation
  without changing call sites
- Result: `NoesisServer` `PR_S4`/`PR_S6` gain a real (not stubbed) source for
  `capability_ok(x)`; should land before `PR_S4`/`PR_S6` are considered complete

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

## NoesisPlatform

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
- Implementation note (`research/Noesis/platform_api.py`): added an unauthenticated
  `POST /rounds/clear` beyond the list above, since `OrderBook.clear_round()` is the
  only thing that produces a `Contract`/`TierClearResult` and none of the listed
  endpoints call it; without it, `GET /contracts/{id}` and
  `GET /rounds/{tier}/latest` could never be populated by an external caller
  `GET /rounds/{tier}/latest` reads from an in-process per-tier cache populated by
  `POST /rounds/clear`, standing in for `PR_M4`'s pub/sub feed until `PR_M4` lands

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

### `PR_P2b`: [ ] Use Postgres Backend

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

### `PR_P4`: [ ] No-charge Credit Ledger for Bid Gating

- Background: the roadmap's `v0.4` explicitly scopes "No charge", but the only PR
  that gates `POST /bids` on a balance is `PR_P3`, which requires real Stripe/crypto
  payment rails; `v0.4` needs the balance-gating _mechanism_ without collecting real
  payment yet
- Add a per-account credit ledger (in-memory, or datastore-backed once `PR_P2` lands)
  seeded with a fixed/free grant of task-credit and no real payment collected; gate
  `POST /bids` (`PR_P1`) on `account_balance >= n_beta * p_beta` exactly as `PR_P3`
  describes, debiting only once a bid is matched into a contract
- Unit tests: a new account starts with the seeded free-credit balance; a bid within
  balance is accepted and debits on match; a bid exceeding balance is rejected before
  reaching the auction
- Result: `POST /bids` is balance-gated end to end with zero real payment collected,
  satisfying the roadmap's `v0.4` "No charge" scope; `PR_P3` upgrades the same gate
  to a real-money funding rail when charging for real is in scope

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