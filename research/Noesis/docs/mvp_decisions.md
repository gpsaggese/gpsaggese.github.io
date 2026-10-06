# Noesis MVP — Decisions

- Every non-obvious choice in the MVP, why it was made, and what else was
  considered. Newest at the bottom of each section
- Each PR adds its decisions here and links them in its description
- Spec: `mvp_prd.md`; plan: `mvp_plan.md`

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

- **Alternatives**: the paper's lower-bound rule (punishes healthy sellers for
  noise); a raw threshold `r_hat < R_min` (`PR_S4`; same problem on small windows)
- **Status**: proposed to GP (changes paper §5 eq. `reliability_lower_bound`)

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
  are kept
- **Why**: real buyers won't label their traffic; uncalibrated questions make the
  checker's own error rate unknown. Evidence: on 10/4 the 8B model answered an easy
  probe ("9.9 vs 9.11") correctly, so easy questions can't expose a model swap
- **Alternatives**: buyer-mixed canaries (PRD v0.1); LLM judge (later)

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
    side track (`mvp_plan.md` §5b) behind the same `Provider` interface
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

## Implementation

### D-11. Async `psycopg` 3 for the gateway, not `helpers.hsql`

- **Decision**: `noesis_db.py` uses `psycopg` 3's async pool
- **Why**: the gateway serves concurrent requests on one event loop; `helpers.hsql`
  (`psycopg2`) blocks it. GP's `postgres_store.py` is unchanged and still uses
  `helpers.hsql`
- **Alternatives**: `helpers.hsql` in a thread pool; an ORM

### D-12. Flat modules next to GP's, original code untouched

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
