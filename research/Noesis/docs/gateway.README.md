# Noesis Gateway

- The gateway is the MVP's `NoesisServer` front door: an OpenAI-compatible HTTP
  server that routes each request to the one provider its contract names, measures
  it, and logs everything the checker needs
- It replaces the `PR_S1` passthrough proxy's fake providers with real ones
  (`PR_S7`), adds an OpenAI-compatible API (`PR_S8`), and the contract field on
  every logged request (`PR_S4`)
- Spec: `mvp_prd.md` §5.1; decisions and alternatives: `mvp_decisions.md`

## Where It Fits

```mermaid
flowchart TD
    buyers["1. Buyers (simulator)<br/>post bids"] --> auction["2. Auction<br/>(milestone 3)"]
    auction --> contract["3. Contract<br/>one per match"]
    contract --> gateway["4. Gateway<br/>(this doc)"]
    gateway <--> providers["Real providers<br/>via OpenRouter"]
    gateway --> prober["5. Prober<br/>(milestone 2)"]
    prober --> checker["6. Checker<br/>(milestone 2)"]
    checker --> reputation["7. Reputation<br/>(milestone 3)"]
    reputation -. "blocked sellers skipped<br/>next round" .-> auction
```

- Built: step 4 and its database. Until the auction exists, contracts are created by
  hand with `POST /admin/dev-contract`

## Files

| File | Role |
|---|---|
| `noesis_settings.py` | `Settings` from `research/Noesis/.env` + environment; loud failure on bad values |
| `noesis_db.py` | Async `psycopg` 3 pool; migrations runner (each `migrations/*.sql` once, in order, all-or-nothing) |
| `noesis_seed.py` | Loads `config/seed.yaml`: demo sellers + buyer/operator accounts; keys stored as SHA-256 hashes |
| `gateway_providers.py` | The `Provider` interface: `ProviderCall` in, `ProviderResult` out; status constants |
| `gateway_openrouter.py` | `OpenRouterProvider`: pinned upstream, no fallbacks, real cost; `classify_error()` blame rules |
| `gateway_fake_provider.py` | `FakeProvider` for tests: scripted latency/failures/answers/impersonation |
| `gateway_routing.py` | API key -> account; contract -> seller -> provider (`Route`); `RoutingError` |
| `gateway_request_log.py` | One `requests` row per provider call; attribution check |
| `gateway_faults.py` | Demo fault switches: per-seller slowdown and model swap |
| `gateway_admin.py` | `create_dev_contract()` until the auction exists |
| `gateway_api.py` | HTTP endpoints |
| `gateway_app.py` | App factory: startup (connect, migrate, seed), error handlers, real server |
| `migrations/001_init.sql` | 10-table MVP schema (`mvp_prd.md` §7) |
| `config/seed.yaml` | The 5 demo sellers (from `findings/openrouter_spike.md`) + accounts |
| `test/test_noesis_*.py`, `test/test_gateway_*.py` | Unit and end-to-end tests; `test/gateway_test_utils.py` has shared DB helpers |

## Import Graph

```mermaid
flowchart LR
    app[gateway_app] --> api[gateway_api]
    app --> db[noesis_db] & seed[noesis_seed] & settings[noesis_settings] & openrouter[gateway_openrouter]
    api --> routing[gateway_routing] & request_log[gateway_request_log] & faults[gateway_faults] & admin[gateway_admin]
    routing --> seed
    request_log --> routing & providers[gateway_providers]
    faults --> providers
    openrouter --> providers
    fake[gateway_fake_provider] --> providers
```

- Only `gateway_openrouter.py` knows about OpenRouter. Routing, logging, and faults
  only see the `Provider` interface, so a direct-provider adapter (e.g., calling Groq's
  own API) is a new file, not a rewrite (`mvp_plan.md` §5b)
- GP's original modules (`batch_call_auction.py`, `passthrough_proxy.py`, ...) are
  untouched; the market work (milestone 3) builds on `batch_call_auction.py`

## One Request, End to End

`POST /v1/chat/completions` with headers `Authorization: Bearer <buyer key>` and
`X-Noesis-Contract: 42`:

1. `parse_bearer()` -> key; `authenticate(kind="buyer")` -> account (401 / 403)
2. `parse_contract_id()` -> 42 (400); `resolve_route()` -> `Route` (404 unknown or
   someone else's, 409 not ACTIVE)
3. Reject `stream=true` (400) and models other than the served one (404)
4. Clamp `max_tokens` to `NOESIS_MAX_TOKENS_CAP` (task definition, PRD D1);
   timeout = `NOESIS_TIMEOUT_MULTIPLIER` x contract `l_max`
5. `call_with_fault()`: apply the seller's demo fault if any, then
   `provider.call(route.provider_slug, call)`
6. `log_request()`: one `requests` row, success or failure
7. Return an OpenAI chat-completion body, or an OpenAI-style error, with headers
   `X-Noesis-Request-Id`, `X-Noesis-Contract`, `X-Noesis-Provider`

- Requests rejected in steps 1-3 never reach a provider and are not logged in
  `requests` (they can't affect any seller's score)

## Blame Rules

| What happened | Status | HTTP to buyer | Counts against seller |
|---|---|---|---|
| Answer received | `ok` | 200 | — |
| Slower than the timeout | `timeout` | 504 | Yes |
| OpenRouter says the provider failed (`"Provider returned error"` or `metadata.provider_name`) | `upstream_error` | 502 | Yes |
| OpenRouter's own 5xx/429 with no provider named | `gateway_error` | 503 | No |
| Our key / credits / pin (401, 402, 403, 404) | `gateway_error` | 503 | No |
| Can't reach OpenRouter | `gateway_error` | 503 | No |
| Malformed request (400, 413, 422) | `buyer_error` | 400 | No |
| 200 with an error inside naming the provider, or no answer | `upstream_error` | 502 | Yes |

- An answer whose `served_by` isn't the pinned provider (or is missing) is logged
  with `attribution_mismatch = true` and excluded from the seller's score
- A garbage answer (e.g., `!!!!!!!!`) is still `ok` here: the gateway only records
  that the provider answered; the checker (milestone 2) grades content

## Demo Faults

| Fault | Effect | Logged as |
|---|---|---|
| `{"extra_latency_s": 3}` | Real 3 s delay before the call; provider gets the remaining timeout; delay >= timeout -> `timeout` | `fault_injected = true`, latency includes the delay |
| `{"model_override": "meta-llama/llama-3.1-8b-instruct"}` | Same provider serves the 8B model; buyer still sees the contracted model name | `model_requested` 70B, `model_served` 8B, `fault_injected = true` |
| `{"model_override": ..., "slug_override": "deepinfra/fp8"}` | Same, when the seller's tag doesn't serve the override model | as above |

- Set with `PUT /admin/faults/{seller_id}`, clear with `DELETE`, list with `GET
  /admin/faults` (operator key). Faults are in memory and reset on restart
- Groq, DeepInfra, Novita, and CoreWeave serve Llama 3.1 8B on OpenRouter; Together
  doesn't, so it can't be a model-swap target

## Endpoints

| Endpoint | Auth | Purpose |
|---|---|---|
| `POST /v1/chat/completions` | buyer | OpenAI-compatible completion under a contract |
| `GET /v1/models` | buyer | Lists the one served model |
| `GET /health` | none | Server + database up |
| `POST /admin/dev-contract` | operator | Create an ACTIVE contract by hand (temporary) |
| `GET/PUT/DELETE /admin/faults[/{seller_id}]` | operator | Demo fault switches |

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | — | Required for the real server |
| `NOESIS_BUYER_1_KEY`, `NOESIS_OPERATOR_KEY` | — | Our API keys, hashed into `accounts` at startup |
| `NOESIS_DATABASE_URL` | `postgresql://noesis:noesis@localhost:5433/noesis_dev` | Server DB |
| `NOESIS_MODEL_ID` | `meta-llama/llama-3.3-70b-instruct` | The one served model (PRD D3) |
| `NOESIS_MAX_TOKENS_CAP` | 256 | Completion cap (PRD D1) |
| `NOESIS_TIMEOUT_MULTIPLIER` | 2.0 | Timeout = this x contract `l_max` |
| `NOESIS_SPEND_CAP_USD` | 50 | Read but not enforced yet (`mvp_decisions.md`) |
| `NOESIS_TEST_DATABASE_URL` | `...:5433/noesis_test` | Test DB (tests skip if unreachable) |

- Template: `research/Noesis/env.example`. Setup: `onboarding.md`

## Known Limits

- Non-streaming only; one model; one tier
- No spend cap of our own yet (the OpenRouter key's credit limit is the cap)
- Fault switches are in memory
- Attribution is OpenRouter's report; direct-provider adapters are the planned fix
  (`mvp_plan.md` §5b)
