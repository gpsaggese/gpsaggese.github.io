# Noesis MVP — Execution Plan

<!-- Public, technical version. Keep business/funding/IP material out of this repo. -->

- **Date**: 2026-09-29
- **Spec**: `mvp_prd.md` (v0.2)
- **Target**: recorded demo that meets all PRD §8 success criteria by **Fri 2026-12-11**
  (~11 weeks, three people part-time)
- **Owners (proposed, PRD D11)**:
  - **J** = Javin Ahuja: gateway + metering math (critical path)
  - **G** = Giles Greene: market, reputation
  - **L** = Lucas: canaries, calibration, prober, simulator, fault injector, tooling,
    dashboard
  - New contributors pick tasks from §3 (see `onboarding.md`)
- **Cadence**: GP's rule is to make progress every day. One commit or one written
  finding per person per day. Short async standup (done / next / blocked). Friday
  demo of whatever runs.

---

## 1. Where GP's code stands (upstream `3f1a5db`)

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

## 2. Milestones and dates

Live status and history: `mvp_progress.md`.

| # | Milestone | Dates | Owner | Exit test | Status |
|---|---|---|---|---|---|
| M0 | Setup + facts | Tue 9/29 – Fri 10/9 | J+G+L | GP's tests pass locally and in CI with no `helpers` monorepo. OpenRouter spike confirms ≥ 3 pinnable providers for the model. Schema frozen. Decisions signed | ✅ 9/30 (GP email pending) |
| M1 | Real server | Mon 10/12 – Fri 10/30 | J (+L: T1.5, T2.1 starts) | Unmodified `openai` Python SDK → our gateway → 3 pinned providers. Every log row has `contract_id` + verified `actual_provider` | ✅ 10/4 (spend cap deferred) |
| M2 | Checker | Mon 11/2 – Fri 11/20 | J + L | Calibrated canary set ≥ 150. Offline replay: injected slowdown and model swap flagged, healthy traffic not flagged, rates within ±5 pts of `detection_math.py` | ⬜ next |
| M3 | Loop closed (parallel with M1–M2) | Mon 10/12 – Fri 11/20 | G | With fake gateway: auction on timer, filter works, contracts go ACTIVE→verdict, a failing seller is blocked, cooled down, put on probation, reinstated | ⬜ |
| M4 | Integrated demo | Mon 11/23 – Fri 12/11 | J+G+L | PRD §8 criteria 1–8 in one recorded 60-min run | ⬜ |

Buffer: Thanksgiving week (11/23) is counted as half capacity. If M2 slips more than
1 week, cut the dashboard to a read-only DB view plus a CLI report. Don't cut the
planted-failure test.

---

## 3. Task breakdown

Every task: branch → PR → tests green in CI → one reviewer (another teammate).
"GP-PR" names the item from `research/Noesis/plan.Noesis.md` it closes or replaces.

### M0 — Setup + facts (all three)

| ID | Task | Owner | Acceptance | GP-PR |
|---|---|---|---|---|
| T0.1 | ✅ 9/29 Team backup repo created (private); this monorepo is primary since 10/6 (PRD D10) | J | — | — |
| T0.2 | ✅ 9/29 Run GP's code without the monorepo: done in the team's backup repo with a small `helpers` stand-in. **In this repo the real `helpers` is used** | J | 63/63 upstream tests pass | resolves D8 |
| T0.3 | ✅ Tooling: deps added to `devops/docker_build/pyproject.toml` (+ `poetry.lock`); local test Postgres on :5433 (`onboarding.md`). GP's CI only runs course dirs, so Noesis tests run locally / in the dev container | L | Tests green locally | — |
| T0.4 | ✅ Superseded: new code lives as flat modules next to GP's (`noesis_*.py`, `gateway_*.py`, tests in `test/`), matching this directory's layout | J | — | — |
| T0.5 | ✅ 9/30 run: 8/11 endpoints usable, 100% attribution (`findings/openrouter_spike.md`) — **OpenRouter spike** `scripts/spike_openrouter.py`: list providers for Llama 3.3 70B. For each: 20 calls with `provider.only=[slug], allow_fallbacks=false`. Record served provider (from response), latency p50/p95, error rate, `usage` cost | J | `docs/findings/openrouter_spike.md` with table. Decide D3/D4 from data | de-risks PR_S7 |
| T0.6 | ✅ `migrations/001_init.sql` (applied automatically by `noesis_db.apply_migrations()`) — Freeze DB schema (PRD §7). All three sign | J+G+L | Merged; all agree | — |
| T0.7 | Sign off PRD decisions D1–D12. Send PRD + this plan to GP with the §11 questions | J | Written replies logged in `mvp_decisions.md` | — |

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
| T2.1 | Canary bank ≥ 300 (`data/canaries/*.jsonl`) + normalizers + exact-match grader | Grader unit tests incl. number formats ("1,000" = "1000") | PR_S11 (simplified) |
| T2.2 | **Calibration run**: each canary 3× per healthy provider + 3× on the 8B model. Keep healthy=100%, weak ≤ 60%. Also measure latency p95 per provider → set `l_max` | `docs/findings/calibration.md`: survivors ≥ 150, weak-model accuracy on survivors, per-provider p95, cost | PR_S11 |
| T2.3 | Prober: for each ACTIVE contract, send `canaries_per_contract` canaries spread over the window, marked `is_canary` server-side only | Integration test with fake provider | new |
| T2.4 | Metering: close windows (n_tasks served or 60 s), compute `r_lat`, `q`, Wilson/Clopper–Pearson upper bounds, verdict + reason, INSUFFICIENT rule | Unit tests reproduce `scripts/detection_math.py` cases. Property test: never FAILED when all requests pass | PR_S4, PR_S6 (corrected) |
| T2.5 | `VerdictSink` interface → reputation (in-process) | Contract test shared with T3.4 | PR_S4 |
| T2.6 | **Offline replay harness**: generate synthetic request streams (healthy / slowed / weak-model at measured accuracies), run metering, report flag rates | Rates within ±5 pts of analytic numbers over 1,000 simulated contracts | new |

### M3 — Loop closed (G, starts 10/12 in parallel)

| ID | Task | Acceptance | GP-PR |
|---|---|---|---|
| T3.1 | Orders get `order_id`, `account_id`, `submitted_at`. `Fill` carries bid/ask order ids (fixes the multi-bid bug). Standing asks persist, bids expire | GP's auction tests still pass. New tests for carry-over + multi-bid buyer | PR_M1 fix |
| T3.2 | Compatibility filter (tier, latency, reliability, seller eligible) + IR-safe uniform pricing (PRD K3) | Tests: incompatible cheap ask skipped. No fill priced outside any matched party's limit (property test over random books) | new |
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
| T4.4 | Three rehearsal 60-min runs. Tune `l_max`, window, canaries/contract if false alarms or misses | J+G+L | Run 3 meets §8 |
| T4.5 | **Recorded run** + exported DB + 1-page results memo (numbers for the results line in PRD §2) | J+G+L | Memo in `docs/results/` |
| T4.6 | Walk GP through it. Collect his feedback on the paper's violation rule change | J | Notes in `docs/decisions.md` |

## 4. Critical path

```
T0.5 spike ─► T1.1 ─► T1.3/T1.4 ─► T2.2 calibration ─► T2.4 metering ─► T4.4 rehearsals ─► T4.5
T3.1 ─► T3.2 ─► T3.3 ─► T3.4 ───────────┘
```

The OpenRouter spike (T0.5) and calibration (T2.2) are the two points where the data
might force a change of plan: model, provider count, canary design. Do them early.

## 5b. Side track: provider independence (not blocking the MVP)

Decision 10/4: OpenRouter stays for the MVP. In parallel, low priority, work out how
Noesis reaches providers without depending on OpenRouter (Tier 2):
- Direct accounts + adapters for the demo sellers (most are OpenAI-compatible, so one
  generic adapter + per-provider config behind the existing `Provider` interface)
- Cross-check: same probes via OpenRouter vs direct, compare answers and latency
- Triggers to switch: pinned≠served >1%, OpenRouter outages distorting scores, policy
  or pricing changes, first paying customer

## 6. After the MVP (not scheduled)

First productize the checker as the **wedge**: a verified-delivery monitoring gateway for teams spending $5k–$50k/mo (PRD §9b). Then, roughly in value order: real design-partner buyers (PII
scrubbing first, GP S-4), direct provider APIs alongside OpenRouter, subtle-degradation
detection (quantization) with a larger canary bank or reference-model agreement,
public price/reputation feed (`PR_M4`), credits ledger (`PR_P4`), cloud deploy
(`PR_P2`), multiple tiers (`PR_M5`), routing/fusion savings (`PR_S2`, `PR_S5`).
