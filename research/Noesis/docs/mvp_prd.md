# Noesis MVP — Product Requirements (v0.2)

<!-- Public, technical version. Keep business/funding/IP material out of this repo. -->

- **Date**: 2026-09-29
- **Author**: Javin Ahuja (drafted with Claude)
- **Supersedes**: `docs/context/Noesis_MVP_PRD_v0.1.pdf` (2026-09-25)
- **Reviewers**: the Noesis team, then GP (Prof. Saggese)
- **Built on**: GP's prototype in this directory (`research/Noesis/`)
- **Implementation status**: see `mvp_progress.md`. The gateway (§5.1) is built: `gateway_*.py`, `noesis_*.py`; design in `gateway.README.md`. Paths like `noesis/metering` in §5 name planned components, not files
- **Status**: Draft. Decisions in §9 marked *Proposed* need a yes from Javin, Lucas, and Giles.
  §9 lists them.

### What changed from v0.1

| # | Change | Why |
|---|---|---|
| 1 | **Speed and quality are scored separately** per contract, not merged into one "success rate" | With 10% test questions, a single merged rate hides quality failures. A swapped weaker model gets caught on only **2.6%** of contracts. Scored separately: **83%** per contract and almost certain across two. See §5.2. |
| 2 | Test questions (canaries) are **sent by Noesis's own prober**, not by the buyer | Real buyers won't label their traffic. Building it this way now means the checker works unchanged with real customers later |
| 3 | Canaries are **calibrated**: we keep only questions every healthy provider gets right | Otherwise the checker's own error rate is unknown (the #1 risk in v0.1) |
| 4 | Contracts have a **lifecycle** (PENDING → ACTIVE → CLOSED → PASSED/FAILED), not a synchronous `fulfill()` call | The buyer sends requests over time, so a contract's result only exists after its window closes. GP's `dispatch_contract()` returns a bool right away, which doesn't fit |
| 5 | Auction pricing rule is specified so that no matched party trades at a price it refused | Adding the speed/reliability filter breaks the "midpoint of the last matched pair" rule (§5.3) |
| 6 | Reputation adds **cooldown + probation** | Blocked sellers get no contracts, so under the paper's formula their score can never recover. Without this, one block lasts forever |
| 7 | Every threshold now has a number (§6) | So "done" is testable |
| 8 | Repo decision resolved: **new repo we own** (this one); GP's code is vendored under `research/Noesis/` | Answered open question 1 |

---

## 1. Problem

Buyers purchase LLM inference by the token at a price the provider sets. Nothing
checks that the provider delivered the promised speed and quality: models can be
quietly quantized or swapped, endpoints can be slow or flaky, and nobody is held
accountable. Routers (OpenRouter etc.) choose providers by price and uptime. They don't
check delivered quality against a contract.

## 2. MVP claim (the one thing we must prove)

> **We can detect when an AI provider fails to deliver the speed or quality it
> promised, with a known false-alarm rate, and feed that back so under-delivering
> providers lose future deals.**

Matching buyers to sellers is a solved problem. **Verified delivery** is the new part
and the part reviewers will probe. If the checker fails, nothing else matters.

**Result we want to be able to state after the MVP:** "We ran N contracts across K
real providers over H hours. We planted a model swap and a slowdown and caught both
within X minutes. The false-alarm rate on healthy providers was Y%. The market
automatically routed volume away from the bad provider."

## 3. Users (MVP)

| Role | Who, in the MVP | What they do |
|---|---|---|
| Buyer | `simulator` (our script) | Posts bids on a schedule; sends ordinary prompts under its active contracts |
| Seller | 3–5 real inference providers serving the **same open model**, reached via OpenRouter with the provider pinned | We post one standing ask per provider on its behalf |
| Prober | Part of Noesis (new) | Sends calibrated canary questions to each seller under each active contract |
| Operator | Javin, Lucas, Giles; GP reviews | Runs the stack, watches the dashboard, triggers planted failures, demos |

Buyers are simulated, so we control traffic, can repeat experiments, and hold no
customer data (no PII/ToS exposure before we have privacy rules; GP's plan open
question S-4).

## 4. Scope

**In (MVP):**
- Real providers behind our gateway (GP `PR_S7`), provider pinned, fallbacks off
- OpenAI-compatible `/v1/chat/completions` (GP `PR_S8`), so buyers use standard SDKs
- Every request tagged with contract id + actual serving provider (`PR_S4` schema part)
- Per-request latency check; per-contract latency reliability
- Canary prober + calibrated canary set + exact-match grader (`PR_S11`, simplified)
- Statistical violation test with the **corrected** rule (`PR_S6`, corrected)
- Auction filters on latency + reliability, blocks low-reputation sellers (new)
- Contract lifecycle through the gateway, replacing the coin flip (`PR_M8`, redesigned)
- Seller reputation with decay, cooldown, probation (`PR_M3`)
- Auction on a timer (`PR_M7`)
- Traffic simulator, fault injector, dashboard
- Standalone repo: GP's `helpers` dependency replaced by a small local shim

**Out (later):** payments/credits (`PR_P3`, `PR_P4`), smart routing/fusion/caching/
distillation (`PR_S2,S3,S5,S10,S12`), multiple tiers and cross-tier matching (`PR_M5`),
public price feed (`PR_M4`), blind bids (`PR_M6`), scored compatibility (`PR_M9`),
cloud deploy (`PR_P2`), LLM-judge grading, streaming responses, real customers.

## 5. Functional requirements

Status column refers to GP's code as of upstream commit `3f1a5db`.

### 5.1 Gateway — `noesis/gateway` (GP: `passthrough_proxy.py`, `platform_api.py`)

Status: in-process `Gateway.call(provider, model, prompt)` with fake providers,
per-char cost, log without contract id. `/completions` is a bespoke shape.

| ID | Requirement |
|---|---|
| G1 | `POST /v1/chat/completions` accepts the OpenAI request shape (`model`, `messages`, `max_tokens`, `temperature`). Non-streaming only. Auth: `Authorization: Bearer <api_key>` → buyer account |
| G2 | Contract binding: request carries header `X-Noesis-Contract: <contract_id>`. Gateway rejects (HTTP 409) if the contract isn't ACTIVE or doesn't belong to the caller. Requests without the header are rejected in MVP |
| G3 | Routing: contract → seller → provider slug. Upstream call to OpenRouter with `provider: {"only": [<slug>], "allow_fallbacks": false}` and the pinned model id |
| G4 | Attribution: gateway reads the provider name OpenRouter reports in the response. If it's not the pinned provider, the request is logged `attribution_mismatch=true` and **excluded** from the seller's score (counted separately on the dashboard) |
| G5 | Enforced `max_tokens` cap = task definition (§9 D1). Gateway clamps larger requests |
| G6 | Timeout = `2 × l_max`. A timeout, provider 408/429/5xx, or an error body from the provider is a **failed** request for the seller (`timeout` / `upstream_error`). A bad request from the buyer (`buyer_error`) or our own problem, like bad key, no credits, or a routing config error (`gateway_error`), is not counted against the seller |
| G7 | Log one row per request (schema §7): contract id, seller, pinned + actual provider, model, prompt, completion, latency, prompt/completion tokens, real cost from OpenRouter `usage`, `is_canary`, status |
| G8 | Hard spend cap: gateway refuses upstream calls once cumulative real cost ≥ `SPEND_CAP_USD` for the calendar month |
| G9 | Fault injection (demo only, off by default, loudly logged): per-seller `extra_latency_ms` and `model_override` (e.g., serve a smaller model while reporting the contracted one to the checker) |

### 5.2 Metering — `noesis/metering` (new; GP `PR_S4`, `PR_S6`, `PR_S11`)

Status: checker foundation implemented; prober, calibration runner, and window
orchestration remain. See `checker.README.md`.

**Two separate reliability measures per contract κ over its window W:**

- **Latency reliability** `r_lat(κ)` = share of eligible *buyer* requests in W
  with `status=ok` and `latency ≤ l_max(κ)`. Canary latency is excluded by
  default because short synthetic prompts are not representative buyer traffic.
- **Quality score** `q(κ)` = share of *canary* requests in W graded correct.

**Violation rule (corrected).** For each measure, compute a one-sided 95% **Wilson**
confidence interval (exact Clopper–Pearson when n < 10). Flag the measure only when the
**upper** bound < the promised level:
- latency violation ⇔ `upper(r_lat) < r_min(κ)`
- quality violation ⇔ `upper(q) < q_min` (market-wide, §6)
- **contract FAILED** ⇔ either sufficiently sampled metric has a conclusive
  violation
- **contract PASSED** ⇔ both metrics have enough evidence and neither violates
  its promise
- Otherwise verdict = `INSUFFICIENT` (not reported to reputation; shown on
  dashboard). A sufficiently sampled failure takes precedence if the other
  metric lacks evidence
- Checker/configuration errors also produce `INSUFFICIENT` and never penalize
  the seller

*Why the upper bound:* the null hypothesis is "seller is compliant". We only reject it
when even the optimistic estimate misses the promise. The paper (§5 eq.
`reliability_lower_bound`) and GP's `PR_S6` flag when the **lower** bound is below
`R_min`, which punishes healthy sellers for noise. Worked numbers (exact binomial, reproduce with
`python3 scripts/detection_math.py`):

| Scenario (n per contract) | Paper rule: flag if lower < promise | **Our rule: flag if upper < promise** |
|---|---|---|
| Healthy, true latency-rel 0.97 vs promise 0.90 (n=50) | **44%** of contracts falsely flagged | 0.002% |
| Healthy canary acc 0.98 vs q_min 0.85 (10 canaries) | **100%** falsely flagged | 0.003% |
| Slowed 5× → latency-rel ≈ 0.2 (n=50) | caught | caught 100% |
| Weaker model, canary acc 0.5 (10 canaries) | — | caught 83% per contract; ≥97% within 2 |
| Same weaker model, v0.1 merged metric (60 req, 10 canaries) | — | caught **2.6%** ← why we split the two measures |

**Canaries:**

| ID | Requirement |
|---|---|
| Q1 | Canary bank: ≥ 300 short-answer questions with one canonical answer (arithmetic word problems, factual one-word answers, simple code-output questions). Stored in `data/canaries/*.jsonl` with `id, prompt, answer, normalizer` |
| Q2 | **Calibration run** (Milestone 2): send each question 3× to every healthy provider of the contracted model and to one weaker reference model. Keep questions where healthy accuracy = 100% and the weaker model's accuracy ≤ 60%. Target ≥ 150 surviving questions. Record per-provider healthy accuracy. This measures the checker's own error rate |
| Q3 | Prober sends `canaries_per_contract` (default 10) canaries to the seller under each ACTIVE contract, spread across the window, formatted like ordinary buyer prompts |
| Q4 | Grader: deterministic normalize (trim, lowercase, strip punctuation / number formatting) + exact match. No LLM judge in MVP |
| Q5 | Canary cost is billed to Noesis (operator), not the buyer, and tracked separately |

**Reporting:**

| ID | Requirement |
|---|---|
| M1 | On window close, write `contract_verdicts` row (§7) with n, r_lat, bounds, n_canary, q, bounds, verdict, reason |
| M2 | Publish verdict to the market's reputation updater (in-process call in MVP; interface kept so it can become a queue) |

### 5.3 Market — `noesis/market` (GP: `batch_call_auction.py`, `contract_dispatch.py`)

Status: order book, per-tier call auction, contract schema, HTTP API exist. Matching
ignores latency/reliability. Fulfillment is `random() < 0.9`. No ids/timestamps on orders.

| ID | Requirement |
|---|---|
| K1 | Keep GP's `Bid`/`Ask`/`OrderBook`/`OrderBookStore` design. Add `order_id`, `account_id`, `submitted_at` (GP's own TODO) |
| K2 | **Compatibility filter**: bid β and ask α can match only if `α.c_level == β.c_level_min`, `α.l_typical ≤ β.l_max`, `α.r_typical ≥ β.r_min`, and seller is `eligible` (§5.4) |
| K3 | **Matching**: bids by `p_max` desc, asks by `p_min` asc (stable). For each bid, fill from compatible asks in order. Accept a pair only if the price interval `[max p_min of accepted asks, min p_max of accepted bids]` stays non-empty. Clearing price `p*` = midpoint of the final interval, uniform for all fills in the tier. Guarantees no party trades at a price it refused |
| K4 | Unmatched **asks** carry over to the next round (standing asks). Unmatched **bids** expire (simulator re-bids). GP's code drops both |
| K5 | Scheduler: clear every `T` seconds (default 10 s demo, configurable). An empty round must not stop the scheduler. Keep `POST /rounds/clear` for tests/debug |
| K6 | Each fill → `Contract` with id, round id, buyer, seller, `n_tasks`, tier, `l_max`, `r_min`, `price`, `state=PENDING`. Activated immediately → `ACTIVE` with `window_start` |
| K7 | Window starts on the first valid buyer request and closes when its targets are served or its deadline elapses. Use 60 s for the demo and at most 300 s for normal MVP runs → `CLOSED` → metering verdict → `PASSED` / `FAILED` / `INSUFFICIENT` |
| K8 | Replace `mock_fulfill` coin-flip: contract result comes from the metering verdict. Keep `mock_fulfill` for unit tests only |

### 5.4 Reputation — `noesis/reputation` (new; GP `PR_M3`)

| ID | Requirement |
|---|---|
| R1 | One score `ρ ∈ [0,1]` per seller. New sellers start at `ρ0 = 0.8` |
| R2 | On each PASSED/FAILED verdict: `ρ ← (1−λ)ρ + λ·1[PASSED]`, λ = 0.3 (paper eq. `reputation_update`). INSUFFICIENT: no update |
| R3 | If `ρ < ρ_min = 0.5` → seller `blocked` for `cooldown_rounds = 6` rounds (1 min at T=10 s). It is excluded from matching (K2) |
| R4 | After cooldown → `probation`: ρ reset to `ρ_min`, at most 1 contract per round until 3 consecutive PASSED, then `eligible` |
| R5 | Every change is stored in `reputation_events` (§7) for the dashboard |

With these values a healthy seller survives one false flag (0.8 → 0.56). Two failed
contracts in a row block it (0.8 → 0.56 → 0.39). The planted-failure seller should be
blocked within ~2–4 contracts (under 1 minute at demo cadence).

### 5.5 Simulator, fault injector, dashboard

| ID | Requirement |
|---|---|
| S1 | Simulator: config-driven (YAML) buyers. Each round posts bids with randomized `n_tasks` (20–60), `p_max`, fixed `l_max`/`r_min` for the tier. For each ACTIVE contract it owns, sends prompts from a prompt pool at a configurable rate via the OpenAI SDK pointed at our gateway |
| S2 | Seller asks: one standing ask per provider (`l_typical`, `r_typical` from calibration; `p_min` = provider's real per-token price × task size × (1 + margin)) |
| S3 | Fault injector CLI: `noesis fault --seller X --latency +3000ms` / `--model-swap <smaller-model>` / `--clear`. Scheduled faults via the simulator config |
| S4 | Dashboard (single page, reads DB): per-round clearing price and volume; contracts table with verdict and reason; per-seller reputation over time with block/probation markers; request log with attribution; spend vs cap; canary calibration stats |
| S5 | `make demo` runs everything locally with one command (Docker Compose: api + postgres + simulator + dashboard) |

## 6. Default parameters (demo)

| Parameter | Value | Note |
|---|---|---|
| Auction interval `T` | 10 s | GP v0.3 target |
| Tier | 1 (`standard`) | |
| Model | Llama 3.3 70B Instruct (confirm ≥ 4 OpenRouter providers in week 1) | D3 |
| Weaker model for calibration and model-swap fault | Llama 3.1 8B Instruct | |
| Task | 1 chat completion, `max_tokens ≤ 256`, prompt ≤ 1,000 tokens | D1 |
| `l_max` | 1.5 × p95 latency of healthy providers in calibration | Set per-model after calibration |
| `r_min` (latency reliability) | 0.90 | Low target because 99.9% can't be verified at MVP volume |
| `q_min` (canary accuracy) | 0.85 | |
| Canaries per contract | 10 | |
| `n_min` / `c_min` | 20 requests / 8 canaries | Below → INSUFFICIENT |
| Window | first valid buyer request until targets complete; 60 s demo, ≤300 s normal MVP | Collection deadline, not polling cadence |
| Confidence | one-sided 95% (z = 1.645) | |
| Reputation | ρ0 0.8, λ 0.3, ρ_min 0.5, cooldown 6 rounds, probation 3 passes | |
| Spend cap | $50/month (Proposed, D9) | 1-hr demo ≈ 1–3 M tokens ≈ $1–3 at 70B prices; verify in week 1 |

## 7. Data model (shared contract between server and market halves; freeze in M0)

```
accounts(account_id PK, kind[buyer|seller|operator], api_key_hash, created_at)
sellers(seller_id PK, account_id FK, provider_slug, provider_name, model_id, state[eligible|blocked|probation],
        reputation, blocked_until_round, probation_passes)
bids(order_id PK, account_id, n_tasks, c_level_min, l_max, r_min, p_max, submitted_at, round_id NULL)
asks(order_id PK, seller_id, n_tasks, c_level, l_typical, r_typical, p_min, submitted_at, standing BOOL)
rounds(round_id PK, tier, started_at, cleared_at, clearing_price NULL, matched_n_tasks,
       unfilled_bid_n_tasks, unfilled_ask_n_tasks)
contracts(contract_id PK, round_id FK, bid_order_id FK, ask_order_id FK, buyer_account_id, seller_id, n_tasks, c_level, l_max, r_min,
          price, state[PENDING|ACTIVE|CLOSED|PASSED|FAILED|INSUFFICIENT], window_start, window_end)
requests(request_id PK, contract_id FK NULL, seller_id, pinned_provider, actual_provider,
         attribution_mismatch BOOL, model_requested, model_served, is_canary BOOL, canary_id NULL,
         canary_correct NULL, prompt, completion, prompt_tokens, completion_tokens, latency_ms,
         status[ok|timeout|upstream_error|buyer_error|gateway_error|refused_cap], cost_usd, fault_injected BOOL,
         created_at)
contract_verdicts(contract_id PK, n, r_lat, r_lat_upper, n_canary, q, q_upper, verdict, reason, computed_at)
reputation_events(id PK, seller_id, contract_id NULL, round_id, rho_before, rho_after, state_before,
                  state_after, created_at)
canaries(canary_id PK, prompt, answer, normalizer, calibrated BOOL, healthy_acc, weak_acc)
```

GP's existing `postgres_store.py` tables (`noesis_*`) are migrated to this schema. Keep the
`*Store` ABC + in-memory implementation pattern for tests.

## 8. Success criteria (all must hold in one recorded live run)

1. **Autonomy**: loop runs ≥ 60 min with ≥ 3 real providers, zero manual steps after `make demo`.
2. **Attribution**: 100% of requests have `contract_id` and `actual_provider`. Attribution
   mismatches < 1% and all excluded from scoring.
3. **Planted slowdown caught**: at minute 15, `+3000 ms` on seller A. First FAILED verdict
   within 2 contract windows. A blocked within 5 min. Share of matched volume A wins
   drops to 0 while blocked.
4. **Planted model swap caught**: at minute 35, seller B serves the 8B model. First FAILED
   (quality) verdict within 3 contracts. B blocked within 5 min.
5. **No false alarms**: healthy sellers: ≤ 1 FAILED verdict each over the hour and **never
   blocked**.
6. **Recovery**: after the fault clears, the seller goes through cooldown → probation →
   eligible.
7. **Legibility**: someone who has never seen the code can read all of the above off
   the dashboard in < 5 min. Test with GP or another reviewer.
8. **Cost**: total real spend for the run ≤ $10.

Deliverables: the recorded run (video + exported DB), a one-page results memo with the
numbers above, and a reproducible `make demo`.

## 9. Decisions

| # | Decision | v0.1 proposal | v0.2 recommendation | Status |
|---|---|---|---|---|
| D1 | Task unit | 1 request, capped tokens | 1 chat completion, ≤ 256 output / ≤ 1,000 input tokens | Proposed |
| D2 | Tiers | 1 | 1 | Proposed |
| D3 | Model | Llama 3.x 70B | Llama 3.3 70B Instruct. Week-1 check that ≥ 4 providers serve it on OpenRouter, otherwise pick the open model with the most providers | Proposed |
| D4 | Reaching providers | OpenRouter, pinned, no fallbacks | OpenRouter, plus verify the served provider in the response (G4). **Alternative: Concentrate AI.** Its default auto-failover breaks attribution, and it may not list one open model from many hosts. Adopt it only if they confirm *"fallback can be disabled per request and the response names the upstream provider"*. Post-MVP: self-hosted LiteLLM or direct provider APIs, so we own attribution | Proposed |
| D5 | Quality method | ~10% canaries mixed into traffic | Prober-injected canaries, calibrated set, **scored separately from latency** | **Changed** |
| D6 | Violation rule | upper CI < promise | Same, Wilson one-sided 95%, applied to each measure separately | Proposed |
| D7 | Auction interval | 10 s | 10 s demo, configurable | Proposed |
| D8 | GP's `helpers` lib | Ask GP; default his dev env | **Use GP's real `helpers` in this repo.** (A ~150-line stand-in exists only in the team's backup repo, to run without the monorepo) | Decided |
| D9 | Who pays API | Us, hard cap | Us, $50/mo cap enforced in gateway (G8). Apply for OpenRouter/provider startup credits | Proposed |
| D10 | Repo | open question | **This monorepo (`research/Noesis/`) is primary; every change lands as a reviewed PR.** The team keeps a private backup repo | Decided |
| D11 | Owners | one server, one market | Gateway + metering math; checker data (canaries, calibration, prober) + simulator + tooling; market + reputation + dashboard. Names in `mvp_plan.md` | Proposed |
| D12 | Stack | — | Python 3.11, FastAPI, Postgres, `httpx`, `openai` SDK (simulator), Streamlit dashboard (fastest path; replace later) | Proposed |

## 10. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Canary check too noisy | Can't tell good from slightly worse | Calibrated set (Q2), clear-cut questions, publish the checker's own error rate. MVP targets gross failures (model swap, slowdown). Subtle quantization detection is a post-MVP research result |
| Canaries distinguishable from real traffic | Provider games the prober | MVP: use buyer-style formatting. Later: generate canaries from real prompt distribution, rotate the bank |
| OpenRouter attribution breaks | Wrong seller blamed | `only` + `allow_fallbacks:false`, verify reported provider, exclude mismatches |
| Too few providers for the model | < 3 sellers | Week-1 check, D3 fallback |
| Top-tier supply is thin | OpenAI, Anthropic, and Google won't bid, so the frontier tier is empty or collusion-prone | MVP uses one open-weight model with many hosts. Frontier tier waits for real supply |
| Gateway intermediary hides who served a request | Reputation assigned to the wrong seller | OpenRouter pinning now. Own the gateway (LiteLLM or direct APIs) before any real customers |
| Competitors add metering | Wedge commoditized | Moat is the longitudinal, request-level verified-quality dataset. Study OpenRouter, Martian, Not Diamond, Artificial Analysis, Concentrate AI (plan §5) |
| High reliability hard to verify | 99.9% needs ~thousands of requests | Demo at 0.90. The minimum-detectable-shortfall table (GP draft `research/ideas/draft.Noesis_Reputation_Detection_Bound.md`) is a paper result |
| Provider ToS on resale/logging | Legal exposure | Read each provider's + OpenRouter's terms before any public claim. MVP has no resale (we're the only buyer) |
| Part-time bandwidth | Slips | GP's rule: daily progress. Weekly demo of whatever runs |

## 11. Open questions for GP

1. OK with the corrected violation rule (upper bound) and splitting latency/quality?
   (Changes paper §5 eq. `reliability_lower_bound` and `PR_S6`.)
2. Server-first order (`mvp_plan.md`) vs the market-first roadmap in `plan.Noesis.md`: fine?
3. Should `PR_S9` (auto-asks from the OpenRouter catalog) and `PR_P4` (free-credit
   ledger) be in the MVP? Both are on the v0.2 / v0.4 lists and currently deferred.
4. Is Noesis its own gateway, or a metering/market layer on top of existing gateways?

## 12. Glossary

Provider: company running a model (Together, Fireworks, DeepInfra…). Gateway: our
proxy. Bid/Ask: buyer/seller order. Tier: quality class. Contract: a matched deal
we check. Window: the requests a contract covers. Canary: test question with a
known answer, sent by our prober. Latency reliability: share of requests that were
on time and succeeded. Quality score: share of canaries correct. Wilson interval: the
confidence interval for a proportion. Reputation ρ: seller score in [0,1]. Attribution:
knowing which provider served a request.
