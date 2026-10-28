# Noesis MVP — Progress

- What's done, what we learned, what's next. Canonical roadmap:
  `plan.md` (roadmap and MVP execution plan); decisions:
  `decisions.md`

## Status

| Milestone | Target | Status |
|---|---|---|
| M0 Setup | 10/9 | ✅ 9/30 |
| M1 Gateway | 10/30 | ✅ PR #597 merged 10/10 (spend cap deferred, `decisions.md` D-9) |
| M2 Checker | 11/20 | 🟨 foundation merged in PR #604 on 10/11; candidate bank, calibration runtime, and scheduled probing implemented 10/17; paid calibration and window close remain |
| M3 Market loop | 11/20 | Not started; builds on `batch_call_auction.py` |
| M4 Demo | 12/11 | Not started |

- Tests: PR #604 added 28 focused checker/database tests; GP's original tests were
  unchanged
- Live-verified: OpenAI SDK -> gateway -> OpenRouter -> Groq / CoreWeave, every
  request logged with contract, seller, served provider, latency, cost

## Built

- Gateway and its database: see the file table in `design.gateway.md`
- Checker foundation: deterministic grading and selection, evidence classification,
  separate latency/quality confidence bounds, verdicts, and PostgreSQL persistence
  (`design.checker.md`)
- Checker runtime: versioned 300-question candidate bank, pluggable capability
  measurement, persisted calibration with an activation gate, secret scheduled
  probing, and server-owned graded evidence
- `scripts/detection_math.py`: exact false-alarm / detection rates for the
  violation rules (`decisions.md` D-1, D-2)
- `scripts/spike_openrouter.py` + `findings/openrouter_spike.md`: real provider
  measurements

## Findings

1. **Pinning works on OpenRouter** (9/30): 8 of 11 Llama 3.3 70B endpoints usable;
   100% of served calls came from the pinned provider; the whole check cost $0.0026
2. **Same model, different quality**: providers serve fp8 / bf16 / fp16 versions;
   Novita (bf16) once answered a math question with `!!!!!!!!`, caught by a
   known-answer probe on the first test
3. **~10x latency spread** for the same model (p95 0.26 s CoreWeave -> 2.44 s
   SambaNova)
4. **The paper's violation rule flags healthy sellers 44–100% of the time**; the
   upper-bound rule ~0.002% (D-1). Merging speed and quality catches a model swap
   2.6% of the time, so they're scored separately (D-2)
5. **Easy questions can't expose a model swap** (10/4): Llama 3.1 8B answered
   "9.9 vs 9.11" correctly; canaries must be calibrated (D-3)
6. **Graders must normalize**: Groq answered `Canberra.`, CoreWeave `Canberra`
7. **Sync folders corrupt git repos**: iCloud duplicated files inside `.git`;
   clone outside synced folders (`onboarding.md`)

## Next (M2 — Checker runtime and data)

1. Run reviewed paid calibration: keep questions 70B always gets right and 8B misses; set
   `l_max` from measured latency
2. Add the window-closing worker and market-facing `VerdictSink`
3. Add offline replay for healthy, slowed, and weak-model traffic

## History

| Date | Done |
|---|---|
| 9/29 | GP's prototype running; PRD v0.2 + plan; schema; spike script |
| 9/30 | OpenRouter spike: 8 usable providers |
| 10/1 | Settings, async DB pool, migrations |
| 10/4 | Seed data; provider interface + OpenRouter adapter (blame rules) + fake provider; routing; request log; OpenAI-compatible endpoints; demo faults. **M1 complete** |
| 10/6 | Ported into `research/Noesis/` following the repo's conventions; first PR prepared |
| 10/10 | PR #597 merged: setup, persistent gateway, OpenRouter adapter, routing, logging, and fault injection |
| 10/11 | PR #604 merged: checker contracts, deterministic grading, evidence/statistics/verdict logic, and persistence |
| 10/17 | Added the v1 candidate bank, live-calibration runtime and report, pluggable capability measurement, and scheduled prober; paid calibration remains pending review |
| 10/18 | PR_S9: supply adapter turns OpenRouter endpoint prices into one ask per seed seller before each round (`market/supply_adapter.py`, `noesis_platform/supply_feed.py`; D-25 to D-27) |
| 10/23 | Split standalone model monitoring and nerfing detection into `PR_S13`-`PR_S16`; no runtime behavior changed |
