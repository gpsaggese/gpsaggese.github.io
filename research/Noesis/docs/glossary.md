# Noesis — Glossary

- Plain-language definitions of the terms used in the code and docs

| Term | Meaning |
|---|---|
| Provider | A company that runs a model and sells access to it (Groq, Together, DeepInfra, ...) |
| Seller | A provider participating in the Noesis market; in the MVP, one per provider (`sellers` table) |
| Buyer | Whoever sends requests and pays; in the MVP, a simulator (`accounts` with kind `buyer`) |
| Operator | Us: runs the system, creates dev contracts, flips demo faults (kind `operator`) |
| Gateway | Our server between buyers and providers; routes, measures, and logs every request |
| OpenRouter | A third-party router that gives one API to many providers; our transport for the MVP |
| Pinning | Forcing OpenRouter to use exactly one provider for a call (`provider.only`, no fallbacks) |
| Slug / tag | OpenRouter's id for one provider endpoint, e.g., `groq`, `deepinfra/turbo` |
| Attribution | Knowing which provider actually produced an answer (`served_by` vs the pinned one) |
| Attribution mismatch | An answer from a different or unnamed provider; excluded from scoring |
| Bid | A buyer's order: how many tasks, minimum tier, max latency, min reliability, max price |
| Ask | A seller's offer: how many tasks, tier, typical latency/reliability, min price |
| Tier | Capability class (e.g., `standard`); the MVP has one |
| Auction / clearing | Matching bids to asks every T seconds and setting one price per tier |
| Contract | A matched deal we can check: tasks, tier, `l_max`, `r_min`, price, state |
| Task | One chat completion, at most 256 output tokens (PRD D1) |
| `l_max` | The contract's latency limit in seconds; the request timeout is 2x this |
| `r_min` | The contract's promised reliability (share of on-time successful requests) |
| Window | The requests a contract covers: its task count or 60 s, whichever comes first |
| Canary / test question | A question with one known answer, sent by our prober to grade quality |
| Calibration | Keeping only canaries every healthy provider gets right and a weaker model misses |
| Prober | The part of Noesis that sends canaries under each active contract |
| Latency reliability | Share of a contract's requests that succeeded within `l_max` |
| Quality score | Share of a contract's canaries answered correctly |
| Confidence interval / upper bound | The range the true rate probably lies in; we flag only if even the upper end misses the promise |
| Verdict | A contract's result: PASSED, FAILED, or INSUFFICIENT (too little data) |
| Reputation (rho) | A seller's score in [0, 1], updated after each verdict; low scores get blocked |
| Cooldown / probation | After a block: sit out N rounds, then return on probation until 3 passes |
| Blame / status | Who caused a failed call: `timeout` / `upstream_error` (seller), `gateway_error` (us), `buyer_error` (buyer) |
| Fault injection | Deliberately slowing a seller or swapping its model to prove the checker catches it |
| Model swap | Serving a weaker model (Llama 3.1 8B) while claiming the contracted one (70B) |
| Quantization (fp8 / bf16 / fp16) | Running a model at lower numeric precision: cheaper, sometimes lower quality |
| p50 / p95 | Typical (median) and near-worst (95th percentile) latency |
| `PR_S7`, `PR_M3`, ... | Work items in GP's roadmap `research/Noesis/plan.Noesis.md` |
