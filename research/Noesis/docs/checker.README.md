# Noesis checker (M2)

The checker answers one narrow question: did a seller deliver the latency and
quality promised by a contract? It converts immutable request records into an
explainable `PASSED`, `FAILED`, or `INSUFFICIENT` verdict. This PR establishes
the deterministic logic and persistence boundary; a later PR will run it on a
schedule and send canary traffic.

## Evaluation flow

1. The first valid buyer request starts the contract evaluation window.
2. Buyer requests provide latency evidence. Canary latency is excluded by
   default because short synthetic prompts would bias buyer-experienced latency.
3. Noesis sends deterministic canaries throughout the window. Their normalized
   exact-match grades provide quality evidence.
4. Buyer mistakes, gateway failures, spend-cap refusals, and attribution
   mismatches are excluded. Seller timeouts and upstream errors are failures.
5. Each metric is assessed only after reaching its own evidence minimum.
6. A one-sided confidence upper bound below the promised rate is a conclusive
   failure. Either metric can fail the contract. Passing requires enough
   evidence for both metrics.
7. Checker/configuration errors produce `INSUFFICIENT` with disposition
   `CHECKER_UNHEALTHY`; they never penalize the seller.
8. Finalization atomically freezes request-level evidence, writes one verdict,
   and transitions the contract out of `ACTIVE`.

## Window policy

- Demo: 60 seconds, with enough traffic to reach 20 buyer requests and 8
  canaries.
- Normal MVP: up to 300 seconds.
- A window may finish early once its target traffic is complete.
- A wall-clock deadline is not itself evidence. If a minimum is still missing,
  the result is `INSUFFICIENT`, unless the other metric already has a
  sufficiently sampled conclusive failure.

The timer is therefore a collection deadline, not a request that blocks for
five minutes and not a periodic database polling loop. An orchestrator should
wake on new request/probe events or the deadline, then read the window once.

## Modules

| Module | Responsibility |
|---|---|
| `checker_models.py` | Immutable inputs and outputs shared by the checker |
| `checker_grading.py` | Versioned deterministic answer normalization/grading |
| `checker_canaries.py` | Canary-bank validation and secret balanced selection |
| `checker_evidence.py` | Seller-attributable evidence and exclusion rules |
| `checker_statistics.py` | Wilson and exact one-sided upper bounds |
| `checker_verdicts.py` | Evidence minima, failure precedence, and verdicts |
| `checker_store.py` | PostgreSQL reads, probe claims, and atomic finalization |
| `migrations/002_checker.sql` | Versioned banks, probes, calibration, evidence |

All decision logic is pure. Database access is kept at the boundary so replaying
the same frozen evidence under the same policy version produces the same result.

## Deliberately not in this PR

- Canary generation or the initial canary dataset
- Live-provider calibration runs
- The prober/scheduler and window-closing worker
- Gateway insertion of graded canary responses
- Reputation updates, dashboard views, and alerting

Those pieces build on this contract in separate, reviewable PRs.
