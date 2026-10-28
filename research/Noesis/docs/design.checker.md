# Noesis checker (M2)

The checker answers one narrow question: did a seller deliver the latency and
quality promised by a contract? It converts immutable request records into an
explainable `PASSED`, `FAILED`, or `INSUFFICIENT` verdict. The separate
`prober/` coordination component calibrates deterministic questions and sends
hidden canary traffic through active contracts.

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

## Measures and thresholds

- Two separate measures per contract `kappa` over its window `W`:
  - **Latency reliability** `r_lat`: share of eligible _buyer_ requests in `W` with
    `status = ok` and `latency <= l_max`
  - **Quality score** `q`: share of _canary_ requests in `W` graded correct
- Violation rule, applied to each measure on its own:
  - Latency: `upper(r_lat) < r_min`
  - Quality: `upper(q) < q_min`, where `q_min` is market-wide, not chosen by the
    buyer
  - `upper` is the one-sided 95% Wilson upper bound (z = 1.645), or the exact
    Clopper-Pearson bound when n < 10
- Why the upper bound and why two measures: `decisions.md` D-1 and D-2
- An `INSUFFICIENT` verdict is not sent to reputation and is shown on the dashboard

Default parameters for the demo:

| Parameter             | Value                                    | Note                                                    |
| :-------------------- | :--------------------------------------- | :------------------------------------------------------ |
| `r_min`               | 0.90                                     | Low target because 99.9% can't be verified at MVP volume |
| `q_min`               | 0.85                                     | Canary accuracy                                         |
| `l_max`               | 1.5 x p95 latency of healthy providers   | Set per model after calibration                         |
| Canaries per contract | 10                                       |                                                         |

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

## Canaries

- A canary is a short question with one known answer, sent by Noesis's own prober
  (not by the buyer) under each `ACTIVE` contract
- **Bank**: arithmetic word problems, factual one-word answers, and simple
  code-output questions
  - Records are JSONL lines with `id`, `bank_version`, `family`, `prompt`, `answer`,
    `normalizer`, and an optional `generator_seed`
  - Bank size and calibration thresholds: `plan.md` T2.1 and T2.2
- **Calibration**: run on every healthy provider and one weaker model
  - Record the healthy accuracy per provider: this measures the checker's own error
    rate
- **Prober**: sends `canaries_per_contract` canaries (default 10), spread across the
  window and formatted like ordinary buyer prompts
- **Grader**: trim, lowercase, strip punctuation and number formatting, then exact
  match. There is no LLM judge in the MVP
- **Cost**: billed to Noesis (the operator), not the buyer, and tracked separately

## Hand-off to reputation

- The checker publishes each verdict to the market's reputation updater
- In the MVP this is an in-process call (`VerdictSink`, `plan.md` T2.5)
- Keep the interface so that it can become a queue later

## Modules

- See `checker/README.md` for the module table

All decision logic is pure. Database access is kept at the boundary so replaying
the same frozen evidence under the same policy version produces the same result.

## Remaining Integration Work

- The reviewed, paid live-provider calibration run and its generated findings
- The window-closing worker that turns frozen request evidence into a verdict
- The proposed runtime architecture and reviewable implementation split are in
  `spec.PR_S4.md`
- Reputation updates, dashboard views, and alerting

Calibration and scheduled probing are run with
`python3 -m research.Noesis.prober.prober_cli`; see `prober/README.md`.
