# PR_S6 — Offline Checker Replay Validation

- Main review PR: `gpsaggese/gpsaggese.github.io#827`
- Tracking issue: `gpsaggese/gpsaggese.github.io#641`
- Paper-validation issue: `gpsaggese/gpsaggese.github.io#684`
- Plan task: `plan.md` T2.6

## Goal

Validate that the production checker turns controlled latency and canary-quality
evidence into verdicts at the rates predicted by the checker's statistical model.
The experiment must be deterministic, runnable without a provider or database,
and produce artifacts that can be reviewed and cited in the technical report.

## Non-goals

- Measuring a live provider or claiming that a model was nerfed
- Replacing the production evidence, window, statistics, or verdict functions
- Testing the gateway, database, market, reputation adapter, or scheduler
- Selecting the final live-provider calibration thresholds

## Architecture

```text
ReplayScenario
      |
      v
ReplayObservationGenerator ---- deterministic buyer and canary rows
      |
      v
Production checker
  build_evidence() -> assess_window_readiness() -> decide_verdict()
      |                                      |
      v                                      v
Observed verdicts                    Exact binomial prediction
      |                                      |
      +------------------+-------------------+
                         v
             aggregate + acceptance checks
                         |
                         v
                  JSON / CSV / Markdown
```

| Component | Responsibility |
| :-------- | :------------- |
| `checker_replay.py` | Define controlled populations and generate reproducible request observations |
| `checker_replay_runner.py` | Run observations through production checker functions and calculate exact analytic rates |
| `checker_replay_report.py` | Aggregate seeds, apply acceptance rules, and render stable artifacts |
| `scripts/run_checker_replay.py` | Provide the executable experiment entry point and a meaningful exit code |
| `docs/findings/checker_replay/` | Store the checked-in protocol results for review and later paper use |

The production checker is the system under test. The replay code does not copy or
reimplement its decision path.

## Frozen Protocol

- Checker promises: `r_min = 0.90`, `q_min = 0.85`, `l_max = 1.5 s`
- Evidence minima: 20 buyer requests and 8 canaries
- Collection targets: 50 buyer requests and 10 canaries
- Window deadline: 60 seconds
- Statistical rule: one-sided 95% upper confidence bound
- Experiment size: 5 seeds x 1,000 contracts x 6 scenarios = 30,000 contracts

| Scenario | Buyer requests | Canaries | Latency compliance | Canary quality | Purpose |
| :------- | -------------: | -------: | -----------------: | -------------: | :------ |
| `healthy` | 50 | 10 | 0.97 | 0.98 | Estimate false alarms |
| `slow_only` | 50 | 10 | 0.60 | 0.98 | Isolate latency detection |
| `weak_model_only` | 50 | 10 | 0.97 | 0.50 | Isolate quality detection |
| `borderline_quality` | 50 | 10 | 0.97 | 0.80 | Exercise uncertainty near `q_min` |
| `borderline_latency` | 50 | 10 | 0.85 | 0.98 | Exercise uncertainty near `r_min` |
| `insufficient_evidence` | 19 | 7 | 1.00 | 1.00 | Verify that a deadline cannot manufacture a verdict |

## Acceptance

For each scenario, enumerate every possible binomial success count and apply the
same confidence-bound rule used by the production checker. The observed aggregate
failure rate must satisfy both:

1. Absolute error no larger than four binomial standard errors, with a 0.5
   percentage-point floor.
2. Absolute error no larger than the roadmap limit of five percentage points.

The command exits nonzero if any scenario misses either rule.

## Design Decisions

- Reuse production checker functions; do not create an experiment-only checker.
- Keep latency and quality as separate evidence and random streams so failure
  attribution remains visible and one stream cannot perturb the other.
- Derive seeds with SHA-256 instead of Python's process-randomized `hash()`.
- Treat the wall-clock limit as a collection deadline, not as evidence.
- Compare simulation with an exact analytic oracle rather than fixed expected
  snapshots alone.
- Emit Markdown for reviewers, CSV for analysis, and JSON for automation.

The alternatives and rationale are recorded in `decisions.md` D-32 through D-35.

## Run and Review

From the repository root:

```bash
research/Noesis/scripts/run_checker_replay.py
```

The review order is:

1. Read this specification and `design.checker.md`.
2. Inspect scenario generation in `checker_replay.py`.
3. Verify production-path reuse and analytic math in
   `checker_replay_runner.py`.
4. Inspect acceptance and artifact rendering in `checker_replay_report.py`.
5. Run the command and compare its output with
   `docs/findings/checker_replay/checker_replay.md`.
6. Run the focused tests listed in `run_checker_replay.how_to_guide.md`.

## Result

All six scenarios passed both acceptance rules over 30,000 contracts. Latency and
quality failures remained separately attributable, healthy traffic produced no
observed false alarms, and every insufficient-evidence contract remained
`INSUFFICIENT`. These findings validate the offline checker implementation against
its model; live-provider calibration and longitudinal nerfing measurements remain
separate research work.
