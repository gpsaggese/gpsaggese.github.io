# Run the Contract-monitoring Demo

- Purpose: exercise the complete `PR_S4` probe-to-verdict flow with visible,
  deterministic outcomes
- Cost: free
- External services: none
- Runtime: a few seconds

## What Runs

The demo uses the production worker composition, readiness rules, evidence
classifier, confidence-bound verdict engine, report builder, and
`RecordingVerdictSink`.

OpenRouter and PostgreSQL are the only mocked boundaries. Their deterministic
fixtures let every reviewer reproduce the same behavior without credentials or
local infrastructure. PostgreSQL transaction behavior is covered separately by
the checker-store integration tests.

The four contracts demonstrate:

1. Healthy buyer and canary evidence produces `PASSED` and publishes.
2. Conclusively slow buyer evidence produces `FAILED` and publishes.
3. Conclusively incorrect canaries produce `FAILED` and publish.
4. A sparse window reaches its deadline, persists `INSUFFICIENT`, and does not
   publish to the market-facing sink.

## Run It

After completing `onboarding.md` and activating the project virtual environment,
run from the repository root:

```bash
> research/Noesis/scripts/demo_contract_monitoring.py
```

Expected summary:

```text
1) Seeded 4 ACTIVE contracts with buyer evidence.
2) Prober persisted 7 scheduled canaries before checker evaluation.
3) Checker evaluated 4 ready windows:
   contract=1 seller=healthy_seller close=TARGETS_REACHED buyer=2 canary=2 verdict=PASSED published=yes
   contract=2 seller=slow_seller close=TARGETS_REACHED buyer=2 canary=2 verdict=FAILED published=yes
   contract=3 seller=weak_seller close=TARGETS_REACHED buyer=2 canary=2 verdict=FAILED published=yes
   contract=4 seller=sparse_seller close=DEADLINE_REACHED buyer=0 canary=1 verdict=INSUFFICIENT published=no
4) Persistence boundary finalized 4 contracts exactly once.
5) RecordingVerdictSink received 3 actionable reports: 1:PASSED, 2:FAILED, 3:FAILED
   Contract 4 stayed sink-silent because INSUFFICIENT is non-actionable.
```

## Verify It

Run the golden-output test:

```bash
> pytest -q research/Noesis/scripts/test/test_demo_contract_monitoring.py
```

Run the complete checker/prober/demo test surface:

```bash
> pytest -q \
    research/Noesis/checker/test \
    research/Noesis/prober/test \
    research/Noesis/scripts/test/test_demo_contract_monitoring.py \
    research/Noesis/test/test_component_imports.py
```

The optional PostgreSQL tests run when the local test database is available and
skip otherwise.

## Scope Boundary

`RecordingVerdictSink` proves the exact market-facing payload and publication
policy. It does not update reputation or seller eligibility. The real adapter is
intentionally deferred to `PR_M3`, as specified by issue #640.
