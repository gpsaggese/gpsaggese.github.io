# Checker offline replay findings

- Overall acceptance: **PASS**
- Seeds: `0,1,2,3,4`
- Contracts per seed and scenario: 1000
- Total simulated contracts: 30000
- Checker promises: latency compliance 90.0%; canary quality 85.0%
- Decision rule: one-sided 95.0% confidence upper bound
- Acceptance: analytic agreement and no more than 5 percentage points

## Aggregate results

| Scenario | Contracts | Passed | Failed | Insufficient | Observed fail | Analytic fail | Difference | Result |
| :------- | --------: | -----: | -----: | -----------: | ------------: | ------------: | ---------: | :----- |
| healthy | 5000 | 5000 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | **PASS** |
| slow_only | 5000 | 1 | 4999 | 0 | 0.9998 | 0.9998 | 0.0000 | **PASS** |
| weak_model_only | 5000 | 912 | 4088 | 0 | 0.8176 | 0.8281 | 0.0105 | **PASS** |
| borderline_quality | 5000 | 4406 | 594 | 0 | 0.1188 | 0.1209 | 0.0021 | **PASS** |
| borderline_latency | 5000 | 3408 | 1592 | 0 | 0.3184 | 0.3319 | 0.0135 | **PASS** |
| insufficient_evidence | 5000 | 0 | 0 | 5000 | 0.0000 | 0.0000 | 0.0000 | **PASS** |

## Failure attribution

| Scenario | Latency failures | Quality failures | Both |
| :------- | ---------------: | ---------------: | ---: |
| healthy | 0 | 0 | 0 |
| slow_only | 4999 | 0 | 0 |
| weak_model_only | 0 | 4088 | 0 |
| borderline_quality | 0 | 594 | 0 |
| borderline_latency | 1592 | 0 | 0 |
| insufficient_evidence | 0 | 0 | 0 |

## Interpretation

The healthy scenario estimates false alarms. The slow-only and
weak-model-only scenarios verify that latency and quality failures
remain separately attributable. Borderline scenarios exercise
uncertain evidence near the promises. The insufficient-evidence
scenario verifies that a deadline does not manufacture a pass or
failure when the minimum evidence is missing.

These are deterministic synthetic replay results. They validate the
checker implementation against its analytic model; they do not make
claims about a live provider or prove that a deployed model was
nerfed.
