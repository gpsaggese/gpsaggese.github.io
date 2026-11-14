# Offline Capability-drift Findings

## Summary

- Overall acceptance: `PASS`
- Provider calls: `0`
- Database writes: `0`
- Trials per scenario: `1000`
- Baseline windows: `5`
- Post-change windows: `15`
- Observations per window: `60`
- Statistical threshold: `p < 0.010`
- Practical threshold: `10.0` percentage points

## Scenario Results

| Scenario | Expected | Detect | False alarm | P90 | Gate |
| :------- | :------- | :----- | :---------- | :-- | :--- |
| `healthy_a_a` | `NO_ALERT` | 0.000 | 0.000 | n/a | `PASS` |
| `full_weak_model_swap` | `DETECT` | 1.000 | 0.000 | 2.000 | `PASS` |
| `small_immaterial_drop` | `NO_ALERT` | 0.027 | 0.027 | 15.000 | `PASS` |
| `threshold_level_drop` | `MEASURE` | 0.873 | 0.000 | 11.000 | `MEASURE` |
| `gradual_degradation` | `DETECT` | 1.000 | 0.000 | 7.000 | `PASS` |
| `partial_substitution_10` | `MEASURE` | 0.044 | 0.000 | 13.000 | `MEASURE` |
| `partial_substitution_25` | `MEASURE` | 0.727 | 0.000 | 12.000 | `MEASURE` |
| `partial_substitution_50` | `DETECT` | 1.000 | 0.000 | 4.000 | `PASS` |
| `latency_only` | `NO_ALERT` | 0.000 | 0.000 | n/a | `PASS` |
| `price_only` | `NO_ALERT` | 0.000 | 0.000 | n/a | `PASS` |
| `availability_failure` | `INSUFFICIENT` | 0.000 | 0.000 | n/a | `PASS` |
| `ungradable_evidence` | `INSUFFICIENT` | 0.000 | 0.000 | n/a | `PASS` |

## Interpretation

- A failed gate is a research result, not a runtime failure
- The detector sees observations and a frozen baseline only
- Scenario labels and intervention timing remain evaluator-only data
- Fingerprint and market decisions remain outside this experiment

## Power Analysis

- Sensitivity points evaluated: `60`
- See `power_analysis.csv` for the sample-size and baseline sweep
- See the SVG plots for detection, false-alarm, and query trade-offs
