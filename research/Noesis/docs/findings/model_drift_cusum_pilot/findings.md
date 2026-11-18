# Preliminary Offline CUSUM Pilot

## Scope

- Status: `PRELIMINARY`
- Provider calls: `0`
- Additional spend: `$0`
- Worker processes: `1`
- Trials per scenario: `250`
- Independent calibration streams: `1000`
- Healthy accuracy: `0.95`
- Target degraded accuracy: `0.85`
- Calibrated CUSUM threshold: `8.080612`
- Calibration false-alarm rate: `0.010`
- Wall time: `11.540 seconds`
- Peak process RSS: `29.0 MB`

## Method

- Each trial contains `900` post-change binary grades (`15` windows x `60` grades).
- The one-sided Bernoulli CUSUM targets a drop from `0.95` to `0.85` and evaluates every grade.
- Its threshold was selected on independent healthy streams to target at most a `1.0%` finite-horizon false-alarm rate.
- The existing Fisher detector estimates a baseline from `5` windows, evaluates 60-grade windows, and requires confirmation.
- Both detectors receive the same seeded planted evidence streams.

## Paired Results

| Scenario | CUSUM detect | Fisher detect | CUSUM median query | Fisher median query |
| :-- | --: | --: | --: | --: |
| `healthy_a_a` | 0.012 | 0.000 | 637.0 | n/a |
| `full_weak_model_swap` | 1.000 | 1.000 | 21.0 | 120.0 |
| `threshold_level_drop` | 1.000 | 0.904 | 93.0 | 240.0 |
| `gradual_degradation` | 1.000 | 1.000 | 181.0 | 300.0 |
| `partial_substitution_25` | 1.000 | 0.800 | 118.0 | 300.0 |
| `partial_substitution_50` | 1.000 | 1.000 | 47.0 | 120.0 |

## Interpretation

- The independent healthy validation produced `3/250` CUSUM alarms (`1.2%`) and `0/250` Fisher alarms (`0.0%`).
- CUSUM detected every planted degradation scenario in this panel and had a lower median query count than Fisher.
- CUSUM was given the frozen healthy rate, while Fisher estimated its baseline from samples; this favors CUSUM and prevents a universal superiority claim.
- This pilot tests feasibility and relative detection delay under one synthetic model. It does not establish production false-alarm rates.

## Planned Longer Experiment

Before production use, repeat calibration with more seeds, multiple healthy baseline levels, baseline-estimation uncertainty, family-level changes, and independent multi-day live windows. Freeze the final threshold only after that evaluation.
