# Issue 645: Model Capability-Drift Experiment

## Summary

- Phase 1 adds an offline, deterministic experiment for model quality degradation.
- The implementation reuses the monitoring observation schema and remains isolated
  from market matching, contract verdicts, and provider reputation.
- The full local run evaluated 12,000 primary trials and passed all gated criteria.

## Architecture

```text
Scenario and hidden intervention
              |
              v
Raw MonitoringObservation records
              |
              v
Per-family window aggregation
              |
              v
Fisher test + minimum effect size
              |
              v
Confirmation state machine
              |
              v
Observed decision and detection delay
              |
              v
Compare with planted truth in the report only
```

The detector cannot read scenario names, expected outcomes, or intervention labels.
This separation prevents the experiment from grading itself with hidden truth.

## Protocol

- Baseline: `5` healthy windows.
- Evaluation: `15` monitoring windows after the intervention.
- Evidence: `60` observations per window by default.
- Healthy accuracy: `95%`.
- Weak-model accuracy: `60%`.
- Significance threshold: `p < 0.01`.
- Minimum material drop: `10` percentage points.
- Confirmation: `2` consecutive degraded windows.
- Primary trials: `1,000` per scenario.
- Power analysis: `100` trials per query-count and baseline-size combination.

## Primary Results

| Scenario | Detection rate | 90th-percentile delay | Result |
| --- | ---: | ---: | --- |
| Healthy A/A | 0.000 | Not applicable | Pass |
| Full weak-model swap | 1.000 | 2 windows | Pass |
| Small 3-point drop | 0.027 | 15 windows | Pass |
| Threshold 10-point drop | 0.873 | 11 windows | Measured |
| Gradual degradation | 1.000 | 7 windows | Pass |
| Partial substitution, 10% | 0.044 | 13 windows | Measured |
| Partial substitution, 25% | 0.727 | 12 windows | Measured |
| Partial substitution, 50% | 1.000 | 4 windows | Pass |
| Latency-only change | 0.000 | Not applicable | Pass |
| Price-only change | 0.000 | Not applicable | Pass |
| Availability failure | 0.000 | Not applicable | Pass |
| Ungradable evidence | 0.000 | Not applicable | Pass |

Availability and ungradable failures produced `INSUFFICIENT` in every trial instead
of being mislabeled as capability degradation.

## Power Findings

- A full model swap was detected in every tested query-budget configuration.
- For a 10-point quality drop with five baseline windows, detection rose from
  `0.18` at 20 observations per window to `0.91` at 60 observations per window.
- A 50% mixed substitution reached `1.00` detection at 40 or more observations per
  window.
- The healthy false-alarm rate remained between `0.00` and `0.01` in the tested
  power configurations.

## Claims and Limitations

- The experiment validates the statistical logic under controlled synthetic
  interventions; it does not prove that a hosted provider changed its model.
- A capability alert means the output distribution is materially worse on the
  selected canary families. It does not identify the underlying cause.
- Fisher testing and confirmation are implemented in Phase 1. CUSUM, MMD, and
  fingerprint-based identity testing remain follow-up experiments.
- The primary detector uses aggregate accuracy over a balanced family sample.
  Per-family metrics are retained but do not yet produce independent verdicts.
- Repeated-window testing still creates a multiple-testing risk. The healthy and
  small-drop controls measure that risk empirically for this protocol.
- Live-provider evidence requires a separate controlled run with a fixed model,
  prompt bank, decoding policy, and time schedule.

## Files

- `monitor/monitor_metrics.py`: Aggregate raw observations into window metrics.
- `monitor/monitor_detection.py`: Baselines, Fisher testing, and detector states.
- `monitor/monitor_capability_experiment.py`: Scenarios and Monte Carlo trials.
- `monitor/monitor_capability_report.py`: JSON, CSV, Markdown, and SVG reports.
- `scripts/run_model_drift_capability_experiment.py`: Reproducible CLI.
- `docs/findings/model_drift_capability/`: Checked experimental artifacts.

## Next Steps

- Review the measured power and choose a live query budget.
- Run an A/A live-provider warm-up before testing any degradation hypothesis.
- Add time-series and model-identity lanes as separate experiments.
- Keep future server updates in component-specific files to avoid conflicts in
  shared progress documents.
