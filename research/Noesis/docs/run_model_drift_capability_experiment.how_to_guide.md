# Run the Model-Drift Capability Experiment

## Summary

- This guide reproduces the offline Phase 1 capability experiment for issue #645.
- The experiment asks whether the checker can detect a hidden drop in model
  accuracy while avoiding alerts on healthy, latency-only, and price-only traffic.
- It makes no provider calls and performs no database or market writes.

## Scope

- Five healthy baseline windows establish the expected accuracy for each canary
  family.
- Fifteen monitoring windows follow a hidden intervention.
- Each window contains 60 deterministic observations distributed across the six
  calibrated canary families.
- The detector receives observations and the baseline, but never the planted
  intervention label.
- The monitored capability series is classified as `NERFED` only after two
  consecutive windows satisfy
  both conditions:
  - The one-sided Fisher exact-test p-value is below `0.01`.
  - The observed accuracy drop is at least `10` percentage points.

## Run the Experiment

From the repository root, run:

```bash
> python3 research/Noesis/scripts/run_model_drift_capability_experiment.py \
    --trials_per_scenario 1000 \
    --power_trials 100 \
    --dst_dir research/Noesis/docs/findings/model_drift_capability \
    -v INFO
```

The command exits with status `0` only when all gated acceptance criteria pass.
Use `--overwrite` when replacing an existing output directory.

## Inspect the Results

The destination directory contains:

- `protocol.json`: Reproducible protocol and detector configuration.
- `trial_results.csv`: Per-trial planted truth and observed detector decision.
- `scenario_summary.csv`: Detection rate and delay by scenario.
- `power_analysis.csv`: Sensitivity across query and baseline-window counts.
- `summary.json`: Machine-readable aggregate results.
- `findings.md`: Human-readable findings, claims, and limitations.
- `detection_power.svg`: Detection rate for every primary scenario.
- `false_alarm_rate.svg`: False-alarm rate for negative controls.
- `power_vs_query_count.svg`: Detection power versus observations per window.

The primary detector uses aggregate accuracy across the balanced family sample.
Per-family counts are preserved for diagnosis and later robustness comparisons.

## Interpret Detector States

- `OK`: The evidence does not meet both degradation thresholds.
- `WARNING`: One degraded window was observed, but confirmation is pending.
- `NERFED`: Two consecutive windows meet both degradation thresholds.
- `INSUFFICIENT`: Too few gradable observations exist to make a quality claim.

`NERFED` is an experimental capability verdict. It does not stop traffic, alter a
market contract, or penalize a provider.

## Run the Tests

Run the Phase 1 unit and CLI tests:

```bash
> pytest -q \
    research/Noesis/monitor/test/test_monitor_metrics.py \
    research/Noesis/monitor/test/test_monitor_detection.py \
    research/Noesis/monitor/test/test_monitor_capability_experiment.py \
    research/Noesis/monitor/test/test_monitor_capability_report.py \
    research/Noesis/scripts/test/test_run_model_drift_capability_experiment.py
```

Run all monitoring regression tests before review:

```bash
> pytest -q research/Noesis/monitor/test research/Noesis/scripts/test
```

## Troubleshoot

- If the destination already exists, pass `--overwrite` or choose another path.
- If a scenario fails, inspect `scenario_summary.csv` before changing thresholds.
- If a window is `INSUFFICIENT`, inspect gradable and available observation counts.
- Preserve the protocol seed when reproducing a result. Change it only for an
  explicitly documented robustness run.
