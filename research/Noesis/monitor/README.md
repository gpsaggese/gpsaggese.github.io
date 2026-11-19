# Monitor

- Runs deterministic canaries against one configured provider/model without a market
  contract
- Persists model quality, availability, latency, token, attribution, and cost
  observations in monitoring-only tables
- Does not write market orders, contracts, verdicts, or seller reputation

## Structure of the Dir

| Dir     | Description                                  |
| :------ | :------------------------------------------- |
| `test/` | Unit and PostgreSQL-backed integration tests |

## Description of Files

| File                 | Description                                      |
| :------------------- | :----------------------------------------------- |
| `monitor_capability_experiment.py` | Offline capability scenarios and trials |
| `monitor_capability_report.py` | Experiment artifacts and plots             |
| `monitor_cli.py`     | No-spend validation and explicit live execution   |
| `monitor_detection.py` | Capability baselines and degradation decisions |
| `monitor_experiment_manifest.py` | Frozen Phase 0 routes, bank, budgets, and gates |
| `monitor_fingerprint_experiment.py` | Blinded Hamming-MMD replay controls |
| `monitor_fingerprint_report.py` | Fingerprint experiment artifacts       |
| `monitor_live_fingerprint.py` | Paid three-pool Lane 2 collection plan |
| `monitor_live_fingerprint_analysis.py` | Live A/A, swap, and mixture analysis |
| `monitor_metrics.py` | Per-window and per-family evidence aggregation    |
| `monitor_models.py`  | Immutable targets, policies, and observations     |
| `monitor_paid_execution.py` | Shared paid-stage safety and execution core |
| `monitor_paid_report.py` | Durable generic paid-stage artifacts         |
| `monitor_paid_smoke.py` | Exact 15-call, $0.25 attribution smoke plan   |
| `monitor_paid_smoke_report.py` | Attribution-smoke artifacts            |
| `monitor_controlled_warmup.py` | Exact 240-call, $10 warm-up plan       |
| `monitor_warmup_benchmark.py` | Frozen 60-item bank and deterministic graders |
| `monitor_runtime.py` | One bounded, sequential monitoring pass           |
| `monitor_store.py`   | Monitoring-only PostgreSQL persistence            |

## Architecture

- `monitor/` coordinates reusable code from `checker/`, `common/`, and the provider
  boundary in `gateway/`
- The runtime selects active calibrated canaries, sends ordinary user prompts through
  the existing `Provider` interface, grades successful responses, and stores each
  observation immediately
- Provider failures are observation outcomes; a database or orchestration failure
  marks the run `FAILED`
- Runs are sequential in this MVP to bound provider load and keep latency
  measurements interpretable

## Commands

- Validate a ten-call plan without spending money or writing to PostgreSQL:

  ```bash
  > python3 -m research.Noesis.monitor.monitor_cli \
      --target groq=meta-llama/llama-3.3-70b-instruct
  ```

- Add `--execute` after reviewing the call count to make provider calls and persist
  the run
- Live execution requires `OPENROUTER_API_KEY`, `NOESIS_DATABASE_URL`, and
  `NOESIS_CANARY_SCHEDULE_SECRET`
- Run the offline capability-drift experiment without provider or database access:

  ```bash
  > python3 research/Noesis/scripts/run_model_drift_capability_experiment.py \
      --trials_per_scenario 1000 \
      --power_trials 100 \
      --dst_dir research/Noesis/docs/findings/model_drift_capability
  ```

- Run the offline fingerprint controls from the pinned reference checkout:

  ```bash
  > NOESIS_MODEL_EQUALITY_REPO=/path/to/model-equality-testing \
    python3 research/Noesis/scripts/run_model_drift_fingerprint_experiment.py \
      --trials_per_scenario 100 \
      --permutations 1000 \
      --dst_dir research/Noesis/docs/findings/model_drift_fingerprint
  ```

- Validate the exact $0.25 attribution-smoke plan without making a provider
  call:

  ```bash
  > python3 research/Noesis/scripts/run_model_drift_attribution_smoke.py \
      --dst_dir research/Noesis/docs/findings/model_drift_attribution_smoke
  ```

- After exporting `OPENROUTER_API_KEY` locally, execute the reviewed smoke plan
  by adding `--execute --approved_cap_usd 0.25`. Do not paste the key into the
  command history or a chat message.

- Validate the exact $10 controlled warm-up after freezing the 60-row bank:

  ```bash
  > python3 research/Noesis/scripts/run_model_drift_controlled_warmup.py \
      --bank_path /absolute/path/to/reviewed-bank.jsonl \
      --dst_dir research/Noesis/docs/findings/model_drift_controlled_warmup
  ```

- Execute that reviewed plan by adding `--execute --approved_cap_usd 10`.
  Both paid runners write the exact plan before the first call and fsync every
  observation before proceeding.

- Validate the 180-call live Lane 2 fingerprint plan without spending money:

  ```bash
  > python3 research/Noesis/scripts/run_model_drift_live_fingerprint.py \
      --dst_dir research/Noesis/docs/findings/model_drift_live_fingerprint
  ```

- Execute the reviewed plan by adding `--execute --approved_cap_usd 2`.
  The plan collects two independent Cloudflare 70B pools and one Cloudflare
  8B pool at the frozen sampling temperature of 1.0.

- Analyze complete saved pools through the pinned reference implementation:

  ```bash
  > NOESIS_MODEL_EQUALITY_REPO=/path/to/model-equality-testing \
    python3 research/Noesis/scripts/analyze_model_drift_live_fingerprint.py \
      --observations_path /path/to/observations.jsonl \
      --dst_dir research/Noesis/docs/findings/model_drift_live_fingerprint
  ```

- Build and validate the versioned bank and private answer file:

  ```bash
  > python3 research/Noesis/scripts/build_model_drift_warmup_bank.py
  ```

- After collection, grade the saved responses without another provider call:

  ```bash
  > python3 research/Noesis/scripts/grade_model_drift_controlled_warmup.py \
      --bank_path research/Noesis/data/model_drift_warmup_bank_v1.jsonl \
      --graders_path \
        research/Noesis/data/model_drift_warmup_grader_configs_v1.jsonl \
      --observations_path /path/to/observations.jsonl \
      --dst_dir /path/to/run-directory
  ```

## Deferred Work

- Recurring schedules, bounded concurrency, retries, heartbeats, and stale-run
  cleanup
- Recurring live windows, stored baselines, dashboards, and alerts
- CUSUM and recurring live fingerprint baselines
- Any market, contract-verdict, or seller-reputation integration
