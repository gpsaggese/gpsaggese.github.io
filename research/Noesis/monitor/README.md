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
| `monitor_cli.py`     | No-spend validation and explicit live execution |
| `monitor_models.py`  | Immutable targets, policies, and observations   |
| `monitor_runtime.py` | One bounded, sequential monitoring pass          |
| `monitor_store.py`   | Monitoring-only PostgreSQL persistence           |

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

## Deferred Work

- Recurring schedules, bounded concurrency, retries, heartbeats, and stale-run
  cleanup
- Window aggregation, baselines, nerfing detection, charts, and alerts
- Any market, contract-verdict, or seller-reputation integration
