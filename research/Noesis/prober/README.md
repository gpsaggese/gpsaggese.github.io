# Prober

- Coordinates the checker and gateway to send server-owned canary traffic
- Builds secret deterministic schedules, claims due slots in PostgreSQL, routes
  probes through the contracted seller, grades responses, and persists evidence
- Does not expose canary IDs or probe metadata in the provider request

## Structure of the Dir

| Dir     | Description                                      |
| :------ | :----------------------------------------------- |
| `test/` | Unit and PostgreSQL-backed integration tests     |

## Description of Files

| File                | Description                                      |
| :------------------ | :----------------------------------------------- |
| `prober_cli.py`     | Safe calibration and scheduled-worker commands  |
| `prober_runtime.py` | One non-blocking scheduled-probe execution pass |
| `prober_worker.py`  | Repeated passes with retry and graceful shutdown |

## Description of Architecture

- `prober/` is a thin coordination layer
- It imports scheduling, grading, models, and persistence from `checker/`
- It imports routing, providers, faults, and request logging from `gateway/`
- Provider requests contain ordinary user messages; canary metadata is added
  only when Noesis writes its internal request row

## Commands

- Validate a calibration plan without spending money or writing to PostgreSQL:

  ```bash
  > python3 -m research.Noesis.prober.prober_cli calibrate \
      --healthy groq=meta-llama/llama-3.3-70b-instruct \
      --healthy together=meta-llama/llama-3.3-70b-instruct \
      --weak deepinfra=meta-llama/llama-3.1-8b-instruct
  ```

- After reviewing the printed call count, add `--execute` to make paid calls.
  The command persists every observation, activates the bank only if at least
  150 canaries survive, and writes `docs/findings/calibration.md`.
- Run one due-probe cycle with `probe --once`, or omit `--once` for a worker
  that polls once per second until `SIGINT` or `SIGTERM`.
- Live modes require `OPENROUTER_API_KEY`, `NOESIS_DATABASE_URL`, and
  `NOESIS_CANARY_SCHEDULE_SECRET` in `.env` or the process environment.

## Failure Behavior

- A provider timeout or upstream error is persisted as seller evidence.
- A gateway/configuration error is recorded but is not blamed on the seller.
- An internal probe failure becomes `FAILED_INTERNAL`; later probes continue.
- A database-level cycle failure is logged and retried by the worker.
- PostgreSQL's unique `(contract_id, probe_slot)` constraint prevents two
  workers from executing the same scheduled slot.
