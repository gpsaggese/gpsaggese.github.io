# Noesis

- Implementation of the \Noesis{} protocol described in `papers/Noesis/*.tex`: a
  two-sided market for LLM inference capacity
- `NoesisMarket`: a batch call-auction that matches buyer/seller orders and
  dispatches cleared contracts to a (mocked) fulfillment layer
- `NoesisServer`: the MVP gateway (OpenAI-compatible, routes each request to the one
  real provider its contract names, logs every call) and the checker that grades
  sellers against their contracts
- `NoesisPlatform`: a thin `fastapi` HTTP surface over the market and the legacy
  proxy, with an optional Postgres backend and a Docker Compose deployment
- See:
  - `docs/README.md` for the index of all docs
  - `docs/design.architecture.md` for the C4-model architecture
  - `docs/onboarding.md` to set up, run, and test the MVP gateway
  - `docs/plan.md` for the protocol implementation roadmap

## Structure of the Dir

| Dir                 | Description                                                         | Concept             |
| :------------------ | :------------------------------------------------------------------ | :------------------ |
| `checker/`          | Grades sellers on latency and quality: pure verdict logic plus its store | `NoesisServer`  |
| `common/`           | Settings, database pool, migrations, demo seed shared by components | MVP infrastructure  |
| `devops/`           | Dockerfiles, Docker Compose deployment, container entrypoint scripts | Deployment         |
| `docs/`             | All docs: architecture, plans, specs, notes, findings, onboarding   | Docs                |
| `gateway/`          | OpenAI-compatible gateway and the legacy in-process proxy           | `NoesisServer`      |
| `market/`           | Call auction and contract dispatch                                  | `NoesisMarket`      |
| `monitor/`          | Standalone provider/model checks without market contracts           | `NoesisServer`      |
| `noesis_platform/`  | HTTP service over the market and the proxy, Postgres stores         | `NoesisPlatform`    |
| `prober/`           | Live calibration and hidden scheduled canary execution              | `NoesisServer`      |
| `scripts/`          | One-off measurement scripts                                         | Tools               |

- Each component keeps its tests in its own `test/` dir
- Allowed imports between components (checked by `test/test_component_imports.py`):

  | Dir                | May import                                        |
  | :----------------- | :------------------------------------------------ |
  | `common/`, `market/` | nothing in `research.Noesis`                    |
  | `gateway/`         | `common/`                                         |
  | `monitor/`         | `checker/`, `common/`, provider modules in `gateway/` |
  | `checker/`         | `common/`, `gateway/gateway_providers.py`         |
  | `noesis_platform/` | `market/`, `gateway/passthrough_proxy.py`         |
  | `prober/`          | `checker/`, `common/`, `gateway/`                 |

## Description of Files

| File               | Description                                                                |
| :----------------- | :------------------------------------------------------------------------- |
| `.dockerignore`    | Restricts the DEV Docker build context to `devops/` and `helpers_root/devops/` |
| `changelog.txt`    | Release log for the `noesis_platform` Docker image                         |
| `conftest.py`      | Pytest bootstrap (symlink to the repo's `conftest.py`)                     |
| `env.example`      | Template for `.env` (git-ignored): OpenRouter key and the gateway's API keys |
| `invoke.yaml`      | `pyinvoke` config (symlink)                                                |
| `pytest.ini`       | Pytest markers and CLI options (symlink)                                   |
| `repo_config.yaml` | Repo, Docker, and S3 metadata read by `helpers.repo_config_utils`          |
| `tasks.py`         | Exposes the `invoke` targets: Docker, AWS, pytest (symlink)                |

## Description of Executables

| Command                                                                          | Description                                                  |
| :------------------------------------------------------------------------------- | :----------------------------------------------------------- |
| `uvicorn research.Noesis.noesis_platform.main:app`                               | Serve the `NoesisPlatform` HTTP API (in-process, no Docker)  |
| `uvicorn research.Noesis.gateway.gateway_app:build_default_app --factory`        | Serve the MVP gateway (see `docs/onboarding.md`)             |
| `scripts/spike_openrouter.py`                                                    | Measure pinned OpenRouter providers (`--dry-run` costs nothing) |
| `scripts/detection_math.py`                                                      | Print violation-rule false-alarm and detection rates         |
| `python3 -m research.Noesis.prober.prober_cli calibrate ...`                     | Validate or execute live canary calibration                  |
| `python3 -m research.Noesis.prober.prober_cli probe`                             | Run the scheduled canary worker                              |
| `python3 -m research.Noesis.monitor.monitor_cli --target PROVIDER=MODEL`          | Validate a standalone monitoring run                         |
| `devops/docker_run/run_docker_noesis.sh`                                         | Build and run the dockerized API plus a Postgres sidecar     |
| `invoke run_fast_tests`                                                          | Run the fast unit test suite                                 |

- Run all tests from the repo root:

  ```bash
  > pytest research/Noesis
  ```

## Description of Workflows

- Local, in-memory dev loop
  - `> uvicorn research.Noesis.noesis_platform.main:app --reload` serves the API with
    the default `NOESIS_DB_BACKEND=memory`; state resets on every restart
  - Exercise it with `POST /bids`, `POST /asks`, `POST /rounds/clear`, then
    `GET /contracts/{contract_id}`
- Containerized dev loop with a persistent backend
  - `devops/docker_run/run_docker_noesis.sh` brings up `noesis_api` next to a
    `noesis_postgres` sidecar, wired through `NOESIS_DB_BACKEND` and `POSTGRES_*` in
    `devops/compose/docker-compose.noesis.yml`
- Order-to-contract flow (C3 diagram in `docs/design.architecture.md`)
  - `OrderBook.submit_bid()` / `submit_ask()` queue orders in the `OrderBookStore`
  - `OrderBook.clear_round()` buckets orders by tier and clears each one
  - `build_contracts()` turns the tier results into `Contract`s;
    `dispatch_contract()` fulfills each through the injected `fulfillment_func`
- Dev notes (build, generate the OpenAPI spec): `docs/notes/dev_notes.md`
