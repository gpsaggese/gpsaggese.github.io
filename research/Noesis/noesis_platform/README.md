# Noesis Platform

- `NoesisPlatform`: a thin `fastapi` HTTP surface over the market and the legacy
  proxy, with an optional Postgres backend
- `main.py` builds the module-level `app` that `uvicorn` serves
- Imports `market/` and `gateway/passthrough_proxy.py`; nothing imports it back
- Architecture: `docs/architecture.md`

## Structure of the Dir

| Dir     | Description                            |
| :------ | :------------------------------------- |
| `test/` | Unit tests, one `test_*.py` per module |

## Description of Files

| File               | Description                                                              | Cluster  |
| :----------------- | :----------------------------------------------------------------------- | :------- |
| `main.py`          | `uvicorn` entry point; picks the memory or Postgres backend              | Entry    |
| `platform_api.py`  | `create_app()`: HTTP endpoints over the auction, dispatch, and proxy     | HTTP     |
| `postgres_store.py` | Postgres implementations of `OrderBookStore`, `ContractStore`, `RequestLogStore` | Database |

## Description of Executables

| Command | Description |
| :------ | :---------- |
| `uvicorn research.Noesis.noesis_platform.main:app --host 0.0.0.0 --port 8000` | Serve the API in-process, no Docker |
| `devops/docker_run/run_docker_noesis.sh [VERSION]` | Run the API plus a Postgres sidecar via Docker Compose |

- Environment: `NOESIS_DB_BACKEND` (`memory` default, or `postgres`),
  `NOESIS_API_KEYS="key1:acct_1,key2:acct_2"` for `X-API-Key` auth, and the
  `POSTGRES_*` connection variables

## Description of Architecture

- `main.py` selects the backend once, at import time; every other module is unaware
  of it
- `postgres_store.py` is imported only inside the `postgres` branch of `main.py`, so
  the default path needs no `psycopg2`
