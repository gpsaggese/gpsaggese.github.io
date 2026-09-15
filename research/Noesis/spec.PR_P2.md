# `PR_P2`: Cloud Deployment - Implementation Spec

- Specifies the Python code, Docker assets, and cloud deployment config for `PR_P2`
  from `research/Noesis/plan.Noesis.md`
- Scope: containerize `PR_P1`'s API into one Docker image; deploy to a single-node
  container service (AWS ECS, Fly.io, or Render); result is a
  `NoesisMarket`/`NoesisServer` instance reachable at a public URL; Kubernetes
  deferred until more than one process needs orchestration
- Roadmap position: `v0.1`, cumulative on `PR_M1`, `PR_M2`, `PR_S1`, `PR_P1` (all
  `[x]` already)
- Specification only: no code in this document has been implemented

## Design Decisions

- Follow this repo's dev-system Docker flow (`devops/` dir,
  `dev.Dockerfile`/`prod.Dockerfile`, `poetry`, `invoke` release tasks, ECR/GHCR)
  instead of `class_project/project_template/Dockerfile*`: every runnable dir here
  already uses it (e.g. `tutorials/tutorial_forecast_as_service/`)
- `main.py` stays a flat file, not a nested `api/` subpackage: matches this dir's
  flat, `rn<abbrev>`-aliased modules
- Add a real, `poetry`-managed `pyproject.toml`, not a `requirements.txt`: resolves
  the gap `architecture.md` flags and matches every other runnable dir's convention
- Cloud target: AWS ECS, task definition/service registered directly, not via this
  repo's `aws_create_prod_task_definition` automation, which reads a
  `shared_configs_bucket_name` unset in any `repo_config.yaml` here

## Trade-off and Alternative Design

- Org-template ECS automation would reuse shared tooling but needs provisioning the
  missing `shared_configs_bucket_name` first (one-time infra work, out of scope);
  direct registration needs nothing extra, at the cost of bypassing that tooling
- Fly.io/Render remain valid per the plan, but neither has release tooling in this
  repo like ECS does; picking one means building deploy tooling from scratch instead
  of reusing `docker_release_dev_image`

## Out of Scope

- Persistent datastore for order book/contract/request log: `PR_P2b`; state stays
  in-memory `List`/`Dict` (`architecture.md` Weakness 6)
- Real payment/credit gating on `POST /bids`: `PR_P3`/`PR_P4`
- Real fulfillment via `Gateway`: `PR_M8`; deployed instance keeps
  `contract_dispatch.mock_fulfill()` as the default `fulfillment_fn`
- Kubernetes, multi-region, autoscaling: deferred per the plan
- Provisioning the ECS cluster/service/load balancer: one-time AWS setup, not
  Python/Docker-asset code

## Current State

- `platform_api.create_app(order_book, gateway, api_keys, *, fulfillment_fn=mock_fulfill) -> fastapi.FastAPI`
  is the only thing to containerize; wraps `OrderBook`/`Gateway` behind HTTP with
  `X-API-Key` auth on `POST /bids`, `POST /asks`, `POST /completions`
- Gaps per `architecture.md`: no launch script hands `create_app()` to `uvicorn`
  (only `TestClient` does today); `api_keys` is a plain constructor arg, not
  env-sourced; no health-check endpoint; the dir has no `devops/`, `changelog.txt`,
  or `repo_config.yaml`

## Implementation

- New entrypoint module, one added endpoint, plus symlinks/Docker assets that make
  `research/Noesis` a runnable, dockerized dir

### `research/Noesis/main.py` (new)

- Builds module-level `app`, run via
  `uvicorn research.Noesis.main:app --host 0.0.0.0 --port 8000`, following
  `tutorials/tutorial_forecast_as_service/api/main.py`'s pattern; calls
  `hdbg.init_logger(verbosity=logging.INFO)` at import time
- `_parse_api_keys(raw: str) -> Dict[str, str]`: parses
  `NOESIS_API_KEYS="key1:acct1,key2:acct2"` (empty string -> `{}`, all writes then
  `401`); `hdbg.dassert_in(":", entry, ...)` fails fast on a malformed entry
- Module level: builds `order_book`, `gateway`, `api_keys` (via `_parse_api_keys`),
  then `app = rnoplapi.create_app(order_book, gateway, api_keys)`

### `platform_api.py`: Add `GET /health`

- One unauthenticated route in `create_app()`: `GET /health -> {"status": "ok"}`,
  polled by every candidate cloud target's health check; does not read
  `order_book`/`gateway` state; update the module docstring's endpoint list per
  `.claude/skills/coding.rules.md`'s "Update Docstrings If Out-of-sync"

### Docker and Runnable-dir Assets

- New real files: `changelog.txt`, `repo_config.yaml`
  (`docker_image_name: noesis_platform`, `container_registry_info`/`s3_bucket_info`
  copied from this repo's top-level `repo_config.yaml`), `.dockerignore`; modeled on
  `tutorial_forecast_as_service/`
- New symlinks: `tasks.py`/`conftest.py` -> `../../tasks.py`/`conftest.py`;
  `invoke.yaml`/`pytest.ini` -> `../../helpers_root/invoke.yaml`/`pytest.ini`, at a
  corrected depth (the precedent's own symlinks are broken, one `../` short)
- `devops/docker_build/`: symlinked boilerplate (`dev.Dockerfile`, `prod.Dockerfile`,
  install scripts) plus a real `pyproject.toml` (`fastapi`, `pydantic`,
  `uvicorn[standard]`, `pytest`)
- `devops/docker_run/`: symlinked boilerplate plus a real `run_docker_noesis.sh`
  launcher and (unlike the precedent) an actual `devops/env/default.env`
- `devops/compose/docker-compose.noesis.yml`: extends the generated
  `tmp.docker-compose.yml`, overriding `command` and `ports: ["8000:8000"]`
- Release: from `research/Noesis/`, `i docker_build_local_image` -> `i docker_bash`
  (smoke test via `pytest research/Noesis/test`) -> `i docker_tag_local_image_as_dev`
  -> `i docker_push_dev_image` -> `i docker_build_prod_image` ->
  `i docker_push_prod_image` (or the `docker_release_*_image` wrappers)

## Interaction with Existing Code

- `create_app()`'s signature is unchanged; only its body gains `GET /health`, so
  existing `TestClient`-based tests are unaffected
- `main.py` is the first non-test caller of `create_app()`
- The new symlinks reuse the repo's generic `invoke`/pytest config unchanged: no
  changes to `tasks.py` or `helpers_root/invoke.yaml`/`pytest.ini` themselves

## Configuration and Secrets

- `NOESIS_API_KEYS`: only required secret; local via `docker compose run -e`/shell
  env (never committed); on ECS via a task-definition `secrets` entry
- `--host`/`--port` are not flags: the compose `command:` or ECS `containerPort` sets
  the bind address
- `fulfillment_fn` stays hardcoded to `mock_fulfill`; real fulfillment is `PR_M8`

## Unit Test Plan

- New `test_main.py`, `Test__parse_api_keys`: `test1` well-formed string -> matching
  `Dict`; `test2` empty string -> `{}`; `test3` malformed entry -> `AssertionError`
- Extend `test_platform_api.py`'s existing class: `GET /health` returns
  `200`/`{"status": "ok"}` without an `X-API-Key` header
- Not tested directly: `main.py`'s module-level `app` (reads env at import time);
  `_parse_api_keys()` is factored out to stay testable
- `test_batch_call_auction.py`, `test_contract_dispatch.py`,
  `test_passthrough_proxy.py`: no changes expected, the regression signal

## Risks and Limitations to Call Out

- Deployed instance still runs mocked fulfillment/stub providers, not a real LLM
  call: state this wherever announced
- No persistence (`PR_P2b`): a redeploy/crash loses every pending bid/ask, contract,
  and request log
- No rate limiting beyond `X-API-Key` on write endpoints; read endpoints stay
  unauthenticated (`architecture.md` Weakness 9)
- First `research/` dir made "runnable"; the only precedent has two known bugs
  (broken symlinks, missing `devops/env/`) this PR must not reproduce
- ECS task-definition automation here is coupled to unconfigured org-wide infra; do
  not assume it alone reaches a running, public service

## Result

- Implemented as specced:
  - `research/Noesis/main.py` (new): module-level `app`, run via
    `uvicorn research.Noesis.main:app --host 0.0.0.0 --port 8000`;
    `_parse_api_keys(raw) -> Dict[str, str]` exactly as specced (empty string ->
    `{}`, `hdbg.dassert_in(":", entry, ...)` on a malformed entry); written together
    with `PR_P2b`'s `NOESIS_DB_BACKEND` switch in the same session (see
    `spec.PR_P2b.md`'s "Result"), since `main.py` did not exist on disk yet when that
    PR landed
  - `platform_api.py`: `GET /health -> {"status": "ok"}`, unauthenticated, reads no
    `order_book`/`gateway` state; module docstring's endpoint list updated
  - Docker/runnable-dir assets: `changelog.txt`, `repo_config.yaml`
    (`docker_image_name: noesis_platform`, `container_registry_info`/
    `s3_bucket_info` copied from the top-level `repo_config.yaml`), `.dockerignore`;
    `tasks.py`/`conftest.py`/`invoke.yaml`/`pytest.ini` symlinks at the corrected
    depth; `devops/docker_build/` (symlinked boilerplate + a real `pyproject.toml`
    with `fastapi`/`pydantic`/ `uvicorn[standard]`/`pytest`, later extended by
    `PR_P2b` with `numpy`/ `pandas`/`psycopg2-binary`, plus a `poetry lock`-generated
    `poetry.lock`); `devops/docker_run/` (symlinked boilerplate + a real, executable
    `run_docker_noesis.sh`)
  - `devops/compose/docker-compose.noesis.yml` and `devops/env/default.env`: landed
    as part of `PR_P2b` (built in a concurrent session), combining this PR's
    `noesis_api` service/`command`/`ports: ["8000:8000"]` baseline with `PR_P2b`'s
    `noesis_postgres` service/volume and env additions, so this PR's own baseline
    requirement for both files is satisfied by that combined version rather than a
    `PR_P2`-only one
  - Verified: `tar -czh .` from `research/Noesis/` (the exact mechanism
    `docker_build_local_image`'s single-arch path uses to assemble the build context)
    dereferences every corrected symlink into real, non-empty file content (e.g
    `dev.Dockerfile` 2117B, `poetry.lock` 95452B, `entrypoint.sh` 2239B), confirming
    the depth fix actually resolves; full `research/Noesis` test suite: 53 passed, 10
    skipped (`test_postgres_ store.py`, needs docker-in-docker, expected per
    `PR_P2b`)
- Deferred, deliberately out of this PR's scope (see "Out of Scope" above): no AWS
  ECS cluster/service/load balancer, RDS, or GHCR/ECR push was provisioned or
  executed here: all one-time manual/infra steps, not Python/Docker-asset code, per
  the "Design Decisions"/"Trade-off" sections above. `i docker_build_local_image` /
  `docker_bash` / `docker_tag_local_ image_as_dev` / `docker_push_*_image` were not
  run end-to-end in this sandbox (no Docker registry credentials); the `tar -czh`
  dereference check above is a lighter-weight proxy for "the symlink/build-context
  wiring is correct", not a substitute for an actual image build. AWS ECS was kept as
  the documented cloud target (direct task-definition/service registration, not the
  org-template automation); Fly.io/Render were not attempted, per "Trade-off and
  Alternative Design" above
- Not yet reachable at a public URL: `plan.Noesis.md`'s `PR_P2` Result line ("a
  `NoesisMarket`/`NoesisServer` instance reachable at a public URL") is unmet until
  the ECS provisioning step above is actually carried out