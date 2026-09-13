# FastAPI Tutorial
- This folder contains the setup for running FastAPI and uvicorn tutorials within
  a containerized environment

## Quick Start
- From the root of the repository, change your directory to the FastAPI tutorial
  folder:
  ```bash
  > cd tutorials/fastapi
  ```

- Once the location has been changed to the repo run the command to build the
  image to run dockers:
  ```bash
  > ./docker_build.sh
  ```

- Once the docker has been built you can then go ahead and run the container and
  launch jupyter notebook using the created image using the command:
  ```bash
  > ./docker_jupyter.sh
  ```

- For more information on the Docker build system refer to [Project template
  README](/class_project/project_template/docker_scripts.README.md)

- Once the `./docker_jupyter.sh` script is running, work through the following
  notebooks in order

## Tutorial Notebooks
- Work through the following notebooks in order:

  - [`fastapi.API.ipynb`](fastapi.API.ipynb): Core FastAPI fundamentals
    - Path and query parameters
    - Request body validation with `pydantic` models
    - Dependency injection with `Depends()`
    - Error handling with `HTTPException`
    - The automatic `/docs`, `/redoc`, and `/openapi.json`

  - [`fastapi_utils.py`](fastapi_utils.py): Utility functions supporting the
    tutorial notebooks

  - [`fastapi.example.ipynb`](fastapi.example.ipynb): End-to-end Book Catalog API
    - Runs a real `uvicorn` server on a background thread
    - Exercises the API with real HTTP requests via `httpx`
    - Covers create, read, update, delete, filtering, and error responses

## Changelog
- 2026-08-10: Initial release

## TODO

1. In-process, no server, no network (what the test suite does)

fastapi.testclient.TestClient wraps create_app() directly — same interface as requests, but no socket, no uvicorn process needed:

import fastapi.testclient
import research.Noesis.batch_call_auction as rnbacaau
import research.Noesis.passthrough_proxy as rnopapro
import research.Noesis.platform_api as rnoplapi

order_book = rnbacaau.OrderBook()
gateway = rnopapro.Gateway()
app = rnoplapi.create_app(order_book, gateway, {"key1": "acct_1"})
client = fastapi.testclient.TestClient(app)

r = client.post(
    "/bids",
    headers={"X-API-Key": "key1"},
    json={"buyer_id": "buyer_1", "n_tasks": 10000, "c_level_min": "frontier",
          "l_max": 2.0, "r_min": 0.999, "p_max": 0.02},
)
print(r.status_code, r.json())
Good for scripts, notebooks, or driving the market without paying HTTP overhead.

2. Real HTTP client against a running server (uvicorn up from earlier)

import requests

BASE_URL = "http://127.0.0.1:8000"
HEADERS = {"X-API-Key": "key1"}

r = requests.post(f"{BASE_URL}/bids", headers=HEADERS, json={
    "buyer_id": "buyer_1", "n_tasks": 10000, "c_level_min": "frontier",
    "l_max": 2.0, "r_min": 0.999, "p_max": 0.02,
})
r.raise_for_status()
print(r.json())

r = requests.post(f"{BASE_URL}/rounds/clear")
print(r.json())
Swap requests for httpx (sync or async with httpx.AsyncClient()) if you need async.

3. Generated typed client from the OpenAPI spec (fits the openapi.json from last turn)

pip install openapi-python-client
openapi-python-client generate --path openapi.json --meta none
Produces a package with a typed function per endpoint (bids_post.sync(client=..., body=BidRequest(...)), response parsed into the ContractResponse/RoundClearResponse models) — best when a client needs IDE autocomplete/type checking against the exact schema, or the API surface will keep changing and you want it to fail loudly on drift.
