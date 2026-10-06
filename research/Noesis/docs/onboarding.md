# Noesis — Onboarding

- Goal: from zero to running the gateway and its tests in about an hour, then
  picking a first task
- Read next: `how_we_work.md` (process), `glossary.md` (terms),
  `gateway.README.md` (design)

## 1. What Noesis Is (5 Minutes)

- A market for LLM inference where providers are held to what they promise: a
  contract covers capability, latency, and reliability, not just tokens
- The new part, and the MVP's focus, is **verifying delivery**: we measure every
  request, test providers with known-answer questions, and lower the reputation of
  sellers that under-deliver
- Background: the paper `papers/Noesis/Noesis.pdf` (by GP), GP's roadmap
  `research/Noesis/plan.Noesis.md`, and the MVP spec `docs/mvp_prd.md`
- Status: `docs/mvp_progress.md`

## 2. Access You Need

| What | From whom | Why |
|---|---|---|
| A GitHub account with SSH keys | You | You'll fork GP's repo and open PRs from your fork (`class_project/how_to_contribute.md`) |
| OpenRouter API key with a small credit limit (~$5) | Create your own at openrouter.ai | Real provider calls; tests don't need it |
| Team chat + the private backup repo | Javin | Coordination, non-public material |

## 3. Setup (macOS / Linux)

- Fork `gpsaggese/gpsaggese.github.io` on GitHub (button "Fork", or
  `gh repo fork gpsaggese/gpsaggese.github.io --clone=false`)
- Clone **your fork** **outside** iCloud/Dropbox-synced folders (e.g., not
  `~/Desktop` or `~/Documents` on macOS): sync services corrupt `.git` directories

  ```bash
  > mkdir -p ~/dev && cd ~/dev
  > git clone git@github.com:{your_username}/gpsaggese.github.io.git
  > cd gpsaggese.github.io
  # GP's repo is `upstream`: pull from it to stay current.
  > git remote add upstream git@github.com:gpsaggese/gpsaggese.github.io.git
  > git fetch upstream
  # `helpers_root` is a submodule.
  > git submodule update --init helpers_root
  ```

- If checkout fails on large files: the repo stores some PDFs with Git LFS. Either
  `brew install git-lfs && git lfs install`, or skip them (the Noesis code doesn't
  need them):

  ```bash
  > git config filter.lfs.smudge "" && git config filter.lfs.process "" \
      && git config filter.lfs.required false
  > git restore --source=HEAD :/
  ```

- Python 3.11 environment (kept outside the repo so it's never committed):

  ```bash
  > uv venv -p 3.11 ~/dev/.venvs/noesis
  > VIRTUAL_ENV=~/dev/.venvs/noesis uv pip install fastapi numpy pandas \
      psycopg2-binary pydantic pytest uvicorn httpx "psycopg[binary,pool]" \
      pyyaml openai async_solipsism tqdm botocore
  ```

  - The last three are needed by GP's `helpers` library at import time
  - The Noesis service dependencies are also declared in
    `research/Noesis/devops/docker_build/pyproject.toml` (GP's Docker image)

- Local Postgres for tests and the dev server (port 5433, user/password `noesis`):

  ```bash
  > docker run -d --name noesis_pg -p 5433:5432 \
      -e POSTGRES_USER=noesis -e POSTGRES_PASSWORD=noesis \
      -e POSTGRES_DB=noesis_test postgres:16
  > docker exec noesis_pg createdb -U noesis noesis_dev
  ```

  - `noesis_test` is wiped by the tests; `noesis_dev` is for the server
  - Tables are created automatically on startup (`migrations/001_init.sql`)

- Secrets: copy the template and fill it in. `research/Noesis/.env` is git-ignored

  ```bash
  > cp research/Noesis/env.example research/Noesis/.env
  # Generate each Noesis key with:
  > python3 -c "import secrets; print(secrets.token_urlsafe(32))"
  ```

## 4. Run the Tests

- From the repo root:

  ```bash
  > export PYTHONPATH=$PWD/helpers_root:$PWD
  > ~/dev/.venvs/noesis/bin/python -m pytest -q research/Noesis/test
  ```

- Expected: all pass; GP's Docker-in-Docker Postgres tests (`test_postgres_store.py`)
  are skipped outside his dev container
- No Postgres running? The Noesis DB tests skip instead of failing; start it
  (section 3) to run them

## 5. Run the Gateway

```bash
> export PYTHONPATH=$PWD/helpers_root:$PWD
> ~/dev/.venvs/noesis/bin/uvicorn research.Noesis.gateway_app:build_default_app \
    --factory --port 8000
```

- Open `http://localhost:8000/docs` to click through the endpoints
- Make a contract and send one real request (costs a fraction of a cent):

  ```bash
  > source research/Noesis/.env
  > curl -s -X POST localhost:8000/admin/dev-contract \
      -H "Authorization: Bearer $NOESIS_OPERATOR_KEY" -H "Content-Type: application/json" \
      -d '{"buyer_id": "buyer_1", "seller_id": "groq"}'
  # Use the returned contract_id below.
  > curl -s localhost:8000/v1/chat/completions \
      -H "Authorization: Bearer $NOESIS_BUYER_1_KEY" -H "X-Noesis-Contract: 1" \
      -H "Content-Type: application/json" \
      -d '{"model": "meta-llama/llama-3.3-70b-instruct",
           "messages": [{"role": "user", "content": "What is 17 * 23?"}]}'
  ```

- Any OpenAI SDK works: set `base_url="http://localhost:8000/v1"`, the buyer key
  as `api_key`, and send the header `X-Noesis-Contract`

## 6. Read the Code in This Order

1. `docs/gateway.README.md`: the picture and the request flow
2. `gateway_providers.py`: the interface everything plugs into
3. `gateway_api.py`: the main endpoint, top to bottom
4. `gateway_routing.py`, `gateway_request_log.py`, `gateway_openrouter.py`
5. `test/test_gateway_api.py`: how the whole thing is exercised
6. GP's `batch_call_auction.py` (the market milestone builds on it)

## 7. Pick a First Task

- Open tasks with owners are in `docs/mvp_plan.md` §3; the next milestone is the
  checker (M2)
- Good first tasks:
  - Add test questions to the canary bank (M2, T2.1): no prior code knowledge needed
  - Add a test for an edge case you find while reading
  - Improve a doc that confused you
- Then follow `how_we_work.md`: plan -> approval -> issue -> branch in your fork
  -> PR to GP's repo

## 8. Common Problems

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: helpers` | `export PYTHONPATH=$PWD/helpers_root:$PWD` from the repo root |
| `ModuleNotFoundError: async_solipsism` (or `tqdm`, `botocore`) | Install it in the venv; `helpers` imports it |
| DB tests all skipped | Postgres not running on :5433 (section 3) |
| `OPENROUTER_API_KEY is required` | Fill in `research/Noesis/.env` |
| Imports hang or are very slow on macOS | Repo or venv inside an iCloud folder; move to `~/dev` |
| `git` shows weird files like `main 2` in `.git` | Same iCloud problem; re-clone outside synced folders |
