### [x] Reorganize `research/Noesis` Into Components

* Repo: umd_classes only (`helpers_root` is not touched)
- [ ] helpers (https://github.com/causify-ai/helpers)
- [x] umd_classes (https://github.com/gpsaggese/gpsaggese.github.io)

* Problem
- `research/Noesis` is one flat dir of ~40 entries: code of four components (market,
  gateway, platform, shared infra), docs, plans, specs, and AI prompts sit side by
  side
- Goal: one dir per component, one `docs/` for all docs, and a `README.md` in each
  dir that explains its files
- Target layout and rules: `spec.reorg.md`; sources: `architecture.md`,
  `papers/Noesis/*.tex`, `docs/mvp_prd.md`
- Closes these items in `docs/gp_todos.md`:
  - Move docs from `research/Noesis` to `research/Noesis/docs`
  - Reorg docs
  - Separate components in dirs (e.g., gateway, auction, ...)
  - Capture content of PR #597 in a README

* Info
- **Type**: improvement
- **Reason of the problem**: the dir grew PR by PR (`PR_M1`/`PR_S1`/`PR_P1`, then the
  MVP gateway) without a component boundary
- **Confidence in the fix**: high
- **Fix complexity**: medium (many import lines; one path anchor, `NOESIS_DIR`,
  changes meaning)
- **Verification plan**:
  - `pytest research/Noesis` before and after each PR: same pass/skip counts, run
    once against the test DB so Postgres tests do not skip
  - `grep -r "research.Noesis.<old module>"` returns nothing after each move
  - `uvicorn` starts both entry points and `GET /health` returns 200
  - `docker compose -f devops/compose/docker-compose.noesis.yml config` is valid
  - `test/test_component_imports.py` passes (dependency rules)
  - Every component dir, `docs/`, `devops/`, `scripts/` has a `README.md` whose file
    list matches `ls`

* Solution

- [x] PR1: Consolidate all docs in `docs/` (see `spec.reorg.md`)
  - Move root docs: `architecture.md`, `plan.*.md`, `spec.*.md`,
    `explain_code.*.md`, `coding.create_specs.md`
  - Group into `docs/plans/`, `docs/specs/`, `docs/notes/`; keep `docs/findings/`
  - `gateway.README.md` becomes `gateway_design.md`
  - Add `docs/README.md`: index and reading order
  - Fix references in code docstrings, docs, and `devops/`
  - Depends on: nothing

- [x] PR2: Extract `market/` (see `spec.reorg.md`)
  - Move `batch_call_auction.py`, `contract_dispatch.py`, and their tests
  - Update importers: `platform_api.py`, `postgres_store.py`, tests
  - Add `market/README.md`
  - Depends on: PR1

- [x] PR3: Extract `common/`, `gateway/`, and `checker/` (see `spec.reorg.md`)
  - `common/`: `noesis_{settings,db,seed}.py`, `migrations/` (incl.
    `002_checker.sql`), `seed.yaml`; fix `NOESIS_DIR` / `MIGRATIONS_DIR` /
    `DEFAULT_SEED_PATH` anchors
  - `gateway/`: 9 `gateway_*.py` and `passthrough_proxy.py`
  - `checker/`: 7 `checker_*.py` and their tests (added after this plan was
    written); `docs/checker.README.md` becomes `checker_design.md`
  - Split `gateway_test_utils.py`: DB helpers to `common/test/noesis_test_utils.py`
  - Update the gateway `uvicorn` target in docs
  - Add `common/README.md`, `gateway/README.md`, and `checker/README.md` (files,
    import graph from `gateway_design.md`)
  - Depends on: PR2 (same importers; restack)

- [x] PR4: Extract `noesis_platform/` (see `spec.reorg.md`)
  - Move `platform_api.py`, `postgres_store.py`, `main.py`, and their tests
  - Update `uvicorn research.Noesis.noesis_platform.main:app` in
    `devops/compose/docker-compose.noesis.yml`, `README.md`, specs
  - Add `noesis_platform/README.md`
  - Depends on: PR2, PR3

- [x] PR5: READMEs, architecture refresh, boundary test
  - Rewrite root `README.md`: component table, how to run; move "GP Notes" to
    `docs/notes/dev_notes.md`; drop the stale `TODO(ai_gp)` (the script exists)
  - Add `devops/README.md` and `scripts/README.md`
  - Update `docs/architecture.md` C2 to one boundary per component dir
  - Add `test/test_component_imports.py`
  - Depends on: PR4
