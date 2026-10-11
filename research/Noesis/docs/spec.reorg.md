# `Noesis`: Reorganize Code Into Components - Implementation Spec

- Reorganize `research/Noesis` from one flat dir (~40 entries: code of four
  components, docs, plans, specs, prompts) into one dir per component, each with a
  `README.md`; no behavior change
- Sources: `architecture.md`, `papers/Noesis/*.tex`, `docs/mvp_prd.md`,
  `docs/gp_todos.md` (the request)
- Roadmap position: independent of `plan.Noesis.md`; later PRs (`PR_M3`, `PR_S4`,
  `PR_S11`) land in the dirs defined here
- Specification only: no code in this document has been implemented

## Design Decisions

- One top-level dir per component, named after the paper / PRD concept
  (**decision**):

  | Dir | Concept | Files moved in |
  | :-- | :-- | :-- |
  | `docs/` | all docs | root `*.md` docs, existing `docs/` |
  | `common/` | shared MVP infra | `noesis_{settings,db,seed}.py`, `migrations/`, `seed.yaml` |
  | `market/` | `NoesisMarket` | `batch_call_auction.py`, `contract_dispatch.py` |
  | `gateway/` | `NoesisServer` front door | 9 `gateway_*.py`, `passthrough_proxy.py` |
  | `noesis_platform/` | `NoesisPlatform` service | `platform_api.py`, `postgres_store.py`, `main.py` |

- `noesis_platform/`, not `platform/`: pytest prepends `research/Noesis` to
  `sys.path` (for `conftest.py`), where a `platform` dir can shadow the stdlib module
- `passthrough_proxy.py` goes to `gateway/`: `mvp_prd.md` section 5.1 names it the
  gateway's ancestor; `gateway/README.md` marks it legacy, imported only by
  `noesis_platform/`
- File names stay unchanged (pure `git mv`): history follows, and `test_*.py`
  basenames stay unique, which pytest needs without `__init__.py` in test dirs
- No `__init__.py`: `research/Noesis` is a namespace package today
- Tests move next to code (`<component>/test/test_<file>.py`); top-level `test/`
  keeps only cross-component tests
- `test/gateway_test_utils.py` is split: DB helpers (`get_test_settings()`,
  `run_with_clean_db()`, `fetch_all()`, `add_*()`) go to `common/test/`; `Gateway`
  and `running_gateway()` stay in `gateway/test/` (fact: only they use
  `gateway_app`); `common/` never imports `gateway/`
- Dependency rules, enforced by a test (layering per `architecture.rules.md`):

  | Dir | May import |
  | :-- | :-- |
  | `common/`, `market/` | nothing in `research.Noesis` |
  | `gateway/` | `common/` |
  | `noesis_platform/` | `market/`, `gateway/passthrough_proxy` |

- `common/` anchors paths: `COMMON_DIR = dirname(__file__)` for `migrations/` and
  `seed.yaml`; `NOESIS_DIR = dirname(COMMON_DIR)` stays the root for `.env`
- READMEs follow `.claude/skills/readme.rules.md`: component dirs, `docs/`,
  `devops/`, `scripts/`; not `test/`, `migrations/`
- No placeholder dirs (`metering/`, `reputation/`, `simulator/`): the PR adding the
  first file creates the dir

## Trade-off and Alternative Design

- Drop the `gateway_` prefix in `gateway/`: reads better, but breaks the unique test
  basenames and `git blame` continuity; follow-up if wanted
- `passthrough_proxy.py` in `noesis_platform/`: removes one dependency edge, but
  hides the proxy from the dir that replaces it
- `legacy/` dir for the HTTP surface: rejected, `POST /rounds/clear` survives (PRD
  K5)

## Out of Scope

- Renaming modules or classes; splitting `platform_api.py` or other modules
- Moving `scripts/` code (`detection_math.py` belongs to a future `metering/`)
- The `section` notation cleanup in docs: tracked in `docs/gp_todos.md`
- Updating the deployed ECS task definition (outside this repo)

## Current State

- Fact: only files under `research/Noesis` import `research.Noesis.*`
- Fact: `noesis_settings.NOESIS_DIR` is `dirname(__file__)`: moving the file breaks
  `.env`, `config/seed.yaml`, `migrations/` lookups silently
- Fact: `uvicorn research.Noesis.main:app` is in
  `devops/compose/docker-compose.noesis.yml`, `README.md`, `spec.PR_P2.md`

## Implementation

- Five stacked PRs (`plan.reorg.md`): `docs/`; `market/`; `common/` + `gateway/`;
  `noesis_platform/`; READMEs + architecture refresh
- Per code PR: `git mv` per `.claude/skills/git.move/SKILL.md`, fix imports,
  aliases, and docs, run the full suite
- `docs/gateway.README.md` becomes `gateway_design.md`; its "Files" and "Import
  Graph" sections move to `gateway/README.md`
- `architecture.md` C2: one `Container_Boundary` per component dir; C3 unchanged

## Interaction with Existing Code

- Every importer updates its `import` line and alias; no re-export shims
  (`coding.fix_from_imports`)
- Entry points become `research.Noesis.noesis_platform.main:app` and
  `research.Noesis.gateway.gateway_app:build_default_app`
- Root `conftest.py`, `pytest.ini`, `invoke.yaml`, `tasks.py` symlinks stay

## Configuration and Secrets

- Not applicable: `.env` and `env.example` stay at the root

## Unit Test Plan

- Existing tests pass with only import-line changes
- New `test/test_component_imports.py`: fails on an import outside the dependency
  table
- New `common/test/test_noesis_settings.py` case: `MIGRATIONS_DIR` and
  `DEFAULT_SEED_PATH` point to existing files

## Risks and Limitations to Call Out

- A wrong `NOESIS_DIR` fails only at gateway startup or in Postgres tests, which
  skip without a DB: run the suite against the test DB before merging
- Out-of-repo users of the old entry-point path (ECS task definition, "GP Notes")
  break until updated
- Stacked PRs share import lines in `platform_api.py`, `postgres_store.py`,
  `main.py`: restack after each merge

## Result (to Fill in Once Implemented)

- Placeholder: record what was built vs. deferred, and any deviation
