# `Noesis`: Reorganize Code Into Components - Implementation Spec

- Reorganize `research/Noesis` from one flat dir (~40 entries: code of five
  components, docs, plans, specs, prompts) into one dir per component, each with a
  `README.md`; no behavior change
- Sources: `architecture.md`, `papers/Noesis/*.tex`, `docs/mvp_prd.md`,
  `docs/gp_todos.md` (the request)
- Roadmap position: independent of `plan.Noesis.md`; later PRs (`PR_M3`, `PR_S4`,
  `PR_S11`) land in the dirs defined here
- Specification only: no code in this document has been implemented

## Design Decisions

### Target Layout

- One top-level dir per component, named after the paper / PRD concept; all docs in
  `docs/`; tests next to the code they test (**decision**)
- `(from X)` marks a move; a file with no annotation stays where it is; `new` marks
  a file that does not exist today

```text
research/Noesis/
├── README.md                    # rewritten: component table, how to run
├── docs/                        # ALL docs
│   ├── README.md                # new: index and reading order
│   ├── architecture.md          # (from ./architecture.md)
│   ├── glossary.md
│   ├── onboarding.md
│   ├── how_we_work.md
│   ├── gateway_design.md        # (from docs/gateway.README.md, minus Files/Graph)
│   ├── checker_design.md        # (from docs/checker.README.md, minus Modules)
│   ├── plans/
│   │   ├── plan.Noesis.md       # (from ./plan.Noesis.md)
│   │   ├── plan.marketing.md    # (from ./plan.marketing.md)
│   │   ├── plan.reorg.md
│   │   ├── mvp_plan.md          # (from docs/)
│   │   └── mvp_progress.md      # (from docs/)
│   ├── specs/
│   │   ├── mvp_prd.md           # (from docs/)
│   │   ├── mvp_decisions.md     # (from docs/)
│   │   ├── spec.PR_P2.md        # (from ./spec.PR_P2.md)
│   │   ├── spec.PR_P2b.md       # (from ./spec.PR_P2b.md)
│   │   └── spec.reorg.md
│   ├── notes/
│   │   ├── explain_code.noesis.md
│   │   ├── explain_code.batch_call_auction.md
│   │   ├── coding.create_specs.md
│   │   ├── dev_notes.md         # new: "GP Notes" section of README.md
│   │   └── gp_todos.md          # (from docs/)
│   └── findings/                # unchanged
├── common/                      # shared MVP infra, data model
│   ├── README.md                # new
│   ├── noesis_settings.py
│   ├── noesis_db.py
│   ├── noesis_seed.py
│   ├── seed.yaml                # (from config/seed.yaml)
│   ├── migrations/              # (from migrations/): 001_init.sql, 002_checker.sql
│   └── test/
│       ├── noesis_test_utils.py # new: DB half of test/gateway_test_utils.py
│       └── test_noesis_{settings,db,seed}.py
├── checker/                     # NoesisServer: grades sellers (landed after the plan)
│   ├── README.md                # new: module table from checker.README.md
│   ├── checker_{canaries,evidence,grading,models,statistics,store,verdicts}.py
│   └── test/test_checker_*.py
├── market/                      # NoesisMarket: matching engine
│   ├── README.md                # new
│   ├── batch_call_auction.py
│   ├── contract_dispatch.py
│   └── test/test_{batch_call_auction,contract_dispatch}.py
├── gateway/                     # NoesisServer front door
│   ├── README.md                # new: Files and Import Graph from gateway.README.md
│   ├── gateway_{admin,api,app,fake_provider,faults,openrouter,providers,
│   │            request_log,routing}.py
│   ├── passthrough_proxy.py     # legacy in-process proxy, used by noesis_platform
│   └── test/
│       ├── gateway_test_utils.py  # Gateway + running_gateway() half
│       └── test_gateway_*.py, test_passthrough_proxy.py
├── noesis_platform/             # NoesisPlatform: HTTP service over market + proxy
│   ├── README.md                # new
│   ├── platform_api.py
│   ├── postgres_store.py
│   ├── main.py
│   └── test/test_{platform_api,postgres_store,main}.py
├── devops/                      # unchanged, plus new README.md
├── scripts/                     # unchanged, plus new README.md
├── test/test_component_imports.py  # new: only cross-component tests live here
└── .dockerignore, changelog.txt, conftest.py, env.example, invoke.yaml,
    pytest.ini, repo_config.yaml, tasks.py   # unchanged tool config at the root
```

### Rules Behind the Layout

- `noesis_platform/`, not `platform/`: pytest prepends `research/Noesis` to
  `sys.path` (for `conftest.py`), where a `platform` dir can shadow the stdlib module
- `passthrough_proxy.py` goes to `gateway/`: `mvp_prd.md` section 5.1 names it the
  gateway's ancestor; `gateway/README.md` marks it legacy, imported only by
  `noesis_platform/`
- File names stay unchanged (pure `git mv`): history follows, and `test_*.py`
  basenames stay unique, which pytest needs without `__init__.py` in test dirs
- No `__init__.py`: `research/Noesis` is a namespace package today
- `test/gateway_test_utils.py` is split because `common/` and `checker/` tests use
  its DB helpers (`get_test_settings()`, `run_with_clean_db()`, `fetch_all()`,
  `add_*()`) while only `Gateway` and `running_gateway()` use `gateway_app`;
  `common/` never imports `gateway/`
- `docs/` groups by kind: `plans/` (what to build, when), `specs/` (what and why),
  `notes/` (explanations, scratch, TODOs); top of `docs/` keeps the docs a new reader
  opens first
- Dependency rules, enforced by `test/test_component_imports.py` (layering per
  `architecture.rules.md`):

  | Dir | May import |
  | :-- | :-- |
  | `common/`, `market/` | nothing in `research.Noesis` |
  | `gateway/` | `common/` |
  | `checker/` | `common/`, `gateway/gateway_providers` (status constants) |
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
- `explain_code.*.md` next to its code (`market/`): rejected, the request is "all
  docs in `docs/`"; each component README links to its note instead

## Out of Scope

- Renaming modules or classes; splitting `platform_api.py` or other modules
- Moving `scripts/` code (`detection_math.py` belongs to a future `metering/`)
- The `section` notation cleanup in docs: tracked in `docs/notes/gp_todos.md`
- Updating the deployed ECS task definition (outside this repo)

## Current State

- Fact: only files under `research/Noesis` import `research.Noesis.*`
- Fact: `noesis_settings.NOESIS_DIR` is `dirname(__file__)`: moving the file breaks
  `.env`, `config/seed.yaml`, `migrations/` lookups silently
- Fact: `uvicorn research.Noesis.noesis_platform.main:app` is in
  `devops/compose/docker-compose.noesis.yml`, `README.md`, `spec.PR_P2.md`

## Implementation

- Five stacked PRs (`plan.reorg.md`): `docs/`; `market/`; `common/` + `gateway/`;
  `noesis_platform/`; READMEs + architecture refresh
- Per code PR: `git mv` per `.claude/skills/git.move/SKILL.md`, fix imports,
  aliases, and docs, run the full suite
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

- Built: all five PRs of `plan.reorg.md` in one working tree; no commit
- Deviation: `checker/` (7 modules, 5 tests, `002_checker.sql`) landed after this
  spec and got its own dir; `checker.README.md` split like `gateway.README.md`
- Deviation: import aliases (`rnbacaau`, ...) were kept, not re-derived: the
  existing aliases are not a strict function of the module path
- Verified: `pytest research/Noesis` gives 123 passed / 38 skipped before and after
  (Postgres tests skip without a test DB); `test_component_imports.py` adds 3 passed
- Not verified: Postgres-backed tests (no test DB), `docker compose config`, a live
  `uvicorn` start
