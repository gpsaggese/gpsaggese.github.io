# Agentic Data Science Decisions

## Process

### D-1. Build on what exists; every PR ships something that runs

- **Decision**: each PR starts from an artifact already in the repo and ends in
  code that runs, not in another plan
- **Why**: the project had many plans and surveys and little code that ran; work
  kept restarting from scratch
- **Alternatives**: more scoping docs before building
- **Status**: accepted

### D-2. One plan, in the format of the planning guide

- **Decision**: `docs/plan.md` replaces `../track_plan.md`; the project keeps
  `plan.md`, `progress.md`, `decisions.md`, and `how_we_work.md`
- **Why**: `docs/planning/all.plan_work.how_to_guide.md` asks for one plan,
  because two plans drift apart (issue #880)
- **Alternatives**: keep `track_plan.md` next to a new plan
- **Status**: accepted

## Bake-off

### D-3. Keep the Bake-off in `tutorials/agentic_eda/bakeoff/` for now

- **Decision**: the harness lives next to the other `agentic_eda` code and follows
  its conventions (absolute imports, `_parse()`/`_main()`, `hunitest` tests)
- **Why**: the track plan put Project 1 code in `tutorials/agentic_eda/`, which
  already holds an EDA agent
- **Alternatives**: `research/agentic_data_science/bakeoff/`, next to
  `schema_agent/`, so that in-house entrants can be imported directly
- **Status**: proposed, see Open Question 1 of the Bake-off Harness in `plan.md`

### D-4. Offline samples by default; never fake real data

- **Decision**: `Dataset.load()` returns a small, deterministic, synthetic sample
  with the real schema; `load(real=True)` reads a cached CSV or a real loader and
  otherwise raises, naming the source
- **Why**: the harness and its tests run with no network access, and a run can
  never silently use fake data where real data was meant
- **Alternatives**: download the real data in every test run
- **Status**: accepted, PR `gpsaggese/tutorials#706`

## RL-for-EDA Benchmark

### D-5. Use existing causal-discovery libraries instead of a new benchmark

- **Decision**: generate graphs with `pgmpy` and run baselines with
  `causal-learn`
- **Why**: graph recovery against a known graph is a settled benchmark task, see
  `../RL_for_auto_EDA_benchmark_scoping.md`; `gCastle` was the first choice, but
  `castle.datasets` failed to import in Colab and it pulls in `torch`
- **Alternatives**: `gCastle`; a generator written from scratch
- **Status**: accepted, PR #587

### D-6. Score the CPDAG on linear-Gaussian data

- **Decision**: `shd(..., cpdag=True)` compares Markov equivalence classes by
  default
- **Why**: from observational linear-Gaussian data, edge direction is identifiable
  only up to the equivalence class, so a raw-DAG score penalizes answers that no
  method can get right
- **Alternatives**: non-Gaussian (LiNGAM) noise, where the full DAG is
  identifiable and the raw-DAG score is fair
- **Status**: proposed, waiting for GP's reading of the first worked example in
  the spec

### D-7. Leave the trained RL policy to the Tukey project

- **Decision**: this project keeps the environment, the scoring, and the
  benchmark; the trained policy is built by the Tukey class project
  (#628, PR #865), which already builds on `../RL_for_auto_EDA_prototype.py`
- **Why**: two teams building the same policy would duplicate work (D-1)
- **Alternatives**: train our own policy in parallel
- **Status**: proposed, to confirm with GP and the Tukey team
