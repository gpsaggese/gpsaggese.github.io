# Agentic Data Science Implementation Plan

## Goal

- Automate data science with agents, the way coding agents automated
  programming: given a dataset and a goal, an agent explores the data, tests
  hypotheses, and reports what it found and how far to trust it
- The hard part is verification: real data has no test suite, so we build
  harnesses that score agents against something checkable
  - The **Bake-off** runs many EDA agents on the same datasets and scores them
    side by side
  - The **RL-for-EDA benchmark** generates data from a known causal graph, so an
    agent's answer is scored exactly
- Background:
  - Vision and roadmap:
    [Google Doc](https://docs.google.com/document/d/1fAao18l5oqeGvmGq4V4S0RGvXggXAFtsiv9gMIt0UlU)
  - Paper: `papers/RL_for_Automated_EDA/`
  - Specs: `research/ideas/in_progress.Comparison_of_Data_Science_Agents.md`,
    `research/ideas/in_progress.RL_for_auto_EDA.md`
  - SOTA: `../RL_for_auto_EDA_sota_review.md`,
    `../RL_for_auto_EDA_benchmark_scoping.md`, `../sota_eda.md`

## Plan Hierarchy

- This file: what to build and in which order
  - It replaces `../track_plan.md`
- `progress.md`: what is done and learned
- `decisions.md`: why it is built this way
- `- [ ]` is an unmerged PR and `- [x]` is a merged PR
- PR ids are in `gpsaggese.github.io`, except `PR_B1`, which was merged in
  `gpsaggese/tutorials` before that repo was archived

## Roadmap

- `v0.1`: two harnesses run end to end: the Bake-off with a dummy agent on four
  datasets, and the RL-for-EDA benchmark scoring a scripted agent against PC and
  GES
  - [x] `PR_R1`
  - [x] `PR_R2`
  - [x] `PR_B1`
- `v0.2`: first real comparison: `schema_agent` and one external agent scored on
  the real UCI Heart and NYC Taxi data
  - [ ] `PR_B2`
  - [ ] `PR_B3`
  - [ ] `PR_B4`
- `v0.3`: insight quality is scored, not only "did it run"
  - [ ] `PR_B5`
  - [ ] `PR_B6`
  - [ ] `PR_R3`
- `v0.4`: every in-house agent is an entrant
  - [ ] `PR_B7`
  - [ ] `PR_B8`
  - [ ] `PR_B9`
- `v1.0`: results table and a "Bake-off in 60 minutes" tutorial
  - [ ] `PR_B10`
- Releases are cumulative: a version also needs the PRs of the versions before it

# Component Specs

## Bake-off Harness

### Goal

- Run every registered agent on every registered dataset, time it, record
  failures without stopping, and write one scorecard
- Code: `research/agentic_data_science/bakeoff/` (`datasets.py`, `agents.py`,
  `scoring.py`, `run_bakeoff.py`)

### Open Questions

1. Resolved: the Bake-off moved to `research/agentic_data_science/bakeoff/`, see
   `decisions.md` D-3
2. Is the class project
   `UmdTask850_MSML610_Fall2026_Comparison_of_Data_Science_Agents` (PR #859)
   building the same comparison? If so, its team should add entrants
   to this harness instead of writing a second one
3. How large a sample of NYC Taxi and Amazon Reviews keeps a full run under a few
   minutes and the API cost low?

## Agent Entrants

### Goal

- Wrap each agent behind `Agent.run(dataset) -> AgentResult`, without changing
  the agent itself
- In-house: `research/agentic_data_science/schema_agent/`,
  `tutorials/agentic_eda/intermediate_v0_timeseries_agent/`, `research/AgenTUI/`,
  the Claude Code agent in `../RL_for_auto_EDA_prototype.py`
- External baseline: PandasAI or Data Interpreter

### Open Questions

1. How do we turn each agent's output (JSON profile, notebook, free text) into a
   list of insights in the same shape?
2. What API budget per full run is acceptable?

## Scoring

### Goal

- Fill the `insight_quality` and `code_quality` columns of the scorecard, which
  are blank since `PR_B1`

### Open Questions

1. LLM judge, or a task set with known answers (InsightBench, DABStep style)? A
   task set may need its own project -> Project
2. On synthetic data from a known graph, insight quality is checkable: does a
   claimed relation match an edge of the graph? This reuses the RL-for-EDA
   scoring, see `PR_B5`

## RL-for-EDA Benchmark

### Goal

- Sample a causal graph, generate data from it, hide the graph, and score an
  agent's recovered graph with SHD and out-of-sample prediction
- Code: `../RL_for_auto_EDA_prototype.py`, `../RL_for_auto_EDA_benchmark.py`
- The trained RL policy is built by the Tukey class project (#628, PR #865) on top
  of this prototype; this project does not duplicate it, see `decisions.md` D-7

### Open Questions

1. Score against the CPDAG (linear-Gaussian data) or generate non-Gaussian data
   and score the raw DAG? See `decisions.md` D-6
2. Should the environment and the scoring move into one module that both the
   Bake-off and Tukey import?
3. Does a policy trained on synthetic graphs transfer to real data? -> Project

## Benchmark Meta-Dataset

### Goal

- Turn the prose surveys into one verified, queryable table of published results
  on data-science agent benchmarks (DataSciBench, DSBench, MLE-Bench, GAIA,
  InsightBench, ...)

### Open Questions

1. Is this one project, or part of AutoEDA EPIC 9 (benchmarking)? -> Project

# PRs

## v0.1: Two Harnesses Run End to End

### `PR_R1`: [x] RL-for-EDA prototype

- Repo: `gpsaggese.github.io`, PR #527
- Synthetic environment, SHD scoring, a scripted agent, and a Claude Code agent
  driven by `cc_lib.PromptSequencer`

### `PR_R2`: [x] Benchmark against PC and GES

- Repo: `gpsaggese.github.io`, PR #587
- `RL_for_auto_EDA_benchmark.py` compares random, scripted, PC, and GES on CPDAG
  SHD over 10 graphs for each of four graph sizes

### `PR_B1`: [x] Bake-off harness skeleton

- Repo: `gpsaggese/tutorials`, PR `gpsaggese/tutorials#706`
- Dataset registry with offline samples, `Agent` interface with `DummyAgent`,
  `Scorecard`, `run_bakeoff.py --list/--run`, 12 unit tests

## v0.2: First Real Comparison

### `PR_B2`: [ ] Load the real UCI Heart and NYC Taxi data

- Background: `Dataset.load(real=True)` in `bakeoff/datasets.py` reads
  `data/<name>.csv` or raises `NotImplementedError`; no dataset has a real loader
- Add a `real_loader` for `uci_heart` (UCI repository, 303 rows) and for
  `nyc_taxi` (one month of TLC Yellow Taxi data, a fixed sample of 10k rows)
- Cache each download in `bakeoff/data/`, which is git-ignored
- Add `--real` to `run_bakeoff.py` to run on real data
- Unit tests: the loaders return the expected columns and dtypes on a small
  fixture CSV, with no network access; `--real` without a cache gives a clear
  error
- Result: the Bake-off runs on two real datasets

### `PR_B3`: [ ] `schema_agent` as an entrant

- Background: `research/agentic_data_science/schema_agent/schema_agent.py data.csv`
  writes `data_profile_report.json` and `data_profile_summary.md`; it needs
  `OPENAI_API_KEY`
- Add `SchemaAgent(Agent)` in `bakeoff/agents.py` that writes the dataset to a
  temp CSV, runs the pipeline, and maps the column roles, quality notes, and
  hypotheses in the JSON report to `AgentResult.insights`
- Register it in `all_agents()`; skip it with a clear reason when the key or the
  code is missing
- Depends on Open Question 1 of the Bake-off Harness: where the code lives
- Unit tests: the mapping from a saved example JSON report to insights; the skip
  path when the key is missing
- Result: the scorecard compares `schema_agent` with `DummyAgent` on UCI Heart and
  NYC Taxi

### `PR_B4`: [ ] External baseline agent

- Background: the comparison spec names PandasAI and Data Interpreter as
  reference agents; none runs in the repo
- Add `PandasAIAgent(Agent)` that asks a fixed set of EDA questions and collects
  the answers as insights
- Pin the version in the requirements of the runnable dir
- Unit tests: the wrapper with the LLM call mocked; the error path is recorded as
  a failure
- Result: an external reference point next to the in-house agents

## v0.3: Score Insight Quality

- `PR_B5`: synthetic causal dataset in the Bake-off, generated by the RL-for-EDA
  environment, where an insight is scored against the true graph
- `PR_B6`: rubric scoring of `insight_quality` and `code_quality` on the real
  datasets
- `PR_R3`: the Claude Code agent from `PR_R1` as an entrant in both harnesses

## v0.4: All In-House Agents

- `PR_B7`: time-series agent from
  `tutorials/agentic_eda/intermediate_v0_timeseries_agent/`
- `PR_B8`: AgenTUI (prompt -> executed notebook -> insights)
- `PR_B9`: the Tukey policy (#628) as an entrant, once it is merged

## v1.0: Results and Tutorial

- `PR_B10`: full agent-by-dataset results table, written up, and a "Bake-off in
  60 minutes" tutorial

## Unscheduled: RL-for-EDA Benchmark

- `PR_R4`: add NOTEARS and DirectLiNGAM to `RL_for_auto_EDA_benchmark.py`, and
  report TPR and FDR next to SHD
- `PR_R5`: shared environment and scoring module, if Open Question 2 is answered
  yes

## Unscheduled: Benchmark Meta-Dataset

- `PR_M1`: schema and store, with two benchmarks loaded by hand
- `PR_M2`: the remaining benchmarks, with a source for each row
- `PR_M3`: cross-benchmark correlation of the models on two or more leaderboards
- `PR_M4`: reproducibility audit of 10 to 20 tasks from one benchmark

## Conventions

- Coding rules: `helpers_root/.claude/rules.md`, `.claude/skills/coding.rules.md`
- Testing rules: `.claude/skills/testing.rules.md`
- Process: `docs/planning/all.plan_work.how_to_guide.md` in `helpers`
