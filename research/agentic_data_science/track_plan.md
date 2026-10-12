# Agentic Data Science Track Plan

**Owners:** Dev Shah, GP Saggese
**Status:** Draft

## Purpose

Turn the EDA-track ideas that already exist in the repo into executed, committed
artifacts. The repo currently holds a large surplus of plans and surveys and very
little that runs. This plan is deliberately **execution-only**: every task below
starts from an artifact that already exists and ends in runnable code or a committed
dataset, never another plan.

## What already exists (build on these — do not restart)

| Area | Artifact in repo | State |
|---|---|---|
| In-house profiler agent | `research/agentic_data_science/schema_agent/` | Runs: CSV -> `data_profile_report.json` + `data_profile_summary.md` |
| In-house EDA agent fragments | `causify-ai/tutorials/agentic_eda/` (`generate_EDA_context.py`, `intermediate_v0_timeseries_agent/`) | Partial, merged |
| Notebook agent | `research/AgenTUI/` | Mature: prompt -> executed notebook |
| Agent-comparison spec | `research/ideas/in_progress.Comparison_of_Data_Science_Agents.md` | 90% specs, 0% executed |
| Benchmark-survey spec | `research/ideas/draft.Benchmarking_Data_Science_Agents.md` | 90% specs, 0% executed |
| Surveys | `sota_eda.md`, `deep-research-report.md`, `agentic_data_science.report.md` | Written, unverified drafts |
| AutoEDA plan/backlog | `AutoEDA_Plan.md`, `AutoEDA_Issues.md` | EPICs mostly not started; EPIC 9 (benchmarking) open |
| RL-for-EDA track | `RL_for_auto_EDA_*` | Prototype + benchmark run live |

## How we work (applies to every PR)

- **One PR = one clear, runnable deliverable**, reviewable in a few days.
- **Commit where the code already lives:** `causify-ai/tutorials/agentic_eda/`
  (Project 1) or `research/agentic_data_science/` (Project 2). No new top-level homes.
- **Claim an existing EPIC/issue** before opening a parallel one. Do not start a third
  copy of the competing `helpers#989` / `#991` drafts.
- **Definition of done = a shipped artifact**: code that runs end to end on one input,
  a committed results file, and a short README. Not a plan doc.
- **First running PR within ~3 days** of a student picking up a task.
- Each project ends with a "Learn X in 60 minutes" tutorial, per the class-project format.

---

## Project 1 — Agentic EDA Bake-off

Run the agents we already have on shared datasets and score their insight output.
Produces the regression harness the AutoEDA effort lacks (EPIC 9).

- **Datasets** (from the comparison spec): UCI Heart Disease, NYC Yellow Taxi (sampled),
  OpenAQ air quality, Amazon Reviews 2023 (All_Beauty).
- **Rubric** (from the comparison spec): insight quality, does-it-run, code quality, runtime.

| PR | Scope | Builds on | Definition of done | Size |
|---|---|---|---|---|
| 1.1 | Harness skeleton: dataset loaders + `Agent` and `Scorecard` interfaces; a dummy agent runs end to end | comparison spec | `run_bakeoff.py --list` shows datasets + agents; dummy agent emits an (empty) scorecard | ~3d, foundation |
| 1.2 | Adapter: run `schema_agent` as an entrant, map its profile output to scorecard fields | `schema_agent/` | schema_agent scored on 2 datasets; results committed | ~3d |
| 1.3 | Adapter: run the `tutorials/agentic_eda` time-series agent + `generate_EDA_context.py` | `tutorials/agentic_eda/` | runs on NYC Taxi; scored; results committed | ~4d |
| 1.4 | Adapter: run AgenTUI (prompt -> executed notebook -> extract insights) | `research/AgenTUI/` | AgenTUI scored on 1 dataset | ~4d |
| 1.5 | External reference agent (PandasAI or Data Interpreter) as a baseline | comparison spec | one external agent scored on 2 datasets | ~3d |
| 1.6 | Scoring: implement the rubric (insight quality via LLM-judge or InsightBench-style tasks; runtime; does-it-run; code quality) | comparison spec, `sota_eda.md` | scorecard computes all columns; full agent x dataset table committed | ~5d |
| 1.7 | Tutorial + results writeup ("bake-off in 60 min") | all above | committed tutorial notebook + results table + README | ~3d |

**Order:** 1.1 unblocks everything. 1.2-1.5 run in parallel once 1.1 lands. 1.6 needs 1.1;
1.7 last.

---

## Project 2 — DS-Benchmark Meta-Dataset

Turn the three prose surveys into one verified, queryable dataset of benchmark results.
The reproducibility harness it builds is exactly EPIC 9's benchmarking need.

- **Benchmarks:** DataSciBench, DSBench, MLE-Bench, GAIA, SWE-bench (core);
  DA-Code, DABStep, InsightBench, BLADE, VisEval (from the surveys).
- **Schema:** `benchmark, model, score, task_type, difficulty, source_url, date`.

| PR | Scope | Builds on | Definition of done | Size |
|---|---|---|---|---|
| 2.1 | Schema + store (SQLite/CSV) + loader; load 2 benchmarks by hand | benchmarking spec | queryable store with >=2 benchmarks, committed | ~3d, foundation |
| 2.2 | Load the remaining benchmarks into the same schema, with provenance | surveys | >=6 benchmarks loaded; source recorded per row | ~5d |
| 2.3 | Taxonomy table: classify by task style / metric / tools | benchmarking spec | committed taxonomy table + notebook | ~3d |
| 2.4 | Cross-benchmark correlation for models on >=2 leaderboards + heatmap | 2.1-2.2 | committed analysis notebook + figure | ~3d |
| 2.5 | Reproducibility audit: replicate 10-20 tasks from one benchmark (e.g., GAIA L1), report discrepancy | benchmarking spec | committed audit notebook + numbers | ~4d |
| 2.6 | Tutorial + writeup; cross-link `sota_eda.md` as the verified upgrade | all above | tutorial + README | ~3d |

**Order:** 2.1 unblocks the rest. 2.2 feeds 2.3/2.4. 2.5 can start early on one benchmark.

---

## Assignment table (to fill in)

| PR | Student | Status | Notes |
|---|---|---|---|
| 1.1 |  | open |  |
| 1.2 |  | open |  |
| 1.3 |  | open |  |
| 1.4 |  | open |  |
| 1.5 |  | open |  |
| 1.6 |  | open |  |
| 1.7 |  | open |  |
| 2.1 |  | open |  |
| 2.2 |  | open |  |
| 2.3 |  | open |  |
| 2.4 |  | open |  |
| 2.5 |  | open |  |

## Onboarding a student to a PR

1. Read the one artifact the PR builds on (column 3) and the comparison/benchmark spec.
2. Open (or claim) the matching issue; branch per the repo naming convention.
3. Ship the smallest running version first, PR within ~3 days, then iterate.

## Open ideas / to discuss (GP + Dev add here)

- Which datasets beyond the four for Project 1?
- Insight-quality scoring: LLM-judge vs. an InsightBench/DABStep-style task set?
- Do we fold Project 1's harness directly into AutoEDA EPIC 9, or keep it standalone first?
- Minimum number of benchmarks for Project 2 to be useful?
- Which PRs are good "first contribution" tasks for a new student (1.1, 2.1)?
