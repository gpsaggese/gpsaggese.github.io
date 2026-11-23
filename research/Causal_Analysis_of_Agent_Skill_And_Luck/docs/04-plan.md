# Causal Analysis of Agent Skill and Luck Implementation Plan

## Goal

- Make the causal success simulation results reproducible and trusted before
  running larger experiments
  - One `config` controls every parameter of an experiment
  - Every important result is frozen by a unit test
- Background:
  - Issue [#707](https://github.com/gpsaggese/gpsaggese.github.io/issues/707)
    and its discussion
  - The paper:
    `papers/Causal_Analysis_of_Agent_Skill_And_Luck/03_experimental_evaluation.tex`

## Plan Hierarchy

- This file: what to build and in which order
- `07-progress.md`: what is done and learned
- `03-decisions.md`: why it is built this way
- `- [ ]` is an unmerged PR and `- [x]` is a merged PR

## Roadmap

- `v0.1`: the baseline experiment (100 agents, 80 periods, seed 42) runs from one
  config, and the example notebook reports only numbers it computes
  - [x] `PR_C1`
  - [ ] `PR_E1`
- Releases are cumulative: a version also needs the PRs of the versions before it

# Component Specs

## Example Notebook

### Goal

- `causal_success_example.ipynb` runs the full analysis on one baseline result:
  - Inequality, associations, causal estimates, and policy comparison
  - Every interpretation follows from the printed output

### Open Questions

1. A longer-run experiment, labeled separately from the baseline

# PRs

## v0.1: Baseline Experiment

### `PR_C1`: [x] One config for the simulation

- Merged as #938 (Niveda): `get_default_config()`, `run_experiment()`, and the
  tests that freeze the baseline

### `PR_E1`: [ ] Example notebook on the shared simulator

- Background: before this rewrite, `causal_success_example.py`
  - Mixed a 200-period run with the 80-period baseline
  - Used `log(capital + 1)` and the spread of fitted effects as uncertainty
  - Had its own `sqrt(allocation)` policy model and reversed or unsupported
    interpretations
- Changes:
  - New `causal_success_analysis_utils.py`: analysis and plotting helpers
  - Notebook consumes `get_default_config()` and `run_experiment()` only
  - Notebook follows the current `notebook.rules.md` and `notebook.template.py`
- Unit tests:
  - `test/test_causal_success_analysis_utils.py`: fast tests on small runs, and slow
    tests that freeze the baseline results
  - The tests do not depend on PyMC
- Result: every number in the notebook comes from one config and is unit tested
- Implementation and validation checklist:
  - [x] Add `causal_success_analysis_utils.py` and its fast tests
  - [x] Rewrite the notebook on the baseline result
    - [x] Setup cells, config, and baseline
    - [x] Inequality, associations, and top performers
    - [x] DML and causal forest on $\log(\text{capital})$, with estimator intervals
    - [x] Policy comparison through `run_experiment()`
    - [x] Comparison with the paper: single seed and average over seeds
    - [x] Conclusions written from the outputs
  - [x] Add slow tests that freeze the notebook results
  - [x] Add the new test file to `test/test_docker_all.py`
  - [x] Sync `.py` and `.ipynb` with Jupytext
  - [x] Verify in the project Docker image: unit tests, the notebook top to
    bottom, the paired script, and the pinned pre-commit hooks
  - [x] Run `test/test_docker_all.py` (on 2026-11-16, before the later
    cleanups)
  - [x] Record the results and the limitations in `07-progress.md`
  - [ ] Review
- Status: implemented and verified, pending publication and review; there is
  no commit or PR yet, so `PR_E1` stays unchecked until it is merged
- Known gap: the official `i lint` could not run, since it needs private ECR
  access

## Conventions

- Coding rules: `.claude/skills/coding.rules.md`
- Testing rules: `.claude/skills/testing.rules.md`
- Notebook rules: `.claude/skills/notebook.rules.md` (current upstream version)
