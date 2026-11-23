# Causal Analysis of Agent Skill and Luck Progress

## Research notes

[Shared research Gdoc](
https://docs.google.com/document/d/1jxGArIb2KFJ0aZXc36U60F4Cnwx6-9KfQWhIkFOrS2I/edit
)

## Status

| Milestone                 | Target date | Status                     | PR                                                                |
| :------------------------ | :---------- | :------------------------- | :---------------------------------------------------------------- |
| `PR_C1`: config and tests | 2026-11-14  | merged                     | [#938](https://github.com/gpsaggese/gpsaggese.github.io/pull/938) |
| `PR_E1`: example notebook | open        | implemented locally, no PR | none                                                              |

## Built

- `causal_success_utils.py`: the simulation and its config (`PR_C1`)
- `causal_success_analysis_utils.py`: the analysis and plotting helpers of
  `causal_success_example.ipynb` (`PR_E1`), see
  [`D1`](03-decisions.md#d1-the-analysis-helpers-live-in-their-own-module)

## Findings

- All the numbers are from `causal_success_example.ipynb`, executed in Docker on
  2026-11-15 and re-executed on 2026-11-21 and 2026-11-23 with the same values,
  with `get_default_config()` (100 agents, 80 periods, seed 42)

1. Baseline inequality: Gini 0.217, top 10% share 18.4%, bottom 50% share 34.5%,
   mean capital 0.888, median 0.816 (Cell 1.2)
2. Associations with log-capital: talent norm r = -0.045, 95% CI [-0.239,
   0.153]; realized beneficial events r = 0.540, 95% CI [0.384, 0.666] (Cell 3.2)
3. Top 10 agents by capital: mean talent rank 42.6 against 50.5 for the
   population, where rank 1 is the most talented (Cell 3.5)
4. Observed average event-level log gain: 0.199 over 232 realized beneficial
   events (Cell 4.2)
5. DML effect of one realized beneficial event on log-capital: 0.157, standard
   error 0.026, 95% CI [0.106, 0.208] (Cell 4.3)
   - The interval contains 0.199, which is consistent with the estimate but
     does not validate it or its identification assumptions
6. Causal forest: average effect 0.159, 95% CI [0.094, 0.224]; the per-agent
   effects vary little relative to their uncertainty: std 0.013 against a mean
   interval width of 0.126 (Cell 4.4)
   - This is descriptive, not a formal test of heterogeneity
7. Policies: every policy run has the same events as the baseline, and no
   agent reaches the capital floor, so a policy only rescales each agent's
   multiplier; `performance` equals `egalitarian` (Cells 5.2 and 5.3)
   - With a binding floor the rescaling fails: e.g., 10 agents with a floor of
     0.5 would imply a total starting capital of 53.1 instead of 110
8. Over seeds 0 to 49: Gini 0.276 (std 0.027, max 0.334), mean realized
   beneficial events 2.17, r(lucky events, log-capital) 0.652, against 0.38,
   4.8, and 0.78 in the paper; seed 42 has the lowest Gini (Cells 6.1 and 6.2)

## Modeling Limitations

- The causal estimates come from one seed (42)
- The spillover rule of the code differs from the paper, and the notebook
  describes the code, see
  [`D3`](03-decisions.md#d3-the-simulator-behavior-is-described-not-changed)
- The policy comparison is exploratory and in-sample, see
  [`D5`](03-decisions.md#d5-the-policy-comparison-is-exploratory)
- The policy plot was dropped to keep the notebook below 500 KB: the tables of
  Cells 5.2 and 5.3 show the same values

## Environment and Reproduction

- Reference environment: the project `Dockerfile`, built as
  `gpsaggese/umd_causal_success_analysis:umdtask707` (Python 3.12.3, numpy
  1.26.4, pandas 3.0.6, scipy 1.17.1, scikit-learn 1.6.1, econml 0.16.0,
  matplotlib 3.9.4, pytest 9.1.1)
- Commands, in the container, with the mount and the environment of
  `docker_cmd.sh` (repo at `/git_root`):
  ```bash
  docker> cd /git_root && python -m pytest -o cache_dir=/tmp/pytest_cache \
      research/Causal_Analysis_of_Agent_Skill_And_Luck/test/test_agent_benchmark_utils.py \
      research/Causal_Analysis_of_Agent_Skill_And_Luck/test/test_causal_success_utils.py \
      research/Causal_Analysis_of_Agent_Skill_And_Luck/test/test_causal_success_analysis_utils.py
  docker> cd /git_root/research/Causal_Analysis_of_Agent_Skill_And_Luck
  docker> jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=-1 causal_success_example.ipynb
  docker> python causal_success_example.py
  ```
- Pair the notebook with the Jupytext of the pinned pre-commit hook (1.19.0):
  the image has Jupytext 1.19.6, whose header the pinned hook rejects
- Run the pinned hooks with
  `pre-commit run -c helpers_root/.pre-commit-config.yaml --files <files>`
- `test/test_docker_all.py` runs `docker_build.sh`, which rewrites the tracked
  `research/docker_build.log`: do not commit that file

## Next

1. Review of `PR_E1`
2. Publish `PR_E1`: commit, push, and open the PR
3. Later, as a separate, labeled task: plan the longer-run experiment

## History

| Date       | Done                                                                    |
| :--------- | :---------------------------------------------------------------------- |
| 2026-11-14 | `PR_C1` merged as #938                                                  |
| 2026-11-15 | `PR_E1` implemented and verified locally; decisions `D1` to `D6` taken  |
| 2026-11-15 | Review fixes of `PR_E1` applied and re-verified; decision `D7` taken    |
| 2026-11-15 | Starting capital below the floor handled; decision `D8` taken           |
| 2026-11-16 | Final notebook wording and docs corrections of `PR_E1`                  |
| 2026-11-16 | Isolated Linux verification; changes moved onto `ed7e50c`               |
| 2026-11-21 | PR-readiness cleanup: Cell 1.3 wording, source links, test helper       |
| 2026-11-22 | Pinned pre-commit hooks pass on the staged files                        |
| 2026-11-23 | Logging, constants, notebook math, and docs cleanup; decision `D9`      |

## Open Work

- `PR_E1`: implemented and verified on
  `UmdTask707_CAASL_Rewrite_example_notebook`, on top of `ed7e50c`; no commit
  or PR exists yet, pending review and publication

## 2026-11-15: `PR_E1` Example notebook on the shared simulator

- **Issue**: [#707](https://github.com/gpsaggese/gpsaggese.github.io/issues/707)
- **What was done**:
  - Added `causal_success_analysis_utils.py` and
    `test/test_causal_success_analysis_utils.py`
  - Rewrote `causal_success_example.py` and `.ipynb` on one baseline config,
    following the current `notebook.rules.md` and `notebook.template.py`
  - Added the new test file to `test/test_docker_all.py`
  - Removed the 200-period experiment, `log(capital + 1)`, the
    `sqrt(allocation)` policy model, and the broken links
- **Testing**: the unit tests and the notebook ran in the reference image, see
  [Environment and Reproduction](#environment-and-reproduction); the analysis
  tests also pass with `pymc` made unimportable
- **Note**: written with Claude Code; the author reviews every change before
  the PR

## 2026-11-15: `PR_E1` Review fixes

- **Issue**: [#707](https://github.com/gpsaggese/gpsaggese.github.io/issues/707)
- **What was done**:
  - `cell5_3_compute_implied_initial_capital()` counts the agents at the
    capital floor and returns NaN when the rescaling does not hold, see
    [`D7`](03-decisions.md#d7-the-starting-capital-is-inferred-only-when-the-floor-never-binds)
    - `cell5_2_run_policy_comparison()` returns the full results of each run,
      so that the capital histories can be checked
  - `cell4_2_compute_log_change_per_event()` returns an `unchanged` row and NaN
    statistics for a kind with no event, instead of raising `ValueError`
  - The notebook describes the causal forest as descriptive, calls 0.199 the
    observed average event-level log gain, qualifies the rescaling with the
    floor, and calls the talent spread a discrepancy inside the paper
- **Testing**: new tests for a binding floor (`Test_cell5_3...test2`), no
  beneficial event (`Test_cell4_2...test3`), and an unchanged event at the
  floor (`Test_cell4_2...test4`)

## 2026-11-15: `PR_E1` Starting capital below the floor

- **Issue**: [#707](https://github.com/gpsaggese/gpsaggese.github.io/issues/707)
- **What was done**:
  - `cell5_2_run_policy_comparison()` and
    `cell5_3_compute_implied_initial_capital()` use each agent's actual
    starting capital, see
    [`D8`](03-decisions.md#d8-the-policy-helpers-use-the-actual-starting-capital)
    - `cell5_3_compute_implied_initial_capital()` no longer takes
      `initial_capital`, and reports `sum_actual_initial`
  - Before: with 10 agents, `initial_capital=1`, `capital_floor=2`, and no
    events, the totals were 10 and 110 and the inferred total was 60, while the
    agents started with 20 and 120
- **Testing**: new tests for that config (`Test_cell5_2...test3`,
  `Test_cell5_3...test3`); the baseline values are unchanged

## 2026-11-16: `PR_E1` Final notebook and docs corrections

- **Issue**: [#707](https://github.com/gpsaggese/gpsaggese.github.io/issues/707)
- **What was done**:
  - Cells 5.2 and 5.3 separate the run without allocation (total 100) from
    the funded policies (total 200)
  - Cell 3.4 defines talent rank 1 and capital rank 1 separately
  - Cell 7.1 says that the seed-42 run does not show a statistically
    distinguishable linear association between talent norm and log-capital
  - Cell 5.2 shows the source of `cell5_2_run_policy_comparison()` and Cell 5.3
    the source of `cell5_3_compute_implied_initial_capital()`
  - `cell6_1_run_seed_sweep()` no longer shows a progress bar, see
    [`D9`](03-decisions.md#d9-the-seed-sweep-has-no-progress-bar)
- **Testing**: the notebook and the paired script run; every table and number
  is unchanged

## 2026-11-16: `PR_E1` Isolated Linux verification

- **Issue**: [#707](https://github.com/gpsaggese/gpsaggese.github.io/issues/707)
- **What was done**:
  - Applied the changes to a disposable Linux clone of
    `UmdTask537_Review_students_research_projects` at `ed7e50c`, the squash
    merge of #938, which has the same tree as the previous base `ffd97f4`
  - Moved the changes onto `UmdTask707_CAASL_Rewrite_example_notebook` at
    `ed7e50c`, so that the original commits of `PR_C1` are not replayed
  - Synchronized the notebook pair with the pinned Jupytext 1.19.0
- **Testing**:
  - `test/test_docker_all.py`: 8 passed (`test_docker_build`,
    `test_docker_cmd`, `test_docker_bash`, and `test1` to `test5`), in a private
    Docker-in-Docker daemon, so that `docker_build.sh` does not overwrite the
    image tags of the host
    - This is the last run of the harness: it predates the changes of
      2026-11-21 and 2026-11-23
  - The 3 test files: 135 passed; the paired script: exit code 0
- **Limitations**:
  - The official `i lint` could not run: it needs `CSFY_ECR_BASE_PATH` and the
    private ECR image `helpers:prod`, and that access is not available
  - The Markdown formatter of `linters2/lint.py` re-wraps and retitles the
    docs, upstream `skills_usage.md` included, so it was not applied

## 2026-11-21: `PR_E1` PR-readiness cleanup

- **Issue**: [#707](https://github.com/gpsaggese/gpsaggese.github.io/issues/707)
- **What was done**:
  - Cell 1.3 no longer attributes the talent standard deviations below 0.15 to
    clipping
    - With seed 42, none of the 300 draws of `create_population()` falls
      outside [0, 1]
    - The sample standard deviations are 0.135, 0.147, and 0.138
  - Re-executed the notebook on the new branch, so that the 9 source links of
    `hintros.print_obj_info()` point to it
  - `_get_figure_titles()` in the tests reads the figure title with the public
    `Figure.get_suptitle()` of Matplotlib 3.9.4, instead of `fig._suptitle`
- **Testing**: the analysis tests (29, 7 slow) and the 3 test files (135)
  pass; every notebook output is unchanged, except the branch of the source
  links

## 2026-11-22: `PR_E1` Pre-commit record

- **Issue**: [#707](https://github.com/gpsaggese/gpsaggese.github.io/issues/707)
- **Testing**: the pinned hooks of `helpers_root/.pre-commit-config.yaml`
  (`pre-commit-hooks` v6.0.0, `ssort` 0.16.0, `ruff` v0.14.14, `jupytext`
  v1.19.0bis) pass on the exact staged files, with no automatic edit

## 2026-11-23: `PR_E1` Logging, constants, and docs cleanup

- **Issue**: [#707](https://github.com/gpsaggese/gpsaggese.github.io/issues/707)
- **PR**: none yet, and no commit exists
- **What was done**:
  - `causal_success_analysis_utils.py`:
    - Every function logs its inputs on entry, and every function that returns
      a value logs a summary of it (shape, count, or keys)
    - The single-use constants of the policy comparison and of the paper
      values moved into `cell5_2_run_policy_comparison()` and
      `cell6_2_compare_with_paper()`; `_TALENT_COLS` stays shared
  - The notebook markdown renders its math with LaTeX: $r$ in Cell 3.2, $M_i$
    in Cell 5.3, and $N(0.5, 0.15)$ on $[0, 1]$ in Cell 6.4
  - Condensed this file, rewrote the status of `04-plan.md`, and added
    [`D9`](03-decisions.md#d9-the-seed-sweep-has-no-progress-bar)
- **Testing**, in the reference image:
  - `test_causal_success_analysis_utils.py`: 29 passed, the 7 slow tests
    included; the 3 test files: 135 passed, 0 skipped, 0 failed
  - The notebook runs top to bottom from a fresh kernel: valid nbformat 4.5,
    34 code cells in order, no error output, ASCII only, 492,014 bytes
    - Every output and figure is unchanged, except the line anchors of the 9
      source links and a temporary file name in the log of the setup cell
    - The 9 source links point to the `def` line of each function on this
      branch
  - The paired script runs: exit code 0
  - The pinned Jupytext 1.19.0: a second `--sync` changes nothing
  - The pinned pre-commit hooks pass on the 9 files
- **Limitations**:
  - `test/test_docker_all.py` was not rerun since 2026-11-16
  - The official `i lint` could not run; the pinned pre-commit hooks were run,
    but they do not replace it
  - The 9 source links resolve on GitHub only after the branch is pushed
  - The setup cell prints a `TqdmWarning` (`IProgress not found`), since
    `ipywidgets` is not in the image
