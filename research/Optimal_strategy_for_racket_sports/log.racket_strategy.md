# Racket Strategy Project Log

- What was done in `research/Optimal_strategy_for_racket_sports/`, newest entry
  first, with links to the issues and PRs
- Parent issue: [#590](https://github.com/gpsaggese/gpsaggese.github.io/issues/590)
  (Implement solution for optimal strategy for racket sports)
- The milestones are defined by GP in a
  [comment on #590](https://github.com/gpsaggese/gpsaggese.github.io/issues/590#issuecomment-6075690143)
  (2026-10-09):
  - **Milestone 1**: a notebook with the tennis / pickleball court where player 1
    serves to (`x1`, `y1`), player 2 moves there and returns toward (`x2`, `y2`)
    with a 2D Gaussian spread (`std_x2`, `std_y2`), and the probability that the
    return is in
  - **Milestone 2**: the probability that player 1 reaches the return, given a move
    speed, and the probability of a winner from player 2
  - In a later
    [comment](https://github.com/gpsaggese/gpsaggese.github.io/issues/590#issuecomment-6128247188)
    of 2026-10-12, GP asked to do the heatmap / next milestone next (#624); the
    design was agreed in
    [#690](https://github.com/gpsaggese/gpsaggese.github.io/issues/690)
- Dates are in UTC, as GitHub reports them
- The plan of GP's initial implementation is `plan.racket_strategy.md`; its one
  spec file per planned PR was deleted in #691, once PR1 to PR7 were done

## Open Work

- The open tasks and the next steps are in the Tasks section of
  `plan.racket_strategy.md`
- Manual widget checks of the Milestone 1 notebook, noted as in progress in #654
- The open research questions are in the Gdoc, under
  [Questions for GP](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit?tab=t.a8h8vb8v4lq8#heading=h.xfydadrg3lvc)

## 2026-11-10: Research Questions and Results in the Gdoc (#694)

- **Issue**: [#694](https://github.com/gpsaggese/gpsaggese.github.io/issues/694),
  from GP's rules for the research process
  ([comment](https://github.com/gpsaggese/gpsaggese.github.io/issues/694#issuecomment-6712939277),
  2026-11-07) and his follow-up: "pls put all the questions / content in the gdoc and
  then you can post links from GH to there"
- **Gdoc**: the research content that lived only in GitHub comments moved there
  - [Experiment 4](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit?tab=t.0#heading=h.xvdn4n7zo860):
    the speed ratio and the reaction time (#789); its 3 figures are from a full run
    of Part 3 of notebook 05 on master `80efe9d9`, which gave the same numbers as the
    notebook
  - [Experiment 5](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit?tab=t.0#heading=h.qrto9so35bnx):
    the plan of #723, part 2, with the decisions of 2026-11-03 and 2026-11-06
  - [Experiment 6](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit?tab=t.0#heading=h.kntokyu5w697):
    the racket RL env of #713, layer 3 (#881), with the figures saved in notebook 06
  - [Questions for GP](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit?tab=t.a8h8vb8v4lq8#heading=h.xfydadrg3lvc):
    12 questions, one comment each with our recommendation, linked with one line from
    #694, #713, #723, #789, #846, #857, and #881
  - The `inspect.getsource()` question of #846 stays on GitHub, since it is about
    code
- **Docs**:
  - `project_management.explanation.md`: the research Gdoc, GP's rules, the life of a
    research question, and the paper as the way to review a research PR, with a Paper
    and Gdoc checklist for its description
  - `design_principles.explanation.md`: P13 rewritten for the Gdoc (it replaced the
    GitHub rule on 2026-11-07), P12 updated, and links to the Gdoc in P4, P10, P14,
    and P16
  - `plan.racket_strategy.md`: links to the Gdoc, and the status of #823, #857, and
    #881, all merged
- **Decided**
  ([GP on #713](https://github.com/gpsaggese/gpsaggese.github.io/issues/713#issuecomment-6751589012),
  2026-11-09, recorded under Experiment 6): the layer 3 success criterion is good
  enough (Q7), and PPO stays on the full grid, with no pruning for now (Q8)
- **Pending GP**
  ([Research process](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit?tab=t.a8h8vb8v4lq8#heading=h.21d3arlkot6c)):
  Q1, one log or two; Q2, the Drive folder of the research Gdocs; Q3, how the general
  rules reach the research guide of helpers; the paper checklist lives in
  `project_management.explanation.md` for now

## 2026-11-07: Racket RL Env and Notebook 06 (#713, Layer 3)

- **Issue**: [#713](https://github.com/gpsaggese/gpsaggese.github.io/issues/713),
  layer 3, after the Gymnasium tutorial (layer 1)
  - GP's decisions on #713: brute force first, with the target learned by RL;
    reuse the Milestone 2 code but not its results; a reward of 1 for a winner; a
    discrete target grid with a programmable step
- **What was done**:
  - `notebooks/racket_sports_06_rl_return_utils.py`: `ReturnEnv`, a one-shot
    Gymnasium env (a contextual bandit) whose step is one call of
    `compute_winner_probability()` with 1 sample; the brute force through the env;
    the comparison with a fresh Milestone 2 map; the pass rule
  - `notebooks/racket_sports_06_rl_return.ipynb`: the env, the brute force,
    Milestone 2 recomputed on the same grid, PPO (`stable-baselines3`) on
    rescaled observations, and the pass rule; the saved outputs are from the full
    run
  - Tests: 14 unit tests, with 3 planted bugs caught, and a notebook test of about
    10 minutes
- **Findings**:
  - SB3 feeds the raw meters to the network, and the constant input of player 1
    at $y = -11.885$ m saturated a quarter to a third of the first layer;
    rescaling the observation to $[-1, 1]$ removed the slow start
  - PPO drifts after about 100k steps on a reward that is almost always 1 (tennis,
    2 of 3 seeds), so the full run uses 75k steps
  - Full run, 9 serves per sport, 1000 points per target, fresh seeds 3, 4, 5:
    - The brute force matches Milestone 2: Spearman 0.98 in tennis and 0.99 in
      pickleball, and each best cell is in the other's region on 15 of 18 serves;
      the 3 misses are ties within sampling noise (0.2 points)
    - The expected regret of PPO is under 0.8 points on all 6 runs, against 11 to
      21 points for a random target
    - The strict rule, every greedy cell in the Milestone 2 region, written before
      the run, passed 4 of 6 runs; seed 5 failed in both sports, since on the flat
      tennis plateau the policy spreads over tied cells and its top cell is fragile
- **Open questions for GP**: which success metric matters for the next step, the
  greedy cell or the expected regret; and whether to try a fix for the late drift,
  tested again with a pass rule written in advance on fresh seeds

## 2026-11-04: Design Principles

- **Issue**: [#723](https://github.com/gpsaggese/gpsaggese.github.io/issues/723)
  (model the spread as a function of the ball speed), part 1, from
  [GP's answer](https://github.com/gpsaggese/gpsaggese.github.io/issues/723#issuecomment-6602349966)
  of 2026-11-03 to the proposal to scale the spread with the ball speed
- **What was done**: `design_principles.explanation.md`, 15 principles, each with
  why, an example from this project, and its source
  - P1 to P3 are GP's from #723: independent variables with separate coupling
    models, notebooks for intuition and code for sweeps, and sweeping to find what
    is robust to what
  - P4 to P15 are the principles we already follow, each checked against the plan,
    this log, the paper, and the GitHub discussions, with a link to where it was
    decided
- **Name**: `.explanation.md`, as the file name convention of
  `all.writing_docs.how_to_guide.md` asks and as `project_management.explanation.md`
  does, instead of the `design_principles.md` of GP's comment

## 2026-11-03: Remove Repetition in the Code and the Tests

- **Issue**: [#823](https://github.com/gpsaggese/gpsaggese.github.io/issues/823)
  (remove redundancy in code and tests, from GP's review of #804), with
  [the plan](https://github.com/gpsaggese/gpsaggese.github.io/issues/823#issuecomment-6591281381)
- **What was done**:
  - Code, with no behavior change: `_show_figure()` in the notebook 01 utils
    ends the redraws of Cell 1.1 and Cell 2.1, and `_build_returner_controls()`
    builds the returner sliders of both widgets of `racket_strategy_utils.py`
  - Tests: each failing check is tested once, in the test of the function that
    owns it; 14 tests that repeated another test are removed; copied setup is
    in shared helpers; the "Tests cover" lists and the `# Expected:` comments
    that repeated the asserts are removed
- **Numbers**: 244 tests to 230; test lines 7,092 to 6,558; source lines 5,230
  to 5,256, since the 2 new helpers replace 2 copied blocks
- **Testing**:
  - Full suite: only the 5 known Docker failures, before and after
  - The same 52 source lines raise a failing check in some test, before and
    after; 13 of them were tested from 2 or 3 test classes, now none
  - Source line coverage: 98.86% before and after, with the same lines missed
  - Notebooks 01 to 05 run top to bottom, 05 in reduced mode
  - The 12 Cell 1.1 and 6 Cell 2.1 widget images are byte identical to master
- **Possible follow-up**: the notebook 05 tests check their outputs piece by
  piece; the "compare whole output" rule of `testing.rules.md` would rewrite
  about 60 asserts without removing any repetition

## 2026-11-01: PRs #804, #809, and #812 Merged

- [PR #804](https://github.com/gpsaggese/gpsaggese.github.io/pull/804), merged
  2026-10-31: red to green heatmaps
  ([#788](https://github.com/gpsaggese/gpsaggese.github.io/issues/788))
- [PR #809](https://github.com/gpsaggese/gpsaggese.github.io/pull/809), merged
  2026-10-31: docs cleanup, notebook names, and plan phases
  ([#691](https://github.com/gpsaggese/gpsaggese.github.io/issues/691) and
  [#787](https://github.com/gpsaggese/gpsaggese.github.io/issues/787)), see the
  2026-10-30 entry
- [PR #812](https://github.com/gpsaggese/gpsaggese.github.io/pull/812), merged
  2026-11-01: notebook 05, variable sensitivity
  ([#789](https://github.com/gpsaggese/gpsaggese.github.io/issues/789), with the
  [#694](https://github.com/gpsaggese/gpsaggese.github.io/issues/694) corner
  sims), see the 2026-10-29 and 2026-10-27 entries
- All 3 PRs are by Aahan Rembersu; the plan marks #691, #787, #788, and #789 as
  done

## 2026-10-30: Docs Cleanup, Notebook Names, and Plan Phases

- **Issues**: [#691](https://github.com/gpsaggese/gpsaggese.github.io/issues/691)
  (clean up the docs, with
  [GP's decisions](https://github.com/gpsaggese/gpsaggese.github.io/issues/691#issuecomment-6309187106))
  and [#787](https://github.com/gpsaggese/gpsaggese.github.io/issues/787) (the 3
  phases in the plan), in one PR
- **What was done**:
  - Deleted `spec.racket_strategy_PR1.md` to `spec.racket_strategy_PR7.md`; the
    plan marks PR1 to PR7 as done, with PR4 and PR7 closed without code, since we
    focus on the 2D model and set the trajectory work aside
  - Renamed the notebooks with a number in their logical order, using
    underscores as `notebook.rules.md` asks, with `git mv`:
    - `racket_shot_probability` (and its utils and test) to
      `racket_sports_01_shot_probability`
    - `racket_strategy.API` to `racket_sports_02_strategy_API`
    - `racket_strategy.exploration` to `racket_sports_03_strategy_exploration`
    - `racket_trajectory.API` (and `racket_trajectory_utils.py`) to
      `racket_sports_04_trajectory_API`
    - `racket_strategy_utils.py` keeps its name, since notebooks 02 and 03 share
      it
  - Plan: the 3 phases of #714 with the open question on the game theory part,
    the #789 finding on the speed ratio, the next steps as tasks, and the task
    statuses
- **Testing**: the full suite has only the 5 known Docker failures, every
  renamed notebook runs top to bottom, and no old name is left outside this log

## 2026-10-29: Speed Ratio and Reaction Time Checks

- **Issue**: [#789](https://github.com/gpsaggese/gpsaggese.github.io/issues/789) (can the two speeds be reduced to one
  control?), with [GP's decision](https://github.com/gpsaggese/gpsaggese.github.io/issues/789#issuecomment-6528212968) of 2026-10-30
- **Notebook**: `notebooks/racket_sports_05_variable_sensitivity.ipynb`, Part 3;
  the checks ran as a local script and moved to the notebook on 2026-10-30
- **Gdoc**: [research results](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit), the #789 entry of the Project section
- **What was run**:
  - Both speeds multiplied by $k$ = 0.5, 2, 4 at $t_r = 0$, same seed, on 5
    settings per sport: 30 pairs of maps compared cell by cell
  - 500 random settings at $t_r = 0$, with a random $k$ from 0.101 to 9.883
  - A grid of 216 maps: 12 ratios $r = v_p / v_b$ x 3 values of $v_p$ x 3 values of
    $t_r$ x 2 sports, for one curve per ratio and for the reaction time comparison
- **Key numbers**:
  - With $t_r = 0$ only the ratio matters: the largest cell difference is 0 on the
    30 pairs and on the 500 random settings, and player 1 reaches more than 1% of
    the landings in 401 of those
  - With $t_r = 0.2$ s the 3 speed pairs of a ratio split, by up to 12.62 points on
    the highest $P(\text{winner})$, since a second quantity enters, the reaction
    distance $v_p t_r$
  - $t_r$ of 0.2 s against 0, at $v_p = 1.5$ m/s, within the ball speeds of the
    widget:

| Largest change | Tennis | Pickleball |
|---|---|---|
| Highest $P(\text{winner})$ (points) | 0.00 | 0.82 |
| Mean $P(\text{winner})$ over the half (points) | 2.34 | 3.95 |
| Best region centroid (m) | 0.26 | 0.45 |
| Highest $P(\text{winner})$ at $v_p = 6$ m/s (points) | 2.40 | 17.52 |

- **GP's decision**: keep the model with all the variables, with $v_b$, $v_p$, and
  $t_r$ as separate controls, no ratio slider, and $v_p$ = 1.5 m/s by default; state
  the ratio result in the paper, with the proof sketch in an appendix
- **Paper**: Section VII-A and the appendix

## 2026-10-27: Corner Sims

- **Issue**: [#694](https://github.com/gpsaggese/gpsaggese.github.io/issues/694), with
  [GP's plan](https://github.com/gpsaggese/gpsaggese.github.io/issues/694#issuecomment-6336301932): the corners of the settings,
  then one setting at a time
- **Notebook**: `notebooks/racket_sports_05_variable_sensitivity.ipynb`, Parts 1 and
  2; the sims ran as a local script and moved to the notebook on 2026-10-30
- **Gdoc**: [research results](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit), Experiment 3
- **What was run**: 952 maps, with seed 1, the default grid, and 5000 samples per
  target
  - 324 corner runs: 9 serves x 3 ball speeds x 3 values of $v_p$ x 2 sides x 2
    sports
  - 178 sweeps: ball speed, $v_p$, $t_r$, `std_x2`, and `std_y2`, one at a time,
    around the defaults and around $v_p$ = 6 m/s
  - 450 serve map runs: a 0.5 m grid of legal deuce serves at 3 settings
- **Key numbers**, as the highest cell of each map (`max_p_winner`):
  - The error spread matters most: with a spread of 3 m the highest
    $P(\text{winner})$ falls to 0.827 (tennis) and 0.695 (pickleball) for `std_x2`,
    and to 0.955 and 0.727 for `std_y2`
  - Around the defaults the speeds move it little: 1.000 in tennis; in pickleball
    0.948 to 0.998 with $v_p$, and 0.979 to 0.998 with the ball speed
  - The corners mostly do not matter: at least 0.99 in 146 of 162 tennis runs and
    108 of 162 pickleball runs
  - The exception is a slow ball (5 m/s) with a fast player 1 (6 m/s): 0.816 to
    0.991 in tennis, and 0.047 to 0.705 in pickleball, depending on the serve
  - There deep serves help player 1: in pickleball the short serves leave 0.680 to
    0.705 and the deep ones 0.047 to 0.092, and the deep serve at the T is the
    lowest in both sports (on the tennis ad side, tied with the other deep serves)
  - Outside that corner, pickleball still goes down to 0.904 at $v_p$ = 6 m/s with
    the default ball
  - The mean over the half moves more: from 0.51 to 0.83 with $v_p$ in pickleball
- **Limit of the metric**: the highest $P(\text{winner})$ is 1.000 for many tennis
  settings, so it cannot rank them
- **Paper**: Section VII-A, Figures 3 and 4

## 2026-10-23: Milestone 2, Probability of a Winning Return

- **Issue**: [#624](https://github.com/gpsaggese/gpsaggese.github.io/issues/624),
  design agreed in [#690](https://github.com/gpsaggese/gpsaggese.github.io/issues/690)
- **PR**: [#715](https://github.com/gpsaggese/gpsaggese.github.io/pull/715), by
  Aahan Rembersu, opened 2026-10-21 as a draft, not merged yet
- **Milestone**: Milestone 2
- **What was done**:
  - New functions in `notebooks/racket_shot_probability_utils.py`, one step at a
    time:
    - Player 1's reach: `get_player1_xy()`, `compute_flight_time()`,
      `compute_is_reached()`
    - One target: `compute_winner_probability()`, the joint event "in and not
      reached" by Monte Carlo, with its standard error
    - The map: `compute_winner_heatmap()` with common random numbers,
      `find_best_region()`, `find_best_target()`, and `plot_winner_heatmap()` on
      a fixed 0 to 1 color scale
    - The widget: `cell2_1_plot_winner_widget()`, with a settings string in the
      comments panel and as a copyable line under the figure
  - A new Part 2 of `notebooks/racket_shot_probability.ipynb` (Cell 2.1), with
    the limitation of #723 in bold at its top
  - Cell 1.1 and Cell 2.1 share one helper for their common controls,
    `_build_shared_controls()`; no other existing function changed
  - The decisions are in the Milestone 2 section of `plan.racket_strategy.md`,
    and the model is in Section IV-H of the paper
- **Main finding**: with the defaults, the best return is the middle of player
  1's half, because player 1 covers only about 0.6 m (a return landing next to
  them flies about 15 m in 0.61 s at 25 m/s)
- **Testing**:
  - Full suite: 183 passed (110 before Milestone 2; 73 new tests); the only
    failures are the 5 tests of `test_docker_racket_strategy.py`, which fail on
    master too
  - Milestone 1 end-to-end check: the same results as master, and the 14 Cell
    1.1 images are byte identical to master
  - Milestone 2 end-to-end check (`e2e_milestone2.py`): all 140 checks pass; the
    notebook rules skip the Part 1 outputs saved on master (see Notebook
    outputs)
    - The widget's numbers equal the functions called directly, for 5 settings
      per sport and side
    - The fault decision equals `_check_serve()` on 168 serve points
    - A result rebuilt from the settings string alone is identical
    - Each control redraws once; slider extremes give no error
  - 47 planted bugs from the 4 phases, each caught by at least one test; the
    same outputs in 3 separate processes; the notebook runs top to bottom for
    both sports
  - In a fresh clone: the same results; linters (`ruff`, `ruff format`,
    `docformatter`, `ssort`, `mypy`) clean
- **Notebook outputs**: the committed notebook keeps the Part 1 outputs that GP
  saved in [#726](https://github.com/gpsaggese/gpsaggese.github.io/pull/726);
  the end-to-end checks skip the outputs already on master and check that the
  new cells have none
- **Limitations**: the spread does not depend on the ball speed, so hitting as
  fast as possible always looks best
  ([#723](https://github.com/gpsaggese/gpsaggese.github.io/issues/723)); the
  bounce is ignored as a 3D effect; player 1 stands at the center of the
  baseline with constant `t_r` and `v_p`
- **Note**: written with help from Claude Code; the author reviewed every change

## 2026-10-14: Fix Private Renames, Serve Direction, and Fault Label

- **Issue**: [#623](https://github.com/gpsaggese/gpsaggese.github.io/issues/623),
  closed by the PR
- **PR**: [#654](https://github.com/gpsaggese/gpsaggese.github.io/pull/654), by
  Aahan Rembersu, opened 2026-10-14, merged 2026-10-14
- **Milestone**: follow-up to Milestone 1
- **What was done**:
  - The 5 functions made private without updating their callers are public again:
    `solve_launch_speed()`, `get_flight_time()`, `get_height_at()`,
    `compute_reachability()`, `compute_score()`
    - Only the 5 definitions and their 7 calls changed
    - This fixed the 14 failing tests
  - `make_serve_situation()` now serves diagonally: the server stands at the box
    center mirrored across the center line
    - E.g., a tennis deuce serve starts at (2.06, -11.885) instead of
      (-2.06, -11.885)
    - The server stays on the baseline, so the library does not depend on the
      notebook's stance
    - The API notebook's Cell 7.1 marks the server, so the diagonal is visible
  - The Milestone 1 "Fault" label no longer covers the serve arrow: it goes beside
    the arrow, on the half where the serve did not land
    - Checked for every fault the sliders can reach, both sports and both sides:
      the closest gap is about 0.76 m
- **Testing**:
  - Full suite (without the Docker test file): 110 passed, 0 failed (before: 93
    passed, 14 failed)
  - 3 new tests for the "Fault" label; the serve tests check the exact server
    position
  - Both API notebooks, the exploration notebook, and the shot probability notebook
    run top to bottom with no errors; each `.py` / `.ipynb` pair is in sync
  - Milestone 1 end-to-end check: 64 passed, 0 failed
  - Linters: `ruff`, `ruff format`, `docformatter`, `ssort` clean; `mypy` reports 5
    `no-any-return` errors in the renamed functions, the same 5 as on master
- **CI note**: the failing `fast_tests.no_docker` check on #619 was not caused by
  these changes
  - It fails on `class_scripts/test/test_count_lecture_slides_uv_run.py`: the `uv`
    environment has only `tabulate`, but the import chain needs `tqdm`
  - Master fails the same way
  - The check runs only `book_springer`, `book_cs_refreshers`, `class_project`,
    and `class_scripts`, so none of the `research/` tests run in CI
- **Note**: written with help from Claude Code; the author reviewed every change

## 2026-10-12: Milestone 1, Serve and Return Probability Notebook

- **Issue**: [#590](https://github.com/gpsaggese/gpsaggese.github.io/issues/590)
- **PR**: [#619](https://github.com/gpsaggese/gpsaggese.github.io/pull/619), by
  Aahan Rembersu, opened 2026-10-12 as a draft, merged 2026-10-12
- **Milestone**: Milestone 1
- The sections below are the PR description, condensed

### Summary

- A notebook that shows the probability that a return lands in, given where it is
  aimed and how much it spreads:
  - Pick deuce or ad; player 1 serves from behind the baseline to (`x1`, `y1`),
    following tennis and pickleball serve rules
  - If the serve is legal, player 2 moves instantly to (`x1`, `y1`) and returns
    toward (`x2`, `y2`); if it is a fault, the notebook says why and there is no
    return
  - The return lands with a 2D Gaussian spread (`std_x2`, `std_y2`)
  - The notebook shows $P_{\mathrm{in}}$, computed exactly and checked by Monte
    Carlo
  - Every input is an `ipywidgets` control; it works for tennis and pickleball
- A standalone model, separate from the physics-based scoring in
  `racket_scoring.py`: the Gaussian sits directly on the landing point, so
  $P_{\mathrm{in}}$ has a closed form

### Model

- $X \sim N(x_2, \sigma_x^2)$ and $Y \sim N(y_2, \sigma_y^2)$, independent, with
  $\sigma_x$ = `std_x2` and $\sigma_y$ = `std_y2`
- In means landing in player 1's half: $|x| \le W/2$ and $-L/2 \le y \le 0$
- Independence plus a rectangular region gives a product of two 1D terms:
  $$
  P_{\mathrm{in}} =
  \left[\Phi\left(\frac{W/2 - x_2}{\sigma_x}\right)
  - \Phi\left(\frac{-W/2 - x_2}{\sigma_x}\right)\right]
  \cdot
  \left[\Phi\left(\frac{0 - y_2}{\sigma_y}\right)
  - \Phi\left(\frac{-L/2 - y_2}{\sigma_y}\right)\right]
  $$

### Serve Rules

- The side (deuce or ad) sets the server's stance: 0.5 m beside the center mark,
  0.3 m behind the baseline (a typical stance, not a rule value)
- Only the box diagonal from the server is legal; lines are in
- Faults are named: wide, into the wrong box, long, or in the kitchen (pickleball,
  including the kitchen line)
- The box stops at the baseline (see the pickleball note in Pre-existing Issues)

### Commits and Files

- 5 commits:
  1. Make `draw_court()` public again (it was renamed private but is called from
     other modules; this fixed 2 failing tests)
  2. Gaussian in-court probability utils and tests
  3. Shot probability widget and notebook
  4. Input validation and comment cleanup
  5. Serve follows tennis and pickleball rules
- Files:
  - `notebooks/racket_shot_probability_utils.py`: the model, the widget, and the
    drawing helpers
  - `notebooks/racket_shot_probability.ipynb` and `.py`: the notebook (Jupytext
    pair)
  - `test/test_racket_shot_probability_utils.py`: the tests
  - `racket_strategy_utils.py`: `draw_court()` rename only
  - `README.md`: one row in the notebooks table

### Testing

- New tests: 43 tests and 19 subtests pass, including random order runs
- Full suite: 93 passed; the 14 failures are pre-existing (see below)
- Independent math check: the closed form matches `scipy`'s bivariate normal CDF
  and 2D numerical integration within 1e-13 over 2,300 random cases
- Monte Carlo agrees with the closed form within chance over 4,000 cases
- Serve legality: 0 mismatches against an independently written rule over 859,452
  points (both sports, both sides)
- Widget stress: 1,235 slider states per sport with no errors or memory leaks,
  plus 200 random states per sport after the serve rule change
- Mutation check: 8 of 8 deliberately planted bugs are caught by the tests
- Bad inputs (`nan`, `inf`, std <= 0, `n_samples` <= 0) raise clear errors
- The notebook runs top to bottom for both sports; outputs are reproducible
- Final end-to-end run from a fresh clone: 98 checks, 0 failures

### Design Choices

- The standard error shown uses $P_{\mathrm{in}}$, since $\hat{P}$ gives 0 when
  every sample lands on one side
- Slider values are rounded to 6 decimals, so the "+" and "-" buttons land exactly
  on 0
- Ellipses are dark and dots are see-through, the reverse of the notebook styling
  rule, because light dotted ellipses disappear inside the dot cloud
- In 2D, the 1 sigma ellipse holds about 39% of the samples and 2 sigma about 86%

### Limitations

- The net is ignored for the return: a short return counts as out only if it lands
  on the wrong side
  - Net clearance could be added by reusing `racket_trajectory.py`
- The `x` and `y` errors are independent (no tilted ellipse)
- The serve point decides whether there is a return, but does not change
  $P_{\mathrm{in}}$ for a legal serve
- Singles width only

### Pre-existing Issues Found

- 5 functions were made private without updating their callers:
  `solve_launch_speed()`, `get_flight_time()`, `get_height_at()`,
  `compute_reachability()`, `compute_score()`
  - This caused the 14 failing tests and broke `plot_trajectory_fan()` and the two
    API notebooks
  - Fixed later in #654
- `racket_scoring.make_serve_situation()` placed a deuce server on the same side as
  the deuce box, so its serves went straight down the court instead of diagonally
  - `get_service_box_region()` itself is correct, and Milestone 1 uses the
    diagonal rule
  - Fixed later in #654
- Pickleball `service_line_m` is 6.71 m but the half length is 6.705 m; the
  Milestone 1 serve check caps the box at the baseline
- Pickleball net height and max ball speed differ from the paper's Table I
- Environment:
  - `pytest` is not in the Docker image
  - `git` reports "dubious ownership" inside the container
  - The partial clone remote format breaks `hgit.py`
  - Test runs leave root-owned files
  - `isort` and `ruff format` disagree on wrapped imports

### Questions and GP's Answers

- In
  [GP's comment](https://github.com/gpsaggese/gpsaggese.github.io/issues/590#issuecomment-6128247188)
  of 2026-10-12:
  - Fix the 5 private renames and `make_serve_situation()` in a separate PR? A
    quick PR is fine: done in #654
  - Next step, a $P_{\mathrm{in}}$ heatmap over every target or net clearance with
    `racket_trajectory.py`? The heatmap (#624); `racket_trajectory.py` is set
    aside for now, since the 3D problem adds complexity without adding much over
    2D
- **Note**: written with help from Claude Code; the author reviewed every change

## 2026-10-09: GP's Initial Implementation

- **Issue**: [#590](https://github.com/gpsaggese/gpsaggese.github.io/issues/590)
- **PR**: [#591](https://github.com/gpsaggese/gpsaggese.github.io/pull/591), by GP,
  opened 2026-09-23, merged 2026-10-09
- **Milestone**: none; the base that the milestones build on
- **What was done**: the PR description is only "#590", so this list comes from the
  PR's files and the status in `plan.racket_strategy.md`:
  - Project scaffold: Docker files from `class_project/project_template/`,
    `README.md`, `requirements.txt` (plan PR1)
  - `racket_params.py`: sport, player, and error parameters, court geometry, and
    the Table I presets `TENNIS` and `PICKLEBALL` (plan PR1)
  - `racket_trajectory.py`: closed-form ball flight, feasible launches, error
    sampling, and landing simulation (plan PR2)
  - `racket_scoring.py`: target grid, Monte Carlo launch table, reachability,
    composite score, and the serve situation (plan PR3)
  - `racket_strategy_utils.py`: court drawing, trajectory fan, heatmaps, and
    widgets (plan PR5, partly done)
  - Notebooks: `racket_strategy.API`, `racket_trajectory.API`, and
    `racket_strategy.exploration` (plan PR5 and PR6)
  - Tests for each module, plus a Docker test
  - The plan and the specs: `plan.racket_strategy.md` and
    `spec.racket_strategy_PR1.md` to `spec.racket_strategy_PR7.md`
- Still open in the plan: PR4 (placement game), PR7 (example notebook), and PR8
  (paper results)
