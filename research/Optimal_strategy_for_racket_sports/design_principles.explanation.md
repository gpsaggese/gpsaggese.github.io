# Design Principles of the Racket Sports Research

- The principles we follow when we build and study the model of this project
  - Each principle has one line, then why, then an example from this project, and
    the issue or comment where it was decided
  - The ids are stable: a new principle takes the next number, and an id is never
    reused
- GP asked for this doc on
  [#723](https://github.com/gpsaggese/gpsaggese.github.io/issues/723#issuecomment-6602349966)
  - Keep it updated: when a discussion gives a new principle, add it in a PR, with a
    link to that discussion: for a research discussion, its Gdoc heading or comment
    (P13)
- How we work day to day is in
  [`project_management.explanation.md`](project_management.explanation.md)
  - The plan is [`plan.racket_strategy.md`](plan.racket_strategy.md), and the log is
    [`log.racket_strategy.md`](log.racket_strategy.md)

## Modeling

### P1: Independent Variables

- **Principle**: all the variables of the model are independent; a coupling between
  them lives in a separate model that generates a set of values together
- **Why**:
  - Coupling the variables inside the model makes it messy
  - The couplings we know are guesses: for the spread and the ball speed, even the
    coefficient of a linear model is unknown
  - A separate model can be changed, swept, or turned off without changing the
    model or the results that do not use it
- **Example**: the spread of the return grows with the ball speed (#723)
  - The notebooks keep `std_x2`, `std_y2`, and the ball speed as 3 separate
    sliders
  - For a sweep, a builder computes `std_x2` and `std_y2` from the ball speed with
    one model, so the 2 spreads move together and the model does not change
- **Note**: independent means separate inputs, not statistically independent errors
  - The `x` and `y` errors of the return are also statistically independent
    (Milestone 1), but that is a separate assumption
- **Source**:
  [GP on #723](https://github.com/gpsaggese/gpsaggese.github.io/issues/723#issuecomment-6602349966),
  points 1 and 2; the unknown coefficient is from
  [GP on #690](https://github.com/gpsaggese/gpsaggese.github.io/issues/690#issuecomment-6336175829)

### P2: Notebooks for Intuition, Code for Sweeps

- **Principle**: the notebooks are for checking things visually and building
  intuition; the sweeps are done in code
- **Why**:
  - A widget shows one setting at a time: good for seeing a behavior, not for
    measuring it over hundreds of settings
  - A sweep in code can be tested, rerun, and saved for the paper
- **Example**:
  - The widgets of notebook 01 show one winner map at a time
  - The sweeps of #694 and #789 are functions in
    `racket_sports_05_variable_sensitivity_utils.py`, e.g., `run_all_sweeps()`,
    `run_corners()`, and `run_ratio_checks()`
  - Notebook 05 runs them in a reduced mode by default and shows the tables and the
    plots; its full runs write the CSV files of Figures 3 and 4 of the paper
- **Source**:
  [GP on #723](https://github.com/gpsaggese/gpsaggese.github.io/issues/723#issuecomment-6602349966),
  point 1; GP's must on
  [#590](https://github.com/gpsaggese/gpsaggese.github.io/issues/590#issuecomment-6075690143):
  all the code is in a `_utils.py` file, with a notebook to visualize it

### P3: Sweep to Find What Is Robust to What

- **Principle**: sweep the variables to find which results are robust to which
  variables, and investigate the results that are not robust
  - Capture the basic behaviors under reasonable assumptions, instead of sweeping
    every variable
- **Why**:
  - GP expected regions of the variable space where the values do not matter, and
    the corner sims found them
  - Sweeping everything costs many runs that say the same thing; the interesting
    part is where the answer changes
- **Example**: the #694 corner sims, then one variable at a time
  - Robust: the highest $P(\text{winner})$ is at least 0.99 in 146 of the 162
    tennis corner runs and in 108 of the 162 pickleball ones
  - Not robust: with a slow ball (5 m/s) and a fast player 1 (6 m/s) it is 0.047 to
    0.705 in pickleball, depending on the serve
  - Investigated: deep serves help player 1, since the return flies farther and
    player 1 has more time to run (paper, Sec. VII-A); the same corner led to #789
- **Source**:
  [GP on #723](https://github.com/gpsaggese/gpsaggese.github.io/issues/723#issuecomment-6602349966),
  point 3;
  [GP on #694](https://github.com/gpsaggese/gpsaggese.github.io/issues/694#issuecomment-6336301932):
  the corners first, then one variable at a time

### P4: Invariances Are Results, Not Controls

- **Principle**: when the results depend on a combination of variables, state it as
  a result, and keep the variables as separate controls
- **Why**:
  - An invariance holds only under its assumptions
  - A model with a combined control could not express a setting where an
    assumption fails, so it follows P1
- **Example**: the speed ratio of #789
  - With $t_r = 0$, $P(\text{winner})$ depends on $v_p$ and $v_b$ only through
    $v_p / v_b$: Proposition 1 of the paper, with a proof sketch in the appendix,
    and identical same-seed maps on 30 scaled pairs and 500 random settings
  - The notebooks keep $v_b$, $v_p$, and $t_r$ as separate sliders, with no ratio
    slider
  - The ratio breaks with $t_r > 0$, since the reaction distance $v_p t_r$ enters,
    and with a spread that grows with the ball speed (#723)
  - Results and figures: the Gdoc,
    [Experiment 4](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit?tab=t.0#heading=h.xvdn4n7zo860)
- **Source**:
  [GP on #789](https://github.com/gpsaggese/gpsaggese.github.io/issues/789#issuecomment-6528212968)

### P5: Start Simple

- **Principle**: start with the simplest model that shows the behavior, as a first
  approximation, and add one piece at a time
- **Why**:
  - The 2D approximation is good enough to understand the first and second order
    effects
  - A simple model is fast to build and to check, and it is the baseline that each
    added piece is compared with
- **Example**:
  - 2D, not 3D: `racket_trajectory.py` is set aside, and the bounce is ignored
  - The spread does not depend on the ball speed (Milestone 2) or on player 2's
    movement (#714), as first approximations followed up in #723 and #786
  - A discrete grid of targets (10 x 15 cells for tennis, 9 x 10 for pickleball)
    instead of a continuous optimizer
  - One-shot experiments before RL (phase 1 of #714)
  - The first RL env has one agent, a binary reward (1 for a winner, else 0), and
    the targets of the Milestone 2 grid as its actions (#713)
- **Source**:
  - [GP on #590](https://github.com/gpsaggese/gpsaggese.github.io/issues/590#issuecomment-6128247188):
    the 3D problem complicates things without adding much over 2D
  - [GP on #690](https://github.com/gpsaggese/gpsaggese.github.io/issues/690#issuecomment-6336175829):
    ignore the bounce, and a spread independent of the ball speed as a first
    approximation
  - [GP on #714](https://github.com/gpsaggese/gpsaggese.github.io/issues/714#issuecomment-6336481113):
    keep the uncertainty of the response fixed at first, and the 3 phases
  - [GP on #713](https://github.com/gpsaggese/gpsaggese.github.io/issues/713#issuecomment-6446476291)
    approved the
    [RL spec](https://github.com/gpsaggese/gpsaggese.github.io/issues/713#issuecomment-6434360767)
    with the binary reward and the grid actions

### P16: Base the Assumptions on Evidence

- **Principle**: when a value or a shape of the model cannot be calibrated, take its
  base from a study or a domain expert, cite the source, and try a few simple
  alternatives next to it
  - A value with no source is marked as an assumption
- **Why**:
  - P3 sweeps under reasonable assumptions: a cited source is what makes an
    assumption reasonable, and lets a reader check it
  - When both the coefficient and the shape of a coupling are unknown, the result
    says how much it depends on them, instead of resting on one guess
- **Example**: the spread vs ball speed of #723
  - Linear is the base, and a knee, a parabola, and a one-sided parabola are tried
    next to it; which one wins is part of the result
  - A factor per direction, $a_y = a$ for the depth and $a_x = a / 2$ for the width:
    [Sudo et al. (2024)](https://doi.org/10.1080/14763141.2024.2381105) found the
    speed-accuracy trade-off of a forehand only in the hitting direction
    - $a_x = a / 2$ is our assumption, between Sudo et al. ($a_x = 0$) and
      $a_x = a_y$
  - The base spread at $v_0$, (0.5, 1.0) m, is larger in depth than in width, as in
    [Delgado-García et al. (2019)](https://doi.org/10.1080/24748668.2019.1647733),
    where the long axis of the error ellipse is almost parallel to the sideline
  - Realistic player profiles from an expert who has researched this will replace
    the variants of the sweeps
  - The variant $v_p = 3.0$ m/s of the plan is marked as an assumption, with no
    source for the number
- **Source**:
  [GP in our meeting, recorded on #723](https://github.com/gpsaggese/gpsaggese.github.io/issues/723#issuecomment-6677605768),
  decisions 2 to 4; now in the Gdoc,
  [Experiment 5](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit?tab=t.0#heading=h.qrto9so35bnx),
  Decisions

## Simulation and Results

### P6: Reproducible from a Settings String and a Seed

- **Principle**: every result can be rebuilt exactly from its settings string, which
  lists every setting, including the seed
- **Why**:
  - Anyone must be able to reproduce a result of the Gdoc, the log, or the paper
    without guessing a setting
  - With the seed in the string, the Monte Carlo result repeats exactly
- **Example**:
  - The winner widget of notebook 01 prints a line like this one, built by
    `build_settings_string()`:
    ```text
    sport=Tennis side=deuce x1=-2.1 y1=3.2 std_x2=0.5 std_y2=1.0 ball_speed=25.0 t_r=0.2 v_p=1.5 seed=1 n_x=10 n_y=15 n_samples=5000
    ```
  - Notebook 05 keeps one string per map in its `settings` column
    (`settings_to_string()`, `parse_settings_string()`)
  - The Milestone 2 check rebuilt a result from the settings string alone and got
    the same numbers
- **Source**:
  [GP on #694](https://github.com/gpsaggese/gpsaggese.github.io/issues/694#issuecomment-6336301932):
  create strings that represent all the settings; the Milestone 2 decisions in the
  plan

### P7: Common Random Numbers Across Targets

- **Principle**: every target of a map uses the same seed, so all the targets share
  the same standard normal draws, shifted to their own aim point
- **Why**: 2 targets then differ because of where they aim, not because of sampling
  noise, so comparing them to find the best one is fair
- **Example**:
  - `compute_winner_heatmap()` gives every target a new
    `np.random.default_rng(seed)` with the same seed
  - The sweeps use the same seed for every map, so the #789 check could compare
    scaled maps cell by cell and find them identical
- **Source**: the Milestone 2 decisions in the plan; the paper, Sec. IV-H

### P8: Joint Events by Monte Carlo

- **Principle**: estimate a joint event by Monte Carlo as one event on the same
  samples, not as a product of probabilities
- **Why**:
  - Both events depend on the same landing point, so a product assumes an
    independence that may not hold
  - GP: "I would not try to be cute and just compute the thing as one event"
- **Example**: `compute_winner_probability()` counts the samples that land in AND are
  not reached by player 1
- **Source**:
  [GP on #690](https://github.com/gpsaggese/gpsaggese.github.io/issues/690#issuecomment-6252896405);
  the paper, Sec. IV-H

### P9: Standard Errors and a Best Region

- **Principle**: report each estimate with its standard error, and the best target
  as a region within 2 standard errors of the highest, not as a single best cell
- **Why**:
  - Many targets can tie at the top, e.g., $P(\text{winner}) = 1$ when every sample
    is a winner
  - A single argmax would pick one of them by sampling noise
- **Example**:
  - `compute_winner_probability()` returns the standard error with the estimate
  - `find_best_region()` keeps every target whose gap to the highest is within 2
    standard errors of that difference, and the map outlines the region
  - `find_best_target()` stars the cell of the region closest to its centroid
- **Source**: the Milestone 2 decisions in the plan; the paper, Sec. IV-H

## Reinforcement Learning

### P10: Reuse Code, Not Results

- **Principle**: the RL work reuses the code of the one-shot experiments, but not
  their results: RL must find the same results on its own
- **Why**:
  - RL finding the same best region on its own checks both the RL and the one-shot
    results
  - Starting the RL from the one-shot answer would make that check circular
- **Example**: the racket env of #713, in notebook 06
  - Its transitions reuse the Milestone 2 sampling and reach functions
  - The target is random and learned by the RL loop, and the brute force through the
    env matches a fresh Milestone 2 map: the Gdoc,
    [Experiment 6](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit?tab=t.0#heading=h.kntokyu5w697)
  - Only phase 3 of #714 may replace parts of the RL with the one-shot results, after
    phase 2 has reproduced them; until then PPO searches the full grid, with no
    pruning
- **Source**:
  [GP on #713](https://github.com/gpsaggese/gpsaggese.github.io/issues/713#issuecomment-6446476291);
  [GP on #713](https://github.com/gpsaggese/gpsaggese.github.io/issues/713#issuecomment-6751589012):
  the full grid, with no pruning, for now; the phases in the plan

## Process

### P11: Finish One Thing, Clean Up, Then Move On

- **Principle**: focus on one thing at a time: finish it, do a quick cleanup, and
  then move on to the next step
- **Why**:
  - A task left half done mixes its loose ends with the next one
  - A cleanup with the same checks passing before and after shows that it changed
    no behavior
- **Example**:
  - #713: PR #838 finished the Gymnasium notebooks, fixed, tested, and cleaned up
    to the notebook rules, before PR #846 moved on to the example notebook
  - #823: the repetition in the code and the tests is removed, with the same
    coverage and the same checked lines before and after, before #723 moves on to
    a new model
- **Source**:
  [GP on #691](https://github.com/gpsaggese/gpsaggese.github.io/issues/691#issuecomment-6309187106):
  "Let's do a quick clean up and then feel free to propose the next steps";
  [GP on #713](https://github.com/gpsaggese/gpsaggese.github.io/issues/713#issuecomment-6336386717):
  focus on one thing at a time, finish it, then move on

### P12: One Issue per Task

- **Principle**: every task or problem gets its own GitHub issue, on the plan; a
  research question about it is asked in the Gdoc, and the issue gets only a link
  (P13)
- **Why**: each task keeps its PR, and the link to its research discussion, in one
  place
- **Example**: GP filed the spread vs ball speed question of #690 as #723; the open
  questions of #723 are in the Gdoc, under
  [Questions for GP](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit?tab=t.a8h8vb8v4lq8#heading=h.xfydadrg3lvc),
  and #723 has one line with the link
- **Details**: the Tasks section of
  [`project_management.explanation.md`](project_management.explanation.md#tasks)
- **Source**:
  [GP on #690](https://github.com/gpsaggese/gpsaggese.github.io/issues/690#issuecomment-6336175829):
  "Again every problem -> GH issue";
  [GP on #694](https://github.com/gpsaggese/gpsaggese.github.io/issues/694#issuecomment-6712939277):
  the research questions go in the Gdoc

### P13: Research Discussions and Decisions in the Gdoc

- **Principle**: the research questions, their discussion, and the research decisions
  live in the research Gdoc, with comments and suggestion mode; GitHub gets only a
  link
  - Coding questions stay on GitHub, and no decision lives only in Slack or in a
    meeting
- **Why**:
  - GP: "stuff in GitHub and in Slack are meant to be lost"
  - In the Gdoc a question sits next to the result it is about, and its answer and
    its decision stay with it
  - A newcomer, or we later, can see what was decided and why
- **Example**: the open questions of #713 and #881 are comments under
  [Questions for GP](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit?tab=t.a8h8vb8v4lq8#heading=h.xfydadrg3lvc)
  in the Gdoc, and each issue has one line with the link
  - GP answered Q7 and Q8 on #713, and both decisions are recorded under
    [Experiment 6](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit?tab=t.0#heading=h.kntokyu5w697),
    Decisions
- **Details**: the Communication and Research Questions sections of
  [`project_management.explanation.md`](project_management.explanation.md#communication)
- **Source**:
  [GP on #694](https://github.com/gpsaggese/gpsaggese.github.io/issues/694#issuecomment-6712939277),
  rules 1 to 3
  - On 2026-11-07 this replaced the rule "every decision is written on GitHub or in
    the repo", from
    [GP on #625](https://github.com/gpsaggese/gpsaggese.github.io/issues/625#issuecomment-6185837906),
    item 8

### P14: The Plan and the Paper Control the Code

- **Principle**: the plan and the paper control the code: GP reviews them, more
  than every line of the code
  - The notebooks contain all the details, the log all the experiments, and the
    paper the interesting experiments
  - Where the experiments are logged, in the log, in the Gdoc, or in both, is pending
    GP:
    [Q1](https://docs.google.com/document/d/19KGM-1a9U5E7CrE4ltljYz7IVtMfNuRzrqgGkkuXUSQ/edit?tab=t.a8h8vb8v4lq8#heading=h.21d3arlkot6c)
- **Why**:
  - Reviewing the plan and the paper scales better than reviewing every line
  - It keeps the code in sync with what the paper claims
- **Example**:
  - GP asked to merge #812 without a detailed review of the code, and to use the
    plan and the paper to control it
  - The #789 result went to the paper as Proposition 1, and the #694 corner sims as
    Sec. VII-A
- **Details**: the Paper section of
  [`project_management.explanation.md`](project_management.explanation.md#paper):
  every research PR names the paper section or the Gdoc heading that it changes
- **Source**:
  [GP on #812](https://github.com/gpsaggese/gpsaggese.github.io/issues/812#issuecomment-6545315777)
  and
  [GP on #789](https://github.com/gpsaggese/gpsaggese.github.io/issues/789#issuecomment-6528212968);
  [GP on #694](https://github.com/gpsaggese/gpsaggese.github.io/issues/694#issuecomment-6712939277):
  control the research through the paper

### P15: Put the Smarts in the Skills

- **Principle**: when a skill misses something, improve the skill instead of fixing
  its output by hand
- **Why**: a fix in a skill applies to every later run and every student, while a
  fix by hand applies once
- **Example**: GP's way to remove the repetition of #823 is to run
  `/coding.factor_common_code` and `/testing.fix_unit_tests`, check that the
  repetition is gone, and add to the skill what it missed
- **Source**:
  [GP on #823](https://github.com/gpsaggese/gpsaggese.github.io/issues/823#issuecomment-6602057562);
  [GP on #713](https://github.com/gpsaggese/gpsaggese.github.io/issues/713#issuecomment-6557078564):
  80 to 90% of the work is automated by the skills, and humans review and steer the
  last mile
