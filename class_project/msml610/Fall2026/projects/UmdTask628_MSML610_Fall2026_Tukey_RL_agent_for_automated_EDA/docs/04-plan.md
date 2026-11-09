# Tukey Implementation Plan

## Goal

- Train an agent that chooses statistical tests on a data table and emits its
  causal graph, rewarded by a verifiable score on synthetic data
- Background: [spec](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/research/ideas/in_progress.RL_for_auto_EDA.md),
  [prototype](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/research/agentic_data_science/RL_for_auto_EDA_prototype.py),
  [paper draft](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/papers/RL_for_Automated_EDA/paper.md)

## Plan Hierarchy

- This file: what to submit and in which order
- [`07-progress.md`](07-progress.md): what is done and learned
- [`03-decisions.md`](03-decisions.md): why it is built this way
- `- [ ]` is an unmerged PR and `- [x]` is a merged PR

## Roadmap

- A first version of M1-M4 exists locally. Each PR brings one part into the
  repo, revalidated, and opens after the review of the previous one
- `v0.1`: plan and docs
  - [ ] `PR_P1`
- `v0.2`: environment, tools, scripted baseline, scoring (M1-M2)
  - [ ] `PR_B1`
- `v0.3`: RLVR training (M3)
  - [ ] `PR_R1`
- `v0.4`: generalization and sim-to-real (M4)
  - [ ] `PR_G1`
  - [ ] `PR_G2`
- `v1.0`: final submission
  - [ ] `PR_F1`
- Releases are cumulative

# Component Specs

## Environment and Data

- Goal: random causal graphs, a train/test split, an environment that hides the
  graph and the test rows
- Open: are [`D1`](03-decisions.md#d1-score-against-the-graph-the-data-can-identify)
  and [`D2`](03-decisions.md#d2-hidden-confounders-are-scored-against-a-pag) the
  right reading of Key Examples 1 and 2?

## Tools and Scoring

- Goal: tools on the training rows; a score combining structure and prediction
- Open: is `lam = 0.5` the right reward (spec Question 1)? See
  [`D3`](03-decisions.md#d3-combine-structure-and-prediction-with-lam--05)

## Policy and Training

- Goal: a tool policy and a graph decoder trained with policy gradients
- Open: does test selection or the decoder carry the gain? Why do GES and
  DirectLiNGAM stay ahead?

## Generalization and Real Data

- Goal: a curriculum and a report on the sim-to-real gap and failure modes
- Open: messy data (spec Question 2) is untested; how does our report relate to
  the paper draft?

# PRs

## v0.1: Plan and Docs

### `PR_P1`: [ ] Plan and docs

- README, plan, decisions, progress log, how we work; docs only
- Acceptance: reviewers comment on `D1`, `D2` and the scope of `PR_B1`
- Owner: Rayhan, Nov 4

## v0.2: Environment, Tools, Baseline and Scoring

### `PR_B1`: [ ] Environment, tools, scripted baseline and scoring (M1-M2)

- One runnable path: generate data, run the tools and the scripted rule, score
- Generators, environment, tools, scripted rule, scoring, PC/GES/DirectLiNGAM
  baselines, the M1-M2 graph sets, Docker (`requirements.in`), two notebooks
- Tests: generator contracts, training-only preprocessing, no leak of the true
  graph or test rows, tools, agreement with the prototype, scoring conventions
- Acceptance: full tests and the Docker build pass on a clean checkout; each
  number names its input file and command. Split it if too large to review
- Owners (proposed): Rayhan, integration; Ronak, environment checks and Docker;
  Sai, tools and baselines checks. Target Nov 8

## v0.3: RLVR Training

### `PR_R1`: [ ] Policy, training and the M3 evaluation

- Features, policy, training phases, evaluation and the confirmatory results
- Tests of the decoder and tool losses, including the entropy, KL and value terms
- Owners (proposed): Rayhan; Sai checks the comparisons, Ronak reproducibility

## v0.4: Generalization and Sim-to-Real

### `PR_G1`: [ ] Curriculum and noise studies (M4)

- State what each comparison does and does not isolate
- Owner (proposed): Ronak

### `PR_G2`: [ ] Real data and the sim-to-real report (M4)

- Sachs and diabetes, reference graphs, failure modes, answers to the spec's
  questions
- Owner (proposed): Sai

## v1.0: Final Submission

### `PR_F1`: [ ] Final README, report and video

- Closes #628. Owners: all three, Dec 7

## Unscheduled: Open Research

### `PR_X1`: [ ] Decoder-only control

- Train without the tool-RL phase, on validation data, to separate test
  selection from decoder training

### `PR_X2`: [ ] Messy-data evaluation

- Evaluate the trained policy and the baselines on new synthetic data with
  missing values and drift (spec Question 2); no retraining

## Team Responsibilities

- Proposed, to confirm as a team: Rayhan, scoring and M3; Ronak, environment,
  Docker, curriculum; Sai, tools and baselines checks, real data, related work
- Each owner delivers changes or analysis with evidence, explains it, and
  reviews one other component
- A proposed role is not completed work; contributions are recorded per PR in
  [`07-progress.md`](07-progress.md)

## Conventions

- [Course project rules](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/class_project/README.md),
  [Docker and dependencies](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/class_project/project_template/docker_scripts.README.md)
