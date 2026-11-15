# Plan

- Goal of the project: show that the effect of one step of a failed coding-agent
  run can be measured by restarting the run from a checkpoint, and that a cheap
  predictor of that effect is feasible
  - Proposal:
    [`draft.Counterfactual_Attribution_of_Agent_Failures.md`](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/research/ideas/draft.Counterfactual_Attribution_of_Agent_Failures.md)
  - Issue: [#677](https://github.com/gpsaggese/gpsaggese.github.io/issues/677)
- Priority: Milestone 4, a cheap predictor of which steps matter and a test of
  whether acting on it improves outcomes
  - Milestone 1 and 2 data are collected in a form that supports it
- Scope, as agreed on the issue: show feasibility of the approach, not run a
  large-scale experiment; any way to scale the experiment down is welcome
- Task (changed 2026-11-10): chess endgames (King and Rook vs. King) instead of
  SWE-rebench. Zaratan has no Docker, and the SWE tasks need a Docker container
  per task and an evaluator that is too slow to scale. The agent (mini-swe-agent
  with Claude Haiku 5.5), the checkpoints, the definitions and the stages are
  unchanged; "solves the task" now means "checkmates before the fifty-move
  rule", judged exactly by the Syzygy tablebase
- Policy (changed 2026-11-11): Haiku 5.5 with thinking on (adaptive, default
  effort, `config/chess_haiku_think.yaml`). With thinking off it won 0 of 10
  pilot games and no checkpoint had a mixed outcome, so the A/A check was not
  possible (`docs/milestone1.md`)
- Deadline: December 1, 2026 (to confirm against the end of the semester,
  December 21)
- Budget: about $500 of API calls for the whole project; the first batch is
  capped at about $150, so that most of it remains for Milestones 3 and 4

# Definitions

- **Step**: one model reply together with the execution of its command
- **Checkpoint**: the conversation and the changed files after a step
- **Fork**: a restore of a checkpoint into a fresh environment, on a new branch
- **`V(j)`**: fraction of independent continuations from the checkpoint after
  step `j` that solve the task (checkmate), with the same Haiku policy
  throughout
- **Repair gain**: `G(k) = V(before step k) - V(after step k)`
  - Positive `G` means re-deciding step `k` helps; negative `G` means the
    original step was helpful
  - This is the negative of the proposal's `Delta_k` (original action minus
    alternative); `G` is used in all code, plots, tables and reports
- **A/A check**: two independently sampled groups of continuations under the
  same action and policy; the spread of their gap is the noise floor

# Roadmap

## Milestone 1: Checkpoint and Resume

- Done on 5 SWE-rebench tasks, before this repo: recording runner, checkpoint
  after every step, restore, exact replay, reference-fix checks
- On chess, in this repo: the same runner without containers (`caaf_*.py`), the
  five-task pilot, replay checks at every checkpoint, and the noise floor at
  checkpoints where Haiku sometimes wins and sometimes fails
  (`caaf_m1.py`, results in `results/m1_think/` and `docs/milestone1.md`)
- Status (2026-11-11): complete. Replay checks pass at every checkpoint; the
  pilot won 3 of 10 games; the A/A gaps at two checkpoints near `V` = 0.5 are
  0.10 and 0.14 with 50 vs 50, both within noise (half-width 0.19)

## Milestone 2: Blame Profiles

- `V` curves and repair gain `G` per step for natural failed runs, with
  uncertainty
- First evidence on sparse vs diffuse failures

## Milestone 3: Taxonomy and Transfer

- Group the profiles, label the blamed steps, and count the failure modes
- Decide whether transfer means a second model or different settings of Haiku

## Milestone 4: Cheap Surrogate and Feedback Loop

- Train a predictor of `G` from features available before each decision, and
  validate it on held-out tasks
- Change the agent scaffold to prevent the top failure modes and measure the
  success-rate change on unseen tasks

# Execution: the 20-Task Batch

## Tasks

- 20 KRvK starting positions in 5 difficulty bins (mate in 2, 3, 4, 6 and 8
  moves with best play), 4 per bin, each with 8 spare half-moves before the
  fifty-move rule; seeded and listed in `data/tasks.json`
  (`python caaf_m1.py tasks`)
- Reserves: one more position for each of the first 4 bins, in the same file
  - A task that cannot be run or scored reliably is documented with the exact
    reason and replaced by a reserve; replacements are never chosen by whether
    Haiku succeeds

## Stage A: Five-Task Pilot

- Tasks: the first position of every bin (`krk-m2-1`, `krk-m3-1`, `krk-m4-1`,
  `krk-m6-1`, `krk-m8-1`)
- For each task:
  - Check with the tablebase that the start is won and the referee scores a
    known mating line as a win (unit tests)
  - Run Haiku twice from a fresh start under one fixed, recorded configuration
  - Save a checkpoint at every decision boundary, with its id and path
  - Store the full trajectory, final board, referee score, tokens and cost;
    keep agent failures separate from API or infrastructure failures
  - Replay-check every checkpoint and report every mismatch
- Report how many runs succeeded, failed, or could not be scored

## Stage B: Check the Milestone 2 Estimator

- `V` curves on the pilot failures, with at least 10 continuations per
  checkpoint
- Direct-arm check of the `V`-curve shortcut on 2-3 natural, nontrivial steps:
  a keep-original-action arm and a re-decide arm from the same before-step
  checkpoint, compared with `V(after)` and `V(before)`
- A/A check at two checkpoints where success is neither near certain nor near
  impossible, ideally 50 continuations per group, with the groups interleaved in
  running and scoring; this completes Milestone 1
- Report the uncertainty, cost per continuation and the projected cost of the
  batch
- The fact that `G` values add up to `V(start)` is true by construction and is
  not evidence that attribution is correct

## Stage C: The Remaining 15 Tasks

- Same collection procedure: two fresh Haiku runs per task, checkpoints at every
  decision boundary, about 40 fresh runs in total
- Task manifest (one row per task) and run manifest (one row per run)
- Keep successful and failed runs; never label planted failures as natural
- If there are fewer than about 15 natural failures, report it and propose a
  measured expansion

## Stage D: Blame Profiles

- Fully sampled profiles for at least 5 natural failed runs with some measurable
  chance of recovery, at about 20-30 continuations per checkpoint
- Sampling and stopping rules stated before looking at the final patterns, with
  flat checkpoints sampled too
- "Sparse" and "diffuse" defined in advance; "no recovery observed under this
  tested policy" with an uncertainty bound, never "impossible"
- Two natural-run profiles repeated from scratch as a stability check
- Planted blunders (e.g. a move that stalemates or hangs the rook) reported
  separately, as positive controls
- Changing the model's settings for one step (e.g. thinking off, or a higher
  effort) is a different intervention and is reported separately

## Milestone 4 Data

- One row per decision with features available before that decision (task,
  difficulty bin, decision index, move type, context length, recent illegal
  moves and format errors) and, separately, the `G` estimate and its
  uncertainty as the label
- No future information as a feature (in particular no tablebase value); task
  and bin ids kept so that the splits are by whole task, preferably by bin

## Scorer and Infrastructure

- The scorer is the Syzygy tablebase (KRvK, 8 KB, checksummed by
  `data/fetch_tablebases.sh`): exact, instant and free, so no scorer cache or
  evaluator machine is needed
- The environment is the board program inside the agent's process: the agent's
  only commands are `play show` and `play move <move>`, and anything else is
  refused without running, so no task container is needed
- Runs anywhere with Python 3.12 and outbound access to the Anthropic API: a
  laptop, a Zaratan node (plain virtualenv, no Docker), or the project's Docker
  image (`./docker_build.sh`, then `./docker_bash.sh`)
- The Docker image holds only Python, Jupyter and `requirements.txt`; the Docker
  CLI, the host-socket mount and the SWE evaluator from PR 1 were removed in
  PR 2

# Pull Requests

| PR | Content | Status |
| :- | :------ | :----- |
| PR 1 | Project template, Docker setup (Docker CLI and host socket for the task containers, pinned evaluator), and this plan | Merged (#785) |
| PR 2 | Chess runner in `caaf_*.py`, unit tests, Stage A pilot, replay checks and the A/A noise floor (Milestone 1 complete), project README, `03-decisions.md`, `07-progress.md`, Docker image without the SWE parts | Next |
| PR 3 | Notebooks, `08-how_we_work.md`, and the research Google Doc | Planned |
| PR 4 | Stage B: `V` curves and the direct-arm check of the shortcut | Planned |
| PR 5 | Stage C: the remaining 15 tasks | Planned |
| PR 6 | Stage D: blame profiles and the Milestone 4 data table | Planned |
