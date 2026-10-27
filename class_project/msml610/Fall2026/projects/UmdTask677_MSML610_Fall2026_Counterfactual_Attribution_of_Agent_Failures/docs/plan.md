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
- Deadline: December 1, 2026 (to confirm against the end of the semester,
  December 21)
- Budget: about $500 of API calls for the whole project; the first batch is
  capped at about $150, so that most of it remains for Milestones 3 and 4

# Definitions

- **Step**: one model reply together with the execution of its command
- **Checkpoint**: the conversation and the changed files after a step
- **Fork**: a restore of a checkpoint into a fresh container, on a new branch
- **`V(j)`**: fraction of independent continuations from the checkpoint after
  step `j` that solve the task, with the same Haiku policy throughout
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
- Remaining: the noise floor at a checkpoint where Haiku sometimes succeeds and
  sometimes fails; the 5 earlier tasks are all-or-nothing for Haiku

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

- 20 SWE-rebench tasks from 20 different repositories, with historical submitted
  successes and failures from another agent (a selection aid, not evidence of
  mixed outcomes for Haiku):
  - `elastic__apm-agent-python-1619`, `litl__backoff-161`, `encode__httpx-386`,
    `networkx__networkx-5354`, `fsspec__filesystem_spec-930`,
    `marshmallow-code__webargs-428`, `ansible__ansible-builder-336`,
    `hynek__structlog-221`, `jaraco__zipp-59`, `dask__zict-64`,
    `IAMconsortium__nomenclature-390`, `markusressel__container-app-conf-61`,
    `chimpler__pyhocon-173`, `Delgan__loguru-1239`,
    `sphinx-gallery__sphinx-gallery-480`, `tophat__syrupy-607`,
    `python-cmd2__cmd2-117`, `yukinarit__pyserde-620`,
    `stephenhillier__starlette_exporter-89`, `Fatal1ty__mashumaro-130`
- Reserves: `fair-workflows__nanopub-140`, `nutechsoftware__alarmdecoder-60`,
  `dchevell__flask-executor-45`, `cleder__fastkml-223`
  - A task that cannot be run or scored reliably is documented with the exact
    reason and replaced by a reserve; replacements are never chosen by whether
    Haiku succeeds

## Stage A: Five-Task Pilot

- Tasks: `encode__httpx-386`, `networkx__networkx-5354`,
  `fsspec__filesystem_spec-930`, `marshmallow-code__webargs-428`,
  `hynek__structlog-221`
- For each task:
  - Check that the unchanged repository fails the official tests and the
    reference fix passes; record the evaluator version and any errors
  - Run Haiku twice from a fresh start under one fixed, recorded configuration
  - Save a checkpoint at every decision boundary, with its id and path
  - Store the full trajectory, final patch, official score, tokens and cost;
    keep agent failures separate from evaluator or infrastructure failures
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
- Report the uncertainty, cost per continuation, evaluator time and the
  projected cost of the batch
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
- Planted premature submissions reported separately, as positive controls
- "Thinking on for one step" is a different intervention and is reported
  separately

## Milestone 4 Data

- One row per decision with features available before that decision (task,
  repository, decision index, action type, context length, recent tool and test
  outcomes) and, separately, the `G` estimate and its uncertainty as the label
- No future information as a feature; task and repository ids kept so that the
  splits are by whole task, preferably by repository

## Scorer Performance

- A local, pinned and verified copy of the files the evaluator downloads from
  GitHub, then base/gold and A/A checks re-run to show the cache changes nothing
- Throughput benchmarked before raising parallelism; CPU, memory, wall time,
  cache hit rate and errors recorded

## Infrastructure

- An x86-64 Linux machine with Docker, outbound internet, about 8 CPUs, 32 GB
  RAM and 200 GB of disk, so the task images run without emulation
  - Zaratan: compute nodes reach the Anthropic API and Docker Hub; Docker or
    Podman availability is still to check
  - Otherwise a cloud VM (e.g., about $27-45 for 72-120 hours), destroyed after
    the batch
- Docker setup: the project image has the Docker CLI and the official evaluator;
  the `docker_*.sh` scripts mount the host's Docker socket, so the task
  containers run on the host's Docker next to the project container
- One task proven end to end on the machine before the pilot; the pilot runs as
  a restartable background job that saves after every run and fork

# Pull Requests

| PR | Content | Status |
| :- | :------ | :----- |
| PR 1 | Project template, Docker setup (Docker CLI and host socket for the task containers, pinned evaluator), and this plan | In review |
| PR 2 | Runner ported into `caaf_*.py`, notebooks, unit tests, project README and docs (`progress.md`, `decisions.md`, `how_we_work.md`) | Next |
| PR 3 | Scorer cache with verification, and one task end to end on the experiment machine | Planned |
| PR 4 | Stage A: five-task pilot, manifests, replay checks | Planned |
| PR 5 | Stage B: estimator checks and the A/A noise floor (Milestone 1 complete) | Planned |
| PR 6 | Stage C: the remaining 15 tasks | Planned |
| PR 7 | Stage D: blame profiles and the Milestone 4 data table | Planned |
