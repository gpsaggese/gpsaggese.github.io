# Skill vs Luck in Agent Benchmarks

MSML610 Fall 2026 research project (group 5: Pratham Bharati, Shraddha
Chandrashekar), tracked in
[issue #716](https://github.com/gpsaggese/gpsaggese.github.io/issues/716)

- The question: how much of an agent leaderboard is skill and how much is luck
  - Luck in which tasks end up in the benchmark
  - Luck in running each agent only once
  - Luck in picking the best of many scaffolds, prompts, or attempts
- The spec is
  [`draft.Skill_vs_Luck_in_Agent_Benchmarks.md`](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/research/ideas/draft.Skill_vs_Luck_in_Agent_Benchmarks.md)
- Everything runs locally in Docker, at no cost, on 3 sources of data:
  - The public per-task results of every SWE-bench Verified submission
  - Our own runs of 6 open coding models on HumanEval+ and MBPP+, served by
    Ollama, with 2 scaffolds and up to 5 runs per task, and 20 prompt variants
    of 1 of them
  - The code that EvalPlus generated in 2023 with many models, including 200
    samples per task at 2 temperatures, scored again here
- The paper draft is
  [`papers/Skill_vs_Luck_in_Agent_Benchmarks/paper.md`](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/papers/Skill_vs_Luck_in_Agent_Benchmarks/paper.md),
  built with `make` in its dir

## Results So Far

- On the pinned snapshot of `SWE-bench/experiments` (commit `40f164d`, 173
  agents x 500 tasks):
  - A single score is only known to about +/- 4 points
    - The median 95% interval is 8.2 points wide
  - Neighbors on the leaderboard cannot be told apart
    - The median gap between neighbors is 0.2 points
    - None of the 172 neighbor pairs differ after a Holm correction
  - 18 agents cannot be told apart from the agent ranked first
  - The benchmark is too small for the gaps it shows
    - Resolving a 1-point gap would take about 14,700 tasks
    - 500 tasks only resolve gaps of about 5.4 points or more
  - The ranking itself is uncertain
    - The top 3 agents share about 94% of the first place, and a typical top-25
      agent has a 95% rank interval about 18 ranks wide
  - Rankings on 2 halves of the tasks agree over all 173 agents (Kendall 0.90)
    but are weak among the top 20 (Kendall 0.25)
  - The leader of one half scores 2.0 points less on the other half
  - The best of 20 equal variants on 500 tasks looks 3.8 points better than it
    is
  - Under a 2PL IRT fit, the 500 tasks are worth about 107 equally informative
    tasks for the top 20
- The analysis and the plots are in `skill_vs_luck.example.ipynb`
- The scorer of the coding benchmarks matches the published EvalPlus scores to
  within 2 tasks, on the 5 public greedy sets checked
  - E.g., on the public greedy samples of GPT-4-Turbo it gives 81.7 on
    HumanEval+ and 73.5 on MBPP+, against the published 81.7 and 73.3
- The predictions for the local runs are in `preregistration.md`, written before
  the full runs and committed before any of them was scored
- On our own runs (3 models x 2 scaffolds, HumanEval+ and MBPP+, 5 runs per task
  at temperature 0.8), in `skill_vs_luck.runs.ipynb`:
  - The greedy scores are within 2 points of the published EvalPlus scores
  - Run-to-run luck is 28% to 52% of the variance of a 1-run outcome
  - Only 18.5% (HumanEval+) and 25.3% (MBPP+) of random 1-run leaderboards keep
    the order of all 6 agents, mostly because a model with and without repair
    is almost the same agent; with the 3 models alone, 76% to 96% do
  - The close pair (CodeQwen 7B and OpenCoder 8B) cannot be told apart even with
    5 runs per task; on HumanEval+ its gap in single runs goes from -2.4 to +6.1
    points
  - In the public EvalPlus samples, going from temperature 0.2 to 0.8 multiplies
    the run variance by 1.8 to 3.0
  - 4 of the 7 pre-registered predictions held (P1, P2, P4, P6), and P3, P5, and
    P7 did not
- Phase 2 (how many SWE-bench records were real, run luck from 1 run, lower
  temperatures, and 3 more models) has its plan and predictions in
  `preregistration_2.md`, committed before any of it was analyzed
  - Study A, in `skill_vs_luck.records.ipynb`: only 2 of the 30 SWE-bench
    Verified records are distinguishable from the record they broke, and they
    carry 33% of the 74.8 points of progress; since June 2024, 27 records in a
    row were each within the noise of the record before (A1 and A2 held, A3 did
    not)
  - Study B, in `skill_vs_luck.one_run.ipynb`, is tuned on the public samples
    only: from 1 run, a 2PL fit of agents run at the same temperature puts the
    run share within 10 points of the truth in 86% of the draws, while adding
    the public greedy agents made every setting worse
    - Its test on our runs, in `skill_vs_luck.one_run_test_set.ipynb`, run once
      after the estimator was committed, failed: the tuned estimate is within
      10 points for only 7 of the 18 new cases (B1 did not hold), mostly
      because it is too high at temperature 0.8; before tuning it is above the
      truth in 16 of the 18 (B2 held)
  - Study C, in `skill_vs_luck.temperatures.ipynb`: at temperature 0.2 runs are
    15% to 21% of the variance of 1 run (37% to 52% at 0.8), 2 greedy runs give
    the same result on every task, and the close pair still cannot be told
    apart; with one-shot, 1-run leaderboards keep the order as often at 0.2 as
    at 0.8 (C1 to C4 held, C5 did not)
  - Study D, in `skill_vs_luck.more_models.ipynb`: with 3 more models, only
    44.4% of the 1-run leaderboards of the 6 models keep the order on MBPP+ (D3
    held); Llama 3.1 8B scores 7.6 points above its published greedy MBPP+
    score (D1 did not hold), and the 3 models within 2.1 points of each other
    in the published MBPP+ scores are 5 to 12 points apart in our runs (D2 did
    not hold)
  - 8 of the 13 phase-2 predictions held (A1, A2, B2, C1 to C4, and D3), and
    A3, B1, C5, D1, and D2 did not
- Milestones 3 and 4 have their plan and 15 predictions in
  `preregistration_3.md`, committed before any of them was run
  - Study E, in `skill_vs_luck.transfer.ipynb`: HumanEval+ and MBPP+ rank 18
    models almost the same way (Spearman 0.98, and 0.99 once task luck is
    removed, with an interval that reaches 1), but not on the same scale: 6 of
    the 7 chat models score 14 to 21 HumanEval+ points more than a typical
    model with the same MBPP+ score, and 2 base models about 4 to 5 points
    less, so 8 of the 18 models are benchmark-specific
    - Under a 2PL fit, HumanEval+ is worth about 119 equally informative tasks
      (72% of its 164) and MBPP+ about 214 (57% of its 378)
    - The 89 models of the EvalPlus leaderboard agree too (Spearman 0.94)
    - 4 of the 6 predictions held (E1, E2, E4, and E6), and E3 and E5 did not
  - Study F, in `skill_vs_luck.selection.ipynb`: on the public samples,
    reporting the best of K runs inflates a score by just what
    `sigma_run x E[max of K]` predicts, from 1.2 points for K = 2 to 4.0 for
    K = 20; on our runs, with only 3 or 5 runs per cell, the inflation is 1.41
    times the prediction at K = 2
  - Study G: the 20 prompt variants of DeepSeek Coder 1.3B score 50.0% to 60.4%
    on HumanEval+ and 41.8% to 54.0% on MBPP+; the best of K variants, picked
    on half of the tasks, scores up to 5.1 points (HumanEval+) and 2.5 points
    (MBPP+) less on the other half
  - Study H: of the 5 corrections of a winner's score, only the paired rule is
    within 1 point of the held-out score in every setting, and it has the
    lowest error, so it is the one we recommend
    - The binomial rule corrects too much (up to -4.9 points) and empirical
      Bayes too little (up to +2.0)
    - The conditional estimate of Andrews, Kitagawa, and McCloskey is right on
      the median, but it misses by up to 7.7 points on average, since it falls
      far below the score when the winner only just won; their hybrid version,
      an extra look, still misses by up to 1.5 points
    - The correction of the spec, `sigma x sqrt(2 ln K)`, corrects even more
      than the binomial rule: -8.0 and -6.4 points at K = 20 (an extra look,
      promised in #716)
  - The reporting protocol, `report_agent_scores()`, takes the SWE-bench
    Verified leader from 79.2% to 75.1%
    - On held-out halves of the SWE-bench tasks (an extra look), the leader
      drops by 1.88 points: the rules that assume equal candidates correct 3.8
      to 6.3 points too much, and no correction is within 1 point, so for a
      leaderboard the drop on held-out halves is the better guide
  - 6 of the 9 predictions of Studies F and G held (F2, F3, and G0 to G3), and
    F1, G4, and G5 did not, so 10 of the 15 predictions of Milestones 3 and 4
    held

## Status of the Milestones

| Milestone                     | Status                                                                          |
| :---------------------------- | :------------------------------------------------------------------------------ |
| 1. Outcome matrix             | Done on SWE-bench Verified, HumanEval+, and MBPP+                               |
| 2. Quantify luck              | Done: task part on SWE-bench, run part on our runs                              |
| 3. Persistence, transfer, IRT | Done: split-half and IRT on SWE-bench, transfer and IRT on HumanEval+ and MBPP+ |
| 4. Selection effects          | Done: winner's curse on real runs and prompt variants, and a correction to use  |

## Structure of the Dir

| Dir                          | Description                                                                                 |
| :--------------------------- | :------------------------------------------------------------------------------------------ |
| `data/`                      | SWE-bench Verified snapshot and the list of the EvalPlus downloads                          |
| `data/coding_runs/`          | Scores of the coding-benchmark solutions, from `score_coding_runs.py`, kept on Google Drive |
| `data/coding_runs_phase2/`   | Scores of the phase-2 runs (Studies B, C, and D), kept on Google Drive                      |
| `data/coding_runs_pilot/`    | Scores of the 20-task pilot, kept on Google Drive                                           |
| `data/coding_runs_variants/` | Scores of the 20 prompt variants of Milestone 4, not committed                              |
| `data/raw/`                  | Downloads and raw model answers, rebuilt by the scripts and not committed                   |
| `docs/`                      | Progress log (`progress.md`) and decisions (`decisions.md`)                                 |
| `test/`                      | Unit tests                                                                                  |

## Description of Files

| File                                    | Description                                                                                                                        | Cluster   |
| :-------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------- | :-------- |
| `bashrc`, `etc_sudoers`                 | Shell and sudo configuration copied into the image                                                                                 | Docker    |
| `docker_*.sh`, `run_jupyter.sh`         | Docker scripts, copied unchanged from `class_project/project_template`                                                             | Docker    |
| `docker_cmd_sandbox.sh`                 | Like `docker_cmd.sh`, for running model-generated code (see below)                                                                 | Docker    |
| `Dockerfile`                            | Python 3.12 slim image with the packages in the 2 requirements files                                                               | Docker    |
| `docs/decisions.md`                     | Non-obvious choices, one entry per decision                                                                                        | Docs      |
| `docs/progress.md`                      | Status, built pieces, findings with their evidence, and history                                                                    | Docs      |
| `fetch_coding_scores.py`                | Download the scores of the coding runs from Google Drive into `data/`                                                              | Data      |
| `fetch_evalplus_data.py`                | Download the EvalPlus datasets and the public EvalPlus samples                                                                     | Data      |
| `fetch_swebench_verified.py`            | Download the public SWE-bench Verified results into `data/`                                                                        | Data      |
| `preregistration.md`                    | Design, analyses, and predictions of the local runs, written before them                                                           | Docs      |
| `preregistration_2.md`                  | Design and predictions of the 4 phase-2 studies, written before them                                                               | Docs      |
| `preregistration_3.md`                  | Design and predictions of Milestones 3 and 4, written before them                                                                  | Docs      |
| `README.md`                             | This file                                                                                                                          | Docs      |
| `requirements.eval.txt`                 | Pinned packages that the EvalPlus scorer needs                                                                                     | Docker    |
| `requirements.txt`                      | Pinned Python packages installed in the image, including `jax` and `numpyro`                                                       | Docker    |
| `run_coding_agents.py`                  | Generate answers to HumanEval+ or MBPP+ tasks with a model served by Ollama                                                        | Runs      |
| `run_coding_experiment.sh`              | Run every model and score every answer                                                                                             | Runs      |
| `run_prompt_variants.sh`                | Generate and score the 20 prompt variants of Milestone 4                                                                           | Runs      |
| `score_coding_runs.py`                  | Score answers or public samples with the EvalPlus tests                                                                            | Runs      |
| `skill_vs_luck.API.ipynb`               | Walkthrough of every function in `skill_vs_luck_utils.py`                                                                          | Notebooks |
| `skill_vs_luck.API.py`                  | Jupytext pair of `skill_vs_luck.API.ipynb`                                                                                         | Notebooks |
| `skill_vs_luck.example.ipynb`           | Milestones 1 to 4 on the SWE-bench Verified leaderboard                                                                            | Notebooks |
| `skill_vs_luck.example.py`              | Jupytext pair of `skill_vs_luck.example.ipynb`                                                                                     | Notebooks |
| `skill_vs_luck.more_models.ipynb`       | Study D: 3 more local models, greedy and at temperature 0.8                                                                        | Notebooks |
| `skill_vs_luck.more_models.py`          | Jupytext pair of `skill_vs_luck.more_models.ipynb`                                                                                 | Notebooks |
| `skill_vs_luck.one_run.ipynb`           | Study B: run luck estimated from 1 run, tuned on the public samples                                                                | Notebooks |
| `skill_vs_luck.one_run.py`              | Jupytext pair of `skill_vs_luck.one_run.ipynb`                                                                                     | Notebooks |
| `skill_vs_luck.one_run_test_set.ipynb`  | Study B: the tuned 1-run estimator tested once on our runs                                                                         | Notebooks |
| `skill_vs_luck.one_run_test_set.py`     | Jupytext pair of `skill_vs_luck.one_run_test_set.ipynb`                                                                            | Notebooks |
| `skill_vs_luck.records.ipynb`           | Study A: each SWE-bench Verified record tested against the one it broke                                                            | Notebooks |
| `skill_vs_luck.records.py`              | Jupytext pair of `skill_vs_luck.records.ipynb`                                                                                     | Notebooks |
| `skill_vs_luck.runs.ipynb`              | Run-to-run luck of the local agents on HumanEval+ and MBPP+                                                                        | Notebooks |
| `skill_vs_luck.runs.py`                 | Jupytext pair of `skill_vs_luck.runs.ipynb`                                                                                        | Notebooks |
| `skill_vs_luck.selection.ipynb`         | Milestone 4: the best of K runs and of K prompt variants, the corrections, and the reporting protocol (Studies F to H)             | Notebooks |
| `skill_vs_luck.selection.py`            | Jupytext pair of `skill_vs_luck.selection.ipynb`                                                                                   | Notebooks |
| `skill_vs_luck.temperatures.ipynb`      | Study C: the 3 old models at temperature 0.2 and in a second greedy run                                                            | Notebooks |
| `skill_vs_luck.temperatures.py`         | Jupytext pair of `skill_vs_luck.temperatures.ipynb`                                                                                | Notebooks |
| `skill_vs_luck.transfer.ipynb`          | Milestone 3 on HumanEval+ and MBPP+: transfer and effective size (Study E)                                                         | Notebooks |
| `skill_vs_luck.transfer.py`             | Jupytext pair of `skill_vs_luck.transfer.ipynb`                                                                                    | Notebooks |
| `skill_vs_luck_runs_utils.py`           | Prompts, Ollama client, scoring, and run-level analysis of the coding runs                                                         | Library   |
| `skill_vs_luck_utils.py`                | Intervals, ranks, paired tests, records, benchmark size, simulation, split-half, IRT, transfer, winner's curse and fixes, protocol | Library   |
| `test/test_fetch_coding_scores.py`      | Unit tests for `fetch_coding_scores.py`                                                                                            | Tests     |
| `test/test_fetch_evalplus_data.py`      | Unit tests for `fetch_evalplus_data.py`                                                                                            | Tests     |
| `test/test_notebooks.py`                | Smoke tests of the notebooks, and tests that freeze the results of Milestones 3 and 4                                              | Tests     |
| `test/test_skill_vs_luck_runs_utils.py` | Unit tests for `skill_vs_luck_runs_utils.py`                                                                                       | Tests     |
| `test/test_skill_vs_luck_utils.py`      | Unit tests for `skill_vs_luck_utils.py`                                                                                            | Tests     |
| `version.sh`                            | Print the package versions inside the image                                                                                        | Docker    |

- The files in `data/`:

  | File                                | Description                                                                   |
  | :---------------------------------- | :---------------------------------------------------------------------------- |
  | `evalplus_downloads.csv`            | Every EvalPlus file used, with its URL, size, and SHA-256                     |
  | `evalplus_leaderboard.csv`          | Snapshot of the EvalPlus leaderboard, the pass@1 of 125 models, on 2026-11-26 |
  | `provenance.json`                   | Sources, pinned commit, build time, and excluded submissions                  |
  | `swebench_verified_api_calls.csv`   | Agents x tasks matrix of model calls, for the submissions that report them    |
  | `swebench_verified_costs.csv`       | Agents x tasks matrix of dollar costs, for the same submissions               |
  | `swebench_verified_outcomes.csv`    | Agents x tasks matrix: 1 if the agent solved the task                         |
  | `swebench_verified_submissions.csv` | Metadata of every submission folder, with the reason for any exclusion        |
  | `swebench_verified_tasks.csv`       | The tasks: repo, human estimate of the fix time, issue date                   |

# Description of Executables

## `fetch_swebench_verified.py`

### What It Does

- Downloads:
  - The per-task results and the metadata of every SWE-bench Verified
    submission, at a pinned commit of `SWE-bench/experiments`
  - The task list, from the Hugging Face dataset `SWE-bench/SWE-bench_Verified`
- Drops the submissions that have no per-task results or whose per-task score
  does not match their reported score
  - The reason is saved for each one
- Writes the CSV files and `provenance.json` into `--dst_dir`
  - This is the only step that needs internet: the notebooks read the saved
    files

### Examples

- Rebuild the data in `data/`, replacing the current files:
  ```bash
  > ./fetch_swebench_verified.py --dst_dir data --overwrite
  ```

- Build the data from a different commit of `SWE-bench/experiments`:
  ```bash
  > ./fetch_swebench_verified.py --dst_dir data --overwrite --sha <commit>
  ```

- Print the debug logs while building:
  ```bash
  > ./fetch_swebench_verified.py --dst_dir data --overwrite -v DEBUG
  ```

## `fetch_coding_scores.py`

### What It Does

- Downloads the scores of the coding runs as 2 archives from the
  [Google Drive folder of the project](https://drive.google.com/drive/folders/1rgI_x5rfX9HXiP0kZUHfyq2Zg8WWuV4y),
  since they are not kept in the repo
  - Phase 1: the 88 `.csv.gz` files of `data/coding_runs/` and
    `data/coding_runs_pilot/`
  - Phase 2: the 18 `.csv.gz` files of `data/coding_runs_phase2/`
- Checks the SHA-256 of each archive, so the notebooks read the same scores
  they were run on
- Keeps the archives in `data/raw/`, and downloads one again only if it is
  missing or different
- Extracts only plain files, and only into the dirs of each archive

### Examples

- Download and extract the scores inside the container (needs internet):
  ```bash
  > ./docker_cmd.sh "cd /git_root/class_project/msml610/Fall2026/projects/UmdTask716_MSML610_Fall2026_Skill_vs_Luck_in_Agent_Benchmarks && python fetch_coding_scores.py"
  ```

## `fetch_evalplus_data.py`

### What It Does

- Downloads, from the GitHub releases of EvalPlus:
  - HumanEval+ v0.1.10 and MBPP+ v0.2.0, the versions that `evalplus==0.3.1`
    pins, into the folder where EvalPlus looks for them
  - The greedy samples of every model on both benchmarks, and 200 samples per
    task at temperatures 0.2 and 0.8 for 8 models on HumanEval+
- Skips the files that are already there, and writes the size and the SHA-256
  of every file into `data/evalplus_downloads.csv`
- Saves a snapshot of the EvalPlus leaderboard, the pass@1 of each model, into
  `data/evalplus_leaderboard.csv`, since the leaderboard keeps changing
- Only downloads the samples: their code runs in `score_coding_runs.py`, in the
  sandbox

### Examples

- Download everything into `data/raw/`:
  ```bash
  > ./fetch_evalplus_data.py
  ```

## `run_coding_agents.py`

### What It Does

- Asks 1 model, served by Ollama on the host, to solve HumanEval+ or MBPP+ tasks,
  with the prompt and the sampling settings of EvalPlus
- 1 cell is 1 task and 1 run, and gives both scaffolds:
  - `one_shot`: the first answer
  - `repair`: if the first answer fails an example shown in the prompt, the
    model sees the error and answers again (`--max_repairs` times)
- Saves each cell as soon as it is done into `data/raw/coding_runs/`, so a
  stopped job continues where it stopped
  - The seed of a cell depends only on the task and the run, so a cell can be
    generated again; the answer is often the same (8 of 9 replays in the pilot),
    but not always, since Ollama reuses its prompt cache between requests
  - The script saves the model version and the settings next to the answers,
    and checks them when a job continues

### Examples

- Generate 5 runs at temperature 0.8 for the first 10 HumanEval+ tasks:
  ```bash
  > ./docker_cmd_sandbox.sh --ollama "python run_coding_agents.py --benchmark humaneval --model opencoder:8b --temperature 0.8 --runs 5 --num_tasks 10"
  ```

## `score_coding_runs.py`

### What It Does

- Scores solutions on the EvalPlus base and plus tests, with the EvalPlus code
- Takes either the answers of `run_coding_agents.py` or a public EvalPlus sample
  zip
- Labels each failure: `syntax_error`, `load_error`, `missing_function`,
  `runtime_error`, `timeout`, `wrong_answer`, or `no_code`
- With `--n_checks 2`, scores each solution twice, to find flaky tests
- Writes 1 row per solution into `data/coding_runs/`

### Examples

- Score the answers of a model, twice each:
  ```bash
  > ./docker_cmd_sandbox.sh "python score_coding_runs.py --runs_file data/raw/coding_runs/humaneval/opencoder_8b_t0.8.jsonl --n_checks 2"
  ```

- Score the first 50 samples per task of a public zip:
  ```bash
  > ./docker_cmd_sandbox.sh "python score_coding_runs.py --zip_file data/raw/evalplus_samples/v0.1.0/code-llama-7b_temp_0.8.zip --max_samples 50"
  ```

## `run_coding_experiment.sh`

- Runs the whole coding-benchmark experiment of `preregistration.md`:
  - 1 greedy run, then 5 runs at temperature 0.8, for the 3 models on both
    benchmarks, adding 1 run at a time for all the models
  - Then scores every answer twice, and the public EvalPlus samples once
- Running it again continues the generation where it stopped, and scores
  everything again
- It took about 23 hours on a MacBook Air M5 (22 to generate the answers and 1
  to score everything), so keep the Mac awake and plugged in:
  ```bash
  > caffeinate -dimsu ./run_coding_experiment.sh
  ```

## `run_prompt_variants.sh`

- Runs the search over prompt variants of Study G in `preregistration_3.md`:
  1 greedy one-shot answer of DeepSeek Coder 1.3B to every task of HumanEval+
  and MBPP+, with each of the 20 prompt variants of `PROMPT_VARIANTS` in
  `skill_vs_luck_runs_utils.py`
  - HumanEval+ goes first for all the variants, so a stopped job leaves
    complete benchmarks
  - The answers go to `data/raw/coding_runs_variants/`, with the variant in
    each file name and in its settings
- `generate` makes the missing answers, so running it again continues where it
  stopped; `score` scores each answer twice into `data/coding_runs_variants/`
- Send 1 request at a time: Ollama can batch parallel requests, which can
  change greedy answers
- Generating took about 7.5 hours on a MacBook Air M5 (2.5 for HumanEval+, 5
  for MBPP+), and scoring about 18 minutes:
  ```bash
  > caffeinate -dimsu ./run_prompt_variants.sh generate
  > ./run_prompt_variants.sh score
  ```

## Docker Scripts

| Command                 | Description                                                    |
| :---------------------- | :------------------------------------------------------------- |
| `docker_build.sh`       | Build the image `gpsaggese/umd_project_skill_vs_luck`          |
| `docker_jupyter.sh`     | Start Jupyter Lab on `localhost:8888`, opened in this dir      |
| `docker_cmd.sh`         | Run 1 command in a new container, with the repo at `/git_root` |
| `docker_cmd_sandbox.sh` | Run 1 command that executes model-generated code               |
| `docker_bash.sh`        | Open a shell in a new container                                |
| `docker_exec.sh`        | Open a shell in the running container                          |
| `docker_clean.sh`       | Remove the image                                               |

- The template scripts are documented in
  `class_project/project_template/docker_scripts.README.md`
- `docker_cmd_sandbox.sh` differs from `docker_cmd.sh` in these ways:
  - The container has no network, unless the first argument is `--ollama`,
    which turns the network on so that it can reach the Ollama server on the host
  - The repo is read-only, except for this project's dir
  - It starts in this project's dir, and points EvalPlus to the datasets in
    `data/raw/`, so scoring works offline

# Description of Workflows

## Run the Notebooks

- Build the image, download the scores of the coding runs that the notebooks
  read (see `fetch_coding_scores.py`), and start Jupyter, from this dir:
  ```bash
  > ./docker_build.sh
  > ./docker_cmd.sh "cd /git_root/class_project/msml610/Fall2026/projects/UmdTask716_MSML610_Fall2026_Skill_vs_Luck_in_Agent_Benchmarks && python fetch_coding_scores.py"
  > ./docker_jupyter.sh
  ```

- Open `http://localhost:8888`, then run `skill_vs_luck.API.ipynb` first,
  `skill_vs_luck.example.ipynb` second, `skill_vs_luck.records.ipynb` (the
  SWE-bench records) third, `skill_vs_luck.runs.ipynb` (the local runs)
  fourth, and `skill_vs_luck.one_run.ipynb` (run luck from 1 run) fifth

- Then run the 3 notebooks of phase 2, which read the scores in
  `data/coding_runs_phase2/`: `skill_vs_luck.one_run_test_set.ipynb` (the test
  of Study B), `skill_vs_luck.temperatures.ipynb` (Study C), and
  `skill_vs_luck.more_models.ipynb` (Study D)

- Then `skill_vs_luck.transfer.ipynb` (Milestone 3 on HumanEval+ and MBPP+,
  Study E), and last `skill_vs_luck.selection.ipynb` (Milestone 4, Studies F to
  H, about 6 minutes), which also reads the scores of the prompt variants in
  `data/coding_runs_variants/`, made by `run_prompt_variants.sh`

## Report a Leaderboard with the Protocol

- `report_agent_scores()` in `skill_vs_luck_utils.py` is the reporting
  protocol of the project; it takes an agents x tasks matrix of 0/1 outcomes
  and gives, for each agent, best first:
  - The score with its 95% bootstrap interval
  - The 95% rank interval and the chance of coming out first
  - The gap to the next agent, the Holm-adjusted p-value of the paired test
    against it, and the number of tasks that would resolve that gap at 80%
    power
  - For the leader, the score corrected for having been picked as the best,
    by default with the paired rule that Study H recommends

- The paired rule assumes that the candidates are about equal, as variants of
  1 agent are; on a leaderboard of very different agents no correction passed
  (`skill_vs_luck.selection.ipynb`, Cell 4.3), so report there the drop of the
  leader on held-out halves of the tasks next to its score, e.g., the `none`
  row of `summarize_corrections()` on the trials of
  `simulate_variant_selection(outcomes, outcomes.shape[0])`

- E.g., on the SWE-bench Verified snapshot, inside the container:
  ```python
  import skill_vs_luck_utils as utils

  outcomes, submissions, tasks = utils.load_swebench_dataset("data")
  report = utils.report_agent_scores(outcomes, correction="paired")
  report.head(15)
  ```

## Run a Notebook as a Script

- Run the paired `.py` file of a notebook from top to bottom inside the
  container:
  ```bash
  > ./docker_cmd.sh "cd /git_root/class_project/msml610/Fall2026/projects/UmdTask716_MSML610_Fall2026_Skill_vs_Luck_in_Agent_Benchmarks && python skill_vs_luck.example.py"
  ```

## Run the Unit Tests

- Run the tests inside the container:
  ```bash
  > ./docker_cmd_sandbox.sh "python -m pytest test/test_skill_vs_luck_utils.py test/test_skill_vs_luck_runs_utils.py test/test_fetch_coding_scores.py test/test_fetch_evalplus_data.py"
  ```

- Pass the test files explicitly
  - `class_project/msml610/conftest.py` stops `pytest` from collecting tests in
    a directory under `msml610`

- Run the tests of the notebooks, after downloading the scores:
  ```bash
  > ./docker_cmd_sandbox.sh "python -m pytest test/test_notebooks.py -m slow"
  ```
  - Each experiment notebook starts with a `config` dict that holds all its
    parameters; the smoke tests (`slow`, 2 to 16 seconds each, about 1 minute
    in all) run every notebook with a smaller `config`
  - The `superslow` tests (`-m superslow`, about 6 minutes) run the notebooks
    of Milestones 3 and 4 with their full `config`, and check that the
    pre-registered results do not change
  - A test is skipped when the scores it needs are not downloaded

## Run the Coding-Benchmark Experiment

- Install Ollama on the host and pull the 3 models, since Docker on a Mac cannot
  use the GPU:
  ```bash
  > brew install --cask ollama-app
  > ollama pull deepseek-coder:1.3b-instruct-q4_K_M
  > ollama pull codeqwen:7b-chat-v1.5-q4_K_M
  > ollama pull opencoder:8b
  ```

- Build the image, download the EvalPlus data, and run the experiment:
  ```bash
  > ./docker_build.sh
  > ./docker_cmd.sh "cd /git_root/class_project/msml610/Fall2026/projects/UmdTask716_MSML610_Fall2026_Skill_vs_Luck_in_Agent_Benchmarks && python fetch_evalplus_data.py"
  > caffeinate -dimsu ./run_coding_experiment.sh
  ```

- Then the 20 prompt variants of Milestone 4 (see `run_prompt_variants.sh`)

## Rebuild the Data

- Open a shell in the container and run the fetch script (needs internet):
  ```bash
  > ./docker_bash.sh
  > cd /git_root/class_project/msml610/Fall2026/projects/UmdTask716_MSML610_Fall2026_Skill_vs_Luck_in_Agent_Benchmarks
  > ./fetch_swebench_verified.py --dst_dir data --overwrite
  ```
