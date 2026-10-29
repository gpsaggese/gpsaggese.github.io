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
  - Our own runs of 3 open coding models on HumanEval+ and MBPP+, served by
    Ollama, with 2 scaffolds and 5 runs per task
  - The code that EvalPlus generated in 2023 with many models, including 200
    samples per task at 2 temperatures, scored again here

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
- The analysis and the plots are in `skill_vs_luck.example.ipynb`
- The scorer of the coding benchmarks matches the published EvalPlus scores to
  within 2 tasks, on the 5 public greedy sets checked
  - E.g., on the public greedy samples of GPT-4-Turbo it gives 81.7 on
    HumanEval+ and 73.5 on MBPP+, against the published 81.7 and 73.3
- The predictions for the local runs are in `preregistration.md`, written before
  the full runs and committed before any of them was scored

## Status of the Milestones

| Milestone                     | Status                                                         |
| :---------------------------- | :------------------------------------------------------------- |
| 1. Outcome matrix             | SWE-bench Verified done; HumanEval+ and MBPP+ runs in progress |
| 2. Quantify luck              | Task-sampling part done; run-to-run part pre-registered        |
| 3. Persistence, transfer, IRT | Planned                                                        |
| 4. Selection effects          | Planned                                                        |

## Structure of the Dir

| Dir                 | Description                                                                        |
| :------------------ | :--------------------------------------------------------------------------------- |
| `data/`             | SWE-bench Verified snapshot, list of the EvalPlus downloads, and the coding scores |
| `data/coding_runs/` | Scores of the coding-benchmark solutions, from `score_coding_runs.py`              |
| `data/raw/`         | Downloads and raw model answers, rebuilt by the scripts and not committed          |
| `test/`             | Unit tests                                                                         |

## Description of Files

| File                                    | Description                                                                 | Cluster   |
| :-------------------------------------- | :-------------------------------------------------------------------------- | :-------- |
| `bashrc`, `etc_sudoers`                 | Shell and sudo configuration copied into the image                          | Docker    |
| `docker_*.sh`, `run_jupyter.sh`         | Docker scripts, copied unchanged from `class_project/project_template`      | Docker    |
| `docker_cmd_sandbox.sh`                 | Like `docker_cmd.sh`, for running model-generated code (see below)          | Docker    |
| `Dockerfile`                            | Python 3.12 slim image with the packages in the 2 requirements files        | Docker    |
| `fetch_evalplus_data.py`                | Download the EvalPlus datasets and the public EvalPlus samples              | Data      |
| `fetch_swebench_verified.py`            | Download the public SWE-bench Verified results into `data/`                 | Data      |
| `preregistration.md`                    | Design, analyses, and predictions of the local runs, written before them    | Docs      |
| `README.md`                             | This file                                                                   | Docs      |
| `requirements.eval.txt`                 | Pinned packages that the EvalPlus scorer needs                              | Docker    |
| `requirements.txt`                      | Pinned Python packages installed in the image                               | Docker    |
| `run_coding_agents.py`                  | Generate answers to HumanEval+ or MBPP+ tasks with a model served by Ollama | Runs      |
| `run_coding_experiment.sh`              | Run every model and score every answer                                      | Runs      |
| `score_coding_runs.py`                  | Score answers or public samples with the EvalPlus tests                     | Runs      |
| `skill_vs_luck.API.ipynb`               | Walkthrough of every function in `skill_vs_luck_utils.py`                   | Notebooks |
| `skill_vs_luck.API.py`                  | Jupytext pair of `skill_vs_luck.API.ipynb`                                  | Notebooks |
| `skill_vs_luck.example.ipynb`           | Milestones 1 and 2 on the SWE-bench Verified leaderboard                    | Notebooks |
| `skill_vs_luck.example.py`              | Jupytext pair of `skill_vs_luck.example.ipynb`                              | Notebooks |
| `skill_vs_luck_runs_utils.py`           | Prompts, Ollama client, scoring, and run-level analysis of the coding runs  | Library   |
| `skill_vs_luck_utils.py`                | Data loading, intervals, paired tests, benchmark size, simulation, plots    | Library   |
| `test/test_skill_vs_luck_runs_utils.py` | Unit tests for `skill_vs_luck_runs_utils.py`                                | Tests     |
| `test/test_skill_vs_luck_utils.py`      | Unit tests for `skill_vs_luck_utils.py`                                     | Tests     |
| `version.sh`                            | Print the package versions inside the image                                 | Docker    |

- The files in `data/`:

  | File                                | Description                                                                |
  | :---------------------------------- | :------------------------------------------------------------------------- |
  | `evalplus_downloads.csv`            | Every EvalPlus file used, with its URL, size, and SHA-256                  |
  | `provenance.json`                   | Sources, pinned commit, build time, and excluded submissions               |
  | `swebench_verified_api_calls.csv`   | Agents x tasks matrix of model calls, for the submissions that report them |
  | `swebench_verified_costs.csv`       | Agents x tasks matrix of dollar costs, for the same submissions            |
  | `swebench_verified_outcomes.csv`    | Agents x tasks matrix: 1 if the agent solved the task                      |
  | `swebench_verified_submissions.csv` | Metadata of every submission folder, with the reason for any exclusion     |
  | `swebench_verified_tasks.csv`       | The tasks: repo, human estimate of the fix time, issue date                |

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

## `fetch_evalplus_data.py`

### What It Does

- Downloads, from the GitHub releases of EvalPlus:
  - HumanEval+ v0.1.10 and MBPP+ v0.2.0, the versions that `evalplus==0.3.1`
    pins, into the folder where EvalPlus looks for them
  - The greedy samples of every model on both benchmarks, and 200 samples per
    task at temperatures 0.2 and 0.8 for 8 models on HumanEval+
- Skips the files that are already there, and writes the size and the SHA-256
  of every file into `data/evalplus_downloads.csv`
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
  - The model version and the settings are saved next to the answers, and are
    checked when a job continues

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
- It takes more than a day on a MacBook Air M5 (about 21 hours for the answers
  alone), so keep the Mac awake and plugged in:
  ```bash
  > caffeinate -dimsu ./run_coding_experiment.sh
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
    which lets it reach the Ollama server on the host
  - The repo is read-only, except for this project's dir
  - It starts in this project's dir, and points EvalPlus to the datasets in
    `data/raw/`, so scoring works offline

# Description of Workflows

## Run the Notebooks

- Build the image and start Jupyter, from this dir:
  ```bash
  > ./docker_build.sh
  > ./docker_jupyter.sh
  ```

- Open `http://localhost:8888`, then run `skill_vs_luck.API.ipynb` first and
  `skill_vs_luck.example.ipynb` second

## Run a Notebook as a Script

- Run the paired `.py` file of a notebook from top to bottom inside the
  container:
  ```bash
  > ./docker_cmd.sh "cd /git_root/class_project/msml610/Fall2026/projects/UmdTask716_MSML610_Fall2026_Skill_vs_Luck_in_Agent_Benchmarks && python skill_vs_luck.example.py"
  ```

## Run the Unit Tests

- Run the tests inside the container:
  ```bash
  > ./docker_cmd_sandbox.sh "python -m pytest test/test_skill_vs_luck_utils.py test/test_skill_vs_luck_runs_utils.py"
  ```

- Pass the test files explicitly
  - `class_project/msml610/conftest.py` stops `pytest` from collecting tests in
    a directory under `msml610`

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

## Rebuild the Data

- Open a shell in the container and run the fetch script (needs internet):
  ```bash
  > ./docker_bash.sh
  > cd /git_root/class_project/msml610/Fall2026/projects/UmdTask716_MSML610_Fall2026_Skill_vs_Luck_in_Agent_Benchmarks
  > ./fetch_swebench_verified.py --dst_dir data --overwrite
  ```
