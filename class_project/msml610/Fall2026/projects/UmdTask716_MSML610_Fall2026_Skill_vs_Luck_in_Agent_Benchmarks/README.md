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
- Everything runs locally in Docker, at no cost
  - The data is the public per-task results of every SWE-bench Verified
    submission, saved once in `data/`

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

## Status of the Milestones

| Milestone                     | Status                                                        |
| :---------------------------- | :------------------------------------------------------------ |
| 1. Outcome matrix             | SWE-bench Verified done; local HumanEval+ and MBPP+ runs next |
| 2. Quantify luck              | Task-sampling part done; run-to-run part needs the local runs |
| 3. Persistence, transfer, IRT | Planned                                                       |
| 4. Selection effects          | Planned                                                       |

## Structure of the Dir

| Dir     | Description                                                                       |
| :------ | :-------------------------------------------------------------------------------- |
| `data/` | Snapshot of the SWE-bench Verified results, built by `fetch_swebench_verified.py` |
| `test/` | Unit tests for `skill_vs_luck_utils.py`                                           |

## Description of Files

| File                               | Description                                                              | Cluster   |
| :--------------------------------- | :----------------------------------------------------------------------- | :-------- |
| `bashrc`, `etc_sudoers`            | Shell and sudo configuration copied into the image                       | Docker    |
| `docker_*.sh`, `run_jupyter.sh`    | Docker scripts, copied unchanged from `class_project/project_template`   | Docker    |
| `Dockerfile`                       | Python 3.12 slim image with the packages in `requirements.txt`           | Docker    |
| `fetch_swebench_verified.py`       | Download the public SWE-bench Verified results into `data/`              | Data      |
| `README.md`                        | This file                                                                | Docs      |
| `requirements.txt`                 | Pinned Python packages installed in the image                            | Docker    |
| `skill_vs_luck.API.ipynb`          | Walkthrough of every function in `skill_vs_luck_utils.py`                | Notebooks |
| `skill_vs_luck.API.py`             | Jupytext pair of `skill_vs_luck.API.ipynb`                               | Notebooks |
| `skill_vs_luck.example.ipynb`      | Milestones 1 and 2 on the SWE-bench Verified leaderboard                 | Notebooks |
| `skill_vs_luck.example.py`         | Jupytext pair of `skill_vs_luck.example.ipynb`                           | Notebooks |
| `skill_vs_luck_utils.py`           | Data loading, intervals, paired tests, benchmark size, simulation, plots | Library   |
| `test/test_skill_vs_luck_utils.py` | Unit tests for `skill_vs_luck_utils.py`                                  | Tests     |
| `version.sh`                       | Print the package versions inside the image                              | Docker    |

- The files in `data/`:

  | File                                | Description                                                                |
  | :---------------------------------- | :------------------------------------------------------------------------- |
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

## Docker Scripts

| Command             | Description                                                    |
| :------------------ | :------------------------------------------------------------- |
| `docker_build.sh`   | Build the image `gpsaggese/umd_project_skill_vs_luck`          |
| `docker_jupyter.sh` | Start Jupyter Lab on `localhost:8888`, opened in this dir      |
| `docker_cmd.sh`     | Run 1 command in a new container, with the repo at `/git_root` |
| `docker_bash.sh`    | Open a shell in a new container                                |
| `docker_exec.sh`    | Open a shell in the running container                          |
| `docker_clean.sh`   | Remove the image                                               |

- These are the template scripts, documented in
  `class_project/project_template/docker_scripts.README.md`

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
  > ./docker_cmd.sh "cd /git_root/class_project/msml610/Fall2026/projects/UmdTask716_MSML610_Fall2026_Skill_vs_Luck_in_Agent_Benchmarks && python -m pytest test/test_skill_vs_luck_utils.py"
  ```

- Pass the test file explicitly
  - `class_project/msml610/conftest.py` stops `pytest` from collecting tests in
    a directory under `msml610`

## Rebuild the Data

- Open a shell in the container and run the fetch script (needs internet):
  ```bash
  > ./docker_bash.sh
  > cd /git_root/class_project/msml610/Fall2026/projects/UmdTask716_MSML610_Fall2026_Skill_vs_Luck_in_Agent_Benchmarks
  > ./fetch_swebench_verified.py --dst_dir data --overwrite
  ```
