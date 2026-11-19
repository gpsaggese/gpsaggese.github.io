# Counterfactual Attribution of Agent Failures

- Issue: [#677](https://github.com/gpsaggese/gpsaggese.github.io/issues/677)
- Plan: [`docs/04-plan.md`](docs/04-plan.md); decisions:
  [`docs/03-decisions.md`](docs/03-decisions.md); progress log:
  [`docs/07-progress.md`](docs/07-progress.md); Milestone 1 results:
  [`docs/milestone1.md`](docs/milestone1.md)
- Idea: when an agent fails, restart the run from a checkpoint before each step,
  let the same agent re-decide, and measure how much the chance of success
  changes; the steps whose re-decision helps most are the ones to blame

## The Task

- Agent: [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) 2.4.6,
  extended by subclassing, with Claude Haiku 5.5 (thinking on; thinking off
  was the first Milestone 1 policy, see `docs/milestone1.md`)
- Task: King and Rook vs. King endgames. The agent plays White through two
  commands, `play show` and `play move <move>`; Black plays the best tablebase
  defence. The agent wins if it checkmates before the fifty-move rule
- Ground truth: the Syzygy tablebase gives the exact value of every position,
  so every run is scored exactly and for free

## Files

| File | Contents |
| :--- | :------- |
| `caaf_store.py` | Checkpoint store (content-addressed blobs, SQLite manifest), hash-chained tapes, spending ledger |
| `caaf_chess.py` | Board program and sealed environment, tablebase values, referee, start positions |
| `caaf_agent.py` | Recording agent (two hooks, a checkpoint after every step) and `record_run`, `fork`, `run_to_end` |
| `caaf_utils.py` | Parallel forks, checkpoint lookups, statistics (Wilson, Newcombe, Fisher) |
| `caaf_m1.py` | Milestone 1 experiments: `tasks`, `pilot`, `replay`, `screen`, `aa`, `report` |
| `config/chess_haiku_think.yaml` | Model (thinking on), prompts, step limit and the budget cap |
| `config/chess_haiku.yaml` | The same with thinking off (first Milestone 1 policy) |
| `data/tasks.json` | The 20 tasks, 4 reserves and the 5-task pilot |
| `data/fetch_tablebases.sh` | Downloads and checks the KRvK tables (8 KB) |
| `results/m1_think/`, `results/m1/` | Experiment summaries, thinking on and off (the stores in `runs/` are not in git) |
| `test/test_caaf.py` | Unit tests; no API calls |

## What a Checkpoint Holds

| Half | Contents |
| :--- | :------- |
| Context | Rendered prompts, the full message list as sent, model id and settings, step and cost counters, agent config, harness commit, tablebase hashes |
| Environment | The game record (start position, moves, status) |
| Manifest | Checkpoint id, run, branch, step, parent, both hashes; branches with their parent checkpoint, intervention and outcome |
| Tape | One hash-chained line per model call (input hash, reply, usage) and per command (output, exit code) |

- Write order: context blob, environment blob, then the manifest row in one
  transaction; blobs are fsynced and renamed into place
- A fork restores both halves, checks their hashes, and starts a new branch;
  interventions: `free` (the model decides), `replay` (replies read from a
  tape), `force` (a planted reply)

## How to Run

- With Python 3.12 (a laptop, or a Zaratan node; no Docker needed):
  ```bash
  > python -m venv .venv && .venv/bin/pip install -r requirements.txt
  > bash data/fetch_tablebases.sh
  > .venv/bin/python -m pytest -q -c /dev/null test/test_caaf.py
  ```
- The experiments need `ANTHROPIC_API_KEY` in the environment; the ledger stops
  all calls at the cap in the config. `CAAF_CONFIG` picks the config and
  `CAAF_RUN` the run name (stores in `runs/<name>`, results in
  `results/<name>`):
  ```bash
  > set -a; source ~/.cf-m1.env; set +a
  > export CAAF_CONFIG=chess_haiku_think.yaml CAAF_RUN=m1_think
  > .venv/bin/python caaf_m1.py pilot
  > .venv/bin/python caaf_m1.py replay
  > .venv/bin/python caaf_m1.py screen
  > .venv/bin/python caaf_m1.py aa
  > .venv/bin/python caaf_m1.py report
  ```
- The same commands run in the project's Docker image (`./docker_build.sh`,
  then `./docker_bash.sh`)
