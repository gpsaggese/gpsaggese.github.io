# Counterfactual Attribution of Agent Failures: Decisions

## Task and Environment

### `D1` (Chess endgames replace SWE-rebench as the task)

- **Decision**: the agent plays King and Rook vs. King endgames as White and
  must checkmate before the fifty-move rule; the Syzygy tablebase scores every
  game
- **Why**: Zaratan has no Docker, and the SWE-rebench tasks need one container
  per task and an evaluator too slow to run thousands of continuations; the
  tablebase gives exact scores, instantly and for free
- **Alternatives**: SWE-rebench on a machine with Docker (no such machine, and
  too slow and costly to scale); other games without an exact scorer
- **Status**: accepted (2026-11-10)

### `D2` (The board runs inside the agent's process, without a container)

- **Decision**: the agent's only commands are `play show` and `play move
  <move>`; anything else is refused without running, and the environment's whole
  state is the game record (a small JSON document)
- **Why**: the environment is sealed by construction, its state is saved and
  restored exactly, and it runs in a plain virtualenv on a laptop or Zaratan
- **Alternatives**: a real shell in a container (needs Docker, and the state is
  a filesystem)
- **Status**: accepted (2026-11-10)

### `D3` (The tablebase is both the referee and Black's defender)

- **Decision**: Black plays the best tablebase defence (lowest value for White,
  then longest resistance, ties broken by the first move in UCI order)
- **Why**: Black is a fixed function of the position, so the environment is
  deterministic and every difference between continuations comes from the agent
- **Alternatives**: a chess engine (slower, and not exact in every position) or
  a random Black (adds noise that is not the agent's)
- **Status**: accepted (2026-11-10)

## Policy

### `D4` (Haiku 5.5 plays with thinking on)

- **Decision**: the policy is Claude Haiku 5.5 with adaptive thinking at the
  default effort (`config/chess_haiku_think.yaml`)
- **Why**: with thinking off, Haiku won 0 of 10 pilot games and 15 of 16
  still-won checkpoints never recovered, so no checkpoint had a mixed outcome
  for the A/A check; with thinking on it won 3 of 10, and 12 of 30 checkpoints
  had `V` in [0.3, 0.7] (`docs/milestone1.md`)
- **Alternatives**: a stronger model such as Sonnet 5.5 (several times the
  cost); the A/A check at a lower `V` (a weaker check)
- **Status**: accepted (2026-11-11); supersedes the thinking-off policy of the
  original plan, which stays as `config/chess_haiku.yaml`

### `D5` (Earlier replies are sent back as text, without thinking blocks)

- **Decision**: the conversation sent to the model keeps each earlier reply's
  text only; the thinking and the raw response are kept on the tape
- **Why**: the conversation in a checkpoint is then exactly what the model
  receives, so a restored run sends byte-identical input and replays are exact
  (62/62 replay checks); the API allows leaving out earlier thinking blocks
- **Alternatives**: sending the thinking blocks back (the conversation depends
  on signed, account-bound blocks, and the model's input is harder to check)
- **Status**: accepted (2026-11-11)

## Infrastructure

### `D6` (A shared ledger caps the API spending)

- **Decision**: every model call is checked against, and charged to, one
  file-locked ledger (`runs/api_usage.jsonl`); no call starts once the cap in
  the config is reached
- **Why**: hundreds of continuations run in parallel; a hard cap prevents a bug
  from spending the budget
- **Alternatives**: a limit per run (does not bound the total)
- **Status**: accepted (2026-11-10); cap $20 for Milestone 1, $2.59 spent

### `D7` (Experiment stores stay out of git)

- **Decision**: the checkpoint stores (`runs/`, 218 MB after Milestone 1) and the
  tablebase files (`data/syzygy/`, downloaded and checksummed by
  `data/fetch_tablebases.sh`) are git-ignored; the small JSON summaries in
  `results/` are committed; the stores go to the project's Google Drive dir
- **Why**: the class rules forbid binary and large files in the repo
- **Alternatives**: none
- **Status**: accepted (2026-11-11); uploaded to Google Drive on 2026-11-11
  (`docs/milestone1.md`, "Original Runs on Google Drive")
