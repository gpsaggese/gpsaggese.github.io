# Counterfactual Attribution of Agent Failures: Progress

## Status

| Milestone | Target date | Status | PR |
| :-------- | :---------- | :----- | :- |
| M1: Checkpoint and Resume | - | Done (2026-11-11), not yet merged | PR 2 (to open) |
| M2: Blame Profiles | December 1, 2026 (project) | Not started | - |
| M3: Taxonomy and Transfer | December 1, 2026 (project) | Not started | - |
| M4: Cheap Surrogate and Feedback Loop | December 1, 2026 (project) | Not started | - |

- Tracking: GitHub project board and milestones on the fork
  (`linuteresa/gpsaggese.github.io`, issues #1-#16)

## Built

- Checkpoint store, tapes and spending ledger: `caaf_store.py`
- Board environment, tablebase referee and start positions: `caaf_chess.py`
- Recording agent, `record_run`, `fork` and `run_to_end`: `caaf_agent.py`
- Parallel forks and statistics: `caaf_utils.py`
- Milestone 1 experiments: `caaf_m1.py`
- Unit tests, no API calls: `test/test_caaf.py` (11 tests)
- What each file holds and how to run it: `README.md`

## Findings

1. Checkpoint and resume is exact: every replay check passes at every
   checkpoint, one step ahead (model input hash, command output, environment and
   conversation after the step) and to the end of the game (moves, outcome,
   final environment)
   - 37/37 checkpoints (thinking off), 62/62 (thinking on)
   - Data: `results/m1/replay.json`, `results/m1_think/replay.json`
2. With thinking off, Haiku 5.5 won 0 of 10 pilot games; in 9 it left the rook
   undefended next to Black's king within its first 3 moves
   - Data: `results/m1/pilot.json`
3. With thinking on, Haiku 5.5 won 3 of 10 pilot games; 5 of the 7 failures are
   the fifty-move rule (no progress), 2 a lost rook
   - Data: `results/m1_think/pilot.json`
4. With thinking on, 12 of 30 still-won checkpoints have `V` in [0.3, 0.7] (10
   continuations each); with no spare half-moves `V` is 0 (0/30), with any
   slack about 0.3-0.45
   - Data: `results/m1_think/screen.json`
5. The A/A noise floor is a half-width of about 0.19 in `V` at 50 vs 50
   continuations: the gaps at two checkpoints near `V` = 0.5 are +0.10 and
   -0.14, both 95% intervals contain 0, Fisher p = 0.42 and 0.23
   - At 20 and 30 continuations per group the half-width is 0.28 and 0.24
   - Data: `results/m1_think/aa.json`
6. A continuation costs about $0.005 with thinking on; Milestone 1 cost $2.59 in
   total
   - Data: `results/m1_think/summary.json`

## Next

1. Get PR 2 (#917) reviewed and merged
2. Create the research Google Doc
3. Stage B: `V` curves on the pilot failures and the direct-arm check of the
   shortcut (`04-plan.md`)
4. Stage C: the remaining 15 tasks

## History

| Date | Done |
| :--- | :--- |
| 2026-11-10 | Task changed to chess endgames, see [`D1`](03-decisions.md#d1-chess-endgames-replace-swe-rebench-as-the-task); runner, store, tests and the Milestone 1 experiments written; unit tests 11/11 and a mock-model dry run pass |
| 2026-11-10 | Pilot, replay checks and screening with thinking off: 0/10 won, 37/37 replay checks, no mixed checkpoint, so no A/A check ($0.12) |
| 2026-11-10 | GitHub project, milestones M1-M4 and issues #1-#16 created on the fork |
| 2026-11-11 | Policy changed to thinking on, see [`D4`](03-decisions.md#d4-haiku-55-plays-with-thinking-on); pilot 3/10 won, 62/62 replay checks, 12 mixed checkpoints, A/A within noise at both checkpoints ($2.47); Milestone 1 complete, write-up in `milestone1.md` |
| 2026-11-11 | Docs renamed to the class layout (`04-plan.md`); `03-decisions.md` and this log added; SWE-based Milestone 1 handoff deleted |
| 2026-11-11 | Docker setup changed for chess: Docker CLI, host-socket mount, SWE evaluator and `requirements.eval.txt` removed; the image builds natively on arm64 and the 11 unit tests pass inside it |
| 2026-11-11 | Reproduction notes added to `milestone1.md` (environment, tablebase hashes, seeds and positions, model settings, commands, ledger by phase in `results/ledger_summary.json`); checked: tasks regenerate byte-identically from the seeds and the 62 replay checks pass again on a copy of the store with no API calls |
| 2026-11-11 | PR 2 opened (#917) and merged with upstream master; `docker_jupyter.sh` tested on the new image; checkpoint stores and ledger archived (66 MB, sha256 recorded) and put on the project's UMD Google Drive folder, one subfolder per run, with instructions in `milestone1.md` for getting them and for running with your own API key |

## Open Work

- PR #917: waiting for review
- Research Google Doc: not started (PR 3)
