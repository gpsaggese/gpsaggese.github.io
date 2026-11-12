# Milestone 1: Checkpoint and Resume on Chess

- Dates: 2026-11-10 to 2026-11-11
- Tasks: the five pilot positions in `data/tasks.json` (`krk-m2-1`,
  `krk-m3-1`, `krk-m4-1`, `krk-m6-1`, `krk-m8-1`), 8 spare half-moves each
- Two policies, both Claude Haiku 5.5 (`anthropic/claude-haiku-5-5`), no
  sampling parameters:

| Policy | Config | Results | Run name |
| :----- | :----- | :------ | :------- |
| Thinking off | `config/chess_haiku.yaml` | `results/m1/` | `m1` |
| Thinking on (adaptive, default effort) | `config/chess_haiku_think.yaml` | `results/m1_think/` | `m1_think` |

- Total API cost: $2.59 of the $20 cap ($0.12 thinking off, $2.47 thinking on)

## Summary

- **Milestone 1 is complete with thinking on.** Thinking off was tried first,
  as planned, but Haiku then almost never wins, so there was no checkpoint for
  the A/A check; thinking on is the policy from here on
- **Checkpoint and resume works.** Every replay check passes at every
  checkpoint, one step ahead and to the end of the game: 37/37 (thinking off)
  and 62/62 (thinking on)
- **Pilot:** thinking off won 0/10, thinking on won 3/10
- **Noise floor:** at two checkpoints with `V` near 0.5, two groups of 50
  continuations under the same policy differ by 0.10 and 0.14; both 95%
  intervals contain 0 (half-width about 0.19) and neither Fisher test is
  significant. So with 50 vs 50, a difference in `V` below about 0.19 cannot be
  told apart from noise
- **Cost per continuation:** about $0.005 with thinking on

## Stage A: Pilot

- Thinking off: 10 runs, 0 won, 0 infrastructure errors, $0.006

| Task | Run 1 | Run 2 | Win lost at White move |
| :--- | :---- | :---- | :--------------------- |
| krk-m2-1 | insufficient material | insufficient material | 1, 1 |
| krk-m3-1 | insufficient material | insufficient material | 1, 1 |
| krk-m4-1 | insufficient material | insufficient material | 1, 1 |
| krk-m6-1 | fifty-move rule | insufficient material | 2, 2 |
| krk-m8-1 | insufficient material | insufficient material | 3, 3 |

- Thinking on: 10 runs, 3 won, 0 infrastructure errors, $0.05

| Task | Run 1 | Run 2 | Win lost at White move |
| :--- | :---- | :---- | :--------------------- |
| krk-m2-1 | **checkmate** | **checkmate** | - |
| krk-m3-1 | fifty-move rule | fifty-move rule | 2, 2 |
| krk-m4-1 | fifty-move rule | insufficient material | 5, 1 |
| krk-m6-1 | fifty-move rule | **checkmate** | 5, - |
| krk-m8-1 | insufficient material | fifty-move rule | 3, 5 |

- "Insufficient material" means Black captured the rook
- Thinking off, Haiku leaves the rook undefended next to Black's king in 9 of 10
  games (e.g. `krk-m4-1`: Rg3+ with Black's king on g2, while its reasoning
  calls the rook "safe"). The tapes show it sees a correct board, FEN and list
  of legal moves, so these are model errors, not a harness bug
- Thinking on, the main failure changes: 5 of 7 failures are the fifty-move rule
  (the kings shuffle without progress, e.g. both `krk-m3-1` runs repeat Kc3,
  Kd3, Ke3 while Black's king goes back and forth), and 2 are a lost rook
- Both runs of a task often play the same first moves, so the policy is close
  to deterministic at the start, and diverges later

## Replay Checks

- Every checkpoint that has a next model call, no API calls
- One step ahead, from a restored checkpoint with the recorded reply:

| Check | Thinking off | Thinking on |
| :---- | :----------- | :---------- |
| Model input hash equals the recorded one | 37/37 | 62/62 |
| Command output equals the recorded one | 37/37 | 62/62 |
| Environment after the step equals the next checkpoint | 37/37 | 62/62 |
| Conversation after the step equals the next checkpoint | 37/37 | 62/62 |

- To the end, replaying the rest of the tape:

| Check | Thinking off | Thinking on |
| :---- | :----------- | :---------- |
| Same moves | 37/37 | 62/62 |
| Same outcome | 37/37 | 62/62 |
| Same final environment | 37/37 | 62/62 |
| Replay tape hash chain intact | 37/37 | 62/62 |
| Source tape hash chains intact | 10/10 | 10/10 |

- With thinking on, the conversation sent back to the model holds each earlier
  reply's text only, without its thinking blocks (the API allows this; the
  model then answers without its earlier reasoning). The thinking is kept on
  the tape, so replays stay exact

## Screening for Mixed Checkpoints

- Candidates: up to 4 checkpoints per run where the position is still won
  (before the pivot); 10 free continuations each

| Result | Thinking off | Thinking on |
| :----- | :----------- | :---------- |
| Candidates (forks) | 16 (160) | 30 (300) |
| `V` in [0.3, 0.7] | 0 | 12 |
| Never won (0/10) | 15 | 9 |
| Always won (10/10) | 0 | 3 |
| Other | 1 (2/10) | 6 |
| Cost | $0.11 | $1.47 |

- Thinking on, by spare half-moves left at the checkpoint (beyond perfect
  play):

| Spare half-moves | Checkpoints | Wins | 95% Wilson interval |
| :--------------- | :---------- | :--- | :------------------ |
| 0 | 3 | 0/30 | [0.00, 0.11] |
| 2 | 3 | 9/30 | [0.17, 0.48] |
| 4 | 8 | 30/80 | [0.28, 0.48] |
| 6 | 4 | 18/40 | [0.31, 0.60] |
| 8 | 12 | 44/120 | [0.29, 0.46] |

  - With no spare moves (only perfect play still wins) Haiku never wins; with
    any slack, `V` is about 0.3 to 0.45
- Consistency check for free: the two runs of each task share a start position,
  screened separately: 7 and 8 wins (`krk-m2-1`), 2 and 3 (`krk-m3-1`), 3 and 3
  (`krk-m4-1`), 2 and 1 (`krk-m6-1`), 2 and 0 (`krk-m8-1`), all within noise
- 0/10 means "no recovery observed under this policy" (95% interval [0, 0.28]),
  not "impossible"

## A/A Noise Floor

- The two screened checkpoints with `V` closest to 0.5, from different tasks;
  50 continuations per group, groups interleaved when run and scored, the same
  policy and action in both groups

| Checkpoint | Screen `V` | Group A | Group B | Gap | 95% CI (Newcombe) | Fisher p |
| :--------- | :--------- | :------ | :------ | :-- | :---------------- | :------- |
| `krk-m4-1`, after step 4 | 0.5 | 26/50 | 21/50 | +0.10 | [-0.09, +0.28] | 0.42 |
| `krk-m2-1`, after step 1 | 0.5 | 19/50 | 26/50 | -0.14 | [-0.32, +0.05] | 0.23 |

- Both intervals contain 0 and neither test is significant: no false alarm
- 0 infrastructure errors in 200 continuations; cost $0.96
- Width of the 95% interval for a difference in `V` near 0.5, by continuations
  per group (this sets how many continuations later stages need):

| Continuations per group | 10 | 20 | 30 | 50 |
| :---------------------- | :- | :- | :- | :- |
| Half-width | 0.37 | 0.28 | 0.24 | 0.19 |

## Consequences for the Next Stages

- The plan's Stage D sampling (20-30 continuations per checkpoint) resolves
  differences in `V` of about 0.25-0.30; smaller repair gains are reported as
  "not resolved", or sampled more at the checkpoints that matter
- At about $0.005 per continuation, a fully sampled profile of a 10-step run at
  25 continuations per checkpoint costs about $1.25, so Stages B-D fit well
  inside the first $150 batch
- Thinking off stays available as a planted-failure control (it hangs the rook
  early and reliably)

## Reproduce

### Environment

- Code: the `caaf_*.py` files of this PR, unchanged since the runs (last edited
  2026-11-10 21:51 local time, first run 22:55). Every checkpoint records the
  base commit (`9e1391ed`) with `harness_dirty: true`, since the runs were made
  before the code was committed
- Python 3.12 with the pinned `requirements.txt`: `mini-swe-agent==2.4.6`,
  `litellm==1.104.2`, `chess==1.11.2`, `pyyaml==6.0.3`
  - Or the project's Docker image (`./docker_build.sh`), Python 3.12.15, whose
    package versions are in `docker_build.version.log`
- `ANTHROPIC_API_KEY` in the environment, from a file outside the repo (e.g.
  `~/.cf-m1.env` with one line `ANTHROPIC_API_KEY=...`); never committed
- No Docker needed at run time; outbound HTTPS to the Anthropic API only

### Tablebase

- Syzygy KRvK tables from `https://tablebase.lichess.ovh/tables/standard/`
  (`3-4-5-wdl/KRvK.rtbw`, `3-4-5-dtz/KRvK.rtbz`), downloaded and checked
  against lichess's published list by `data/fetch_tablebases.sh`:

| File | sha256 |
| :--- | :----- |
| `KRvK.rtbw` | `386fbde73308e49a4207836922c68b30b664e83c5a37f7fa37305a15cd16f2f1` |
| `KRvK.rtbz` | `cab59f42e75c2a25da3939231850a79fa838593fff5c28d7772da92062ed965a` |

- The same two hashes are stored in every checkpoint (`tablebase_sha256`)

### Tasks and Seeds

- `python caaf_m1.py tasks` writes `data/tasks.json`: for difficulty bin `i`
  (0-4, mate in 2, 3, 4, 6, 8 moves, i.e. DTZ 3, 5, 7, 11, 15 plies), 5 positions
  from `start_positions(5, seed=677 + i, dtz, slack_plies=8)`: the first 4 are
  tasks, the 5th a reserve (reserves kept for the first 4 bins)
- Checked on 2026-11-11: regenerating in a fresh Python 3.12 environment gives a
  byte-identical file (sha256
  `d73e4d03b5e87aa85bfbbe51bd1335905727a7600afe4e1efe440893e0908710`)
- Pilot positions (White to move; the halfmove clock leaves 8 spare half-moves):

| Task | Seed | Mate in | Start FEN |
| :--- | :--- | :------ | :-------- |
| `krk-m2-1` | 677 | 2 | `4R3/8/8/8/7K/8/8/7k w - - 89 1` |
| `krk-m3-1` | 678 | 3 | `8/8/8/8/1R1K4/8/8/3k4 w - - 87 1` |
| `krk-m4-1` | 679 | 4 | `8/8/8/8/8/3R4/4K1k1/8 w - - 85 1` |
| `krk-m6-1` | 680 | 6 | `4K3/k7/8/6R1/8/8/8/8 w - - 81 1` |
| `krk-m8-1` | 681 | 8 | `8/8/8/8/4K3/k7/8/7R w - - 77 1` |

### Model Settings

| Setting | Thinking on (`chess_haiku_think.yaml`) | Thinking off (`chess_haiku.yaml`) |
| :------ | :------------------------------------- | :-------------------------------- |
| Model | `anthropic/claude-haiku-5-5` via litellm | same |
| Thinking | `{type: adaptive}`, default effort | `{type: disabled}` |
| `max_tokens` | 16000 (thinking counts toward it) | 2000 |
| Sampling | no temperature, top-p, top-k or seed (Haiku 5.5 takes no seed and rejects non-default sampling) | same |
| Prompt caching | `set_cache_control: default_end` | same |
| Earlier replies sent back | text only, no thinking blocks | text only |
| Step limit | White moves left before the fifty-move rule + 6 | same |
| Format errors | stop after 3 in a row | same |

- The system and task prompts are in the config files; every checkpoint stores
  the rendered prompts, the model id and the settings it ran with

### Commands

- Run each step from the repo root, in order; every block is one command to
  copy and run. Only steps 6, 8 and 9 call the API (about $2.50 and about an
  hour in total); every other step is free

1. Go to the project folder:
   ```bash
   cd class_project/msml610/Fall2026/projects/UmdTask677_MSML610_Fall2026_Counterfactual_Attribution_of_Agent_Failures
   ```
2. Create a Python 3.12 environment with the pinned packages:
   ```bash
   python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
   ```
3. Download and check the tablebase files:
   ```bash
   bash data/fetch_tablebases.sh
   ```
4. Run the unit tests (11 tests, free, no API):
   ```bash
   .venv/bin/python -m pytest -q -c /dev/null test/test_caaf.py
   ```
5. Load the API key and pick the thinking-on setup (results go to
   `results/m1_think/`):
   ```bash
   set -a; source ~/.cf-m1.env; set +a; export CAAF_CONFIG=chess_haiku_think.yaml CAAF_RUN=m1_think
   ```
6. Play the 10 pilot games (5 tasks x 2):
   ```bash
   .venv/bin/python caaf_m1.py pilot --tasks 5 --runs 2 --workers 8
   ```
7. Run the replay checks (no API calls, free):
   ```bash
   .venv/bin/python caaf_m1.py replay --workers 8
   ```
8. Screen: 30 checkpoints x 10 = 300 restarts:
   ```bash
   .venv/bin/python caaf_m1.py screen --per-run 4 --n 10 --workers 8
   ```
9. Run the A/A test: 2 checkpoints x 50 vs 50:
   ```bash
   .venv/bin/python caaf_m1.py aa --checkpoints 2 --n 50 --workers 8
   ```
10. Write the summary:
    ```bash
    .venv/bin/python caaf_m1.py report
    ```

- Thinking off (`results/m1/`): the same with
  `CAAF_CONFIG=chess_haiku.yaml CAAF_RUN=m1`; `aa` stops there with "only 0
  mixed checkpoints"
- What each step does:
  - `pilot`: 2 fresh games for each of the 5 pilot tasks, a checkpoint after
    every move
  - `replay`: for every checkpoint with a next model call, restore it, feed the
    recorded reply, and compare one step ahead and to the end of the game; no
    API calls
  - `screen`: up to 4 still-won checkpoints per game, evenly spaced, 10 free
    continuations each (300 with thinking on)
  - `aa`: the 2 screened checkpoints with `V` in [0.3, 0.7] closest to 0.5, from
    different tasks; 50 + 50 continuations, A and B alternating in the job queue
  - `report`: `results/<run>/summary.json`

### What Reproduces Exactly

- **Replay checks: exactly**, given the checkpoint store. Checked on 2026-11-11:
  rerunning `replay` on a copy of `runs/m1_think`, with no API key set, gives
  62/62 again and adds no row to the ledger
- **Task positions: exactly**, from the seeds (see above)
- **Candidate and A/A checkpoint choice: exactly**, given the pilot store and
  the screening results (both rules are deterministic)
- **Pilot, screening and A/A outcomes: statistically, not exactly.** Haiku is
  sampled and takes no seed, so a rerun plays new games; win rates should agree
  within the noise measured here (about +/-0.19 at 50 continuations per group,
  +/-0.37 at 10)
- The checkpoint stores (`runs/m1`, `runs/m1_think`, 218 MB) are not in git;
  they are on Google Drive (next section), so the replay checks can be rerun
  on the original runs

### Original Runs on Google Drive

- Folder:
  [MSML610_Fall2026_Counterfactual_Attribution_of_Agent_Failures](https://drive.google.com/drive/folders/15c2YBv6f9FU5enMjIqUqoKwgKxpwcG6T)
  (view access on request; to move to the class folder)
- It holds one archive of `runs/` (both stores and the spending ledger) split
  into 10 parts, `part00` to `part09`, and a `README.txt`
  - Joined archive: 68,983,596 bytes (66 MB), 218 MB unpacked, sha256
    `d63ef2d94a695280cffa1896bbe201e0f2d3f79def9b90751801c6f09e3a7172`
- To get the original runs, download the 10 parts into this project folder and
  run each step from there:

1. Join the parts:
   ```bash
   cat MSML610_Fall2026_Counterfactual_Attribution_of_Agent_Failures.M1_runs.tar.gz.part* > MSML610_Fall2026_Counterfactual_Attribution_of_Agent_Failures.M1_runs.tar.gz
   ```
2. Check the archive (the output must start with `d63ef2d9`):
   ```bash
   shasum -a 256 MSML610_Fall2026_Counterfactual_Attribution_of_Agent_Failures.M1_runs.tar.gz
   ```
3. Unpack it, which recreates `runs/` (git-ignored, so it cannot be committed by
   accident):
   ```bash
   tar -xzf MSML610_Fall2026_Counterfactual_Attribution_of_Agent_Failures.M1_runs.tar.gz
   ```
4. Rerun the 62 replay checks on the original games (no API calls, free):
   ```bash
   CAAF_CONFIG=chess_haiku_think.yaml CAAF_RUN=m1_think .venv/bin/python caaf_m1.py replay
   ```

### Running It With Your Own API Key

- Everything a run writes stays in your clone: the stores in `runs/<run name>/`
  and the ledger in `runs/api_usage.jsonl` (both git-ignored), and the summaries
  in `results/<run name>/` (in git once you commit them)
- Use your own run name, e.g. `CAAF_RUN=m1_think_<your name>`: with
  `CAAF_RUN=m1_think` a run overwrites the published `results/m1_think/` in your
  clone
- The cap in the config ($20) guards the ledger on one machine, so each person
  has their own cap; it is not a shared team budget
- Your API key does not change the experiment: the model, its settings, the
  prompts, the tasks and the tablebase are fixed by the config, the seeds and
  the checksums, and every checkpoint records them
- Your games will still differ from ours, because Haiku is sampled without a
  seed: every run, by anyone, plays new games. Expect:
  - **The same** replay checks (62/62 on our stores, and every check passing on
    your own) and the same task positions
  - **Different** individual games, moves and checkpoints, so different
    screening candidates and A/A checkpoints
  - **Statistically the same** win rates: pilot, screening and A/A numbers
    should agree with ours within the noise above (about +/-0.19 at 50
    continuations per group, +/-0.37 at 10)
- A gap larger than the noise means something changed: check the model id,
  the config and the tablebase hashes recorded in your checkpoints against the
  ones here. If they all match, the remaining explanation is a change on the
  model provider's side
- To share your runs, commit `results/<run name>/` on a branch, and upload
  `runs/<run name>/` to Drive the same way (one archive, split into parts of
  about 7 MB if the upload is capped at 10 MB)
- For Milestone 2, analyse the shared stores rather than fresh pilots, so that
  everyone works on the same games

### Spending Ledger

- `runs/api_usage.jsonl`: one row per model call (time, model, uncached input,
  cache writes, cache reads, output tokens, cost), appended under a file lock by
  every thread; no call starts once the total reaches the cap minus a $0.25
  margin
- Cap $20 (both configs); prices $0.10 input and $0.50 output per million tokens,
  cache writes x1.25 and cache reads x0.1 the input price
- Per-phase totals, also in `results/ledger_summary.json` (the ledger itself is
  on Google Drive with the stores):

| Phase | Calls | Input tokens | Output tokens | Cost |
| :---- | :---- | :----------- | :------------ | :--- |
| Thinking off: pilot | 37 | 63,965 | 6,303 | $0.006 |
| Thinking off: screen | 757 | 1,795,379 | 122,012 | $0.114 |
| Thinking on: pilot | 62 | 135,429 | 96,186 | $0.053 |
| Thinking on: screen | 1,661 | 4,315,756 | 2,709,000 | $1.467 |
| Thinking on: A/A | 828 | 1,945,858 | 1,809,714 | $0.955 |
| Replay checks (both) | 0 | 0 | 0 | $0 |
| **Total** | **3,345** | **8,256,387** | **4,743,215** | **$2.595** |
