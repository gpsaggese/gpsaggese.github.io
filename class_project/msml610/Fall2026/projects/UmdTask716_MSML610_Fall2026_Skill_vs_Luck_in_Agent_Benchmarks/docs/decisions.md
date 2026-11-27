# Skill vs Luck in Agent Benchmarks Decisions

## Ranking

### D-1. Split tied ranks and tied first places

- **Decision**: tied agents share the average of the ranks they span, and an
  `n`-way tie for first gives each agent `1 / n` of the first place
- **Why**: the top 2 agents both solve exactly 396 of the 500 tasks, so the
  rule decides the chance of coming out first for both; splitting keeps the
  chances summing to 1 in every resample
- **Alternatives**: the best shared rank, like the leaderboard (counts a 2-way
  tie as 2 wins), or breaking ties at random (adds noise for no gain)
- **Status**: accepted

## Item response theory

### D-2. Fit the 2PL model with SVI

- **Decision**: fit the full 173 x 500 matrix with variational inference
  (`AutoNormal` guide), and keep NUTS as an option
- **Why**: SVI takes about 4 seconds on a laptop CPU, and on a 40 x 150 subset
  it matches NUTS (ability correlation 0.997, ratio of posterior stds 0.98)
- **Alternatives**: NUTS with 1 chain on the full matrix (slower, and not run
  on it yet), a Rasch model (no discrimination, so no effective size)
- **Status**: proposed

### D-3. Measure the effective size with the participation ratio

- **Decision**: the effective size is `(sum I_i)^2 / sum I_i^2`, with `I_i`
  the Fisher information of task `i` averaged over the agents of interest
- **Why**: it is the number of tasks when all carry the same information and
  1 when a single task carries all of it, like the effective number of trades
  used in #733; the count of tasks with mixed outcomes is reported next to it
- **Alternatives**: the number of tasks with a discrimination above a
  threshold (depends on the threshold), the tasks holding 90% of the
  information (reported in the plot)
- **Status**: proposed

## Data and process

### D-4. Keep the scores of the coding runs on Google Drive

- **Decision**: the 106 `.csv.gz` score files stay out of git, in 2 archives
  in the Drive folder of the project; `fetch_coding_scores.py` downloads them,
  checks their SHA-256, and extracts them
- **Why**: GP asked on #808 not to keep data files in the repo; the checksum
  makes sure the notebooks read the same scores they were run on, and the
  extraction only writes plain files into the dirs of each archive
- **Alternatives**: committing the files (against the class rules), Git LFS
  (not set up in the class repo), downloading without a checksum (a changed
  file would go unnoticed)
- **Status**: accepted

### D-5. Merge `master` into a branch instead of rebasing it

- **Decision**: when a branch falls behind `master`, merge `master` into it
- **Why**: a rebase rewrites the pushed commits, including the
  pre-registration commits whose dates show the predictions came before the
  results, and needs a force-push; GP squash-merges each PR, so `master` ends
  up the same either way
- **Alternatives**: rebase and force-push (loses the dated commits on the
  branch)
- **Status**: accepted

### D-6. Break ties with a jitter in the conditional estimate

- **Decision**: before the conditional estimate of the winner (correction C4 of
  `preregistration_3.md`), add a uniform jitter of up to half a task to each
  score, give the winner the largest jitter among the variants tied with it,
  and add `1 / (12 n^2)` to the variance of each score
- **Why**: scores move by whole tasks, so variants often tie, and the normal
  theory of the estimate needs no ties; picking the best jittered score is the
  same as breaking the ties at random. In 1,500 simulated benchmarks of equal
  variants per setting, run before any result of Milestone 4, the median error
  was +0.33 points with the jitter against +2.64 with the bounds moved by half
  a task (20 variants, 82 tasks), and +0.19 against +0.78 (2 variants)
- **Alternatives**: moving the bounds by half a task (biased up), no change
  (a tie for first makes the estimate fall back to the raw score)
- **Status**: accepted

### D-7. Add the hybrid estimate as an extra look, not as a sixth correction

- **Decision**: keep the 5 corrections and rule H of `preregistration_3.md` as
  written, and add the hybrid estimate of Andrews, Kitagawa, and McCloskey,
  with `beta = 0.005`, next to them as an extra look (Cell 3.2 of
  `skill_vs_luck.selection.ipynb`), where rule H is applied again with it as a
  fourth candidate
- **Why**: Study F showed the conditional estimate (C4) with a mean error of
  -2 to -4 points and a root mean squared error of 11 to 19 points; when the
  winner only just won, the estimate falls far below the score, at times below
  0. The hybrid estimate is the fix of the same paper: it also assumes that the
  winner's mean is in an interval that covers all the means with a 99.5%
  chance, so it stays within about 3 standard deviations of the score. Adding
  it to rule H after seeing Study F would let a result pick the rule. It was
  added on 2026-11-27, before any analysis of the prompt variants was run, and
  it uses its own random draws, so the 5 corrections give the same numbers as
  before (checked on 600 random cases)
- **Alternatives**: replacing C4 (changes a pre-registered correction after a
  result), clipping C4 to 0 to 1 (hides the misses without fixing them), no
  extra look (leaves open whether a better version of C4 would pass)
- **Status**: accepted
