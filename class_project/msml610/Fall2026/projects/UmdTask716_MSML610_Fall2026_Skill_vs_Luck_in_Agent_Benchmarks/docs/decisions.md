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
