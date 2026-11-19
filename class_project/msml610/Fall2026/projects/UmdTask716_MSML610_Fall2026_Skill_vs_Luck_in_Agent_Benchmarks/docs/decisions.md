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
