# Skill Luck Sim

Synthetic simulation of the winner's curse when picking the best of K agent configs
from a benchmark, for issue gpsaggese/gpsaggese.github.io#541

- Draws K configs with close true skills, N tasks, and S seeds per task
- Picks the config with the best observed score and measures how far that score is
  above its true resolve rate
- Checks two fixes (re-running the winner on fresh seeds, or on fresh tasks) and
  bootstrap error bars on each config's score
- No API calls and no real agents: `numpy`, `pandas`, `matplotlib`, `pytest` only

This folder is separate from `research/Causal_Analysis_of_Agent_Skill_And_Luck/`,
which is a different simulation, and does not import from it

## Description of Files

| File                     | Description                                                                 | Cluster    |
| ------------------------ | --------------------------------------------------------------------------- | ---------- |
| `README.md`              | This file: model, how to run, results, limits                               | Docs       |
| `bootstrap.py`           | Cluster bootstrap CIs on config scores and their coverage of the true rate  | Estimation |
| `estimate.py`            | Scores, winner, curse, rank error, fresh-seed and fresh-task re-evaluation  | Estimation |
| `generate.py`            | Model parameters, draws of configs, tasks, outcomes, exact true rates       | Generation |
| `plotting.py`            | The three figures                                                           | Plotting   |
| `results/`               | CSVs and PNGs written by `run_sweep.py`                                     | Outputs    |
| `run_sweep.py`           | Runs the (K, N, S) grid and writes `results/`                               | Executable |
| `test/test_bootstrap.py` | Tests for `bootstrap.py`                                                    | Tests      |
| `test/test_estimate.py`  | Tests for `estimate.py`                                                     | Tests      |
| `test/test_generate.py`  | Tests for `generate.py`                                                     | Tests      |

## The Model

- Config k has a true skill `theta_k ~ Normal(mu_theta, sigma_theta)`
- Task i has a difficulty `b_i ~ Normal(mu_b, sigma_b)`
- Config k and task i have an interaction `u_ki ~ Normal(0, sigma_u)`, so a config
  can be good at tasks another config is bad at
- `P(resolve) = sigmoid(theta_k - b_i + u_ki)`, and each seed is an independent
  Bernoulli draw with that probability
- The true resolve rate `Ybar_k` is the expectation over the task population, not
  over the N sampled tasks
  - I compute it by Gauss-Hermite numerical integration, using that the logit is
    `Normal(theta_k - mu_b, sigma_b^2 + sigma_u^2)` over random tasks
- Observed score: mean over the N tasks and S seeds. Winner: the highest observed
  score
- The mapping to the paper's `Ybar_A` and `L_A(Y)` is in the docstring of
  `generate.py`

- Defaults (assumed, not fitted to any real benchmark):
  - `mu_theta = 1.0`, `sigma_theta = 0.2`, `mu_b = 0`, `sigma_b = 2.0`,
    `sigma_u = 1.0`
  - This gives true rates around 64% with a spread of 2.7 points across configs
  - With K = 10, the best and second best config differ by 1.4 points on average,
    and the best and worst by 8.1 points

## Description of Executables

### `run_sweep.py`

- What it does:
  - Runs 1000 replications for each of the 64 settings with K in {3, 5, 10, 20},
    N in {25, 50, 100, 500}, S in {1, 3, 5, 10}
  - Runs bootstrap coverage at K = 5 for each (N, S), with 400 replications and
    500 bootstrap draws per interval
  - Writes `curse_sweep.csv`, `coverage_sweep.csv`, and three PNGs to `results/`
  - Takes about 5 minutes on one core. Each setting has its own seed derived from
    `--seed`, so results do not depend on run order

- Run the tests from the repo root:
  ```bash
  > python -m pytest skill_luck_sim
  ```

- Run the full sweep:
  ```bash
  > PYTHONPATH=. python skill_luck_sim/run_sweep.py
  ```

- Run a quick smoke version into another directory:
  ```bash
  > PYTHONPATH=. python skill_luck_sim/run_sweep.py \
      --n_reps 50 --n_coverage_reps 10 --n_boot 100 --out_dir tmp.run_sweep
  ```

## Results

All numbers below come from `results/` with `--seed 0`. Monte Carlo standard errors
are about 0.001 to 0.003 for the curse and about 0.005 for coverage

### Winner's Curse

![Winner's curse](results/fig1.curse.png)

- The winner's observed score is above its true rate in every setting
  - K = 10, N = 100, S = 3: 2.8 points too high
  - K = 20, N = 25, S = 1: 13.3 points too high
  - K = 3, N = 500, S = 10: 0.2 points too high
- More candidates make it worse, and more tasks help more than more seeds
  - At K = 10 and S = 3, going from 25 to 500 tasks cuts the curse from 7.0 to 0.6
    points
  - At K = 10 and N = 25, going from 1 to 10 seeds cuts it from 10.7 to 4.9 points
  - The grid has 8 pairs of settings with the same total runs per config. In all 8,
    the setting with more tasks had the smaller curse. E.g., at K = 10, N = 500,
    S = 1 gives 1.5 points and N = 50, S = 10 gives 3.0 points

### Picking the Right Config

![P(winner is truly best)](results/fig2.prob_best.png)

- The observed winner is often not the truly best config
  - K = 10, N = 100, S = 3: right 46% of the time
  - K = 20, N = 500, S = 10: right 70% of the time
- When it is wrong, the loss is small: at K = 10, N = 100, S = 3 the winner's true
  rate is on average 1.1 points below the best config's
- Rankings below the top are noisy: at K = 20, N = 25, S = 1 each config's observed
  rank is off by 5.4 places on average

### Fixing the Curse

- Re-running the winner on N fresh tasks with S seeds gives an unbiased estimate
  - Pooled over all 64 settings, the mean error is +0.0002, and 1 of 64 settings
    is beyond 2.58 standard errors from zero
  - Cost: N S extra runs, i.e. 1/K of the original search. 10% at K = 10
- Re-running the winner on the same tasks with fresh seeds removes only part of
  the curse
  - It removes 85% to 92% at S = 1 but only 27% to 38% at S = 10 (K = 10)
  - Reason: the winner was also selected for its luck on these specific tasks (the
    `u_ki` terms), and new seeds keep that luck. With more seeds, more of the curse
    comes from task luck and less from seed luck
  - Cost: same as fresh tasks, N S extra runs

### Error Bars

![Bootstrap coverage](results/fig3.coverage.png)

- Resampling tasks as the cluster gives 95% intervals that cover the true rate
  93.5% of the time on average for N between 25 and 100, and 94.8% at N = 500
  - So it undercovers by about 1.5 points for small N. That matches the known
    small-sample bias of the percentile bootstrap, but I have not checked the
    cause
- Resampling tasks and then seeds inside each task gives wider intervals: 95.1% on
  average, from 92.6% to 96.8%
  - This is not a sign that two-stage is the right method. With S = 1 it is the
    same as task resampling, and the test case with iid outcomes shows it
    overcovers (98%) because it counts seed noise twice
  - In this model the extra width happens to cancel the small-N undercoverage
- The coverage standard errors treat the K intervals of one replication as
  independent. They share the same tasks, so the real standard errors are a bit
  larger

## Assumptions That Could Make a Real Benchmark Behave Differently

- Task difficulty is Normal on the logit scale. Real per-task results are often
  bimodal, with many tasks solved by every config or by none
- `u_ki` is independent across tasks. Real agents fail in correlated ways, e.g.
  one scaffold breaks on every task from the same repository
- Seeds are independent Bernoulli draws. Real reruns can share cached state,
  rate limits, or flaky test environments
- Configs are drawn independently with one skill spread. A real scaffold search
  produces variants of one agent whose interactions `u_ki` are correlated, which
  shrinks the task-luck part of the curse
- The selection rule is a plain argmax over all K at once. Real searches are often
  sequential and adaptive, which can make the curse larger
- Real benchmarks have a fixed task set, e.g. 500 tasks for SWE-bench Verified.
  Fresh tasks may not exist, and then fresh seeds are the only fix available, with
  the limits shown above
- All parameter values are guesses. The next step is to fit `sigma_theta`,
  `sigma_b`, and `sigma_u` to real per-task, per-seed results
