# Causal Analysis of Agent Skill and Luck Decisions

## Example Notebook

### `D1` (The analysis helpers live in their own module)

- Status: accepted
- `causal_success_utils.py` keeps the shared simulation and the config
- `causal_success_analysis_utils.py` holds the analysis and plotting helpers of the
  example notebook
  - The notebook imports it as `analysis`, next to `utils`
  - No helper only forwards a call to `causal_success_utils.py`

### `D2` (One baseline result for the whole notebook)

- Status: accepted
- The example notebook runs `run_experiment(get_default_config())` once
  - 100 agents, 80 periods, seed 42
- Every summary, plot, estimate, and policy comparison derives from that config
- A longer-run experiment is a separate, labeled task

### `D3` (The simulator behavior is described, not changed)

- Status: accepted
- The spillover check runs on every lucky event, also when the selected agent does
  not capitalize on it
  - The paper says only a successful lucky event creates a spillover
  - The notebook describes the code, and lists the difference with the paper
- The treatment is the count of realized beneficial events, including spillovers

### `D4` (Valid estimator uncertainty)

- Status: accepted
- The DML effect is reported with the standard error of the estimator
- The causal forest reports the interval of each conditional effect
- The spread of fitted per-agent effects is not used as uncertainty
- The causal assumptions are stated next to the estimates

### `D5` (The policy comparison is exploratory)

- Status: accepted
- Every policy runs through `run_experiment()` with the same seed
- `performance` equals `egalitarian` when all agents start with the same capital
- The CATE-based allocation is a heuristic, evaluated in-sample
- The `sqrt(allocation)` model of the old notebook is removed

### `D6` (Paper values are compared, not reproduced)

- Status: accepted
- The notebook separates the single-seed baseline from averages over seeds
- The parameters are not tuned to match the paper

### `D7` (The starting capital is inferred only when the floor never binds)

- Status: accepted
- A policy rescales each agent's multiplier only if the capital floor never
  binds
  - A bound floor makes the multiplier of an event depend on the capital
- `cell5_3_compute_implied_initial_capital()` reports the agents at the floor
  and returns NaN instead of a wrong inference

### `D8` (The policy helpers use the actual starting capital)

- Status: accepted
- The simulator raises an `initial_capital` below `capital_floor` to the floor
- `cell5_2_run_policy_comparison()` and
  `cell5_3_compute_implied_initial_capital()` read each agent's starting
  capital from its capital history, not from the config
  - The config is accepted as is, since the simulator accepts it
  - Starting at the floor is not an event, so it does not stop the rescaling

### `D9` (The seed sweep has no progress bar)

- Status: accepted
- An exception to the "Use Progress Bar" rule of `coding.rules.md`
- `cell6_1_run_seed_sweep()` runs 50 seeds without a `tqdm` progress bar
  - A progress bar saved about 50 lines of stderr in the executed notebook
