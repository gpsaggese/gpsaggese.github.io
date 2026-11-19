# RLlib Optimal Trade Execution

Train RLlib agents to liquidate an order of SPY shares within one trading day,
minimizing the implementation shortfall (bps) versus the arrival price, and
compare them with TWAP, an immediate full sale, and the Almgren-Chriss
schedule.

- Issue: [#686](https://github.com/gpsaggese/gpsaggese.github.io/issues/686)
- Spec:
  [RLlib_Project_Description.md](../../../../project_descriptions/MSML610/RLlib_Project_Description.md),
  Project 1

This is a running doc: status, results, and decisions are updated with each
milestone, newest first.

## Status

| Milestone | Content | Status |
|---|---|---|
| M1 | Container, environment, SPY calibration, TWAP and sell-all baselines | Done |
| M2 | `rllib.API.ipynb`: RLlib API on CartPole | Done |
| M3 | `rllib.example.ipynb`: DQN agent, evaluation by VIX regime, order-size sweep | Done |
| Bonus | PPO vs DQN, Almgren-Chriss variance penalty and schedule | Not started |

## Results

### M3: DQN on the Execution Environment

Setup:

- Train on the 2015-2019 calibration; test on 1,000 episodes calibrated on
  2020-2024, split by VIX regime (519 calm and 481 stressed episodes, in
  proportion to the days)
  - A day is calm if the previous VIX close is below 20
- Actions: multiples 0 to 3x of the TWAP pace (remaining inventory over
  remaining steps), so TWAP is the action 1.0 at every step
- Rewards: `shortfall` (the spec reward) and `expected_cost` (the same minus
  the P&L of the price shock on the shares still held, which removes the
  price noise without changing the expected return)
- DQN: 60 iterations (~60,000 env steps), 3 training seeds per agent; the
  DQN rows average the seeds
- Every policy runs on the same test paths, so the win rate compares each
  episode with TWAP

Test results at 100,000 shares, all episodes (bps):

| Policy | Shortfall mean ± std | Expected cost | Expected cost std over seeds | Win rate vs TWAP |
|---|---|---|---|---|
| TWAP | 1.67 ± 53.51 | 0.208 | | |
| Sell-all | 0.54 ± 0.10 | 0.541 | | 0.49 |
| DQN (`shortfall`) | 1.00 ± 37.58 | 0.224 | 0.004 | 0.57 |
| DQN (`expected_cost`) | 1.00 ± 49.58 | 0.254 | 0.013 | 0.56 |

Order-size sweep, `expected_cost` agent, expected cost (bps):

| Order | TWAP | Sell-all | DQN | DQN / TWAP |
|---|---|---|---|---|
| 1% ADV (0.73M shares) | 0.449 | 2.890 | 0.760 ± 0.172 | 1.69x |
| 5% ADV (3.67M shares) | 1.564 | 13.765 | 2.442 ± 0.109 | 1.56x |
| 10% ADV (7.34M shares) | 2.957 | 27.360 | 4.596 ± 0.327 | 1.55x |

Findings:

- With linear impact and no risk penalty, TWAP has the lowest expected cost;
  DQN is 8-22% above it at 100,000 shares and 1.5-1.7x TWAP for larger
  orders
- The shortfall means are dominated by price noise (~38-68 bps std), so
  compare the expected cost, not the shortfall mean or the win rate
- Both DQN agents sell faster than TWAP early in the day (inventory plot in
  the notebook); the `expected_cost` agent also keeps ~17% of the order until
  the last step
- Selling early lowers the shortfall std (38-50 vs 54 bps) and wins slightly
  more than half the episodes, but pays more impact on average
- The spread over seeds is small next to the gap to TWAP, so the gap is not
  a training accident

### M2: RLlib API on CartPole

- `rllib.API.ipynb` covers `register_env`, `PPOConfig`, `DQNConfig`,
  `Algorithm.train()`, `Algorithm.evaluate()`, the `RLModule`, and checkpoints
- Results vary between runs, since the parallel env runners make training
  nondeterministic

| | PPO | DQN |
|---|---|---|
| Env steps sampled | 32,000 | 12,000 |
| Best mean episode return | 500 (solved at 24,000 steps) | 235 |
| Evaluation return (10 episodes) | 486 | n/a |

### M1: Calibration and Baselines

Setup:

- Order: sell 100,000 SPY shares in 20 steps within one trading day
- Data: SPY 1-minute bars built from NYSE TAQ (WRDS), regular hours,
  2015-2024
- Volatility: std of the open-to-close log return
- Impact: coefficients from Almgren, Thum, Hauptmann, and Li (2005), scaled by
  SPY volatility, linear in the traded fraction of daily volume
- Spread: half the volume-weighted effective spread, paid on every share sold
- Baselines: 1,000 episodes each, same price paths for both

| | 2015-2019 (calibration) | 2020-2024 (test) |
|---|---|---|
| Trading days | 1258 | 1258 |
| Open-to-close daily vol (%) | 0.68 | 0.94 |
| Avg daily volume (M shares) | 73.4 | 66.4 |
| Order as % of ADV | 0.136 | 0.151 |
| Half spread (bps) | 0.268 | 0.179 |
| gamma (permanent) | 0.00213 | 0.00295 |
| eta (temporary) | 0.0192 | 0.0267 |
| TWAP shortfall, mean ± std (bps) | 0.21 ± 37.17 | 0.10 ± 51.47 |
| Sell-all shortfall, mean ± std (bps) | 0.53 ± 0.00 | 0.58 ± 0.00 |
| gamma fitted on TAQ signed volume (comparison only) | 0.0955 (t = 58.9, R^2 = 0.127) | 0.1044 (t = 35.9, R^2 = 0.051) |

Findings:

- 100,000 shares is 0.14% of SPY's daily volume, so impact and spread cost
  together are below 1 bp
- TWAP shortfall is driven by price noise (~37-51 bps std), not by impact
- The order size needs to be a larger fraction of daily volume for the agent
  to have a cost to optimize
- The gamma fitted on the market's signed volume is 35-45x the Almgren et al.
  value: it measures the whole market's order flow, and fast traders trade
  after price moves, so it overstates the impact of one order

## Decisions

- 2026-11-14: 100,000 shares stays the headline, with the 1/5/10% ADV sweep
  next to it; results average 3 training seeds per agent, since one run of
  the same notebook gave 5.52 and the next 4.18 bps at 10% ADV
- 2026-11-05: Actions as multiples of the TWAP pace instead of fractions of
  the remaining inventory, since no fraction action can follow TWAP; tuned on
  the 2015-2019 market only, so the test set is used once
- 2026-10-28: `expected_cost` reward next to the spec's `shortfall` reward,
  since the price noise is ~100x the cost differences between schedules
- 2026-10-26: NYSE TAQ on WRDS as the data source, replacing Alpaca
  - Covers 2015-2019, as in the spec (Alpaca starts in 2016)
  - Check against Alpaca 2016-2024: all 2,264 trading days match, close
    within 0.0 bps (median), volume ~95% of Alpaca
  - Prices, spreads, and trade signs use only regular trades; block trades
    (average-price, contingent) print away from the quotes and are counted
    only in volume
  - Trades are signed buy or sell by the quote rule; trades at the midpoint
    are unsigned
- 2026-10-26: Impact coefficients from Almgren, Thum, Hauptmann, and Li
  (2005), since the spec calibrates volatility and volume from data, and the
  Almgren-Chriss bonus uses the same coefficients; the fitted gamma is
  reported only as a check
- 2026-10-22: SPY 1-minute bars instead of daily data, since higher resolution
  is better

## Pending

- Drop cancelled and corrected trades (`tr_corr` other than 00), which the
  WRDS consolidated trades keep; they are under 0.01% of SPY trades

## Files

- `rllib_utils.py`: `ExecutionEnv` Gymnasium environment, TWAP and sell-all
  baselines, episode runner, `train_dqn()`, plots
- `taq_utils.py`: SPY 1-minute bars from NYSE TAQ on WRDS
- `data_utils.py`: SPY (Alpaca) and VIX download, calibration, VIX regimes,
  M1 table
- `test/`: unit tests
- `rllib.API.ipynb`, `rllib.example.ipynb`: notebooks (M2, M3)
- `Dockerfile`, `requirements.txt`, `docker_*.sh`: container, see
  `docker_scripts.README.md`

## How to Run

- Build the container and start Jupyter
  ```bash
  > ./docker_build.sh
  > ./docker_jupyter.sh
  ```

- Download the TAQ bars into `data/taq/` (not committed; WRDS data must not
  be shared) with a WRDS account, the username in `WRDS_USERNAME`, and a
  `~/.pgpass` file; needs `pip install wrds pyarrow`
  ```bash
  > WRDS_USERNAME=<user> PYTHONPATH=$(git rev-parse --show-toplevel):$(git rev-parse --show-toplevel)/helpers_root python -c \
      'import class_project.msml610.Fall2026.projects.UmdTask686_MSML610_Fall2026_RLlib.taq_utils as t; t.download_spy_taq(2015, 2024)'
  ```

- Download the VIX and the Alpaca bars (used for the TAQ check) into `data/`
  with a free Alpaca account, passing the keys as `ALPACA_API_KEY_ID` and
  `ALPACA_API_SECRET_KEY`
  ```bash
  > docker run --rm --env-file <keys.env> -v $(git rev-parse --show-toplevel):/git_root \
      -w /git_root -e PYTHONPATH=/git_root:/git_root/helpers_root \
      gpsaggese/umd_project_rllib python -c \
      'import class_project.msml610.Fall2026.projects.UmdTask686_MSML610_Fall2026_RLlib.data_utils as d; d.download_vix(); d.download_spy(2016, 2024)'
  ```

- Reproduce the M1 table
  ```python
  import class_project.msml610.Fall2026.projects.UmdTask686_MSML610_Fall2026_RLlib.data_utils as d
  print(d.compute_m1_table({"2015-2019": (2015, 2019), "2020-2024": (2020, 2024)}))
  ```

- Run the unit tests
  ```bash
  > pytest test/test_rllib_utils.py test/test_data_utils.py test/test_taq_utils.py
  ```

## Data Citation

Wharton Research Data Services (WRDS) was used in preparing this project. This
service and the data available thereon constitute valuable intellectual
property and trade secrets of WRDS and/or its third-party suppliers.
