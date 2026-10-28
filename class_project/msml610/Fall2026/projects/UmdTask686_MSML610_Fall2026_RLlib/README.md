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
| M2 | `rllib.API.ipynb`: RLlib API on CartPole | Not started |
| M3 | `rllib.example.ipynb`: DQN and PPO agents, evaluation, plots | Not started |
| Bonus | PPO vs DQN, Almgren-Chriss variance penalty and schedule | Not started |

## Results

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
- Order size as a percentage of ADV, and a sweep over sizes (M3)

## Files

- `rllib_utils.py`: `ExecutionEnv` Gymnasium environment, TWAP and sell-all
  baselines
- `taq_utils.py`: SPY 1-minute bars from NYSE TAQ on WRDS
- `data_utils.py`: SPY (Alpaca) and VIX download, calibration, M1 table
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
