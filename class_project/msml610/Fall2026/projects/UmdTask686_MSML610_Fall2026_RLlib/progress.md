# RLlib Optimal Trade Execution Progress

- **Issue**: [#686](https://github.com/gpsaggese/gpsaggese.github.io/issues/686)
- **Author**: Simran Kharbanda (@Sim2200)
- **Format**: follows `docs/planning/all.document_research.how_to_guide.md`.
  The notebooks hold the details, this log holds the findings, the README holds
  the decisions, and the Google Doc holds the visual evidence
- **Google Doc**:
  [Research results - MSML610 RLlib Project 1](https://docs.google.com/document/d/1E73NTJcTl_BcskJEdCJQoa3q1EduAlgtuekzBBLWOGU/edit)

## Status

| Milestone | Scope                                                   | Status      | PR                                                                 |
| :-------- | :------------------------------------------------------ | :---------- | :----------------------------------------------------------------- |
| 1         | Container, `ExecutionEnv`, calibration, baselines       | Done        | [#725](https://github.com/gpsaggese/gpsaggese.github.io/pull/725) |
| 2         | `rllib.API.ipynb`: RLlib API on CartPole                | Done        | [#796](https://github.com/gpsaggese/gpsaggese.github.io/pull/796) |
| 3         | `rllib.example.ipynb`: DQN on the execution environment | Done        | [#873](https://github.com/gpsaggese/gpsaggese.github.io/pull/873) |

## Built

- `rllib_utils.py`: `ExecutionEnv` (Gymnasium), TWAP and sell-all baselines,
  episode runner, greedy policy, summary table, `train_dqn()`, plots
- `data_utils.py`: Alpaca and FRED VIX downloads, calibration, VIX regime labels,
  `compute_m1_table()`
- `taq_utils.py`: SPY 1-minute bars from NYSE TAQ on WRDS
- `rllib.API.ipynb` / `.py`: RLlib API tutorial on CartPole-v1
- `rllib.example.ipynb` / `.py`: DQN against TWAP on the calibrated market
- `test/`: unit tests for `rllib_utils`, `data_utils`, `taq_utils`
- `README.md`: setup, results, and how to run

## Findings

1. 100,000 shares is 0.14% of SPY's daily volume, so impact and spread together
   cost below 1 bp, and the TWAP shortfall is driven by the price noise (~37-51
   bps std)
   - Data: README, section "M1: Calibration and Baselines"
2. NYSE TAQ matches Alpaca on all 2,264 trading days of 2016-2024: close within
   0.0 bps (median), volume ~95% of Alpaca
3. The permanent impact fitted on TAQ signed volume is 35-45x the Almgren et al.
   (2005) value (0.0955 vs 0.00213 on 2015-2019), since it measures the whole
   market's order flow; it is reported only as a check
4. On CartPole-v1, PPO solves the task at 24,000 env steps and DQN reaches 235
   - Data: `rllib.API.ipynb`, section "Compare PPO and DQN"
5. With fraction-of-inventory actions, DQN sold early, held ~20% of the order
   until the last step, and dumped it; no fraction action matches TWAP
6. The price noise (~38-53 bps std per episode) is ~100x the cost differences
   between schedules, so the shortfall reward is mostly noise; the
   `expected_cost` reward is ~100x less noisy
7. TWAP-multiple actions help DQN; training 2.5x longer and hiding the price do
   not (tuned on the 2015-2019 market only)
8. With TWAP-multiple actions, averaged over 3 training seeds, DQN's expected
   cost is 8-22% above TWAP at 100,000 shares (0.224 and 0.254 vs 0.208 bps)
   and 1.5-1.7x TWAP for larger orders (1/5/10% ADV)
   - Data: `rllib.example.ipynb`, sections "Evaluate the Policy" and "Order Size
     Sweep"
9. Two runs of the same notebook gave DQN 5.52 vs 4.18 bps at 10% ADV, so one
   training run is not enough to compare policies

## Next

1. Almgren-Chriss variance penalty and its analytic schedule
2. PPO vs DQN on the execution environment

## Open Work

- [#873](https://github.com/gpsaggese/gpsaggese.github.io/pull/873): M3, in review
