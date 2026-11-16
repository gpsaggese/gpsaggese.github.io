# RLlib Optimal Trade Execution Progress

- **Issue**: [#686](https://github.com/gpsaggese/gpsaggese.github.io/issues/686)
- **Author**: Simran Kharbanda (@Sim2200)
- **Format**: follows `docs/planning/all.document_research.how_to_guide.md`.
  The notebooks hold the details, this log holds the findings and the decisions,
  and the Google Doc holds the visual evidence
- **Google Doc**:
  [Research results - MSML610 RLlib Project 1](https://docs.google.com/document/d/1ENZxEE3-JiKPs9pAgXvejf14-bE478Qb0qmD4JKIaq8/edit)
- **Dates**: UTC

## Status

| Milestone | Scope                                                   | Status      | PR                                                                 |
| :-------- | :------------------------------------------------------ | :---------- | :----------------------------------------------------------------- |
| 1         | Container, `ExecutionEnv`, calibration, baselines       | Done        | [#725](https://github.com/gpsaggese/gpsaggese.github.io/pull/725) |
| 2         | `rllib.API.ipynb`: RLlib API on CartPole                | Done        | [#796](https://github.com/gpsaggese/gpsaggese.github.io/pull/796) |
| 3         | `rllib.example.ipynb`: DQN on the execution environment | In progress | [#873](https://github.com/gpsaggese/gpsaggese.github.io/pull/873) |

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

## History

| Date       | Done                                                                                          |
| :--------- | :-------------------------------------------------------------------------------------------- |
| 2026-10-18 | Opened #686 with the objective, tasks, and milestones; set up the fork and the local repo  |
| 2026-10-22 | Office hours with GP: Project 1, SPY intraday data, 1-minute bars; opened #725                |
| 2026-10-24 | First calibration on Alpaca 1-minute bars, 2016-2019; M1 table and README                     |
| 2026-10-25 | `rllib.API.ipynb` (M2); code-quality skills applied to M1                                     |
| 2026-10-26 | Moved M1 to NYSE TAQ 2015-2024 and added the spread cost; decided on Almgren et al. impact    |
| 2026-10-27 | Started M3: VIX regime labels, one calibration per regime, `train_dqn()`                      |
| 2026-10-28 | #725 merged; opened #796; first M3 run; added the `expected_cost` reward                      |
| 2026-11-01 | Notebook and code-quality skills applied to M2; reran `rllib.API.ipynb`                       |
| 2026-11-03 | M2 cleanup; #796 ready for review                                                             |
| 2026-11-05 | Tuned DQN on the 2015-2019 market; template dependency policy                                 |
| 2026-11-06 | Opened #873 (draft); reran `rllib.example.ipynb` with TWAP-multiple actions                   |
| 2026-11-07 | #796 merged                                                                                   |
| 2026-11-11 | Rebased M3 on master; resolved the import-order conflict from #881                            |
| 2026-11-12 | Started the research Google Doc                                                               |
| 2026-11-14 | Decided: 100,000-share headline with the ADV sweep, 3 seeds, Almgren-Chriss then PPO vs DQN   |
| 2026-11-15 | Reran `rllib.example.ipynb` with 3 training seeds per agent; M3 results in the README        |

## Open Work

- [#873](https://github.com/gpsaggese/gpsaggese.github.io/pull/873): M3, draft
