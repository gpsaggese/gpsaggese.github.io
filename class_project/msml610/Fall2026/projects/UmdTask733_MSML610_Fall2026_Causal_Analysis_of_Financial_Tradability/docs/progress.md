# Causal Analysis of Financial Tradability Progress

- Issue: #733
- Research Google Doc: [Research Log](https://docs.google.com/document/d/1ttpZT3kWVXq5jMZn3nx1ijmtH97VfAclRmSa9Z2EYsc/edit)
- Decisions: [decisions.md](decisions.md)

## Status

| Milestone | Target date | Status | PR |
| :-------- | :---------- | :----- | :- |
| Step 0: Exercise 1 from #381 (PnL vs hit rate) | Oct 29 | Done | #794 |
| Step 1: Data, cleaning, features, labels | Nov 5 | In progress | |
| Step 2: Predictability vs horizon, hit rate and PnL with costs | Nov 5 | Not started | |
| Step 3: Causal analysis and placebo tests | Nov 19 | Not started | |
| Step 4: Best horizon per regime, walk-forward backtest | Nov 26 | Not started | |
| Step 5: Write-up, video, final PR | Dec 3 | Not started | |

## Built

- `tradability_utils.py`: loads Binance prices, computes returns, simulates the
  PnL for a given hit rate, computes the effective number of trades
- `exercise1.ipynb`: Exercise 1 with plots and observations
- Docker setup from `class_project/project_template`

## Findings

1. At a 51% hit rate, 1h trading made money in every simulation, 1d in about
   75-80% of them: 59,127 hourly vs 2,464 daily trades over the same 6.75
   years (BTC and ETH, 2020-01-01 to 2026-10-01)
   - Data: Binance public market data, `exercise1.ipynb`
2. The spec formula p* = 1/2 + 1.645 / (2 sqrt(N)) is too optimistic: at its
   value for 1h (0.5034), the chance of profit is only about 80%, not 95%
   - Cause: fat tails, a few large moves dominate the PnL
   - Fix: use N_eff = (sum |r|)^2 / sum r^2 instead of N (about 21,000 for 1h
     and 1,100 for 1d). The corrected p* (0.5056 for 1h, about 0.525 for 1d)
     matches where the simulation crosses 95%
3. Drawing a fixed number of hits (Exercise 1) and drawing each hit
   independently (spec) agree at 1h but not at 1d: about 0.52 vs 0.525 for a
   95% chance of profit

## Next

1. Step 1: 5-minute OHLCV data, cleaning, features, and labels for horizons
   5m, 15m, 1h, 4h, 1d
2. Step 2: predictability vs horizon with baseline models, and the minimum hit
   rate after costs

## History

| Date | Done |
| :--- | :--- |
| 2026-10-22 | Filed #733 |
| 2026-10-23 | GP approved the topic; start from scratch, use #381 for ideas only |
| 2026-10-28 | Posted plan and related research on #733 |
| 2026-10-28 | PR #794 merged: Exercise 1, Docker setup, effective-N correction |
| 2026-10-28 | Started the research Google Doc, as GP asked on #794 |

## Open Work

- Step 1 spec: agreed in the team, to be linked from PR 1

## 2026-10-28: UmdTask733_MSML610_Fall2026_Causal_Analysis_of_Financial_Tradability

- **Issue**: #733, stays open (multi-PR project)
- **PR**: #794, by @oladri-renuka, opened 2026-10-27, merged 2026-10-28
- **Milestone**: Step 0
- **What was done**:
  - Exercise 1 from #381: PnL distribution vs hit rate for BTC and ETH at 1h
    and 1d
  - Effective-N correction to the minimum hit rate formula
  - Docker setup from the project template
- **Testing**: ran the notebook end to end inside Docker; `ruff check` and
  `ruff format` pass with the repo config
- **CI note**: `fast_tests.no_docker` failed on this branch, and also fails on
  `master` on the same day, so it was not caused by this PR
