# Causal Analysis of Financial Tradability Progress

- Issue: #733
- Research Google Doc: [Research Log](https://docs.google.com/document/d/1ttpZT3kWVXq5jMZn3nx1ijmtH97VfAclRmSa9Z2EYsc/edit)
- Decisions: [decisions.md](decisions.md)

## Status

| Milestone | Target date | Status | PR |
| :-------- | :---------- | :----- | :- |
| Step 0: Exercise 1 from #381 (PnL vs hit rate) | Oct 29 | Done | #794 |
| Step 0: Exercise 1 on SPY (asked on #733) | Oct 29 | In review | #820 |
| Step 1: Data, cleaning, features, labels | Nov 5 | In review | #820 |
| Step 2: Predictability vs horizon, hit rate and PnL with costs | Nov 5 | In review | #820 |
| Step 3: Causal analysis and placebo tests | Nov 19 | Not started | |
| Step 4: Best horizon per regime, walk-forward backtest | Nov 26 | Not started | |
| Step 5: Write-up, video, final PR | Dec 3 | Not started | |

## Built

- `tradability_utils.py`: loads Binance prices, computes returns, simulates the
  PnL for a given hit rate, computes the effective number of trades
- `exercise1.ipynb`: Exercise 1 with plots and observations
- `exercise1_spy.ipynb`: Exercise 1 on SPY next to BTC (Yahoo Finance data)
- `data_utils.py`: loads 5-minute OHLCV bars, cleans them, computes features
  and labels per horizon
- `test/test_data_utils.py`: unit tests, including no look-ahead and no
  feature or label spanning a gap
- `tradability.data.ipynb`: data quality, features, labels per horizon
- `predict_utils.py`: walk-forward split, always buy, momentum, and logistic
  regression, and the hit rate and PnL of a set of trades
- `tradability.models.ipynb`: the three models per horizon, tested on
  2022-2026, with and without costs
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
4. The N_eff correction also works for SPY: in all 5 cases (SPY and BTC, 1h
   and 1d) the corrected p* is where the simulation crosses 95%
   - SPY needs a higher hit rate than BTC at the same bar size because it
     trades fewer hours: about 52.5% for SPY 1h vs 51% for BTC 1h
   - What matters is the number of trades, not the size of the moves: daily
     SPY and daily BTC both need about 53%
   - Data: Yahoo Finance and Binance, `exercise1_spy.ipynb`
5. The 5-minute Binance data is clean: no duplicates, 0.07% missing bars. BTC
   and ETH share the same 15 gaps (Binance outages), all at least 1 hour, so
   the data is split into 16 segments
   - Data: `tradability.data.ipynb`
6. Both the share of up moves and p* go up with the horizon. Always buying is
   above p* at 1h and 4h but below it at 5m and 1d, because of the 2020-2026
   uptrend. This is the baseline step 2 has to beat
   - Data: `tradability.data.ipynb`, section "Labels per horizon"
7. A hit rate above p* does not mean a profit. The logistic regression is
   right 52% of the time from 5m to 4h (p* is 50.2-51.3%), but the moves it
   gets wrong are 6-10% bigger, so it makes 0-2 bp per trade. The weighted hit
   rate (share of |r| in the right direction) is the number to compare with
   0.5
   - Data: `tradability.models.ipynb`, test 2022-2026
8. With a 10 bp cost per trade, no model makes money at any horizon. The hit
   rate needed goes from over 100% at 5m to 55-56% at 1d, and the best PnL
   before costs is 9.5 bp per trade (ETH, 1d). Costs push the best horizon
   to the longest one
   - Data: `tradability.models.ipynb`, section "Costs"
9. Short moves tend to reverse: momentum is right only 46-49% of the time
   from 5m to 4h

## Next

1. Step 3: causal analysis of the horizon effect by regime (volatility,
   trend), with placebo tests
2. Use the weighted hit rate and PnL, not the plain hit rate, to judge
   models

## History

| Date | Done |
| :--- | :--- |
| 2026-10-22 | Filed #733 |
| 2026-10-23 | GP approved the topic; start from scratch, use #381 for ideas only |
| 2026-10-28 | Posted plan and related research on #733 |
| 2026-10-28 | PR #794 merged: Exercise 1, Docker setup, effective-N correction |
| 2026-10-28 | Started the research Google Doc, as GP asked on #794 |
| 2026-10-29 | GP: results "very interesting", asked for SPY; PR #800 merged |
| 2026-10-29 | Exercise 1 on SPY; step 1 data loading and cleaning |
| 2026-10-30 | Step 1 features and labels; step 2 models per horizon |

## Open Work

- Zero-volume bars right before a gap: part of the gap or not?
- Yahoo keeps only 730 days of hourly data: commit the SPY 1h CSV?

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
