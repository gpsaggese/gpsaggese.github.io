# Causal Analysis of Financial Tradability Progress

- Issue: #733
- Research Google Doc: [Research Log](https://docs.google.com/document/d/1ttpZT3kWVXq5jMZn3nx1ijmtH97VfAclRmSa9Z2EYsc/edit)
- Decisions: [03-decisions.md](03-decisions.md)

## Status

| Milestone | Target date | Status | PR |
| :-------- | :---------- | :----- | :- |
| Step 0: Exercise 1 from #381 (PnL vs hit rate) | Oct 29 | Done | #794 |
| Step 0: Exercise 1 on SPY (asked on #733) | Oct 29 | Done | #820 |
| Step 1: Data, cleaning, features, labels | Nov 5 | Done | #820 |
| Step 2: Predictability vs horizon, hit rate and PnL with costs | Nov 5 | Done | #820 |
| Follow-ups from GP's answers: stale bars, SPY overnight bar, cost sweep, data sources | Nov 12 | In review | |
| Gap analysis (#904) | Nov 19 | In review | |
| Step 3: What happens per horizon in different market conditions (no causal analysis yet, `D17`) | Nov 19 | In review | |
| Robustness: do the results hold with a different target? | Nov 26 | In progress: other targets in review, volatile markets next | #905 |
| Step 4: Best horizon per regime, walk-forward backtest | Nov 26 | Not started | |
| Step 5: Write-up, video, final PR | Dec 3 | Not started | |

## Built

- `tradability_utils.py`: loads Binance and Yahoo prices, computes returns,
  drops the overnight bar, simulates the PnL for a given hit rate, computes
  the effective number of trades
- `exercise1.ipynb`: Exercise 1 with plots and observations
- `exercise1_spy.ipynb`: Exercise 1 on SPY next to BTC (Yahoo Finance data)
- `data_utils.py`: loads 5-minute OHLCV bars, cleans them, computes features
  and labels per horizon
- `test/test_data_utils.py`: unit tests, including no look-ahead and no
  feature or label spanning a gap
- `test/test_predict_utils.py`, `test/test_tradability_utils.py`,
  `test/test_regime_utils.py`: unit tests for the models, the metrics, the
  cost sweep, the overnight bar, and the regimes (including no look-ahead)
- `tradability.data.ipynb`: data quality, features, labels per horizon
- `tradability.gaps.ipynb`: gap analysis (#904): when the gaps happen, and
  how many bars, labels, and features they cost
- `regime_utils.py`: volatility and trend regimes at each bar, from past
  data only
- `tradability.regimes.ipynb`: step 3, the step 2 models and trades split by
  regime
- `predict_utils.py`: walk-forward split, always buy, momentum, and logistic
  regression, the same linear model trained on other targets, and the hit
  rate and PnL of a set of trades at one or several costs
- `tradability.targets.ipynb`: robustness (#905), the step 2 model trained on
  four targets
- `tradability.models.ipynb`: the three models per horizon, tested on
  2022-2026, with costs from 0 to 20 bp
- `data_sources.reference.md`: where each dataset comes from and how it is
  made
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
   - Without the first hourly bar of each day (the overnight move, `D16`),
     the corrected p* for SPY 1h barely changes: 0.5246 to 0.5250
   - Data: Yahoo Finance and Binance, `exercise1_spy.ipynb`
5. The 5-minute Binance data is clean: no duplicates, 0.07% missing bars. BTC
   and ETH share the same 15 gaps (Binance outages), all at least 1 hour, so
   the data is split into 16 segments. 41 of the 42 zero-volume bars are
   right before an outage and are now treated as part of the gap (`D15`)
   - Data: `tradability.data.ipynb`
6. Both the share of up moves and p* go up with the horizon. Always buying is
   above p* at 1h and 4h but below it at 5m and 1d, because of the 2020-2026
   uptrend. This is the baseline step 2 has to beat
   - Data: `tradability.data.ipynb`, section "Labels per horizon"
7. A hit rate above p* does not mean a profit. The logistic regression is
   right 52% of the time from 5m to 4h (p* is 50.2-51.3%), but the moves it
   gets wrong are 5-10% bigger, so it makes 0-2 bp per trade. The weighted hit
   rate (share of |r| in the right direction) is the number to compare with
   0.5
   - Data: `tradability.models.ipynb`, test 2022-2026
8. With a 10 bp cost per trade, no model makes money at any horizon. The hit
   rate needed goes from over 100% at 5m to 55-56% at 1d, and the best PnL
   before costs is 9.4 bp per trade (ETH, 1d). Costs push the best horizon
   to the longest one
   - Data: `tradability.models.ipynb`, section "Costs"
9. Short moves tend to reverse: momentum is right only 46-49% of the time
   from 5m to 4h
10. Even a small cost decides the best horizon. From 5m to 1h every model
    breaks even at 0.3 bp per trade or less, so at 2 bp all of them lose
    money; at 4h the best break-even cost is 1.6 bp. Only 1d survives: at
    5 bp, always buying on BTC (2.1 bp per trade, from the uptrend) and the
    logistic regression on ETH (4.4 bp) still make money
    - Data: `tradability.models.ipynb`, section "Cost sweep"
11. The gaps cost little data directly (0.07% of the bars, at most 0.65% of
    the labels at 1d), but 4.0% of the bars have no features, because the
    7-day windows restart after each gap. 7 of the 15 gaps start at 02:00
    UTC, and there is none after 2023-03-24
    - Data: `tradability.gaps.ipynb`
12. The step 2 average hides two markets. The logistic regression has no
    edge in calm and normal markets, but in volatile ones (about 30% of
    2022-2026) its PnL per trade grows with the horizon: 1.7-2.2 bp at 1h,
    7.7-10.9 bp at 4h (2.6-3.5 standard errors from zero on both coins), and
    28-35 bp at 1d (noisier). At 4h it is about the 10 bp cost. It is
    positive in 9 of 10 coin-years at 4h, but smaller in 2026
    - Data: `tradability.regimes.ipynb`
13. After a rising week, always buying makes a little more at every horizon,
    and short moves reverse in every regime
14. The step 2 conclusion does not depend on the target. Trained on the
    sign weighted by |r|, on the moves bigger than 10 bp only, or on the
    return itself, the model still has a weighted hit rate of 0.50 +/- 0.01
    from 5m to 1h, the best horizon is still 1d, and nothing makes money at
    10 bp. Training on the large moves only gives almost the same trades;
    the weighted and return targets are worse at 4h and 1d
    - Data: `tradability.targets.ipynb`

## Next

1. Robustness, part 2: does the volatile-market result hold with the other
   targets? (#905)
2. Step 4: trade only in volatile markets, walk forward, with costs

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
| 2026-10-30 | Step 1 features and labels |
| 2026-10-31 | Step 2 models per horizon, with costs |
| 2026-11-07 | PR #820 merged. GP answered the open questions in the gdoc, and on #733: "worth a publication if we can keep working a bit on it after class" |
| 2026-11-10 | To-do tab in the gdoc; filed #904 and #905; SPY 1h file saved in the project Drive folder |
| 2026-11-10 | Follow-ups from GP's answers: stale bars, SPY overnight bar, cost sweep, data sources |
| 2026-11-11 | Gap analysis (#904) |
| 2026-11-11 | PR #908 merged ("Very well done") |
| 2026-11-12 | Step 3: results by regime |
| 2026-11-12 | Robustness: the step 2 model trained on other targets (#905) |

## Open Work

- A wrong call loses the move, or counts as zero? (`D4`, GP is
  double-checking)
- Let the 7-day feature windows run across gaps of a few hours, to get back
  most of the 4% of bars without features? (#904)
- Treat the zero-volume bar on 2021-01-11 08:05 (both coins) as missing?
  (#904)

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
