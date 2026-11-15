# Registered 5/10/20-session development comparison

Generated October 21, 2026 from the fixed 2023 A / AMZN / MSFT archive. The 36
registered company × horizon × target × pool combinations completed in
19.23 seconds. No new data requests, tensor inference,
model downloads, parameter search or company selection occurred. This is additional
development analysis; previous ten-session outcomes were already known. It provides
neither a fresh blind confirmation nor established alpha.

## Protocol and relation to the course

The [course idea](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/research/ideas/draft.Train_Financial_Sentinment_Analysis_Using_Prices.md#implementation-plan)
asks for multiple horizons and raw/excess price-response comparisons. **5, 10 and
20 sessions are this project's registered choices; the course does not prescribe
that list.** The independent registration timestamp is `2026-10-21T19:54:50.430657+00:00`,
before the new five/twenty-session results. Its SHA-256 is
`f37d525426e10e87d23651866976d8e4235ccb57a71e87cc61f401ed1eb7031d`. The company pool, March 9–December 1 exclusive
source window, September 1 cutoff, two-session entry delay and fixed model
parameters match the existing 2023 development study. All three original full-pool
ten-session excess-target results are reused after exact split/input/hash checks.
Those files and the previously delivered PDF remain unchanged.

Two explicitly named targets are evaluated separately:

* `raw_price_return`: company daily-bar `o` at exit divided by entry `o`, minus one.
* `excess_price_return`: that raw return minus SPY's return on the same endpoints.

Both use observed provider split-adjusted prices and **exclude cash dividends**.
Neither substitutes for the unchanged strict ten-session total-return target,
which still has zero eligible labels. Classes are below −0.5%, within ±0.5%, and
above +0.5% for the selected target. They describe price responses, not prose tone.
FinBERT's positive-minus-negative score is a text feature calibrated only on training
rows. Date-only headlines enter on the second following XNYS session. Historical
text versions and publication-time availability remain unverified; actual capture
receipts are from 2026, and no historical capture timestamp has been invented.

## Coverage, prices and split maturity

The unchanged archive contains 9,419 vendor-tagged headline-company records and
451 company-days. All 451 have both observed endpoints and prior momentum features
for each horizon; none is excluded for missing endpoints or calendar coverage.
The horizon-specific reductions below are solely outcomes that cross the training
cutoff. All training exits are strictly before `2023-09-01T00:00:00Z`; test entries
are September 1 or later. Source/event proxy groups are formed on the complete graph,
including excluded bridging rows, before train/test purging. No proxy group crosses
the retained within-company split. There is no additional fixed embargo. This is
an outcome-maturity purge, not a proof that test observations are independent.

| Company | Sessions | Observed days | Train | Test | Maturity excluded | Last train outcome | Last test outcome |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 5 | 85 | 56 | 28 | 1 | 2023-08-30 | 2023-12-05 |
| A | 10 | 85 | 52 | 28 | 5 | 2023-08-31 | 2023-12-12 |
| A | 20 | 85 | 46 | 28 | 11 | 2023-08-30 | 2023-12-27 |
| AMZN | 5 | 181 | 115 | 61 | 5 | 2023-08-31 | 2023-12-11 |
| AMZN | 10 | 181 | 110 | 61 | 10 | 2023-08-31 | 2023-12-18 |
| AMZN | 20 | 181 | 100 | 61 | 20 | 2023-08-31 | 2024-01-03 |
| MSFT | 5 | 185 | 115 | 65 | 5 | 2023-08-31 | 2023-12-11 |
| MSFT | 10 | 185 | 110 | 65 | 10 | 2023-08-31 | 2023-12-18 |
| MSFT | 20 | 185 | 100 | 65 | 20 | 2023-08-31 | 2024-01-03 |

Totals are **286/154 train/test at five sessions; 272/154 at ten; 246/154 at twenty**.
Thus 440/426/400 used observations are distinct from the 451 observed labels. The
common-eligible pool intersects train IDs and test IDs separately across all three
horizons before fitting: A 46/28, AMZN 100/61 and MSFT 100/65, totaling 246/154.
All three common comparisons therefore use exactly the same rows and text/price
inputs, but different horizon targets. No intersection uses returns, metrics or
FinBERT cache availability. Test IDs happen to be unchanged here; training IDs shrink.
Within-test labels overlap; no IID significance test or annualized performance is reported.

Each retained capture has 255 company bars plus 255 SPY bars and an independent
255-session calendar, January 9, 2023–January 12, 2024. Observed pagination terminates
and no retained calendar date is missing for either symbol. Hashes bind every source,
calendar, protocol and price-page file. Daily timestamps are aware, unique, at New
York midnight, with both DST offsets audited. Calendar logical entry/exit clocks are
09:30 New York. However, [Alpaca's aggregation documentation](https://docs.alpaca.markets/us/docs/market-data-faq#how-are-bars-aggregated)
defines the daily timestamp as the day anchor and `o` through eligible-trade update
rules. These bars do **not certify the exact trade timestamp, official opening auction
price or an executable fill**. Endpoints align to scheduled session dates; price
execution-time certification is false. This limitation applies equally to the retained
ten-session result and is explicit rather than silently changing its prices.

## Models and cache coverage

The four fixed controls are training mean, 5/20-session prior-open momentum Ridge,
headline TF-IDF Ridge, and combined momentum/TF-IDF Ridge. TF-IDF vocabulary/IDF and
all scalers are fitted on training data only: max 2,000 features, min_df 2, ngrams
1–2, Ridge alpha 10, LSQR tolerance 1e-8. There is no tuning across horizons.

Two added controls use the previously saved quantized Xenova/finbert receipts:
tone-only Ridge and price-plus-tone Ridge, with separate training-only calibration.
The cached model revision is `8f269abebfdd9009d7d9b5e96af7e5c6bfe50b20`.
Inputs are exact original headlines, averaged within company-day after deduplicating
identical titles. No truncation is accepted and no new inference is run.

For full five-session training, A has 4 uncovered rows / 10 unique title hashes;
AMZN 5 / 94; MSFT 5 / 85. Both FinBERT controls are marked **Unavailable for the
entire corresponding company/horizon pool** (for both targets). All four baselines
retain every eligible row. The ten/twenty-session and all common-pool comparisons
have complete cached coverage. No difficult row is selectively discarded. The
machine-readable metrics table preserves all 216 potential model slots, including
12 unavailable slots; 204 slots have measured results.

## Prediction error

MAE below is in percentage points, separately by target. Raw MAE naturally grows
with horizon; the smaller five-session values do not select a preferred horizon.
Every model should be compared with its same-horizon, same-pool training-mean
control. RMSE, direction accuracy, class balanced accuracy, macro F1 and MAE/control
ratios are retained in `delivery/metrics.csv` and the per-case JSON reports.

### Horizon-specific eligible rows

| Company | Sessions | Target | Mean | Price | TF-IDF | Price+TF-IDF | FinBERT | Price+FinBERT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 5 | Excess | 3.6782 | 3.8773 | 3.7058 | 3.9285 | Unavailable | Unavailable |
| A | 5 | Raw | 4.1370 | 4.6760 | 4.1498 | 4.7469 | Unavailable | Unavailable |
| A | 10 | Excess | 5.6557 | 5.3556 | 5.5324 | 5.2875 | 5.5836 | 5.4223 |
| A | 10 | Raw | 6.3187 | 5.9988 | 6.2479 | 5.9804 | 6.2887 | 5.9960 |
| A | 20 | Excess | 10.8798 | 10.1901 | 10.3295 | 9.6350 | 10.9163 | 10.2684 |
| A | 20 | Raw | 11.4087 | 12.0276 | 11.3772 | 11.9476 | 11.5094 | 11.9937 |
| AMZN | 5 | Excess | 2.3806 | 2.6952 | 2.3460 | 2.6310 | Unavailable | Unavailable |
| AMZN | 5 | Raw | 3.4491 | 3.7240 | 3.4283 | 3.6993 | Unavailable | Unavailable |
| AMZN | 10 | Excess | 3.2371 | 3.5316 | 3.1994 | 3.4728 | 3.3011 | 3.7507 |
| AMZN | 10 | Raw | 4.5279 | 4.8793 | 4.5329 | 4.8318 | 4.5981 | 4.9648 |
| AMZN | 20 | Excess | 5.1135 | 4.9963 | 5.0656 | 4.9402 | 5.1228 | 5.0181 |
| AMZN | 20 | Raw | 6.2755 | 6.2369 | 6.2264 | 6.2001 | 6.2855 | 6.2744 |
| MSFT | 5 | Excess | 1.5862 | 1.5703 | 1.5710 | 1.5544 | Unavailable | Unavailable |
| MSFT | 5 | Raw | 2.3100 | 2.3523 | 2.2924 | 2.3381 | Unavailable | Unavailable |
| MSFT | 10 | Excess | 2.5281 | 2.4826 | 2.5195 | 2.4740 | 2.4791 | 2.4564 |
| MSFT | 10 | Raw | 3.7429 | 3.7447 | 3.7615 | 3.7568 | 3.7497 | 3.7550 |
| MSFT | 20 | Excess | 4.5199 | 5.7230 | 4.4711 | 5.6820 | 4.2663 | 5.3730 |
| MSFT | 20 | Raw | 4.9844 | 5.9516 | 4.9798 | 5.9084 | 4.7831 | 5.6864 |

### Common eligible rows: identical IDs at every horizon

| Company | Sessions | Target | Mean | Price | TF-IDF | Price+TF-IDF | FinBERT | Price+FinBERT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 5 | Excess | 3.6837 | 3.5944 | 3.6842 | 3.6155 | 3.7235 | 3.6839 |
| A | 5 | Raw | 4.0794 | 4.2286 | 4.0716 | 4.2783 | 4.0875 | 4.2083 |
| A | 10 | Excess | 5.6752 | 5.5502 | 5.5474 | 5.4667 | 5.6054 | 5.7092 |
| A | 10 | Raw | 6.2133 | 5.6765 | 6.1544 | 5.6748 | 6.2018 | 5.7073 |
| A | 20 | Excess | 10.8798 | 10.1901 | 10.3295 | 9.6350 | 10.9163 | 10.2684 |
| A | 20 | Raw | 11.4087 | 12.0276 | 11.3772 | 11.9476 | 11.5094 | 11.9937 |
| AMZN | 5 | Excess | 2.5011 | 3.0479 | 2.4710 | 2.9935 | 2.5020 | 3.0885 |
| AMZN | 5 | Raw | 3.5796 | 4.0381 | 3.5522 | 4.0082 | 3.5780 | 4.0678 |
| AMZN | 10 | Excess | 3.4829 | 3.8875 | 3.4375 | 3.8419 | 3.4948 | 4.0102 |
| AMZN | 10 | Raw | 4.8996 | 5.3615 | 4.8717 | 5.3319 | 4.9056 | 5.3914 |
| AMZN | 20 | Excess | 5.1135 | 4.9963 | 5.0656 | 4.9402 | 5.1228 | 5.0181 |
| AMZN | 20 | Raw | 6.2755 | 6.2369 | 6.2264 | 6.2001 | 6.2855 | 6.2744 |
| MSFT | 5 | Excess | 1.5799 | 1.5013 | 1.5629 | 1.4958 | 1.6140 | 1.6292 |
| MSFT | 5 | Raw | 2.3071 | 2.4010 | 2.2841 | 2.3904 | 2.3086 | 2.5103 |
| MSFT | 10 | Excess | 2.5311 | 2.4316 | 2.5159 | 2.4121 | 2.4609 | 2.3844 |
| MSFT | 10 | Raw | 3.7355 | 3.6798 | 3.7474 | 3.6820 | 3.7297 | 3.6673 |
| MSFT | 20 | Excess | 4.5199 | 5.7230 | 4.4711 | 5.6820 | 4.2663 | 5.3730 |
| MSFT | 20 | Raw | 4.9844 | 5.9516 | 4.9798 | 5.9084 | 4.7831 | 5.6864 |

For the excess target, common-pool MAE divided by its own training-mean MAE is:

| Company | Sessions | Mean | Price | TF-IDF | Price+TF-IDF | FinBERT | Price+FinBERT |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 5 | 1.0000 | 0.9758 | 1.0001 | 0.9815 | 1.0108 | 1.0001 |
| A | 10 | 1.0000 | 0.9780 | 0.9775 | 0.9633 | 0.9877 | 1.0060 |
| A | 20 | 1.0000 | 0.9366 | 0.9494 | 0.8856 | 1.0033 | 0.9438 |
| AMZN | 5 | 1.0000 | 1.2187 | 0.9880 | 1.1969 | 1.0004 | 1.2349 |
| AMZN | 10 | 1.0000 | 1.1162 | 0.9870 | 1.1031 | 1.0034 | 1.1514 |
| AMZN | 20 | 1.0000 | 0.9771 | 0.9906 | 0.9661 | 1.0018 | 0.9813 |
| MSFT | 5 | 1.0000 | 0.9502 | 0.9892 | 0.9467 | 1.0215 | 1.0312 |
| MSFT | 10 | 1.0000 | 0.9607 | 0.9940 | 0.9530 | 0.9722 | 0.9421 |
| MSFT | 20 | 1.0000 | 1.2662 | 0.9892 | 1.2571 | 0.9439 | 1.1887 |

The changes are inconsistent across company, target and horizon. For example,
common-pool MSFT price-plus-FinBERT improves ten-session excess MAE to 2.3844 pp
from the 2.5311 pp mean control, but its twenty-session MAE is 5.3730 pp versus
4.5199 pp. MSFT twenty-session tone alone is 4.2663 pp; A twenty-session combined
TF-IDF is 9.6350 pp versus the 10.8798 pp mean. These are inspected development
comparisons with overlapping labels and small company-specific test sets. No
model/horizon is promoted as a selected, confirmed predictor, and a lower prediction
error is not evidence of incremental trading return.

## Transaction-cost sensitivity and overlap

For each company/horizon/pool/target, test rows are sorted by entry index and sample
ID. The existing greedy selector keeps a row only after the prior selected exit;
adjacent exit/entry endpoints are allowed. Every available model and always-company,
SPY and cash uses the same selected intervals. A positive prediction holds the
company, otherwise cash; excess-target signs do not imply a SPY hedge. Each active
interval is fully closed before the next. Costs are **assumed** 0/10/20 round-trip
bps, split equally over entry and exit, with factor
`(1 - bps/20000)^2 × (1 + observed price return)`. Actual broker commissions,
regulatory/borrow/financing charges and spreads are unknown; these are not Alpaca
account deductions. Cash interest and dividends are excluded.

The full-pool excess-target illustrative table below reports cumulative simulated
price return (%) with active round trips and session exposure. It is a sparse
interval diagnostic, not a portfolio backtest. Five/ten/twenty-session selectors
cover different numbers of intervals and endpoints; their cumulative returns are
not a valid horizon ranking. Both named targets and both pools, all available
controls, cash and all cost scenarios are exported to `delivery/cost_scenarios.csv`.

| Company | Sessions | Control | Intervals | Active trips | Held sessions | Gross 0bp % | Net assumed 10bp % | Net assumed 20bp % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 5 | price_text_ridge | 11 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| A | 5 | always_company | 11 | 11 | 55 | 10.4262 | 9.2179 | 8.0222 |
| A | 5 | always_spy | 11 | 11 | 55 | 3.2807 | 2.1506 | 1.0322 |
| A | 10 | price_text_ridge | 6 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| A | 10 | always_company | 6 | 6 | 60 | 5.5004 | 4.8692 | 4.2414 |
| A | 10 | always_spy | 6 | 6 | 60 | 1.5546 | 0.9470 | 0.3426 |
| A | 20 | price_text_ridge | 3 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| A | 20 | always_company | 3 | 3 | 60 | 5.7419 | 5.4250 | 5.1090 |
| A | 20 | always_spy | 3 | 3 | 60 | 0.3258 | 0.0252 | -0.2746 |
| AMZN | 5 | price_text_ridge | 13 | 11 | 55 | 2.9332 | 1.8069 | 0.6923 |
| AMZN | 5 | always_company | 13 | 13 | 65 | 2.9364 | 1.6066 | 0.2933 |
| AMZN | 5 | always_spy | 13 | 13 | 65 | 0.4612 | -0.8367 | -2.1184 |
| AMZN | 10 | price_text_ridge | 7 | 7 | 70 | 4.3491 | 3.6210 | 2.8976 |
| AMZN | 10 | always_company | 7 | 7 | 70 | 4.3491 | 3.6210 | 2.8976 |
| AMZN | 10 | always_spy | 7 | 7 | 70 | 1.8668 | 1.1561 | 0.4499 |
| AMZN | 20 | price_text_ridge | 4 | 4 | 80 | 10.1144 | 9.6747 | 9.2365 |
| AMZN | 20 | always_company | 4 | 4 | 80 | 10.1144 | 9.6747 | 9.2365 |
| AMZN | 20 | always_spy | 4 | 4 | 80 | 4.9143 | 4.4953 | 4.0779 |
| MSFT | 5 | price_text_ridge | 13 | 12 | 60 | 6.1697 | 4.9030 | 3.6508 |
| MSFT | 5 | always_company | 13 | 13 | 65 | 10.6064 | 9.1774 | 7.7663 |
| MSFT | 5 | always_spy | 13 | 13 | 65 | 0.4612 | -0.8367 | -2.1184 |
| MSFT | 10 | price_text_ridge | 7 | 7 | 70 | 11.9344 | 11.1534 | 10.3775 |
| MSFT | 10 | always_company | 7 | 7 | 70 | 11.9344 | 11.1534 | 10.3775 |
| MSFT | 10 | always_spy | 7 | 7 | 70 | 1.8668 | 1.1561 | 0.4499 |
| MSFT | 20 | price_text_ridge | 4 | 2 | 40 | 12.0812 | 11.8572 | 11.6336 |
| MSFT | 20 | always_company | 4 | 4 | 80 | 12.7916 | 12.3413 | 11.8925 |
| MSFT | 20 | always_spy | 4 | 4 | 80 | 4.9143 | 4.4953 | 4.0779 |

At ten sessions the previously observed combined TF-IDF exposure remains all cash
for A and always-company for AMZN/MSFT. At other horizons exposures change, but no
registered criterion or independent sample establishes persistent incremental
economic benefit. Overlapping company-day labels are never compounded together;
companies are not pooled into a fictitious capital curve.

## Reproduce and deliver

From the repository root, using the existing research Python environment:

```bash
python research/financial_sentiment_using_prices/multi_horizon.py \
  --repo . \
  --registration research/financial_sentiment_using_prices/multi_horizon_2023_protocol.json \
  --output-dir reports/financial_sentiment_runs/NEW_PRIVATE_MULTI_HORIZON_RUN
```

Choose a new output directory; an existing path is rejected. No credentials or
network calls are required. The separate `financial_sentiment.multi_horizon.ipynb`
audits hash-bound archives, independently recomputes endpoints and maturity, repeats
the fixed offline fit in a fresh private directory, and checks predictions and
metrics against this saved run. All **7 code cells executed without error in 22.15 seconds**;
all 36 predictions/metrics/cost comparisons reproduced exactly. The executed notebook
and HTML are delivered with the CSV tables. **408 research tests passed** (12 added horizon tests). The original
strict module, prior report/PDF and ten-session artifacts are checked unchanged.
Docker remains absent; this is local execution, not a container-run certification.

## Remaining course work and independent research gates

1. Implement and test the requested Loughran–McDonald lexicon baseline and the
   price-label text-training/fine-tuning component described in the course idea.
   The current TF-IDF predictor is learned; FinBERT is frozen tone plus Ridge,
   not a transformer fine-tuned on price labels.
2. Build the separate large-move attribution evaluation: predeclare a move threshold
   and preceding-news window, rank candidate news, allow unexplained moves, and
   hand-label an evaluation event-day set independently of model outputs. Report
   unexplained fraction and false-attribution rate. The idea's 3-sigma example is
   not a compulsory threshold; current URL/title proxy groups are not that gold set.
3. For the owner's original strict research target, resolve actual publication/version
   evidence and opening-price total-return corporate-action evidence; preserve daily
   capture receipts and wait for labels to mature. Historical retrospective results
   cannot certify this target or trading alpha.
4. Execute Docker/course packaging after the owner provides the permitted Docker
   environment, then finish required narrative/video and owner-authorized formal
   submission. No installation, issue, push or PR is performed here.

This bounded multi-horizon step is complete. The remaining items are explicit
requirements and evidence gates, rather than a claim that the whole course project
or a profitable trading system is finished.
