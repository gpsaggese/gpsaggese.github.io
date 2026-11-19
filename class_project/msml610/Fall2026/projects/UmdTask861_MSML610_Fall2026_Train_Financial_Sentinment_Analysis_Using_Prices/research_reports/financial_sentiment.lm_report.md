# Loughran–McDonald baseline: registered 2023 development comparison

October 21, 2026. The official project text explicitly names the
**Loughran–McDonald dictionary** in Milestone 2. Here “LM” means that financial
sentiment dictionary, rather than a generic language model. The independent
source check is the [official project idea](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/research/ideas/draft.Train_Financial_Sentinment_Analysis_Using_Prices.md#implementation-plan).
Its retrieved UTF-8 source is bound by SHA-256
`e5c032416b5bae4dbb1e4aab3731ad895e2de0442cac110b9c10571973677f2d`; it matches the previously captured
project source. No transformer fine-tuning is performed in this slice.

All **36 fixed company/horizon/target/pool cases completed in
3.16 seconds**. Two LM controls add 72 measured
model slots. Together with unchanged prior controls, 276 of 288 potential slots
have measured results; 12 prior full-pool five-session FinBERT slots remain
explicitly unavailable. LM has no unavailable company-day or selectively removed
row. Outcomes were already inspected in earlier development work; this additional
comparison is not a fresh blind test, point-in-time historical simulation or alpha
confirmation.

## Official source, permission and version

The [Notre Dame SRAF dictionary page](https://sraf.nd.edu/loughranmcdonald-master-dictionary/)
permits free academic research and directs commercial users to request a separate
license. This run is local MSML610 academic work. No commercial Quant.ai deployment
or redistribution permission is asserted. The raw dictionary, permission-page HTML
and source receipts stay in the ignored private experiment directory; they are not
bundled into application code, a public repository, an export or the old PDF.

The official page was updated March 2026 and directly links
`Loughran-McDonald_MasterDictionary_1993-2025.csv`. That exact public CSV was
downloaded anonymously at 2026-10-21T20:14:17.631046Z: 9,093,460 bytes,
SHA-256 `e2d1328682bab7d2187684fb9f5420bb730401c9eefc00daf835edd203f4859d`.
No login, new credential, payment, form submission or explicit legal-acceptance
step was used. The retrieval receipt binds request/final URLs, actual receipt
timestamps, status 200, byte counts and all source hashes. Reruns are offline.

The loaded master contains **86,553 words, 347 active positive and 2,345 active
negative words**. Current positive category flags are active; zero and negative
removed-category flags are inactive (7 positive and 10 negative removed flags).
No active positive/negative flag is later than 2023 in this file, but that does
**not** certify a 2023 dictionary snapshot: the current master and category
removals/versions can differ from what was available then. We do not infer or
backdate an as-of dictionary from the flags. Together with unknown historical
headline versions and later model availability, this remains retrospective research.

## Registration, score and train-only calibration

Registration `2026-10-21T20:15:49.625854+00:00`, SHA-256
`3b86571ca47601ebd56b86d2527dcbfa6eea0c3e43eefdf9e7492946fe7c6343`, was written before any new LM outcome was
examined. It fixes source hashes, score/aggregation, all sample bindings, model
parameters, coverage failures, costs and a 300-second analysis / 330-second outer
budget. No hyperparameter search, model download or tensor inference is run.

For each exact original headline:

1. Split into ASCII letter runs, uppercase, retaining runs of at least two letters.
   Do not stem, lemmatize, remove stopwords or use negation rules.
2. Count positive and negative token **occurrences** using current active flags.
   Repeated words count. The headline score is `(positive - negative) / tokens`.
3. Deduplicate identical titles within the saved company-day, preserving original
   order, and require their newline join to equal the existing text feature. Average
   those headline scores equally; do not weight by length or add article bodies.

This is the registered project scoring convention, not a claim that the course
or dictionary authors mandate this denominator. A title with tokens but no polarity
hits receives numeric zero with a `NoPolarityMatch` diagnostic. That absence is not
independent evidence of neutral meaning. A missing title, invalid mapping or a
zero-token title makes the company-day unavailable; any required unavailable row
makes both new LM controls unavailable for the entire case, while preserving all
existing rows/baselines. This fallback was tested and was not needed on real rows.

`lm_tone_ridge` uses the company-day LM score. `price_lm_ridge` combines it with
the same prior-open 5/20-session momentum inputs as the existing price Ridge.
Each company's horizon/target/pool has its own StandardScaler and Ridge alpha10,
LSQR tolerance1e-8, fitted only on training rows. The transformer is not trained;
the dictionary is fixed and is not learned from test data. The ±0.5% price-class
band applies to the selected price-response target, not LM prose tone.

## Unchanged dataset, coverage and split

The source remains A / AMZN / MSFT, March 9–December 1 exclusive, 2023, with a
September 1 cutoff and a two-session date-only entry delay. Both raw company and
SPY-excess split-adjusted daily `o` price returns are evaluated at 5/10/20 sessions.
These horizons are project choices. Cash dividends are excluded; the original
strict ten-session total-return track is unchanged and still has zero eligible
labels. Daily `t` certifies an aggregation date, not an exact auction/trade/fill
timestamp; historical source receipt/publication availability is not certified.

All prior row order, sample IDs, mature chronology, full-graph source/event proxy
purging, common-ID intersections, labels and prior predictions are hash-checked
and reused. Existing baselines are not refitted. Training results mature strictly
before `2023-09-01T00:00:00Z`. Horizon-specific train/test counts are 286/154 at
five sessions, 272/154 at ten, and 246/154 at twenty, with 14/25/51 maturity
exclusions from 451 observed company-days. Common comparisons remain exactly
246 train / 154 test at every horizon. Within-test outcomes overlap; no IID
significance, portfolio pooling or new horizon selection is claimed.

| Company | Source records | Distinct-title uses | Company-days | Available LM rows | No-hit title uses | No-hit company-days |
| --- | --- | --- | --- | --- | --- | --- |
| A | 195 | 194 | 85 | 85 | 148 | 54 |
| AMZN | 4422 | 4301 | 181 | 181 | 3126 | 2 |
| MSFT | 4802 | 4667 | 185 | 185 | 3340 | 1 |

There are **7,293 globally distinct titles** across all 451 company-days and
9,162 within-company-day distinct-title uses. **5,235 distinct titles
(71.78%) have no positive/negative hit**. This is a substantial
dictionary-coverage limitation, not an excuse to remove zero-score rows. The
57 no-hit company-days and all remaining rows stay in their original eligible
cases. Raw source records (9,419), distinct-title uses (9,162), unique texts
(7,293) and eligible model rows are different denominators. Prior FinBERT inference
covered its narrower eligible pool; its missing full-five-session controls are
kept unavailable rather than silently replaced by LM coverage.

## MAE and descriptive paired comparisons

MAE is in percentage points. Compare each model within the same company, target,
horizon and sample pool. Lower raw MAE at shorter horizons does not select a horizon.
All eight controls, RMSE, direction accuracy, class balanced accuracy/macro F1,
MAE divided by its own training-mean control, per-row absolute losses and descriptive
LM-minus-control loss differences are saved. There are no p-values from overlapping
labels and no performance-dependent redefinition of the pool.

### Horizon-specific rows

| Company | Sessions | Target | Mean | Price | TF-IDF | Price+TF-IDF | FinBERT | Price+FinBERT | LM | Price+LM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 5 | Excess | 3.6782 | 3.8773 | 3.7058 | 3.9285 | Unavailable | Unavailable | 3.6570 | 3.8481 |
| A | 5 | Raw | 4.1370 | 4.6760 | 4.1498 | 4.7469 | Unavailable | Unavailable | 3.9760 | 4.4783 |
| A | 10 | Excess | 5.6557 | 5.3556 | 5.5324 | 5.2875 | 5.5836 | 5.4223 | 5.6397 | 5.4085 |
| A | 10 | Raw | 6.3187 | 5.9988 | 6.2479 | 5.9804 | 6.2887 | 5.9960 | 6.2571 | 5.9904 |
| A | 20 | Excess | 10.8798 | 10.1901 | 10.3295 | 9.6350 | 10.9163 | 10.2684 | 10.8473 | 10.1915 |
| A | 20 | Raw | 11.4087 | 12.0276 | 11.3772 | 11.9476 | 11.5094 | 11.9937 | 11.4305 | 12.1111 |
| AMZN | 5 | Excess | 2.3806 | 2.6952 | 2.3460 | 2.6310 | Unavailable | Unavailable | 2.4691 | 2.7788 |
| AMZN | 5 | Raw | 3.4491 | 3.7240 | 3.4283 | 3.6993 | Unavailable | Unavailable | 3.4863 | 3.7830 |
| AMZN | 10 | Excess | 3.2371 | 3.5316 | 3.1994 | 3.4728 | 3.3011 | 3.7507 | 3.2700 | 3.5562 |
| AMZN | 10 | Raw | 4.5279 | 4.8793 | 4.5329 | 4.8318 | 4.5981 | 4.9648 | 4.5304 | 4.8789 |
| AMZN | 20 | Excess | 5.1135 | 4.9963 | 5.0656 | 4.9402 | 5.1228 | 5.0181 | 5.1563 | 5.0530 |
| AMZN | 20 | Raw | 6.2755 | 6.2369 | 6.2264 | 6.2001 | 6.2855 | 6.2744 | 6.3650 | 6.3173 |
| MSFT | 5 | Excess | 1.5862 | 1.5703 | 1.5710 | 1.5544 | Unavailable | Unavailable | 1.5819 | 1.5886 |
| MSFT | 5 | Raw | 2.3100 | 2.3523 | 2.2924 | 2.3381 | Unavailable | Unavailable | 2.3700 | 2.4289 |
| MSFT | 10 | Excess | 2.5281 | 2.4826 | 2.5195 | 2.4740 | 2.4791 | 2.4564 | 2.5030 | 2.4535 |
| MSFT | 10 | Raw | 3.7429 | 3.7447 | 3.7615 | 3.7568 | 3.7497 | 3.7550 | 3.7327 | 3.7219 |
| MSFT | 20 | Excess | 4.5199 | 5.7230 | 4.4711 | 5.6820 | 4.2663 | 5.3730 | 4.4320 | 5.4659 |
| MSFT | 20 | Raw | 4.9844 | 5.9516 | 4.9798 | 5.9084 | 4.7831 | 5.6864 | 4.7912 | 5.6097 |

### Common eligible rows

| Company | Sessions | Target | Mean | Price | TF-IDF | Price+TF-IDF | FinBERT | Price+FinBERT | LM | Price+LM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 5 | Excess | 3.6837 | 3.5944 | 3.6842 | 3.6155 | 3.7235 | 3.6839 | 3.6700 | 3.6315 |
| A | 5 | Raw | 4.0794 | 4.2286 | 4.0716 | 4.2783 | 4.0875 | 4.2083 | 3.9335 | 4.1687 |
| A | 10 | Excess | 5.6752 | 5.5502 | 5.5474 | 5.4667 | 5.6054 | 5.7092 | 5.6735 | 5.7507 |
| A | 10 | Raw | 6.2133 | 5.6765 | 6.1544 | 5.6748 | 6.2018 | 5.7073 | 6.1648 | 5.7305 |
| A | 20 | Excess | 10.8798 | 10.1901 | 10.3295 | 9.6350 | 10.9163 | 10.2684 | 10.8473 | 10.1915 |
| A | 20 | Raw | 11.4087 | 12.0276 | 11.3772 | 11.9476 | 11.5094 | 11.9937 | 11.4305 | 12.1111 |
| AMZN | 5 | Excess | 2.5011 | 3.0479 | 2.4710 | 2.9935 | 2.5020 | 3.0885 | 2.6291 | 3.1163 |
| AMZN | 5 | Raw | 3.5796 | 4.0381 | 3.5522 | 4.0082 | 3.5780 | 4.0678 | 3.6762 | 4.1045 |
| AMZN | 10 | Excess | 3.4829 | 3.8875 | 3.4375 | 3.8419 | 3.4948 | 4.0102 | 3.5655 | 3.9320 |
| AMZN | 10 | Raw | 4.8996 | 5.3615 | 4.8717 | 5.3319 | 4.9056 | 5.3914 | 4.9734 | 5.3761 |
| AMZN | 20 | Excess | 5.1135 | 4.9963 | 5.0656 | 4.9402 | 5.1228 | 5.0181 | 5.1563 | 5.0530 |
| AMZN | 20 | Raw | 6.2755 | 6.2369 | 6.2264 | 6.2001 | 6.2855 | 6.2744 | 6.3650 | 6.3173 |
| MSFT | 5 | Excess | 1.5799 | 1.5013 | 1.5629 | 1.4958 | 1.6140 | 1.6292 | 1.5810 | 1.5674 |
| MSFT | 5 | Raw | 2.3071 | 2.4010 | 2.2841 | 2.3904 | 2.3086 | 2.5103 | 2.3512 | 2.5029 |
| MSFT | 10 | Excess | 2.5311 | 2.4316 | 2.5159 | 2.4121 | 2.4609 | 2.3844 | 2.5091 | 2.3989 |
| MSFT | 10 | Raw | 3.7355 | 3.6798 | 3.7474 | 3.6820 | 3.7297 | 3.6673 | 3.7267 | 3.6612 |
| MSFT | 20 | Excess | 4.5199 | 5.7230 | 4.4711 | 5.6820 | 4.2663 | 5.3730 | 4.4320 | 5.4659 |
| MSFT | 20 | Raw | 4.9844 | 5.9516 | 4.9798 | 5.9084 | 4.7831 | 5.6864 | 4.7912 | 5.6097 |

LM changes are mixed. Full-pool ten-session excess `price_lm_ridge` MAE is A
5.4085 pp versus price 5.3556; AMZN 3.5562 versus 3.5316; MSFT 2.4535 versus
2.4826. On common rows, the corresponding values are A 5.7507 versus 5.5502,
AMZN 3.9320 versus 3.8875, MSFT 2.3989 versus 2.4316. MSFT's twenty-session
tone-only LM is 4.4320 pp versus mean 4.5199, while combined LM is 5.4659 pp.
These descriptive differences do not establish consistent incremental predictive
power, and the observed MSFT improvement does not validate a live strategy.

## Assumed transaction costs and exposure

Each case reuses exactly the previous greedy non-overlap interval IDs. All available
models use those intervals. Positive predictions hold the company, otherwise cash;
an excess prediction does not create a SPY hedge. Active windows close/reopen with
factor `(1 - assumed_roundtrip_bps/20000)^2 × (1 + company price return)`.
Costs are **assumed** 0/10/20 round-trip bps; actual broker commissions, regulatory
charges, spread, borrow/financing and fill quality are unknown. No dividend or cash
interest is included. Interval counts and holding exposure differ by horizon, so
cumulative values do not rank horizons. Companies are not pooled into one capital curve.

The following uses horizon-specific excess labels. Values are cumulative simulated
price returns (%), with active round trips. The full 1,152 model/cost scenarios,
including raw targets, common rows, prior controls, always-company/SPY/cash and
held-session counts, remain in the CSV and JSON artifacts.

| Company | Sessions | LM control | Intervals | Active trips | Gross 0bp % | Net assumed 10bp % | Net assumed 20bp % | Price control 10bp % | Always-company 10bp % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 5 | lm_tone_ridge | 11 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 9.2179 |
| A | 5 | price_lm_ridge | 11 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 9.2179 |
| A | 10 | lm_tone_ridge | 6 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4.8692 |
| A | 10 | price_lm_ridge | 6 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4.8692 |
| A | 20 | lm_tone_ridge | 3 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5.4250 |
| A | 20 | price_lm_ridge | 3 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5.4250 |
| AMZN | 5 | lm_tone_ridge | 13 | 13 | 2.9364 | 1.6066 | 0.2933 | 1.8069 | 1.6066 |
| AMZN | 5 | price_lm_ridge | 13 | 10 | 1.6376 | 0.6260 | -0.3759 | 1.8069 | 1.6066 |
| AMZN | 10 | lm_tone_ridge | 7 | 7 | 4.3491 | 3.6210 | 2.8976 | 3.6210 | 3.6210 |
| AMZN | 10 | price_lm_ridge | 7 | 7 | 4.3491 | 3.6210 | 2.8976 | 3.6210 | 3.6210 |
| AMZN | 20 | lm_tone_ridge | 4 | 4 | 10.1144 | 9.6747 | 9.2365 | 9.6747 | 9.6747 |
| AMZN | 20 | price_lm_ridge | 4 | 4 | 10.1144 | 9.6747 | 9.2365 | 9.6747 | 9.6747 |
| MSFT | 5 | lm_tone_ridge | 13 | 11 | 5.3117 | 4.1594 | 3.0191 | 4.9004 | 9.1774 |
| MSFT | 5 | price_lm_ridge | 13 | 8 | 10.4600 | 9.5797 | 8.7059 | 4.9004 | 9.1774 |
| MSFT | 10 | lm_tone_ridge | 7 | 7 | 11.9344 | 11.1534 | 10.3775 | 11.1534 | 11.1534 |
| MSFT | 10 | price_lm_ridge | 7 | 6 | 8.9375 | 8.2857 | 7.6374 | 11.1534 | 11.1534 |
| MSFT | 20 | lm_tone_ridge | 4 | 3 | 14.1982 | 13.8560 | 13.5147 | 6.6760 | 12.3413 |
| MSFT | 20 | price_lm_ridge | 4 | 2 | 12.0812 | 11.8572 | 11.6336 | 6.6760 | 12.3413 |

Sparse examples include MSFT five-session combined LM (9.5797% at assumed10bp,
8 active intervals versus always-company9.1774% over13) and twenty-session LM
tone (13.8560% over3 versus always-company12.3413% over4). These are reported
alongside weaker cases: MSFT ten-session combined LM8.2857% versus price/always-company
11.1534%; AMZN five-session combined LM0.6261% versus price1.8069%. They are
development interval diagnostics with different exposure, unverified execution
prices and small samples, not confirmation of stable alpha. Prior baseline cost
results and predictions remain numerically identical.

## Reproduce, validation and remaining work

From the repository root, with the existing research Python:

```bash
RESEARCH_PYTHON=/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11
"$RESEARCH_PYTHON" research/financial_sentiment_using_prices/lm_baseline.py \
  --repo "$PWD" \
  --registration research/financial_sentiment_using_prices/lm_2023_protocol.json \
  --output-dir reports/financial_sentiment_runs/my_new_LM_run
```

The versioned private dictionary/receipts and prior fixed archives must already
exist. Existing output paths are rejected; reruns do not download a dictionary or
accept terms. **422 research tests pass** (14 new LM tests). The independent
`financial_sentiment.lm.ipynb` verifies source hashes and exact scoring, training-only
calibration, sample alignment, all old controls and costs, then reproduces the
fixed 36-case comparison in a fresh private directory. Executed notebook/HTML and
validation receipt are separate artifacts. All **7 code cells executed in 7.86 seconds with zero errors**,
and all 36 predictions, fitted calibrations, metrics, paired losses, costs and
company-day/headline scores reproduced exactly. No previous report or PDF is rewritten.

Remaining course requirements: price-label transformer fine-tuning; independently
hand-labeled large-move attribution evaluation with unexplained cases and false
attribution rates; Docker execution and owner-authorized course narrative/video and
formal submission. Original strict research additionally needs real publication/
version clocks, total-return corporate-action evidence and mature prospective
labels. This bounded LM slice does not begin those extra experiments or imply
commercial license, blind-test or profitability completion.
