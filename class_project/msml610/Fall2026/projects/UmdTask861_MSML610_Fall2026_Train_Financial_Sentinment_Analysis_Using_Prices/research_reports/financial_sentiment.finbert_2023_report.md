# Registered expanded FinBERT comparison on the fixed 2023 rows

This additional development comparator was registered at 2026-10-21T19:33:26.395104+00:00, after the original four-model development results had been inspected and before new FinBERT inference/evaluation. It is not a fresh blind test. The exact existing sample order, split/news/prediction hashes, date boundary, titles and calibration settings were frozen. No company, row or hyperparameter was changed from the registered pool after results.

| Company | Train / test | Distinct required headlines | Required headline uses | Delivered paired rows | Coverage failures / truncation |
| --- | --- | ---: | ---: | ---: | --- |
| A | 52 / 28 | 168 | 178 | 80 | 0 / none |
| AMZN | 110 / 61 | 3726 | 4075 | 171 | 0 / none |
| MSFT | 110 / 65 | 4034 | 4457 | 175 | 0 / none |

All **426** original paired company-days (272 train / 154 test) are covered. The **8,710 headline uses** are the same distinct-title company-day input budget as TF-IDF after its original common exclusions, not all 9,419 raw headline-company records. There are 7,928 distinct headlines summed separately by company. A shared within-run cache required **6,948 actual unique text inferences**, which is a compute optimization, not another fitted cross-company price model or a count of independent events. No difficult row was removed.

The local cached quantized Xenova/finbert revision and all three weight/tokenizer/config hashes were checked before inference. Mean positive-minus-negative tone was calibrated with separate training-only StandardScaler and Ridge(alpha10) models, with and without the unchanged prior-price momentum. The transformer remains frozen; its softmax is not a stock-return probability. No project-specific transformer fine-tuning is claimed. Exact historical model availability and pretraining overlap remain unverified.

Real inference completed in **128.94 seconds**, within the fixed shared 900-second wall budget (930-second outer process cap), using four CPU threads. Every input was untruncated, with a maximum 55 tokens against the 512-token model limit. No model download, network inference, new credential, paid compute or trade was used. Seven new artificial boundary tests cover cache reuse, budget exhaustion, partial output, changed packets, whole-company failure and old-output preservation. **396 research tests passed in 9.549 seconds**.

The failure rule was fixed before inference: any invalid, missing, truncated or budget-incomplete required headline makes the entire company's added comparator Unavailable; the old four baselines and full denominator remain. Partial headline receipts and the reason are retained. Actual coverage failures in this run were zero.

### Same-row six-control errors

| Model | A MAE (pp) | AMZN MAE (pp) | MSFT MAE (pp) |
| --- | ---: | ---: | ---: |
| train_mean | 5.6557 | 3.2371 | 2.5281 |
| price_ridge | 5.3556 | 3.5316 | 2.4826 |
| text_tfidf_ridge | 5.5324 | 3.1994 | 2.5195 |
| price_text_ridge | 5.2875 | 3.4728 | 2.4740 |
| finbert_tone_ridge | 5.5836 | 3.3011 | 2.4791 |
| price_finbert_ridge | 5.4223 | 3.7507 | 2.4564 |

Adding FinBERT to price features worsens A and AMZN error. MSFT MAE is slightly lower: 2.4564pp versus 2.4826pp for price-only and 2.4740pp for price-plus-TF-IDF. This inspected, small, overlapping retrospective test does not establish significance, stable predictive skill or alpha. All companies and controls remain reported; there was no winner-based promotion or retuning. Complete RMSE, direction/class metrics, fitted coefficients/scalers, predictions and paired losses are retained in each company comparison JSON.

### Unchanged intervals and illustrative costs

Both added controls use the original 6 / 7 / 7 non-overlapping per-company intervals; the other overlapping test rows are not compounded. There is no pooled capital curve. Values below are simulated split-adjusted price returns, excluding cash dividends; assumed round-trip costs are illustrative, not actual broker fees. Actual charges remain unknown.

| Company / control | Roundtrips | Gross 0bp | Assumed 10bp | Assumed 20bp |
| --- | ---: | ---: | ---: | ---: |
| A / finbert_tone_ridge | 0 | 0.0000% | 0.0000% | 0.0000% |
| A / price_finbert_ridge | 0 | 0.0000% | 0.0000% | 0.0000% |
| A / always_company | 6 | 5.5004% | 4.8692% | 4.2414% |
| A / always_spy | 6 | 1.5546% | 0.9470% | 0.3426% |
| AMZN / finbert_tone_ridge | 7 | 4.3491% | 3.6210% | 2.8976% |
| AMZN / price_finbert_ridge | 7 | 4.3491% | 3.6210% | 2.8976% |
| AMZN / always_company | 7 | 4.3491% | 3.6210% | 2.8976% |
| AMZN / always_spy | 7 | 1.8668% | 1.1561% | 0.4499% |
| MSFT / finbert_tone_ridge | 7 | 11.9344% | 11.1534% | 10.3775% |
| MSFT / price_finbert_ridge | 7 | 11.9344% | 11.1534% | 10.3775% |
| MSFT / always_company | 7 | 11.9344% | 11.1534% | 10.3775% |
| MSFT / always_spy | 7 | 1.8668% | 1.1561% | 0.4499% |

Both new controls stay in cash on A's selected intervals. On AMZN/MSFT they hold the company in all seven intervals, exactly matching always-company exposure and returns at every cost scenario. Therefore the positive simulated returns provide no incremental text-timing evidence.

The updated `financial_sentiment.expansion.ipynb` appends this comparison to the original six-cell baseline and reproduces all nine code cells offline from saved tone receipts; no tensor inference or network access occurs in notebook replay. Each added model's recorded training IDs and feature means are checked against training rows only. Six-model predictions and fitted calibration files must exactly match the first evaluation. Older executed notebooks remain untouched. Current outputs are under `reports/financial_sentiment_runs/finbert_2023_20261021/`; `comparison_summary.json`, per-company `comparison/`, `inference_manifest.json`, raw tone receipts and the bounded-process log retain actual outcomes.

### Exact course-scope check: horizons and attribution

The [official research idea](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/research/ideas/draft.Train_Financial_Sentinment_Analysis_Using_Prices.md#implementation-plan), retrieved October 21, is marked draft. Its implementation plan (source lines 96-116) calls for labels across multiple horizons using both raw and excess returns, later-period comparisons with FinBERT/LM/momentum, and a predictive-power table across horizons. It does **not** prescribe exact 5/10/20-session choices. The present 2023 comparison covers only the fixed ten-session excess-price-return target; it does not complete that sensitivity study.

Attribution requires large-move detection, preceding-news ranking, an unexplained option, checking against hand-labeled known event days, and unexplained/false-attribution rates. Annotator count and blinding are not specified. Independent model-blinded gold annotation is the project's quality requirement, not an invented quotation; assistant self-review is not independent validation. No attribution experiment started in this slice.

Source Git blob: `cb30c4ac326c9a45382af956ecc1a6ffd57efb02`; exact source/hash/line evidence is retained privately. These chosen-project specifications are distinct from the [generic grading rubric](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/class_project/README.md#grading-rubric). Docker execution, formal course submission and video remain unresolved; no installation or issue/PR submission occurred. Original strict total-return evidence, LM, horizons, transformer fine-tuning and independently evaluated attribution remain unfinished.
