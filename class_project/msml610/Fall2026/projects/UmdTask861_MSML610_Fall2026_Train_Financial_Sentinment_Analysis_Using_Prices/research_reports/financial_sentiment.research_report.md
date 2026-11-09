# Price-derived financial sentiment: real-data development report

MSML610, Group 19 — local evidence as of October 21, 2026.

Two separate development studies are reported: the original A 2016-2020 six-control slice, and the additionally registered 2023 A/AMZN/MSFT four-control slice (272 train / 154 test across separate company models). Neither is certified point-in-time alpha. The later section explains source failures, different date coverage and the unchanged original strict target.

Latest validation: 389 research tests passed in 5.644 seconds. The original A notebook completed eight code cells twice; the additional 2023 three-company notebook completed six code cells. The original 373-test comparison checkpoint remains preserved in its receipt.

The bounded end-to-end slice now runs from archived real headlines and opening prices through chronological weak labels, six fixed model controls, cost scenarios and an executable notebook. It does not establish alpha. The original strict ten-session total-return target remains separate and has zero eligible labels in the audited archive.

## Dataset and availability

| Item | Verified count / meaning |
| --- | --- |
| Original A headlines | 356 selected from 8,922 complete CSV records in a 2 MiB prefix |
| Source dates | 181 reported publication dates, 2016–2020 |
| Canonical article URLs | 356; no duplicate URL/title in this selected archive |
| Title relevance | 188 headlines explicitly mention Agilent; vendor tags for the rest are not independent relevance labels |
| Observed company-day outcomes | 180 |
| Paired train/test | 91 / 87 |
| Common exclusions | 2 early company-days without both prior 5/20-session momentum features |
| Price bars | A 1,144; SPY 1,144; complete requested pagination, unique observed calendar endpoints |
| Strict original eligible labels | 0 |

The fixed FNSPID revision is `bf9189c41527198897d1af3e17b1a0095279fc45`. Only original headlines are inputs, not supplied summaries, sentiment labels or article bodies. Captures occurred in 2026. Historical headline versions, availability at the reported date, independent company relevance and full-market coverage are unknown. The source prefix is alphabetically limited and must not be treated as the full corpus. A denotes Agilent Technologies (NYSE); the existing protocol retains issuer mapping references and the direct snapshot's HTTP 403 limitation.

## Label contract and chronology

The retrospective target is `(company exit open / company entry open - 1) - (SPY exit open / SPY entry open - 1)`. These split-adjusted prices exclude cash dividends. Date-only reported news enters on the second following XNYS session and exits ten sessions later. Classes are -1 below -0.005, +1 above +0.005, otherwise 0. This is price-derived weak supervision, not text-tone sentiment. It is not the original opening-price total-return target.

The fixed cutoff is 2019-01-01. Training entries span 2016-02-18–2018-11-26, with the last training outcome on 2018-12-11. Test entries span 2019-01-07–2020-06-09, with the last outcome on 2020-06-23. Training outcomes crossing the boundary are purged. Canonical URL and nearby-headline proxy groups are formed on the full graph, including excluded bridges; shared retained train/test groups are zero. No additional fixed-length embargo is imposed. Proxy grouping does not certify every real-world event identity.

All 87 test labels are used for predictive metrics. They overlap, so they are not independent observations. For cost compounding, 25 fixed non-overlapping intervals are selected; 62 overlapping rows are not compounded. No IID significance, Sharpe, annualized performance or causal attribution is claimed.

## Fixed model comparison

All models use the same 91 training and 87 test company-days. Price inputs use only prior-session opening momentum, never the entry open or future prices. TF-IDF vocabulary/IDF, scalers and Ridge models are fitted on training only. Alpha is fixed at 10 and no test-set hyperparameter search was performed. The transformer remains frozen; only its return calibration is fitted.

FinBERT consumes the same 338 distinct original headlines (352 headline uses) as the paired TF-IDF pool. Per-headline tone is `P(positive)-P(negative)`, averaged within company-day. All inputs were untruncated. The two additional models are tone-only Ridge and price-plus-tone Ridge. The FinBERT softmax is not a stock-rise probability. This additional comparator was registered after the original A test results had already been inspected; it is an inspected development comparison, not a fresh blind holdout. Historical model availability and pretraining overlap were not independently verified.

| Model | MAE (percentage points) | RMSE (percentage points) | Direction accuracy | Class balanced accuracy | Class macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train_mean | 2.5764 | 3.4473 | 0.5977 | 0.3333 | 0.1036 |
| price_ridge | 2.6543 | 3.5217 | 0.5632 | 0.3136 | 0.2888 |
| text_tfidf_ridge | 2.6013 | 3.4584 | 0.5862 | 0.2936 | 0.1299 |
| price_text_ridge | 2.6850 | 3.5377 | 0.5977 | 0.2520 | 0.2372 |
| finbert_tone_ridge | 2.7058 | 3.5875 | 0.5057 | 0.3295 | 0.2395 |
| price_finbert_ridge | 2.8376 | 3.7207 | 0.5517 | 0.2302 | 0.2297 |

The training-mean predictor has the lowest MAE in this sample. FinBERT does not improve error; the combined FinBERT/price model is worse. TF-IDF does slightly better than the price-only model on MAE, but not than the mean control. These data do not establish incremental market predictive skill. No winner was promoted into an automated trading system.

The pinned model is `Xenova/finbert`, revision `8f269abebfdd9009d7d9b5e96af7e5c6bfe50b20`, quantized ONNX on local CPU (`onnxruntime 1.22.1`). Model SHA256: `56cd19c7068ca5670c326816845bb7d7edda508b68a228697a1460d0e821ce76`. The original offline Hub loading eventually completed after a local-library read delay. The added explicit-cache path was actually rerun and returned exactly equal per-headline probabilities, aggregates and metadata. No model or dependency was downloaded.

## Illustrative cost evaluation

These are cumulative simulated price returns over the same 25 sparse intervals, excluding dividends. Positive predicted relative return selects long company, otherwise cash. Positions close between intervals; cash earns no interest. There are no shorts or borrow costs. They are not continuous daily portfolio returns or account PnL. Costs are assumed round-trip execution scenarios split over both sides, not verified Alpaca fees. Actual broker charges are unknown.

| Model/control | Roundtrips | Gross (0 bps) | Assumed 10 bps round trip | Assumed 20 bps round trip |
| --- | ---: | ---: | ---: | ---: |
| train_mean | 25 | 37.1334% | 33.7467% | 30.4421% |
| price_ridge | 19 | 0.0458% | -1.8376% | -3.6865% |
| text_tfidf_ridge | 24 | 37.2817% | 34.0253% | 30.8446% |
| price_text_ridge | 18 | 20.0009% | 17.8597% | 15.7556% |
| finbert_tone_ridge | 23 | 39.5449% | 36.3712% | 33.2682% |
| price_finbert_ridge | 18 | 2.9539% | 1.1169% | -0.6883% |
| always_company | 25 | 37.1334% | 33.7467% | 30.4421% |
| always_spy | 25 | 19.1789% | 16.2357% | 13.3637% |
| cash | 0 | 0.0000% | 0.0000% | 0.0000% |

Tone-only's positive simulated return does not override its worse prediction error. It selects 23/25 company intervals and has substantial long-market exposure. With one company, inspected outcomes and sparse overlapping events, a positive return or difference from always-long is insufficient evidence of alpha.

## Predeclared expansion and failed coverage

The pool A, AMZN, MSFT was fixed before inspecting expanded performance. Dates, cutoff, delay, horizon, alpha and the maximum 32 MiB source-prefix budget were fixed. PLTR remains in the prospective pool because it lacks comparable 2016–2020 listing history.

| Symbol | Window headlines in the 32 MiB prefix | Reported source days | Train / test company-days | Result |
| --- | ---: | ---: | --- | --- |
| A | 356 | 181 | 91 / 87 paired | Existing inspected slice |
| AMZN | 268 | 35 | 0 / 26 | Only 2020-04-27–2020-06-02; insufficient training rows; no fitted expanded performance |
| MSFT | 0 | 0 observed | Unavailable | Not reached by bounded prefix; not evidence that MSFT had no news |

The prefix contains 141,696 complete records; its incomplete final record is discarded. AMZN/SPY price pages are complete with 1,144 bars each. The AMZN baseline command refused insufficient training rows before publishing an evaluation directory. Its audit and retained raw capture remain available. Obtaining earlier AMZN/MSFT coverage needs a newly registered source query/byte budget, not a cutoff changed to make these test-only rows trainable. No expanded performance was used to select companies.

## Reproduction and daily use

From the Quant repository root, the already-tested manual daily command is:

```bash
./research/financial_sentiment_using_prices/run_daily.sh --overlap-minutes 1440
```

First use looks back three calendar days; later runs resume the checkpoint with 24-hour overlap and end at actual invocation time minus 20 minutes for the existing SIP access. Exact aware `--start-at`/`--end-at` past windows are supported; reruns preserve versions and cannot roll the checkpoint back. Post-16:00, overnight and weekend news are retained. Daily bars refresh from New York midnight independently of the intraday news window. Failed/capped news pagination leaves the checkpoint unchanged. Four real runs retained 31 unique article IDs and 22 bodies; repeated IDs are not new independent data. This is a manual collector, not an installed scheduler or an automatically validated cross-company predictor.

Offline baseline replay:

```bash
python3.11 research/financial_sentiment_using_prices/run_historical.py \
  --input-dir reports/financial_sentiment_runs/historical_development_20261021/capture \
  --output-dir reports/financial_sentiment_runs/my_new_baseline
```

The research notebook `financial_sentiment.research.ipynb` and paired `.py` rerun the frozen baseline, recompute all 180 labels from raw endpoints, assert split/group isolation, verify identical FinBERT inputs/receipts, refit both calibrated controls and export figures to a fresh private run. It needs the installed research dependencies and retained private archive; it does not silently download missing data. `comparison_protocol_20261021.json` records the additional comparison and expanded pool. The English README supplies the prepare/infer/evaluate commands for a new offline tone run.

## Course requirement check and remaining delivery gaps

The [class guide](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/class_project/README.md) requires local Python/Jupyter work, Docker execution, prescribed issue/branch/PR organization and a 10–20 minute walkthrough video. The listed grading categories total 40 points: completeness 10 and six 5-point categories covering Docker, documentation, complexity, code, PRs and understanding. The [research idea](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/research/ideas/draft.Train_Financial_Sentinment_Analysis_Using_Prices.md) asks for later-period price-label comparisons including FinBERT, momentum and Loughran–McDonald, multiple horizons and move attribution with an unexplained option and independently checked false attribution.

| Requirement | Current evidence | Remaining work |
| --- | --- | --- |
| Real data / chronological labels | Reproducible retrospective price-return slice and strict audit | Original total-return evidence; broader dated text coverage |
| Trained price-label model | Training-only TF-IDF/Ridge and numeric calibration | Transformer fine-tuning aspiration remains unfulfilled |
| Fair controls | Fixed six-model same-headline comparison | Versioned authorized Loughran–McDonald dictionary; frozen horizon comparison |
| Attribution | Existing review/prototype only | Predeclared large-move set, unexplained class and independent gold annotations |
| Notebook / documentation | New research notebook, paired Python, real negative-result report | Broader final study and independent clean-environment reproduction |
| Docker | Existing Dockerfile/scripts | Docker executable unavailable on this Mac; additional `Dockerfile.research` prepared with the research dependencies but build/run not verified |
| Course issue / PR | Group 19 selection identified | Formal issue number/branch assignment unverified; no PR created in this task |
| Video / submission | Execution evidence available | Owner presentation, 10–20 minute recording and authorized submission |

The owner-provided November 4 checkpoint and end-November target are tracked in the Chinese primary plan; the course guide delegates dates to the class schedule. Group 19 is not assumed to be issue #19. The project is not yet a complete final submission.

## Evidence files

Local evidence root: `reports/financial_sentiment_runs/course_delivery_20261021/`.
- `tone_request/`, `tone_inference/`, `tone_inference_cached/`: exact budget, model identity, probabilities, hashes and actual inference completion timestamps.
- `comparison/`: six-model report, predictions, fitted calibration parameters, paired losses and provenance.
- `expansion_amzn_capture/`: complete requested price pagination and bounded public source receipt.
- `expansion_amzn_audit.json`: honest zero-training expansion failure.
- `notebook_delivery/` and `notebook_delivery_v2/`: executed notebook and HTML (v2 improves figure layout); per-execution run contains raw-price label checks, figures and receipt.
- `full_research_tests.log` and `delivery_receipt.json`: final local validation and file hashes.

Raw text, price responses and model caches remain private; derived report tables do not redistribute the corpus. No push, PR, trade, paid resource or new credential was used in this task.


## Additional 2023 development study: completed, separately registered

The original 2016-2020 expanded source query remains unsuccessful. This additional study was registered at 2026-10-21T19:06:46.788993+00:00 from observed source coverage, before new price queries/outcomes. A/AMZN/MSFT, source dates 2023-03-09 through 2023-12-01 exclusive and a 2023-09-01 split were fixed. The same two-session delay, ten-session price-return target, alpha10 and original four controls were used. This date choice was not based on model performance. It does not replace the original experiment, strict total-return target or a final blind holdout.

### Source retrieval and failed original coverage

The pinned All_external.csv file is 5,731,397,037 bytes. Exact retained ranges show the local A block reproduces the original 356 selected titles, AMZN's local block contains only the same 268 April-June 2020 titles, and MSFT is absent from the located ordered region. The pinned nasdaq_exteral_data.csv is 23,232,979,597 bytes. Its actual company boundary dates are A 2022-09-11 to 2023-12-16, AMZN 2023-03-09 to 2023-12-16, MSFT 2022-04-26 to 2023-12-16. All three original date regions are unavailable in that source. These records are retained as coverage failures, not evidence that those companies had no news.

The extractor verifies whole multiline CSV records, actual byte positions, local ticker/date order, exact Content-Range, hashes and neighboring boundaries. The second file has a floating leading CSV index. Blank ticker tags are an explicit missing partition and never company samples. Ambiguous records, contradictory order, failed responses or exhausted registered budgets fail with retained evidence. Global corpus ordering/completeness is not independently certified.

The final finite aggregate ceiling was 256 MiB and 200 range responses, with 96 MiB per declared company/date region, registered before full second-source retrieval. Actual aggregate transfer was **107,723,488 bytes in 139 responses** across both files, counting earlier attempts once and preserving reused original receipts. The final second-source archive accounts for 99,786,823 bytes. No full multi-gigabyte corpus download, credentials or paid service were used. Raw article bodies are retained only as unavoidable bytes within source records and are not model inputs.

### Sample denominators and validation

There are **9,419 headline-company records, 8,294 distinct canonical URLs overall, and 1,125 URLs associated with multiple companies**. They are not 9,419 independent economic events. All URLs are unique within each company. Vendor ticker tags are not independent relevance annotations; literal issuer-name mentions occur in 80 A titles, 1,402 AMZN titles and 1,325 MSFT titles, which is a simple count, not a relevance gold label. No post-result relevance filter was applied.

| Company | Headline records | Observed day outcomes | Paired train | Paired test | Cross-cutoff training outcomes removed | Non-overlapping cost intervals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 195 | 85 | 52 | 28 | 5 | 6 |
| AMZN | 4422 | 181 | 110 | 61 | 10 | 7 |
| MSFT | 4802 | 185 | 110 | 65 | 10 | 7 |

All **451** observed relative-price labels and prior-price features were independently recomputed from retained raw bars in the executed expansion notebook. Removing 25 immature training outcomes gives **272 training + 154 test** rows across separate company evaluations. Each company and SPY has 255 daily bars with complete requested pagination. There is no price-endpoint or prior-momentum exclusion in this study. The original strict track still has zero eligible labels.

| Company | First training entry | Last training outcome | First test entry | Last test entry | Last test outcome |
| --- | --- | --- | --- | --- | --- |
| A | 2023-03-13 | 2023-08-31 | 2023-09-05 | 2023-11-28 | 2023-12-12 |
| AMZN | 2023-03-13 | 2023-08-31 | 2023-09-01 | 2023-12-04 | 2023-12-18 |
| MSFT | 2023-03-13 | 2023-08-31 | 2023-09-01 | 2023-12-04 | 2023-12-18 |

All retained training outcomes precede the September 1 boundary. Per-company connected proxy groups shared by retained train/test are zero, and direct source IDs shared across combined train/test are also zero. Proxy event identities are not independently certified. Models are fitted separately per company; local proxy identifiers are not a joint cross-company event graph. Test labels overlap, the companies share market exposure, and no combined independence, significance or capital-return claim is made. There is no extra fixed embargo.

### Frozen predictive comparisons

| Model | A MAE (pp) | AMZN MAE (pp) | MSFT MAE (pp) |
| --- | ---: | ---: | ---: |
| train_mean | 5.6557 | 3.2371 | 2.5281 |
| price_ridge | 5.3556 | 3.5316 | 2.4826 |
| text_tfidf_ridge | 5.5324 | 3.1994 | 2.5195 |
| price_text_ridge | 5.2875 | 3.4728 | 2.4740 |

The combined model has slightly lower MAE than price-only for each company, but AMZN's training-mean and text-only controls are substantially better than its combined model. Gains for MSFT are small. These are descriptive development results with final historical text, short tests and overlapping labels; they do not establish alpha. All companies and controls are reported. Expanded FinBERT/LM and transformer fine-tuning were not run in this step, and the old A FinBERT experiment was not repeated. Full RMSE, direction/class metrics, class counts and per-sample predictions remain in the JSON reports.

### Separate company cost scenarios

Returns below are cumulative simulated split-adjusted price returns for each company's own fixed sparse intervals. They are not pooled portfolio returns, verified fills or account PnL. Round-trip 0/10/20bp scenarios are assumed execution costs, not actual broker charges; cash dividends, interest, borrow/shorts and annualization are excluded.

| Company / model | Roundtrips | Gross 0bp | Assumed 10bp round trip | Assumed 20bp round trip |
| --- | ---: | ---: | ---: | ---: |
| A / train_mean | 0 | 0.0000% | 0.0000% | 0.0000% |
| A / price_ridge | 0 | 0.0000% | 0.0000% | 0.0000% |
| A / text_tfidf_ridge | 0 | 0.0000% | 0.0000% | 0.0000% |
| A / price_text_ridge | 0 | 0.0000% | 0.0000% | 0.0000% |
| A / always_company | 6 | 5.5004% | 4.8692% | 4.2414% |
| A / always_spy | 6 | 1.5546% | 0.9470% | 0.3426% |
| A / cash | 0 | 0.0000% | 0.0000% | 0.0000% |
| AMZN / train_mean | 7 | 4.3491% | 3.6210% | 2.8976% |
| AMZN / price_ridge | 7 | 4.3491% | 3.6210% | 2.8976% |
| AMZN / text_tfidf_ridge | 7 | 4.3491% | 3.6210% | 2.8976% |
| AMZN / price_text_ridge | 7 | 4.3491% | 3.6210% | 2.8976% |
| AMZN / always_company | 7 | 4.3491% | 3.6210% | 2.8976% |
| AMZN / always_spy | 7 | 1.8668% | 1.1561% | 0.4499% |
| AMZN / cash | 0 | 0.0000% | 0.0000% | 0.0000% |
| MSFT / train_mean | 7 | 11.9344% | 11.1534% | 10.3775% |
| MSFT / price_ridge | 7 | 11.9344% | 11.1534% | 10.3775% |
| MSFT / text_tfidf_ridge | 7 | 11.9344% | 11.1534% | 10.3775% |
| MSFT / price_text_ridge | 7 | 11.9344% | 11.1534% | 10.3775% |
| MSFT / always_company | 7 | 11.9344% | 11.1534% | 10.3775% |
| MSFT / always_spy | 7 | 1.8668% | 1.1561% | 0.4499% |
| MSFT / cash | 0 | 0.0000% | 0.0000% | 0.0000% |

The combined model takes no company position in A's selected windows. In AMZN and MSFT it holds the company in all seven selected intervals, so its simulated returns exactly equal always-company: at assumed 10bp round trip, 3.6210% and 11.1534%, respectively. These positive values therefore provide no incremental timing-return evidence from text. A's modeled combined return is 0%, versus 4.8692% for always-company under the same assumed cost.

### Reproduction, Docker blocker and current completion checklist

The new `financial_sentiment.expansion.ipynb` and paired Python file completed all six code cells offline. It recomputes the 451 labels and momentum features, checks chronology and event separation, reruns the fixed controls into new directories and verifies exact hashes for predictions, fitted models, company-days and splits against the first CLI evaluation. No new network request or credential access is made by the notebook. Its executed notebook/HTML, figures and receipt are under `reports/financial_sentiment_runs/expanded_ranges_20261021/`. The actual repository passed **389 tests in 5.644 seconds**, including 16 new range/CSV/coverage boundaries.

Docker was checked on PATH, common Docker/OrbStack locations and Unix sockets. The actual executable helper `run_course_container.sh --check` returned **127: no docker executable on PATH**; Bash syntax passed. No daemon, build, container execution, installation or permission/context change occurred. The prepared helper builds a unique local image, then disables runtime network and mounts only research/archives, without backend credentials. Docker build/run remains unverified.

Useful commands from the repository root (output directories must be new):

```bash
./research/financial_sentiment_using_prices/run_course_container.sh --check
./research/financial_sentiment_using_prices/run_daily.sh --overlap-minutes 1440
python3.11 -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=120 \
  --output-dir reports/financial_sentiment_runs/my_expansion_export \
  research/financial_sentiment_using_prices/financial_sentiment.expansion.ipynb
```

Remaining course completion items:
- Original strict total-return corporate-action, text-version and availability evidence.
- A genuinely frozen later evaluation, fair expanded FinBERT/LM and horizon comparisons.
- Transformer fine-tuning if feasible locally; no such training is claimed here.
- Independently annotated attribution with an unexplained class and false-attribution evaluation.
- Docker build and notebook execution when an approved local runtime exists.
- Confirmed formal course issue/branch/PR, final presentation and 10-20 minute owner video.

The source-coverage gap now has a usable, explicitly separate three-company development slice. No original target was silently substituted, and no push, PR, trade, purchase, paid compute or new credential occurred.
