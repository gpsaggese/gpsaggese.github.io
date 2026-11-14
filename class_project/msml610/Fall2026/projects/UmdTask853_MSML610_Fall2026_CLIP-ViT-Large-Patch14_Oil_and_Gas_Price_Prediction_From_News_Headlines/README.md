# When the News Breaks: Forecasting Oil and Natural Gas Returns with Frozen CLIP

**MSML610 · Advanced Machine Learning · Fall 2026**  
**Group 18 · Individual tutorial project · Author: Shashank Ashoka (`@shashk09-coder`)**  
**Course project:** [Project 4 — Oil and Gas Price Prediction From News Headlines](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/class_project/project_descriptions/MSML610/CLIP-ViT-Large-Patch14_Project_Description.md) · **[Master issue #853](https://github.com/gpsaggese/gpsaggese.github.io/issues/853)** · **[Implementation PR #863](https://github.com/gpsaggese/gpsaggese.github.io/pull/863)**

## The question

Oil and gas prices react to information that is not visible in yesterday's closing price. A production cut, a pipeline outage, an unexpected inventory report, or a geopolitical escalation can change the market's view of tomorrow. But an appealing headline is not necessarily a useful forecasting signal: publication times are messy, stories are repeated, and daily commodity returns are notoriously difficult to predict.

I started from the simpler question: **how far can we get without reading the news?** I then built two headline datasets, learned frozen text representations with **CLIP ViT-L/14**, and asked whether those representations add anything to a price-only model under a chronological evaluation. The aim is an executable *Learn CLIP in 60 Minutes* tutorial and a carefully controlled forecasting example—not a claim that CLIP can reliably predict commodity markets.

> **Main result.** On the 2010–2021 news-overlap experiment, **FNSPID + CLIP PCA** reduced WTI test RMSE from **0.040257** (price-only Ridge) to **0.040131** (**0.31% relative improvement**, 581 test observations). Its paired bootstrap RMSE-gain interval, **[−0.000070, +0.000303]**, includes zero. For Henry Hub, the best observed headline improvement was **ABC + FinBERT PCA** (0.087308 → 0.087241; **0.08%**), also inconclusive. CLIP did **not** improve Henry Hub RMSE. These are exploratory findings, not evidence of a robust forecasting edge.

**Contents:** [Price data and baselines](#1-first-establish-a-price-only-baseline) · [ABC headlines](#2-abc-news-what-an-oil-keyword-does-not-tell-us) · [FNSPID](#3-fnspid-from-stock-news-mentions-to-unique-energy-headlines) · [CLIP and fusion](#4-how-clip-enters-the-forecast) · [Results](#5-results-abc-versus-fnspid) · [Reproduction](#7-reproduce-the-experiments) · [Next steps](#8-next-steps-toward-a-live-multimodal-system)

---

## 1. First establish a price-only baseline

### 1.1 Prices and target construction

I retrieved historical daily **WTI crude oil**, **Brent crude oil**, and **Henry Hub natural gas** spot-price observations from FRED/EIA. These are three related but distinct series: WTI is the primary crude-oil target, Henry Hub is the natural-gas target, and Brent provides an additional crude-oil benchmark.

| Commodity | FRED series | Role | Input data |
|:--|:--|:--|:--|
| WTI crude | [`DCOILWTICO`](https://fred.stlouisfed.org/series/DCOILWTICO) | Primary target | Daily spot prices |
| Brent crude | [`DCOILBRENTEU`](https://fred.stlouisfed.org/series/DCOILBRENTEU) | Additional price-only benchmark | Daily spot prices |
| Henry Hub natural gas | [`DHHNGSP`](https://fred.stlouisfed.org/series/DHHNGSP) | Second target | Daily spot prices |

After parsing dates, sorting observations, and treating missing/nonpositive quotes as invalid, I construct the log return between **consecutive available observations**:

$$
r_t=\log(P_t)-\log(P_{t-1}),\qquad y_t=r_{t+1}.
$$

The prediction features are formed only from information known at the forecast origin: lagged returns (up to 30 observations in the extended benchmark), five-observation momentum, and trailing 20-observation volatility. Returns are defined over **observed price quotes**, not necessarily consecutive calendar days; this matters around weekends and holidays. Evaluation uses chronological train/validation/test splits with a five-observation embargo at split boundaries.

**Implementation:** [`forecast_utils.py`](forecast_utils.py) · [`run_forecasting.py`](run_forecasting.py) · [`price_only_benchmarks/benchmark_models.py`](price_only_benchmarks/benchmark_models.py) · [`price_only_benchmarks/forecast_utils.py`](price_only_benchmarks/forecast_utils.py)

### 1.2 Eleven price-only forecasting models

The price-only experiment was deliberately broad. A low-capacity linear model is a useful reference only if it is tested against stronger nonlinear and temporal alternatives.

| Family | Model | What it tests |
|:--|:--|:--|
| Naive | Zero return | Whether predicting no change is sufficient |
| Naive | Historical training mean | Constant-return reference; denominator for out-of-sample R² |
| Linear | Ridge | Regularized autoregression on engineered price features |
| Linear | ElasticNet | Sparse/regularized linear feature selection |
| Tree ensemble | HistGradientBoosting | Nonlinear thresholds in lag/momentum/volatility features |
| Tree ensemble | XGBoost | Boosted nonlinear interactions |
| Tree ensemble | LightGBM | Efficient gradient-boosted regression |
| Sequence | LSTM | Recurrent price-history dependence |
| Sequence | GRU | Gated recurrence with fewer parameters |
| Sequence | TCN | Dilated temporal convolutional dependencies |
| Transformer | **PatchTST-inspired model** | Patch-based temporal encoding of price sequences |

The compact PatchTST-inspired implementation is **not a reproduction of the complete published PatchTST system**. It borrows the idea of tokenizing a time series into patches from **Nie et al., [*A Time Series Is Worth 64 Words: Long-term Forecasting with Transformers* (ICLR 2023)](https://arxiv.org/abs/2211.14730)**. See [`price_only_benchmarks/patchtst_model.py`](price_only_benchmarks/patchtst_model.py) and the [implementation notes](price_only_benchmarks/PATCHTST_README.md).

**Broader price-only benchmark — selected lowest-RMSE models per asset**

| Asset | Lowest observed test-RMSE model | RMSE (log-return units) |
|:--|:--|--:|
| WTI | **Ridge** | **0.031203** |
| Brent | **TCN** | **0.036206** |
| Henry Hub | **LSTM** | **0.109875** |

These results come from a **separate, broader price-only evaluation**. They are *not* directly comparable to the 2010–2021 ABC/FNSPID benchmark below: the data windows and test observations differ. Full per-model metrics should be exported from the local benchmark results rather than interpolated from this summary. The complete model implementations, including model-specific evaluation scripts, live under [`price_only_benchmarks/`](price_only_benchmarks/); results are kept locally instead of committing generated tables or datasets to Git.

The deeper lesson from this stage was methodological: **model complexity alone does not guarantee lower return-forecast error**. A fair news comparison needs the *same target dates* and a strong, reproducible price-only Ridge reference.

---

## 2. ABC News: what an “oil” keyword does not tell us

The first textual source was [*A Million News Headlines*](https://www.kaggle.com/datasets/therohk/million-headlines): **1,244,184 Australian Broadcasting Corporation (ABC) headlines**, spanning approximately **2003–2021**. The source is broad general news, not a dedicated energy feed. Its useful forecasting representation is essentially a **two-field row (publication date, headline text)**, without a dependable intraday release time.

A basic `oil|gas` search is deceptively noisy. For example, a story about **olive oil** contains the word *oil* but is not, by itself, evidence about crude supply or WTI prices. The same problem arises with cooking oil, essential oils, and figurative uses of *gas*. Conversely, a war or a shipping interruption may matter enormously for energy prices **without mentioning oil in the headline**. This creates a precision–recall trade-off that a regular expression cannot eliminate.

### 2.1 A layered filter, not a single keyword

1. **Candidate retrieval:** Search case-insensitively with word boundaries for oil-, gas-, and market-specific terminology: `crude`, `WTI`, `Brent`, `OPEC`, `petroleum`, `refinery`, `LNG`, `natural gas`, `Henry Hub`, pipelines, drilling, and related terms. Broader geopolitical and disruption terms are retained as **candidates for further review**, not automatically treated as energy stories.
2. **Contextual disambiguation:** Reject clearly unrelated uses such as *olive oil* or *essential oils* when no energy context is present; do not blindly exclude whole stories based on a single word because a mixed-context headline can still be relevant.
3. **Direct-energy selection:** Favor explicit energy-market context for the main reproducible experiment. Keep broad geopolitical headlines in a separate audit pool so speculative relevance does not silently pollute the labels.
4. **Temporal normalization and deduplication:** Normalize headline text, retain the available publication date, and remove duplicates using the source-specific audit rules.
5. **Leakage-safe alignment:** Treat date-only news as available **after** its publication date, rather than guessing that a story was seen before the market's observation cutoff.

Examples below illustrate the distinction; they are **examples of the rule**, not claims that these exact strings occurred in the source:

| Headline pattern | Decision | Reason |
|:--|:--|:--|
| “Olive oil prices rise after poor harvest” | Exclude from direct crude/gas pool | Agricultural food oil, not petroleum |
| “OPEC announces crude production cut” | Include as direct energy | Explicit petroleum supply event |
| “LNG export terminal shuts after outage” | Include as direct energy | Explicit natural-gas infrastructure |
| “Regional conflict escalates near shipping route” | Broad/contextual review | Could be energy-relevant, but not demonstrable from keyword alone |

### 2.2 ABC audit results

| Stage | Headlines |
|:--|--:|
| Total ABC source corpus | **1,244,184** |
| Broad energy-related regex candidates | **~11,724** |
| Explicit oil/gas keyword union | **6,305** |
| Direct candidates after off-topic filtering | **5,987** |
| Selected headlines in the 2010–2021 aligned experiment | **2,970** |

The **5,987** count and the **2,970** count refer to different processing scopes: the former is a direct-energy candidate pool in the source audit; the latter is the selected subset used by the date-restricted encoding/forecasting run. They should not be presented as successive deletions from an identical time window.

**Code:** [`audit_abc_energy.py`](audit_abc_energy.py) for source inspection and filtering; [`news_fusion/prepare_news.py`](news_fusion/prepare_news.py) for the final date-restricted manifest. The ABC source is useful as a clean general-news baseline, but its sparse direct-energy coverage limits the number of forecast dates on which a headline embedding can contribute new information.

---

## 3. FNSPID: from stock-news mentions to unique energy headlines

To test whether coverage was the bottleneck, I evaluated an energy-focused subset of [FNSPID](https://huggingface.co/datasets/beachside1234/FNSPID), using a third-party hosted version of its stock-associated news tables. Two source Parquet tables contained approximately **13.06 million** and **15.55 million** rows. Rather than materializing the complete files locally, I used **DuckDB/HTTP streaming** to select news associated with **15 energy-company tickers** and then applied the energy-topic filter.

Unlike ABC’s date-and-headline rows, these are **stock-associated article records** that must be interpreted using their news text, date, and security linkage. FNSPID presents a different problem from ABC: a single article can be linked to multiple securities. **Rows are stock–article mentions, not necessarily distinct news stories.** Treating each mention as independent would overcount popular stories, inflate headline counts, and bias the daily embedding toward duplicated coverage.

### 3.1 Extraction and deduplication

| Stage | Records |
|:--|--:|
| First hosted stock-news table | **13,057,514** rows |
| Second hosted stock-news table | **15,549,299** rows |
| Rows attached to selected energy tickers | **88,333** |
| Raw energy-related candidate mentions | **23,671** |
| Unique article/date/headline candidates after deduplication | **12,602** |
| Unique direct-energy candidates | **12,205** |
| Broad/context-only candidates | **397** |

Of the **12,602** unique candidates, approximately **9,891** matched oil terminology and **3,417** matched gas terminology. These categories **overlap**, so they must not be summed to calculate the total. Deduplication removed **11,069** repeated candidate mentions.

The extracted unique-candidate set spans **2009–2023**, with **3,594** distinct direct-energy news days in the overall audit. This offers substantially better temporal density than ABC, but has an important selection bias: **news linked to energy equities is not a representative sample of all macroeconomic or commodity-market news**.

### 3.2 Publication-time caveat

Approximately **99.72%** of selected-ticker raw timestamps appeared at **00:00 UTC**, suggesting date-level placeholders rather than reliable publication times. I therefore **did not assume midnight was the real publication instant**. Headlines were aligned conservatively after their stated publication date. We cannot claim precise intraday causal ordering from this dataset.

**Code:** [`msml610_source_audit/fnspid_energy_extract.py`](msml610_source_audit/fnspid_energy_extract.py) · [`msml610_source_audit/audit_fnspid_gdelt.py`](msml610_source_audit/audit_fnspid_gdelt.py) · [FNSPID extraction notes](msml610_source_audit/FNSPID_ENERGY_EXTRACTION_README.md). I also audited [GDELT's archive](msml610_source_audit/gdelt_archive_audit.py): the pilot yielded four successful 15-minute snapshots, not continuous historical news coverage, so **GDELT was excluded from the quantitative forecasting comparison**.

---

## 4. How CLIP enters the forecast

CLIP is a **representation model**, not a price predictor. The deployed experiment uses the pretrained **[`openai/clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14)** text tower with its weights frozen; the downstream forecasting model learns how to combine the text signal with historical prices.

### 4.1 From headlines to a daily market feature

For each selected headline $h_i$, CLIP ViT-L/14's text encoder produces a **768-dimensional** representation $z_i$. The configured train-fitted PCA compresses these vectors to a much smaller number of features (approximately **20 components** in the course design). The encoder remains in evaluation mode, and feature extraction uses inference-only execution. The headline vectors are grouped by eligible date and pooled into a daily representation. **PCA is fitted on training observations only**, then applied to validation and test dates, so future information cannot influence the feature basis.

$$
\hat{r}_{t+1} =
g_{\mathrm{Ridge}}
\left(
\left[
x_t^{\mathrm{price}},
\mathrm{PCA}_{\mathrm{train}}
\left(z_t^{\mathrm{day}}\right)
\right]
\right)
$$

Here $x_t^{\mathrm{price}}$ contains lagged returns, momentum, and volatility. On days without eligible articles, the pipeline uses a consistent missing-news representation rather than inventing headlines. Importantly, the same historical price features and prediction dates are used for price-only and price-plus-news comparisons.

### 4.2 What the ablations test

| Variant | Purpose |
|:--|:--|
| Zero return / training mean | Sanity-check naive references |
| **Price-only Ridge** | Main baseline: how much history explains without text |
| Price + headline count | Whether *news activity* alone contributes |
| Price + TF–IDF | Whether conventional lexical features are sufficient |
| Price + FinBERT sentiment | Whether financial tone improves forecasts |
| Price + FinBERT CLS/PCA | Domain-oriented language representation comparator |
| **Price + CLIP/PCA** | Main frozen-CLIP headline embedding model |
| Price + full CLIP representation | Ablation of dimensionality reduction |
| Price + CLIP event-prompt scores | Direct test of interpretable event affinities |
| Shuffled CLIP dates | Negative control for coincidental text–return relationships |

CLIP prompt features use descriptions such as *oil supply disruption*, *weak oil demand*, and *OPEC production cut*. Similarity to a prompt reflects **semantic affinity**, not a probability of a particular price direction. Because CLIP's text encoder was trained jointly with images and image captions rather than energy-market reports, FinBERT is included as a domain-relevant comparator rather than assuming CLIP is superior by design.

**Code:** [`clip.API.ipynb`](clip.API.ipynb) for the introduction to frozen inference and prompt similarity · [`clip.example.ipynb`](clip.example.ipynb) for the applied demonstration · [`clip_features.py`](clip_features.py) for reusable text features · [`news_fusion/encode_news.py`](news_fusion/encode_news.py) for batched extraction · [`news_fusion/run_forecast.py`](news_fusion/run_forecast.py) for the common-date forecasting experiment.

---

## 5. Results: ABC versus FNSPID

### 5.1 A controlled comparison

Both news sources were evaluated over the **same 2010–2021 modeling window**, with a chronological split and a five-observation embargo. A source-comparison audit explicitly confirmed **identical target dates and realized returns** for ABC and FNSPID within each asset. Thus, differences in these tables reflect the text feature source/model rather than selecting an easier set of price outcomes.

| Forecast target | Train / validation / test | Held-out date range | Test observations |
|:--|:--|:--|--:|
| WTI next-observation return | 1,904 / 435 / 581 | 2019-07-25 – 2021-12-31 | **581** |
| Henry Hub next-observation return | 1,964 / 449 / 600 | 2019-08-13 – 2021-12-31 | **600** |

The two datasets cover different headline populations: **ABC is broad general news** and **FNSPID is filtered stock-linked financial news**. Dataset size alone is not a measure of information quality, but usable *news-day coverage* is an essential constraint.

| Test-set news coverage | ABC | FNSPID |
|:--|--:|--:|
| **WTI** | **91 / 581 days (15.66%)** | **525 / 581 days (90.36%)** |
| **Henry Hub** | **23 / 600 days (3.83%)** | **338 / 600 days (56.33%)** |

On the WTI test period, FNSPID supplies a usable text signal on **more than five times as many days** as ABC. For Henry Hub the difference is larger still; nevertheless, better coverage does not automatically yield lower error.

### 5.2 Held-out RMSE across all news-fusion variants

**Lower is better.** RMSE is reported in **raw log-return units**, not dollars per barrel or percentage classification accuracy. Every column shares the same test outcomes *within its asset*. Values are rounded to six decimals.

| Model | WTI + ABC ↓ | WTI + FNSPID ↓ | Henry Hub + ABC ↓ | Henry Hub + FNSPID ↓ |
|:--|--:|--:|--:|--:|
| Zero return | 0.040324 | 0.040324 | 0.087823 | 0.087823 |
| Historical mean | 0.040322 | 0.040322 | 0.087828 | 0.087828 |
| **Price-only Ridge** | **0.040257** | **0.040257** | **0.087308** | **0.087308** |
| Price + news count | 0.040287 | 0.040254 | 0.087302 | 0.087358 |
| Price + TF–IDF | 0.040285 | 0.040252 | 0.087300 | 0.087359 |
| Price + FinBERT sentiment | 0.040274 | 0.040250 | 0.087308 | 0.087372 |
| Price + FinBERT CLS/PCA | 0.040318 | 0.040281 | **0.087241** | 0.087321 |
| **Price + CLIP/PCA** | 0.040321 | **0.040131** | 0.087316 | 0.087486 |
| Price + full CLIP | 0.040323 | **0.040131** | 0.087316 | 0.087480 |
| Price + CLIP event prompts | 0.040280 | 0.040255 | 0.087306 | 0.087355 |
| Shuffled-date CLIP control | 0.040301 | 0.040239 | 0.087328 | 0.087526 |

**Main takeaway:** for WTI, the strongest point estimates come from **FNSPID + CLIP embeddings**. For Henry Hub, **the price-only model remains strongest on FNSPID**, while ABC + FinBERT gives a very small apparent gain despite sparse coverage. The result is **source-dependent and target-dependent**, not a blanket demonstration that semantic embeddings beat price history.

### 5.3 Magnitude, direction and uncertainty

Beyond RMSE, I tracked out-of-sample $R^2$ relative to the training-mean forecast and directional accuracy, then compared paired errors using bootstrap intervals. The correct question is not merely whether one RMSE is lower, but whether the difference could plausibly be sampling noise.

| Comparison | Price-only Ridge | News-augmented model | Relative RMSE change | OOS R² (news model) | 95% CI for absolute RMSE gain |
|:--|--:|--:|--:|--:|:--|
| **WTI · FNSPID · CLIP PCA** | 0.040257 | **0.040131** | **+0.313%** | 0.009458 | **[−0.000070, +0.000303]** |
| Henry Hub · ABC · FinBERT PCA | 0.087308 | **0.087241** | +0.077% | 0.013337 | [−0.000020, +0.000183] |
| WTI · ABC · CLIP PCA | **0.040257** | 0.040321 | −0.159% | 0.000056 | [−0.000143, +0.000020] |
| Henry Hub · FNSPID · CLIP PCA | **0.087308** | 0.087486 | −0.204% | 0.007785 | [−0.000509, +0.000085] |

A **positive RMSE gain** means news reduced forecasting error. Both positive-gain intervals **contain zero**, so the improvements are **not statistically resolved** at the 95% level in this exploratory comparison. Many representations were inspected on the same held-out interval; model selection on that test set can also bias the apparent winner. A future evaluation needs an untouched time period and, ideally, rolling-origin re-estimation.

The WTI directional-accuracy result is another caution: **price-only Ridge achieved about 51.22%**, while **FNSPID + CLIP PCA achieved about 48.78%** on the scored nonzero-direction cases. CLIP's small RMSE improvement did **not** translate into better prediction of the sign of a return. This is an error-reduction experiment, **not** a demonstrated trading strategy.

**Evaluation code:** [`news_fusion/run_forecast.py`](news_fusion/run_forecast.py) · [`news_fusion/compare_sources.py`](news_fusion/compare_sources.py) · [`plot_forecast_errors.py`](plot_forecast_errors.py). Generated CSVs, cached embeddings, and source datasets reside in local `results/` and `data/` directories and are not committed with the course project.

### 5.4 Interpretation

There are three distinct findings.

1. **Coverage matters for whether a text model can even contribute.** ABC provided eligible WTI headlines on only 15.66% of test dates; FNSPID provided them on 90.36%. On days without eligible news, the combined model has little new text information to use.
2. **CLIP can provide a modest additional continuous signal for WTI, but the evidence is weak.** The point-estimate RMSE difference on FNSPID is favorable and larger than the shuffled-date control, yet the bootstrap interval crosses zero and sign accuracy declines.
3. **The gas market does not behave like the oil market.** Greater FNSPID coverage for Henry Hub did not improve RMSE. The missing ingredient may be gas-specific information (storage, weather, LNG flows, regional transport constraints), not merely a larger or more general news corpus.

A stronger study would stratify errors by the presence of news, compare high-volatility days with ordinary days, audit the timing of each article, and reserve a genuinely untouched evaluation period. Those are **future validation steps**, not completed results.

---

## 6. What this project teaches about CLIP

The tutorial is built around a practical progression: **load a pretrained model → extract frozen representations → score event prompts → aggregate text over time → fit a small downstream model → test whether the representation adds information**. `clip.API.ipynb` isolates the API concepts; `clip.example.ipynb` connects them to a real forecasting problem.

CLIP's text-only use is intentionally a limited slice of its full capacity. Its original design learns a *joint* image–text representation. The present experiments use **news headlines only**; they do not yet consume article images, price-chart screenshots, or a live feed. Treating this as a baseline rather than an endpoint motivates the next stage.

---

## 7. Reproduce the experiments

### 7.1 Repository structure

```text
.
├── README.md
├── Dockerfile / requirements.txt / docker_*.sh
├── clip.API.ipynb                  # pretrained CLIP API and semantic probes
├── clip.example.ipynb              # applied tutorial
├── clip_features.py                # current reusable text feature code
├── audit_abc_energy.py             # ABC candidate audit
├── news_fusion/
│   ├── prepare_news.py             # source manifests and date alignment
│   ├── encode_news.py              # frozen CLIP / FinBERT feature extraction
│   ├── run_forecast.py             # joint evaluation and bootstrap
│   └── compare_sources.py         # common-target ABC/FNSPID comparison
├── price_only_benchmarks/          # linear, tree, recurrent and patch baselines
├── msml610_source_audit/          # FNSPID extraction and GDELT feasibility
├── test/                          # targeted implementation tests
└── progress.md                     # implementation log
```

The official project specification also calls for a **`clip_utils.py`** entry point. The current reusable code is in `clip_features.py`; a clean `clip_utils.py` interface and a fresh-container verification remain **submission tasks**, not claims of completed work.

### 7.2 Environment and execution

From the project directory in the course repository, use the provided course Docker/Jupyter scripts:

```bash
./docker_build.sh
./docker_jupyter.sh
```

In Jupyter, execute [`clip.API.ipynb`](clip.API.ipynb) before [`clip.example.ipynb`](clip.example.ipynb). The original API notebook demonstrates frozen text inference and prompt similarity. The applied pipeline requires separately obtained FRED prices and ABC/FNSPID source data; these source datasets and Hugging Face model weights are **not committed to GitHub**.

The data preparation, encoding, and forecasting CLIs expose their arguments through `--help`:

```bash
python -m news_fusion.prepare_news --help
python -m news_fusion.encode_news --help
python -m news_fusion.run_forecast --help
```

With the **already-prepared** source manifest, text features, and local price CSVs, the WTI/Henry Hub evaluation is run as:

```bash
python -m news_fusion.run_forecast \
  --manifest results/news_fusion/fnspid/news_manifest.csv \
  --features-dir results/news_fusion/fnspid \
  --prices-dir data \
  --assets wti,henry_hub \
  --output-dir results/news_fusion/fnspid/evaluation
```

To confirm common test outcomes across the two completed source evaluations:

```bash
python -m news_fusion.compare_sources \
  --abc-dir results/news_fusion/abc/evaluation \
  --fnspid-dir results/news_fusion/fnspid/evaluation \
  --output results/news_fusion/source_comparison.csv
```

Test the two test suites separately, because the forecasting utilities and benchmark utilities have different import roots:

```bash
PYTHONPATH=. python -m pytest -q --noconftest \
  test/test_clip_features.py \
  test/test_forecast_utils.py \
  test/test_news_fusion.py

PYTHONPATH=price_only_benchmarks:. python -m pytest -q --noconftest \
  price_only_benchmarks/test/test_benchmark_models.py \
  price_only_benchmarks/test/test_patchtst.py
```

**Reproducibility limitations:** this is not yet a one-command independent rerun. The source downloads may require accounts, network access, or provider permissions; the local experiment outputs are intentionally excluded from Git; and the course Docker dependency configuration needs a fresh-build check to ensure all notebook dependencies are available. No live-crawler system has been deployed.

---

## 8. Next steps: toward a live multimodal system

The next stage is to stop treating the news stream as a fixed historical CSV. A practical system would monitor **permitted feeds/APIs or licensed access** every day, retain precise timestamps and source provenance, and extract text alongside **image-based information** such as commodity charts and infographics. This is where CLIP's **joint image–text representation** could be used more fully. The crawler/scheduler collects content; **CLIP embeds it**; a separately trained forecasting model predicts returns.

### 8.0 Dataset for Energy headlines Extending energy-news coverage with GDELT and domain-specific sources:

Future experiments will incorporate GDELT to recover geopolitical and energy-related events that may be underrepresented in ABC News and FNSPID, including armed conflicts, sanctions, OPEC production decisions, supply-chain disruptions, refinery outages, and natural gas shortages. This will be complemented by energy-focused reporting from OilPrice.com and other authorized news sources. Unlike keyword-only filtering, the proposed pipeline will extract event entities, affected commodities, geographic context, temporal relationships, and asset-specific bullish, bearish, or neutral impact signals. These structured representations will be fused with frozen CLIP text and visual embeddings and historical price features, enabling future evaluation of event-aware forecasting across different market regimes and prediction horizons. All news will be aligned using verified publication timestamps to prevent look-ahead bias.

### 8.1 Candidate news sources

These are **planned sources, not currently ingested feeds**. Access, copyright, robots policies, paywalls, licensing, and rate limits must be checked for each provider; a public webpage is not permission to scrape and republish its contents.

| Focus | Prospective sources | Contribution |
|:--|:--|:--|
| Global crude and macro energy | [Reuters Energy](https://www.reuters.com/business/energy/), [OilPrice.com](https://oilprice.com/), [Bloomberg Energy](https://www.bloomberg.com/industries/energy), [Financial Times Energy](https://www.ft.com/energy) | OPEC policy, sanctions, demand, production, inventories, market-moving developments |
| Oil/gas industry and US gas | [Rigzone](https://www.rigzone.com/news/), [Natural Gas Intelligence](https://www.naturalgasintel.com/) | Field operations, pipelines, LNG, weather and storage-sensitive Henry Hub coverage |
| Gulf and geopolitics | [The National — Energy](https://www.thenationalnews.com/business/energy/), [Arab News — Energy](https://www.arabnews.com/energy), [Al Jazeera — Oil & Gas](https://www.aljazeera.com/tag/oil-and-gas/) | Regional supply, transport disruptions, policy and conflict context |
| Asian demand and trade | [ET EnergyWorld](https://energy.economictimes.indiatimes.com/news/oil-and-gas/), [SCMP Energy](https://www.scmp.com/topics/energy) | Indian/Chinese demand, imports, refining, LNG and trade policy |

For price and contract-market reference data, the prospective sources are [CME WTI futures](https://www.cmegroup.com/markets/energy/crude-oil/light-sweet-crude.quotes.html), [CME Henry Hub natural-gas futures](https://www.cmegroup.com/markets/energy/natural-gas/natural-gas.quotes.html), and [ICE Brent futures](https://www.ice.com/products/219/Brent-Crude-Futures/data). **Spot-price targets and futures-price targets must be modeled separately**; their contract conventions and delivery horizons are not interchangeable.

### 8.2 From keyword matches to asset-conditioned news events

A headline is not inherently bullish or bearish. Its **market impact is conditional on the commodity, the affected entity, the event mechanism, and the information already priced in**. For instance, an unexpected OPEC supply cut may support crude prices, whereas an unexpected increase in US natural-gas inventories may put downward pressure on Henry Hub. Generic positive/negative sentiment labels are therefore insufficient: a headline can describe a negative real-world event while implying a *positive* price response through reduced supply.

I plan to replace the current keyword-only representation with an **entity–event–impact layer**. Each incoming article will be deduplicated and linked to (i) named entities and geographic exposure (OPEC, producing countries, LNG terminals, storage facilities, pipelines), (ii) event type (supply disruption, demand shock, inventory surprise, sanctions, outages, weather), (iii) affected asset (WTI, Brent, Henry Hub), and (iv) an **asset-specific directional hypothesis**: bullish (+), bearish (−), or ambiguous/neutral (0). These tags would serve as *structured model inputs*, not assumed ground-truth price movements.

| Illustrative news event | Relevant entity or mechanism | WTI hypothesis | Henry Hub hypothesis |
|:--|:--|:--:|:--:|
| Unexpected OPEC crude output cut | OPEC; crude supply contracts | **+** | **0 / context-dependent** |
| US gas storage substantially above expectations | EIA gas inventory; supply surplus | **0 / indirect** | **−** |
| Colder-than-forecast US winter temperatures | Heating demand; regional gas balances | **0 / indirect** | **+**, conditional on inventories and region |
| Weaker global manufacturing and oil-demand forecasts | Demand slowdown | **−** | **0 / context-dependent** |

*The signs above are economic hypotheses, not measured outcomes or implemented classification results. Event direction should be validated against subsequent returns on strictly held-out dates, controlling for market expectations and simultaneous news.*

Rather than averaging every headline equally, I would learn a **time- and asset-conditioned attention weight**. A report about a US storage surprise should matter more for Henry Hub than a broad article about a foreign oil producer; an unexpected supply disruption should matter differently from a routine earnings headline. The resulting event signal would retain the news timestamp, article provenance, CLIP representation, relevant entity, uncertainty, and potential market direction. This also permits an interpretable *which-news-moved-the-forecast* analysis, while avoiding claims that correlation proves causation.

### 8.3 A Time-VLM-inspired joint text–vision–time-series predictor

This extension is motivated by **Zhong et al., [*Time-VLM: Exploring Multimodal Vision-Language Models for Augmented Time Series Forecasting* (ICML 2025)](https://proceedings.mlr.press/v267/zhong25a.html)** ([code](https://github.com/CityMind-Lab/ICML25-TimeVLM)). Time-VLM combines three complementary branches: a **Retrieval-Augmented Learner (RAL)** for historical temporal patterns, a **Vision-Augmented Learner (VAL)** that turns time series into images, and a **Text-Augmented Learner (TAL)** for contextual descriptions. Their representations are fused for forecasting. I propose **adapting this principle to asynchronous, real-world energy news**, rather than claiming the Time-VLM model has already been reproduced here.

- **Historical retrieval (RAL-inspired).** Maintain a temporally valid memory of past price–event windows, retrieving analogues such as inventory surprises, sanctions, weather-related gas demand and OPEC announcements. Retrieval must only access material published before each forecast cutoff, with no test-period labels leaking into the memory.
- **Visual price representation (VAL-inspired).** Convert historical WTI/Henry Hub return windows into normalized chart images, and optionally incorporate licensed article charts or supply/inventory figures. Encode these with CLIP's **frozen vision tower** (or test a specialist time-series vision encoder). Chart generation must use only observations available at prediction time. Because CLIP was not pretrained specifically for financial charts, a measured gain cannot be assumed.
- **News and event representation (TAL-inspired).** Encode raw headlines with CLIP's **frozen text tower**, add source/timestamp information and structured entity–event–impact tags, and pool news with recency-sensitive, asset-specific attention. Unlike Time-VLM's generated time-series descriptions, this branch would ingest **real, timestamped external headlines**, possibly alongside descriptions generated strictly from historical data.
- **Joint prediction and attribution.** Fuse price, visual, retrieved-history, and text/event features through a small gated or cross-attention prediction head. Train a return-regression objective and, optionally, a separately evaluated directional head predicting whether the next observed return is positive, negative, or near zero. Report event-level attribution as model sensitivity rather than a causal explanation.

The proposed asset- and horizon-conditional system can be summarized as:

$$
\mathbf{u}_t^{(a)} =
\mathrm{Fuse}\left(
\mathbf{x}_t^{(a)},
\mathbf{z}_{t,\mathrm{text}}^{(a)},
\mathbf{z}_{t,\mathrm{vision}}^{(a)},
\mathbf{m}_t^{(a)}
\right)
$$

$$
\hat{r}_{t+h}^{(a)} =
g_{\mathrm{return},h}\left(\mathbf{u}_t^{(a)}\right),
\qquad
\hat{p}_{t+h}^{(a)} =
P\left(
r_{t+h}^{(a)} > 0
\mid
\mathbf{u}_t^{(a)}
\right)
$$

Here $a$ specifies the commodity (e.g., WTI or Henry Hub), $h$ the forecasting horizon, $\mathbf z_{t,\mathrm{text}}$ CLIP news features, $\mathbf z_{t,\mathrm{vision}}$ chart/image features, and $\mathbf m_t$ a strictly historical retrieved context. The *directional probability* is a proposed trained output, **not** a calibrated probability inferred directly from CLIP's cosine similarity. The sign model and return model would be evaluated independently.

### 8.4 Roadmap: from offline benchmark to monitored forecasts

1. **Acquire with provenance.** Integrate permitted/licensed feeds, retain publication time and timezone, article ID, URL, rights and revisions; fetch market prices from official or authorized market-data providers. A scheduler collects content; CLIP does not itself crawl websites.
2. **Filter and label candidate events.** Combine high-recall retrieval with named-entity linking, entity-specific event detection, source-quality controls, deduplication and uncertain bullish/bearish/neutral hypotheses for each target commodity. Avoid labeling news by looking at future price movements at inference time.
3. **Test the multimodal contribution.** Compare price-only, price + raw news, price + entity/event tags, price + charts, and full fusion under identical chronological test dates. Include shuffled-headline, shuffled-image, removed-event and historical-retrieval ablations. Use RMSE, OOS $R^2$, directional accuracy, properly calibrated uncertainty and bootstrap confidence intervals.
4. **Model each horizon separately.** Begin with next-observation forecasts, then test weekly and monthly horizons using their own non-overlapping label definitions and validation splits. Year-ahead scenarios require macroeconomic drivers and uncertainty models; repeatedly applying a daily return model is not a reliable annual forecast.
5. **Deploy only after prospective evaluation.** Run a scheduled inference service with fixed forecast cutoffs, input snapshots, data/version tracking, drift monitoring and prospective paper-trading-style evaluation *without* claiming trading profitability. Retain zero-return and price-only Ridge as hard reference baselines.

**Research question.** Can a model learn *which headline matters to which commodity, in which direction, and under which temporal context*—and does that structured, cross-modal information consistently improve out-of-sample forecasts beyond historical prices? The current FNSPID/CLIP experiment motivates this question but does not yet answer it.

---

## 9. References and limitations

1. **Radford et al.** [*Learning Transferable Visual Models From Natural Language Supervision*](https://arxiv.org/abs/2103.00020), ICML 2021 — CLIP architecture and contrastive image–text training.
2. **Nie et al.** [*A Time Series Is Worth 64 Words: Long-term Forecasting with Transformers*](https://arxiv.org/abs/2211.14730), ICLR 2023 — PatchTST inspiration, not claimed as a fully reproduced model.
3. **Zhong, Ruan, Jin, Li, Wen, and Liang.** [*Time-VLM: Exploring Multimodal Vision-Language Models for Augmented Time Series Forecasting*](https://proceedings.mlr.press/v267/zhong25a.html), ICML 2025 — retrieval, visual time-series and text branches with frozen vision-language models; [official code](https://github.com/CityMind-Lab/ICML25-TimeVLM).
4. **Aldabagh, Zheng, Najand, Kalari, and Mukkamala.** [*An Agentic LLM-Based System for Short-Term Crude Oil Price Forecasting*](https://www.scitepress.org/PublishedPapers/2026/148164/), ICEIS 2026 — related news-driven forecasting direction.
5. **Data:** [FRED WTI](https://fred.stlouisfed.org/series/DCOILWTICO), [FRED Brent](https://fred.stlouisfed.org/series/DCOILBRENTEU), [FRED Henry Hub](https://fred.stlouisfed.org/series/DHHNGSP), [ABC Million News Headlines](https://www.kaggle.com/datasets/therohk/million-headlines), and the audited [FNSPID hosted dataset](https://huggingface.co/datasets/beachside1234/FNSPID).

**Interpretation boundary.** All reported RMSEs describe retrospective prediction of log returns, not a live trading result or proof of causation. The best observed improvements are small, their 95% intervals include zero, publication timestamps remain a limitation, and the current feature search has not been validated against a new untouched holdout. This repository is a working CLIP learning-and-implementation project with an exploratory applied benchmark.
### Additional References — Event-Driven and Multimodal Forecasting

1. **Cen, C., Liu, H., Hu, B., et al. (2026).** *Hierarchical Semantic Extraction and Heterogeneous Graph Neural Networks for Event-Driven Time Series Forecasting.* Scientific Reports, 16, 26302.  
   [Paper](https://doi.org/10.1038/s41598-026-56483-9)  
   **Relevance:** Hierarchical extraction of events, entity-level sentiment, causal relationships, and heterogeneous graph modeling for crude oil forecasting.

2. **Zhang, Y., Yang, W., Wang, J., Ma, Q., & Xiong, J. (2025).** *CAMEF: Causal-Augmented Multi-Modality Event-Driven Financial Forecasting by Integrating Time Series Patterns and Salient Macroeconomic Announcements.* ACM SIGKDD (KDD 2025).  
   [Paper](https://doi.org/10.1145/3711896.3736872) | [arXiv](https://arxiv.org/abs/2502.04592) | [Code](https://github.com/lakebodhi/CAMEF)  
   **Relevance:** Causal modeling, multimodal time-series fusion, macroeconomic announcements, and counterfactual event augmentation.

3. **Wang, M., Cohen, S. B., & Ma, T. (2024).** *Modeling News Interactions and Influence for Financial Market Prediction.* Findings of EMNLP 2024, pp. 3302–3314.  
   [Paper](https://aclanthology.org/2024.findings-emnlp.189/) | [PDF](https://aclanthology.org/2024.findings-emnlp.189.pdf) | [arXiv](https://arxiv.org/abs/2410.10614)  
   **Relevance:** FININ models interactions between news events, their delayed market effects, and their relationships with historical financial observations.

4. **Foroutan, P., & Lahmiri, S. (2024).** *Deep Learning Systems for Forecasting the Prices of Crude Oil and Precious Metals.* Financial Innovation, 10, 111.  
   [Paper](https://doi.org/10.1186/s40854-024-00637-z)  
   **Relevance:** Deep learning benchmarks for commodity price forecasting, including temporal models and ensemble learning methods.
