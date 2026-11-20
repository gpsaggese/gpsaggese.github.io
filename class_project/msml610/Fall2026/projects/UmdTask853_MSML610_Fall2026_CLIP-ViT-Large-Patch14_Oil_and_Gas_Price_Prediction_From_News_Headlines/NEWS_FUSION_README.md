# MSML610 | Frozen CLIP vs FinBERT energy news forecasting (v1)

**Status**: runnable experimental implementation, *not* verified real-data results.

The exact user-provided keyword flags from the earlier ABC / FNSPID audit are read **unchanged**. This pipeline forecasts next observed commodity log return for WTI (`DCOILWTICO`), Brent (`DCOILBRENTEU`), and Henry Hub (`DHHNGSP`) using 30 price-return lags, 5-quote momentum, and 20-quote volatility plus news features. All prices are positive, and the April 2020 nonpositive WTI quote is excluded from returns touching it. It uses one 65/15/20 chronological split per asset with 5 excluded rows between segments, and all models share precisely the same price dates for a given source.

**Data sources**:
- ABC audited candidates: `results/abc_audit/candidate_headlines.csv`, 2003–2021
- FNSPID extracted unique direct candidates: `results/source_audit/fnspid_extract/fnspid_unique_direct_candidates.csv`, 2009–2023
- Combined ABC+FNSPID: exact date/title deduplication, source membership retained
- GDELT: 4 sampled 15-minute archives, **coverage demonstration only**, not an experimental baseline

**Fixed common evaluation dates: 2010-01-01 through 2021-12-31.** This is a common *target-period* window, not a verified first-publication-time alignment. For every headline with publication calendar date `d`, the earliest eligible target price date is the first positive observed FRED price date `t>d` (never same day). FNSPID midnight UTC timestamps are *not* interpreted as intraday availability; GDELT `gkg_seen_utc` is an observation time, not necessarily publication. No-news dates are kept with zero-vector features, count=0 and indicator=0. The target is next observed quote return ending on `t` relative to the prior observed quote; price predictors use data no later than the prior observed quote. FRED observed dates are not guaranteed release times, so this is a retrospective research benchmark, **not live-trading validation**.

## Models (all supervised heads are Ridge)

| ID | Features |
|---|---|
| `zero_return` | predict zero |
| `historical_mean` | constant train+validation mean |
| `price_ridge` | standardized price features |
| `price_news_count_ridge` | price + daily headline count and has-news indicator |
| `price_tfidf_ridge` | price + count + training-only TF-IDF unigrams/bigrams |
| `price_finbert_sentiment_ridge` | price + count + FinBERT mean positive/negative/neutral probabilities |
| `price_finbert_cls_pca_ridge` | price + count + frozen FinBERT CLS embeddings, PCA20 |
| `price_clip_pca_ridge` | price + count + frozen CLIP ViT-L/14 text projections, PCA20 |
| `price_clip_event_ridge` | price + count + CLIP cosine similarity to 6 fixed commodity-event prompts |
| `price_clip_full_ridge` | price + count + CLIP PCA20 + event-prompt similarities |
| `price_minilm_pca_ridge` | optional price + count + MiniLM sentence embeddings, PCA20 |
| `shuffled_clip_control` | CLIP vectors shuffled among news-present days within each split, counts unchanged |

CLIP: `openai/clip-vit-large-patch14` (77 token max). FinBERT: `ProsusAI/finbert` (128 token max); three probabilities are explicitly ordered `positive, negative, neutral`; 768-D CLS from last hidden layer. MiniLM (optional): `sentence-transformers/all-MiniLM-L6-v2` (256 tokens), masked mean pooling. Encoders are frozen; batch encodings are cached in hashed shards. Review weights' applicable licenses (FinBERT model card has noncommercial terms). Do not distribute weights in the course repository.

### Install (WSL `.venv`)

```bash
python -m pip install 'numpy>=1.26' 'pandas>=2.1' 'scikit-learn>=1.4' 'scipy>=1.12' 'pytest>=8' \
  'torch>=2.3' 'transformers==4.53.2' safetensors
```

For GPU, use a CUDA-enabled compatible PyTorch build in Colab / your environment. Confirm `python -c 'import torch; print(torch.cuda.is_available())'`. On CPU, frozen CLIP ViT-L/14 will be slower.

### Phase A: prepare manifests, choose direct or strict before modeling

Run from project root after unzipping this package (it creates `news_fusion/`).

```bash
for s in abc fnspid; do
  python -m news_fusion.prepare_news --source "$s" \
    --start 2010-01-01 --end 2021-12-31 \
    --selection direct --output-dir "results/news_fusion/$s"
done
# Optional, matches both data sources and deduplicates their overlap:
python -m news_fusion.prepare_news --source combined \
  --start 2010-01-01 --end 2021-12-31 \
  --selection direct --output-dir results/news_fusion/combined
```

### Phase B: actual frozen inference

Start with FinBERT on ABC, then CLIP on ABC. Each model reads *exactly the headline manifest*; do not change the manifest between extraction and forecast.

```bash
python -m news_fusion.encode_news \
  --manifest results/news_fusion/abc/news_manifest.csv \
  --encoder finbert --output results/news_fusion/abc/finbert.npz \
  --batch-size 32 --device auto

python -m news_fusion.encode_news \
  --manifest results/news_fusion/abc/news_manifest.csv \
  --encoder clip --output results/news_fusion/abc/clip.npz \
  --batch-size 8 --device auto
```

For FNSPID change `abc` to `fnspid` in both paths, and similarly for `combined`. Optional MiniLM: `--encoder minilm --output .../minilm.npz --batch-size 64`.

CLIP encoding also writes `clip_prompts.npz` with 12 **fixed, preregistered** positive/negative supply and demand event prompts (6 for oil, 6 for Henry Hub). This enables a compact event-score ablation without retraining CLIP. Brent uses oil prompts.

**Resumption**: `--cache-dir .cache/news_fusion` (default) keeps verified `.npz` shards. `--revision main` is the default; pin immutable commit SHAs before final publication. The metadata records the model ID/revision, torch/transformers versions and actual device. For a source-independent deduplicated cache across manifests, improvements to content-hash-per-headline would be needed; v1 caches by full manifest hash.

### Phase C: evaluations after encoding (same 2010–2021 window)

```bash
python -m news_fusion.run_forecast \
  --manifest results/news_fusion/abc/news_manifest.csv \
  --features-dir results/news_fusion/abc \
  --prices-dir data --start 2010-01-01 --end 2021-12-31 \
  --output-dir results/news_fusion/abc/evaluation

python -m news_fusion.run_forecast \
  --manifest results/news_fusion/fnspid/news_manifest.csv \
  --features-dir results/news_fusion/fnspid \
  --prices-dir data --start 2010-01-01 --end 2021-12-31 \
  --output-dir results/news_fusion/fnspid/evaluation
```

The evaluator autodetects which `.npz` exist. **Without embeddings**, it still runs valid price-only, count, and TF-IDF baselines, so you can validate the pipeline immediately. `--quick` is *diagnostics only*, not for your professor's RMSE table.

### Test / inspect

```bash
python -m pytest -q --noconftest test/test_news_fusion.py
cat results/news_fusion/abc/evaluation/test_leaderboard.csv
cat results/news_fusion/fnspid/evaluation/test_leaderboard.csv
```

Files: `test_leaderboard.csv`, `test_predictions.csv`, `paired_rmse_bootstrap.csv`, `model_configs.json`, `coverage_and_splits.json`, `protocol.json`. Bootstrap confidence intervals use 1,000 paired circular moving-block resamples of residual pairs (block length 5); they condition on fitted models, do not account for model selection or training variance, and should not be presented as causal evidence.

### Strong scientific caveats for PR / midterm review

- The already-examined 2019–2026 price-only RMSE leaderboard is **not** the appropriate comparator for these 2010–2021 news-source experiments; the pipeline retrains price-only Ridge on the **same** dates.
- No-news dates are never dropped to inflate apparent performance.
- We cannot claim FNSPID news publication timestamps are genuine midnight UTC. Date-only mapping is conservative but does not establish public release time of FRED spot quotes.
- WTI/Brent/Henry Hub news relevance needs manual checking and source-specific deduplication. FinBERT financial sentiment is **not** the sign of the commodity return.
- `source=combined` is an ablation, not necessarily a stronger corpus. Duplicates are removed on date and normalized title; syndicated stories with different titles may remain.
- GDELT snapshots lack sufficient temporal continuity to report a reliable forecast improvement. Include GDELT in **data-source feasibility**, not the main forecasting results.
- All test results after initial inspection are exploratory. Decide final comparison metrics and model-selection criteria using train/validation only before further test inspection.
- Do not commit `data/`, `results/`, `.cache/`, `.npz`, article content, model weights, or large news manifests to GitHub. Put figures in professor's shared Google Doc, code/short numerical results in PR and `progress.md`.

### Verify common target dates across ABC and FNSPID (after both evaluations)

```bash
python -m news_fusion.compare_sources \
  --abc-dir results/news_fusion/abc/evaluation \
  --fnspid-dir results/news_fusion/fnspid/evaluation \
  --output results/news_fusion/source_comparison.csv
```

The comparator **fails** if any evaluated asset's test dates or actual returns differ across the source experiments, or if the sets of evaluated assets differ. A separate table reports RMSE changes against the corresponding price-only Ridge baseline on the same dates. Do not select news source based on test performance after inspection.
