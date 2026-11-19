# Train Financial Sentiment Analysis Using Price Response as Labels

Author: @ypeng12. MSML610 Fall2026, individual research project.
[Project issue #861](https://github.com/gpsaggese/gpsaggese.github.io/issues/861).
[Assigned specification](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/research/ideas/draft.Train_Financial_Sentinment_Analysis_Using_Prices.md).

This is a working **checkpoint**, not a claim of complete research or profitable
trading. Price-supervised text learning, chronological evaluation and explicit
negative results are the focus. The portable example actually refits four models
on observed numerical features and independently checks every test prediction.
It also audits published BERT results and strategy curves. These are three
different experiments; their dates, sample counts and purposes are not combined.

## Run locally

```bash
./docker_build.sh
./docker_bash.sh -lc 'python financial_sentiment_utils.py'
./docker_bash.sh -lc 'jupyter nbconvert --to notebook --execute financial_sentiment.API.ipynb --output-dir executed'
./docker_bash.sh -lc 'jupyter nbconvert --to notebook --execute financial_sentiment.example.ipynb --output-dir executed'
./docker_jupyter.sh
```

Open the token-bearing Jupyter URL printed by the last command, replacing port
8888 with 8890. The host port binds to localhost. No broker credentials are needed
for these notebooks, and they make no network calls. The Dockerfile and base
requirements are unchanged from the official course template, retrieved at
upstream commit `eb4b86393557cc6477cb7266b50e1a73f82e0bc8` on November 4, 2026.
Shell wrappers are adapted to run from this self-contained directory inside
either the course fork or Quant. Docker/Colima is used explicitly on this Mac.

## Files and reproducibility boundary

| Files | Purpose |
| --- | --- |
| `financial_sentiment.API.ipynb` / `.py` | Input schema, provenance, units and timing contracts |
| `financial_sentiment.example.ipynb` / `.py` | Actual four-model refit, nine-model comparison and strategy audit |
| `financial_sentiment_utils.py` | Independent NumPy ridge fit, aggregate checks and label arithmetic |
| `data/historical_features.json` | Observed, train-only transformed numerical features; no article text or vocabulary |
| `public_results/` | Unmodified public BERT and strategy receipts with SHA256 manifest |
| `research_sources/` | Original collection, labels, model, fine-tuning and attribution implementations/tests |
| `research_reports/` | Historical research reports; dates distinguish their scope from this checkpoint |
| `source_manifest.json` | Quant source commit and exact exported file hashes |
| `PRESENTATION.md` | Student walkthrough and remaining submission steps |

The numerical packet is derived from 356 FNSPID headlines and 1,144 daily opening
prices each for Agilent and SPY. It yields 91 training and 87 later test rows after
common eligibility exclusions. The cutoff is January 1, 2019, with training
outcomes strictly before it. TF-IDF and scaling are fitted only on training.
The notebook independently refits models on those features; it does **not**
recreate the original article-to-feature transformation. No credentials, account
records, raw news corpus or private model weights are published.

Source dataset: [FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID), immutable
revision `bf9189c41527198897d1af3e17b1a0095279fc45`; original source links remain
in the retained private archive. Prices were collected using authorized Alpaca
market-data access. Numerical features and relative returns are derived research
data, not a redistribution of raw articles or account activity.

To reproduce preprocessing, use `research_sources/historical_backfill.py` and
`run_historical.py` with the original protocol in the Quant source repository and
your own authorized market-data access. A fresh capture must remain a new archive;
never backdate its receipt or overwrite previous evidence. The complete 2023
raw archive/checkpoints are not bundled and are currently incomplete locally.

## Research findings and limits

- The 2023 study compares nine models on the same 154 test rows. BERT improves
  AMZN MAE against the training mean, but does not consistently improve direction
  or all companies. Overlapping outcomes and inspected development samples
  preclude an independent-alpha claim.
- The separate 2026 B/C test has negative text increment: approximately -$84.07
  at the frozen 2 bps per-side cost scenario. These are simulations, not actual
  fees or account PnL. Daily-close and intraday drawdowns are distinct.
- Raw/excess 5/10/20-session research and FinBERT/LM reports are included as dated
  evidence. Default notebook execution does not retrain those models.
- Move detection and time-bounded retrieval are implemented. Independent human
  event labels, false-attribution rates and real unexplained rates remain pending.
  Zero qualified news does not mean all market moves have no explanation.

Quant remains the single development source. `scripts/export_course_project.py`
creates this course snapshot using an allowlist. Changes belong in Desktop/Quant,
then a new export is committed to the course branch; do not maintain two sources.
