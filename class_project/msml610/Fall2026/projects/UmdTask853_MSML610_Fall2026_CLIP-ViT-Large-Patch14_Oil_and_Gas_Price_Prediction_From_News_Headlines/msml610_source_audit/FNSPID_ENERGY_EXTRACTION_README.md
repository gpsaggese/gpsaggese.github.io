# MSML610: FNSPID headline-only extraction and relevance audit

This script performs an **audit**, not a forecasting experiment. It retrieves only
`Date`, `Article_title`, `Stock_symbol`, `Url`, and `Publisher` for fifteen
selected energy-related tickers from two remote FNSPID Parquet files.

## Setup

From the MSML610 project root in WSL with `.venv` active:

```bash
python -m pip install pandas duckdb
mkdir -p msml610_source_audit results/source_audit
```

Copy `fnspid_energy_extract.py` into `msml610_source_audit/` or unzip the supplied package.

## Run

```bash
set -o pipefail
python msml610_source_audit/fnspid_energy_extract.py \
  --source both \
  --output-dir results/source_audit/fnspid_extract \
  2>&1 | tee results/source_audit/fnspid_extract_run.log
```

The script queries ~88k ticker-associated records in the first audit; remote
transfer and runtime depend on Parquet row grouping and Hugging Face server
availability. If one source fails, the metadata records the failure and saves
results from any successful source. Do not interpret a partial extract as a
complete source comparison.

## Outputs

- `fnspid_energy_summary.json`: counts, source errors and timestamp risks.
- `fnspid_candidate_mentions.csv`: all raw ticker-associated regex matches.
- `fnspid_unique_candidates.csv`: deduplicated by normalized headline + date;
  includes matched tickers, sources, source-mention count, and original URL.
- `fnspid_unique_direct_candidates.csv`: unique records with OIL or GAS flags.
- `fnspid_energy_by_year.csv`: yearly coverage of the unique candidate pool.
- `fnspid_candidate_mentions_by_source.csv`: source-specific mention counts.
- `fnspid_manual_review.csv`: stratified unannotated random headline samples.

## Methodology

- The user's original regex definitions are **unchanged**.
- All five flag types are retained, while direct oil or gas headlines can be
  analyzed separately from broad-only candidates.
- Source-specific mentions may repeat the same article under different tickers.
  The dataset also contains syndicated stories. Cross-source deduplication is
  based on normalized headline and publication date. That strategy is simple
  and auditable but can over-merge same-title news on the same day.
- FNSPID timestamps are **not assumed to be trustworthy intraday timestamps**.
  In the first scan, 88,090 of 88,333 selected-ticker rows were midnight UTC.
  The exported `eligible_feature_date_not_before` is simply the calendar day
  after the quoted publication date; it is a **lower bound** for use in later
  alignment, not a verified intraday release timestamp. For a price origin on
  day `t`, require headline `publication_date < t`; do not use same-day news.
- No model has been fit, and no observed forecasting performance is claimed.
- Do not commit bulk source articles or locally extracted output files to GitHub.

## Checks after run

```bash
python - <<'PY'
from pathlib import Path
import json
import pandas as pd
base=Path('results/source_audit/fnspid_extract')
meta=json.loads((base/'fnspid_energy_summary.json').read_text())
for key in ['selected_ticker_mentions','raw_candidate_mentions','unique_candidate_articles_date_title','unique_direct_articles','unique_oil_articles','unique_gas_articles','unique_direct_days','midnight_share']:
    print(key, meta[key])
print('Source statuses:', meta['source_files'])
print(pd.read_csv(base/'fnspid_energy_by_year.csv').to_string(index=False))
print(pd.read_csv(base/'fnspid_manual_review.csv').head(15)[['publication_date','headline_text','review_group']].to_string(index=False))
PY
```
