# FNSPID/GDELT pilot recovery after HTTP 502/503/429

Original pilot: 5 successful / 40 attempted FNSPID Dataset Viewer pages (500 rows of 4,000 requested); two GDELT DOC API calls failed with HTTP 429. These are retrieval failures, not zero coverage.

## GDELT archival pilot (recommended now)

The new `gdelt_archive_audit.py` fetches raw public GDELT GKG 2.0 *15-minute ZIP archives* from `data.gdeltproject.org`, so it does not call the DOC API. It parses `<PAGE_TITLE>` from the final GKG column (field 27), distinguishes GDELT ingestion timestamp from extracted article publication timestamp, and applies the **original exact keyword regex** by importing `news_flags` from the existing `audit_fnspid_gdelt.py`.

```bash
python msml610_source_audit/gdelt_archive_audit.py \
  --include-latest \
  --output-dir results/source_audit/gdelt_archives \
  2>&1 | tee results/source_audit/gdelt_archive_run.log
```

The default historical snapshots are single 15-minute slices at midday UTC on 2020-01-15, 2021-01-15, 2023-01-15, and 2025-01-15, plus the latest slice. Some timestamps may be missing from the archive; those failures are logged. One 15-minute slice is **not a day** and cannot be used to estimate historical daily coverage.

Saved files include `gdelt_gkg_summary.json`, `gdelt_gkg_energy_candidates.csv`, `gdelt_gkg_manual_review.csv`, and `gdelt_gkg_all_sampled_titles.csv`.

GDELT references:
- https://data.gdeltproject.org/gdeltv2/lastupdate.txt
- https://blog.gdeltproject.org/gkg-2-0-now-includes-page-titles/
- https://blog.gdeltproject.org/new-gkg-2-0-article-metadata-fields/

## FNSPID

Do not re-run 40 pages against the failing `datasets-server.huggingface.co/rows` server. The current pilot has 500 rows, enough for schema validation **only**.

Official authors: https://huggingface.co/datasets/Zihan1004/FNSPID

An unaffiliated community converted version provides `Stock_news/All_external.parquet` (1.7 GB) and `Stock_news/nasdaq_exteral_data.parquet` (6.44 GB):
https://huggingface.co/datasets/beachside1234/FNSPID/tree/main/Stock_news

Test its Parquet listing without downloading a multi-GB file:

```bash
python - <<'PY'
import requests
url = 'https://huggingface.co/api/datasets/beachside1234/FNSPID/parquet'
r = requests.get(url, timeout=40)
print('HTTP', r.status_code)
print(r.text[:2200])
PY
```

The availability of Parquet shards and whether they can be sampled efficiently must be established before proceeding. Do not assume the converted repo is identical to the official release without hash/schema/coverage verification.

## Scientific caveats

- GDELT's `gkg_seen_utc` is GDELT's processing/observation timestamp, not guaranteed source publication time. `publication_time_reported_utc` is extracted from the publisher when available and may be absent or inaccurate.
- For a strict forecasting backtest, use only information known by the prediction cutoff. Do not treat late-discovered articles as earlier available information.
- Do not use a few 15-minute snapshots to estimate daily news coverage or relative superiority to ABC.
- News text is copyrighted by publishers; store source links and a research-only local extract, not article text in the GitHub PR.
