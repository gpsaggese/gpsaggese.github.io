# MSML610 FNSPID / GDELT source audit pilot

Run: `python -m pip install requests pandas`, then `python audit_fnspid_gdelt.py --source both --fnspid-pages 40 --gdelt-days 7 --output-dir results/source_audit`.

This script **samples** public FNSPID_external (third-party repackaging by sabareesh88) via Hugging Face Dataset Viewer `/rows` and recent GDELT DOC API results. It does not download the primary FNSPID 5.73 GB / 23.2 GB CSVs or scrape publisher sites.

The code applies the user's original five-category candidate regex. These flags are NOT verified relevance labels. Audit outputs report sample results only; samples should NOT be compared with ABC full-corpus counts as totals.

GDELT `seendate` is the GDELT observation timestamp, not guaranteed actual article publication time. The DOC API has a recent-history window and max 250 matches/query; the retrieved list is not exhaustive. For a serious multi-year backtest, design a historical GKG/BigQuery export separately.

FNSPID timestamps may be date-only or source-dependent; treat them conservatively and verify before any forecast alignment. This replica is a convenience for a first pilot, not a guaranteed duplicate of the canonical upstream corpus. The upstream authors' licensing statements should be checked before redistribution or commercial usage.

Keep raw output files and large samples outside GitHub; commit only the reusable audit script, README, concise summary and versioned criteria.
