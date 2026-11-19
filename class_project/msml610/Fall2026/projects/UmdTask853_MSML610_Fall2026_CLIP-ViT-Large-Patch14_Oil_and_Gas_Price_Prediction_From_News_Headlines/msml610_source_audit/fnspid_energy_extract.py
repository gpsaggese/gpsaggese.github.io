#!/usr/bin/env python3
"""Reproducible FNSPID energy-news audit; uses the user's original regex verbatim.

Default mode reads only five headline/provenance columns for selected energy
company tickers from the published FNSPID community Parquet conversion.
No news/article text is downloaded. Requires pandas and duckdb.

Output headline dates are NOT verified intraday event timestamps.
Use headlines dated strictly BEFORE a commodity feature date in backtests.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


def rx(pattern):
    return re.compile(r"\b(?:" + pattern + r")\b", re.IGNORECASE)

OIL = rx(
    r"oil|crude|petroleum|opec|brent|wti|"
    r"refin(?:ery|eries|ing|ers?)|oil\s*(?:fields?|rigs?|sands?)|"
    r"gasoline|petrol|diesel|jet\s+fuel"
)

GAS = rx(
    r"natural\s+gas|natgas|lng|henry\s+hub|shale\s+gas|"
    r"gas\s+(?:storage|pipelines?|supply|production|demand|"
    r"exports?|imports?|inventor(?:y|ies)|fields?|wells?|fired)|"
    r"oil\s*(?:and|&|/)\s*gas"
)

AMBIGUOUS_GAS = rx(
    r"gas\s+(?:prices?|futures?|markets?|bills?|shortages?|crisis)"
)

CONTEXT = rx(
    r"energy|fuel|pipelines?|tankers?|terminals?|eia|"
    r"hormuz|red\s+sea|suez|bab\s+el\s+mandeb|black\s+sea|"
    r"middle\s+east|persian\s+gulf|"
    r"russia(?:n)?|ukrain(?:e|ian)|iran(?:ian)?|iraq(?:i)?|"
    r"saudi(?:\s+arabia)?|venezuela(?:n)?|liby(?:a|an)|"
    r"qatar|kuwait|uae|texas|louisiana|"
    r"gulf\s+(?:of\s+mexico|coast)|permian"
)

DRIVER = rx(
    r"wars?|conflicts?|attacks?|attacked|strikes?|struck|"
    r"bomb(?:s|ed|ing)?|explosions?|sanctions?|embargo(?:es)?|"
    r"blockades?|disrupt(?:ions?|ed|s)?|outages?|shutdowns?|"
    r"halts?|halted|shut|restarts?|ceasefire|peace\s+talks|"
    r"production|output|exports?|imports?|inventor(?:y|ies)|"
    r"stockpiles?|reserves?|discover(?:y|ies)|cuts?|"
    r"hurricanes?|heat\s*waves?|cold\s+snaps?|"
    r"freez(?:e|es|ing)|winter\s+storms?|polar\s+vortex"
)

DEMAND = rx(
    r"recession|economic\s+(?:slowdown|growth)|"
    r"manufacturing\s+(?:pmi|activity)|"
    r"electric\s+(?:vehicles?|cars?)|ev\s+(?:sales|adoption)|"
    r"solar\s+(?:power|plants?|farms?|capacity)|"
    r"wind\s+(?:power|farms?|capacity)"
)


def news_flags(headline):
    text = re.sub(r"[\u2010-\u2015-]", " ", headline)
    text = re.sub(r"\s+", " ", text).strip()
    flags = {
        "oil_candidate": bool(OIL.search(text)),
        "gas_candidate": bool(GAS.search(text)),
        "ambiguous_gas": bool(AMBIGUOUS_GAS.search(text)),
        "indirect_candidate": bool(CONTEXT.search(text) and DRIVER.search(text)),
        "demand_candidate": bool(DEMAND.search(text)),
    }
    flags["keep_candidate"] = any(flags.values())
    return flags


SOURCES = {
    "external": (
        "https://huggingface.co/datasets/beachside1234/FNSPID/"
        "resolve/main/Stock_news/All_external.parquet"
    ),
    "nasdaq": (
        "https://huggingface.co/datasets/beachside1234/FNSPID/"
        "resolve/main/Stock_news/nasdaq_exteral_data.parquet"
    ),
}
TICKERS = [
    "XOM", "CVX", "COP", "OXY", "EOG", "LNG", "EQT", "KMI", "CTRA",
    "SLB", "HAL", "BKR", "VLO", "MPC", "PSX",
]
FLAGS = ["oil_candidate", "gas_candidate", "ambiguous_gas", "indirect_candidate", "demand_candidate", "keep_candidate"]


def canonical_title(value: str) -> str:
    s = unicodedata.normalize("NFKC", str(value)).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def fetch_remote_sources(tickers: list[str], selection: list[str]) -> tuple[pd.DataFrame, dict]:
    import duckdb

    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    con.execute("SET threads=2")
    con.execute("SET memory_limit='2GB'")
    quoted = ",".join("'" + t.replace("'", "''") + "'" for t in tickers)
    query = f'''
        SELECT
          "Date" AS raw_timestamp,
          "Article_title" AS headline_text,
          UPPER(TRIM("Stock_symbol")) AS stock_symbol,
          "Url" AS article_url,
          "Publisher" AS publisher
        FROM read_parquet(?)
        WHERE UPPER(TRIM("Stock_symbol")) IN ({quoted})
    '''
    result, outcome = [], {}
    for name in selection:
        print(f"[FNSPID] Reading selected tickers from {name} ...", flush=True)
        try:
            frame = con.execute(query, [SOURCES[name]]).df()
            frame["source"] = name
            outcome[name] = {"status": "success", "selected_ticker_rows": len(frame)}
            result.append(frame)
            print(f"[FNSPID] {name}: {len(frame):,} records", flush=True)
        except Exception as exc:
            outcome[name] = {"status": "error", "error": str(exc)[:700]}
            print(f"[FNSPID] {name}: FAILED {str(exc)[:500]}", flush=True)
    if not result:
        raise RuntimeError(f"No FNSPID sources could be read: {outcome}")
    return pd.concat(result, ignore_index=True), outcome


def audit(frame: pd.DataFrame, output: Path, *, statuses=None, seed=42) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    columns = ["raw_timestamp", "headline_text", "stock_symbol", "article_url", "publisher", "source"]
    for col in columns:
        if col not in frame:
            frame[col] = ""
    frame = frame[columns].copy()
    frame["headline_text"] = frame["headline_text"].fillna("").astype(str).str.strip()
    frame["raw_timestamp"] = frame["raw_timestamp"].fillna("").astype(str).str.strip()
    frame["stock_symbol"] = frame["stock_symbol"].fillna("").astype(str).str.upper()
    frame["source"] = frame["source"].fillna("").astype(str)
    frame["article_url"] = frame["article_url"].fillna("").astype(str)
    frame["publisher"] = frame["publisher"].fillna("").astype(str)

    total = len(frame)
    raw_midnight = frame["raw_timestamp"].str.contains(r"\b00:00:00\s+UTC\b", regex=True)
    # Parse UTC as text, preserving timestamp provenance. Explicit clock times
    # are NOT assumed to be accurate; source authors had timezone issues.
    timestamp = pd.to_datetime(frame["raw_timestamp"], errors="coerce", utc=True, format="mixed")
    frame["publication_date"] = timestamp.dt.strftime("%Y-%m-%d")
    frame["timestamp_midnight_utc"] = raw_midnight
    frame["normalized_title"] = frame["headline_text"].map(canonical_title)
    valid = frame["normalized_title"].ne("") & frame["publication_date"].notna()
    invalid_count = int((~valid).sum())
    frame = frame.loc[valid].copy()
    if frame.empty:
        raise ValueError("No valid dated headlines found")

    flag_df = pd.DataFrame([news_flags(s) for s in frame["headline_text"]], index=frame.index)
    for c in FLAGS:
        frame[c] = flag_df[c]
    frame["direct_candidate"] = frame["oil_candidate"] | frame["gas_candidate"]
    frame["broad_only"] = frame["keep_candidate"] & ~frame["direct_candidate"]
    frame["eligible_feature_date_not_before"] = (
        pd.to_datetime(frame["publication_date"]) + pd.Timedelta(days=1)
    ).dt.strftime("%Y-%m-%d")
    # User-provided rules are applied before any removal. Stock association
    # is not treated as proof of energy relevance.
    matches = frame.loc[frame["keep_candidate"]].copy()
    matches.to_csv(output / "fnspid_candidate_mentions.csv", index=False)

    # The same syndicated article may appear under multiple ticker symbols
    # and sources. Preserve all ticker/source memberships in the output.
    matches = matches.sort_values(["publication_date", "normalized_title", "source", "stock_symbol"])
    key = ["publication_date", "normalized_title"]
    meta = (matches.groupby(key, sort=False, dropna=False)
            .agg(ticker_symbols=("stock_symbol", lambda x: ",".join(sorted(set(x)))),
                 sources=("source", lambda x: ",".join(sorted(set(x)))),
                 matching_mentions=("source", "size"),
                 distinct_urls=("article_url", lambda x: len(set(v for v in x if v))),
                 publishers=("publisher", lambda x: ",".join(sorted(set(v for v in x if v))[:6])))
            .reset_index())
    distinct = matches.drop_duplicates(key, keep="first").drop(columns=["stock_symbol", "source", "publisher"])
    distinct = distinct.merge(meta, on=key, how="left", validate="one_to_one")
    distinct = distinct.sort_values(["publication_date", "normalized_title"])
    distinct.to_csv(output / "fnspid_unique_candidates.csv", index=False)
    distinct.loc[distinct["direct_candidate"]].to_csv(output / "fnspid_unique_direct_candidates.csv", index=False)

    years = distinct.assign(year=distinct["publication_date"].str[:4])
    by_year = (years.groupby("year", as_index=False)
               .agg(unique_candidates=("keep_candidate", "size"),
                    direct=("direct_candidate", "sum"),
                    oil=("oil_candidate", "sum"),
                    gas=("gas_candidate", "sum"),
                    broad_only=("broad_only", "sum")))
    by_year.to_csv(output / "fnspid_energy_by_year.csv", index=False)

    by_source = (matches.groupby("source", as_index=False)
                 .agg(candidate_mentions=("keep_candidate", "size"),
                      direct_mentions=("direct_candidate", "sum"),
                      oil_mentions=("oil_candidate", "sum"),
                      gas_mentions=("gas_candidate", "sum")))
    by_source.to_csv(output / "fnspid_candidate_mentions_by_source.csv", index=False)

    # Manual review is explicitly a stratified convenience sample, not
    # a random population estimate of precision.
    sampled = []
    for name, subset in [
        ("oil_only", distinct.loc[distinct["oil_candidate"] & ~distinct["gas_candidate"]]),
        ("gas_only", distinct.loc[distinct["gas_candidate"] & ~distinct["oil_candidate"]]),
        ("oil_and_gas", distinct.loc[distinct["oil_candidate"] & distinct["gas_candidate"]]),
        ("broad_only", distinct.loc[distinct["broad_only"]]),
    ]:
        if not subset.empty:
            item = subset.sample(min(30, len(subset)), random_state=seed).copy()
            item["review_group"] = name
            sampled.append(item)
    if sampled:
        review = pd.concat(sampled, ignore_index=True)
        review["relevance_label"] = ""
        review["commodity"] = ""
        review["review_notes"] = ""
        review.to_csv(output / "fnspid_manual_review.csv", index=False)

    overview = {
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": statuses or {},
        "regex_policy": "user-provided OIL, GAS, AMBIGUOUS_GAS, CONTEXT+DRIVER, DEMAND unchanged",
        "selected_tickers": TICKERS,
        "selected_ticker_mentions": total,
        "invalid_date_or_empty_title": invalid_count,
        "midnight_utc_mentions": int(raw_midnight.sum()),
        "midnight_share": round(float(raw_midnight.mean()), 6),
        "raw_candidate_mentions": len(matches),
        "unique_candidate_articles_date_title": len(distinct),
        "unique_direct_articles": int(distinct["direct_candidate"].sum()),
        "unique_oil_articles": int(distinct["oil_candidate"].sum()),
        "unique_gas_articles": int(distinct["gas_candidate"].sum()),
        "unique_broad_only_articles": int(distinct["broad_only"].sum()),
        "duplicate_mentions_removed_by_title_date": len(matches) - len(distinct),
        "first_publication_date": distinct["publication_date"].min() if len(distinct) else None,
        "last_publication_date": distinct["publication_date"].max() if len(distinct) else None,
        "unique_direct_days": int(distinct.loc[distinct["direct_candidate"], "publication_date"].nunique()),
        "unique_oil_days": int(distinct.loc[distinct["oil_candidate"], "publication_date"].nunique()),
        "unique_gas_days": int(distinct.loc[distinct["gas_candidate"], "publication_date"].nunique()),
        "notes": [
            "Partial ticker-associated subset, not the full 28.6M source corpus.",
            "Deduplication based on normalized article title plus publication date; may merge distinct same-title reports.",
            "Do NOT use publication-day headlines for that day's price prediction. Treat dates as day-level until verified.",
            "Gaps, systematic relevance and source terms require further audits before model use.",
        ],
    }
    (output / "fnspid_energy_summary.json").write_text(json.dumps(overview, indent=2))
    print("\n=== FNSPID ENERGY EXTRACTION SUMMARY ===")
    for k, v in overview.items():
        if k not in ("source_files", "selected_tickers", "notes"):
            print(f"{k}: {v}")
    print("\nSaved:", output)
    return overview


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="results/source_audit/fnspid_extract")
    ap.add_argument("--source", choices=["external", "nasdaq", "both"], default="both")
    ap.add_argument("--offline-csv", help="local CSV with raw_timestamp, headline_text, stock_symbol, article_url, publisher, source")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.offline_csv:
        frame = pd.read_csv(args.offline_csv)
        statuses = {"synthetic_or_local_csv": args.offline_csv}
    else:
        selected = list(SOURCES) if args.source == "both" else [args.source]
        frame, statuses = fetch_remote_sources(TICKERS, selected)
    audit(frame, Path(args.output_dir), statuses=statuses, seed=args.seed)


if __name__ == "__main__":
    main()
