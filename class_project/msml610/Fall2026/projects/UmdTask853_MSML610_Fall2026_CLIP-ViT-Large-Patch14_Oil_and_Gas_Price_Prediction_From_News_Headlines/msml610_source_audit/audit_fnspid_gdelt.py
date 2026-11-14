#!/usr/bin/env python3
"""MSML610 pilot: sample FNSPID and collect recent GDELT headline metadata.

Does not download original 29 GB FNSPID CSVs. Does not scrape article pages.
Samples are NOT complete historical coverage or a training corpus.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import time
from pathlib import Path

import pandas as pd
import requests

HF_REPACK = "sabareesh88/FNSPID_external"
HF_BASE = "https://datasets-server.huggingface.co"
GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"
ENERGY_SYMBOLS = set("XOM CVX COP OXY EOG DVN FANG SLB HAL BKR APA MRO MPC VLO PSX KMI WMB LNG CTRA EQT AR RRC SWN CHK NFE BP SHEL SHEL.L ENB TRP OKE PXD".split())


def rx(pattern: str) -> re.Pattern:
    return re.compile(r"\b(?:" + pattern + r")\b", re.IGNORECASE)

# Exactly the user's preliminary candidate regex. Categories are NOT verified relevance labels.
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
AMBIGUOUS_GAS = rx(r"gas\s+(?:prices?|futures?|markets?|bills?|shortages?|crisis)")
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


def news_flags(headline: str) -> dict:
    text = re.sub(r"[\u2010-\u2015-]", " ", str(headline))
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


def json_get(session: requests.Session, url: str, params: dict, timeout=60) -> dict:
    resp = session.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    try:
        obj = resp.json()
    except ValueError as exc:
        raise RuntimeError(f"Expected JSON from {resp.url}; response begins {resp.text[:160]!r}") from exc
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected JSON object from {resp.url}")
    return obj


def collect_fnspid(session: requests.Session, batches: int, seed: int) -> tuple[pd.DataFrame, dict]:
    # Third-party repackaging of FNSPID external news, not the canonical upstream CSV.
    # Do not infer overall date coverage or total energy-news counts from this pilot.
    count = 13_057_514  # Repository metadata fallback; overridden by live dataset-viewer metadata.
    actual_size = False
    errors = []
    try:
        obj = json_get(session, f"{HF_BASE}/size", {"dataset": HF_REPACK})
        for item in obj.get("size", {}).get("splits", []):
            if item.get("split") == "train" and item.get("config") == "default":
                count = int(item.get("num_rows", item.get("num_examples", count)))
                actual_size = True
                break
    except (requests.RequestException, RuntimeError, TypeError, ValueError) as exc:
        errors.append(f"size endpoint: {exc}")
    rng = random.Random(seed)
    collected = []
    for i in range(batches):
        # Randomized systematic sample across ROW OFFSETS (not randomly selected dates).
        low = int((i / batches) * max(count - 100, 0))
        high = int(((i + 1) / batches) * max(count - 100, 0))
        offset = rng.randint(low, max(low, high))
        try:
            data = json_get(session, f"{HF_BASE}/rows", {
                "dataset": HF_REPACK, "config": "default", "split": "train",
                "offset": offset, "length": 100,
            })
            for obj in data.get("rows", []):
                r = obj.get("row", {})
                collected.append({
                    "source": "FNSPID_external_repack_sample",
                    "title": str(r.get("Article_title") or ""),
                    "date": str(r.get("Date") or "")[:10],
                    "timestamp_raw": str(r.get("Date") or ""),
                    "symbol": str(r.get("Stock_symbol") or ""),
                    "publisher": str(r.get("Publisher") or ""),
                    "url": str(r.get("Url") or ""),
                    "sample_offset": offset,
                })
            print(f"[FNSPID] page {i+1}/{batches}: {len(data.get('rows', []))} rows", flush=True)
        except (requests.RequestException, RuntimeError, TypeError, ValueError) as exc:
            errors.append(f"offset {offset}: {str(exc)[:250]}")
            print(f"[FNSPID] page {i+1}: failed ({str(exc)[:130]})", flush=True)
            if len(errors) >= 6 and not collected:
                break
        time.sleep(0.2)
    df = pd.DataFrame(collected)
    metadata = {"source": HF_REPACK, "size_metadata_rows": count,
                "size_from_live_endpoint": actual_size, "attempted_pages": batches,
                "sampled_rows": len(df), "errors": errors[:12],
                "sampling_note": "offset-sampled pilot; not complete FNSPID news archive"}
    return df, metadata


def collect_gdelt(session: requests.Session, days: int) -> tuple[pd.DataFrame, dict]:
    queries = {
        "oil": '("crude oil" OR "Brent crude" OR OPEC OR WTI)',
        "natural_gas": '("natural gas" OR LNG OR "Henry Hub")',
    }
    collected, errors = [], []
    query_counts = {}
    for name, query in queries.items():
        try:
            data = json_get(session, GDELT_API, {
                "query": query, "mode": "ArtList", "format": "json",
                "maxrecords": 250, "sort": "DateDesc", "timespan": f"{days}d",
            })
            articles = data.get("articles", [])
            query_counts[name] = len(articles)
            for a in articles:
                seen = str(a.get("seendate") or "")
                date = seen[:8]
                date = f"{date[:4]}-{date[4:6]}-{date[6:8]}" if len(date)==8 and date.isdigit() else ""
                collected.append({
                    "source": "GDELT_DOC_recent_article_sample",
                    "title": str(a.get("title") or ""),
                    "date": date,  # The date when GDELT SEEN the article, not guaranteed publication date.
                    "timestamp_raw": seen,
                    "symbol": "",
                    "publisher": str(a.get("domain") or ""),
                    "url": str(a.get("url") or ""),
                    "query_group": name,
                    "source_country": str(a.get("sourcecountry") or ""),
                })
            print(f"[GDELT] {name}: {len(articles)} returned (capped at 250)", flush=True)
        except (requests.RequestException, RuntimeError, TypeError, ValueError) as exc:
            errors.append(f"{name}: {str(exc)[:250]}")
            print(f"[GDELT] {name} query failed: {str(exc)[:150]}", flush=True)
        time.sleep(5)  # conservative rate limiting
    df = pd.DataFrame(collected)
    if not df.empty:
        df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    return df, {"source": "GDELT DOC 2.0", "timespan_days": days,
                "query_counts_before_dedupe": query_counts,
                "sampled_unique_articles": len(df), "errors": errors,
                "sampling_note": "recent result lists; 250 per query cap, not exhaustive historical archive",
                "timing_note": "GDELT seendate is when GDELT observed it, NOT guaranteed publisher time"}


def audit_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if df.empty:
        return df, {"sample_rows": 0, "matched_oil": 0, "matched_gas": 0,
                    "matched_any": 0, "unique_dates_in_sample": 0}
    df = df.copy()
    flags = df["title"].fillna("").map(news_flags).apply(pd.Series)
    df = pd.concat([df.reset_index(drop=True), flags.reset_index(drop=True)], axis=1)
    df["energy_ticker"] = df["symbol"].str.upper().isin(ENERGY_SYMBOLS)
    valid_dates = pd.to_datetime(df["date"], errors="coerce")
    return df, {
        "sample_rows": int(len(df)),
        "matched_oil": int(df.oil_candidate.sum()),
        "matched_gas": int(df.gas_candidate.sum()),
        "matched_direct": int((df.oil_candidate | df.gas_candidate).sum()),
        "matched_any": int(df.keep_candidate.sum()),
        "matched_energy_tickers": int(df.energy_ticker.sum()),
        "unique_dates_in_sample": int(valid_dates.nunique()),
        "date_min_sample": str(valid_dates.min().date()) if valid_dates.notna().any() else None,
        "date_max_sample": str(valid_dates.max().date()) if valid_dates.notna().any() else None,
        "duplicate_title_date_sample": int(df.duplicated(subset=["date", "title"]).sum()),
        "note": "All counts apply only to retrieved samples, NOT full source coverage",
    }


def write_outputs(output: Path, fn: pd.DataFrame, gd: pd.DataFrame, metadata: dict, seed: int):
    output.mkdir(parents=True, exist_ok=True)
    for tag, frame in (("fnspid", fn), ("gdelt", gd)):
        if not frame.empty:
            frame.to_csv(output / f"{tag}_pilot.csv", index=False)
            review = pd.concat([
                frame[frame.keep_candidate].sample(min(25, int(frame.keep_candidate.sum())), random_state=seed),
                frame[~frame.keep_candidate].sample(min(25, int((~frame.keep_candidate).sum())), random_state=seed),
            ], ignore_index=True)
            review["human_relevance_0_1_2"] = ""
            review["review_notes"] = ""
            review.to_csv(output / f"{tag}_manual_review.csv", index=False)
    (output / "source_audit_summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["both", "fnspid", "gdelt"], default="both")
    ap.add_argument("--fnspid-pages", type=int, default=40,
                    help="100 sampled rows per page; 40 -> up to 4000 rows")
    ap.add_argument("--gdelt-days", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=Path, default=Path("results/source_audit"))
    args = ap.parse_args()
    if args.fnspid_pages < 1 or not (1 <= args.gdelt_days <= 90):
        ap.error("Require --fnspid-pages >=1 and --gdelt-days between 1 and 90")
    session = requests.Session()
    session.headers.update({"User-Agent": "MSML610-educational-source-audit/0.1 (academic feasibility study)"})
    metadata = {"retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "sample_seed": args.seed, "original_abc_audit_reference":
                {"source_rows": 1244184, "explicit_oil_or_gas_candidates": 6305,
                 "note": "ABC full counts are not directly comparable to pilot samples"}}
    fdf, gdf = pd.DataFrame(), pd.DataFrame()
    if args.source in ("both", "fnspid"):
        fdf, fm = collect_fnspid(session, args.fnspid_pages, args.seed)
        fdf, s = audit_frame(fdf)
        metadata["fnspid"] = {**fm, "sample_audit": s}
    if args.source in ("both", "gdelt"):
        gdf, gm = collect_gdelt(session, args.gdelt_days)
        gdf, s = audit_frame(gdf)
        metadata["gdelt"] = {**gm, "sample_audit": s}
    write_outputs(args.output_dir, fdf, gdf, metadata, args.seed)
    print("\n=== SOURCE AUDIT PILOT SUMMARY ===")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print("\nSaved:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
