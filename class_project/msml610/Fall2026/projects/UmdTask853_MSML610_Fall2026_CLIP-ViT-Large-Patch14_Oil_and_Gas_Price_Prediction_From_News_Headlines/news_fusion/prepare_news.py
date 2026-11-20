#!/usr/bin/env python3
"""Normalize already-audited ABC, FNSPID, and GDELT headline candidates.

Never changes the user's five-category regex rules. Input flags were produced
by the previously supplied audit scripts. 'strict' is an OPTIONAL deterministic
false-positive exclusion ablation; 'direct' is the primary preregistered rule.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

BAD_CONTEXT = re.compile(
    r"\b(?:olive|coconut|palm|vegetable|fish|essential)\s+oils?\b|"
    r"\bpetrol\s+sniff\w*\b|\b(?:nickel|nikel|aluminium|aluminum)\s+refiner(?:y|ies)\b|"
    r"\bbrent\s+harvey\b|\bpetrol\s+station\s+(?:robbery|robbed|attack|burglary)\b",
    re.IGNORECASE,
)


def bool_series(x: pd.Series) -> pd.Series:
    return x.astype("string").fillna("").str.lower().isin(["true", "1", "yes", "t"])


def normalized_title(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def date_series(values: pd.Series, *, compact: bool = False) -> pd.Series:
    if compact:
        parsed = pd.to_datetime(values.astype("string").str.replace(r"\.0$", "", regex=True),
                                format="%Y%m%d", errors="coerce", utc=True)
    else:
        parsed = pd.to_datetime(values, errors="coerce", utc=True, format="mixed")
    return parsed.dt.tz_convert(None).dt.normalize()


def load_one(source: str, path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if source == "abc":
        headline_col = "headline_text"
        if "date" in raw:
            dates = date_series(raw["date"])
        elif "publish_date" in raw:
            dates = date_series(raw["publish_date"], compact=True)
        else:
            raise ValueError("ABC candidate audit needs date or publish_date")
        source_field = "abc"
    elif source == "fnspid":
        headline_col = "headline_text"
        if "publication_date" not in raw:
            raise ValueError("FNSPID unique candidate file missing publication_date")
        dates = date_series(raw["publication_date"])
        source_field = "fnspid"
    elif source == "gdelt":
        headline_col = "title"
        if "gkg_seen_utc" not in raw:
            raise ValueError("GDELT source must have gkg_seen_utc")
        # GDELT first-observed timestamp is a conservative availability proxy.
        dates = date_series(raw["gkg_seen_utc"])
        source_field = "gdelt_pilot"
    else:
        raise ValueError(f"Unknown source: {source}")
    needed = [headline_col, "oil_candidate", "gas_candidate"]
    if any(x not in raw for x in needed):
        raise ValueError(f"Missing columns: {set(needed).difference(raw.columns)}")
    df = pd.DataFrame({"published_date": dates,
                       "headline_text": raw[headline_col].astype("string").fillna("").str.strip(),
                       "oil_candidate": bool_series(raw["oil_candidate"]),
                       "gas_candidate": bool_series(raw["gas_candidate"]),
                       "source": source_field})
    return df


def prepare(sources: dict[str, str], *, start: str, end: str, selection: str = "direct") -> tuple[pd.DataFrame, dict]:
    """Deduplicate same date/title across sources while OR-combining flags."""
    if selection not in ("direct", "strict"):
        raise ValueError("selection must be direct or strict")
    if "gdelt" in sources and len(sources) == 1:
        # May prepare a GDELT manifest, but prediction requires stronger archives.
        pass
    frames = [load_one(source, path) for source, path in sources.items()]
    frame = pd.concat(frames, ignore_index=True)
    n_raw = len(frame)
    frame = frame.loc[frame.published_date.notna() & frame.headline_text.ne("")].copy()
    frame = frame.loc[frame.oil_candidate | frame.gas_candidate].copy()
    before_strict = len(frame)
    if selection == "strict":
        frame = frame.loc[~frame.headline_text.str.contains(BAD_CONTEXT, na=False)].copy()
    frame = frame.loc[frame.published_date.between(pd.Timestamp(start), pd.Timestamp(end))].copy()
    frame["title_normalized"] = frame.headline_text.map(normalized_title)
    # Exact same date+title can be published under multiple stock symbols.
    group_cols = ["published_date", "title_normalized"]
    frame = frame.sort_values(["published_date", "title_normalized", "source"])
    distinct = (frame.groupby(group_cols, as_index=False, sort=True)
                .agg(headline_text=("headline_text", "first"),
                     oil_candidate=("oil_candidate", "max"),
                     gas_candidate=("gas_candidate", "max"),
                     source=("source", lambda v: ",".join(sorted(set(v))))))
    distinct["headline_id"] = distinct.apply(
        lambda row: hashlib.sha256((row.published_date.strftime("%Y-%m-%d") + "\x00" +
                                     row.title_normalized).encode("utf-8")).hexdigest()[:24], axis=1)
    distinct["published_date"] = distinct.published_date.dt.strftime("%Y-%m-%d")
    distinct = distinct[["headline_id", "published_date", "headline_text", "oil_candidate", "gas_candidate", "source"]]
    if distinct.empty:
        raise ValueError("No direct headlines in the selected time window")
    meta = {"sources": sources, "selection": selection, "start": start, "end": end,
            "raw_candidate_rows_loaded": n_raw, "direct_before_optional_exclusions": before_strict,
            "unique_direct_headlines": len(distinct),
            "unique_oil": int(distinct.oil_candidate.sum()),
            "unique_gas": int(distinct.gas_candidate.sum()),
            "unique_dates": int(distinct.published_date.nunique()),
            "note": "Source flag rules unchanged from original user regex; source-specific candidates are not manual ground truth."}
    return distinct, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["abc", "fnspid", "combined", "gdelt"], required=True)
    ap.add_argument("--abc-csv", default="results/abc_audit/candidate_headlines.csv")
    ap.add_argument("--fnspid-csv", default="results/source_audit/fnspid_extract/fnspid_unique_direct_candidates.csv")
    ap.add_argument("--gdelt-csv", default="results/source_audit/gdelt_archives/gdelt_gkg_energy_candidates.csv")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2021-12-31")
    ap.add_argument("--selection", choices=["direct", "strict"], default="direct")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    available = {"abc": args.abc_csv, "fnspid": args.fnspid_csv, "gdelt": args.gdelt_csv}
    sources = ({"abc": args.abc_csv, "fnspid": args.fnspid_csv}
               if args.source == "combined" else {args.source: available[args.source]})
    frame, metadata = prepare(sources, start=args.start, end=args.end, selection=args.selection)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "news_manifest.csv", index=False)
    (out / "manifest_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("=== HEADLINE MANIFEST ===")
    print(json.dumps(metadata, indent=2))
    print("Saved:", out / "news_manifest.csv")
    if args.source == "gdelt":
        print("[WARNING] GDELT input is a snapshot pilot, not comprehensive historical coverage.")


if __name__ == "__main__":
    main()
