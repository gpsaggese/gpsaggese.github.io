#!/usr/bin/env python3
"""Audit ABC headlines with the project's fixed oil/gas candidate regex rules.

Usage: python audit_abc_energy.py --csv data/abcnews-date-text.csv
Outputs stay in results/abc_audit/ (not intended for Git).
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

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


def normalize(headline):
    text = re.sub(r"[\u2010-\u2015-]", " ", headline)
    return re.sub(r"\s+", " ", text).strip()


def news_flags(headline):
    text = normalize(str(headline))
    flags = {
        "oil_candidate": bool(OIL.search(text)),
        "gas_candidate": bool(GAS.search(text)),
        "ambiguous_gas": bool(AMBIGUOUS_GAS.search(text)),
        "indirect_candidate": bool(CONTEXT.search(text) and DRIVER.search(text)),
        "demand_candidate": bool(DEMAND.search(text)),
    }
    flags["keep_candidate"] = any(flags.values())
    return flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/abcnews-date-text.csv")
    parser.add_argument("--output-dir", default="results/abc_audit")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv, dtype={"publish_date": "string", "headline_text": "string"})
    expected = {"publish_date", "headline_text"}
    if not expected.issubset(df.columns):
        raise ValueError(f"Missing expected columns: {expected - set(df.columns)}")

    df["date"] = pd.to_datetime(df["publish_date"], format="%Y%m%d", errors="coerce")
    normalized = (df["headline_text"].fillna("")
                  .str.replace(r"[\u2010-\u2015-]", " ", regex=True)
                  .str.replace(r"\s+", " ", regex=True).str.strip())
    rules = {
        "oil_candidate": OIL,
        "gas_candidate": GAS,
        "ambiguous_gas": AMBIGUOUS_GAS,
        "indirect_candidate": None,
        "demand_candidate": DEMAND,
    }
    for key, regex in rules.items():
        if key != "indirect_candidate":
            df[key] = normalized.str.contains(regex, na=False)
    df["indirect_candidate"] = (
        normalized.str.contains(CONTEXT, na=False) &
        normalized.str.contains(DRIVER, na=False)
    )
    flag_cols = list(rules)
    df["keep_candidate"] = df[flag_cols].any(axis=1)
    df["explicit_oil_or_gas"] = df["oil_candidate"] | df["gas_candidate"]
    df["broad_only"] = df["keep_candidate"] & ~df["explicit_oil_or_gas"]
    df["duplicate_date_headline"] = df.duplicated(["publish_date", "headline_text"])

    # Count calendar-day coverage across the original source date range.
    valid_dates = df["date"].dropna()
    days = pd.date_range(valid_dates.min(), valid_dates.max(), freq="D")
    daily = (df.loc[df["date"].notna()]
             .groupby("date")[["keep_candidate", "explicit_oil_or_gas", "oil_candidate", "gas_candidate"]]
             .sum().reindex(days, fill_value=0))
    daily.index.name = "date"
    daily.to_csv(out / "daily_coverage.csv")

    yearly = (df.loc[df["date"].notna()].assign(year=lambda x: x.date.dt.year)
              .groupby("year")[["keep_candidate", "explicit_oil_or_gas", "oil_candidate", "gas_candidate", "broad_only"]]
              .sum())
    yearly["source_headlines"] = df.loc[df["date"].notna()].groupby(df["date"].dt.year).size()
    yearly.to_csv(out / "yearly_coverage.csv")

    # Overlapping flags are retained; no automatic assignment of broad candidates to gas.
    selection = df.loc[df["keep_candidate"]].copy()
    selection.to_csv(out / "candidate_headlines.csv", index=False)
    sample_masks = {
        "oil_only": df.oil_candidate & ~df.gas_candidate,
        "gas_only": df.gas_candidate & ~df.oil_candidate,
        "both": df.oil_candidate & df.gas_candidate,
        "ambiguous_without_explicit": df.ambiguous_gas & ~df.explicit_oil_or_gas,
        "indirect_only": df.indirect_candidate & ~df.explicit_oil_or_gas,
        "demand_only": df.demand_candidate & ~df.explicit_oil_or_gas & ~df.indirect_candidate,
    }
    samples = []
    for label, mask in sample_masks.items():
        group = df.loc[mask].sample(n=min(int(mask.sum()), 15), random_state=42).copy()
        group["review_group"] = label
        group["human_relevant"] = ""
        group["human_commodity"] = ""
        group["human_notes"] = ""
        samples.append(group)
    pd.concat(samples, ignore_index=True).to_csv(out / "manual_review.csv", index=False)

    def popular(regex, mask):
        counts = Counter()
        for text in normalized.loc[mask]:
            counts.update(m.lower() for m in regex.findall(text))
        return counts.most_common(20)

    summary = {
        "source_rows": int(len(df)),
        "date_from": str(valid_dates.min().date()),
        "date_to": str(valid_dates.max().date()),
        "missing_dates": int(df.date.isna().sum()),
        "missing_headlines": int(df.headline_text.isna().sum()),
        "duplicate_date_headline_pairs": int(df.duplicate_date_headline.sum()),
        "flag_counts_overlapping": {name: int(df[name].sum()) for name in flag_cols},
        "keep_candidate": int(df.keep_candidate.sum()),
        "explicit_oil_or_gas": int(df.explicit_oil_or_gas.sum()),
        "broad_only": int(df.broad_only.sum()),
        "calendar_days": int(len(days)),
        "no_candidate_news_days": int((daily.keep_candidate == 0).sum()),
        "no_explicit_news_days": int((daily.explicit_oil_or_gas == 0).sum()),
        "no_explicit_news_share": round(float((daily.explicit_oil_or_gas == 0).mean()), 4),
        "top_oil_matches": popular(OIL, df.oil_candidate),
        "top_gas_matches": popular(GAS, df.gas_candidate),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== ABC DATA QUALITY & ENERGY HEADLINE AUDIT ===")
    for name, value in summary.items():
        if not name.startswith("top_"):
            print(f"{name}: {value}")
    print("\nSaved:", out.resolve())
    print("Review manual_review.csv before approving broad candidates for FinBERT.")


if __name__ == "__main__":
    main()
