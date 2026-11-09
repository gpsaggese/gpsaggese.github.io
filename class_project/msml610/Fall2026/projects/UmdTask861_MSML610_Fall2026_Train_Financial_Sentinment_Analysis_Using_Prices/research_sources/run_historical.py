"""Offline retrospective headline/price development experiment, not live alpha.

The strict prospective total-return contract is unchanged. This separate track
uses observed split-adjusted opening-price returns (cash dividends excluded),
unknown historical text versions, delayed date-only headlines and source/event
proxy purging. Training transforms never use test values or outcome labels.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

from price_labels import _connected_groups

NY = ZoneInfo("America/New_York")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def canonical_url(value):
    p = urlsplit(value.strip())
    if p.scheme not in ("http", "https") or not p.netloc:
        return ""
    query = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
             if not k.lower().startswith("utm_") and k.lower() not in {"guccounter", "cmpid", "ref"}]
    return urlunsplit(("https", p.netloc.lower(), p.path.rstrip("/"), urlencode(sorted(query)), ""))


def group_news(news):
    """Conservative identity proxies; no claim of verified economic events."""
    items = []
    for index, n in enumerate(news):
        headline = " ".join(re.findall(r"[a-z0-9]+", n["headline"].lower()))
        uri = canonical_url(n.get("url", ""))
        source = digest((uri or n["source_date"] + ":" + headline).encode())
        items.append({**n, "sample_id": f"article-{index}", "source_ids": ["url-or-text:" + source],
                      "event_cluster_ids": [], "normalized_headline": headline})
    # Windowed near-duplicate links are only a conservative proxy. Do not merge
    # recurring generic titles across years. No price/outcome is inspected here.
    ordered = sorted(range(len(items)), key=lambda i: items[i]["source_date"])
    for offset, i in enumerate(ordered):
        left = items[i]
        day = datetime.fromisoformat(left["source_date"])
        a = set(left["normalized_headline"].split())
        for j in ordered[offset + 1:]:
            right = items[j]
            if (datetime.fromisoformat(right["source_date"]) - day).days > 7:
                break
            b = set(right["normalized_headline"].split())
            if a and b and len(a & b) >= 4 and len(a & b) / len(a | b) >= 0.85:
                token = f"proxy-near-headline:{i}:{j}"
                left["event_cluster_ids"].append(token)
                right["event_cluster_ids"].append(token)
    return items


def price_class(value, band=0.005):
    """Price-derived relative-return class, not positive/negative prose tone."""
    if not math.isfinite(value) or band < 0 or not math.isfinite(band):
        raise ValueError("Require finite return and nonnegative neutral band")
    return 1 if value > band else -1 if value < -band else 0


def build_rows(news, prices, calendar, protocol):
    """Aggregate one delayed company-day; features end strictly before entry."""
    symbol = protocol["symbol"]
    horizon, delay = protocol["horizon_sessions"], protocol["entry_delay_sessions"]
    if type(horizon) is not int or horizon not in (5, 10, 20) or type(delay) is not int or delay < 1:
        raise ValueError("Require horizon 5/10/20 and positive whole-session delay")
    sessions = calendar["sessions"]
    days = [s["date"] for s in sessions]
    if days != sorted(set(days)):
        raise ValueError("Calendar must have distinct ordered dates")
    grouped, outside = defaultdict(list), []
    for item in group_news(news):
        entry = bisect_right(days, item["source_date"]) + delay - 1
        if entry >= len(days):
            outside.append({"sample_id": item["sample_id"], "reasons": ["no_delayed_entry_session"]})
        else:
            grouped[entry].append(item)
    rows = []
    for index, articles in sorted(grouped.items()):
        row = {"sample_id": f"{symbol}:{days[index]}", "symbol": symbol,
               "entry_session_index": index, "entry_at": sessions[index]["open_at"],
               "source_ids": sorted({x for a in articles for x in a["source_ids"]}),
               "event_cluster_ids": sorted({x for a in articles for x in a["event_cluster_ids"]}),
               "source_dates": sorted({a["source_date"] for a in articles}),
               "text": "\n".join(dict.fromkeys(a["headline"] for a in articles)),
               "source_row_indices": [a["source_row_index"] for a in articles],
               "historical_availability_verified": False, "status": "Retrospective",
               "exclusion_reasons": [], "label_status": "Unavailable"}
        rows.append(row)
        exit_index = index + horizon
        if exit_index >= len(days):
            row["exclusion_reasons"].append("future_endpoint_outside_calendar")
            continue
        row.update(exit_at=sessions[exit_index]["open_at"], exit_session_index=exit_index)
        keys = [(s, days[j]) for s in (symbol, "SPY") for j in (index, exit_index)]
        if any(key not in prices for key in keys):
            row["exclusion_reasons"].append("missing_observed_price_endpoint")
            continue
        stock = prices[(symbol, days[exit_index])] / prices[(symbol, days[index])] - 1
        benchmark = prices[("SPY", days[exit_index])] / prices[("SPY", days[index])] - 1
        row.update(stock_return=stock, spy_return=benchmark, excess_return=stock - benchmark,
                   price_class=price_class(stock - benchmark, protocol["class_neutral_band"]),
                   label_status="RetrospectiveAvailable")
        features = []
        for period in (5, 20):
            # Prior session's open is known before today's prediction; entry
            # open, daily close and all future prices are excluded from inputs.
            before, old = index - 1, index - 1 - period
            if old < 0 or any((symbol, days[k]) not in prices for k in (before, old)):
                row["exclusion_reasons"].append(f"missing_prior_momentum_{period}")
            else:
                features.append(prices[(symbol, days[before])] / prices[(symbol, days[old])] - 1)
        if len(features) == 2:
            row["price_features"] = features
    return rows, outside


def retrospective_split(rows, test_start):
    """Mature chronology and transitive purging, without certifying eligibility."""
    cutoff = test_start + "T00:00:00Z"
    groups = _connected_groups(rows)  # reuse full source graph, including excluded bridges
    train, test, excluded = [], [], []
    for row in rows:
        reasons = list(row["exclusion_reasons"])
        if not reasons:
            if row["entry_at"][:10] < test_start:
                if row["exit_at"] >= cutoff:
                    reasons.append("training_outcome_crosses_time_boundary")
                else:
                    train.append(row)
            else:
                test.append(row)
        if reasons:
            excluded.append({"sample_id": row["sample_id"], "reasons": reasons})
    test_groups = {groups[r["sample_id"]] for r in test}
    retained = []
    for row in train:
        if groups[row["sample_id"]] in test_groups:
            excluded.append({"sample_id": row["sample_id"], "reasons": ["source_or_proxy_event_linked_to_test"]})
        else:
            retained.append(row)
    return {"train": retained, "test": test, "excluded": excluded,
            "test_start": test_start, "availability_track": "retrospective_unknown_versions",
            "event_identity": "URL and windowed headline proxies; not independent event verification"}


def evaluate_models(split, protocol):
    import numpy as np
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, balanced_accuracy_score, f1_score

    train, test = split["train"], split["test"]
    if len(train) < 10 or len(test) < 5:
        raise ValueError("Insufficient paired train/test rows for this bounded development experiment")
    configuration = dict(protocol["tfidf"])
    configuration["ngram_range"] = tuple(configuration["ngram_range"])
    vectorizer = TfidfVectorizer(**configuration)
    text_train = vectorizer.fit_transform([r["text"] for r in train])
    text_test = vectorizer.transform([r["text"] for r in test])
    scaler = StandardScaler().fit([r["price_features"] for r in train])
    price_train = csr_matrix(scaler.transform([r["price_features"] for r in train]))
    price_test = csr_matrix(scaler.transform([r["price_features"] for r in test]))
    y = np.asarray([r["excess_return"] for r in train])
    target = np.asarray([r["excess_return"] for r in test])
    designs = {"price_ridge": (price_train, price_test), "text_tfidf_ridge": (text_train, text_test),
               "price_text_ridge": (hstack([price_train, text_train]).tocsr(), hstack([price_test, text_test]).tocsr())}
    predictions = {"train_mean": np.full(len(test), y.mean())}
    fitted = {"train_mean": {"intercept": float(y.mean()), "coefficients": []}}
    for name, (x, x_test) in designs.items():
        model = Ridge(alpha=protocol["ridge_alpha"], solver="lsqr", tol=1e-8).fit(x, y)
        predictions[name] = model.predict(x_test)
        fitted[name] = {"intercept": float(model.intercept_), "coefficients": model.coef_.tolist()}
    metrics, rows = {}, []
    band = protocol["class_neutral_band"]
    classes = [price_class(float(v), band) for v in target]
    for name, values in predictions.items():
        if not np.isfinite(values).all():
            raise ValueError("Non-finite model prediction")
        predicted_classes = [price_class(float(v), band) for v in values]
        metrics[name] = {"samples": len(test), "mae": float(mean_absolute_error(target, values)),
                         "rmse": float(math.sqrt(mean_squared_error(target, values))),
                         "relative_direction_accuracy": float(np.mean(np.sign(values) == np.sign(target))),
                         "price_class_balanced_accuracy": float(balanced_accuracy_score(classes, predicted_classes)),
                         "price_class_macro_f1": float(f1_score(classes, predicted_classes, labels=[-1, 0, 1], average="macro", zero_division=0))}
    for index, source in enumerate(test):
        row = {k: source[k] for k in ["sample_id", "entry_at", "exit_at", "entry_session_index", "exit_session_index",
                                       "stock_return", "spy_return", "excess_return", "price_class", "source_ids"]}
        row["predictions"] = {name: float(value[index]) for name, value in predictions.items()}
        rows.append(row)
    artifacts = {"vocabulary": vectorizer.vocabulary_, "idf": vectorizer.idf_.tolist(),
                 "price_means": scaler.mean_.tolist(), "price_scales": scaler.scale_.tolist(),
                 "models": fitted, "training_sample_ids": [r["sample_id"] for r in train],
                 "test_sample_ids": [r["sample_id"] for r in test]}
    # numpy integer vocabulary positions must have portable JSON values.
    artifacts["vocabulary"] = {k: int(v) for k, v in artifacts["vocabulary"].items()}
    return metrics, rows, artifacts


def cost_evaluation(predictions, cost_bps):
    """Non-overlapping long/cash intervals; modeled costs, not broker charges."""
    selected, next_entry = [], -1
    for row in sorted(predictions, key=lambda r: (r["entry_session_index"], r["sample_id"])):
        if row["entry_session_index"] >= next_entry:
            selected.append(row)
            next_entry = row["exit_session_index"]
    names = list(predictions[0]["predictions"]) if predictions else []
    output = {}
    for name in [*names, "always_company", "always_spy", "cash"]:
        scenarios = []
        for bps in cost_bps:
            if not 0 <= bps < 20000 or not math.isfinite(bps):
                raise ValueError("Assumed cost must be finite and nonnegative")
            capital, trades = 1.0, 0
            for row in selected:
                active = name in ("always_company", "always_spy") or (name != "cash" and row["predictions"][name] > 0)
                if active:
                    ret = row["spy_return"] if name == "always_spy" else row["stock_return"]
                    # Round-trip assumption split equally across both sides;
                    # capital is fully closed between intervals, conservatively.
                    factor = (1 - bps / 20000) ** 2 * (1 + ret)
                    capital *= factor
                    trades += 1
            scenarios.append({"assumed_roundtrip_cost_bps": bps, "cumulative_simulated_price_return": capital - 1,
                              "roundtrip_trades": trades})
        output[name] = scenarios
    return {"mode": "retrospective_non_overlapping_long_cash_price_return",
            "eligible_intervals": len(selected), "selected_sample_ids": [r["sample_id"] for r in selected],
            "overlapping_test_rows_not_compounded": len(predictions) - len(selected),
            "actual_broker_charges": None, "cash_dividends_included": False,
            "annualized": False, "is_executable_strategy_evidence": False, "models": output,
            "limitations": ["Fixed illustrative execution-cost scenarios, not actual Alpaca fees",
                            "No interest on cash; long/cash only, no borrow or short positions",
                            "News-selected sparse intervals, closing/reopening each interval; not a daily portfolio track record",
                            "Prices are split-adjusted, exclude cash dividends and do not guarantee fills"]}


def run(input_dir, output_dir):
    requested_output = Path(output_dir)
    if requested_output.exists() or requested_output.is_symlink():
        raise ValueError("Choose a new output directory, not an existing path or symlink")
    source, output = Path(input_dir).resolve(), requested_output.resolve()
    sources = [Path(__file__), Path(_connected_groups.__code__.co_filename)]
    generation_hashes = {p.name: digest(p.read_bytes()) for p in sources}
    if output.is_relative_to(source) or output.exists():
        raise ValueError("Choose a new output directory outside the archive")
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["status"] != "complete_bounded_capture":
        raise ValueError("Cannot evaluate an incomplete capture")
    expected = manifest["files_sha256"]
    for name, sha in expected.items():
        if Path(name).name != name or digest((source / name).read_bytes()) != sha:
            raise ValueError("Archive hash mismatch: " + name)
    protocol = json.loads((source / "protocol.json").read_text())
    if digest((source / "protocol.json").read_bytes()) != manifest["protocol_sha256"]:
        raise ValueError("Protocol hash mismatch")
    news = json.loads((source / "news.json").read_text())
    calendar = json.loads((source / "calendar.json").read_text())
    prices, counts = {}, Counter()
    allowed_days = {s["date"] for s in calendar["sessions"]}
    price_requests = [r for r in manifest["requests"] if r["file"].startswith("prices_")]
    if not price_requests or manifest.get("bars_summary", {}).get("pagination_complete") is not True:
        raise ValueError("Price pagination is not certified complete for the requested query")
    prior_token = None
    for page_index, request in enumerate(price_requests):
        if request["file"] not in expected or request.get("status") != 200:
            raise ValueError("Unbound or failed price response")
        if request["params"].get("page_token") != prior_token:
            raise ValueError("Price pagination chain is broken")
        if not request["file"].startswith("prices_"):
            continue
        if request["params"].get("adjustment") != "split" or request["params"].get("timeframe") != "1Day":
            raise ValueError("This track requires declared split-adjusted daily bars")
        payload = json.loads((source / request["file"]).read_text())
        prior_token = payload.get("next_page_token")
        if prior_token != request.get("next_page_token"):
            raise ValueError("Price pagination token does not match retained response")
        if (page_index < len(price_requests) - 1) != bool(prior_token):
            raise ValueError("Price page sequence is incomplete or has extra pages")
        for symbol, bars in payload["bars"].items():
            for bar in bars:
                day = datetime.fromisoformat(bar["t"].replace("Z", "+00:00")).astimezone(NY).date().isoformat()
                value = bar["o"]
                if day not in allowed_days or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError("Invalid observed opening price/calendar date")
                if (symbol, day) in prices:
                    raise ValueError("Duplicate price endpoint")
                prices[(symbol, day)] = value
                counts[symbol] += 1
    if dict(counts) != manifest["bars_summary"]["records"] or any(not counts[s] for s in [protocol["symbol"], "SPY"]):
        raise ValueError("Price count/required-symbol audit failed")
    rows, outside = build_rows(news, prices, calendar, protocol)
    split = retrospective_split(rows, protocol["test_start"])
    metrics, predictions, models = evaluate_models(split, protocol)
    costs = cost_evaluation(predictions, protocol["assumed_roundtrip_cost_bps"])
    generated = datetime.now(timezone.utc).isoformat()
    report = {"status": "Retrospective development experiment completed", "data_kind": "observed_retrospective",
              "generated_at": generated, "predictions_actually_generated_at": generated,
              "historical_availability_verified": False, "strict_prospective_eligible_labels": 0,
              "market_predictive_skill_established": False, "protocol": protocol,
              "headlines": len(news), "company_days": len(rows), "price_bars": dict(counts),
              "label_status_counts": dict(Counter(r["label_status"] for r in rows)),
              "train_rows": len(split["train"]), "test_rows": len(split["test"]),
              "exclusion_counts": dict(Counter(reason for r in split["excluded"] for reason in r["reasons"])),
              "class_counts": {p: dict(Counter(r["price_class"] for r in split[p])) for p in ("train", "test")},
              "metrics": metrics, "cost_evaluation": costs,
              "training_clock_policy": "Historical outcome realization precedes cutoff; actual capture/version availability remains unknown",
              "limits": protocol["limitations"], "provenance": {"input_files_sha256": expected,
              "manifest_sha256": digest((source / "manifest.json").read_bytes()),
              "protocol_sha256": manifest["protocol_sha256"],
              "versions": {p: importlib.metadata.version(p) for p in ["numpy", "scipy", "scikit-learn", "exchange_calendars"]}}}
    if any(digest((source / name).read_bytes()) != sha for name, sha in expected.items()):
        raise ValueError("Input changed during evaluation")
    if any(digest(p.read_bytes()) != generation_hashes[p.name] for p in sources):
        raise ValueError("Generation source changed during evaluation")
    output.mkdir(mode=0o700, parents=True)
    (output / ".gitignore").write_text("*\n")
    documents = {"report.json": report, "company_days.json": rows, "split.json": split,
                 "predictions.json": predictions, "fitted_models.json": models, "outside_calendar.json": outside}
    for name, value in documents.items():
        save(output / name, value)
    for name, sha in expected.items():
        if digest((source / name).read_bytes()) != sha:
            raise ValueError("Input changed during evaluation")
    save(output / "run_manifest.json", {"generated_at": generated,
         "generation_sources_sha256": generation_hashes,
         "outputs_sha256": {name: digest((output / name).read_bytes()) for name in documents}})
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args(argv)
    r = run(a.input_dir, a.output_dir)
    print(json.dumps({k: r[k] for k in ["status", "headlines", "company_days", "train_rows", "test_rows", "metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
