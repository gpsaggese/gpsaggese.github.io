"""Fixed offline FinBERT tone controls on the same historical paired samples.

Prepare/evaluate use the research Python; infer can use the existing isolated
NLP Python. No subprocess is launched and no model downloaded by importing this
module. This is an additional development comparator, not a fresh blind test.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

MODEL_ID = "Xenova/finbert"
REVISION = "8f269abebfdd9009d7d9b5e96af7e5c6bfe50b20"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save_new(directory, documents):
    directory = Path(directory)
    if directory.exists() or directory.is_symlink():
        raise FileExistsError("Choose a new output directory")
    directory.mkdir(mode=0o700, parents=True)
    (directory / ".gitignore").write_text("*\n")
    for name, data in documents.items():
        (directory / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def validate_outputs(base):
    base = Path(base)
    m = load(base / "run_manifest.json")
    for name, expected in m["outputs_sha256"].items():
        if Path(name).name != name or sha((base / name).read_bytes()) != expected:
            raise ValueError("Base output hash mismatch")


def prepare_packet(capture, base, registration):
    capture, base = Path(capture), Path(base)
    validate_outputs(base)
    manifest = load(capture / "manifest.json")
    if manifest["status"] != "complete_bounded_capture":
        raise ValueError("Capture is incomplete")
    for name, expected in manifest["files_sha256"].items():
        if Path(name).name != name or sha((capture / name).read_bytes()) != expected:
            raise ValueError("Capture hash mismatch")
    policy = load(registration)["finbert"]
    if policy["model_id"] != MODEL_ID or policy["revision"] != REVISION or policy["ridge_alpha"] != 10.0:
        raise ValueError("Comparison differs from the registered fixed protocol")
    split, news = load(base / "split.json"), load(capture / "news.json")
    by_index = {r["source_row_index"]: r for r in news}
    if len(by_index) != len(news):
        raise ValueError("Duplicate raw source positions")
    rows, seen = [], set()
    for row in split["train"] + split["test"]:
        if row["sample_id"] in seen:
            raise ValueError("Duplicate paired sample")
        seen.add(row["sample_id"])
        texts = list(dict.fromkeys(by_index[i]["headline"] for i in row["source_row_indices"]))
        if not texts or "\n".join(texts) != row["text"]:
            raise ValueError("FinBERT and TF-IDF input budgets differ")
        rows.append({"sample_id": row["sample_id"], "headlines": texts,
                     "headline_hashes": [sha(t.encode()) for t in texts]})
    return {"schema_version": "same-headline-tone-request-1.0", "rows": rows,
            "model_id": MODEL_ID, "revision": REVISION,
            "source_split_sha256": sha((base / "split.json").read_bytes()),
            "source_news_sha256": sha((capture / "news.json").read_bytes()),
            "registration_sha256": sha(Path(registration).read_bytes()),
            "inputs_contain_prices_or_targets": False}


def infer_packet(packet, classifier=None, cached_snapshot=None):
    if packet["model_id"] != MODEL_ID or packet["revision"] != REVISION:
        raise ValueError("Model identity mismatch")
    texts = []
    seen = set()
    for row in packet["rows"]:
        if set(row) != {"sample_id", "headlines", "headline_hashes"}:
            raise ValueError("Inference packet contains unexpected fields")
        if row["sample_id"] in seen or not row["headlines"] or len(set(row["headlines"])) != len(row["headlines"]):
            raise ValueError("Duplicate sample or empty/repeated headline budget")
        seen.add(row["sample_id"])
        if [sha(t.encode()) for t in row["headlines"]] != row["headline_hashes"]:
            raise ValueError("Headline hash mismatch")
        texts.extend(row["headlines"])
    unique = list(dict.fromkeys(texts))
    if classifier is None:
        from sentiment_model import FinBertTone
        classifier = FinBertTone(offline=True, cached_snapshot=cached_snapshot)
    metadata = classifier.metadata
    if metadata["model_id"] != MODEL_ID or metadata["revision"] != REVISION:
        raise ValueError("Loaded model differs from registered revision")
    predictions = classifier.predict(unique)  # only text, never returns/prices
    if len(predictions) != len(unique):
        raise ValueError("Missing headline inference")
    cache = {}
    for text, result in zip(unique, predictions):
        if result.get("input_sha256") != sha(text.encode()) or result.get("status") != "Available":
            raise ValueError("Inference result not bound to input")
        if result.get("truncated") is not False:
            raise ValueError("Truncated/unknown input budget; comparison refused")
        scores = result["scores"]
        if set(scores) != {"positive", "neutral", "negative"} or not all(
                isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0 <= x <= 1
                for x in scores.values()) or abs(sum(scores.values()) - 1) > 1e-5:
            raise ValueError("Invalid tone probabilities")
        cache[sha(text.encode())] = result
    rows = []
    for row in packet["rows"]:
        values = [cache[h]["scores"]["positive"] - cache[h]["scores"]["negative"] for h in row["headline_hashes"]]
        rows.append({"sample_id": row["sample_id"], "tone_score": sum(values) / len(values),
                     "headline_hashes": row["headline_hashes"], "headline_count": len(values)})
    return {"schema_version": "same-headline-tone-features-1.0", "samples": rows,
            "actual_inference_completed_at": datetime.now(timezone.utc).isoformat(),
            "model": metadata, "unique_headlines": len(unique), "headline_uses": len(texts),
            "headline_results": cache, "any_truncated": False,
            "source_split_sha256": packet["source_split_sha256"],
            "source_news_sha256": packet["source_news_sha256"],
            "registration_sha256": packet["registration_sha256"],
            "aggregation": "Mean of positive-minus-negative softmax across identical distinct headlines",
            "historical_model_availability_verified": False,
            "pretraining_overlap_independently_verified": False}


def validate_features(packet, features):
    """Recompute every aggregate from the bound per-headline inference receipts."""
    if len({r["sample_id"] for r in packet["rows"]}) != len(packet["rows"]):
        raise ValueError("Duplicate requested sample")
    requested = {r["sample_id"]: r for r in packet["rows"]}
    if len({r["sample_id"] for r in features["samples"]}) != len(features["samples"]):
        raise ValueError("Duplicate inferred sample")
    if set(requested) != {r["sample_id"] for r in features["samples"]}:
        raise ValueError("Different paired sample pool")
    expected_hashes = set()
    for row in features["samples"]:
        req = requested[row["sample_id"]]
        hashes = [sha(t.encode()) for t in req["headlines"]]
        if not hashes or hashes != req["headline_hashes"] or hashes != row["headline_hashes"] or row["headline_count"] != len(hashes):
            raise ValueError("Changed per-sample headline budget")
        values = []
        for h in hashes:
            result = features["headline_results"][h]
            scores = result["scores"]
            if result.get("input_sha256") != h or result.get("status") != "Available" or result.get("truncated") is not False:
                raise ValueError("Invalid headline receipt")
            if set(scores) != {"positive", "neutral", "negative"} or not all(
                    isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0 <= x <= 1
                    for x in scores.values()) or abs(sum(scores.values()) - 1) > 1e-5:
                raise ValueError("Invalid tone probabilities")
            values.append(scores["positive"] - scores["negative"])
        if not math.isfinite(row["tone_score"]) or not math.isclose(row["tone_score"], sum(values) / len(values), abs_tol=1e-12):
            raise ValueError("Tone feature differs from headline receipts")
        expected_hashes.update(hashes)
    if set(features["headline_results"]) != expected_hashes:
        raise ValueError("Unexpected headline receipts")


def evaluate_tone_controls(base, packet_path, features_path, registration, output):
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, balanced_accuracy_score, f1_score
    from run_historical import cost_evaluation, price_class

    base = Path(base)
    source_hash = sha(Path(__file__).read_bytes())
    validate_outputs(base)
    packet, features = load(packet_path), load(features_path)
    if features.get("request_file_sha256") != sha(Path(packet_path).read_bytes()):
        raise ValueError("Tone features do not bind the request bytes")
    if features["any_truncated"] is not False:
        raise ValueError("Different headline budgets")
    for field in ("source_split_sha256", "source_news_sha256", "registration_sha256"):
        if packet[field] != features[field]:
            raise ValueError("Tone source binding mismatch")
    if packet["registration_sha256"] != sha(Path(registration).read_bytes()):
        raise ValueError("Changed registration")
    if packet["source_split_sha256"] != sha((base / "split.json").read_bytes()):
        raise ValueError("Changed base split")
    if features["model"]["model_id"] != MODEL_ID or features["model"]["revision"] != REVISION:
        raise ValueError("Wrong model")
    validate_features(packet, features)
    split, report = load(base / "split.json"), load(base / "report.json")
    tone = {r["sample_id"]: r for r in features["samples"]}
    requested = {r["sample_id"]: r for r in packet["rows"]}
    ids = [r["sample_id"] for r in split["train"] + split["test"]]
    if len(tone) != len(features["samples"]) or set(tone) != set(ids) or set(requested) != set(ids):
        raise ValueError("Different paired sample pool")
    for identity in ids:
        if tone[identity]["headline_hashes"] != requested[identity]["headline_hashes"]:
            raise ValueError("Changed per-sample headline budget")
        if not math.isfinite(tone[identity]["tone_score"]):
            raise ValueError("Invalid tone feature")
    train, test = split["train"], split["test"]
    y = np.asarray([r["excess_return"] for r in train])
    truth = np.asarray([r["excess_return"] for r in test])
    baseline = load(base / "predictions.json")
    if [r["sample_id"] for r in baseline] != [r["sample_id"] for r in test]:
        raise ValueError("Base predictions have another row order")
    predictions = deepcopy(baseline)
    metrics, models = deepcopy(report["metrics"]), {}
    alpha = load(registration)["finbert"]["ridge_alpha"]
    if alpha != report["protocol"]["ridge_alpha"]:
        raise ValueError("Different ridge calibration budget")
    for name, include_price in [("finbert_tone_ridge", False), ("price_finbert_ridge", True)]:
        def design(rows):
            return [(r["price_features"] if include_price else []) + [tone[r["sample_id"]]["tone_score"]] for r in rows]
        scaler = StandardScaler().fit(design(train))
        model = Ridge(alpha=alpha, solver="lsqr", tol=1e-8).fit(scaler.transform(design(train)), y)
        values = model.predict(scaler.transform(design(test)))
        if not np.isfinite(values).all():
            raise ValueError("Invalid calibrated prediction")
        classes = [price_class(float(v), report["protocol"]["class_neutral_band"]) for v in values]
        targets = [r["price_class"] for r in test]
        metrics[name] = {"samples": len(test), "mae": float(mean_absolute_error(truth, values)),
                         "rmse": float(math.sqrt(mean_squared_error(truth, values))),
                         "relative_direction_accuracy": float(np.mean(np.sign(values) == np.sign(truth))),
                         "price_class_balanced_accuracy": float(balanced_accuracy_score(targets, classes)),
                         "price_class_macro_f1": float(f1_score(targets, classes, labels=[-1, 0, 1], average="macro", zero_division=0))}
        models[name] = {"training_sample_ids": [r["sample_id"] for r in train],
                        "means": scaler.mean_.tolist(), "scales": scaler.scale_.tolist(),
                        "intercept": float(model.intercept_), "coefficients": model.coef_.tolist(), "alpha": alpha}
        for row, value in zip(predictions, values):
            row["predictions"][name] = float(value)
    costs = cost_evaluation(predictions, report["protocol"]["assumed_roundtrip_cost_bps"])
    if costs["selected_sample_ids"] != report["cost_evaluation"]["selected_sample_ids"]:
        raise ValueError("Cost evaluation intervals changed")
    losses = [{"sample_id": row["sample_id"], "model": model,
               "absolute_loss": abs(row["predictions"][model] - row["excess_return"]),
               "loss_minus_price_ridge": abs(row["predictions"][model] - row["excess_return"]) - abs(row["predictions"]["price_ridge"] - row["excess_return"])}
              for row in predictions for model in metrics]
    result = {"status": "Same-sample retrospective development comparison completed", "train_rows": len(train),
              "test_rows": len(test), "metrics": metrics, "cost_evaluation": costs,
              "finbert": {k: features[k] for k in ["model", "unique_headlines", "headline_uses", "aggregation", "any_truncated", "actual_inference_completed_at"]},
              "headline_budget_identical": True, "historical_availability_verified": False,
              "market_predictive_skill_established": False, "original_strict_eligible_labels": 0,
              "comparison_scope": "Added comparator on already inspected development test; no fresh holdout or significance claim",
              "limitations": ["Pretraining overlap and historical model release availability not independently verified",
                              "FinBERT softmax describes tone; training-only calibration predicts relative price return",
                              "Cash dividends excluded; single company; date-only final historical headline versions"],
              "actual_evaluation_completed_at": datetime.now(timezone.utc).isoformat()}
    provenance = {"files_sha256": {str(p): sha(Path(p).read_bytes()) for p in [packet_path, features_path, registration, base / "split.json", base / "predictions.json"]},
                  "source_sha256": {"finbert_comparison.py": source_hash}}
    if sha(Path(__file__).read_bytes()) != source_hash or any(sha(Path(p).read_bytes()) != h for p, h in provenance["files_sha256"].items()):
        raise ValueError("Input or source changed during evaluation")
    save_new(output, {"report.json": result, "predictions.json": predictions,
                      "fitted_tone_models.json": models, "paired_losses.json": losses, "provenance.json": provenance})
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--capture", required=True, type=Path)
    prep.add_argument("--base", required=True, type=Path)
    prep.add_argument("--registration", required=True, type=Path)
    prep.add_argument("--output-dir", required=True, type=Path)
    inf = sub.add_parser("infer")
    inf.add_argument("--packet", required=True, type=Path)
    inf.add_argument("--output-dir", required=True, type=Path)
    inf.add_argument("--cached-snapshot", type=Path, help="Pinned local cache; avoids importing the Hub client")
    ev = sub.add_parser("evaluate")
    for name in ["base", "packet", "features", "registration", "output-dir"]:
        ev.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args(argv)
    if a.action == "prepare":
        packet = prepare_packet(a.capture, a.base, a.registration)
        save_new(a.output_dir, {"tone_request.json": packet})
        print(json.dumps({"samples": len(packet["rows"]), "status": "Prepared identical headline budget"}))
    elif a.action == "infer":
        result = infer_packet(load(a.packet), cached_snapshot=a.cached_snapshot)
        result["request_file_sha256"] = sha(a.packet.read_bytes())
        save_new(a.output_dir, {"tone_features.json": result})
        print(json.dumps({k: result[k] for k in ["unique_headlines", "headline_uses", "any_truncated"]}))
    else:
        result = evaluate_tone_controls(a.base, a.packet, a.features, a.registration, a.output_dir)
        print(json.dumps(result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
