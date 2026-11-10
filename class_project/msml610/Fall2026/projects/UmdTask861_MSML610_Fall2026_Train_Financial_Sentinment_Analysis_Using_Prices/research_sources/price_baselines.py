"""Offline, paired baseline comparisons for the price-label research contract.

No market data, model downloads, trading modules, or third-party packages are used.
The bundled demonstration is synthetic software-validation material, not evidence
of predictive skill. Returns and prediction errors use decimal return units.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


PRICE_FEATURES = ("momentum_5d", "momentum_20d")
TONE_FEATURE = "tone_score"
MODEL_FEATURES = {
    "train_mean": (),
    "momentum_ridge": ("momentum_5d",),
    "tone_ridge": (TONE_FEATURE,),
    "price_ridge": PRICE_FEATURES,
    "price_tone_ridge": (*PRICE_FEATURES, TONE_FEATURE),
}


def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{field} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _solve(matrix, target):
    """Solve a small ridge normal equation with partial pivoting."""
    augmented = [list(row) + [value] for row, value in zip(matrix, target)]
    for column in range(len(target)):
        pivot = max(range(column, len(target)), key=lambda r: abs(augmented[r][column]))
        if abs(augmented[pivot][column]) < 1e-14:
            raise ValueError("The training design is numerically singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(len(target)):
            if row == column:
                continue
            scale = augmented[row][column]
            augmented[row] = [a - scale * b for a, b in zip(augmented[row], augmented[column])]
    coefficients = [row[-1] for row in augmented]
    if not all(math.isfinite(x) for x in coefficients):
        raise ValueError("The training fit produced non-finite coefficients")
    return coefficients


def _fit(rows, features, alpha):
    targets = [row["excess_return"] for row in rows]
    intercept = sum(targets) / len(targets)
    means = {f: sum(row["feature_values"][f] for row in rows) / len(rows) for f in features}
    scales = {f: math.sqrt(sum((row["feature_values"][f] - means[f]) ** 2 for row in rows)
                           / len(rows)) or 1.0 for f in features}
    if not all(math.isfinite(value) for value in [intercept, *means.values(), *scales.values()]):
        raise ValueError("Training values exceed the supported numeric range")
    design = [[(row["feature_values"][f] - means[f]) / scales[f] for f in features] for row in rows]
    matrix = [[sum(x[a] * x[b] for x in design) + (alpha if a == b else 0.0)
               for b in range(len(features))] for a in range(len(features))]
    right = [sum(x[a] * (target - intercept) for x, target in zip(design, targets))
             for a in range(len(features))]
    coefficients = _solve(matrix, right) if features else []
    return {"features": list(features), "intercept": intercept, "means": means,
            "scales": scales, "coefficients": dict(zip(features, coefficients)),
            "ridge_alpha": alpha, "fitted_sample_ids": [row["sample_id"] for row in rows]}


def _predict(fitted, row):
    result = fitted["intercept"] + sum(
        fitted["coefficients"][f] * (row["feature_values"][f] - fitted["means"][f]) / fitted["scales"][f]
        for f in fitted["features"])
    if not math.isfinite(result):
        raise ValueError("Prediction must be finite")
    return result


def _sign(value):
    return (value > 0) - (value < 0)


def compare_baselines(split, *, data_kind, ridge_alpha=1.0):
    """Fit fixed models on one complete-case training set and compare paired tests.

    ``split`` must come from ``price_labels.make_time_split``. Neither model
    selection nor standardization uses test features or outcomes. All models use
    the same intersection of samples; missing features are never replaced by 0.
    This function does not tune hyperparameters or estimate trading returns.
    """
    if data_kind not in {"synthetic", "observed"}:
        raise ValueError("data_kind must explicitly be synthetic or observed")
    alpha = _number(ridge_alpha, "ridge_alpha")
    if alpha <= 0:
        raise ValueError("ridge_alpha must be positive and fixed before evaluation")
    needed = (*PRICE_FEATURES, TONE_FEATURE)
    paired, exclusions, seen = {}, [], set()
    for partition in ("train", "test"):
        rows = split.get(partition)
        if not isinstance(rows, list):
            raise ValueError("split must contain train and test lists")
        paired[partition] = []
        for original in rows:
            sample_id = original.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
                raise ValueError("Sample IDs must be nonempty and unique across both partitions")
            seen.add(sample_id)
            if original.get("status") != "Eligible" or original.get("label_status") != "Available":
                raise ValueError("Comparisons require eligible samples with available labels")
            features = original.get("feature_values", {})
            if not isinstance(features, dict):
                raise ValueError("feature_values must be an object")
            missing = [f for f in needed if features.get(f) is None]
            if original.get("excess_return") is None:
                missing.append("excess_return")
            if missing:
                exclusions.append({"sample_id": sample_id, "partition": partition,
                                   "reason": "missing_paired_fields", "fields": missing})
                continue
            paired[partition].append({"sample_id": sample_id,
                                      "feature_values": {f: _number(features[f], f) for f in needed},
                                      "excess_return": _number(original["excess_return"], "excess_return")})
    report = {
        "schema_version": "price-baseline-comparison-1.0",
        "mode": "Synthetic contract check" if data_kind == "synthetic" else "Observed-data research contract",
        "evaluation_status": "Not evaluated",
        "units": "decimal excess return over SPY",
        "configuration": {"horizon_sessions": split.get("horizon_sessions"), "ridge_alpha": alpha,
                          "model_features": {name: list(features) for name, features in MODEL_FEATURES.items()},
                          "hyperparameter_search": False, "scaling_fit": "training samples only"},
        "coverage": {"train_before_pairing": len(split["train"]), "test_before_pairing": len(split["test"]),
                     "paired_train": len(paired["train"]), "paired_test": len(paired["test"])},
        "paired_train_ids": [r["sample_id"] for r in paired["train"]],
        "paired_test_ids": [r["sample_id"] for r in paired["test"]],
        "excluded": exclusions, "models": {}, "paired_loss_deltas": [],
        "limitations": [
            "Synthetic examples verify software contracts only; they provide no market-performance evidence.",
            "Input time and provenance declarations require independent source audits.",
            "Overlapping multi-session outcomes are correlated; row counts are not independent evidence.",
            "No confidence intervals, model selection, transaction costs, portfolio simulation or A/B results are estimated.",
            "A pretrained tone feature is not a price probability; its linear mapping is fitted only on training outcomes.",
        ],
    }
    if len(paired["train"]) < max(5, len(needed) + 2) or not paired["test"]:
        report["reason"] = "Insufficient paired eligible samples (at least five training rows and one test row)."
        return report
    for name, features in MODEL_FEATURES.items():
        fitted = _fit(paired["train"], features, alpha)
        predictions = []
        for row in paired["test"]:
            estimate = _predict(fitted, row)
            actual = row["excess_return"]
            predictions.append({"sample_id": row["sample_id"], "observed_excess_return": actual,
                                "predicted_excess_return": estimate, "absolute_error": abs(estimate - actual)})
        n = len(predictions)
        report["models"][name] = {"fit": fitted, "predictions": predictions,
                                  "metrics": {"samples": n,
                                              "mae": sum(r["absolute_error"] for r in predictions) / n,
                                              "mse": sum(r["absolute_error"] ** 2 for r in predictions) / n,
                                              "direction_accuracy": sum(_sign(r["observed_excess_return"]) == _sign(r["predicted_excess_return"])
                                                                        for r in predictions) / n}}
    for candidate, reference in (("price_tone_ridge", "price_ridge"), ("tone_ridge", "train_mean")):
        deltas = [{"sample_id": left["sample_id"], "absolute_error_delta": left["absolute_error"] - right["absolute_error"]}
                  for left, right in zip(report["models"][candidate]["predictions"], report["models"][reference]["predictions"])]
        report["paired_loss_deltas"].append({"candidate": candidate, "reference": reference,
                                              "meaning": "candidate absolute error minus reference; negative is lower error",
                                              "mean_absolute_error_delta": sum(r["absolute_error_delta"] for r in deltas) / len(deltas),
                                              "samples": deltas})
    report["evaluation_status"] = "Synthetic check completed" if data_kind == "synthetic" else "Research comparison completed"
    return report


def synthetic_dataset():
    """Construct artificial consecutive-day sessions, explicitly not an exchange calendar."""
    ny = ZoneInfo("America/New_York")
    days = [date(2026, 2, 1) + timedelta(days=i) for i in range(80)]
    sessions = [{"date": day.isoformat(), "open_at": datetime.combine(day, time(9, 30), ny).isoformat(),
                 "close_at": datetime.combine(day, time(16), ny).isoformat()} for day in days]
    values = {"DEMO": [100 * math.exp(.001 * i + .02 * math.sin(i / 4)) for i in range(80)],
              "SPY": [100 * math.exp(.0003 * i) for i in range(80)]}
    prices = [{"symbol": symbol, "session_date": day.isoformat(), "open_value": sequence[i],
               "return_basis": "total_return_index", "available_at": (datetime.combine(day, time(9, 30), ny) + timedelta(minutes=1)).isoformat(),
               "provenance_id": f"synthetic-price-{symbol}-{i}"}
              for symbol, sequence in values.items() for i, day in enumerate(days)]
    observations = []
    for i in range(21, 66):
        day = days[i]
        features = {f"momentum_{h}d": (values["DEMO"][i-1] / values["DEMO"][i-1-h] - 1)
                    - (values["SPY"][i-1] / values["SPY"][i-1-h] - 1) for h in (5, 20)}
        features[TONE_FEATURE] = .8 * math.sin(i / 7)
        at = lambda hour, minute: datetime.combine(day, time(hour, minute), ny).isoformat()
        past_available = (datetime.combine(days[i-1], time(9, 30), ny) + timedelta(minutes=1)).isoformat()
        observations.append({"sample_id": f"synthetic-{i}", "symbol": "DEMO", "decision_session": day.isoformat(),
                             "source_ids": [f"synthetic-news-{i}"], "event_cluster_ids": [f"synthetic-event-{i}"],
                             "published_at": at(7, 30), "updated_at": at(7, 30), "response_received_at": at(8, 0),
                             "extraction_completed_at": at(8, 50), "prediction_completed_at": at(9, 0),
                             "historical_availability_verified": True, "feature_values": features,
                             "feature_provenance": {f: {"available_at": at(8, 0) if f == TONE_FEATURE else past_available,
                                                        "provenance_id": f"synthetic-feature-{f}-{i}"} for f in features}})
    return {"data_kind": "synthetic", "synthetic_fixture": True,
            "as_of": datetime.combine(days[-1], time(17), ny).isoformat(),
            "calendar": {"calendar_id": "SYNTHETIC-CONSECUTIVE-DAYS-NOT-AN-EXCHANGE", "calendar_version": "fixture-1",
                         "complete": True, "sessions": sessions}, "prices": prices, "observations": observations,
            "evaluation": {"training_cutoff": datetime.combine(days[47], time(8), ny).isoformat(),
                           "test_start": sessions[48]["open_at"], "test_end": sessions[62]["open_at"]},
            "notes": ["All values, text features and sessions are artificial software fixtures.",
                      "Weekend sessions are intentional; this is not a market calendar or historical performance."]}


def evaluate_dataset(dataset, *, horizon_sessions=10, ridge_alpha=1.0):
    if __package__:
        from .price_labels import build_labels, make_time_split
    else:
        from price_labels import build_labels, make_time_split
    if dataset.get("data_kind") not in {"synthetic", "observed"}:
        raise ValueError("Dataset must declare data_kind as synthetic or observed")
    if dataset["data_kind"] == "observed" and (dataset.get("synthetic_fixture") is True
            or str(dataset.get("calendar", {}).get("calendar_id", "")).upper().startswith("SYNTHETIC")):
        raise ValueError("Synthetic fixtures cannot be relabeled as observed data")
    labels = build_labels(dataset, horizon_sessions=horizon_sessions)
    settings = dataset["evaluation"]
    split = make_time_split(labels, settings["training_cutoff"], settings["test_start"], settings["test_end"])
    split["horizon_sessions"] = horizon_sessions
    report = compare_baselines(split, data_kind=dataset["data_kind"], ridge_alpha=ridge_alpha)
    report["label_coverage"] = {status: sum(row["status"] == status for row in labels)
                                for status in ("Eligible", "Pending", "Unavailable", "Retrospective")}
    report["split"] = {k: v for k, v in split.items() if k not in {"train", "test"}}
    return labels, split, report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate fixed price/tone baselines offline in a new private run directory.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="JSON dataset following the documented price-label contract")
    source.add_argument("--demo", action="store_true", help="Use synthetic software fixtures, never market-performance evidence")
    parser.add_argument("--output-dir", required=True, type=Path, help="Must not already exist")
    parser.add_argument("--horizon-sessions", type=int, choices=(5, 10, 20), default=10)
    args = parser.parse_args(argv)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError("A new output directory is required")
    encoded = ((json.dumps(synthetic_dataset(), indent=2, allow_nan=False) + "\n").encode()
               if args.demo else args.input.read_bytes())
    dataset = json.loads(encoded)
    labels, split, report = evaluate_dataset(dataset, horizon_sessions=args.horizon_sessions)
    provenance = {"schema_version": "price-baseline-run-1.0", "generated_at": datetime.now(timezone.utc).isoformat(),
                  "input_sha256": hashlib.sha256(encoded).hexdigest(), "input_kind": dataset["data_kind"],
                  "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "label_module_sha256": hashlib.sha256(Path(__file__).with_name("price_labels.py").read_bytes()).hexdigest(),
                  "corporate_actions_module_sha256": hashlib.sha256(Path(__file__).with_name("corporate_actions.py").read_bytes()).hexdigest()}
    documents = {"labels.json": labels, "split.json": split, "comparison.json": report}
    if args.demo:
        documents["synthetic_input.json"] = dataset
    serialized = {name: json.dumps(value, indent=2, allow_nan=False) + "\n" for name, value in documents.items()}
    provenance["outputs"] = {name: {"bytes": len(data.encode("utf-8")),
                                    "sha256": hashlib.sha256(data.encode("utf-8")).hexdigest()}
                             for name, data in serialized.items()}
    serialized["run_manifest.json"] = json.dumps(provenance, indent=2, allow_nan=False) + "\n"
    args.output_dir.mkdir(parents=True, mode=0o700)
    (args.output_dir / ".gitignore").write_text("*\n", encoding="utf-8")
    for name, data in serialized.items():
        with (args.output_dir / name).open("x", encoding="utf-8") as stream:
            stream.write(data)
    print(json.dumps({"evaluation_status": report["evaluation_status"], "mode": report["mode"],
                      "label_coverage": report["label_coverage"], "coverage": report["coverage"]}))
    return report


if __name__ == "__main__":
    main()
