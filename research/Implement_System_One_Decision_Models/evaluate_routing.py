"""Evaluate frozen support-ticket predictions and write plots and metrics.

Run from the repository root:

> python research/Implement_System_One_Decision_Models/evaluate_routing.py
"""

import argparse
import hashlib
import json
import math
import os
import statistics
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import research.Implement_System_One_Decision_Models.ticket_routing as troute

# #############################################################################
# Paths and validation
# #############################################################################


PROJECT_DIR = os.path.dirname(__file__)
DEV_PATH = os.path.join(PROJECT_DIR, "dev_predictions.json")
TEST_PATH = os.path.join(PROJECT_DIR, "test_predictions.json")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "evaluation")


def _load_report(path: str, split: str) -> Dict[str, Any]:
    """Load predictions and verify their relation to the frozen dataset."""
    with open(path, encoding="utf-8") as stream:
        report = json.load(stream)
    if report["split"] != split:
        raise ValueError(f"Expected {split} predictions in {path}")
    if report["question"] != troute.ROUTE_QUESTION:
        raise ValueError("Prompt differs from the frozen application")
    if report["action_costs"] != troute.ACTION_COSTS:
        raise ValueError("Cost table differs from the frozen application")
    if report["model_revision"] != troute.MODEL_REVISION:
        raise ValueError("Model revision differs from the frozen application")
    if report["model_sha256"] != troute.MODEL_SHA256:
        raise ValueError("Model file differs from the frozen application")
    with open(troute.DATA_PATH, "rb") as stream:
        dataset_sha256 = hashlib.sha256(stream.read()).hexdigest()
    if report["dataset_sha256"] != dataset_sha256:
        raise ValueError("Prediction dataset hash differs from current tickets")
    cases = troute.load_cases(split)
    records = report["records"]
    if len(records) != len(cases):
        raise ValueError("Prediction count differs from the frozen split")
    for record, case in zip(records, cases):
        if (record["id"], record["true_route"], record["text"]) != (
            case["id"],
            case["label"],
            case["text"],
        ):
            raise ValueError(
                f"Prediction and source case differ at {case['id']}"
            )
        probabilities = record["probabilities"]
        if set(probabilities) != set(troute.ROUTES) or not math.isclose(
            sum(probabilities.values()), 1.0, abs_tol=1e-3
        ):
            raise ValueError(f"Invalid probabilities for {case['id']}")
        if record["top_route"] != max(probabilities, key=probabilities.get):
            raise ValueError(
                f"Top route differs from probabilities for {case['id']}"
            )
        if record["policy_action"] != troute.choose_min_cost_action(
            probabilities
        ):
            raise ValueError(
                f"Policy action differs from costs for {case['id']}"
            )
    return report


# #############################################################################
# Metrics
# #############################################################################


def classification_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute multiclass accuracy, Brier score, and negative log likelihood."""
    n_cases = len(records)
    if n_cases == 0:
        raise ValueError("No predictions to evaluate")
    accuracy = (
        sum(record["top_route"] == record["true_route"] for record in records)
        / n_cases
    )
    brier = (
        sum(
            sum(
                (
                    record["probabilities"][route]
                    - (route == record["true_route"])
                )
                ** 2
                for route in troute.ROUTES
            )
            for record in records
        )
        / n_cases
    )
    nll = (
        -sum(
            math.log(max(record["probabilities"][record["true_route"]], 1e-15))
            for record in records
        )
        / n_cases
    )
    return {"accuracy": accuracy, "multiclass_brier": brier, "nll": nll}


def reliability_bins(
    records: List[Dict[str, Any]], n_bins: int = 5
) -> Tuple[List[Dict[str, Any]], float]:
    """Group top-label accuracy and confidence into equal-width bins."""
    bins = []
    total = len(records)
    ece = 0.0
    for index in range(n_bins):
        lower = index / n_bins
        upper = (index + 1) / n_bins
        members = [
            record
            for record in records
            if lower <= max(record["probabilities"].values()) < upper
            or (
                index == n_bins - 1
                and max(record["probabilities"].values()) == 1.0
            )
        ]
        count = len(members)
        mean_confidence = (
            statistics.mean(
                max(record["probabilities"].values()) for record in members
            )
            if members
            else None
        )
        accuracy = (
            sum(
                record["top_route"] == record["true_route"] for record in members
            )
            / count
            if members
            else None
        )
        if members:
            ece += count / total * abs(accuracy - mean_confidence)
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    return bins, ece


def confusion_table(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """Count top-route predictions by true route."""
    table = {
        route: {predicted: 0 for predicted in troute.ROUTES}
        for route in troute.ROUTES
    }
    for record in records:
        table[record["true_route"]][record["top_route"]] += 1
    return table


def policy_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Measure review coverage, routed error, and realized illustrative cost."""
    routed = [
        record for record in records if record["policy_action"] != "review"
    ]
    return {
        "coverage": len(routed) / len(records),
        "routed_count": len(routed),
        "review_count": len(records) - len(routed),
        "routed_error_rate": (
            sum(
                record["policy_action"] != record["true_route"]
                for record in routed
            )
            / len(routed)
            if routed
            else None
        ),
        "mean_realized_cost": statistics.mean(
            troute.actual_cost(record["policy_action"], record["true_route"])
            for record in records
        ),
        "top_route_mean_realized_cost": statistics.mean(
            troute.actual_cost(record["top_route"], record["true_route"])
            for record in records
        ),
    }


def selective_curve(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sweep routed coverage in order of increasing predicted route cost."""
    ranked = sorted(
        records,
        key=lambda record: min(
            record["expected_costs"][route] for route in troute.ROUTES
        ),
    )
    curve = []
    errors = 0
    for count, record in enumerate(ranked, start=1):
        route = min(troute.ROUTES, key=record["expected_costs"].get)
        errors += route != record["true_route"]
        curve.append(
            {
                "coverage": count / len(records),
                "routed_error_rate": errors / count,
            }
        )
    return curve


# #############################################################################
# Figures and report
# #############################################################################


def _plot_reliability(bins: List[Dict[str, Any]], output_path: str) -> None:
    """Save top-label reliability with per-bin sample counts."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#555555", label="perfect")
    nonempty = [item for item in bins if item["count"]]
    ax.plot(
        [item["mean_confidence"] for item in nonempty],
        [item["accuracy"] for item in nonempty],
        marker="o",
        color="#287bb5",
        label="test bins",
    )
    for item in nonempty:
        ax.annotate(
            f"n={item['count']}",
            (item["mean_confidence"], item["accuracy"]),
            xytext=(-37 if item["mean_confidence"] > 0.9 else 5, -10),
            textcoords="offset points",
        )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean top-label probability",
        ylabel="Top-label accuracy",
    )
    ax.set_title("Ticket routing reliability (36 synthetic test cases)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_coverage(
    curve: List[Dict[str, Any]], policy: Dict[str, Any], output_path: str
) -> None:
    """Save routed error versus coverage and mark the chosen review policy."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        [point["coverage"] for point in curve],
        [point["routed_error_rate"] for point in curve],
        color="#287bb5",
        label="lowest predicted route cost first",
    )
    if policy["routed_count"]:
        ax.scatter(
            [policy["coverage"]],
            [policy["routed_error_rate"]],
            color="#c34f3d",
            s=70,
            label="fixed review cost policy",
            zorder=3,
        )
    max_error = max(point["routed_error_rate"] for point in curve)
    ax.set(
        xlim=(0, 1),
        ylim=(0, max(0.15, max_error * 1.25)),
        xlabel="Fraction routed automatically",
        ylabel="Error among routed cases",
    )
    ax.set_title("Coverage and routed error (36 synthetic test cases)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def evaluate(dev_path: str, test_path: str, output_dir: str) -> Dict[str, Any]:
    """Validate frozen artifacts, calculate metrics, and save figures."""
    dev = _load_report(dev_path, "dev")
    test = _load_report(test_path, "test")
    dev_records = dev["records"]
    test_records = test["records"]
    majority_route = max(
        troute.ROUTES,
        key=lambda route: sum(
            record["true_route"] == route for record in dev_records
        ),
    )
    bins, ece = reliability_bins(test_records)
    policy = policy_metrics(test_records)
    curve = selective_curve(test_records)
    results = {
        "n_dev": len(dev_records),
        "n_test": len(test_records),
        "dataset_sha256": test["dataset_sha256"],
        "model_revision": test["model_revision"],
        "model_sha256": test["model_sha256"],
        "llama_cpp_commit": test["llama_cpp_commit"],
        "prompt_version": test["prompt_version"],
        "dev": classification_metrics(dev_records),
        "test": classification_metrics(test_records),
        "majority_route": majority_route,
        "majority_accuracy": sum(
            record["true_route"] == majority_route for record in test_records
        )
        / len(test_records),
        "top_label_ece_5_bins": ece,
        "reliability_bins": bins,
        "top_route_confusion": confusion_table(test_records),
        "policy": policy,
        "selective_curve": curve,
        "median_request_seconds": statistics.median(
            record["seconds"] for record in test_records
        ),
        "errors": [
            {
                "id": record["id"],
                "true_route": record["true_route"],
                "top_route": record["top_route"],
                "policy_action": record["policy_action"],
                "probabilities": record["probabilities"],
            }
            for record in test_records
            if record["top_route"] != record["true_route"]
        ],
    }
    os.makedirs(output_dir, exist_ok=True)
    _plot_reliability(bins, os.path.join(output_dir, "reliability.png"))
    _plot_coverage(curve, policy, os.path.join(output_dir, "coverage.png"))
    with open(
        os.path.join(output_dir, "results.json"), "w", encoding="utf-8"
    ) as stream:
        json.dump(results, stream, indent=2)
        stream.write("\n")
    return results


def main() -> None:
    """Evaluate saved development and test predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", default=DEV_PATH)
    parser.add_argument("--test", default=TEST_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()
    results = evaluate(args.dev, args.test, args.output_dir)
    print(
        json.dumps(
            {"test": results["test"], "policy": results["policy"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
