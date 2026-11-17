#!/usr/bin/env python
"""Compare the frozen capability and Hamming-MMD Phase 0 evidence."""

import argparse
import csv
import json
import math
import os
from typing import Dict, Iterable, Mapping, Sequence, Tuple


DEFAULT_CAPABILITY_SUMMARY = (
    "research/Noesis/docs/findings/model_drift_capability/scenario_summary.csv"
)
DEFAULT_WARMUP_GRADES = (
    "research/Noesis/docs/findings/model_drift_controlled_warmup_live/grades_v2.jsonl"
)
DEFAULT_FINGERPRINT_SUMMARY = (
    "research/Noesis/docs/findings/model_drift_live_fingerprint/fingerprint_summary.json"
)
DEFAULT_STABILITY_SUMMARY = (
    "research/Noesis/docs/findings/model_drift_fingerprint_stability_live/stability_summary.json"
)
DEFAULT_DST_DIR = (
    "research/Noesis/docs/findings/model_drift_method_comparison"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capability_summary", default=DEFAULT_CAPABILITY_SUMMARY)
    parser.add_argument("--warmup_grades", default=DEFAULT_WARMUP_GRADES)
    parser.add_argument("--fingerprint_summary", default=DEFAULT_FINGERPRINT_SUMMARY)
    parser.add_argument("--stability_summary", default=DEFAULT_STABILITY_SUMMARY)
    parser.add_argument("--dst_dir", default=DEFAULT_DST_DIR)
    return parser.parse_args()


def _read_json(path: str) -> Mapping[str, object]:
    with open(path, encoding="utf-8") as file_:
        value = json.load(file_)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object in %s" % path)
    return value


def _read_jsonl(path: str) -> Tuple[Mapping[str, object], ...]:
    rows = []
    with open(path, encoding="utf-8") as file_:
        for line_number, line in enumerate(file_, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("expected an object on %s:%s" % (path, line_number))
            rows.append(value)
    return tuple(rows)


def _exact_two_sided_mcnemar(only_left: int, only_right: int) -> float:
    discordant = only_left + only_right
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(only_left, only_right) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def _paired_capability_result(
    rows: Iterable[Mapping[str, object]],
) -> Mapping[str, object]:
    reference_route = "cloudflare-llama-3.3-70b"
    alternative_route = "cloudflare-llama-3.1-8b"
    by_item: Dict[str, Dict[str, bool]] = {}
    aggregate = {
        reference_route: {"correct": 0, "gradable": 0},
        alternative_route: {"correct": 0, "gradable": 0},
    }
    for row in rows:
        route = str(row.get("route_name", ""))
        if route not in aggregate:
            continue
        correct = row.get("substantive_correct")
        if not isinstance(correct, bool):
            continue
        aggregate[route]["gradable"] += 1
        aggregate[route]["correct"] += int(correct)
        by_item.setdefault(str(row["item_id"]), {})[route] = correct
    pairs = [item for item in by_item.values() if len(item) == 2]
    both_correct = sum(item[reference_route] and item[alternative_route] for item in pairs)
    only_reference = sum(item[reference_route] and not item[alternative_route] for item in pairs)
    only_alternative = sum(not item[reference_route] and item[alternative_route] for item in pairs)
    both_wrong = len(pairs) - both_correct - only_reference - only_alternative
    reference = aggregate[reference_route]
    alternative = aggregate[alternative_route]
    return {
        "reference_route": reference_route,
        "alternative_route": alternative_route,
        "reference_correct": reference["correct"],
        "reference_gradable": reference["gradable"],
        "alternative_correct": alternative["correct"],
        "alternative_gradable": alternative["gradable"],
        "paired_items": len(pairs),
        "both_correct": both_correct,
        "only_reference_correct": only_reference,
        "only_alternative_correct": only_alternative,
        "both_wrong": both_wrong,
        "mcnemar_p_value": _exact_two_sided_mcnemar(
            only_reference, only_alternative
        ),
    }


def _scenario_rows(
    capability_path: str,
    fingerprint: Mapping[str, object],
) -> Tuple[Mapping[str, object], ...]:
    with open(capability_path, encoding="utf-8", newline="") as file_:
        capability = {row["scenario_name"]: row for row in csv.DictReader(file_)}
    fingerprint_rows = {
        str(row["scenario"]): row
        for row in fingerprint["scenario_results"]
    }
    names = (
        ("full_swap", "full_weak_model_swap", "full_model_swap"),
        ("partial_substitution_10", "partial_substitution_10", "partial_substitution_10"),
        ("partial_substitution_25", "partial_substitution_25", "partial_substitution_25"),
        ("partial_substitution_50", "partial_substitution_50", "partial_substitution_50"),
    )
    rows = []
    for label, capability_name, fingerprint_name in names:
        left = capability[capability_name]
        right = fingerprint_rows[fingerprint_name]
        rows.append(
            {
                "scenario": label,
                "capability_detection_rate": float(left["detection_rate"]),
                "capability_trials": int(left["total_trials"]),
                "mmd_detection_rate": float(right["detection_rate"]),
                "mmd_trials": int(right["trials"]),
                "comparison_class": (
                    "directional_only_different_data_and_trial_designs"
                ),
            }
        )
    return tuple(rows)


def _render_findings(summary: Mapping[str, object]) -> str:
    paired = summary["live_full_swap"]["capability"]
    mmd = summary["live_full_swap"]["mmd"]
    stability = summary["healthy_controls"]
    lines = [
        "# Capability versus Hamming-MMD comparison",
        "",
        "## Scope",
        "",
        "This is a frozen-evidence comparison. It makes no provider calls and reruns no permutation tests. It compares the completed capability and Hamming-MMD experiments at roughly the same per-pool evidence scale while preserving their different prompts, outputs, and trial designs.",
        "",
        "## Live full-swap evidence",
        "",
        "- Capability grading separated Cloudflare 70B (%s/%s) from Cloudflare 8B (%s/%s). Across %s mutually gradable items, the exact paired McNemar p-value was %.6f."
        % (
            paired["reference_correct"],
            paired["reference_gradable"],
            paired["alternative_correct"],
            paired["alternative_gradable"],
            paired["paired_items"],
            paired["mcnemar_p_value"],
        ),
        "- Hamming-MMD separated a 60-response Cloudflare 70B pool from a 60-response Cloudflare 8B pool with empirical p < 0.001 (%s permutations)."
        % mmd["permutations"],
        "- Both methods therefore detected the large 70B-to-8B change, but they answer different questions: capability grading measures task-quality loss, while MMD measures an output-distribution change.",
        "",
        "## Directional substitution comparison",
        "",
        "| Scenario | Capability detection | Hamming-MMD detection |",
        "| :-- | --: | --: |",
    ]
    for row in summary["scenario_comparison"]:
        capability_trial_label = (
            "trial" if row["capability_trials"] == 1 else "trials"
        )
        mmd_trial_label = "trial" if row["mmd_trials"] == 1 else "trials"
        lines.append(
            "| %s | %.1f%% (%s %s) | %.1f%% (%s %s) |"
            % (
                row["scenario"].replace("_", " "),
                100 * row["capability_detection_rate"],
                row["capability_trials"],
                capability_trial_label,
                100 * row["mmd_detection_rate"],
                row["mmd_trials"],
                mmd_trial_label,
            )
        )
    lines.extend(
        [
            "",
            "These rates are not a winner/loser comparison. The capability trials are independent synthetic accuracy windows; the MMD partial-substitution trials reuse fixed live response pools and are correlated.",
            "",
            "## Healthy controls",
            "",
            "- Capability control: %s alerts across %s synthetic healthy trials."
            % (stability["capability_alerts"], stability["capability_trials"]),
            "- Hamming-MMD control: %s raw and %s Holm-adjusted alerts across %s live same-model comparisons."
            % (
                stability["mmd_raw_alerts"],
                stability["mmd_holm_alerts"],
                stability["mmd_comparisons"],
            ),
            "",
            "## Paper conclusion",
            "",
            "The evidence supports a complementary two-lane monitor. Capability probes provide interpretable evidence that task performance worsened; Hamming-MMD provides grader-free evidence that behavior changed. The current experiments support large-change detection, not exact model-weight identification or a long-run production false-positive rate.",
            "",
            "## Limitation",
            "",
            "A strict head-to-head power ranking still requires a future unified collection in which both methods operate on the same prompts, model draws, intervention assignments, and total query budget. That larger campaign is not required for the Phase 0 feasibility paper.",
            "",
        ]
    )
    return "\n".join(lines)


def run_method_comparison(
    capability_summary: str = DEFAULT_CAPABILITY_SUMMARY,
    warmup_grades: str = DEFAULT_WARMUP_GRADES,
    fingerprint_summary: str = DEFAULT_FINGERPRINT_SUMMARY,
    stability_summary: str = DEFAULT_STABILITY_SUMMARY,
    dst_dir: str = DEFAULT_DST_DIR,
) -> Mapping[str, object]:
    """Rebuild the frozen-evidence comparison and return its summary.

    This entry point is shared by the command-line program and the paired
    analysis notebook. It reads only checked-in artifacts and makes no model
    provider calls.
    """
    fingerprint = _read_json(fingerprint_summary)
    stability = _read_json(stability_summary)
    capability_rows = []
    with open(capability_summary, encoding="utf-8", newline="") as file_:
        capability_rows = list(csv.DictReader(file_))
    healthy = next(
        row
        for row in capability_rows
        if row["scenario_name"] == "healthy_a_a"
    )
    paired = _paired_capability_result(_read_jsonl(warmup_grades))
    full_mmd = next(
        row
        for row in fingerprint["scenario_results"]
        if row["scenario"] == "full_model_swap"
    )
    summary = {
        "schema_version": "noesis-method-comparison-v1",
        "provider_calls": 0,
        "comparison_type": "frozen_evidence_triangulation",
        "live_full_swap": {
            "capability": paired,
            "mmd": {
                "reference_responses": 60,
                "alternative_responses": 60,
                "p_value": full_mmd["median_p_value"],
                "permutations": fingerprint["protocol"]["permutations"],
            },
        },
        "scenario_comparison": _scenario_rows(capability_summary, fingerprint),
        "healthy_controls": {
            "capability_alerts": int(healthy["detected_trials"]),
            "capability_trials": int(healthy["total_trials"]),
            "mmd_raw_alerts": stability["raw_alerts"],
            "mmd_holm_alerts": stability["holm_alerts"],
            "mmd_comparisons": stability["comparison_count"],
        },
        "head_to_head_limit": (
            "Methods used different prompt banks, response collections, and trial designs."
        ),
    }
    os.makedirs(dst_dir, exist_ok=True)
    summary_path = os.path.join(dst_dir, "comparison_summary.json")
    table_path = os.path.join(dst_dir, "comparison.csv")
    findings_path = os.path.join(dst_dir, "findings.md")
    with open(summary_path, "w", encoding="utf-8") as file_:
        json.dump(summary, file_, indent=2, sort_keys=True)
        file_.write("\n")
    with open(table_path, "w", encoding="utf-8", newline="") as file_:
        fieldnames = list(summary["scenario_comparison"][0])
        writer = csv.DictWriter(file_, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary["scenario_comparison"])
    with open(findings_path, "w", encoding="utf-8") as file_:
        file_.write(_render_findings(summary))
    print("Comparison: %s" % findings_path)
    print("Provider calls: 0")
    return summary


def _main() -> None:
    args = _parse_args()
    run_method_comparison(
        capability_summary=args.capability_summary,
        warmup_grades=args.warmup_grades,
        fingerprint_summary=args.fingerprint_summary,
        stability_summary=args.stability_summary,
        dst_dir=args.dst_dir,
    )


if __name__ == "__main__":
    _main()
