"""
Render a human-readable report from persisted calibration outputs.

Import as:

import research.Noesis.checker.checker_calibration_report as rnochcare
"""

from typing import Mapping, Sequence

import research.Noesis.checker.checker_models as rnochmod


def _format_accuracy(value: object) -> str:
    if value is None:
        return "n/a"
    return "%.1f%%" % (100 * float(value))


def _format_latency(value: object) -> str:
    if value is None:
        return "n/a"
    return "%.1f" % float(value)


def render_calibration_report(
    run_id: int,
    bank_version: str,
    policy: rnochmod.CalibrationPolicy,
    summary: Mapping[str, object],
    minimum_survivors: int,
    activated: bool,
) -> str:
    """
    Render reproducible calibration findings as Markdown.
    """
    candidate_count = int(summary["candidate_count"])
    accepted_count = int(summary["accepted_count"])
    rejected_count = int(summary["rejected_count"])
    total_calls = int(summary["total_calls"])
    total_cost = float(summary["total_cost_usd"])
    target_stats = summary["target_stats"]
    if not isinstance(target_stats, Sequence):
        raise ValueError("target_stats must be a sequence")
    gate = "PASS" if accepted_count >= minimum_survivors else "FAIL"
    lines = [
        "# Canary calibration findings",
        "",
        "- Run id: `%s`" % run_id,
        "- Bank version: `%s`" % bank_version,
        "- Repetitions per target: %s" % policy.repetitions,
        "- Healthy acceptance threshold: %.1f%%"
        % (100 * policy.healthy_required_accuracy),
        "- Weak-model maximum accuracy: %.1f%%"
        % (100 * policy.weak_maximum_accuracy),
        "- Candidate canaries: %s" % candidate_count,
        "- Accepted survivors: %s" % accepted_count,
        "- Rejected candidates: %s" % rejected_count,
        "- Minimum-survivor gate: **%s** (%s required)"
        % (gate, minimum_survivors),
        "- Bank activated: **%s**" % ("yes" if activated else "no"),
        "- Provider calls: %s" % total_calls,
        "- Recorded cost: $%.6f" % total_cost,
        "",
        "## Target measurements",
        "",
        "| Role | Provider | Model | OK/calls | All acc. | Survivor acc."
        " | p95 ms | Cost $ |",
        "| :--- | :------- | :---- | -------: | -------: | ------------:"
        " | -----: | -----: |",
    ]
    for raw_stat in target_stats:
        if not isinstance(raw_stat, Mapping):
            raise ValueError("each target stat must be a mapping")
        lines.append(
            "| %s | `%s` | `%s` | %s/%s | %s | %s | %s | %.6f |"
            % (
                raw_stat["role"],
                raw_stat["provider_slug"],
                raw_stat["model_id"],
                raw_stat["ok_calls"],
                raw_stat["calls"],
                _format_accuracy(raw_stat["accuracy"]),
                _format_accuracy(raw_stat["survivor_accuracy"]),
                _format_latency(raw_stat["latency_p95_ms"]),
                float(raw_stat["cost_usd"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Only canaries answered correctly on every healthy-target attempt",
            "and at or below the weak-model threshold are eligible for",
            "activation. Provider failures make a candidate incomplete rather",
            "than incorrectly penalizing a model. The p95 measurements above",
            "latency-limit selection; they are not probe-window verdicts.",
            "",
        ]
    )
    return "\n".join(lines)
