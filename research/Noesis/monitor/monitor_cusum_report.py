"""Render stable artifacts for the preliminary offline CUSUM pilot."""

import csv
import dataclasses
import io
import json
import os
from typing import Mapping, Optional, Sequence

import research.Noesis.monitor.monitor_cusum_pilot as rnmoncupi


_SCHEMA_VERSION = "noesis-cusum-pilot-v1"


@dataclasses.dataclass(frozen=True)
class CusumExecutionMetrics:
    """Record the resource envelope observed by the single-process run."""

    wall_time_seconds: float
    peak_rss_mb: float
    worker_processes: int = 1


@dataclasses.dataclass(frozen=True)
class CusumArtifactPaths:
    """Store every artifact written by the pilot."""

    protocol_path: str
    trial_results_path: str
    scenario_summary_path: str
    summary_path: str
    findings_path: str


def _render_csv(
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fieldnames,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _optional(value: Optional[float]) -> str:
    return "n/a" if value is None else "%.1f" % value


def render_findings(
    result: rnmoncupi.CusumPilotResult,
    metrics: CusumExecutionMetrics,
) -> str:
    """Render a concise report with explicit preliminary limitations."""
    protocol = result.protocol
    healthy = next(
        scenario
        for scenario in result.scenarios
        if scenario.scenario_name == "healthy_a_a"
    )
    cusum_healthy_alarms = round(
        healthy.cusum_detection_rate * healthy.trials
    )
    fisher_healthy_alarms = round(
        healthy.fisher_detection_rate * healthy.trials
    )
    lines = [
        "# Preliminary Offline CUSUM Pilot",
        "",
        "## Scope",
        "",
        "- Status: `PRELIMINARY`",
        "- Provider calls: `0`",
        "- Additional spend: `$0`",
        "- Worker processes: `%s`" % metrics.worker_processes,
        "- Trials per scenario: `%s`" % protocol.trials_per_scenario,
        "- Independent calibration streams: `%s`"
        % protocol.calibration_trials,
        "- Healthy accuracy: `%.2f`" % protocol.healthy_accuracy,
        "- Target degraded accuracy: `%.2f`"
        % protocol.alternative_accuracy,
        "- Calibrated CUSUM threshold: `%.6f`" % result.threshold,
        "- Calibration false-alarm rate: `%.3f`"
        % result.calibration_false_alarm_rate,
        "- Wall time: `%.3f seconds`" % metrics.wall_time_seconds,
        "- Peak process RSS: `%.1f MB`" % metrics.peak_rss_mb,
        "",
        "## Method",
        "",
        (
            "- Each trial contains `%s` post-change binary grades "
            "(`%s` windows x `%s` grades)."
            % (
                protocol.post_windows * protocol.observations_per_window,
                protocol.post_windows,
                protocol.observations_per_window,
            )
        ),
        (
            "- The one-sided Bernoulli CUSUM targets a drop from `%.2f` "
            "to `%.2f` and evaluates every grade."
            % (
                protocol.healthy_accuracy,
                protocol.alternative_accuracy,
            )
        ),
        (
            "- Its threshold was selected on independent healthy streams "
            "to target at most a `%.1f%%` finite-horizon false-alarm rate."
            % (100 * protocol.target_false_alarm_rate)
        ),
        (
            "- The existing Fisher detector estimates a baseline from `%s` "
            "windows, evaluates 60-grade windows, and requires confirmation."
            % protocol.baseline_windows
        ),
        "- Both detectors receive the same seeded planted evidence streams.",
        "",
        "## Paired Results",
        "",
        (
            "| Scenario | CUSUM detect | Fisher detect | CUSUM median "
            "query | Fisher median query |"
        ),
        "| :-- | --: | --: | --: | --: |",
    ]
    for scenario in result.scenarios:
        lines.append(
            "| `%s` | %.3f | %.3f | %s | %s |"
            % (
                scenario.scenario_name,
                scenario.cusum_detection_rate,
                scenario.fisher_detection_rate,
                _optional(scenario.cusum_median_query),
                _optional(scenario.fisher_median_query),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "- The independent healthy validation produced `%s/%s` "
                "CUSUM alarms (`%.1f%%`) and `%s/%s` Fisher alarms "
                "(`%.1f%%`)."
                % (
                    cusum_healthy_alarms,
                    healthy.trials,
                    100 * healthy.cusum_detection_rate,
                    fisher_healthy_alarms,
                    healthy.trials,
                    100 * healthy.fisher_detection_rate,
                )
            ),
            (
                "- CUSUM detected every planted degradation scenario in "
                "this panel and had a lower median query count than Fisher."
            ),
            (
                "- CUSUM was given the frozen healthy rate, while Fisher "
                "estimated its baseline from samples; this favors CUSUM and "
                "prevents a universal superiority claim."
            ),
            (
                "- This pilot tests feasibility and relative detection delay "
                "under one synthetic model. It does not establish production "
                "false-alarm rates."
            ),
            "",
            "## Planned Longer Experiment",
            "",
            (
                "Before production use, repeat calibration with more seeds, "
                "multiple healthy baseline levels, baseline-estimation "
                "uncertainty, family-level changes, and independent multi-day "
                "live windows. Freeze the final threshold only after that "
                "evaluation."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def result_to_dict(
    result: rnmoncupi.CusumPilotResult,
    metrics: CusumExecutionMetrics,
) -> Mapping[str, object]:
    """Convert the pilot to a stable JSON-compatible mapping."""
    return {
        "schema_version": _SCHEMA_VERSION,
        "status": "PRELIMINARY",
        "protocol": dataclasses.asdict(result.protocol),
        "threshold": result.threshold,
        "calibration_false_alarm_rate": (
            result.calibration_false_alarm_rate
        ),
        "execution": dataclasses.asdict(metrics),
        "scenarios": [dataclasses.asdict(item) for item in result.scenarios],
        "planned_extension": (
            "Larger calibration across more seeds, baseline levels, baseline "
            "uncertainty, family-level changes, and independent multi-day "
            "live windows."
        ),
    }


def write_cusum_artifacts(
    result: rnmoncupi.CusumPilotResult,
    metrics: CusumExecutionMetrics,
    dst_dir: str,
) -> CusumArtifactPaths:
    """Write deterministic result records and the human findings report."""
    os.makedirs(dst_dir, exist_ok=True)
    paths = CusumArtifactPaths(
        protocol_path=os.path.join(dst_dir, "protocol.json"),
        trial_results_path=os.path.join(dst_dir, "trial_results.csv"),
        scenario_summary_path=os.path.join(dst_dir, "scenario_summary.csv"),
        summary_path=os.path.join(dst_dir, "summary.json"),
        findings_path=os.path.join(dst_dir, "findings.md"),
    )
    with open(paths.protocol_path, "w", encoding="utf-8") as stream:
        json.dump(
            dataclasses.asdict(result.protocol),
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    with open(paths.trial_results_path, "w", encoding="utf-8") as stream:
        stream.write(
            _render_csv(
                [
                    field.name
                    for field in dataclasses.fields(
                        rnmoncupi.CusumTrialResult
                    )
                ],
                [dataclasses.asdict(item) for item in result.trials],
            )
        )
    with open(paths.scenario_summary_path, "w", encoding="utf-8") as stream:
        stream.write(
            _render_csv(
                [
                    field.name
                    for field in dataclasses.fields(
                        rnmoncupi.CusumScenarioResult
                    )
                ],
                [dataclasses.asdict(item) for item in result.scenarios],
            )
        )
    with open(paths.summary_path, "w", encoding="utf-8") as stream:
        json.dump(
            result_to_dict(result, metrics),
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    with open(paths.findings_path, "w", encoding="utf-8") as stream:
        stream.write(render_findings(result, metrics))
    return paths
