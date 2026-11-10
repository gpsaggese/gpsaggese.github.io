"""
Render reproducible artifacts for the offline capability-drift experiment.

Import as:

import research.Noesis.monitor.monitor_capability_report as rnmoncare
"""

import csv
import dataclasses
import html
import io
import json
import os
from typing import Dict, Mapping, Optional, Sequence

import research.Noesis.monitor.monitor_capability_experiment as rnmoncaex

_SCHEMA_VERSION = "noesis-capability-drift-v1"


# #############################################################################
# Artifact models
# #############################################################################


@dataclasses.dataclass(frozen=True)
class CapabilityArtifactPaths:
    """
    Store paths written for one capability experiment.
    """

    protocol_path: str
    trial_results_path: str
    scenario_summary_path: str
    power_analysis_path: str
    summary_path: str
    findings_path: str
    detection_power_plot_path: str
    false_alarm_plot_path: str
    power_query_plot_path: str


# #############################################################################
# Stable records
# #############################################################################


def _scenario_inputs(
    scenario: rnmoncaex.CapabilityScenario,
) -> Dict[str, object]:
    record = dataclasses.asdict(scenario)
    record["expectation"] = scenario.expectation.value
    return record


def _scenario_record(
    result: rnmoncaex.CapabilityScenarioResult,
) -> Dict[str, object]:
    return {
        "name": result.scenario.name,
        "inputs": _scenario_inputs(result.scenario),
        "total_trials": result.total_trials,
        "detected_trials": result.detected_trials,
        "detection_rate": result.detection_rate,
        "false_alarm_rate": result.false_alarm_rate,
        "insufficient_rate": result.insufficient_rate,
        "median_detection_window": result.median_detection_window,
        "p90_detection_window": result.p90_detection_window,
        "accepted": result.accepted,
    }


def experiment_to_dict(
    result: rnmoncaex.CapabilityExperimentResult,
    power_points: Sequence[rnmoncaex.CapabilityPowerPoint],
) -> Dict[str, object]:
    """
    Convert complete results into a stable JSON-compatible record.
    """
    return {
        "schema_version": _SCHEMA_VERSION,
        "passed": result.passed,
        "protocol": dataclasses.asdict(result.protocol),
        "scenarios": [
            _scenario_record(scenario) for scenario in result.scenarios
        ],
        "power_analysis": [dataclasses.asdict(point) for point in power_points],
    }


def render_capability_json(
    result: rnmoncaex.CapabilityExperimentResult,
    power_points: Sequence[rnmoncaex.CapabilityPowerPoint],
) -> str:
    """
    Render complete results as deterministic JSON.
    """
    return (
        json.dumps(
            experiment_to_dict(result, power_points),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def render_protocol_json(
    protocol: rnmoncaex.CapabilityExperimentProtocol,
) -> str:
    """
    Render the frozen protocol independently from measured results.
    """
    return (
        json.dumps(dataclasses.asdict(protocol), indent=2, sort_keys=True) + "\n"
    )


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


def render_trial_results_csv(
    result: rnmoncaex.CapabilityExperimentResult,
) -> str:
    """
    Render one flat record for every seeded trial.
    """
    rows = [dataclasses.asdict(trial) for trial in result.trials]
    return _render_csv(
        [
            "scenario_name",
            "seed",
            "detected",
            "detection_window",
            "queries_to_detection",
            "final_state",
            "false_alarm",
            "baseline_accuracy",
        ],
        rows,
    )


def render_scenario_summary_csv(
    result: rnmoncaex.CapabilityExperimentResult,
) -> str:
    """
    Render one flat aggregate record for every scenario.
    """
    rows = []
    for scenario in result.scenarios:
        rows.append(
            {
                "scenario_name": scenario.scenario.name,
                "expectation": scenario.scenario.expectation.value,
                "total_trials": scenario.total_trials,
                "detected_trials": scenario.detected_trials,
                "detection_rate": scenario.detection_rate,
                "false_alarm_rate": scenario.false_alarm_rate,
                "insufficient_rate": scenario.insufficient_rate,
                "median_detection_window": (scenario.median_detection_window),
                "p90_detection_window": scenario.p90_detection_window,
                "accepted": scenario.accepted,
            }
        )
    return _render_csv(
        [
            "scenario_name",
            "expectation",
            "total_trials",
            "detected_trials",
            "detection_rate",
            "false_alarm_rate",
            "insufficient_rate",
            "median_detection_window",
            "p90_detection_window",
            "accepted",
        ],
        rows,
    )


def render_power_analysis_csv(
    points: Sequence[rnmoncaex.CapabilityPowerPoint],
) -> str:
    """
    Render sample-size sensitivity records.
    """
    return _render_csv(
        [
            "scenario_name",
            "observations_per_window",
            "baseline_windows",
            "trials",
            "detection_rate",
            "false_alarm_rate",
            "median_detection_window",
        ],
        [dataclasses.asdict(point) for point in points],
    )


# #############################################################################
# Findings and plots
# #############################################################################


def _format_optional(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return "%.3f" % value


def render_capability_markdown(
    result: rnmoncaex.CapabilityExperimentResult,
    power_points: Sequence[rnmoncaex.CapabilityPowerPoint],
) -> str:
    """
    Render a concise, reviewable experiment finding.
    """
    protocol = result.protocol
    lines = [
        "# Offline Capability-drift Findings",
        "",
        "## Summary",
        "",
        "- Overall acceptance: `%s`" % ("PASS" if result.passed else "FAIL"),
        "- Provider calls: `0`",
        "- Database writes: `0`",
        "- Trials per scenario: `%s`" % protocol.trials_per_scenario,
        "- Baseline windows: `%s`" % protocol.baseline_windows,
        "- Post-change windows: `%s`" % protocol.post_windows,
        "- Observations per window: `%s`" % protocol.observations_per_window,
        "- Statistical threshold: `p < %.3f`" % protocol.alpha,
        "- Practical threshold: `%.1f` percentage points"
        % (100 * protocol.minimum_quality_drop),
        "",
        "## Scenario Results",
        "",
        "| Scenario | Expected | Detect | False alarm | P90 | Gate |",
        "| :------- | :------- | :----- | :---------- | :-- | :--- |",
    ]
    for scenario in result.scenarios:
        accepted = (
            "MEASURE"
            if scenario.accepted is None
            else "PASS"
            if scenario.accepted
            else "FAIL"
        )
        lines.append(
            "| `%s` | `%s` | %.3f | %.3f | %s | `%s` |"
            % (
                scenario.scenario.name,
                scenario.scenario.expectation.value,
                scenario.detection_rate,
                scenario.false_alarm_rate,
                _format_optional(scenario.p90_detection_window),
                accepted,
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A failed gate is a research result, not a runtime failure",
            "- The detector sees observations and a frozen baseline only",
            (
                "- Scenario labels and intervention timing remain "
                "evaluator-only data"
            ),
            "- Fingerprint and market decisions remain outside this experiment",
            "",
            "## Power Analysis",
            "",
            "- Sensitivity points evaluated: `%s`" % len(power_points),
            "- See `power_analysis.csv` for the sample-size and baseline sweep",
            (
                "- See the SVG plots for detection, false-alarm, and query "
                "trade-offs"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _render_bar_svg(
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    y_label: str,
) -> str:
    width = 1000
    height = 560
    left = 90
    top = 60
    plot_width = width - left - 30
    plot_height = height - top - 190
    bar_gap = 8
    bar_width = (plot_width - bar_gap * (len(values) - 1)) / len(values)
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%s" height="%s">'
        % (width, height),
        '<rect width="100%%" height="100%%" fill="white"/>',
        '<text x="500" y="32" text-anchor="middle" '
        'font-family="sans-serif" font-size="22">%s</text>' % html.escape(title),
        '<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="black"/>'
        % (left, top, left, top + plot_height),
        '<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="black"/>'
        % (left, top + plot_height, left + plot_width, top + plot_height),
        '<text x="20" y="%s" transform="rotate(-90 20 %s)" '
        'font-family="sans-serif" font-size="14">%s</text>'
        % (
            top + plot_height / 2,
            top + plot_height / 2,
            html.escape(y_label),
        ),
    ]
    for tick in range(6):
        value = tick / 5
        y_value = top + plot_height * (1 - value)
        elements.append(
            '<text x="%s" y="%s" text-anchor="end" '
            'font-family="sans-serif" font-size="12">%.1f</text>'
            % (left - 8, y_value + 4, value)
        )
    for index, (label, value) in enumerate(zip(labels, values)):
        x_value = left + index * (bar_width + bar_gap)
        bar_height = max(0.0, min(1.0, value)) * plot_height
        y_value = top + plot_height - bar_height
        elements.append(
            '<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" '
            'fill="#4c78a8"/>' % (x_value, y_value, bar_width, bar_height)
        )
        elements.append(
            '<text x="%.2f" y="%s" transform="rotate(45 %.2f %s)" '
            'font-family="sans-serif" font-size="11">%s</text>'
            % (
                x_value + bar_width / 2,
                top + plot_height + 16,
                x_value + bar_width / 2,
                top + plot_height + 16,
                html.escape(label),
            )
        )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def _render_power_query_svg(
    points: Sequence[rnmoncaex.CapabilityPowerPoint],
) -> str:
    selected = [
        point
        for point in points
        if point.scenario_name == "threshold_level_drop"
        and point.baseline_windows == 5
    ]
    selected.sort(key=lambda point: point.observations_per_window)
    labels = [str(point.observations_per_window) for point in selected]
    values = [point.detection_rate for point in selected]
    return _render_bar_svg(
        labels,
        values,
        "10-point-drop Power by Window Size",
        "Detection rate",
    )


# #############################################################################
# Artifact writing
# #############################################################################


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as file_:
        file_.write(text)


def write_capability_artifacts(
    result: rnmoncaex.CapabilityExperimentResult,
    power_points: Sequence[rnmoncaex.CapabilityPowerPoint],
    output_dir: str,
) -> CapabilityArtifactPaths:
    """
    Write deterministic machine-readable, review, and plot artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = CapabilityArtifactPaths(
        protocol_path=os.path.join(output_dir, "protocol.json"),
        trial_results_path=os.path.join(output_dir, "trial_results.csv"),
        scenario_summary_path=os.path.join(output_dir, "scenario_summary.csv"),
        power_analysis_path=os.path.join(output_dir, "power_analysis.csv"),
        summary_path=os.path.join(output_dir, "summary.json"),
        findings_path=os.path.join(output_dir, "findings.md"),
        detection_power_plot_path=os.path.join(
            output_dir, "detection_power.svg"
        ),
        false_alarm_plot_path=os.path.join(output_dir, "false_alarm_rate.svg"),
        power_query_plot_path=os.path.join(
            output_dir, "power_vs_query_count.svg"
        ),
    )
    _write_text(paths.protocol_path, render_protocol_json(result.protocol))
    _write_text(
        paths.trial_results_path,
        render_trial_results_csv(result),
    )
    _write_text(
        paths.scenario_summary_path,
        render_scenario_summary_csv(result),
    )
    _write_text(
        paths.power_analysis_path,
        render_power_analysis_csv(power_points),
    )
    _write_text(
        paths.summary_path,
        render_capability_json(result, power_points),
    )
    _write_text(
        paths.findings_path,
        render_capability_markdown(result, power_points),
    )
    labels = [scenario.scenario.name for scenario in result.scenarios]
    _write_text(
        paths.detection_power_plot_path,
        _render_bar_svg(
            labels,
            [scenario.detection_rate for scenario in result.scenarios],
            "Capability-drift Detection Rate",
            "Detection rate",
        ),
    )
    _write_text(
        paths.false_alarm_plot_path,
        _render_bar_svg(
            labels,
            [scenario.false_alarm_rate for scenario in result.scenarios],
            "Capability-drift False-alarm Rate",
            "False-alarm rate",
        ),
    )
    _write_text(
        paths.power_query_plot_path,
        _render_power_query_svg(power_points),
    )
    return paths
