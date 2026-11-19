"""
Render artifacts for the offline behavioral-fingerprint controls.

Import as:

import research.Noesis.monitor.monitor_fingerprint_report as rnmonfire
"""

import csv
import dataclasses
import importlib.metadata
import io
import json
import os
import platform
from typing import Dict, Mapping, Sequence

import research.Noesis.monitor.monitor_fingerprint_experiment as rnmonfiex

_SCHEMA_VERSION = "noesis-fingerprint-control-v1"


@dataclasses.dataclass(frozen=True)
class FingerprintArtifactPaths:
    """
    Store paths written for one fingerprint control experiment.
    """

    protocol_path: str
    trial_results_path: str
    scenario_summary_path: str
    summary_path: str
    findings_path: str


def _scenario_record(
    result: rnmonfiex.FingerprintScenarioResult,
) -> Dict[str, object]:
    return {
        "name": result.scenario.name,
        "substitution_fraction": result.scenario.substitution_fraction,
        "expectation": result.scenario.expectation.value,
        "total_trials": result.total_trials,
        "substituted_count": result.substituted_count,
        "detection_rate": result.detection_rate,
        "false_alarm_rate": result.false_alarm_rate,
        "median_p_value": result.median_p_value,
        "accepted": result.accepted,
    }


def experiment_to_dict(
    result: rnmonfiex.FingerprintExperimentResult,
) -> Dict[str, object]:
    """
    Convert the complete result to a stable JSON-compatible record.
    """
    return {
        "schema_version": _SCHEMA_VERSION,
        "validation_scope": "synthetic-implementation-control",
        "provider_calls": 0,
        "passed": result.passed,
        "protocol": dataclasses.asdict(result.protocol),
        "runtime": {
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "torch": importlib.metadata.version("torch"),
        },
        "scenarios": [_scenario_record(item) for item in result.scenarios],
    }


def render_fingerprint_json(
    result: rnmonfiex.FingerprintExperimentResult,
) -> str:
    """
    Render complete results as deterministic JSON.
    """
    return json.dumps(experiment_to_dict(result), indent=2, sort_keys=True) + "\n"


def _render_csv(
    fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]
) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=fieldnames, lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def render_trial_results_csv(
    result: rnmonfiex.FingerprintExperimentResult,
) -> str:
    """
    Render one record for every seeded trial.
    """
    return _render_csv(
        [
            "scenario_name",
            "seed",
            "substituted_count",
            "p_value",
            "statistic",
            "detected",
            "false_alarm",
        ],
        [dataclasses.asdict(trial) for trial in result.trials],
    )


def render_scenario_summary_csv(
    result: rnmonfiex.FingerprintExperimentResult,
) -> str:
    """
    Render one aggregate record for every scenario.
    """
    return _render_csv(
        [
            "name",
            "substitution_fraction",
            "expectation",
            "total_trials",
            "substituted_count",
            "detection_rate",
            "false_alarm_rate",
            "median_p_value",
            "accepted",
        ],
        [_scenario_record(item) for item in result.scenarios],
    )


def render_fingerprint_markdown(
    result: rnmonfiex.FingerprintExperimentResult,
) -> str:
    """
    Render a concise finding with the validation boundary made explicit.
    """
    lines = [
        "# Offline Behavioral-fingerprint Control Findings",
        "",
        "## Scope",
        "",
        "- Validation type: `synthetic implementation control`",
        "- Provider calls: `0`",
        "- Live-provider conclusions supported: `none`",
        "- Overall acceptance: `%s`" % ("PASS" if result.passed else "FAIL"),
        "- Statistic: `mmd_hamming`",
        "- P-value: permutation test with `%s` permutations"
        % result.protocol.permutations,
        "",
        "## Results",
        "",
        "| Scenario | Swap | Count | Detect | False alarm | Median p | Gate |",
        "| :------- | ---: | ----: | -----: | ----------: | -------: | :--- |",
    ]
    for item in result.scenarios:
        gate = (
            "MEASURE"
            if item.accepted is None
            else "PASS" if item.accepted else "FAIL"
        )
        lines.append(
            "| `%s` | %.0f%% | %s | %.3f | %.3f | %.4f | `%s` |"
            % (
                item.scenario.name,
                100 * item.scenario.substitution_fraction,
                item.substituted_count,
                item.detection_rate,
                item.false_alarm_rate,
                item.median_p_value,
                gate,
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A/A uses independent draws from the same synthetic generator",
            "- Partial swaps use exact seeded counts, not Bernoulli sampling",
            "- The evaluator owns swap labels; the detector sees only two pools",
            "- Passing proves the experiment wiring can separate these controls",
            "- It does not validate power on archived or live model responses",
            "",
        ]
    )
    return "\n".join(lines)


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as file_:
        file_.write(text)


def write_fingerprint_artifacts(
    result: rnmonfiex.FingerprintExperimentResult,
    output_dir: str,
) -> FingerprintArtifactPaths:
    """
    Write deterministic machine-readable and reviewer-facing artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = FingerprintArtifactPaths(
        protocol_path=os.path.join(output_dir, "protocol.json"),
        trial_results_path=os.path.join(output_dir, "trial_results.csv"),
        scenario_summary_path=os.path.join(output_dir, "scenario_summary.csv"),
        summary_path=os.path.join(output_dir, "summary.json"),
        findings_path=os.path.join(output_dir, "findings.md"),
    )
    _write_text(
        paths.protocol_path,
        json.dumps(dataclasses.asdict(result.protocol), indent=2, sort_keys=True)
        + "\n",
    )
    _write_text(paths.trial_results_path, render_trial_results_csv(result))
    _write_text(paths.scenario_summary_path, render_scenario_summary_csv(result))
    _write_text(paths.summary_path, render_fingerprint_json(result))
    _write_text(paths.findings_path, render_fingerprint_markdown(result))
    return paths
