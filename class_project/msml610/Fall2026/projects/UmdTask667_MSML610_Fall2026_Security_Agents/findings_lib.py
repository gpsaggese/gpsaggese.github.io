"""
Store the findings of every agent in one normalized format.

A run of one agent on the benchmark produces two files in
`results/normalized/<agent>/`:
- `<run_id>.findings.jsonl`: one finding per line, with `FINDING_COLUMNS`
- `<run_id>.meta.json`: tool version, model, timing, and cost of the run

Agents scan anonymized copies of the snippets (see
`benchmark_lib.anonymize_benchmark()`), so raw findings point to case files and
stripped line numbers. `attach_snippet_ids()` maps them back to snippet ids and
original line numbers.

Import as:

import findings_lib as finlib
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd

import helpers.hdbg as hdbg
import helpers.hio as hio

_LOG = logging.getLogger(__name__)

# #############################################################################
# Constants
# #############################################################################

# Columns of a normalized finding.
FINDING_COLUMNS = (
    # Id of the snippet, filled by `attach_snippet_ids()`.
    "snippet_id",
    # File the agent reported, relative to the scanned dir.
    "case_file",
    "agent",
    "model",
    "run_id",
    # Rule or check that fired, if the agent reports one.
    "rule_id",
    # 1-based line in the original snippet after `attach_snippet_ids()`, or
    # `NO_LINE` if the agent reports no location.
    "line",
    "cwe",
    # One of `SEVERITIES`, or empty if the agent reports none.
    "severity",
    # Free-form confidence reported by the agent (e.g., `high`, `0.8`).
    "confidence",
    "explanation",
    "fix",
)

SEVERITIES = ("critical", "high", "medium", "low")

# Line of a finding whose agent reports no location (e.g., Vulnhuntr). It is
# never shifted to original lines and never counts as localizing a bug.
NO_LINE = 0

# Map SARIF levels to severities.
_SARIF_LEVEL_TO_SEVERITY = {"error": "high", "warning": "medium", "note": "low"}

_CWE_RE = re.compile(r"CWE-(\d+)", re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r"^(HIGH|MEDIUM|LOW) CONFIDENCE$", re.IGNORECASE)


# #############################################################################
# Helpers
# #############################################################################


def extract_cwe(texts: List[str]) -> Optional[int]:
    """
    Return the first CWE id mentioned in a list of texts.

    E.g., `["security", "CWE-89: Improper Neutralization ..."]` -> `89`

    :param texts: texts to search, in priority order
    :return: CWE id, or `None` if no text mentions one
    """
    for text in texts:
        match = _CWE_RE.search(str(text))
        if match:
            cwe = int(match.group(1))
            return cwe
    return None


def severity_from_score(score: float) -> str:
    """
    Map a CVSS-like score in [0, 10] to a severity.

    Uses the CVSS v3 bands: 9.0+ critical, 7.0+ high, 4.0+ medium, else low.
    """
    hdbg.dassert_lte(0.0, score)
    hdbg.dassert_lte(score, 10.0)
    if score >= 9.0:
        severity = "critical"
    elif score >= 7.0:
        severity = "high"
    elif score >= 4.0:
        severity = "medium"
    else:
        severity = "low"
    return severity


def make_finding(**kwargs: Any) -> Dict[str, Any]:
    """
    Build a finding with all `FINDING_COLUMNS`, filling missing ones.

    Missing text fields become `""`, a missing `cwe` becomes `None`.

    :param kwargs: values for some of `FINDING_COLUMNS`
    :return: finding with exactly `FINDING_COLUMNS`
    """
    for key in kwargs:
        hdbg.dassert_in(key, FINDING_COLUMNS, "Unknown finding field")
    finding: Dict[str, Any] = {col: "" for col in FINDING_COLUMNS}
    finding["cwe"] = None
    finding.update(kwargs)
    hdbg.dassert_isinstance(finding["line"], int)
    if finding["severity"]:
        hdbg.dassert_in(finding["severity"], SEVERITIES)
    return finding


# #############################################################################
# SARIF
# #############################################################################


def _get_rule_index(run: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Index the rules of a SARIF run by id.
    """
    rules = run.get("tool", {}).get("driver", {}).get("rules", [])
    rule_index = {rule["id"]: rule for rule in rules}
    return rule_index


def _parse_sarif_result(
    result: Dict[str, Any], rule: Dict[str, Any], *, agent: str
) -> Optional[Dict[str, Any]]:
    """
    Convert one SARIF result into a finding.

    :return: finding, or `None` if the result has no source location
    """
    locations = result.get("locations", [])
    if not locations:
        _LOG.warning("Skipping result without location: %s", result.get("ruleId"))
        return None
    physical = locations[0].get("physicalLocation", {})
    uri = physical.get("artifactLocation", {}).get("uri", "")
    line = physical.get("region", {}).get("startLine")
    if line is None:
        return None
    rule_props = rule.get("properties", {})
    result_props = result.get("properties", {})
    tags = list(result_props.get("tags", [])) + list(rule_props.get("tags", []))
    message = result.get("message", {}).get("text", "")
    # Look for the CWE in tags first, then in the message.
    cwe = extract_cwe(tags + [str(rule_props.get("cwe", "")), message])
    # Prefer a numeric severity score, then the SARIF level.
    score = result_props.get("security-severity") or rule_props.get(
        "security-severity"
    )
    if score:
        severity = severity_from_score(float(score))
    else:
        level = result.get("level") or rule.get("defaultConfiguration", {}).get(
            "level", ""
        )
        severity = _SARIF_LEVEL_TO_SEVERITY.get(level, "")
    # Confidence is a tag (e.g., `MEDIUM CONFIDENCE`) or the rule precision.
    confidence = ""
    for tag in tags:
        match = _CONFIDENCE_RE.match(tag)
        if match:
            confidence = match.group(1).lower()
            break
    if not confidence:
        confidence = str(rule_props.get("precision", ""))
    finding = make_finding(
        case_file=os.path.basename(uri),
        agent=agent,
        rule_id=result.get("ruleId", ""),
        line=int(line),
        cwe=cwe,
        severity=severity,
        confidence=confidence,
        explanation=message,
    )
    return finding


def read_sarif(path: str, *, agent: str) -> List[Dict[str, Any]]:
    """
    Read the findings from a SARIF 2.1.0 file.

    :param path: path to the SARIF file
    :param agent: name of the agent that produced it
    :return: findings, with `snippet_id` still empty
    """
    sarif = hio.from_json(path)
    findings = []
    for run in sarif.get("runs", []):
        rule_index = _get_rule_index(run)
        for result in run.get("results", []):
            rule = rule_index.get(result.get("ruleId", ""), {})
            finding = _parse_sarif_result(result, rule, agent=agent)
            if finding is not None:
                findings.append(finding)
    _LOG.debug("Read %s findings from '%s'", len(findings), path)
    return findings


# #############################################################################
# Bandit
# #############################################################################


def read_bandit_json(path: str, *, agent: str = "bandit") -> List[Dict[str, Any]]:
    """
    Read the findings from a Bandit JSON report (`bandit -f json`).

    :param path: path to the JSON report
    :param agent: name of the agent
    :return: findings, with `snippet_id` still empty
    """
    report = hio.from_json(path)
    findings = []
    for result in report.get("results", []):
        cwe = result.get("issue_cwe", {}).get("id")
        finding = make_finding(
            case_file=os.path.basename(result["filename"]),
            agent=agent,
            rule_id=f"{result['test_id']}:{result['test_name']}",
            line=int(result["line_number"]),
            cwe=int(cwe) if cwe else None,
            severity=result["issue_severity"].lower(),
            confidence=result["issue_confidence"].lower(),
            explanation=result["issue_text"],
        )
        findings.append(finding)
    _LOG.debug("Read %s findings from '%s'", len(findings), path)
    return findings


# #############################################################################
# Mapping to snippets
# #############################################################################


def attach_snippet_ids(
    findings: List[Dict[str, Any]], case_map: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    Map findings on case files back to snippet ids and original lines.

    Findings on files that are not in `case_map` are dropped with a warning.

    :param findings: findings with `case_file` and stripped `line`
    :param case_map: table from `benchmark_lib.anonymize_benchmark()`
    :return: findings with `snippet_id` and original `line` (`NO_LINE` is
        kept as is)
    """
    index = case_map.set_index("case_file")
    mapped = []
    for finding in findings:
        case_file = finding["case_file"]
        if case_file not in index.index:
            _LOG.warning("Dropping finding on unknown file '%s'", case_file)
            continue
        row = index.loc[case_file]
        finding = dict(finding)
        finding["snippet_id"] = row["snippet_id"]
        if int(finding["line"]) != NO_LINE:
            finding["line"] = int(finding["line"]) + int(row["line_offset"])
        mapped.append(finding)
    return mapped


# #############################################################################
# Storage
# #############################################################################


def get_run_paths(results_dir: str, agent: str, run_id: str) -> Dict[str, str]:
    """
    Return the paths of the files of a run.

    :return: dict with keys `findings` and `meta`
    """
    agent_dir = os.path.join(results_dir, agent)
    paths = {
        "findings": os.path.join(agent_dir, f"{run_id}.findings.jsonl"),
        "meta": os.path.join(agent_dir, f"{run_id}.meta.json"),
    }
    return paths


def write_run(
    findings: List[Dict[str, Any]],
    meta: Dict[str, Any],
    *,
    results_dir: str,
    agent: str,
    run_id: str,
) -> Dict[str, str]:
    """
    Write the findings and metadata of a run, replacing any previous files.

    :param findings: findings with `snippet_id` filled
    :param meta: run metadata (e.g., tool version, model, seconds)
    :return: paths of the files written
    """
    paths = get_run_paths(results_dir, agent, run_id)
    lines = []
    for finding in findings:
        hdbg.dassert_eq(tuple(finding.keys()), FINDING_COLUMNS)
        hdbg.dassert_ne(finding["snippet_id"], "", "Finding has no snippet id")
        lines.append(json.dumps(finding, sort_keys=False))
    txt = "\n".join(lines) + ("\n" if lines else "")
    hio.to_file(paths["findings"], txt)
    meta = dict(meta, agent=agent, run_id=run_id, n_findings=len(findings))
    hio.to_json(paths["meta"], meta)
    _LOG.info("Wrote %s findings to '%s'", len(findings), paths["findings"])
    return paths


def read_findings(results_dir: str) -> pd.DataFrame:
    """
    Read all findings of all agents and runs.

    :param results_dir: dir with one subdir per agent
    :return: one row per finding with `FINDING_COLUMNS`
    """
    hdbg.dassert_dir_exists(results_dir)
    rows = []
    for agent in sorted(os.listdir(results_dir)):
        agent_dir = os.path.join(results_dir, agent)
        if not os.path.isdir(agent_dir):
            continue
        for file_name in sorted(os.listdir(agent_dir)):
            if not file_name.endswith(".findings.jsonl"):
                continue
            txt = hio.from_file(os.path.join(agent_dir, file_name))
            rows.extend(json.loads(line) for line in txt.split("\n") if line)
    df = pd.DataFrame(rows, columns=list(FINDING_COLUMNS))
    df["cwe"] = df["cwe"].astype("Int64")
    return df


def read_runs(results_dir: str) -> pd.DataFrame:
    """
    Read the metadata of all runs.

    :param results_dir: dir with one subdir per agent
    :return: one row per run
    """
    hdbg.dassert_dir_exists(results_dir)
    rows = []
    for agent in sorted(os.listdir(results_dir)):
        agent_dir = os.path.join(results_dir, agent)
        if not os.path.isdir(agent_dir):
            continue
        for file_name in sorted(os.listdir(agent_dir)):
            if file_name.endswith(".meta.json"):
                rows.append(hio.from_json(os.path.join(agent_dir, file_name)))
    df = pd.DataFrame(rows)
    return df
