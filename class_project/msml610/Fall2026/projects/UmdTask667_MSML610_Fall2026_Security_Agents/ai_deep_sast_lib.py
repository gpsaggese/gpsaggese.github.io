"""Read Cisco's published findings without using benchmark labels."""

import json
import os
from typing import Any, Dict, List

import findings_lib as finlib


def read_report(
    report_file: str, index_file: str
) -> List[Dict[str, Any]]:
    """
    Normalize published true-positive verdicts from Cisco's JSON report.

    Cisco retains function locations, not exact vulnerable statement lines.
    Use the indexed function start; use zero when no function can be resolved.
    The runner preserves zero as an unknown original line after remapping.
    """
    with open(report_file, encoding="utf-8") as stream:
        report = json.load(stream)
    with open(index_file, encoding="utf-8") as stream:
        index = json.load(stream)
    functions = list(index["functions"].values())
    findings = []
    for item in report["findings"]:
        if item.get("verdict") != "true-positive":
            continue
        case_file = os.path.basename(item["file_path"])
        matches = [
            func for func in functions
            if os.path.basename(func["file_path"]) == case_file
            and item.get("function_name")
            in (func.get("name"), func.get("qualified_name"))
        ]
        line = int(matches[0]["start_line"]) if len(matches) == 1 else 0
        score = item.get("severity_cvss")
        cwe = str(item.get("cwe") or "")
        if cwe.isdigit():
            cwe = f"CWE-{cwe}"
        severity = (
            finlib.severity_from_score(float(score))
            if score is not None else
            {"ERROR": "high", "WARNING": "medium", "INFO": "low"}.get(
                item.get("severity_tier"), ""
            )
        )
        findings.append(finlib.make_finding(
            case_file=case_file,
            agent="ai_deep_sast",
            line=line,
            cwe=finlib.extract_cwe([cwe]),
            severity=severity,
            explanation="\n\n".join(
                str(item.get(field) or "")
                for field in ("description", "evidence_report")
            ).strip(),
        ))
    return findings


def check_scan(summary: Dict[str, Any], log_text: str) -> None:
    """Reject failed or partial scans instead of scoring them as clean code."""
    if not summary.get("success"):
        raise RuntimeError("Cisco scan failed; inspect the raw scan logs")

    def check_node(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("errors") or node.get("circuit_breaker"):
                raise RuntimeError("Cisco scan has analysis errors")
            for value in node.values():
                check_node(value)

    check_node(summary.get("phases", {}))
    if "[ERROR]" in log_text or "Failed to parse LLM response" in log_text:
        raise RuntimeError("Cisco scan logged an analysis or parsing error")
