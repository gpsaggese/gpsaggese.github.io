"""Check Cisco report normalization and incomplete-scan rejection."""

import json

import pytest

import ai_deep_sast_lib as cislib


@pytest.mark.parametrize("cwe", ["CWE-89", "89", 89])
def test_published_findings_use_index_locations(tmp_path, cwe):
    report = tmp_path / "report.json"
    index = tmp_path / "index.json"
    finding = {
        "verdict": "true-positive", "file_path": "/cases/case_a.py",
        "function_name": "handler", "cwe": cwe,
        "severity_cvss": 9.8, "description": "SQL injection",
        "evidence_report": "Input reaches query",
    }
    report.write_text(json.dumps({"findings": [
        finding, dict(finding, verdict="needs-review"),
        dict(finding, function_name="unknown"),
    ]}))
    index.write_text(json.dumps({"functions": {"handler": {
        "file_path": "/cases/case_a.py", "name": "handler",
        "qualified_name": "handler", "start_line": 12,
    }}}))
    findings = cislib.read_report(str(report), str(index))
    assert len(findings) == 2
    assert findings[0]["case_file"] == "case_a.py"
    assert findings[0]["line"] == 12
    assert findings[0]["cwe"] == 89
    assert findings[0]["severity"] == "critical"
    assert findings[1]["line"] == 0


@pytest.mark.parametrize("summary, log_text", [
    ({"success": False}, ""),
    ({"success": True, "phases": {
        "detection": {"rule_based": {"errors": 1}}
    }}, ""),
    ({"success": True, "phases": {
        "detection": {"exploratory": {"circuit_breaker": True}}
    }}, ""),
    ({"success": True}, "Failed to parse LLM response as JSON"),
    ({"success": True}, "[ERROR] Error analysing function"),
])
def test_incomplete_scans_are_not_scored(summary, log_text):
    with pytest.raises(RuntimeError):
        cislib.check_scan(summary, log_text)


def test_complete_scan_is_accepted():
    cislib.check_scan({"success": True, "phases": {
        "detection": {"rule_based": {"errors": 0}},
        "triage": {"errors": 0},
    }}, "")
