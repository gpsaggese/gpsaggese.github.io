"""Normalize Cisco's Semgrep-first local quick-scan reports."""

import json
import os
import re

import findings_lib as finlib


def extract_remediation(text):
    """Extract a remediation section without inventing a suggested fix."""
    match = re.search(
        r"(?im)^\s*(?:#{1,6}\s*)?(?:7[.)]\s*)?\*{0,2}"
        r"Remediation\*{0,2}\s*:?[ \t]*",
        text,
    )
    if not match:
        return ""
    section = text[match.end():]
    end = re.search(
        r"(?im)^\s*(?:8[.)]\s|#{1,6}\s|\*{0,2}Defen[cs]e in Depth)",
        section,
    )
    return section[:end.start()].strip() if end else section.strip()


def read_report(path):
    """Retain every upstream finding; quick mode enriches rather than filters."""
    with open(path, encoding="utf-8") as stream:
        report = json.load(stream)
    if not report["report_metadata"]["llm_enabled"]:
        raise ValueError("Expected an AI-enabled quick scan")
    findings = []
    for item in report["findings"]:
        analysis = item["ai_analysis"]
        if analysis["status"] not in ("success", "skipped", "skipped_rule"):
            raise ValueError("Quick scan contains failed LLM analyses")
        text = analysis.get("message", "")
        cwes = item.get("metadata", {}).get("cwe", [])
        if not isinstance(cwes, list):
            cwes = [cwes]
        cwes = [f"CWE-{v}" if str(v).isdigit() else str(v) for v in cwes]
        findings.append(finlib.make_finding(
            case_file=os.path.basename(item["file"]),
            agent="ai_deep_sast_quick",
            line=int(item["start_line"]),
            rule_id=item["rule_id"],
            cwe=finlib.extract_cwe(cwes),
            severity={"ERROR": "high", "WARNING": "medium", "INFO": "low"}[
                item["severity"]
            ],
            confidence=str(item.get("metadata", {}).get("confidence", "")),
            explanation=item["semgrep_message"] + "\n\n" + text,
            fix=extract_remediation(text) if analysis["status"] == "success"
            else "",
        ))
    return findings


def parse_timings(stderr):
    """Read llama.cpp's evaluated prompt tokens and generation runs."""
    prompt = re.search(r"prompt eval time\s*=.*?/\s*(\d+) tokens", stderr)
    output = re.search(r"(?m)^.*?:\s+eval time\s*=.*?/\s*(\d+) runs", stderr)
    return {
        "tokens_in": int(prompt.group(1)) if prompt else None,
        "tokens_out": int(output.group(1)) if output else None,
    }
