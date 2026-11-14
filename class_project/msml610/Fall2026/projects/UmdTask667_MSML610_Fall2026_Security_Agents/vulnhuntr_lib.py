"""
Parse the output of Vulnhuntr (Protect AI) into normalized findings.

Vulnhuntr writes a JSON log (`vulnhuntr.log`) with one report per model answer.
Each report has an analysis, a PoC, a 0-10 confidence score, and a list of
vulnerability types, but no line number, so findings use
`findings_lib.NO_LINE`.

Import as:

import vulnhuntr_lib as vullib
"""

import json
import logging
from typing import Any, Dict, List, Optional

import helpers.hdbg as hdbg

import findings_lib as finlib

_LOG = logging.getLogger(__name__)

# #############################################################################
# Constants
# #############################################################################

# Vulnhuntr's vulnerability types and the CWE we record for each.
# NOTE: RCE covers command injection (78), code injection (94 / 95), and unsafe
# deserialization (502); 94 is the closest single CWE.
VULN_TYPE_TO_CWE = {
    "LFI": 22,
    "AFO": 73,
    "SQLI": 89,
    "XSS": 79,
    "SSRF": 918,
    "IDOR": 639,
    "RCE": 94,
}

# Status of one snippet, stored in `status.json` next to its log.
STATUS_OK = "ok"
STATUS_PARSE_ERROR = "parse_error"
STATUS_API_ERROR = "api_error"
STATUS_TIMEOUT = "timeout"
STATUS_FAILED = "failed"
# The provider refused a call because a daily limit is used up.
STATUS_DAILY_LIMIT = "daily_limit"
# A call was larger than the per-minute token limit.
STATUS_TOO_LARGE = "too_large"

# #############################################################################
# Log parsing
# #############################################################################


def parse_vulnhuntr_log(text: str) -> Dict[str, Any]:
    """
    Extract the reports from the JSON log written by one Vulnhuntr run.

    Vulnhuntr logs one JSON object per line. The ones used here:
    - `Initial analysis complete`, with `report`
    - `Performing vuln-specific analysis`, with `vuln_type`
    - `Secondary analysis complete`, with `secondary_analysis_report`

    :param text: content of `vulnhuntr.log`
    :return: dict with:
        - `initial`: initial report, or `None` if missing
        - `final`: last secondary report for each vulnerability type
        - `n_requests`: number of model answers that were parsed
    """
    initial: Optional[Dict[str, Any]] = None
    final: Dict[str, Dict[str, Any]] = {}
    n_requests = 0
    vuln_type = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            _LOG.warning("Skipping a log line that is not JSON")
            continue
        name = event.get("event", "")
        if name == "Initial analysis complete":
            initial = event["report"]
            n_requests += 1
        elif name == "Performing vuln-specific analysis":
            vuln_type = event["vuln_type"]
        elif name == "Secondary analysis complete":
            hdbg.dassert(vuln_type, "Secondary report before any vuln type")
            final[vuln_type] = event["secondary_analysis_report"]
            n_requests += 1
    parsed = {"initial": initial, "final": final, "n_requests": n_requests}
    return parsed


def reports_to_findings(
    parsed: Dict[str, Any],
    *,
    case_file: str,
    agent: str,
    min_confidence: int,
) -> List[Dict[str, Any]]:
    """
    Turn the final report of each vulnerability type into a finding.

    :param parsed: output of `parse_vulnhuntr_log()`
    :param case_file: snippet the reports are about
    :param agent: name of the agent
    :param min_confidence: lowest confidence score (0-10) to keep
    :return: findings with `line` set to `NO_LINE`
    """
    findings = []
    for vuln_type, report in sorted(parsed["final"].items()):
        score = int(report.get("confidence_score", 0))
        if score < min_confidence:
            continue
        explanation = str(report.get("analysis", "")).strip()
        poc = str(report.get("poc", "")).strip()
        if poc:
            explanation = f"{explanation} PoC: {poc}"
        finding = finlib.make_finding(
            case_file=case_file,
            agent=agent,
            rule_id=vuln_type,
            line=finlib.NO_LINE,
            cwe=VULN_TYPE_TO_CWE.get(vuln_type),
            severity="",
            confidence=f"{score / 10:.1f}",
            explanation=explanation,
            fix="",
        )
        findings.append(finding)
    return findings


def classify_failure(returncode: int, stderr: str) -> str:
    """
    Return the status of a Vulnhuntr run from its exit code and stderr.
    """
    if returncode == 0:
        status = STATUS_OK
    elif "Validation failed" in stderr or "ValidationError" in stderr:
        status = STATUS_PARSE_ERROR
    elif "RateLimitError" in stderr or "APIStatusError" in stderr:
        status = STATUS_API_ERROR
    elif "APIConnectionError" in stderr:
        status = STATUS_API_ERROR
    else:
        status = STATUS_FAILED
    return status


def summarize_calls(text: str) -> Dict[str, Any]:
    """
    Summarize the API call log written by `docker/vulnhuntr_hooks/`.

    :param text: content of `groq_calls.jsonl`, one JSON object per response
    :return: dict with:
        - `calls`: all records
        - `n_ok`: responses with HTTP 200
        - `tokens_in`, `tokens_out`: tokens of the successful calls
        - `daily_limit`: a call hit a daily limit
        - `too_large`: a call exceeded the per-minute token limit
        - `rate_limits`: rate limit headers of the last successful call
    """
    calls = [json.loads(line) for line in text.splitlines() if line.strip()]
    ok = [call for call in calls if call.get("status") == 200]
    last_headers = {}
    if ok:
        last_headers = {k: v for k, v in ok[-1].items() if k.startswith("x-")}
    summary = {
        "calls": calls,
        "n_ok": len(ok),
        "tokens_in": sum(int(call.get("tokens_in") or 0) for call in ok),
        "tokens_out": sum(int(call.get("tokens_out") or 0) for call in ok),
        "daily_limit": any(call.get("daily_limit") for call in calls),
        "too_large": any(call.get("too_large") for call in calls),
        "rate_limits": last_headers,
    }
    return summary
