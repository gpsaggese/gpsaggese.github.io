import json
import logging
import os

import helpers.hunit_test as hunitest

import findings_lib as finlib
import vulnhuntr_lib as vullib

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_parse_vulnhuntr_log
# #############################################################################


class Test_parse_vulnhuntr_log(hunitest.TestCase):
    """
    Test extracting reports from a Vulnhuntr log.
    """

    def test1(self) -> None:
        """
        Test a real log from the SEED-001 (SQL injection) smoke run.

        The input is the unedited `vulnhuntr.log` (renamed to `.jsonl`, since
        `*.log` is git-ignored) from running Vulnhuntr
        ead88c5 with `openai/gpt-oss-120b` on Groq on the anonymized SEED-001:
        one initial report and one SQLI report, both with confidence 9.
        """
        # Prepare inputs.
        path = os.path.join(self.get_input_dir(), "vulnhuntr_log.jsonl")
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        # Run test.
        parsed = vullib.parse_vulnhuntr_log(text)
        # Check outputs.
        self.assertEqual(parsed["n_requests"], 2)
        self.assertEqual(parsed["initial"]["vulnerability_types"], ["SQLI"])
        self.assertEqual(list(parsed["final"]), ["SQLI"])
        self.assertEqual(parsed["final"]["SQLI"]["confidence_score"], 9)

    def test2(self) -> None:
        """
        Test a log where the initial analysis finds nothing.
        """
        # Prepare inputs.
        report = {"confidence_score": 0, "vulnerability_types": []}
        text = "\n".join(
            [
                json.dumps({"event": "No README summary found"}),
                json.dumps(
                    {"event": "Initial analysis complete", "report": report}
                ),
            ]
        )
        # Run test.
        parsed = vullib.parse_vulnhuntr_log(text)
        # Check outputs.
        self.assertEqual(parsed["n_requests"], 1)
        self.assertEqual(parsed["final"], {})


# #############################################################################
# Test_reports_to_findings
# #############################################################################


class Test_reports_to_findings(hunitest.TestCase):
    """
    Test turning final reports into normalized findings.
    """

    def test1(self) -> None:
        """
        Test the threshold, CWE mapping, and the missing line.
        """
        # Prepare inputs.
        parsed = {
            "final": {
                "SQLI": {"confidence_score": 9, "analysis": "A.", "poc": "P"},
                "XSS": {"confidence_score": 6, "analysis": "B.", "poc": ""},
            }
        }
        # Run test.
        findings = vullib.reports_to_findings(
            parsed, case_file="case_aaaa0000.py", agent="vulnhuntr",
            min_confidence=7,
        )
        # Check outputs.
        self.assertEqual(len(findings), 1)
        finding = findings[0]
        self.assertEqual(finding["rule_id"], "SQLI")
        self.assertEqual(finding["cwe"], 89)
        self.assertEqual(finding["line"], finlib.NO_LINE)
        self.assertEqual(finding["confidence"], "0.9")
        self.assertEqual(finding["explanation"], "A. PoC: P")
        self.assertEqual(finding["fix"], "")


# #############################################################################
# Test_classify_failure
# #############################################################################


class Test_classify_failure(hunitest.TestCase):
    """
    Test the status of a run from its exit code and stderr.
    """

    def test1(self) -> None:
        """
        Test success, a schema failure, a rate limit, and an unknown crash.
        """
        # Prepare inputs.
        cases = [
            (0, ""),
            (1, "vulnhuntr.LLMs.LLMError: Validation failed"),
            (1, "vulnhuntr.LLMs.RateLimitError: Request was rate-limited"),
            (1, "Segmentation fault"),
        ]
        # Run test.
        actual = [vullib.classify_failure(code, err) for code, err in cases]
        # Check outputs.
        expected = ["ok", "parse_error", "api_error", "failed"]
        self.assertEqual(actual, expected)


# #############################################################################
# Test_summarize_calls
# #############################################################################


class Test_summarize_calls(hunitest.TestCase):
    """
    Test summarizing the API call log written by the container hook.
    """

    def test1(self) -> None:
        """
        Test a retried per-minute limit, a success, and a daily limit.
        """
        # Prepare inputs.
        records = [
            {"status": 429, "daily_limit": False, "too_large": False},
            {
                "status": 200,
                "tokens_in": 2500,
                "tokens_out": 700,
                "x-ratelimit-remaining-requests": "997",
            },
            {"status": 429, "daily_limit": True, "too_large": False},
        ]
        text = "\n".join(json.dumps(record) for record in records)
        # Run test.
        summary = vullib.summarize_calls(text)
        # Check outputs.
        self.assertEqual(summary["n_ok"], 1)
        self.assertEqual((summary["tokens_in"], summary["tokens_out"]), (2500, 700))
        self.assertTrue(summary["daily_limit"])
        self.assertFalse(summary["too_large"])
        self.assertEqual(
            summary["rate_limits"], {"x-ratelimit-remaining-requests": "997"}
        )
