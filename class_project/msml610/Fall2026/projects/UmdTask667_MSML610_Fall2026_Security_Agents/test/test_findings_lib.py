import logging
import os

import pandas as pd

import helpers.hunit_test as hunitest

import findings_lib as finlib

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_extract_cwe
# #############################################################################


class Test_extract_cwe(hunitest.TestCase):
    """
    Test finding a CWE id in a list of texts.
    """

    def test1(self) -> None:
        """
        Test that the first text mentioning a CWE wins.
        """
        # Prepare inputs.
        texts = ["security", "CWE-89: SQL Injection", "CWE-20"]
        # Run test.
        actual = finlib.extract_cwe(texts)
        # Check outputs.
        self.assertEqual(actual, 89)

    def test2(self) -> None:
        """
        Test that `None` is returned when no text mentions a CWE.
        """
        # Prepare inputs.
        texts = ["security", "MEDIUM CONFIDENCE"]
        # Run test.
        actual = finlib.extract_cwe(texts)
        # Check outputs.
        self.assertIsNone(actual)


# #############################################################################
# Test_severity_from_score
# #############################################################################


class Test_severity_from_score(hunitest.TestCase):
    """
    Test mapping CVSS-like scores to severities.
    """

    def test1(self) -> None:
        """
        Test the four CVSS v3 bands, including their lower bounds.
        """
        # Prepare inputs.
        scores = [9.0, 7.0, 4.0, 3.9]
        # Prepare outputs.
        expected = ["critical", "high", "medium", "low"]
        # Run test.
        actual = [finlib.severity_from_score(score) for score in scores]
        # Check outputs.
        self.assertEqual(actual, expected)


# #############################################################################
# Test_read_sarif
# #############################################################################


class Test_read_sarif(hunitest.TestCase):
    """
    Test reading findings from a SARIF file.
    """

    def test1(self) -> None:
        """
        Test a trimmed Semgrep SARIF file with two results.

        The input was produced by running Semgrep 1.182.1 with `p/python` and
        `p/owasp-top-ten` on the prototype snippets, keeping one SQL and one
        HTML result.

        Semgrep tags its Flask SQL rule with CWE-704, not CWE-89, so the
        reader must report 704.
        """
        # Prepare inputs.
        path = os.path.join(self.get_input_dir(), "semgrep.sarif")
        # Run test.
        findings = finlib.read_sarif(path, agent="semgrep")
        # Check outputs.
        df = pd.DataFrame(findings)[
            ["case_file", "line", "cwe", "severity", "confidence"]
        ]
        actual = df.to_string(index=False)
        expected = """
                 case_file  line  cwe severity confidence
        case_f90c5e6a.py    21  704     high     medium
        case_07a81681.py    23   79   medium     medium
        """
        self.assert_equal(actual, expected, dedent=True, fuzzy_match=True)


# #############################################################################
# Test_attach_snippet_ids
# #############################################################################


class Test_attach_snippet_ids(hunitest.TestCase):
    """
    Test mapping findings on case files back to snippets.
    """

    def test1(self) -> None:
        """
        Test that lines are shifted and unknown files are dropped.
        """
        # Prepare inputs.
        findings = [
            finlib.make_finding(case_file="case_aaaa0000.py", line=3),
            finlib.make_finding(case_file="case_unknown.py", line=5),
        ]
        case_map = pd.DataFrame(
            {
                "snippet_id": ["SEED-001"],
                "case_file": ["case_aaaa0000.py"],
                "line_offset": [19],
            }
        )
        # Run test.
        mapped = finlib.attach_snippet_ids(findings, case_map)
        # Check outputs.
        self.assertEqual(len(mapped), 1)
        self.assertEqual(mapped[0]["snippet_id"], "SEED-001")
        self.assertEqual(mapped[0]["line"], 22)

    def test2(self) -> None:
        """
        Test that a finding with no location keeps `NO_LINE`.
        """
        # Prepare inputs.
        findings = [
            finlib.make_finding(case_file="case_aaaa0000.py", line=finlib.NO_LINE)
        ]
        case_map = pd.DataFrame(
            {
                "snippet_id": ["SEED-001"],
                "case_file": ["case_aaaa0000.py"],
                "line_offset": [19],
            }
        )
        # Run test.
        mapped = finlib.attach_snippet_ids(findings, case_map)
        # Check outputs.
        self.assertEqual(mapped[0]["snippet_id"], "SEED-001")
        self.assertEqual(mapped[0]["line"], finlib.NO_LINE)


# #############################################################################
# Test_write_run
# #############################################################################


class Test_write_run(hunitest.TestCase):
    """
    Test writing and reading back a run.
    """

    def test1(self) -> None:
        """
        Test that findings and metadata round-trip.
        """
        # Prepare inputs.
        results_dir = self.get_scratch_space()
        findings = [
            finlib.make_finding(
                snippet_id="SEED-001", agent="bandit", line=40, cwe=89
            ),
            finlib.make_finding(snippet_id="SAFE-002", agent="bandit", line=39),
        ]
        meta = {"tool_version": "1.0"}
        # Run test.
        finlib.write_run(
            findings, meta, results_dir=results_dir, agent="bandit", run_id="r1"
        )
        df = finlib.read_findings(results_dir)
        df_runs = finlib.read_runs(results_dir)
        # Check outputs.
        self.assertEqual(df["snippet_id"].tolist(), ["SEED-001", "SAFE-002"])
        self.assertEqual(df["cwe"].tolist(), [89, pd.NA])
        self.assertEqual(df_runs.loc[0, "n_findings"], 2)
        self.assertEqual(df_runs.loc[0, "tool_version"], "1.0")
