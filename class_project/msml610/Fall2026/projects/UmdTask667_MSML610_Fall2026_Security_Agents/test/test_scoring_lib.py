import logging
import math

import pandas as pd

import helpers.hunit_test as hunitest

import findings_lib as finlib
import scoring_lib as scolib

_LOG = logging.getLogger(__name__)


def _get_labels() -> pd.DataFrame:
    """
    Return labels for 2 pairs: SQL (bug on line 40) and XSS (bug on line 39).
    """
    labels = pd.DataFrame(
        {
            "snippet_id": ["SEED-001", "SAFE-001", "SEED-002", "SAFE-002"],
            "vulnerable": [True, False, True, False],
            "cwe": [89, 89, 79, 79],
            "bug_lines": [[40], [], [39], []],
            "pair_id": ["P-001", "P-001", "P-002", "P-002"],
        }
    )
    return labels


def _get_findings(rows: list) -> pd.DataFrame:
    """
    Build a findings table from `(snippet_id, line, cwe)` tuples.
    """
    findings = [
        finlib.make_finding(
            snippet_id=sid, agent="a", run_id="r1", line=line, cwe=cwe
        )
        for sid, line, cwe in rows
    ]
    df = pd.DataFrame(findings, columns=list(finlib.FINDING_COLUMNS))
    return df


# #############################################################################
# Test_cwe_matches
# #############################################################################


class Test_cwe_matches(hunitest.TestCase):
    """
    Test matching reported CWEs to labelled CWEs.
    """

    def test1(self) -> None:
        """
        Test exact, related, unrelated, and missing CWEs.
        """
        # Prepare inputs.
        cases = [(89, 89), (943, 89), (704, 89), (None, 89)]
        # Prepare outputs.
        expected = [True, True, False, False]
        # Run test.
        actual = [scolib.cwe_matches(pred, act) for pred, act in cases]
        # Check outputs.
        self.assertEqual(actual, expected)


# #############################################################################
# Test_score_snippets
# #############################################################################


class Test_score_snippets(hunitest.TestCase):
    """
    Test per-snippet scoring with the line window and CWE match.
    """

    def test1(self) -> None:
        """
        Test that the window is inclusive and a wrong CWE is line-only.
        """
        # Prepare inputs.
        labels = _get_labels()
        findings = _get_findings(
            [
                # 3 lines away with the right CWE: strict match.
                ("SEED-001", 43, 89),
                # On the bug line with the wrong CWE: line match only.
                ("SEED-002", 39, 704),
            ]
        )
        # Run test.
        scored = scolib.score_snippets(labels, findings)
        # Check outputs.
        actual = scored[
            ["snippet_id", "flagged", "localized_line", "localized_strict"]
        ].to_string(index=False)
        expected = """
        snippet_id  flagged  localized_line  localized_strict
          SEED-001     True            True              True
          SAFE-001    False           False             False
          SEED-002     True            True             False
          SAFE-002    False           False             False
        """
        self.assert_equal(actual, expected, dedent=True, fuzzy_match=True)

    def test2(self) -> None:
        """
        Test that a finding 4 lines away is outside the default window.
        """
        # Prepare inputs.
        labels = _get_labels()
        findings = _get_findings([("SEED-001", 44, 89)])
        # Run test.
        scored = scolib.score_snippets(labels, findings)
        # Check outputs.
        row = scored.set_index("snippet_id").loc["SEED-001"]
        self.assertTrue(row["flagged"])
        self.assertFalse(row["localized_line"])

    def test3(self) -> None:
        """
        Test that a finding with no location flags but never localizes.

        The bug is on line 2, within the window of line 0, so only the explicit
        `NO_LINE` rule keeps it from localizing.
        """
        # Prepare inputs.
        labels = _get_labels()
        labels.at[0, "bug_lines"] = [2]
        findings = _get_findings([("SEED-001", finlib.NO_LINE, 89)])
        # Run test.
        scored = scolib.score_snippets(labels, findings)
        # Check outputs.
        row = scored.set_index("snippet_id").loc["SEED-001"]
        self.assertTrue(row["flagged"])
        self.assertFalse(row["localized_line"])
        self.assertFalse(row["localized_strict"])


# #############################################################################
# Test_compute_metrics
# #############################################################################


class Test_compute_metrics(hunitest.TestCase):
    """
    Test the run-level metrics on a small hand-computed case.
    """

    def test1(self) -> None:
        """
        Test one TP, one FP, one FN, one TN.

        - SEED-001 is flagged with the right line and CWE: TP
        - SAFE-001 is flagged: FP
        - SEED-002 is not flagged: FN
        - SAFE-002 is not flagged: TN
        """
        # Prepare inputs.
        labels = _get_labels()
        findings = _get_findings([("SEED-001", 40, 89), ("SAFE-001", 41, 89)])
        scored = scolib.score_snippets(labels, findings)
        # Run test.
        metrics = scolib.compute_metrics(scored)
        # Check outputs.
        self.assertEqual(
            (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]),
            (1, 1, 1, 1),
        )
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)
        self.assertAlmostEqual(metrics["fpr"], 0.5)
        self.assertAlmostEqual(metrics["fdr"], 0.5)
        self.assertAlmostEqual(metrics["mcc"], 0.0)
        self.assertAlmostEqual(metrics["youden_j"], 0.0)
        self.assertAlmostEqual(metrics["recall_strict"], 0.5)
        self.assertAlmostEqual(metrics["finding_precision"], 0.5)
        # Neither pair has its vulnerable snippet flagged and its twin clean.
        self.assertAlmostEqual(metrics["paired_accuracy"], 0.0)

    def test2(self) -> None:
        """
        Test that an agent with no findings gets NaN precision, not an error.
        """
        # Prepare inputs.
        labels = _get_labels()
        findings = _get_findings([])
        scored = scolib.score_snippets(labels, findings)
        # Run test.
        metrics = scolib.compute_metrics(scored)
        # Check outputs.
        self.assertEqual(metrics["fn"], 2)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertTrue(math.isnan(metrics["precision"]))


# #############################################################################
# Test_score_runs
# #############################################################################


class Test_score_runs(hunitest.TestCase):
    """
    Test scoring several runs, including agents that report no locations.
    """

    def helper(self, rows: list) -> dict:
        """
        Score one run with the given findings and return its metrics.
        """
        labels = _get_labels()
        findings = _get_findings(rows)
        runs = pd.DataFrame({"agent": ["a"], "run_id": ["r1"]})
        scores = scolib.score_runs(labels, findings, runs)
        metrics = scores.iloc[0].to_dict()
        return metrics

    def test1(self) -> None:
        """
        Test that a run with no locations gets N/A localization metrics.
        """
        # Run test.
        metrics = self.helper(
            [("SEED-001", finlib.NO_LINE, 89), ("SAFE-001", finlib.NO_LINE, 89)]
        )
        # Check outputs.
        self.assertEqual((metrics["tp"], metrics["fp"]), (1, 1))
        self.assertAlmostEqual(metrics["recall"], 0.5)
        for key in ("recall_line", "recall_strict", "cwe_accuracy"):
            self.assertTrue(math.isnan(metrics[key]), key)
        self.assertTrue(math.isnan(metrics["finding_precision"]))

    def test2(self) -> None:
        """
        Test that a run with at least one location keeps its numbers.
        """
        # Run test.
        metrics = self.helper(
            [("SEED-001", 40, 89), ("SEED-002", finlib.NO_LINE, 79)]
        )
        # Check outputs.
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["recall_line"], 0.5)
        self.assertAlmostEqual(metrics["recall_strict"], 0.5)
