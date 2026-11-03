import logging

import helpers.hunit_test as hunitest

import research.Estimate_academic_paper_quality.epq_paths as epqpaths

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_stage_files
# #############################################################################


class Test_stage_files(hunitest.TestCase):
    """
    Test the names of the files that a stage writes in the output dir.
    """

    def test1(self) -> None:
        """
        Test the files of the stages are named by the order of the executable.
        """
        # Prepare inputs.
        out = "run"
        # Run test.
        actual = [
            epqpaths.get_input_copy_path(out),
            epqpaths.get_papers_path(out),
            epqpaths.get_issues_path(out, "resolve"),
            epqpaths.get_issues_path(out, "download"),
            epqpaths.get_issues_path(out, "extract"),
            epqpaths.get_issues_path(out, "rate"),
            epqpaths.get_report_path(out),
            epqpaths.get_all_issues_path(out),
            epqpaths.get_references_path(out),
            epqpaths.get_summary_path(out, "resolve"),
            epqpaths.get_summary_path(out, "download"),
            epqpaths.get_summary_path(out, "extract"),
            epqpaths.get_summary_path(out, "rate"),
            epqpaths.get_summary_path(out, "report"),
        ]
        # Check outputs.
        expected = [
            "run/00-paper_list.csv",
            "run/01-resolve.paper_list.json",
            "run/01-resolve.issues.json",
            "run/02-download.issues.json",
            "run/03-extract.issues.json",
            "run/04-rate.issues.json",
            "run/04-report.report.csv",
            "run/04-report.issues.json",
            "run/04-report.references.md",
            "run/01-resolve.summary.txt",
            "run/02-download.summary.txt",
            "run/03-extract.summary.txt",
            "run/04-rate.summary.txt",
            "run/04-report.summary.txt",
        ]
        self.assertEqual(actual, expected)

    def test2(self) -> None:
        """
        Test the stages `rate` and `report` share the number of their script.
        """
        # Run test.
        rate = epqpaths.get_stage_path("run", "rate", "x.json")
        report = epqpaths.get_stage_path("run", "report", "x.json")
        # Check outputs.
        self.assertEqual(rate, "run/04-rate.x.json")
        self.assertEqual(report, "run/04-report.x.json")

    def test3(self) -> None:
        """
        Test a stage that does not exist is rejected.
        """
        # Run test.
        with self.assertRaises(AssertionError):
            epqpaths.get_stage_path("run", "unknown", "x.json")

    def test4(self) -> None:
        """
        Test the stage `report` has no stage issues file.
        """
        # Run test.
        with self.assertRaises(AssertionError):
            epqpaths.get_issues_path("run", "report")

    def test5(self) -> None:
        """
        Test the cache and the logs are in `state/`, and the files of a paper
        are in `papers/` with no number.
        """
        # Run test.
        actual = [
            epqpaths.get_cache_dir("run"),
            epqpaths.get_log_dir("run"),
            epqpaths.get_pdf_path("run", "p1"),
            epqpaths.get_markdown_path("run", "p1"),
            epqpaths.get_features_path("run", "p1"),
            epqpaths.get_rating_path("run", "p1"),
        ]
        # Check outputs.
        expected = [
            "run/state/cache",
            "run/state/logs",
            "run/papers/p1.pdf",
            "run/papers/p1.md",
            "run/papers/p1.features.json",
            "run/papers/p1.rating.json",
        ]
        self.assertEqual(actual, expected)
