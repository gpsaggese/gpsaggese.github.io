import logging
import os
from typing import List

import helpers.hio as hio
import helpers.hunit_test as hunitest

import research.Estimate_academic_paper_quality.epq_summary as epqsumm
import research.Estimate_academic_paper_quality.epq_types as epqtypes
import research.Estimate_academic_paper_quality.test.epq_test_utils as epqteuti

_LOG = logging.getLogger(__name__)


def _make_summary(
    *,
    stage: str = "download",
    n_total: int = 20,
    n_done_before: int = 12,
    n_no_input: int = 0,
    n_ok: int = 6,
    n_failed: int = 2,
    dry_run: bool = False,
) -> epqtypes.StageSummary:
    """
    Build the summary of the stage `download`, with 6 papers done in the run.

    :return: summary with two issues, unless `dry_run` is True
    """
    issue_counts = (("download_failed", 1), ("no_download_link", 1))
    if dry_run:
        issue_counts = ()
    summary = epqtypes.StageSummary(
        stage=stage,
        unit="paper",
        n_total=n_total,
        n_done_before=n_done_before,
        n_no_input=n_no_input,
        n_ok=n_ok,
        n_failed=n_failed,
        outputs=(("out/papers/<BASE>.pdf", 6), ("out/papers/<BASE>.md", 6)),
        issues_path="out/02-download.issues.json",
        issue_counts=issue_counts,
        extra=(),
        dry_run=dry_run,
    )
    return summary


# #############################################################################
# Test_count_issues
# #############################################################################


class Test_count_issues(hunitest.TestCase):
    """
    Test `epq_summary.count_issues()`.
    """

    def test1(self) -> None:
        """
        Test the issues are counted by status, sorted by label.
        """
        # Prepare inputs.
        issues = [
            epqteuti.make_issue(stage="download", status="no_download_link"),
            epqteuti.make_issue(stage="download", status="download_failed"),
            epqteuti.make_issue(stage="download", status="no_download_link"),
        ]
        add_stage = False
        # Run test.
        actual = epqsumm.count_issues(issues, add_stage=add_stage)
        # Check outputs.
        expected = (("download_failed", 1), ("no_download_link", 2))
        self.assertEqual(actual, expected)

    def test2(self) -> None:
        """
        Test the label has the stage, for the issues of many stages.
        """
        # Prepare inputs.
        issues = [
            epqteuti.make_issue(stage="extract", status="extraction_failed"),
            epqteuti.make_issue(stage="download", status="download_failed"),
        ]
        add_stage = True
        # Run test.
        actual = epqsumm.count_issues(issues, add_stage=add_stage)
        # Check outputs.
        expected = (
            ("download.download_failed", 1),
            ("extract.extraction_failed", 1),
        )
        self.assertEqual(actual, expected)

    def test3(self) -> None:
        """
        Test no issue gives no count.
        """
        # Prepare inputs.
        issues: List[epqtypes.Issue] = []
        add_stage = False
        # Run test.
        actual = epqsumm.count_issues(issues, add_stage=add_stage)
        # Check outputs.
        self.assertEqual(actual, ())


# #############################################################################
# Test_format_stage_summary
# #############################################################################


class Test_format_stage_summary(hunitest.TestCase):
    """
    Test `epq_summary.format_stage_summary()`.
    """

    def test1(self) -> None:
        """
        Test the counts, the output files, and the issues file of a run.
        """
        # Prepare inputs.
        summary = _make_summary()
        summary_path = "out/02-download.summary.txt"
        # Run test.
        actual = epqsumm.format_stage_summary(summary, summary_path=summary_path)
        # Check outputs.
        expected = """
        Stage `download` summary
          Papers in input:        20
          Skipped, done before:   12
          Processed in this run:   8
          Completed successfully:  6
          Failed:                  2
          Completed in total:     18 of 20
          Output files written in this run:
            'out/papers/<BASE>.pdf': 6 files
            'out/papers/<BASE>.md': 6 files
          Issues file: 'out/02-download.issues.json' (2 issues)
            download_failed: 1
            no_download_link: 1
          Summary file: 'out/02-download.summary.txt'
        """
        self.assert_equal(actual, expected, dedent=True)

    def test2(self) -> None:
        """
        Test a run with no issue, one file, and the items skipped for no input.
        """
        # Prepare inputs.
        summary = epqtypes.StageSummary(
            stage="extract",
            unit="paper",
            n_total=5,
            n_done_before=0,
            n_no_input=2,
            n_ok=3,
            n_failed=0,
            outputs=(("out/papers/<BASE>.features.json", 1),),
            issues_path="out/03-extract.issues.json",
            issue_counts=(),
            extra=("LLM cost in this run: 0.0600 USD",),
            dry_run=False,
        )
        summary_path = "out/03-extract.summary.txt"
        # Run test.
        actual = epqsumm.format_stage_summary(summary, summary_path=summary_path)
        # Check outputs.
        expected = """
        Stage `extract` summary
          Papers in input:        5
          Skipped, no input:      2
          Processed in this run:  3
          Completed successfully: 3
          Failed:                 0
          Completed in total:     3 of 5
          Output files written in this run:
            'out/papers/<BASE>.features.json': 1 file
          Issues file: 'out/03-extract.issues.json' (0 issues)
          LLM cost in this run: 0.0600 USD
          Summary file: 'out/03-extract.summary.txt'
        """
        self.assert_equal(actual, expected, dedent=True)

    def test3(self) -> None:
        """
        Test a dry run says what it would process and lists no file.
        """
        # Prepare inputs.
        summary = _make_summary(n_ok=0, n_failed=0, dry_run=True)
        summary_path = ""
        # Run test.
        actual = epqsumm.format_stage_summary(summary, summary_path=summary_path)
        # Check outputs.
        expected = """
        [DRY_RUN] Stage `download` summary
          Papers in input:      20
          Skipped, done before: 12
          Would process:         8
          No file is written
        """
        self.assert_equal(actual, expected, dedent=True)

    def test4(self) -> None:
        """
        Test the unit is `line` for the stage `resolve`.
        """
        # Prepare inputs.
        summary = epqtypes.StageSummary(
            stage="resolve",
            unit="line",
            n_total=3,
            n_done_before=0,
            n_no_input=0,
            n_ok=2,
            n_failed=1,
            outputs=(("out/01-resolve.paper_list.json", 1),),
            issues_path="out/01-resolve.issues.json",
            issue_counts=(("unresolved", 1),),
            extra=("Papers after the dedup: 1",),
            dry_run=False,
        )
        summary_path = "out/01-resolve.summary.txt"
        # Run test.
        actual = epqsumm.format_stage_summary(summary, summary_path=summary_path)
        # Check outputs.
        expected = """
        Stage `resolve` summary
          Lines in input:         3
          Processed in this run:  3
          Completed successfully: 2
          Failed:                 1
          Completed in total:     2 of 3
          Output files written in this run:
            'out/01-resolve.paper_list.json': 1 file
          Issues file: 'out/01-resolve.issues.json' (1 issue)
            unresolved: 1
          Papers after the dedup: 1
          Summary file: 'out/01-resolve.summary.txt'
        """
        self.assert_equal(actual, expected, dedent=True)

    def test5(self) -> None:
        """
        Test an item that is in no group is an error.
        """
        # Prepare inputs.
        summary = _make_summary(n_total=21)
        summary_path = ""
        # Run test.
        with self.assertRaises(AssertionError) as cm:
            epqsumm.format_stage_summary(summary, summary_path=summary_path)
        # Check outputs.
        self.assertIn("must be in one group", str(cm.exception))


# #############################################################################
# Test_emit_stage_summary
# #############################################################################


class Test_emit_stage_summary(hunitest.TestCase):
    """
    Test `epq_summary.emit_stage_summary()`.
    """

    def test1(self) -> None:
        """
        Test the file `NN-<stage>.summary.txt` has the text of the summary.
        """
        # Prepare inputs.
        output_dir = os.path.join(self.get_scratch_space(), "out")
        summary = _make_summary()
        summary_path = os.path.join(output_dir, "02-download.summary.txt")
        # Run test.
        epqsumm.emit_stage_summary(summary, output_dir)
        # Check outputs.
        expected = epqsumm.format_stage_summary(
            summary, summary_path=summary_path
        )
        self.assert_equal(hio.from_file(summary_path), expected + "\n")

    def test2(self) -> None:
        """
        Test a dry run writes no file.
        """
        # Prepare inputs.
        output_dir = os.path.join(self.get_scratch_space(), "out")
        summary = _make_summary(n_ok=0, n_failed=0, dry_run=True)
        # Run test.
        epqsumm.emit_stage_summary(summary, output_dir)
        # Check outputs.
        self.assertFalse(os.path.exists(output_dir))
