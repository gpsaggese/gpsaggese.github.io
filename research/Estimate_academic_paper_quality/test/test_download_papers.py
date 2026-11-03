import logging
import os
import sys
from typing import List
from unittest import mock

import helpers.hio as hio
import helpers.hunit_test as hunitest

import research.Estimate_academic_paper_quality.download_papers as dowpapers
import research.Estimate_academic_paper_quality.epq_paths as epqpaths
import research.Estimate_academic_paper_quality.epq_report as epqrepor
import research.Estimate_academic_paper_quality.epq_types as epqtypes
import research.Estimate_academic_paper_quality.test.epq_test_utils as epqteuti

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test__download
# #############################################################################


class Test__download(hunitest.TestCase):
    """
    Test the stage of `download_papers.py` with a stub of the download tool.
    """

    def test1(self) -> None:
        """
        Test a download that fails is an issue and the run goes on.
        """
        # Prepare inputs.
        scratch = self.get_scratch_space()
        output_dir = os.path.join(scratch, "out")
        papers = [
            epqteuti.make_paper(
                base_name="p1", download_inputs=("https://x.example/fail",)
            ),
            epqteuti.make_paper(base_name="p2", download_inputs=()),
        ]
        tool_path = epqteuti.make_stub_tool(scratch)
        env = {"EPQ_STUB_CALLS": os.path.join(scratch, "calls.txt")}
        # Run test.
        with mock.patch.dict(os.environ, env):
            issues = dowpapers._download(
                papers,
                output_dir,
                tool_path=tool_path,
                log_level="INFO",
                dry_run=False,
            )
        # Check outputs.
        actual = [(i.base_name, i.status, i.reason) for i in issues]
        expected = [
            (
                "p1",
                "download_failed",
                "https://x.example/fail: Error: HTTP 404 for the input",
            ),
            ("p2", "no_download_link", "no public PDF found"),
        ]
        self.assertEqual(actual, expected)
        # The issues go to a file, for `report_papers.py`.
        path = epqpaths.get_issues_path(output_dir, "download")
        self.assertEqual(epqrepor.read_issues(path), issues)

    def test2(self) -> None:
        """
        Test a converted paper is not downloaded again.
        """
        # Prepare inputs.
        scratch = self.get_scratch_space()
        output_dir = os.path.join(scratch, "out")
        pdf_path = os.path.join(scratch, "source.pdf")
        epqteuti.create_pdf(pdf_path, ["Sharpe ratio of 1.4"])
        calls_path = os.path.join(scratch, "calls.txt")
        env = {"EPQ_STUB_CALLS": calls_path, "EPQ_STUB_PDF": pdf_path}
        papers = [
            epqteuti.make_paper(
                base_name="p1", download_inputs=("https://x.example/good",)
            )
        ]
        tool_path = epqteuti.make_stub_tool(scratch)

        def _run() -> List[epqtypes.Issue]:
            with mock.patch.dict(os.environ, env):
                return dowpapers._download(
                    papers,
                    output_dir,
                    tool_path=tool_path,
                    log_level="INFO",
                    dry_run=False,
                )

        # Run test: the first run downloads the paper.
        issues = _run()
        n_calls = len(hio.from_file(calls_path).split("\n"))
        # Run test: the second run calls nothing.
        rerun_issues = _run()
        # Check outputs.
        self.assertEqual((issues, rerun_issues), ([], []))
        self.assertTrue(
            os.path.isfile(epqpaths.get_markdown_path(output_dir, "p1"))
        )
        self.assertEqual(len(hio.from_file(calls_path).split("\n")), n_calls)

    def test3(self) -> None:
        """
        Test `dry_run` calls no tool and writes no file.
        """
        # Prepare inputs.
        scratch = self.get_scratch_space()
        output_dir = os.path.join(scratch, "out")
        papers = [
            epqteuti.make_paper(
                base_name="p1", download_inputs=("https://x.example/good",)
            )
        ]
        tool_path = epqteuti.make_stub_tool(scratch)
        calls_path = os.path.join(scratch, "calls.txt")
        env = {"EPQ_STUB_CALLS": calls_path}
        # Run test.
        with mock.patch.dict(os.environ, env):
            issues = dowpapers._download(
                papers,
                output_dir,
                tool_path=tool_path,
                log_level="INFO",
                dry_run=True,
            )
        # Check outputs.
        self.assertEqual(issues, [])
        self.assertFalse(os.path.exists(calls_path))
        self.assertFalse(os.path.exists(output_dir))


# #############################################################################
# Test_download_papers_py
# #############################################################################


class Test_download_papers_py(hunitest.TestCase):
    """
    Test the command line of `download_papers.py`.
    """

    def run_main(self, argv: List[str]) -> None:
        """
        Run `_main()` with the given flags.

        :param argv: flags of the script
        """
        parser = dowpapers._parse()
        with mock.patch.object(sys, "argv", ["download_papers.py"] + argv):
            dowpapers._main(parser)

    def test1(self) -> None:
        """
        Test a missing papers CSV stops the run.
        """
        # Prepare inputs.
        scratch = self.get_scratch_space()
        argv = [
            f"--input={scratch}/missing.csv",
            f"--output_dir={scratch}/out",
        ]
        # Run test.
        with self.assertRaises(AssertionError) as cm:
            self.run_main(argv)
        # Check outputs.
        self.assertIn("Run resolve_papers.py", str(cm.exception))

    def test2(self) -> None:
        """
        Test `--dry_run` writes no file, not even the dirs of the output.
        """
        # Prepare inputs.
        scratch = self.get_scratch_space()
        papers_file = os.path.join(scratch, "papers.csv")
        epqrepor.write_papers_csv(papers_file, [epqteuti.make_paper()])
        output_dir = os.path.join(scratch, "out")
        argv = [
            f"--input={papers_file}",
            f"--output_dir={output_dir}",
            "--dry_run",
        ]
        # Run test.
        self.run_main(argv)
        # Check outputs.
        self.assertFalse(os.path.exists(output_dir))
