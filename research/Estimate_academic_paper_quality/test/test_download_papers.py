import logging
import os
import re
import sys
import time
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
                min_delay_sec=0.0,
                max_delay_sec=0.0,
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
        # The summary counts the two papers as failed.
        expected_summary = """
        Stage `download` summary
          Papers in input:        2
          Processed in this run:  2
          Completed successfully: 0
          Failed:                 2
          Completed in total:     0 of 2
          Elapsed time: <TIME>
          Output files written in this run:
            'out/paper_pdf/<BASE>.pdf': 0 files
            'out/paper_md/<BASE>.md': 0 files
          Issues file: 'out/02-download.issues.json' (2 issues)
            download_failed: 1
            no_download_link: 1
          Summary file: 'out/02-download.summary.txt'
        """
        actual_summary = epqteuti.read_summary(output_dir, "download")
        self.assert_equal(actual_summary, expected_summary, dedent=True)

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
                    min_delay_sec=0.0,
                    max_delay_sec=0.0,
                    dry_run=False,
                )

        # Run test: the first run downloads the paper.
        issues = _run()
        n_calls = len(hio.from_file(calls_path).split("\n"))
        first_summary = epqteuti.read_summary(output_dir, "download")
        # Run test: the second run calls nothing.
        rerun_issues = _run()
        # Check outputs.
        self.assertEqual((issues, rerun_issues), ([], []))
        self.assertTrue(
            os.path.isfile(epqpaths.get_markdown_path(output_dir, "p1"))
        )
        self.assertEqual(len(hio.from_file(calls_path).split("\n")), n_calls)
        # The first summary has the paper as completed, the second as done before.
        self.assertIn("Completed successfully: 1", first_summary)
        self.assertIn("'out/paper_md/<BASE>.md': 1 file\n", first_summary)
        expected_rerun_summary = """
        Stage `download` summary
          Papers in input:        1
          Skipped, done before:   1
          Processed in this run:  0
          Completed successfully: 0
          Failed:                 0
          Completed in total:     1 of 1
          Elapsed time: <TIME>
          Output files written in this run:
            'out/paper_pdf/<BASE>.pdf': 0 files
            'out/paper_md/<BASE>.md': 0 files
          Issues file: 'out/02-download.issues.json' (0 issues)
          Summary file: 'out/02-download.summary.txt'
        """
        rerun_summary = epqteuti.read_summary(output_dir, "download")
        self.assert_equal(rerun_summary, expected_rerun_summary, dedent=True)

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
        sleeps: List[float] = []
        # Run test.
        with mock.patch.dict(os.environ, env):
            issues = dowpapers._download(
                papers,
                output_dir,
                tool_path=tool_path,
                log_level="INFO",
                min_delay_sec=1.0,
                max_delay_sec=2.0,
                dry_run=True,
                sleep_func=sleeps.append,
            )
        # Check outputs.
        self.assertEqual(issues, [])
        self.assertFalse(os.path.exists(calls_path))
        self.assertFalse(os.path.exists(output_dir))
        self.assertEqual(sleeps, [])

    def _download_failing(
        self,
        base_names: List[str],
        *,
        min_delay_sec: float,
        max_delay_sec: float,
        output_dir: str = "",
    ) -> List[float]:
        """
        Download papers that fail and record the waits between the downloads.

        :param base_names: base names of the papers, each with one input
        :param min_delay_sec: shortest wait
        :param max_delay_sec: longest wait
        :param output_dir: output dir, empty is `out` in the scratch space
        :return: the waits, in order
        """
        scratch = self.get_scratch_space()
        output_dir = output_dir or os.path.join(scratch, "out")
        papers = [
            epqteuti.make_paper(
                base_name=name, download_inputs=("https://x.example/fail",)
            )
            for name in base_names
        ]
        tool_path = epqteuti.make_stub_tool(scratch)
        env = {"EPQ_STUB_CALLS": os.path.join(scratch, "calls.txt")}
        sleeps: List[float] = []
        with mock.patch.dict(os.environ, env):
            dowpapers._download(
                papers,
                output_dir,
                tool_path=tool_path,
                log_level="INFO",
                min_delay_sec=min_delay_sec,
                max_delay_sec=max_delay_sec,
                dry_run=False,
                sleep_func=sleeps.append,
            )
        return sleeps

    def test4(self) -> None:
        """
        Test n downloads wait n - 1 times, and never before the first one.
        """
        # Run test.
        sleeps = self._download_failing(
            ["p1", "p2", "p3"], min_delay_sec=2.5, max_delay_sec=2.5
        )
        # Check outputs.
        self.assertEqual(sleeps, [2.5, 2.5])

    def test5(self) -> None:
        """
        Test each wait is a random time in the range.
        """
        # Run test.
        sleeps = self._download_failing(
            ["p1", "p2", "p3", "p4", "p5"], min_delay_sec=1.0, max_delay_sec=3.0
        )
        # Check outputs.
        self.assertEqual(len(sleeps), 4)
        for sleep in sleeps:
            self.assertGreaterEqual(sleep, 1.0)
            self.assertLessEqual(sleep, 3.0)
        # The waits are not all the same.
        self.assertGreater(len(set(sleeps)), 1)

    def test6(self) -> None:
        """
        Test a paper that is already converted causes no wait.
        """
        # Prepare inputs.
        scratch = self.get_scratch_space()
        output_dir = os.path.join(scratch, "out")
        hio.create_dir(epqpaths.get_pdf_dir(output_dir), incremental=True)
        hio.create_dir(epqpaths.get_markdown_dir(output_dir), incremental=True)
        epqteuti.create_pdf(
            epqpaths.get_pdf_path(output_dir, "p1"), ["Sharpe ratio of 1.4"]
        )
        hio.to_file(
            epqpaths.get_markdown_path(output_dir, "p1"), "Sharpe ratio " * 50
        )
        # Run test: `p1` is skipped and `p2` is the first download.
        sleeps = self._download_failing(
            ["p1", "p2"],
            min_delay_sec=1.0,
            max_delay_sec=1.0,
            output_dir=output_dir,
        )
        # Check outputs.
        self.assertEqual(sleeps, [])

    def test7(self) -> None:
        """
        Test the elapsed time of the summary includes the wait between downloads.
        """
        # Prepare inputs.
        scratch = self.get_scratch_space()
        output_dir = os.path.join(scratch, "out")
        papers = [
            epqteuti.make_paper(
                base_name=name, download_inputs=("https://x.example/fail",)
            )
            for name in ["p1", "p2"]
        ]
        tool_path = epqteuti.make_stub_tool(scratch)
        env = {"EPQ_STUB_CALLS": os.path.join(scratch, "calls.txt")}
        wait_sec = 0.3
        # Run test: the wait is real, so the stage takes at least `wait_sec`.
        with mock.patch.dict(os.environ, env):
            dowpapers._download(
                papers,
                output_dir,
                tool_path=tool_path,
                log_level="INFO",
                min_delay_sec=wait_sec,
                max_delay_sec=wait_sec,
                dry_run=False,
                sleep_func=time.sleep,
            )
        # Check outputs.
        text = hio.from_file(epqpaths.get_summary_path(output_dir, "download"))
        match = re.search(r"Elapsed time: ([\d.]+) sec", text)
        self.assertIsNotNone(match)
        elapsed_sec = float(match.group(1))  # type: ignore[union-attr]
        self.assertGreaterEqual(elapsed_sec, wait_sec - 0.05)
        self.assertLess(elapsed_sec, 30.0)


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
        Test a missing papers file stops the run.
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
        papers_file = os.path.join(scratch, "papers.json")
        epqrepor.write_papers(papers_file, [epqteuti.make_paper()])
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

    def test3(self) -> None:
        """
        Test the default input is the papers file in the output dir.
        """
        # Prepare inputs.
        output_dir = os.path.join(self.get_scratch_space(), "out")
        argv = [f"--output_dir={output_dir}", "--dry_run"]
        # Run test: there is no papers file in the output dir yet.
        with self.assertRaises(AssertionError) as cm:
            self.run_main(argv)
        # Check outputs.
        self.assertIn("01-resolve.paper_list.json", str(cm.exception))
        # Run test: the dry run finds the papers file.
        epqrepor.write_papers(
            epqpaths.get_papers_path(output_dir), [epqteuti.make_paper()]
        )
        self.run_main(argv)

    def test4(self) -> None:
        """
        Test a shortest wait that is longer than the longest wait stops the run.
        """
        # Prepare inputs.
        scratch = self.get_scratch_space()
        papers_file = os.path.join(scratch, "papers.json")
        epqrepor.write_papers(papers_file, [epqteuti.make_paper()])
        argv = [
            f"--input={papers_file}",
            f"--output_dir={scratch}/out",
            "--min_delay_sec=5",
            "--max_delay_sec=1",
            "--dry_run",
        ]
        # Run test.
        with self.assertRaises(AssertionError):
            self.run_main(argv)
