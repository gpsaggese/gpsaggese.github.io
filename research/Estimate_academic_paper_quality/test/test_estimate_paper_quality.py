import logging
import os
import sys
from typing import List
from unittest import mock

import helpers.hio as hio
import helpers.hunit_test as hunitest

import research.Estimate_academic_paper_quality.estimate_paper_quality as estpaqua

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_build_command
# #############################################################################


class Test_build_command(hunitest.TestCase):
    """
    Test the command line of each action.
    """

    def build(
        self,
        action: str,
        *,
        input_csv: str = "in.csv",
        model: str = "",
        force_extract: bool = False,
        refresh_citations: bool = False,
        log_level: str = "INFO",
    ) -> str:
        """
        Build the command of an action, with defaults for the other flags.

        :param action: action to run
        :param input_csv: CSV for `resolve_papers.py`
        :param model: LLM for `extract_features.py`
        :param force_extract: if True, extract again the papers that are done
        :param refresh_citations: if True, ask for fresh citation counts
        :param log_level: logging level to forward to the executable
        :return: command line
        """
        cmd = estpaqua._build_command(
            action,
            f"{action}.py",
            input_csv=input_csv,
            output_dir="out",
            model=model,
            force_extract=force_extract,
            refresh_citations=refresh_citations,
            log_level=log_level,
        )
        return cmd

    def test1(self) -> None:
        """
        Test only `resolve_papers.py` gets `--input`.
        """
        # Run test.
        actual = self.build("resolve_papers")
        # Check outputs.
        expected = "resolve_papers.py --input in.csv --output_dir out -v INFO"
        self.assert_equal(actual, expected)

    def test2(self) -> None:
        """
        Test `download_papers.py` gets only the output dir and the log level.
        """
        # Run test.
        actual = self.build(
            "download_papers",
            model="m",
            force_extract=True,
            refresh_citations=True,
        )
        # Check outputs.
        expected = "download_papers.py --output_dir out -v INFO"
        self.assert_equal(actual, expected)

    def test3(self) -> None:
        """
        Test `extract_features.py` gets the model and `--force_extract` only
        when asked.
        """
        # Run test.
        default = self.build("extract_features")
        asked = self.build(
            "extract_features", model="gpt-4o-mini", force_extract=True
        )
        # Check outputs.
        self.assert_equal(
            default, "extract_features.py --output_dir out -v INFO"
        )
        expected = (
            "extract_features.py --output_dir out --model gpt-4o-mini "
            "--force_extract -v INFO"
        )
        self.assert_equal(asked, expected)

    def test4(self) -> None:
        """
        Test `report_papers.py` gets `--refresh_citations` only when asked.
        """
        # Run test.
        default = self.build("report_papers")
        asked = self.build("report_papers", refresh_citations=True)
        # Check outputs.
        self.assert_equal(default, "report_papers.py --output_dir out -v INFO")
        expected = (
            "report_papers.py --output_dir out --refresh_citations -v INFO"
        )
        self.assert_equal(asked, expected)

    def test5(self) -> None:
        """
        Test the log level is forwarded to the executable.
        """
        # Run test.
        actual = self.build("report_papers", log_level="DEBUG")
        # Check outputs.
        expected = "report_papers.py --output_dir out -v DEBUG"
        self.assert_equal(actual, expected)

    def test6(self) -> None:
        """
        Test a path with a space is quoted.
        """
        # Run test.
        actual = self.build("resolve_papers", input_csv="my papers.csv")
        # Check outputs.
        expected = (
            "resolve_papers.py --input 'my papers.csv' --output_dir out -v INFO"
        )
        self.assert_equal(actual, expected)

    def test7(self) -> None:
        """
        Test an action that does not exist is rejected.
        """
        # Run test.
        with self.assertRaises(AssertionError):
            self.build("rate")


# #############################################################################
# Test_format_timings
# #############################################################################


class Test_format_timings(hunitest.TestCase):
    """
    Test the table with the time of each action.
    """

    def test1(self) -> None:
        """
        Test one row per action in order, then the total, with aligned values.
        """
        # Prepare inputs.
        timings = [("resolve_papers", 12.34), ("download_papers", 65.0)]
        # Run test.
        actual = estpaqua._format_timings(timings)
        # Check outputs.
        expected = """
        Time of each action
          resolve_papers:  12.3 sec
          download_papers: 1 min 5.0 sec
          Total:           1 min 17.3 sec
        """
        self.assert_equal(actual, expected, dedent=True)

    def test2(self) -> None:
        """
        Test the total of one action is the time of that action.
        """
        # Prepare inputs.
        timings = [("report_papers", 3725.0)]
        # Run test.
        actual = estpaqua._format_timings(timings)
        # Check outputs.
        expected = """
        Time of each action
          report_papers: 1 h 2 min 5.0 sec
          Total:         1 h 2 min 5.0 sec
        """
        self.assert_equal(actual, expected, dedent=True)

    def test3(self) -> None:
        """
        Test that no action is an error, since there is nothing to report.
        """
        # Run test.
        with self.assertRaises(AssertionError):
            estpaqua._format_timings([])


# #############################################################################
# Test_run_action
# #############################################################################


class Test_run_action(hunitest.TestCase):
    """
    Test that an action runs its command, and that a dry run does not.
    """

    def test1(self) -> None:
        """
        Test the command of the action runs.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "done.txt")
        # Run test.
        elapsed_sec = estpaqua._run_action(
            "resolve_papers", f"touch {path}", dry_run=False
        )
        # Check outputs.
        self.assertTrue(os.path.exists(path))
        self.assertGreater(elapsed_sec, 0.0)

    def test2(self) -> None:
        """
        Test `dry_run` does not run the command of the action.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "done.txt")
        # Run test.
        elapsed_sec = estpaqua._run_action(
            "resolve_papers", f"touch {path}", dry_run=True
        )
        # Check outputs.
        self.assertFalse(os.path.exists(path))
        self.assertEqual(elapsed_sec, 0.0)


# #############################################################################
# Test_estimate_paper_quality_py
# #############################################################################


class Test_estimate_paper_quality_py(hunitest.TestCase):
    """
    Test the command line of `estimate_paper_quality.py`.
    """

    def run_main(self, argv: List[str]) -> None:
        """
        Run `_main()` with the given flags.

        :param argv: flags of the script
        """
        parser = estpaqua._parse()
        with mock.patch.object(
            sys, "argv", ["estimate_paper_quality.py"] + argv
        ):
            estpaqua._main(parser)

    def test1(self) -> None:
        """
        Test the action `resolve_papers` needs `--input`.
        """
        # Prepare inputs.
        output_dir = os.path.join(self.get_scratch_space(), "out")
        argv = [f"--output_dir={output_dir}", "--dry_run"]
        # Run test.
        with self.assertRaises(AssertionError) as cm:
            self.run_main(argv)
        # Check outputs.
        self.assertIn("needs `--input`", str(cm.exception))

    def test2(self) -> None:
        """
        Test a missing input CSV stops the run before any action.
        """
        # Prepare inputs.
        scratch = self.get_scratch_space()
        argv = [
            f"--input={scratch}/missing.csv",
            f"--output_dir={scratch}/out",
            "--dry_run",
        ]
        # Run test.
        with self.assertRaises(AssertionError):
            self.run_main(argv)

    def test3(self) -> None:
        """
        Test `--dry_run` runs no action and writes no file.
        """
        # Prepare inputs.
        scratch = self.get_scratch_space()
        input_csv = os.path.join(scratch, "papers.csv")
        hio.to_file(input_csv, "Lazy Prices\n")
        output_dir = os.path.join(scratch, "out")
        argv = [
            f"--input={input_csv}",
            f"--output_dir={output_dir}",
            "--dry_run",
        ]
        # Run test.
        self.run_main(argv)
        # Check outputs.
        self.assertFalse(os.path.exists(output_dir))

    def test4(self) -> None:
        """
        Test the action `resolve_papers` is not needed to run the other actions.
        """
        # Prepare inputs.
        output_dir = os.path.join(self.get_scratch_space(), "out")
        argv = [
            f"--output_dir={output_dir}",
            "--skip_action=resolve_papers",
            "--dry_run",
        ]
        # Run test.
        self.run_main(argv)
        # Check outputs.
        self.assertFalse(os.path.exists(output_dir))

    def test5(self) -> None:
        """
        Test an action that does not exist is rejected.
        """
        # Prepare inputs.
        output_dir = os.path.join(self.get_scratch_space(), "out")
        argv = [f"--output_dir={output_dir}", "--only_action=rate", "--dry_run"]
        # Run test.
        with self.assertRaises(AssertionError):
            self.run_main(argv)
