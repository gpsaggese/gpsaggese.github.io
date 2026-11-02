import csv
import logging
import os
import sys
from typing import Any, List
from unittest import mock

import helpers.hcache_simple as hcacsimp
import helpers.hio as hio
import helpers.hunit_test as hunitest
import pytest

import research.Estimate_academic_paper_quality.epq_pipeline as epqpipel
import research.Estimate_academic_paper_quality.epq_report as epqrepor
import research.Estimate_academic_paper_quality.resolve_papers as resopapers
import research.Estimate_academic_paper_quality.test.epq_test_utils as epqteuti

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_resolve_papers_py
# #############################################################################


class Test_resolve_papers_py(hunitest.TestCase):
    """
    Test the command line of `resolve_papers.py`.
    """

    @pytest.fixture(autouse=True)
    def setup_teardown_test(self):
        """
        Disable the cache of the helpers during each test.

        `download_utils.detect_doi()` caches its answers on disk: the cache is
        off, so that the tests write nothing outside the scratch space.
        """
        hcacsimp.enable_caching(False)
        yield
        hcacsimp.enable_caching(True)

    def make_input_csv(self, inputs: List[str]) -> str:
        """
        Write an input CSV in the scratch space.

        :param inputs: values of the column `input`
        :return: path of the CSV
        """
        path = os.path.join(self.get_scratch_space(), "papers_in.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["input"])
            writer.writerows([[text] for text in inputs])
        return path

    def run_main(self, argv: List[str], sources: Any) -> None:
        """
        Run `_main()` with the given flags and a fake client of the APIs.

        :param argv: flags of the script
        :param sources: client of the APIs that replaces the real one
        """
        old_cache_dir = hcacsimp.get_cache_dir()
        self.addCleanup(hcacsimp.set_cache_dir, old_cache_dir)
        parser = resopapers._parse()
        with (
            mock.patch.object(sys, "argv", ["resolve_papers.py"] + argv),
            mock.patch.object(epqpipel, "make_sources", return_value=sources),
        ):
            resopapers._main(parser)

    def test1(self) -> None:
        """
        Test the papers CSV and the issues CSV of a list with a bad row.
        """
        # Prepare inputs.
        scratch = self.get_scratch_space()
        input_csv = self.make_input_csv(
            ["2001.00001", "Unknown Title", "", "# a comment", "2001.00001v2"]
        )
        papers_csv = os.path.join(scratch, "out", "papers.csv")
        sources = epqteuti.FakeBibSources(
            by_id={
                "arxiv:2001.00001": epqteuti.make_record(
                    source="arxiv",
                    s2_paper_id="",
                    arxiv_id="2001.00001",
                    pdf_urls=("https://arxiv.org/pdf/2001.00001",),
                )
            }
        )
        argv = [f"--input={input_csv}", f"--output={papers_csv}"]
        # Prepare outputs.
        expected_papers = """
        reference,year,authors,title,doi,arxiv_id,s2_paper_id,source,base_name,download_inputs,line_nos,key,aliases
        "2020, Cohen et al., ""Lazy Prices"" (https://arxiv.org/pdf/2001.00001)",2020,"Cohen, Lauren|Malloy, Christopher",Lazy Prices,10.1111/jofi.12885,2001.00001,,arxiv,2020.Cohen_et_al.Lazy_Prices,https://arxiv.org/pdf/2001.00001,1|5,lazy prices|cohen,doi:10.1111/jofi.12885|arxiv:2001.00001
        """
        expected_issues = """
        line_no,input,base_name,stage,status,reason,candidates
        2,Unknown Title,,resolve,unresolved,no source has a paper titled 'Unknown Title',
        """
        # Run test.
        self.run_main(argv, sources)
        # Check outputs.
        actual_papers = hio.from_file(papers_csv).replace("\r\n", "\n")
        self.assert_equal(actual_papers, expected_papers, dedent=True)
        issues_csv = os.path.join(scratch, "out", "papers.issues.csv")
        actual_issues = hio.from_file(issues_csv).replace("\r\n", "\n")
        self.assert_equal(actual_issues, expected_issues, dedent=True)

    def test2(self) -> None:
        """
        Test `--issues_output` selects the path of the issues CSV.
        """
        # Prepare inputs.
        scratch = self.get_scratch_space()
        input_csv = self.make_input_csv(["Unknown Title"])
        papers_csv = os.path.join(scratch, "papers.csv")
        issues_csv = os.path.join(scratch, "bad_rows.csv")
        argv = [
            f"--input={input_csv}",
            f"--output={papers_csv}",
            f"--issues_output={issues_csv}",
        ]
        # Run test.
        self.run_main(argv, epqteuti.FakeBibSources())
        # Check outputs.
        self.assertEqual(len(epqrepor.read_issues_csv(issues_csv)), 1)
        self.assertFalse(
            os.path.exists(epqpipel.get_resolve_issues_path(papers_csv))
        )

    def test3(self) -> None:
        """
        Test `--dry_run` makes no lookup and writes no file.
        """
        # Prepare inputs.
        scratch = self.get_scratch_space()
        input_csv = self.make_input_csv(["2001.00001", "Unknown Title"])
        papers_csv = os.path.join(scratch, "out", "papers.csv")
        sources = epqteuti.FakeBibSources()
        argv = [f"--input={input_csv}", f"--output={papers_csv}", "--dry_run"]
        # Run test.
        self.run_main(argv, sources)
        # Check outputs.
        self.assertEqual(sources.calls, [])
        self.assertFalse(os.path.exists(os.path.dirname(papers_csv)))

    def test4(self) -> None:
        """
        Test an input CSV with no column `input` stops the run.
        """
        # Prepare inputs.
        scratch = self.get_scratch_space()
        input_csv = os.path.join(scratch, "papers_in.csv")
        hio.to_file(input_csv, "title\nLazy Prices\n")
        argv = [
            f"--input={input_csv}",
            f"--output={os.path.join(scratch, 'papers.csv')}",
        ]
        # Run test.
        with self.assertRaises(AssertionError) as cm:
            self.run_main(argv, epqteuti.FakeBibSources())
        # Check outputs.
        self.assertIn("has no column ['input']", str(cm.exception))

    def test5(self) -> None:
        """
        Test a missing input CSV stops the run.
        """
        # Prepare inputs.
        scratch = self.get_scratch_space()
        argv = [
            f"--input={scratch}/missing.csv",
            f"--output={scratch}/papers.csv",
        ]
        # Run test.
        with self.assertRaises(AssertionError):
            self.run_main(argv, epqteuti.FakeBibSources())
