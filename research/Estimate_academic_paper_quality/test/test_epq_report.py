import dataclasses
import logging
import os
from typing import List

import helpers.hio as hio
import helpers.hunit_test as hunitest

import research.Estimate_academic_paper_quality.epq_report as epqrepor
import research.Estimate_academic_paper_quality.epq_types as epqtypes
import research.Estimate_academic_paper_quality.test.epq_test_utils as epqteuti

_LOG = logging.getLogger(__name__)


def _make_issue(
    *, line_no: int, stage: str, status: str, input: str = "x", reason: str = "r"
) -> epqtypes.Issue:
    """
    Build an issue.
    """
    return epqtypes.Issue(
        line_no=line_no,
        input=input,
        base_name="",
        stage=stage,
        status=status,
        reason=reason,
        candidates=(),
    )


# #############################################################################
# Test_issues_json
# #############################################################################


class Test_issues_json(hunitest.TestCase):
    """
    Test `write_issues()` and `read_issues()`.
    """

    def test1(self) -> None:
        """
        Test issues with candidates are read back equal.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "issues.download.json")
        issue = epqtypes.Issue(
            line_no=3,
            input="Momentum",
            base_name="",
            stage="download",
            status="download_failed",
            reason="HTTP 403 for the last input",
            candidates=('1993, Jegadeesh et al., "Momentum" (???)',),
        )
        # Run test.
        epqrepor.write_issues(path, [issue])
        actual = epqrepor.read_issues(path)
        # Check outputs.
        self.assert_equal(str(actual), str([issue]))

    def test2(self) -> None:
        """
        Test a file with another schema version is rejected.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "issues.download.json")
        hio.to_file(path, '{"schema_version": 0, "issues": []}')
        # Run test.
        with self.assertRaises(AssertionError) as cm:
            epqrepor.read_issues(path)
        # Check outputs.
        self.assertIn("old schema", str(cm.exception))


# #############################################################################
# Test_read_input_csv
# #############################################################################


class Test_read_input_csv(hunitest.TestCase):
    """
    Test `epq_report.read_input_csv()`.
    """

    def test1(self) -> None:
        """
        Test rows are numbered from 1, and other columns and a BOM are ignored.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "in.csv")
        txt = (
            "\ufeffnote,input\n"
            'n1,"Lazy Prices (Cohen, Malloy)"\n'
            "n2,2001.00001\n"
            "n3,\n"
        )
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write(txt)
        # Prepare outputs.
        expected = [
            (1, "Lazy Prices (Cohen, Malloy)"),
            (2, "2001.00001"),
            (3, ""),
        ]
        # Run test.
        actual = epqrepor.read_input_csv(path)
        # Check outputs.
        self.assertEqual(actual, expected)

    def test2(self) -> None:
        """
        Test a file with no column `input` is rejected.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "in.csv")
        hio.to_file(path, "title\nLazy Prices\n")
        # Run test.
        with self.assertRaises(AssertionError) as cm:
            epqrepor.read_input_csv(path)
        # Check outputs.
        self.assertIn("has no column ['input']", str(cm.exception))


# #############################################################################
# Test_papers_csv
# #############################################################################


class Test_papers_csv(hunitest.TestCase):
    """
    Test `epq_report.write_papers_csv()` and `epq_report.read_papers_csv()`.
    """

    def test1(self) -> None:
        """
        Test papers are read back equal, with the lists of different sizes.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "papers.csv")
        paper1 = epqteuti.make_paper(
            line_nos=(3, 17),
            reference='2020, Cohen et al., "Lazy, Prices" (???)',
        )
        paper2 = dataclasses.replace(
            epqteuti.make_paper(year=2019, download_inputs=()),
            aliases=(),
            authors=(),
            title='A "quoted", title',
        )
        papers = [paper1, paper2]
        # Run test.
        epqrepor.write_papers_csv(path, papers)
        actual = epqrepor.read_papers_csv(path)
        # Check outputs.
        self.assert_equal(str(actual), str(papers))

    def test2(self) -> None:
        """
        Test the content of the papers CSV.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "papers.csv")
        paper = epqteuti.make_paper(line_nos=(3, 17))
        # Prepare outputs.
        expected = """
        reference,year,authors,title,doi,arxiv_id,s2_paper_id,source,base_name,download_inputs,line_nos,key,aliases
        "2020, Cohen et al., ""Lazy Prices"" (https://arxiv.org/pdf/2001.00001)",2020,"Cohen, Lauren|Malloy, Christopher",Lazy Prices,10.1111/jofi.12885,2001.00001,s2lazy,semantic_scholar,2020.Cohen_et_al.Lazy_Prices,https://arxiv.org/pdf/2001.00001,3|17,lazy prices|cohen,doi:10.1111/jofi.12885
        """
        # Run test.
        epqrepor.write_papers_csv(path, [paper])
        # Check outputs.
        actual = hio.from_file(path).replace("\r\n", "\n")
        self.assert_equal(actual, expected, dedent=True)

    def test3(self) -> None:
        """
        Test a CSV that is not a papers CSV is rejected.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "papers.csv")
        hio.to_file(path, "input\nLazy Prices\n")
        # Run test.
        with self.assertRaises(AssertionError) as cm:
            epqrepor.read_papers_csv(path)
        # Check outputs.
        self.assertIn("has no column", str(cm.exception))


# #############################################################################
# Test_read_issues_csv
# #############################################################################


class Test_read_issues_csv(hunitest.TestCase):
    """
    Test `epq_report.read_issues_csv()`.
    """

    def test1(self) -> None:
        """
        Test issues with and without candidates are read back equal.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "issues.csv")
        issues = [
            epqtypes.Issue(
                line_no=1,
                input="A title, with a comma",
                base_name="",
                stage="resolve",
                status="ambiguous",
                reason="2 different papers match the title",
                candidates=("c1", "c2"),
            ),
            _make_issue(line_no=5, stage="resolve", status="unresolved"),
        ]
        epqrepor.write_issues_csv(path, issues)
        # Run test.
        actual = epqrepor.read_issues_csv(path)
        # Check outputs.
        self.assert_equal(str(actual), str(issues))


# #############################################################################
# Test_features_file
# #############################################################################


class Test_features_file(hunitest.TestCase):
    """
    Test `write_features_file()` and `read_features_file()`.
    """

    def test1(self) -> None:
        """
        Test the features of a paper are read back equal.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "p.features.json")
        features = epqteuti.make_paper_features()
        # Run test.
        epqrepor.write_features_file(path, features)
        actual = epqrepor.read_features_file(path)
        # Check outputs.
        self.assert_equal(str(actual), str(features))

    def test2(self) -> None:
        """
        Test a features file with an old schema asks for a new extraction.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "p.features.json")
        hio.to_file(path, '{"schema_version": 99}')
        # Run test.
        with self.assertRaises(AssertionError) as cm:
            epqrepor.read_features_file(path)
        # Check outputs.
        self.assertIn("--force_extract", str(cm.exception))


# #############################################################################
# Test_rating_file
# #############################################################################


class Test_rating_file(hunitest.TestCase):
    """
    Test `write_rating_file()` and `read_rating_file()`.
    """

    def test1(self) -> None:
        """
        Test the rating of a paper is read back equal.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "p.rating.json")
        quality = epqteuti.make_quality()
        # Run test.
        epqrepor.write_rating_file(path, quality)
        actual = epqrepor.read_rating_file(path)
        # Check outputs.
        self.assert_equal(str(actual), str(quality))

    def test2(self) -> None:
        """
        Test a rating file with an old schema asks for a new rating.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "p.rating.json")
        hio.to_file(path, '{"schema_version": 99}')
        # Run test.
        with self.assertRaises(AssertionError) as cm:
            epqrepor.read_rating_file(path)
        # Check outputs.
        self.assertIn("--action rate", str(cm.exception))


# #############################################################################
# Test_write_references
# #############################################################################


class Test_write_references(hunitest.TestCase):
    """
    Test `epq_report.write_references()`.
    """

    def test1(self) -> None:
        """
        Test one bullet is written for each paper, in the given order.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "references.md")
        papers = [
            epqteuti.make_paper(
                reference='2020, Cohen et al., "Lazy Prices" (???)'
            ),
            epqteuti.make_paper(reference='2009, Pearl, "Causality" (???)'),
        ]
        # Prepare outputs.
        expected = """
        - 2020, Cohen et al., "Lazy Prices" (???)
        - 2009, Pearl, "Causality" (???)
        """
        # Run test.
        epqrepor.write_references(path, papers)
        # Check outputs.
        self.assert_equal(hio.from_file(path), expected, dedent=True)


# #############################################################################
# Test_report_csv
# #############################################################################


class Test_report_csv(hunitest.TestCase):
    """
    Test `build_report_rows()` and `write_report()`.
    """

    def test1(self) -> None:
        """
        Test a paper with no features file is not a row, and the order is kept.
        """
        # Prepare inputs.
        paper1 = epqteuti.make_paper(base_name="b1", year=2021)
        paper2 = epqteuti.make_paper(base_name="b2", year=2020)
        paper3 = epqteuti.make_paper(base_name="b3", year=2019)
        features = {
            "b3": epqteuti.make_paper_features(base_name="b3"),
            "b1": epqteuti.make_paper_features(base_name="b1"),
        }
        ratings = {"b3": epqteuti.make_quality(), "b1": epqteuti.make_quality()}
        # Run test.
        rows = epqrepor.build_report_rows(
            [paper1, paper2, paper3], features, ratings
        )
        # Check outputs.
        actual = [row["base_name"] for row in rows]
        self.assertEqual(actual, ["b1", "b3"])

    def test3(self) -> None:
        """
        Test a paper with a features file and no rating file is not a row.
        """
        # Prepare inputs.
        paper1 = epqteuti.make_paper(base_name="b1")
        paper2 = epqteuti.make_paper(base_name="b2")
        features = {
            "b1": epqteuti.make_paper_features(base_name="b1"),
            "b2": epqteuti.make_paper_features(base_name="b2"),
        }
        ratings = {"b2": epqteuti.make_quality()}
        # Run test.
        rows = epqrepor.build_report_rows([paper1, paper2], features, ratings)
        # Check outputs.
        actual = [row["base_name"] for row in rows]
        self.assertEqual(actual, ["b2"])

    def test2(self) -> None:
        """
        Test the content of `report.csv`, with `unknown` for a missing score.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "report.csv")
        paper = epqteuti.make_paper(
            reference='2020, Cohen et al., "Lazy, Prices" (???)'
        )
        features = {paper.base_name: epqteuti.make_paper_features()}
        ratings = {paper.base_name: epqteuti.make_quality()}
        rows = epqrepor.build_report_rows([paper], features, ratings)
        # Prepare outputs.
        expected = """
        reference,paper_quality,performance,recency,asset_class,theoretical_domain,mechanism,page_length_score,affiliation_score,citations_score,base_name
        "2020, Cohen et al., ""Lazy, Prices"" (???)",4,"Sharpe 1.4, out of sample",2020,US equities,unknown,associational,3,5,unknown,2020.Cohen_et_al.Lazy_Prices
        """
        # Run test.
        epqrepor.write_report(path, rows)
        # Check outputs.
        actual = hio.from_file(path).replace("\r\n", "\n")
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_write_issues_csv
# #############################################################################


class Test_write_issues_csv(hunitest.TestCase):
    """
    Test `epq_report.write_issues_csv()`.
    """

    def test1(self) -> None:
        """
        Test issues are sorted by stage then line, and candidates are joined.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "issues.csv")
        issues: List[epqtypes.Issue] = [
            _make_issue(line_no=2, stage="download", status="download_failed"),
            _make_issue(
                line_no=5,
                stage="resolve",
                status="unresolved",
                input="A title, with a comma",
            ),
            _make_issue(line_no=1, stage="resolve", status="ambiguous"),
        ]
        issues[2] = epqtypes.Issue(
            line_no=1,
            input="x",
            base_name="",
            stage="resolve",
            status="ambiguous",
            reason="r",
            candidates=("c1", "c2"),
        )
        # Prepare outputs.
        expected = """
        line_no,input,base_name,stage,status,reason,candidates
        1,x,,resolve,ambiguous,r,c1|c2
        5,"A title, with a comma",,resolve,unresolved,r,
        2,x,,download,download_failed,r,
        """
        # Run test.
        epqrepor.write_issues_csv(path, issues)
        # Check outputs.
        actual = hio.from_file(path).replace("\r\n", "\n")
        self.assert_equal(actual, expected, dedent=True)
