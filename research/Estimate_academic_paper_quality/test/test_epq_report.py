import dataclasses
import json
import logging
import os

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


def _make_paper_issue(base_name: str, stage: str, status: str) -> epqtypes.Issue:
    """
    Build the issue of a paper on line 1.
    """
    return epqtypes.Issue(
        line_no=1,
        input="x",
        base_name=base_name,
        stage=stage,
        status=status,
        reason="r",
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
        Test a file with no header: the first row is an entry, and a title with
        commas stays whole, quoted or not.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "in.csv")
        txt = (
            "Lazy Prices (John Campbell; Xin Zheng)\n"
            "\n"
            "Lazy Prices (Cohen, Malloy)\n"
            '"Lazy Prices (Cohen, Malloy)"\n'
            "2001.00001\n"
        )
        hio.to_file(path, txt)
        # Prepare outputs.
        expected = [
            (1, "Lazy Prices (John Campbell; Xin Zheng)"),
            (2, "Lazy Prices (Cohen, Malloy)"),
            (3, "Lazy Prices (Cohen, Malloy)"),
            (4, "2001.00001"),
        ]
        # Run test.
        actual = epqrepor.read_input_csv(path)
        # Check outputs.
        self.assertEqual(actual, expected)

    def test3(self) -> None:
        """
        Test a header `input` in any case is skipped, and an empty file is empty.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "in.csv")
        hio.to_file(path, "Input\nLazy Prices\n")
        empty_path = os.path.join(self.get_scratch_space(), "empty.csv")
        hio.to_file(empty_path, "")
        # Run test.
        actual = epqrepor.read_input_csv(path)
        actual_empty = epqrepor.read_input_csv(empty_path)
        # Check outputs.
        self.assertEqual(actual, [(1, "Lazy Prices")])
        self.assertEqual(actual_empty, [])


# #############################################################################
# Test_papers_json
# #############################################################################


class Test_papers_json(hunitest.TestCase):
    """
    Test `epq_report.write_papers()` and `epq_report.read_papers()`.
    """

    def test1(self) -> None:
        """
        Test papers are read back equal, with the lists of different sizes.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "papers.json")
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
        epqrepor.write_papers(path, papers)
        actual = epqrepor.read_papers(path)
        # Check outputs.
        self.assertEqual(actual, papers)

    def test2(self) -> None:
        """
        Test a separator, a quote, and a non-ASCII char in an item survive.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "papers.json")
        paper = dataclasses.replace(
            epqteuti.make_paper(),
            authors=("Smith | Jones", 'O"Brien', "Gu\u00e9rin"),
            aliases=("a|b",),
        )
        # Run test.
        epqrepor.write_papers(path, [paper])
        actual = epqrepor.read_papers(path)
        # Check outputs.
        self.assertEqual(actual, [paper])

    def test3(self) -> None:
        """
        Test the content of the papers file: a list is a JSON array.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "papers.json")
        paper = epqteuti.make_paper(line_nos=(3, 17))
        # Run test.
        epqrepor.write_papers(path, [paper])
        # Check outputs.
        data = json.loads(hio.from_file(path))
        self.assertEqual(data["schema_version"], 2)
        self.assertEqual(len(data["papers"]), 1)
        item = data["papers"][0]
        self.assertEqual(item["line_nos"], [3, 17])
        self.assertEqual(
            item["authors"], ["Cohen, Lauren", "Malloy, Christopher"]
        )
        self.assertEqual(item["year"], 2020)

    def test4(self) -> None:
        """
        Test a papers file with another schema version is rejected.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "papers.json")
        hio.to_file(path, '{"schema_version": 0, "papers": []}')
        # Run test.
        with self.assertRaises(AssertionError) as cm:
            epqrepor.read_papers(path)
        # Check outputs.
        self.assertIn("old schema", str(cm.exception))
        self.assertIn("resolve_papers.py", str(cm.exception))


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
# Test_report_json
# #############################################################################


class Test_report_json(hunitest.TestCase):
    """
    Test `build_report_rows()` and `write_report()`.
    """

    def test1(self) -> None:
        """
        Test a paper with no features file is a failed row, and the order is
        kept.
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
            [paper1, paper2, paper3], features, ratings, []
        )
        # Check outputs.
        actual = [(row["base_name"], row["status"]) for row in rows]
        self.assertEqual(actual, [("b1", "ok"), ("b2", "failed"), ("b3", "ok")])
        self.assertIsNone(rows[1]["paper_quality"])
        self.assertIsNone(rows[1]["issue"])

    def test2(self) -> None:
        """
        Test the content of `report.json`, with `null` for a missing score and
        a quote in the reference that JSON escapes.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "report.json")
        paper = epqteuti.make_paper(
            reference='2020, Cohen et al., "Lazy, Prices" (???)',
            line_nos=(3, 17),
        )
        features = {paper.base_name: epqteuti.make_paper_features()}
        ratings = {paper.base_name: epqteuti.make_quality()}
        rows = epqrepor.build_report_rows([paper], features, ratings, [])
        # Prepare outputs.
        expected = r"""
        {
          "schema_version": 3,
          "rows": [
            {
              "line_nos": [
                3,
                17
              ],
              "status": "ok",
              "reference": "2020, Cohen et al., \"Lazy, Prices\" (???)",
              "paper_quality": 4,
              "performance": "Sharpe 1.4, out of sample",
              "recency": "2020",
              "asset_class": "US equities",
              "theoretical_domain": "unknown",
              "mechanism": "associational",
              "page_length_score": 3,
              "affiliation_score": 5,
              "citations_score": null,
              "base_name": "2020.Cohen_et_al.Lazy_Prices",
              "issue": null
            }
          ]
        }
        """
        # Run test.
        epqrepor.write_report(path, rows)
        # Check outputs.
        actual = hio.from_file(path)
        self.assert_equal(actual, expected, dedent=True)
        # The file is valid JSON and a score is a number, not a string.
        data = json.loads(actual)
        self.assertEqual(data["rows"][0]["paper_quality"], 4)
        self.assertIsNone(data["rows"][0]["citations_score"])

    def test3(self) -> None:
        """
        Test a paper with a features file and no rating file is a failed row
        with the issue and no values.
        """
        # Prepare inputs.
        paper1 = epqteuti.make_paper(base_name="b1")
        paper2 = epqteuti.make_paper(base_name="b2")
        features = {
            "b1": epqteuti.make_paper_features(base_name="b1"),
            "b2": epqteuti.make_paper_features(base_name="b2"),
        }
        ratings = {"b2": epqteuti.make_quality()}
        issues = [_make_paper_issue("b1", "rate", "rating_missing")]
        # Run test.
        rows = epqrepor.build_report_rows(
            [paper1, paper2], features, ratings, issues
        )
        # Check outputs.
        actual = [(row["base_name"], row["status"]) for row in rows]
        self.assertEqual(actual, [("b1", "failed"), ("b2", "ok")])
        self.assertIsNone(rows[0]["performance"])
        self.assertEqual(rows[0]["issue"]["status"], "rating_missing")

    def test4(self) -> None:
        """
        Test an unresolved line is a row with no paper, after the papers.
        """
        # Prepare inputs.
        paper = epqteuti.make_paper(base_name="b1")
        features = {"b1": epqteuti.make_paper_features(base_name="b1")}
        ratings = {"b1": epqteuti.make_quality()}
        issues = [
            _make_issue(
                line_no=9,
                stage="resolve",
                status="unresolved",
                input="B title, with a comma",
            ),
            epqtypes.Issue(
                line_no=4,
                input="A title",
                base_name="",
                stage="resolve",
                status="ambiguous",
                reason="2 different papers match the title",
                candidates=("c1", "c2"),
            ),
        ]
        # Run test.
        rows = epqrepor.build_report_rows([paper], features, ratings, issues)
        # Check outputs.
        actual = [
            (row["line_nos"], row["status"], row["reference"], row["base_name"])
            for row in rows
        ]
        expected = [
            ([1], "ok", paper.reference, "b1"),
            ([4], "failed", "A title", None),
            ([9], "failed", "B title, with a comma", None),
        ]
        self.assertEqual(actual, expected)
        # The row has no value and has the issue, with the candidates as a list.
        self.assertIsNone(rows[1]["paper_quality"])
        expected_issue = {
            "stage": "resolve",
            "status": "ambiguous",
            "reason": "2 different papers match the title",
            "candidates": ["c1", "c2"],
        }
        self.assertEqual(rows[1]["issue"], expected_issue)

    def test5(self) -> None:
        """
        Test a paper with values and an issue is a failed row with both.
        """
        # Prepare inputs.
        paper = epqteuti.make_paper(base_name="b1")
        features = {"b1": epqteuti.make_paper_features(base_name="b1")}
        ratings = {"b1": epqteuti.make_quality()}
        issues = [_make_paper_issue("b1", "download", "download_failed")]
        # Run test.
        rows = epqrepor.build_report_rows([paper], features, ratings, issues)
        # Check outputs.
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "failed")
        self.assertEqual(rows[0]["paper_quality"], 4)
        self.assertEqual(rows[0]["issue"]["stage"], "download")

    def test6(self) -> None:
        """
        Test the issue of the earliest stage is the one in the row of a paper.
        """
        # Prepare inputs.
        paper = epqteuti.make_paper(base_name="b1")
        issues = [
            _make_paper_issue("b1", "rate", "rating_failed"),
            _make_paper_issue("b1", "download", "download_failed"),
        ]
        # Run test.
        rows = epqrepor.build_report_rows([paper], {}, {}, issues)
        # Check outputs.
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["issue"]["status"], "download_failed")

    def test7(self) -> None:
        """
        Test an issue of a paper that is not in the papers file is its own row.
        """
        # Prepare inputs.
        issues = [_make_paper_issue("gone", "download", "download_failed")]
        # Run test.
        rows = epqrepor.build_report_rows([], {}, {}, issues)
        # Check outputs.
        actual = [
            (row["line_nos"], row["status"], row["base_name"]) for row in rows
        ]
        self.assertEqual(actual, [([1], "failed", "gone")])
