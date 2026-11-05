import dataclasses
import datetime
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

import helpers.hcache_simple as hcacsimp
import helpers.hgit as hgit
import helpers.hio as hio
import helpers.hunit_test as hunitest

import research.Estimate_academic_paper_quality.epq_bib_sources as epqbibso
import research.Estimate_academic_paper_quality.epq_paths as epqpaths
import research.Estimate_academic_paper_quality.epq_report as epqrepor
import research.Estimate_academic_paper_quality.epq_types as epqtypes
import research.Estimate_academic_paper_quality.report_papers as repopapers
import research.Estimate_academic_paper_quality.test.epq_test_utils as epqteuti

_LOG = logging.getLogger(__name__)

# DOI of the paper `Lazy Prices` of `epqteuti.make_paper()`.
_DOI = "10.1111/jofi.12885"

_TODAY = datetime.date(2026, 10, 28)


def _make_rating_tiers() -> epqtypes.AffiliationTiers:
    """
    Build a small tier list; the names are only fixtures of the test.
    """
    data = {
        "as_of": "2026-10-28",
        "source": "test fixture",
        "tier_5": [
            {"name": "Test Institute of Technology", "aliases": ["TIT"]},
            {"name": "Test University, Berkeley"},
        ],
        "tier_4": [{"name": "Test University"}],
    }
    return repopapers._parse_tiers(data)


def _make_affiliation(
    institution: str, kind: str = "university"
) -> epqtypes.Affiliation:
    """
    Build the affiliation of an author.
    """
    evidence = epqtypes.Evidence(
        quote=institution, location="byline, p. 1", verified=True
    )
    return epqtypes.Affiliation(
        author="A. Author", institution=institution, kind=kind, evidence=evidence
    )


def _make_citations() -> Dict[str, epqtypes.CitationCount]:
    """
    Build the citation counts of the paper `Lazy Prices`.
    """
    citation = epqtypes.CitationCount(
        count=512, source="Semantic Scholar", fetched_at="2026-10-28"
    )
    return {_DOI: citation}


def _make_tiers(*, harvard_tier_5: bool) -> epqtypes.AffiliationTiers:
    """
    Build a tier list.

    :param harvard_tier_5: if True, `Harvard University` is in the tier 5
    :return: tier list, empty if `harvard_tier_5` is False
    """
    data: Dict[str, Any] = {}
    if harvard_tier_5:
        data = {
            "as_of": "2026-10-28",
            "source": "test fixture",
            "tier_5": [{"name": "Harvard University", "aliases": ["Harvard"]}],
        }
    return repopapers._parse_tiers(data)


def _write_features(output_dir: str, base_names: List[str]) -> None:
    """
    Write the features file of papers, as `extract_features.py` does.

    :param output_dir: dir of all the outputs
    :param base_names: base names of the papers
    """
    for name in base_names:
        epqrepor.write_features_file(
            epqpaths.get_features_path(output_dir, name),
            epqteuti.make_paper_features(base_name=name),
        )


class _RefreshSources(epqteuti.FakeBibSources):
    """
    Fake client of the APIs that records the flag `refresh` of each call.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.refresh_flags: List[bool] = []

    def get_citation_count(
        self,
        *,
        s2_paper_id: str,
        doi: str,
        arxiv_id: str,
        refresh: bool = False,
    ) -> Any:
        self.refresh_flags.append(refresh)
        return super().get_citation_count(
            s2_paper_id=s2_paper_id, doi=doi, arxiv_id=arxiv_id, refresh=refresh
        )


class _FailingSources(epqteuti.FakeBibSources):
    """
    Fake client of the APIs whose citation lookup fails for one paper.
    """

    def get_citation_count(
        self,
        *,
        s2_paper_id: str,
        doi: str,
        arxiv_id: str,
        refresh: bool = False,
    ) -> Any:
        if s2_paper_id == "s2bad":
            raise RuntimeError("Request to 'x' failed\nHTTP 500")
        return super().get_citation_count(
            s2_paper_id=s2_paper_id, doi=doi, arxiv_id=arxiv_id, refresh=refresh
        )


# #############################################################################
# Test__score_page_length
# #############################################################################


class Test__score_page_length(hunitest.TestCase):
    """
    Test `report_papers._score_page_length()`.
    """

    def helper(self, n_pages: Optional[int], expected: str) -> None:
        """
        Test helper for `_score_page_length()`.

        :param n_pages: number of pages
        :param expected: expected `value:score`
        """
        # Run test.
        signal = repopapers._score_page_length(n_pages)
        # Check outputs.
        actual = f"{signal.value}:{signal.score}"
        self.assert_equal(actual, expected)

    def test1(self) -> None:
        """
        Test 35 pages score 5.
        """
        self.helper(35, "35 pages:5")

    def test2(self) -> None:
        """
        Test 34 pages score 4.
        """
        self.helper(34, "34 pages:4")

    def test3(self) -> None:
        """
        Test 10 pages score 3.
        """
        self.helper(10, "10 pages:3")

    def test4(self) -> None:
        """
        Test 5 pages score 2.
        """
        self.helper(5, "5 pages:2")

    def test5(self) -> None:
        """
        Test 4 pages score 1.
        """
        self.helper(4, "4 pages:1")

    def test6(self) -> None:
        """
        Test an unreadable PDF is unknown.
        """
        self.helper(None, "unknown:None")


# #############################################################################
# Test__score_affiliation
# #############################################################################


class Test__score_affiliation(hunitest.TestCase):
    """
    Test `report_papers._score_affiliation()`.
    """

    def helper(
        self, affiliations: List[epqtypes.Affiliation], expected: str
    ) -> None:
        """
        Test helper for `_score_affiliation()`.

        :param affiliations: affiliations of the authors
        :param expected: expected `value:score`
        """
        # Run test.
        signal = repopapers._score_affiliation(
            tuple(affiliations), _make_rating_tiers()
        )
        # Check outputs.
        actual = f"{signal.value}:{signal.score}"
        self.assert_equal(actual, expected)

    def test1(self) -> None:
        """
        Test an alias of a tier 5 entry scores 5.
        """
        # Prepare inputs.
        affiliations = [_make_affiliation("TIT Sloan School of Management")]
        # Prepare outputs.
        expected = "Test Institute of Technology (tier list):5"
        # Run test.
        self.helper(affiliations, expected)

    def test2(self) -> None:
        """
        Test the longest entry wins over a shorter one in the name.
        """
        # Prepare inputs.
        affiliations = [_make_affiliation("The Test University, Berkeley")]
        # Prepare outputs.
        expected = "Test University, Berkeley (tier list):5"
        # Run test.
        self.helper(affiliations, expected)

    def test3(self) -> None:
        """
        Test an alias does not match inside another word.
        """
        # Prepare inputs.
        affiliations = [_make_affiliation("Unlimited Capital Ltd", "company")]
        # Prepare outputs.
        expected = "Unlimited Capital Ltd (company):2"
        # Run test.
        self.helper(affiliations, expected)

    def test4(self) -> None:
        """
        Test an institution out of the list scores by its kind.
        """
        # Prepare inputs.
        affiliations = [_make_affiliation("Other College", "university")]
        # Prepare outputs.
        expected = "Other College (university):3"
        # Run test.
        self.helper(affiliations, expected)

    def test5(self) -> None:
        """
        Test the best author wins and a central bank scores 5.
        """
        # Prepare inputs.
        affiliations = [
            _make_affiliation("Other College", "university"),
            _make_affiliation("Some Reserve Bank", "central_bank"),
        ]
        # Prepare outputs.
        expected = "Some Reserve Bank (central_bank):5"
        # Run test.
        self.helper(affiliations, expected)

    def test6(self) -> None:
        """
        Test a paper with no affiliation scores 1.
        """
        self.helper([], "no affiliation listed:1")


# #############################################################################
# Test__score_citations
# #############################################################################


class Test__score_citations(hunitest.TestCase):
    """
    Test `report_papers._score_citations()`.
    """

    def helper(self, count: Optional[int], year: int, expected: str) -> None:
        """
        Test helper for `_score_citations()`.

        :param count: citation count, `None` if unknown
        :param year: publication year
        :param expected: expected `score|note`
        """
        # Prepare inputs.
        citations = None
        if count is not None:
            citations = epqtypes.CitationCount(
                count=count, source="Semantic Scholar", fetched_at="2026-10-28"
            )
        # Run test.
        signal = repopapers._score_citations(citations, year, _TODAY)
        # Check outputs.
        actual = f"{signal.score}|{signal.note}"
        self.assert_equal(actual, expected)

    def test1(self) -> None:
        """
        Test 40 citations after 3 years score 2.
        """
        self.helper(40, 2023, "2|")

    def test2(self) -> None:
        """
        Test the edges of the bands.
        """
        # Prepare inputs.
        counts = [1000, 999, 300, 299, 50, 49, 10, 9]
        # Prepare outputs.
        expected = [5, 4, 4, 3, 3, 2, 2, 1]
        # Run test.
        actual = [
            repopapers._score_citations(
                epqtypes.CitationCount(c, "Semantic Scholar", "2026-10-28"),
                2000,
                _TODAY,
            ).score
            for c in counts
        ]
        # Check outputs.
        self.assertEqual(actual, expected)

    def test3(self) -> None:
        """
        Test a paper of this year has its citations annualized.
        """
        # Prepare outputs.
        expected = "3|annualized: 30 * 2 / 0.5 years"
        # Run test.
        self.helper(30, 2026, expected)

    def test4(self) -> None:
        """
        Test a paper of last year has its citations annualized.
        """
        # Prepare outputs.
        expected = "2|annualized: 15 * 2 / 1 years"
        # Run test.
        self.helper(15, 2025, expected)

    def test5(self) -> None:
        """
        Test a paper that Semantic Scholar does not have is unknown.
        """
        # Prepare outputs.
        expected = "None|Semantic Scholar does not have the paper"
        # Run test.
        self.helper(None, 2020, expected)


# #############################################################################
# Test__compute_rating
# #############################################################################


class Test__compute_rating(hunitest.TestCase):
    """
    Test `report_papers._compute_rating()`.
    """

    def helper(
        self,
        n_pages: Optional[int],
        institution: str,
        count: Optional[int],
        year: int,
        expected: Optional[int],
    ) -> None:
        """
        Test helper for `_compute_rating()`.

        :param n_pages: number of pages
        :param institution: institution of the only author
        :param count: citation count
        :param year: publication year
        :param expected: expected rating
        """
        # Prepare inputs.
        citations = None
        if count is not None:
            citations = epqtypes.CitationCount(
                count=count, source="Semantic Scholar", fetched_at="2026-10-28"
            )
        inputs = repopapers._RatingInputs(
            n_pages=n_pages,
            affiliations=(_make_affiliation(institution),),
            citations=citations,
            year=year,
            today=_TODAY,
        )
        # Run test.
        quality = repopapers._compute_rating(inputs, _make_rating_tiers())
        # Check outputs.
        self.assertEqual(quality.rating, expected)

    def test1(self) -> None:
        """
        Test the example of the rules: scores 3, 5, 2 give a mean of 3.33 and 3.
        """
        self.helper(12, "TIT", 40, 2023, 3)

    def test2(self) -> None:
        """
        Test a mean of 2.5 is rounded up and not to even.
        """
        # The scores are 2 (5 pages), 3 (other university), unknown citations.
        self.helper(5, "Other College", None, 2020, 3)

    def test3(self) -> None:
        """
        Test an unknown signal is left out of the mean.
        """
        # Only the affiliation is known, and it scores 5.
        self.helper(None, "TIT", None, 2020, 5)

    def test4(self) -> None:
        """
        Test a mean of 1.67 is rounded to 2.
        """
        # The scores are 1 (3 pages), 3 (other university), 1 (5 citations).
        self.helper(3, "Other College", 5, 2000, 2)


# #############################################################################
# Test__parse_tiers
# #############################################################################


class Test__parse_tiers(hunitest.TestCase):
    """
    Test `report_papers._parse_tiers()`.
    """

    def test1(self) -> None:
        """
        Test an empty list in the YAML file gives empty tiers.
        """
        # Prepare inputs.
        data = {"as_of": "", "source": "", "tier_5": [], "tier_4": None}
        # Run test.
        tiers = repopapers._parse_tiers(data)
        # Check outputs.
        self.assertEqual(tiers.tier_5, ())
        self.assertEqual(tiers.tier_4, ())


# #############################################################################
# Test_score_tables_match_rules
# #############################################################################


def _parse_rule_table(section: str, header: str) -> List[Tuple[int, int]]:
    """
    Read the pairs `(lower bound, score)` of a score table of the rules file.

    :param section: name of the `###` section, e.g., `Page Length`
    :param header: name of the second column, e.g., `Pages`
    :return: pairs from the best score to the worst
    """
    path = hgit.find_file_in_git_tree("paper.rules.md")
    text = hio.from_file(path)
    body = text.split(f"### {section}")[1].split("###")[0]
    pairs = []
    for line in body.split("\n"):
        match = re.match(r"^\s*\|\s*([1-5])\s*\|\s*(.+?)\s*\|\s*$", line)
        if match and match.group(2) != header:
            score = int(match.group(1))
            text_cell = match.group(2)
            # `fewer than 5` is the band that starts at 0.
            lower = (
                0
                if "fewer" in text_cell
                else int(re.findall(r"\d+", text_cell)[0])
            )
            pairs.append((lower, score))
    return pairs


class Test_score_tables_match_rules(hunitest.TestCase):
    """
    Test that the score tables of `report_papers` equal the ones of `paper.rules.md`.
    """

    def test1(self) -> None:
        """
        Test the page length bands.
        """
        # Prepare inputs.
        pairs = _parse_rule_table("Page Length", "Pages")
        # Run test.
        actual = [
            (lower, repopapers._score_page_length(lower).score)
            for lower, _ in pairs
        ]
        # Check outputs.
        self.assertEqual(len(pairs), 5)
        self.assertEqual(actual, pairs)

    def test2(self) -> None:
        """
        Test the page length bands just below each lower bound.
        """
        # Prepare inputs.
        pairs = _parse_rule_table("Page Length", "Pages")
        # Run test.
        actual = [
            repopapers._score_page_length(lower - 1).score
            for lower, _ in pairs
            if lower > 0
        ]
        # Check outputs.
        expected = [score - 1 for lower, score in pairs if lower > 0]
        self.assertEqual(actual, expected)

    def test3(self) -> None:
        """
        Test the citation bands.
        """
        # Prepare inputs.
        pairs = _parse_rule_table("Citations", "Citations")
        # Run test.
        actual = []
        for lower, _ in pairs:
            citations = epqtypes.CitationCount(lower, "Semantic Scholar", "x")
            actual.append(
                (
                    lower,
                    repopapers._score_citations(citations, 2000, _TODAY).score,
                )
            )
        # Check outputs.
        self.assertEqual(len(pairs), 5)
        self.assertEqual(actual, pairs)


# #############################################################################
# Test__rate
# #############################################################################


class Test__rate(hunitest.TestCase):
    """
    Test the stage `rate` of `report_papers.py`.
    """

    def rate(
        self,
        papers: List[epqtypes.Paper],
        output_dir: str,
        sources: epqbibso.BibSources,
        *,
        harvard_tier_5: bool = False,
        refresh_citations: bool = False,
        dry_run: bool = False,
    ) -> List[epqtypes.Issue]:
        """
        Run the stage `rate` with a tier list and a fixed date.

        :param papers: papers to rate
        :param output_dir: dir of all the outputs
        :param sources: client of the APIs
        :param harvard_tier_5: if True, `Harvard University` is in the tier 5
        :param refresh_citations: if True, ask for fresh citation counts
        :param dry_run: if True, only log what the stage would do
        :return: issues of the stage
        """
        return repopapers._rate(
            papers,
            output_dir,
            sources=sources,
            tiers=_make_tiers(harvard_tier_5=harvard_tier_5),
            today=_TODAY,
            refresh_citations=refresh_citations,
            dry_run=dry_run,
        )

    def test1(self) -> None:
        """
        Test the rating file has the three signals of the paper.
        """
        # Prepare inputs.
        output_dir = os.path.join(self.get_scratch_space(), "out")
        _write_features(output_dir, ["p1"])
        paper = epqteuti.make_paper(base_name="p1")
        sources = epqteuti.FakeBibSources(citations=_make_citations())
        # Run test.
        issues = self.rate([paper], output_dir, sources)
        # Check outputs.
        self.assertEqual(issues, [])
        quality = epqrepor.read_rating_file(
            epqpaths.get_rating_path(output_dir, "p1")
        )
        # An empty tier list scores a university by its kind.
        self.assertEqual(quality.affiliation.score, 3)
        self.assertEqual(quality.citations.score, 4)
        self.assertEqual(quality.page_length.value, "12 pages")
        self.assertIsNotNone(quality.rating)

    def test2(self) -> None:
        """
        Test a new tier list changes the rating with no new extraction.
        """
        # Prepare inputs.
        output_dir = os.path.join(self.get_scratch_space(), "out")
        _write_features(output_dir, ["p1"])
        features_path = epqpaths.get_features_path(output_dir, "p1")
        features_before = hio.from_file(features_path)
        paper = epqteuti.make_paper(base_name="p1")
        sources = epqteuti.FakeBibSources(citations=_make_citations())
        rating_path = epqpaths.get_rating_path(output_dir, "p1")
        # Run test.
        self.rate([paper], output_dir, sources)
        score_before = epqrepor.read_rating_file(rating_path).affiliation.score
        self.rate([paper], output_dir, sources, harvard_tier_5=True)
        score_after = epqrepor.read_rating_file(rating_path).affiliation.score
        # Check outputs.
        self.assertEqual((score_before, score_after), (3, 5))
        self.assertEqual(hio.from_file(features_path), features_before)

    def test3(self) -> None:
        """
        Test a lookup that fails is an issue and the next paper is rated.
        """
        # Prepare inputs.
        output_dir = os.path.join(self.get_scratch_space(), "out")
        _write_features(output_dir, ["p1", "p2"])
        bad = epqteuti.make_paper(base_name="p1")
        bad = dataclasses.replace(bad, s2_paper_id="s2bad")
        good = epqteuti.make_paper(base_name="p2")
        sources = _FailingSources(citations=_make_citations())
        # Run test.
        issues = self.rate([bad, good], output_dir, sources)
        # Check outputs.
        actual = [(i.base_name, i.stage, i.status, i.reason) for i in issues]
        self.assertEqual(actual, [("p1", "rate", "rating_failed", "HTTP 500")])
        self.assertFalse(
            os.path.exists(epqpaths.get_rating_path(output_dir, "p1"))
        )
        self.assertTrue(
            os.path.isfile(epqpaths.get_rating_path(output_dir, "p2"))
        )
        # The issues go to a file, for the stage `report`.
        path = epqpaths.get_issues_path(output_dir, "rate")
        self.assertEqual(epqrepor.read_issues(path), issues)
        expected_summary = """
        Stage `rate` summary
          Papers in input:        2
          Processed in this run:  2
          Completed successfully: 1
          Failed:                 1
          Completed in total:     1 of 2
          Output files written in this run:
            'out/papers/<BASE>.rating.json': 1 file
          Issues file: 'out/04-rate.issues.json' (1 issue)
            rating_failed: 1
          Summary file: 'out/04-rate.summary.txt'
        """
        actual_summary = epqteuti.read_summary(output_dir, "rate")
        self.assert_equal(actual_summary, expected_summary, dedent=True)

    def test4(self) -> None:
        """
        Test `refresh_citations` asks the client for a fresh count.
        """
        # Prepare inputs.
        output_dir = os.path.join(self.get_scratch_space(), "out")
        _write_features(output_dir, ["p1"])
        paper = epqteuti.make_paper(base_name="p1")
        sources = _RefreshSources(citations=_make_citations())
        # Run test.
        self.rate([paper], output_dir, sources)
        self.rate([paper], output_dir, sources, refresh_citations=True)
        # Check outputs.
        self.assertEqual(sources.refresh_flags, [False, True])

    def test5(self) -> None:
        """
        Test a paper with no features file is skipped with no issue.
        """
        # Prepare inputs.
        output_dir = os.path.join(self.get_scratch_space(), "out")
        paper = epqteuti.make_paper(base_name="p1")
        sources = epqteuti.FakeBibSources(citations=_make_citations())
        # Run test.
        issues = self.rate([paper], output_dir, sources)
        # Check outputs.
        self.assertEqual(issues, [])
        self.assertEqual(sources.calls, [])
        self.assertFalse(
            os.path.exists(epqpaths.get_rating_path(output_dir, "p1"))
        )
        actual_summary = epqteuti.read_summary(output_dir, "rate")
        self.assertIn("Skipped, no input:      1\n", actual_summary)
        self.assertIn("Completed in total:     0 of 1\n", actual_summary)

    def test6(self) -> None:
        """
        Test `dry_run` makes no lookup and writes no file.
        """
        # Prepare inputs.
        output_dir = os.path.join(self.get_scratch_space(), "out")
        _write_features(output_dir, ["p1"])
        paper = epqteuti.make_paper(base_name="p1")
        sources = epqteuti.FakeBibSources(citations=_make_citations())
        # Run test.
        self.rate([paper], output_dir, sources, dry_run=True)
        # Check outputs.
        self.assertEqual(sources.calls, [])
        self.assertFalse(
            os.path.exists(epqpaths.get_rating_path(output_dir, "p1"))
        )
        self.assertFalse(
            os.path.exists(epqpaths.get_issues_path(output_dir, "rate"))
        )
        self.assertFalse(
            os.path.exists(epqpaths.get_summary_path(output_dir, "rate"))
        )


# #############################################################################
# Test__report
# #############################################################################


class Test__report(hunitest.TestCase):
    """
    Test the stage `report` of `report_papers.py`.
    """

    def report(
        self,
        papers: List[epqtypes.Paper],
        output_dir: str,
        *,
        dry_run: bool = False,
    ) -> None:
        """
        Run the stage `report`.

        :param papers: papers of the papers file
        :param output_dir: dir of all the outputs
        :param dry_run: if True, only log what the stage would do
        """
        repopapers._report(papers, output_dir, dry_run=dry_run)

    def test1(self) -> None:
        """
        Test a paper with a rating is an ok row and a paper without one is a
        failed row with its issue.
        """
        # Prepare inputs.
        output_dir = os.path.join(self.get_scratch_space(), "out")
        _write_features(output_dir, ["p1", "p2"])
        epqrepor.write_rating_file(
            epqpaths.get_rating_path(output_dir, "p1"), epqteuti.make_quality()
        )
        papers = [
            epqteuti.make_paper(base_name="p1"),
            epqteuti.make_paper(base_name="p2"),
        ]
        # Run test.
        self.report(papers, output_dir)
        # Check outputs.
        report = json.loads(hio.from_file(epqpaths.get_report_path(output_dir)))
        actual = [(r["base_name"], r["status"]) for r in report["rows"]]
        self.assertEqual(actual, [("p1", "ok"), ("p2", "failed")])
        issue = report["rows"][1]["issue"]
        self.assertEqual(
            (issue["stage"], issue["status"]), ("rate", "rating_missing")
        )
        # The summary counts the paper with no rating as failed.
        expected_summary = """
        Stage `report` summary
          Rows in input:          2
          Processed in this run:  2
          Completed successfully: 1
          Failed:                 1
          Completed in total:     1 of 2
          Output files written in this run:
            'out/04-report.references.md': 1 file
            'out/04-report.report.json': 1 file
          Issues file: 'out/04-report.report.json' (1 issue)
            rate.rating_missing: 1
          Summary file: 'out/04-report.summary.txt'
        """
        actual_summary = epqteuti.read_summary(output_dir, "report")
        self.assert_equal(actual_summary, expected_summary, dedent=True)

    def test2(self) -> None:
        """
        Test a features file with an old schema is a failed row in the report.
        """
        # Prepare inputs.
        output_dir = os.path.join(self.get_scratch_space(), "out")
        paper = epqteuti.make_paper(base_name="p1")
        hio.to_file(
            epqpaths.get_features_path(output_dir, "p1"),
            '{"schema_version": 99}',
        )
        # Run test.
        self.report([paper], output_dir)
        # Check outputs.
        report = json.loads(hio.from_file(epqpaths.get_report_path(output_dir)))
        self.assertEqual(len(report["rows"]), 1)
        row = report["rows"][0]
        self.assertEqual(row["status"], "failed")
        self.assertIsNone(row["paper_quality"])
        self.assertEqual(row["issue"]["status"], "extraction_failed")
        self.assertIn("--force_extract", row["issue"]["reason"])

    def test3(self) -> None:
        """
        Test a paper that failed to rate has one issue in its row, not two.
        """
        # Prepare inputs.
        output_dir = os.path.join(self.get_scratch_space(), "out")
        _write_features(output_dir, ["p1"])
        paper = epqteuti.make_paper(base_name="p1")
        epqrepor.write_issues(
            epqpaths.get_issues_path(output_dir, "rate"),
            [
                epqtypes.Issue(
                    line_no=1,
                    input=paper.reference,
                    base_name="p1",
                    stage="rate",
                    status="rating_failed",
                    reason="HTTP 500",
                    candidates=(),
                )
            ],
        )
        # Run test.
        self.report([paper], output_dir)
        # Check outputs.
        report = json.loads(hio.from_file(epqpaths.get_report_path(output_dir)))
        actual = [(r["base_name"], r["issue"]["status"]) for r in report["rows"]]
        self.assertEqual(actual, [("p1", "rating_failed")])
        # The summary counts the issue once.
        self.assertIn(
            "Issues file: 'out/04-report.report.json' (1 issue)",
            epqteuti.read_summary(output_dir, "report"),
        )

    def test4(self) -> None:
        """
        Test the references keep the order of the papers file.
        """
        # Prepare inputs.
        output_dir = os.path.join(self.get_scratch_space(), "out")
        papers = [
            epqteuti.make_paper(base_name="p1", reference="ref one"),
            epqteuti.make_paper(base_name="p2", reference="ref two"),
        ]
        # Run test.
        self.report(papers, output_dir)
        # Check outputs.
        actual = hio.from_file(epqpaths.get_references_path(output_dir))
        self.assertEqual(actual, "- ref one\n- ref two\n")

    def test5(self) -> None:
        """
        Test `dry_run` writes no file.
        """
        # Prepare inputs.
        output_dir = os.path.join(self.get_scratch_space(), "out")
        papers = [epqteuti.make_paper(base_name="p1")]
        # Run test.
        self.report(papers, output_dir, dry_run=True)
        # Check outputs.
        self.assertFalse(os.path.exists(output_dir))


# #############################################################################
# Test_report_papers_py
# #############################################################################


class Test_report_papers_py(hunitest.TestCase):
    """
    Test the command line of `report_papers.py`.
    """

    def run_main(self, argv: List[str], sources: Any) -> None:
        """
        Run `_main()` with the given flags and a fake client of the APIs.

        :param argv: flags of the script
        :param sources: client of the APIs that replaces the real one
        """
        old_cache_dir = hcacsimp.get_cache_dir()
        self.addCleanup(hcacsimp.set_cache_dir, old_cache_dir)
        parser = repopapers._parse()
        # Use an empty tier list, so that a score does not change when the user
        # edits `affiliation_tiers.yaml`.
        empty_tiers = repopapers._parse_tiers({})
        with (
            mock.patch.object(sys, "argv", ["report_papers.py"] + argv),
            mock.patch.object(epqbibso, "BibSources", return_value=sources),
            mock.patch.object(
                repopapers, "_load_tiers", return_value=empty_tiers
            ),
            mock.patch.object(
                hgit, "find_file_in_git_tree", return_value="tiers.yaml"
            ),
        ):
            repopapers._main(parser)

    def make_papers_file(self, papers: List[epqtypes.Paper]) -> str:
        """
        Write a papers file in the scratch space.

        :param papers: papers of the file
        :return: path of the file
        """
        path = os.path.join(self.get_scratch_space(), "papers.json")
        epqrepor.write_papers(path, papers)
        return path

    def test1(self) -> None:
        """
        Test the stage `rate` and the stage `report` write the report.
        """
        # Prepare inputs.
        papers_file = self.make_papers_file(
            [epqteuti.make_paper(base_name="p1")]
        )
        output_dir = os.path.join(self.get_scratch_space(), "out")
        _write_features(output_dir, ["p1"])
        sources = epqteuti.FakeBibSources(citations=_make_citations())
        argv = [f"--input={papers_file}", f"--output_dir={output_dir}"]
        # Run test.
        self.run_main(argv, sources)
        # Check outputs.
        report = json.loads(hio.from_file(epqpaths.get_report_path(output_dir)))
        self.assertEqual(len(report["rows"]), 1)
        row = report["rows"][0]
        self.assertEqual(row["base_name"], "p1")
        self.assertEqual(
            (row["affiliation_score"], row["citations_score"]), (3, 4)
        )
        self.assertEqual(
            hio.from_file(epqpaths.get_references_path(output_dir)),
            '- 2020, Cohen et al., "Lazy Prices" (https://arxiv.org/pdf/2001.00001)\n',
        )

    def test2(self) -> None:
        """
        Test the stage `report` alone writes the reference list.
        """
        # Prepare inputs.
        papers_file = self.make_papers_file([epqteuti.make_paper()])
        output_dir = os.path.join(self.get_scratch_space(), "out")
        argv = [
            f"--input={papers_file}",
            f"--output_dir={output_dir}",
            "--clear_actions",
            "--action=report",
        ]
        sources = epqteuti.FakeBibSources()
        # Run test.
        self.run_main(argv, sources)
        # Check outputs.
        self.assert_equal(
            hio.from_file(epqpaths.get_references_path(output_dir)),
            '- 2020, Cohen et al., "Lazy Prices" (https://arxiv.org/pdf/2001.00001)\n',
        )
        self.assertEqual(sources.calls, [])
        self.assertFalse(
            os.path.exists(epqpaths.get_issues_path(output_dir, "download"))
        )

    def test3(self) -> None:
        """
        Test the issues of `resolve_papers.py` are failed rows in the report.
        """
        # Prepare inputs.
        papers_file = self.make_papers_file([epqteuti.make_paper()])
        output_dir = os.path.join(self.get_scratch_space(), "out")
        issue = epqtypes.Issue(
            line_no=2,
            input="Unknown Title",
            base_name="",
            stage="resolve",
            status="unresolved",
            reason="no source has a paper titled 'Unknown Title'",
            candidates=(),
        )
        epqrepor.write_issues(
            epqpaths.get_issues_path(output_dir, "resolve"), [issue]
        )
        argv = [
            f"--input={papers_file}",
            f"--output_dir={output_dir}",
            "--clear_actions",
            "--action=report",
        ]
        # Prepare outputs: the paper has no features file, so it is a failed
        # row with no issue, and the unresolved line is the row after it.
        expected = [
            {
                "line_nos": [1],
                "status": "failed",
                "reference": epqteuti.make_paper().reference,
                "base_name": "2020.Cohen_et_al.Lazy_Prices",
                "issue": None,
            },
            {
                "line_nos": [2],
                "status": "failed",
                "reference": "Unknown Title",
                "base_name": None,
                "issue": {
                    "stage": "resolve",
                    "status": "unresolved",
                    "reason": "no source has a paper titled 'Unknown Title'",
                    "candidates": [],
                },
            },
        ]
        # Run test.
        self.run_main(argv, epqteuti.FakeBibSources())
        # Check outputs.
        report = json.loads(hio.from_file(epqpaths.get_report_path(output_dir)))
        keys = ["line_nos", "status", "reference", "base_name", "issue"]
        actual = [{k: row[k] for k in keys} for row in report["rows"]]
        self.assertEqual(actual, expected)
        # The pipeline writes no separate issues file for the report.
        self.assertFalse(
            os.path.exists(os.path.join(output_dir, "04-report.issues.json"))
        )

    def test4(self) -> None:
        """
        Test the candidates of an issue are a list in the report.
        """
        # Prepare inputs.
        papers_file = self.make_papers_file([epqteuti.make_paper()])
        output_dir = os.path.join(self.get_scratch_space(), "out")
        epqrepor.write_issues(
            epqpaths.get_issues_path(output_dir, "resolve"),
            [
                epqtypes.Issue(
                    line_no=7,
                    input="Momentum",
                    base_name="",
                    stage="resolve",
                    status="ambiguous",
                    reason="2 different papers match the title",
                    candidates=("c1", "c2"),
                )
            ],
        )
        argv = [
            f"--input={papers_file}",
            f"--output_dir={output_dir}",
            "--clear_actions",
            "--action=report",
        ]
        # Run test.
        self.run_main(argv, epqteuti.FakeBibSources())
        # Check outputs.
        report = json.loads(hio.from_file(epqpaths.get_report_path(output_dir)))
        row = report["rows"][-1]
        self.assertEqual(
            (row["line_nos"], row["reference"], row["issue"]["status"]),
            ([7], "Momentum", "ambiguous"),
        )
        self.assertEqual(row["issue"]["candidates"], ["c1", "c2"])

    def test5(self) -> None:
        """
        Test a missing papers file stops the run before any lookup.
        """
        # Prepare inputs.
        scratch = self.get_scratch_space()
        argv = [
            f"--input={scratch}/missing.csv",
            f"--output_dir={scratch}/out",
        ]
        # Run test.
        with self.assertRaises(AssertionError) as cm:
            self.run_main(argv, epqteuti.FakeBibSources())
        # Check outputs.
        self.assertIn("Run resolve_papers.py", str(cm.exception))

    def test6(self) -> None:
        """
        Test the default input is the papers file in the output dir.
        """
        # Prepare inputs.
        output_dir = os.path.join(self.get_scratch_space(), "out")
        epqrepor.write_papers(
            epqpaths.get_papers_path(output_dir),
            [epqteuti.make_paper(base_name="p1")],
        )
        argv = [
            f"--output_dir={output_dir}",
            "--clear_actions",
            "--action=report",
        ]
        # Run test: there is no resolve issues file, which is only a warning.
        self.run_main(argv, epqteuti.FakeBibSources())
        # Check outputs.
        actual = hio.from_file(epqpaths.get_references_path(output_dir))
        self.assertIn("Lazy Prices", actual)
        self.assertFalse(
            os.path.exists(epqpaths.get_issues_path(output_dir, "resolve"))
        )

    def test7(self) -> None:
        """
        Test `--dry_run` makes no lookup and writes no file.
        """
        # Prepare inputs.
        papers_file = self.make_papers_file(
            [epqteuti.make_paper(base_name="p1")]
        )
        output_dir = os.path.join(self.get_scratch_space(), "out")
        _write_features(output_dir, ["p1"])
        sources = epqteuti.FakeBibSources(citations=_make_citations())
        argv = [
            f"--input={papers_file}",
            f"--output_dir={output_dir}",
            "--dry_run",
        ]
        # Run test.
        self.run_main(argv, sources)
        # Check outputs.
        self.assertEqual(sources.calls, [])
        self.assertFalse(
            os.path.exists(epqpaths.get_rating_path(output_dir, "p1"))
        )
        self.assertFalse(os.path.exists(epqpaths.get_report_path(output_dir)))
