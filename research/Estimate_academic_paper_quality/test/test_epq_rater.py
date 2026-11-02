import datetime
import logging
import re
from typing import List, Optional, Tuple

import helpers.hgit as hgit
import helpers.hio as hio
import helpers.hunit_test as hunitest

import research.Estimate_academic_paper_quality.epq_rater as epqrater
import research.Estimate_academic_paper_quality.epq_types as epqtypes

_LOG = logging.getLogger(__name__)

_TODAY = datetime.date(2026, 10, 28)


def _make_tiers() -> epqtypes.AffiliationTiers:
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
    return epqrater.parse_tiers(data)


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


# #############################################################################
# Test_score_page_length
# #############################################################################


class Test_score_page_length(hunitest.TestCase):
    """
    Test `epq_rater.score_page_length()`.
    """

    def helper(self, n_pages: Optional[int], expected: str) -> None:
        """
        Test helper for `score_page_length()`.

        :param n_pages: number of pages
        :param expected: expected `value:score`
        """
        # Run test.
        signal = epqrater.score_page_length(n_pages)
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
# Test_score_affiliation
# #############################################################################


class Test_score_affiliation(hunitest.TestCase):
    """
    Test `epq_rater.score_affiliation()`.
    """

    def helper(
        self, affiliations: List[epqtypes.Affiliation], expected: str
    ) -> None:
        """
        Test helper for `score_affiliation()`.

        :param affiliations: affiliations of the authors
        :param expected: expected `value:score`
        """
        # Run test.
        signal = epqrater.score_affiliation(tuple(affiliations), _make_tiers())
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
# Test_score_citations
# #############################################################################


class Test_score_citations(hunitest.TestCase):
    """
    Test `epq_rater.score_citations()`.
    """

    def helper(self, count: Optional[int], year: int, expected: str) -> None:
        """
        Test helper for `score_citations()`.

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
        signal = epqrater.score_citations(citations, year, _TODAY)
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
            epqrater.score_citations(
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
# Test_rate_paper
# #############################################################################


class Test_rate_paper(hunitest.TestCase):
    """
    Test `epq_rater.rate_paper()`.
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
        Test helper for `rate_paper()`.

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
        inputs = epqrater.RatingInputs(
            n_pages=n_pages,
            affiliations=(_make_affiliation(institution),),
            citations=citations,
            year=year,
            today=_TODAY,
        )
        # Run test.
        quality = epqrater.rate_paper(inputs, _make_tiers())
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
# Test_parse_tiers
# #############################################################################


class Test_parse_tiers(hunitest.TestCase):
    """
    Test `epq_rater.parse_tiers()`.
    """

    def test1(self) -> None:
        """
        Test an empty list in the YAML file gives empty tiers.
        """
        # Prepare inputs.
        data = {"as_of": "", "source": "", "tier_5": [], "tier_4": None}
        # Run test.
        tiers = epqrater.parse_tiers(data)
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
    Test that the score tables of `epq_rater` equal the ones of `paper.rules.md`.
    """

    def test1(self) -> None:
        """
        Test the page length bands.
        """
        # Prepare inputs.
        pairs = _parse_rule_table("Page Length", "Pages")
        # Run test.
        actual = [
            (lower, epqrater.score_page_length(lower).score)
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
            epqrater.score_page_length(lower - 1).score
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
                (lower, epqrater.score_citations(citations, 2000, _TODAY).score)
            )
        # Check outputs.
        self.assertEqual(len(pairs), 5)
        self.assertEqual(actual, pairs)
