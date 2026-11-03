import csv
import logging
import os
import sys
from typing import Any, List, Tuple
from unittest import mock

import helpers.hcache_simple as hcacsimp
import helpers.hio as hio
import helpers.hunit_test as hunitest
import pytest

import research.Estimate_academic_paper_quality.epq_bib_sources as epqbibso
import research.Estimate_academic_paper_quality.epq_paths as epqpaths
import research.Estimate_academic_paper_quality.epq_report as epqrepor
import research.Estimate_academic_paper_quality.epq_types as epqtypes
import research.Estimate_academic_paper_quality.resolve_papers as resopapers
import research.Estimate_academic_paper_quality.test.epq_test_utils as epqteuti

_LOG = logging.getLogger(__name__)


def _make_input_line(
    *,
    kind: str = "title",
    text: str = "Lazy Prices",
    key: str = "Lazy Prices",
    line_no: int = 1,
) -> epqtypes.InputLine:
    """
    Build a classified line.
    """
    return epqtypes.InputLine(
        line_no=line_no, text=text, kind=kind, key=key, author_hints=()
    )


class _NoCacheTestCase(hunitest.TestCase):
    """
    Base of the tests that call `_parse_line()`.

    `download_utils.detect_doi()` caches its answers on disk: the cache is off,
    so that the tests write nothing outside the scratch space.
    """

    @pytest.fixture(autouse=True)
    def setup_teardown_test(self):
        """
        Disable the cache of the helpers during each test.
        """
        hcacsimp.enable_caching(False)
        yield
        hcacsimp.enable_caching(True)


# #############################################################################
# Test__join_wrapped_lines
# #############################################################################


class Test__join_wrapped_lines(hunitest.TestCase):
    """
    Test `resolve_papers._join_wrapped_lines()`.
    """

    def test1(self) -> None:
        """
        Test a wrapped bullet is joined, and the line is the first one.
        """
        # Prepare inputs.
        raw_lines = [
            "- Lazy Prices",
            "  (Lauren Cohen; Christopher Malloy)",
            "- Momentum",
        ]
        # Prepare outputs.
        expected = [
            (1, "Lazy Prices (Lauren Cohen; Christopher Malloy)"),
            (3, "Momentum"),
        ]
        # Run test.
        actual = resopapers._join_wrapped_lines(raw_lines)
        # Check outputs.
        self.assertEqual(actual, expected)

    def test2(self) -> None:
        """
        Test blank lines and comments end an entry and are skipped.
        """
        # Prepare inputs.
        raw_lines = ["# Papers", "- A", "", "  stray indent", "// note", "B"]
        # Prepare outputs.
        expected = [(2, "A"), (4, "stray indent"), (6, "B")]
        # Run test.
        actual = resopapers._join_wrapped_lines(raw_lines)
        # Check outputs.
        self.assertEqual(actual, expected)

    def test3(self) -> None:
        """
        Test lines with no bullet are one entry each.
        """
        # Prepare inputs.
        raw_lines = ["1706.03762", "10.1111/jofi.12885"]
        # Prepare outputs.
        expected = [(1, "1706.03762"), (2, "10.1111/jofi.12885")]
        # Run test.
        actual = resopapers._join_wrapped_lines(raw_lines)
        # Check outputs.
        self.assertEqual(actual, expected)


# #############################################################################
# Test__parse_line
# #############################################################################


class Test__parse_line(_NoCacheTestCase):
    """
    Test `resolve_papers._parse_line()`.
    """

    def helper(self, text: str, expected: str) -> None:
        """
        Test helper for `_parse_line()`.

        :param text: text of the line
        :param expected: expected `kind|key|hints`
        """
        # Run test.
        input_line = resopapers._parse_line(3, text)
        # Check outputs.
        assert input_line is not None
        actual = f"{input_line.kind}|{input_line.key}|{input_line.author_hints}"
        self.assert_equal(actual, expected)

    def test1(self) -> None:
        """
        Test an arXiv URL, with the version removed.
        """
        self.helper("https://arxiv.org/abs/1706.03762v5", "arxiv|1706.03762|()")

    def test2(self) -> None:
        """
        Test an arXiv PDF URL, a prefixed id, and a bare id.
        """
        # Run test.
        kinds = [
            resopapers._parse_line(1, text).key  # type: ignore[union-attr]
            for text in [
                "https://arxiv.org/pdf/1706.03762.pdf",
                "arXiv:1706.03762",
                "1706.03762",
            ]
        ]
        # Check outputs.
        self.assertEqual(kinds, ["1706.03762"] * 3)

    def test3(self) -> None:
        """
        Test a DOI is not taken as an arXiv id even if it contains one.
        """
        self.helper(
            "https://doi.org/10.1016/j.jfineco.2019.12345",
            "doi|10.1016/j.jfineco.2019.12345|()",
        )

    def test4(self) -> None:
        """
        Test a bare DOI.
        """
        self.helper("10.1111/JOFI.12885", "doi|10.1111/jofi.12885|()")

    def test5(self) -> None:
        """
        Test an SSRN URL and an SSRN DOI are `ssrn`, not `doi`.
        """
        # Run test.
        keys = [
            resopapers._parse_line(1, text)
            for text in [
                "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5277078",
                "10.2139/ssrn.5277078",
            ]
        ]
        # Check outputs.
        self.assertEqual(
            [(k.kind, k.key) for k in keys], [("ssrn", "5277078")] * 2
        )

    def test6(self) -> None:
        """
        Test a publisher URL.
        """
        self.helper(
            "https://example.com/paper.html",
            "url|https://example.com/paper.html|()",
        )

    def test7(self) -> None:
        """
        Test a title with the authors in parentheses.
        """
        self.helper(
            "Lazy Prices (Lauren Cohen; Christopher Malloy)",
            "title|Lazy Prices|('Lauren Cohen', 'Christopher Malloy')",
        )

    def test8(self) -> None:
        """
        Test a title with no authors, and a bullet marker.
        """
        self.helper("- Lazy Prices", "title|Lazy Prices|()")

    def test9(self) -> None:
        """
        Test a title whose authors are separated by commas.
        """
        self.helper(
            "Good Carry (Geert Bekaert, George Panayotov)",
            "title|Good Carry|('Geert Bekaert', 'George Panayotov')",
        )

    def test10(self) -> None:
        """
        Test an existing local PDF.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "paper.pdf")
        epqteuti.create_pdf(path, ["x"])
        # Run test.
        input_line = resopapers._parse_line(1, path)
        # Check outputs.
        assert input_line is not None
        self.assertEqual((input_line.kind, input_line.key), ("local_pdf", path))

    def test11(self) -> None:
        """
        Test a path to a PDF that does not exist is a title.
        """
        self.helper("/no/such/file.pdf", "title|/no/such/file.pdf|()")

    def test12(self) -> None:
        """
        Test an empty line and the two kinds of comment give `None`.
        """
        # Run test.
        actual = [
            resopapers._parse_line(1, t) for t in ["", "   ", "# a", "// b"]
        ]
        # Check outputs.
        self.assertEqual(actual, [None] * 4)


# #############################################################################
# Test__title_similarity
# #############################################################################


class Test__title_similarity(hunitest.TestCase):
    """
    Test `resolve_papers._title_similarity()`.
    """

    def test1(self) -> None:
        """
        Test titles that differ by case, accents, and punctuation are equal.
        """
        # Run test.
        actual = resopapers._title_similarity(
            "Schölkopf: A Test!", "scholkopf a test"
        )
        # Check outputs.
        self.assertEqual(actual, 1.0)

    def test2(self) -> None:
        """
        Test a partial title is a prefix at a word boundary.
        """
        # Run test.
        actual = resopapers._title_similarity("Lazy", "Lazy Prices")
        # Check outputs.
        self.assertEqual(actual, 0.95)

    def test3(self) -> None:
        """
        Test a prefix inside a word is not a prefix.
        """
        # Run test.
        actual = resopapers._title_similarity("Lazy Pri", "Lazy Prices")
        # Check outputs.
        self.assertLess(actual, 0.95)
        self.assertGreater(actual, 0.8)

    def test4(self) -> None:
        """
        Test an empty title has no similarity.
        """
        self.assertEqual(resopapers._title_similarity("", "Lazy Prices"), 0.0)


# #############################################################################
# Test__match_candidates
# #############################################################################


class Test__match_candidates(hunitest.TestCase):
    """
    Test `resolve_papers._match_candidates()`.
    """

    def helper(
        self,
        query: str,
        hints: Tuple[str, ...],
        candidates: List[epqtypes.BibRecord],
        expected: str,
    ) -> None:
        """
        Test helper for `_match_candidates()`.

        :param query: title of the line
        :param hints: authors in parentheses
        :param candidates: records found by title
        :param expected: expected `status|n records in group|n candidates`
        """
        # Run test.
        match = resopapers._match_candidates(query, hints, candidates)
        # Check outputs.
        actual = f"{match.status}|{len(match.group)}|{len(match.candidates)}"
        self.assert_equal(actual, expected)

    def test1(self) -> None:
        """
        Test the records of two sources of one paper are one match.
        """
        # Prepare inputs.
        candidates = [
            epqteuti.make_record(),
            epqteuti.make_record(source="crossref", s2_paper_id=""),
        ]
        # Run test.
        self.helper("Lazy Prices", (), candidates, "matched|2|0")

    def test2(self) -> None:
        """
        Test a partial title matches.
        """
        self.helper("Lazy", (), [epqteuti.make_record()], "matched|1|0")

    def test3(self) -> None:
        """
        Test a candidate below the threshold is dropped.
        """
        # Prepare inputs.
        candidates = [epqteuti.make_record(title="Completely Different Paper")]
        # Run test.
        self.helper("Lazy Prices", (), candidates, "unresolved|0|0")

    def test4(self) -> None:
        """
        Test two papers with the same title are ambiguous.
        """
        # Prepare inputs.
        candidates = [
            epqteuti.make_record(
                title="Momentum",
                authors=("Jegadeesh, Narasimhan",),
                doi="10.1/a",
                s2_paper_id="a",
            ),
            epqteuti.make_record(
                title="Momentum",
                authors=("Carhart, Mark",),
                doi="10.1/b",
                s2_paper_id="b",
            ),
        ]
        # Run test.
        self.helper("Momentum", (), candidates, "ambiguous|0|2")

    def test5(self) -> None:
        """
        Test the author hints pick one of two papers with the same title.
        """
        # Prepare inputs.
        candidates = [
            epqteuti.make_record(
                title="Momentum",
                authors=("Jegadeesh, Narasimhan",),
                doi="10.1/a",
                s2_paper_id="a",
            ),
            epqteuti.make_record(
                title="Momentum",
                authors=("Carhart, Mark",),
                doi="10.1/b",
                s2_paper_id="b",
            ),
        ]
        # Run test.
        self.helper("Momentum", ("Mark Carhart",), candidates, "matched|1|0")

    def test6(self) -> None:
        """
        Test hints that match no author leave the line unresolved.
        """
        # Prepare inputs.
        candidates = [epqteuti.make_record()]
        # Run test.
        self.helper(
            "Lazy Prices", ("Someone Else",), candidates, "unresolved|0|0"
        )

    def test7(self) -> None:
        """
        Test one exact title wins over a longer title with the same prefix.
        """
        # Prepare inputs.
        candidates = [
            epqteuti.make_record(
                title="Momentum and Reversal",
                authors=("Doe, J",),
                doi="10.1/a",
                s2_paper_id="a",
            ),
            epqteuti.make_record(
                title="Momentum",
                authors=("Roe, R",),
                doi="10.1/b",
                s2_paper_id="b",
            ),
        ]
        # Run test.
        self.helper("Momentum", (), candidates, "matched|1|0")

    def test8(self) -> None:
        """
        Test a preprint and its journal version with other DOIs are one paper.
        """
        # Prepare inputs.
        candidates = [
            epqteuti.make_record(doi="10.1111/jofi.12885", s2_paper_id="a"),
            epqteuti.make_record(
                source="arxiv",
                doi="",
                arxiv_id="2001.00001",
                s2_paper_id="",
                year=2019,
                record_type="posted-content",
            ),
        ]
        # Run test.
        self.helper("Lazy Prices", (), candidates, "matched|2|0")

    def test9(self) -> None:
        """
        Test a family name with several words matches its last word.
        """
        # Prepare inputs.
        candidates = [
            epqteuti.make_record(
                title="Advances", authors=("Lopez de Prado, Marcos",)
            )
        ]
        # Run test.
        self.helper(
            "Advances", ("Marcos Lopez de Prado",), candidates, "matched|1|0"
        )


# #############################################################################
# Test__build_paper
# #############################################################################


class Test__build_paper(hunitest.TestCase):
    """
    Test `resolve_papers._build_paper()`.
    """

    def test1(self) -> None:
        """
        Test the download inputs are in the order of the design.
        """
        # Prepare inputs.
        group = (
            epqteuti.make_record(
                arxiv_id="2001.00001",
                ssrn_id="123",
                pdf_urls=("https://oa.example/lazy.pdf",),
            ),
        )
        input_line = _make_input_line(
            kind="url",
            text="https://publisher.example/lazy",
            key="https://publisher.example/lazy",
        )
        # Prepare outputs.
        expected = (
            "https://publisher.example/lazy|"
            "https://arxiv.org/pdf/2001.00001|"
            "https://oa.example/lazy.pdf|"
            "https://unpaywall.example/lazy.pdf|"
            "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=123"
        )
        # Run test.
        paper = resopapers._build_paper(
            group, input_line, unpaywall_url="https://unpaywall.example/lazy.pdf"
        )
        # Check outputs.
        self.assert_equal("|".join(paper.download_inputs), expected)
        self.assert_equal(
            paper.reference,
            '2020, Cohen et al., "Lazy Prices" (https://publisher.example/lazy)',
        )

    def test2(self) -> None:
        """
        Test a local PDF is the first input but not the link of the reference.
        """
        # Prepare inputs.
        group = (epqteuti.make_record(arxiv_id="2001.00001"),)
        input_line = _make_input_line(
            kind="local_pdf", text="/data/lazy.pdf", key="/data/lazy.pdf"
        )
        # Run test.
        paper = resopapers._build_paper(group, input_line, unpaywall_url="")
        # Check outputs.
        self.assertEqual(paper.download_inputs[0], "/data/lazy.pdf")
        self.assertIn("(https://arxiv.org/pdf/2001.00001)", paper.reference)

    def test3(self) -> None:
        """
        Test the mark of a paper with no public link, with and without a DOI.
        """
        # Prepare inputs.
        with_doi = (epqteuti.make_record(),)
        no_doi = (epqteuti.make_record(doi=""),)
        # Prepare outputs.
        expected = (
            '2020, Cohen et al., "Lazy Prices" (https://doi.org/10.1111/jofi.12885 ???)|'
            '2020, Cohen et al., "Lazy Prices" (???)'
        )
        # Run test.
        actual = "|".join(
            resopapers._build_paper(
                g, _make_input_line(), unpaywall_url=""
            ).reference
            for g in [with_doi, no_doi]
        )
        # Check outputs.
        self.assert_equal(actual, expected)

    def test4(self) -> None:
        """
        Test the base name and the facts come from the canonical record.
        """
        # Prepare inputs.
        group = (
            epqteuti.make_record(
                source="arxiv", year=2019, title="Lazy Prices (preprint)"
            ),
            epqteuti.make_record(),
        )
        # Run test.
        paper = resopapers._build_paper(
            group, _make_input_line(), unpaywall_url=""
        )
        # Check outputs.
        self.assert_equal(paper.base_name, "2020.Cohen_et_al.Lazy_Prices")
        self.assertEqual((paper.source, paper.year), ("semantic_scholar", 2020))

    def test5(self) -> None:
        """
        Test the canonical record is Crossref `journal-article`, then any
        Crossref, then arXiv, when Semantic Scholar has no record.
        """
        # Prepare inputs.
        journal = epqteuti.make_record(source="crossref", year=2021, title="J")
        posted = epqteuti.make_record(
            source="crossref", year=2020, record_type="posted-content", title="P"
        )
        arxiv = epqteuti.make_record(source="arxiv", year=2019, title="A")
        # Run test.
        actual = [
            resopapers._build_paper(
                group, _make_input_line(), unpaywall_url=""
            ).year
            for group in [(arxiv, posted, journal), (arxiv, posted), (arxiv,)]
        ]
        # Check outputs.
        self.assertEqual(actual, [2021, 2020, 2019])


# #############################################################################
# Test__merge_papers_and_sort_papers
# #############################################################################


class Test__merge_papers_and_sort_papers(hunitest.TestCase):
    """
    Test `_merge_papers()`, `_sort_papers()`, and `_finalize_papers()`.
    """

    def test1(self) -> None:
        """
        Test two lines of one paper give one paper with a local PDF first.
        """
        # Prepare inputs.
        paper1 = epqteuti.make_paper(line_nos=(1,))
        paper2 = epqteuti.make_paper(
            line_nos=(4,),
            download_inputs=("/data/lazy.pdf", "https://x.example/y.pdf"),
        )
        # Run test.
        merged = resopapers._merge_papers([paper1, paper2])
        # Check outputs.
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].line_nos, (1, 4))
        self.assertEqual(
            merged[0].download_inputs,
            (
                "/data/lazy.pdf",
                "https://arxiv.org/pdf/2001.00001",
                "https://x.example/y.pdf",
            ),
        )

    def test2(self) -> None:
        """
        Test two papers that share only a DOI alias are merged.
        """
        # Prepare inputs.
        paper1 = epqteuti.make_paper(line_nos=(1,))
        paper2 = epqteuti.make_paper(line_nos=(2,))
        paper2 = epqteuti.make_paper(line_nos=(2,))
        paper2 = paper2.__class__(
            **{**paper2.__dict__, "key": "other title|cohen"}
        )
        # Run test.
        merged = resopapers._merge_papers([paper1, paper2])
        # Check outputs.
        self.assertEqual(len(merged), 1)

    def test3(self) -> None:
        """
        Test two different papers are kept.
        """
        # Prepare inputs.
        paper1 = epqteuti.make_paper()
        paper2 = epqteuti.make_paper(line_nos=(2,))
        paper2 = paper2.__class__(
            **{**paper2.__dict__, "key": "other|doe", "aliases": ()}
        )
        # Run test.
        merged = resopapers._merge_papers([paper1, paper2])
        # Check outputs.
        self.assertEqual(len(merged), 2)

    def test4(self) -> None:
        """
        Test papers are sorted by year down, then author, then title.
        """

        # Prepare inputs.
        def _paper(year: int, author: str, title: str) -> epqtypes.Paper:
            paper = epqteuti.make_paper(year=year)
            return paper.__class__(
                **{**paper.__dict__, "authors": (author,), "title": title}
            )

        papers = [
            _paper(2019, "Zed, A", "B"),
            _paper(2021, "Doe, J", "Z"),
            _paper(2019, "Abe, B", "Z"),
            _paper(2019, "Abe, B", "A"),
        ]
        # Run test.
        actual = [
            (p.year, p.authors[0], p.title)
            for p in resopapers._sort_papers(papers)
        ]
        # Check outputs.
        expected = [
            (2021, "Doe, J", "Z"),
            (2019, "Abe, B", "A"),
            (2019, "Abe, B", "Z"),
            (2019, "Zed, A", "B"),
        ]
        self.assertEqual(actual, expected)


# #############################################################################
# Test__resolve_lines
# #############################################################################


class Test__resolve_lines(_NoCacheTestCase):
    """
    Test `resolve_papers._resolve_lines()` with canned records.
    """

    def test1(self) -> None:
        """
        Test a list with each kind of outcome.
        """
        # Prepare inputs.
        lazy_s2 = epqteuti.make_record()
        lazy_cr = epqteuti.make_record(source="crossref", s2_paper_id="")
        ribeiro_s2 = epqteuti.make_record(
            title="Why Should I Trust You?",
            authors=("Ribeiro, Marco Tulio", "Singh, Sameer"),
            year=2016,
            doi="",
            arxiv_id="1602.04938",
            s2_paper_id="s2rib",
        )
        ribeiro_ax = epqteuti.make_record(
            source="arxiv",
            title="Why Should I Trust You?",
            authors=("Ribeiro, Marco Tulio", "Singh, Sameer"),
            year=2016,
            record_type="posted-content",
            doi="",
            arxiv_id="1602.04938",
            s2_paper_id="",
            pdf_urls=("https://arxiv.org/pdf/1602.04938",),
        )
        momentum1 = epqteuti.make_record(
            title="Momentum",
            authors=("Jegadeesh, Narasimhan",),
            year=1993,
            doi="",
            s2_paper_id="m1",
        )
        momentum2 = epqteuti.make_record(
            title="Momentum",
            authors=("Carhart, Mark",),
            year=1997,
            doi="",
            s2_paper_id="m2",
        )
        sources = epqteuti.FakeBibSources(
            by_id={
                "arxiv:1602.04938": ribeiro_ax,
                "s2:ARXIV:1602.04938": ribeiro_s2,
                "s2:DOI:10.1111/jofi.12885": lazy_s2,
                "crossref:10.1111/jofi.12885": lazy_cr,
            },
            by_title={
                "s2:Lazy Prices": [lazy_s2],
                "crossref:Lazy Prices": [lazy_cr],
                "s2:Momentum": [momentum1, momentum2],
            },
        )
        raw_lines = [
            "# Papers",
            "- Lazy Prices",
            "  (Lauren Cohen; Christopher Malloy)",
            "- 1602.04938",
            "- 10.1111/jofi.12885",
            "- Unknown Paper Title Xyz",
            "- Momentum",
        ]
        # Prepare outputs.
        expected = """
        2020, Cohen et al., "Lazy Prices" (https://doi.org/10.1111/jofi.12885 ???) lines=(2, 5)
        2016, Ribeiro et al., "Why Should I Trust You?" (https://arxiv.org/pdf/1602.04938) lines=(4,)
        6 unresolved no source has a paper titled 'Unknown Paper Title Xyz' []
        7 ambiguous 2 different papers match the title ['1993, Jegadeesh, "Momentum" (???)', '1997, Carhart, "Momentum" (???)']
        """
        # Run test.
        result = resopapers._resolve_lines(raw_lines, sources)
        # Check outputs.
        lines = [f"{p.reference} lines={p.line_nos}" for p in result.papers]
        lines += [
            f"{i.line_no} {i.status} {i.reason} {list(i.candidates)}"
            for i in result.issues
        ]
        self.assert_equal("\n".join(lines), expected, dedent=True)

    def test2(self) -> None:
        """
        Test a URL line that gives a DOI in its meta tags.
        """
        # Prepare inputs.
        url = "https://publisher.example/lazy"
        meta = epqtypes.UrlMeta(
            pdf_path="", doi="10.1111/jofi.12885", title="", authors=()
        )
        sources = epqteuti.FakeBibSources(
            by_id={"s2:DOI:10.1111/jofi.12885": epqteuti.make_record()},
            url_meta={url: meta},
        )
        # Run test.
        result = resopapers._resolve_lines([url], sources)
        # Check outputs.
        self.assertEqual(len(result.papers), 1)
        self.assertEqual(result.papers[0].download_inputs[0], url)

    def test3(self) -> None:
        """
        Test a URL line with only a title and authors in its meta tags.
        """
        # Prepare inputs.
        url = "https://publisher.example/lazy"
        meta = epqtypes.UrlMeta(
            pdf_path="", doi="", title="Lazy Prices", authors=("Cohen, Lauren",)
        )
        sources = epqteuti.FakeBibSources(
            by_title={"s2:Lazy Prices": [epqteuti.make_record()]},
            url_meta={url: meta},
        )
        # Run test.
        result = resopapers._resolve_lines([url], sources)
        # Check outputs.
        self.assertEqual(len(result.papers), 1)

    def test4(self) -> None:
        """
        Test a URL line for a page with no paper is unresolved.
        """
        # Prepare inputs.
        sources = epqteuti.FakeBibSources()
        # Run test.
        result = resopapers._resolve_lines(
            ["https://nowhere.example/x"], sources
        )
        # Check outputs.
        self.assertEqual(
            [(i.status, i.reason) for i in result.issues],
            [("unresolved", "the page has no DOI and no title")],
        )

    def test5(self) -> None:
        """
        Test a local PDF is resolved through its title and kept as an input.
        """
        # Prepare inputs.
        path = os.path.join(self.get_scratch_space(), "lazy.pdf")
        epqteuti.create_pdf(path, ["body"], title_text="Lazy Prices")
        sources = epqteuti.FakeBibSources(
            by_title={"s2:Lazy Prices": [epqteuti.make_record()]}
        )
        # Run test.
        result = resopapers._resolve_lines([path], sources)
        # Check outputs.
        self.assertEqual(len(result.papers), 1)
        self.assertEqual(result.papers[0].download_inputs[0], path)

    def test6(self) -> None:
        """
        Test Unpaywall is asked for the DOI of a matched paper.
        """
        # Prepare inputs.
        sources = epqteuti.FakeBibSources(
            by_id={"s2:DOI:10.1111/jofi.12885": epqteuti.make_record()},
            unpaywall={"10.1111/jofi.12885": "https://oa.example/lazy.pdf"},
        )
        # Run test.
        result = resopapers._resolve_lines(["10.1111/jofi.12885"], sources)
        # Check outputs.
        self.assertEqual(
            result.papers[0].download_inputs, ("https://oa.example/lazy.pdf",)
        )

    def test7(self) -> None:
        """
        Test a path to a PDF that does not exist is an issue with no lookup.
        """
        # Prepare inputs.
        sources = epqteuti.FakeBibSources()
        # Run test.
        result = resopapers._resolve_lines(["/no/such/file.pdf"], sources)
        # Check outputs.
        actual = [(i.status, i.reason) for i in result.issues]
        expected = [
            ("unresolved", "the file '/no/such/file.pdf' does not exist")
        ]
        self.assertEqual(actual, expected)
        self.assertEqual(sources.calls, [])

    def test8(self) -> None:
        """
        Test a source that is down is skipped when the other one answers.
        """

        # Prepare inputs.
        class _Sources(epqteuti.FakeBibSources):
            def search_s2(
                self, query: str, limit: int
            ) -> List[epqtypes.BibRecord]:
                raise RuntimeError("Request to 's2' failed: HTTP 429")

        sources = _Sources(
            by_title={
                "crossref:Lazy Prices": [
                    epqteuti.make_record(source="crossref", s2_paper_id="")
                ]
            }
        )
        # Run test.
        result = resopapers._resolve_lines(["Lazy Prices"], sources)
        # Check outputs.
        self.assertEqual(len(result.papers), 1)
        self.assertEqual(result.papers[0].source, "crossref")

    def test9(self) -> None:
        """
        Test an error is raised when every source is down.
        """

        # Prepare inputs.
        class _Sources(epqteuti.FakeBibSources):
            def search_s2(
                self, query: str, limit: int
            ) -> List[epqtypes.BibRecord]:
                raise RuntimeError("Request to 's2' failed: HTTP 429")

            def search_crossref(
                self, query: str, limit: int
            ) -> List[epqtypes.BibRecord]:
                raise RuntimeError("Request to 'crossref' failed: HTTP 503")

        # Run test.
        with self.assertRaises(RuntimeError) as cm:
            resopapers._resolve_lines(["Lazy Prices"], _Sources())
        # Check outputs.
        self.assertIn("HTTP 503", str(cm.exception))

    def test10(self) -> None:
        """
        Test the authors are added to the Crossref query but not to the other.
        """
        # Prepare inputs.
        sources = epqteuti.FakeBibSources()
        # Run test.
        resopapers._resolve_lines(
            ["Lazy Prices (Lauren Cohen; Chris Malloy)"], sources
        )
        # Check outputs.
        expected = [
            "search_s2 Lazy Prices",
            "search_crossref Lazy Prices Lauren Cohen Chris Malloy",
        ]
        self.assertEqual(sources.calls, expected)


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
            mock.patch.object(epqbibso, "BibSources", return_value=sources),
        ):
            resopapers._main(parser)

    def test1(self) -> None:
        """
        Test the papers file and the issues file of a list with a bad row.
        """
        # Prepare inputs.
        scratch = self.get_scratch_space()
        input_csv = self.make_input_csv(
            ["2001.00001", "Unknown Title", "", "# a comment", "2001.00001v2"]
        )
        output_dir = os.path.join(scratch, "out")
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
        argv = [f"--input={input_csv}", f"--output_dir={output_dir}"]
        # Prepare outputs.
        expected_papers = [
            (
                "2020.Cohen_et_al.Lazy_Prices",
                (1, 5),
                ("doi:10.1111/jofi.12885", "arxiv:2001.00001"),
                ("https://arxiv.org/pdf/2001.00001",),
            )
        ]
        expected_issues = [
            (
                2,
                "Unknown Title",
                "resolve",
                "unresolved",
                "no source has a paper titled 'Unknown Title'",
            )
        ]
        # Run test.
        self.run_main(argv, sources)
        # Check outputs.
        papers = epqrepor.read_papers(epqpaths.get_papers_path(output_dir))
        actual_papers = [
            (p.base_name, p.line_nos, p.aliases, p.download_inputs)
            for p in papers
        ]
        self.assertEqual(actual_papers, expected_papers)
        issues = epqrepor.read_issues(
            epqpaths.get_issues_path(output_dir, "resolve")
        )
        actual_issues = [
            (i.line_no, i.input, i.stage, i.status, i.reason) for i in issues
        ]
        self.assertEqual(actual_issues, expected_issues)
        # The names of the files are inferred from the output dir.
        actual_files = sorted(f for f in os.listdir(output_dir) if f != "state")
        expected_files = [
            "00-paper_list.csv",
            "01-resolve.issues.json",
            "01-resolve.paper_list.json",
        ]
        self.assertEqual(actual_files, expected_files)

    def test2(self) -> None:
        """
        Test the input CSV is copied to the output dir as `00-paper_list.csv`.
        """
        # Prepare inputs.
        scratch = self.get_scratch_space()
        input_csv = self.make_input_csv(["Unknown Title"])
        output_dir = os.path.join(scratch, "out")
        argv = [f"--input={input_csv}", f"--output_dir={output_dir}"]
        # Run test.
        self.run_main(argv, epqteuti.FakeBibSources())
        # Check outputs.
        copy_path = epqpaths.get_input_copy_path(output_dir)
        self.assertEqual(os.path.basename(copy_path), "00-paper_list.csv")
        self.assertEqual(hio.from_file(copy_path), hio.from_file(input_csv))

    def test3(self) -> None:
        """
        Test `--dry_run` makes no lookup and writes no file.
        """
        # Prepare inputs.
        scratch = self.get_scratch_space()
        input_csv = self.make_input_csv(["2001.00001", "Unknown Title"])
        output_dir = os.path.join(scratch, "out")
        sources = epqteuti.FakeBibSources()
        argv = [
            f"--input={input_csv}",
            f"--output_dir={output_dir}",
            "--dry_run",
        ]
        # Run test.
        self.run_main(argv, sources)
        # Check outputs: no lookup, no dir, and so no copy of the input.
        self.assertEqual(sources.calls, [])
        self.assertFalse(os.path.exists(output_dir))

    def test4(self) -> None:
        """
        Test an input CSV with no header: the first row is a paper.
        """
        # Prepare inputs.
        scratch = self.get_scratch_space()
        input_csv = os.path.join(scratch, "papers_in.csv")
        hio.to_file(input_csv, "2001.00001\n")
        # Run test.
        input_lines = resopapers._read_input_lines(input_csv)
        # Check outputs.
        self.assertEqual(
            [(x.line_no, x.kind, x.key) for x in input_lines],
            [(1, "arxiv", "2001.00001")],
        )

    def test5(self) -> None:
        """
        Test a missing input CSV stops the run.
        """
        # Prepare inputs.
        scratch = self.get_scratch_space()
        argv = [
            f"--input={scratch}/missing.csv",
            f"--output_dir={scratch}/out",
        ]
        # Run test.
        with self.assertRaises(AssertionError):
            self.run_main(argv, epqteuti.FakeBibSources())

    def test6(self) -> None:
        """
        Test the real input CSV parses with no exception and one kind per row.
        """
        # Prepare inputs.
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "paper_list.csv",
        )
        # Run test.
        input_lines = resopapers._read_input_lines(path)
        # Check outputs.
        kinds = {line.kind for line in input_lines}
        self.assertEqual(kinds, {"title"})
        self.assertGreater(len(input_lines), 10)
        # The authors in parentheses are the hints of the first row.
        first = input_lines[0]
        self.assertEqual(
            first.author_hints, ("John Campbell", "Xin Zheng", "Dexin Zhou")
        )

    def test7(self) -> None:
        """
        Test the error of a lookup for one line does not stop the other lines.
        """

        # Prepare inputs.
        class _Sources(epqteuti.FakeBibSources):
            def lookup_arxiv(self, arxiv_id: str) -> Any:
                if arxiv_id == "1111.11111":
                    raise RuntimeError("Request to 'x' failed: HTTP 500")
                return super().lookup_arxiv(arxiv_id)

        sources = _Sources(
            by_id={
                "arxiv:2001.00001": epqteuti.make_record(
                    source="arxiv", s2_paper_id="", arxiv_id="2001.00001"
                )
            }
        )
        scratch = self.get_scratch_space()
        input_csv = self.make_input_csv(["1111.11111", "2001.00001"])
        papers_file = os.path.join(scratch, "papers.json")
        issues_file = os.path.join(scratch, "issues.json")
        # Run test.
        result = resopapers._resolve(
            input_csv, papers_file, issues_file, sources
        )
        # Check outputs.
        self.assertEqual(len(result.papers), 1)
        self.assertEqual(len(epqrepor.read_papers(papers_file)), 1)
        self.assertEqual(len(epqrepor.read_issues(issues_file)), 1)
        actual = [(i.line_no, i.status, i.reason) for i in result.issues]
        reason = (
            "no record for arXiv id '1111.11111' "
            "(a source failed: Request to 'x' failed: HTTP 500)"
        )
        expected = [(1, "unresolved", reason)]
        self.assertEqual(actual, expected)

    def test8(self) -> None:
        """
        Test an input that is already the copy in the output dir is not an error.
        """
        # Prepare inputs.
        scratch = self.get_scratch_space()
        output_dir = os.path.join(scratch, "out")
        input_csv = self.make_input_csv(["Unknown Title"])
        hio.create_dir(output_dir, incremental=True)
        copy_path = epqpaths.get_input_copy_path(output_dir)
        hio.to_file(copy_path, hio.from_file(input_csv))
        argv = [f"--input={copy_path}", f"--output_dir={output_dir}"]
        # Run test.
        self.run_main(argv, epqteuti.FakeBibSources())
        # Check outputs.
        self.assertEqual(hio.from_file(copy_path), hio.from_file(input_csv))
        issues = epqrepor.read_issues(
            epqpaths.get_issues_path(output_dir, "resolve")
        )
        self.assertEqual(len(issues), 1)

    def test9(self) -> None:
        """
        Test the old flag `--output` is rejected, not read as `--output_dir`.
        """
        # Prepare inputs.
        scratch = self.get_scratch_space()
        input_csv = self.make_input_csv(["Unknown Title"])
        argv = [f"--input={input_csv}", f"--output={scratch}/papers.csv"]
        # Run test.
        with self.assertRaises(SystemExit):
            self.run_main(argv, epqteuti.FakeBibSources())
        # Check outputs.
        self.assertFalse(os.path.exists(os.path.join(scratch, "papers.csv")))
