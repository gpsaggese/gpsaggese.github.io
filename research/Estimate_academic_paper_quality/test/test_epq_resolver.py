import logging
import os
from typing import List, Tuple

import helpers.hcache_simple as hcacsimp
import helpers.hio as hio
import helpers.hunit_test as hunitest
import pytest

import research.Estimate_academic_paper_quality.epq_resolver as epqresol
import research.Estimate_academic_paper_quality.epq_types as epqtypes
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
    Base of the tests that call `parse_line()`.

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
# Test_join_wrapped_lines
# #############################################################################


class Test_join_wrapped_lines(hunitest.TestCase):
    """
    Test `epq_resolver.join_wrapped_lines()`.
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
        actual = epqresol.join_wrapped_lines(raw_lines)
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
        actual = epqresol.join_wrapped_lines(raw_lines)
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
        actual = epqresol.join_wrapped_lines(raw_lines)
        # Check outputs.
        self.assertEqual(actual, expected)


# #############################################################################
# Test_parse_line
# #############################################################################


class Test_parse_line(_NoCacheTestCase):
    """
    Test `epq_resolver.parse_line()`.
    """

    def helper(self, text: str, expected: str) -> None:
        """
        Test helper for `parse_line()`.

        :param text: text of the line
        :param expected: expected `kind|key|hints`
        """
        # Run test.
        input_line = epqresol.parse_line(3, text)
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
            epqresol.parse_line(1, text).key  # type: ignore[union-attr]
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
            epqresol.parse_line(1, text)
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
        input_line = epqresol.parse_line(1, path)
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
        actual = [epqresol.parse_line(1, t) for t in ["", "   ", "# a", "// b"]]
        # Check outputs.
        self.assertEqual(actual, [None] * 4)

    def test13(self) -> None:
        """
        Test the real input file parses with no exception and one kind per line.
        """
        # Prepare inputs.
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "paper_list.md",
        )
        raw_lines = hio.from_file(path).split("\n")
        # Run test.
        input_lines = epqresol.parse_lines(raw_lines)
        # Check outputs.
        kinds = {line.kind for line in input_lines}
        self.assertEqual(kinds, {"title"})
        self.assertGreater(len(input_lines), 10)
        # A wrapped bullet is one line with its authors.
        first = input_lines[0]
        self.assertEqual(
            first.author_hints, ("John Campbell", "Xin Zheng", "Dexin Zhou")
        )


# #############################################################################
# Test_title_similarity
# #############################################################################


class Test_title_similarity(hunitest.TestCase):
    """
    Test `epq_resolver.title_similarity()`.
    """

    def test1(self) -> None:
        """
        Test titles that differ by case, accents, and punctuation are equal.
        """
        # Run test.
        actual = epqresol.title_similarity(
            "Schölkopf: A Test!", "scholkopf a test"
        )
        # Check outputs.
        self.assertEqual(actual, 1.0)

    def test2(self) -> None:
        """
        Test a partial title is a prefix at a word boundary.
        """
        # Run test.
        actual = epqresol.title_similarity("Lazy", "Lazy Prices")
        # Check outputs.
        self.assertEqual(actual, 0.95)

    def test3(self) -> None:
        """
        Test a prefix inside a word is not a prefix.
        """
        # Run test.
        actual = epqresol.title_similarity("Lazy Pri", "Lazy Prices")
        # Check outputs.
        self.assertLess(actual, 0.95)
        self.assertGreater(actual, 0.8)

    def test4(self) -> None:
        """
        Test an empty title has no similarity.
        """
        self.assertEqual(epqresol.title_similarity("", "Lazy Prices"), 0.0)


# #############################################################################
# Test_match_candidates
# #############################################################################


class Test_match_candidates(hunitest.TestCase):
    """
    Test `epq_resolver.match_candidates()`.
    """

    def helper(
        self,
        query: str,
        hints: Tuple[str, ...],
        candidates: List[epqtypes.BibRecord],
        expected: str,
    ) -> None:
        """
        Test helper for `match_candidates()`.

        :param query: title of the line
        :param hints: authors in parentheses
        :param candidates: records found by title
        :param expected: expected `status|n records in group|n candidates`
        """
        # Run test.
        match = epqresol.match_candidates(query, hints, candidates)
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
# Test_build_paper
# #############################################################################


class Test_build_paper(hunitest.TestCase):
    """
    Test `epq_resolver.build_paper()`.
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
        paper = epqresol.build_paper(
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
        paper = epqresol.build_paper(group, input_line, unpaywall_url="")
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
            epqresol.build_paper(
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
        paper = epqresol.build_paper(group, _make_input_line(), unpaywall_url="")
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
            epqresol.build_paper(
                group, _make_input_line(), unpaywall_url=""
            ).year
            for group in [(arxiv, posted, journal), (arxiv, posted), (arxiv,)]
        ]
        # Check outputs.
        self.assertEqual(actual, [2021, 2020, 2019])


# #############################################################################
# Test_merge_papers_and_sort_papers
# #############################################################################


class Test_merge_papers_and_sort_papers(hunitest.TestCase):
    """
    Test `merge_papers()`, `sort_papers()`, and `finalize_papers()`.
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
        merged = epqresol.merge_papers([paper1, paper2])
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
        merged = epqresol.merge_papers([paper1, paper2])
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
        merged = epqresol.merge_papers([paper1, paper2])
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
            (p.year, p.authors[0], p.title) for p in epqresol.sort_papers(papers)
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
# Test_resolve_lines
# #############################################################################


class Test_resolve_lines(_NoCacheTestCase):
    """
    Test `epq_resolver.resolve_lines()` with canned records.
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
        result = epqresol.resolve_lines(raw_lines, sources)
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
        result = epqresol.resolve_lines([url], sources)
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
        result = epqresol.resolve_lines([url], sources)
        # Check outputs.
        self.assertEqual(len(result.papers), 1)

    def test4(self) -> None:
        """
        Test a URL line for a page with no paper is unresolved.
        """
        # Prepare inputs.
        sources = epqteuti.FakeBibSources()
        # Run test.
        result = epqresol.resolve_lines(["https://nowhere.example/x"], sources)
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
        result = epqresol.resolve_lines([path], sources)
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
        result = epqresol.resolve_lines(["10.1111/jofi.12885"], sources)
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
        result = epqresol.resolve_lines(["/no/such/file.pdf"], sources)
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
        result = epqresol.resolve_lines(["Lazy Prices"], sources)
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
            epqresol.resolve_lines(["Lazy Prices"], _Sources())
        # Check outputs.
        self.assertIn("HTTP 503", str(cm.exception))

    def test10(self) -> None:
        """
        Test the authors are added to the Crossref query but not to the other.
        """
        # Prepare inputs.
        sources = epqteuti.FakeBibSources()
        # Run test.
        epqresol.resolve_lines(
            ["Lazy Prices (Lauren Cohen; Chris Malloy)"], sources
        )
        # Check outputs.
        expected = [
            "search_s2 Lazy Prices",
            "search_crossref Lazy Prices Lauren Cohen Chris Malloy",
        ]
        self.assertEqual(sources.calls, expected)
