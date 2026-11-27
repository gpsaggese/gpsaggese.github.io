import datetime
import json
import logging
import os
import time
from typing import Dict, List, Tuple

import helpers.hunit_test as hunitest

import research.Estimate_academic_paper_quality.epq_bib_sources as epqbibso
import research.Estimate_academic_paper_quality.test.epq_test_utils as epqteuti

_LOG = logging.getLogger(__name__)

_JSON = "application/json"

_S2_PAPER = {
    "paperId": "abc123",
    "title": "Lazy  Prices",
    "year": 2020,
    "authors": [{"name": "Lauren Cohen"}, {"name": "Christopher Malloy"}],
    "externalIds": {"DOI": "10.1111/jofi.12885", "ArXiv": "2001.00001"},
    "openAccessPdf": {"url": "https://example.org/lazy.pdf"},
    "citationCount": 512,
    "publicationTypes": ["JournalArticle"],
}

_CROSSREF_WORK = {
    "DOI": "10.1111/jofi.12885",
    "title": ["Lazy Prices"],
    "author": [
        {"family": "Cohen", "given": "Lauren"},
        {"family": "Malloy", "given": "Christopher"},
    ],
    "issued": {"date-parts": [[2020, 5]]},
    "type": "journal-article",
}

_ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1602.04938v3</id>
    <published>2016-02-16T18:55:41Z</published>
    <title>Why Should I Trust You?:
  Explaining the Predictions</title>
    <author><name>Marco Tulio Ribeiro</name></author>
    <author><name>Sameer Singh</name></author>
  </entry>
</feed>
"""


def _answer(
    status: int, body: object, content_type: str = _JSON
) -> Tuple[int, bytes, str]:
    """
    Build an answer of the local server.

    :param status: HTTP status
    :param body: JSON object, or the text or bytes to send
    :param content_type: content type of the answer
    :return: status, body, and content type
    """
    if isinstance(body, bytes):
        data = body
    elif isinstance(body, str):
        data = body.encode("utf-8")
    else:
        data = json.dumps(body).encode("utf-8")
    return status, data, content_type


class _ServerTestCase(hunitest.TestCase):
    """
    Base of the tests that need a local server and a client of the APIs.
    """

    def start(
        self, routes: Dict[str, List[Tuple[int, bytes, str]]]
    ) -> epqteuti.LocalServer:
        """
        Start a local server that stops at the end of the test.

        :param routes: queue of answers by path
        :return: server
        """
        server = epqteuti.LocalServer(routes)
        self.addCleanup(server.stop)
        return server

    def make_sources(
        self,
        server: epqteuti.LocalServer,
        *,
        email: str = "me@example.org",
        s2_api_key: str = "",
        cache_mode: str = "",
    ) -> epqbibso.BibSources:
        """
        Build a client of the APIs that talks to a local server.

        :param server: local server
        :param email: contact email
        :param s2_api_key: key of Semantic Scholar
        :param cache_mode: cache mode
        :return: client with the cache in the scratch space
        """
        return epqbibso.BibSources(
            email=email,
            s2_api_key=s2_api_key,
            cache_dir=os.path.join(self.get_scratch_space(), "cache"),
            cache_mode=cache_mode,
            settings=epqteuti.make_settings(server.url),
        )


# #############################################################################
# Test_name_to_family_first
# #############################################################################


class Test_name_to_family_first(hunitest.TestCase):
    """
    Test `epq_bib_sources.name_to_family_first()`.
    """

    def helper(self, name: str, expected: str) -> None:
        """
        Test helper for `name_to_family_first()`.

        :param name: author name
        :param expected: expected name as `Family, Given`
        """
        # Run test.
        actual = epqbibso.name_to_family_first(name)
        # Check outputs.
        self.assert_equal(actual, expected)

    def test1(self) -> None:
        """
        Test the form `Given Family`.
        """
        self.helper("Lauren Cohen", "Cohen, Lauren")

    def test2(self) -> None:
        """
        Test a suffix after a comma is dropped.
        """
        self.helper("Robert J. Jackson, Jr.", "Jackson, Robert J.")

    def test3(self) -> None:
        """
        Test the form `Family, Given` is kept.
        """
        self.helper("Cohen, Lauren", "Cohen, Lauren")

    def test4(self) -> None:
        """
        Test a single name.
        """
        self.helper("Plato", "Plato")

    def test5(self) -> None:
        """
        Test an empty name.
        """
        self.helper("", "")


# #############################################################################
# Test_get
# #############################################################################


class Test_get(_ServerTestCase):
    """
    Test the throttle, the retry, and the cache of `BibSources` through
    `lookup_s2()`.
    """

    def test1(self) -> None:
        """
        Test two `429` answers are retried until a `200`.
        """
        # Prepare inputs.
        path = "/graph/v1/paper/DOI:10.1/x"
        server = self.start(
            {
                path: [
                    _answer(429, {}),
                    _answer(429, {}),
                    _answer(200, _S2_PAPER),
                ]
            }
        )
        sources = self.make_sources(server)
        # Run test.
        record = sources.lookup_s2("DOI:10.1/x")
        # Check outputs.
        self.assertEqual(record.s2_paper_id, "abc123")
        self.assertEqual(server.counts[path], 3)

    def test2(self) -> None:
        """
        Test a `404` is a stable answer, so it is cached.
        """
        # Prepare inputs.
        path = "/graph/v1/paper/DOI:10.1/x"
        server = self.start({path: [_answer(404, {})]})
        sources = self.make_sources(server)
        # Run test.
        first = sources.lookup_s2("DOI:10.1/x")
        second = sources.lookup_s2("DOI:10.1/x")
        # Check outputs.
        self.assertIsNone(first)
        self.assertIsNone(second)
        self.assertEqual(server.counts[path], 1)

    def test3(self) -> None:
        """
        Test an error after the last retry is raised, is not cached, and the API
        is skipped for the rest of the run.
        """
        # Prepare inputs.
        path = "/graph/v1/paper/DOI:10.1/x"
        server = self.start({path: [_answer(500, {})]})
        sources = self.make_sources(server)
        # Run test.
        with self.assertRaises(RuntimeError) as cm1:
            sources.lookup_s2("DOI:10.1/x")
        with self.assertRaises(RuntimeError) as cm2:
            sources.lookup_s2("DOI:10.1/x")
        # Check outputs: 5 attempts in the first call, none in the second.
        self.assertEqual(server.counts[path], 5)
        self.assertIn("HTTP 500", str(cm1.exception))
        self.assertIn("skipped", str(cm2.exception))
        # A new client tries again, since the error was not cached.
        with self.assertRaises(RuntimeError):
            self.make_sources(server).lookup_s2("DOI:10.1/x")
        self.assertEqual(server.counts[path], 10)

    def test4(self) -> None:
        """
        Test a `403` is not retried and does not disable the API.
        """
        # Prepare inputs.
        path = "/graph/v1/paper/DOI:10.1/x"
        server = self.start({path: [_answer(403, {})]})
        sources = self.make_sources(server)
        # Run test.
        with self.assertRaises(RuntimeError):
            sources.lookup_s2("DOI:10.1/x")
        with self.assertRaises(RuntimeError):
            sources.lookup_s2("DOI:10.1/x")
        # Check outputs.
        self.assertEqual(server.counts[path], 2)

    def test5(self) -> None:
        """
        Test the key of Semantic Scholar and the email are sent.
        """
        # Prepare inputs.
        path = "/graph/v1/paper/DOI:10.1/x"
        server = self.start({path: [_answer(200, _S2_PAPER)]})
        sources = self.make_sources(server, s2_api_key="KEY")
        # Run test.
        sources.lookup_s2("DOI:10.1/x")
        # Check outputs.
        headers = server.headers[path]
        self.assertEqual(headers["x-api-key"], "KEY")
        self.assertIn("mailto:me@example.org", headers["user-agent"])

    def test6(self) -> None:
        """
        Test `HIT_CACHE_OR_ABORT` serves the cache and aborts on a miss.
        """
        # Prepare inputs.
        path = "/graph/v1/paper/DOI:10.1/x"
        server = self.start({path: [_answer(200, _S2_PAPER)]})
        self.make_sources(server).lookup_s2("DOI:10.1/x")
        offline = self.make_sources(server, cache_mode="HIT_CACHE_OR_ABORT")
        # Run test.
        record = offline.lookup_s2("DOI:10.1/x")
        # Check outputs.
        self.assertEqual(record.s2_paper_id, "abc123")
        self.assertEqual(server.counts[path], 1)
        with self.assertRaises(ValueError):
            offline.lookup_s2("DOI:10.1/other")

    def test7(self) -> None:
        """
        Test `REFRESH_CACHE` asks the server again.
        """
        # Prepare inputs.
        path = "/graph/v1/paper/DOI:10.1/x"
        server = self.start({path: [_answer(200, _S2_PAPER)]})
        self.make_sources(server).lookup_s2("DOI:10.1/x")
        refresh = self.make_sources(server, cache_mode="REFRESH_CACHE")
        # Run test.
        refresh.lookup_s2("DOI:10.1/x")
        # Check outputs.
        self.assertEqual(server.counts[path], 2)

    def test8(self) -> None:
        """
        Test `DISABLE_CACHE` writes no cache file.
        """
        # Prepare inputs.
        path = "/graph/v1/paper/DOI:10.1/x"
        server = self.start({path: [_answer(200, _S2_PAPER)]})
        sources = self.make_sources(server, cache_mode="DISABLE_CACHE")
        # Run test.
        sources.lookup_s2("DOI:10.1/x")
        sources.lookup_s2("DOI:10.1/x")
        # Check outputs.
        self.assertEqual(server.counts[path], 2)
        cache_dir = os.path.join(self.get_scratch_space(), "cache", "http")
        self.assertFalse(os.path.exists(cache_dir))

    def test9(self) -> None:
        """
        Test two requests to the same API are at least the throttle apart.
        """
        # Prepare inputs.
        server = self.start(
            {
                "/graph/v1/paper/DOI:10.1/x": [_answer(200, _S2_PAPER)],
                "/graph/v1/paper/DOI:10.1/y": [_answer(200, _S2_PAPER)],
            }
        )
        settings = epqteuti.make_settings(server.url)
        settings = epqbibso.HttpSettings(
            **{**settings.__dict__, "s2_throttle_sec": 0.3}
        )
        sources = epqbibso.BibSources(
            email="",
            s2_api_key="",
            cache_dir=os.path.join(self.get_scratch_space(), "cache"),
            settings=settings,
        )
        # Run test.
        start = time.monotonic()
        sources.lookup_s2("DOI:10.1/x")
        sources.lookup_s2("DOI:10.1/y")
        elapsed = time.monotonic() - start
        # Check outputs.
        self.assertGreaterEqual(elapsed, 0.29)


# #############################################################################
# Test_lookups
# #############################################################################


class Test_lookups(_ServerTestCase):
    """
    Test the parsing of the answers of each API into `BibRecord`.
    """

    def test1(self) -> None:
        """
        Test a Semantic Scholar paper.
        """
        # Prepare inputs.
        path = "/graph/v1/paper/DOI:10.1111/jofi.12885"
        sources = self.make_sources(
            self.start({path: [_answer(200, _S2_PAPER)]})
        )
        # Prepare outputs.
        expected = (
            "BibRecord(source='semantic_scholar', title='Lazy Prices', "
            "authors=('Cohen, Lauren', 'Malloy, Christopher'), year=2020, "
            "record_type='journal-article', doi='10.1111/jofi.12885', "
            "arxiv_id='2001.00001', ssrn_id='', s2_paper_id='abc123', "
            "pdf_urls=('https://example.org/lazy.pdf',))"
        )
        # Run test.
        actual = sources.lookup_s2("DOI:10.1111/jofi.12885")
        # Check outputs.
        self.assert_equal(str(actual), expected)

    def test2(self) -> None:
        """
        Test a search of Semantic Scholar, and a search with no result.
        """
        # Prepare inputs.
        server = self.start(
            {"/graph/v1/paper/search": [_answer(200, {"data": [_S2_PAPER] * 2})]}
        )
        sources = self.make_sources(server)
        # Run test.
        records = sources.search_s2("Lazy Prices", 5)
        # Check outputs.
        self.assertEqual(len(records), 2)
        self.assertIn("query=Lazy+Prices", server.requests[0])
        self.assertIn("limit=5", server.requests[0])

    def test3(self) -> None:
        """
        Test a search of Semantic Scholar with no `data` key.
        """
        # Prepare inputs.
        server = self.start(
            {"/graph/v1/paper/search": [_answer(200, {"total": 0})]}
        )
        sources = self.make_sources(server)
        # Run test.
        actual = sources.search_s2("Nothing", 5)
        # Check outputs.
        self.assertEqual(actual, [])

    def test4(self) -> None:
        """
        Test a Crossref work and the SSRN id of an SSRN DOI.
        """
        # Prepare inputs.
        work = {
            **_CROSSREF_WORK,
            "DOI": "10.2139/ssrn.5277078",
            "type": "posted-content",
        }
        server = self.start(
            {"/works/10.2139/ssrn.5277078": [_answer(200, {"message": work})]}
        )
        sources = self.make_sources(server)
        # Prepare outputs.
        expected = (
            "BibRecord(source='crossref', title='Lazy Prices', "
            "authors=('Cohen, Lauren', 'Malloy, Christopher'), year=2020, "
            "record_type='posted-content', doi='10.2139/ssrn.5277078', "
            "arxiv_id='', ssrn_id='5277078', s2_paper_id='', pdf_urls=())"
        )
        # Run test.
        actual = sources.lookup_crossref("10.2139/ssrn.5277078")
        # Check outputs.
        self.assert_equal(str(actual), expected)
        self.assertIn("mailto=me%40example.org", server.requests[0])

    def test10(self) -> None:
        """
        Test a name in capitals is changed to title case.
        """
        # Prepare inputs.
        work = {
            **_CROSSREF_WORK,
            "author": [
                {"family": "COHEN", "given": "LAUREN"},
                {"family": "McDonald", "given": "J. R."},
            ],
        }
        server = self.start({"/works/10.1/x": [_answer(200, {"message": work})]})
        sources = self.make_sources(server)
        # Run test.
        actual = sources.lookup_crossref("10.1/x")
        # Check outputs.
        self.assertEqual(actual.authors, ("Cohen, Lauren", "McDonald, J. R."))

    def test5(self) -> None:
        """
        Test a search of Crossref.
        """
        # Prepare inputs.
        message = {"items": [_CROSSREF_WORK, _CROSSREF_WORK]}
        server = self.start({"/works": [_answer(200, {"message": message})]})
        sources = self.make_sources(server)
        # Run test.
        records = sources.search_crossref("Lazy Prices", 5)
        # Check outputs.
        self.assertEqual([r.year for r in records], [2020, 2020])
        self.assertIn("query.bibliographic=Lazy+Prices", server.requests[0])

    def test6(self) -> None:
        """
        Test an arXiv paper, with the version removed from the id.
        """
        # Prepare inputs.
        server = self.start(
            {"/api/query": [_answer(200, _ARXIV_FEED, "application/atom+xml")]}
        )
        sources = self.make_sources(server)
        # Prepare outputs.
        expected = (
            "BibRecord(source='arxiv', title='Why Should I Trust You?: "
            "Explaining the Predictions', authors=('Ribeiro, Marco Tulio', "
            "'Singh, Sameer'), year=2016, record_type='posted-content', doi='', "
            "arxiv_id='1602.04938', ssrn_id='', s2_paper_id='', "
            "pdf_urls=('https://arxiv.org/pdf/1602.04938',))"
        )
        # Run test.
        actual = sources.lookup_arxiv("1602.04938")
        # Check outputs.
        self.assert_equal(str(actual), expected)

    def test7(self) -> None:
        """
        Test an arXiv id that does not exist gives no record.
        """
        # Prepare inputs.
        feed = (
            '<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
            "<id>http://arxiv.org/api/errors</id><title>Error</title>"
            "</entry></feed>"
        )
        server = self.start(
            {"/api/query": [_answer(200, feed, "application/atom+xml")]}
        )
        sources = self.make_sources(server)
        # Run test.
        actual = sources.lookup_arxiv("9999.99999")
        # Check outputs.
        self.assertIsNone(actual)

    def test8(self) -> None:
        """
        Test Unpaywall gives the PDF of the best open-access location.
        """
        # Prepare inputs.
        body = {"best_oa_location": {"url_for_pdf": "https://oa.example/x.pdf"}}
        server = self.start({"/v2/10.1/x": [_answer(200, body)]})
        sources = self.make_sources(server)
        # Run test.
        actual = sources.lookup_unpaywall("10.1/x")
        # Check outputs.
        self.assert_equal(str(actual), "https://oa.example/x.pdf")

    def test9(self) -> None:
        """
        Test Unpaywall is skipped with no email and when there is no location.
        """
        # Prepare inputs.
        server = self.start(
            {"/v2/10.1/x": [_answer(200, {"best_oa_location": None})]}
        )
        no_email = self.make_sources(server, email="")
        with_email = self.make_sources(server)
        # Run test.
        skipped = no_email.lookup_unpaywall("10.1/x")
        no_pdf = with_email.lookup_unpaywall("10.1/x")
        # Check outputs.
        self.assertIsNone(skipped)
        self.assertIsNone(no_pdf)
        self.assertEqual(server.counts["/v2/10.1/x"], 1)


# #############################################################################
# Test_get_citation_count
# #############################################################################


class Test_get_citation_count(_ServerTestCase):
    """
    Test `BibSources.get_citation_count()`.
    """

    def test1(self) -> None:
        """
        Test the count has the date of the lookup.
        """
        # Prepare inputs.
        path = "/graph/v1/paper/abc123"
        server = self.start({path: [_answer(200, {"citationCount": 512})]})
        sources = self.make_sources(server)
        # Prepare outputs.
        today = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
        expected = f"CitationCount(count=512, source='Semantic Scholar', fetched_at='{today}')"
        # Run test.
        actual = sources.get_citation_count(
            s2_paper_id="abc123", doi="", arxiv_id=""
        )
        # Check outputs.
        self.assert_equal(str(actual), expected)

    def test2(self) -> None:
        """
        Test the DOI is tried when Semantic Scholar does not know the id.
        """
        # Prepare inputs.
        server = self.start(
            {
                "/graph/v1/paper/abc123": [_answer(404, {})],
                "/graph/v1/paper/DOI:10.1/x": [
                    _answer(200, {"citationCount": 7})
                ],
            }
        )
        sources = self.make_sources(server)
        # Run test.
        actual = sources.get_citation_count(
            s2_paper_id="abc123", doi="10.1/x", arxiv_id=""
        )
        # Check outputs.
        self.assertEqual(actual.count, 7)

    def test3(self) -> None:
        """
        Test a paper that Semantic Scholar does not have gives `None`.
        """
        # Prepare inputs.
        server = self.start({"/graph/v1/paper/DOI:10.1/x": [_answer(404, {})]})
        sources = self.make_sources(server)
        # Run test.
        actual = sources.get_citation_count(
            s2_paper_id="", doi="10.1/x", arxiv_id=""
        )
        # Check outputs.
        self.assertIsNone(actual)

    def test4(self) -> None:
        """
        Test `refresh` ignores the cached count.
        """
        # Prepare inputs.
        path = "/graph/v1/paper/abc123"
        server = self.start({path: [_answer(200, {"citationCount": 5})]})
        sources = self.make_sources(server)
        sources.get_citation_count(s2_paper_id="abc123", doi="", arxiv_id="")
        # Run test.
        sources.get_citation_count(
            s2_paper_id="abc123", doi="", arxiv_id="", refresh=True
        )
        # Check outputs.
        self.assertEqual(server.counts[path], 2)


# #############################################################################
# Test_fetch_url_meta
# #############################################################################


class Test_fetch_url_meta(_ServerTestCase):
    """
    Test `BibSources.fetch_url_meta()`.
    """

    def test1(self) -> None:
        """
        Test the meta tags of an HTML page.
        """
        # Prepare inputs.
        html = """
        <html><head>
        <meta name="citation_title" content="Lazy Prices">
        <meta name="citation_author" content="Lauren Cohen">
        <meta name="citation_author" content="Christopher Malloy">
        <meta name="citation_doi" content="10.1111/jofi.12885">
        </head><body>x</body></html>
        """
        server = self.start({"/page": [_answer(200, html, "text/html")]})
        sources = self.make_sources(server)
        # Prepare outputs.
        expected = (
            "UrlMeta(pdf_path='', doi='10.1111/jofi.12885', title='Lazy Prices', "
            "authors=('Cohen, Lauren', 'Malloy, Christopher'))"
        )
        # Run test.
        actual = sources.fetch_url_meta(f"{server.url}/page")
        # Check outputs.
        self.assert_equal(str(actual), expected)

    def test2(self) -> None:
        """
        Test a PDF is saved in the cache and its path is returned.
        """
        # Prepare inputs.
        pdf = b"%PDF-1.4 fake content"
        server = self.start(
            {"/paper.pdf": [_answer(200, pdf, "application/pdf")]}
        )
        sources = self.make_sources(server)
        # Run test.
        actual = sources.fetch_url_meta(f"{server.url}/paper.pdf")
        # Check outputs.
        self.assertTrue(actual.pdf_path.endswith(".pdf"))
        with open(actual.pdf_path, "rb") as f:
            self.assertEqual(f.read(), pdf)

    def test3(self) -> None:
        """
        Test the second call is served by the cache.
        """
        # Prepare inputs.
        server = self.start(
            {"/page": [_answer(200, "<html></html>", "text/html")]}
        )
        sources = self.make_sources(server)
        # Run test.
        sources.fetch_url_meta(f"{server.url}/page")
        sources.fetch_url_meta(f"{server.url}/page")
        # Check outputs.
        self.assertEqual(server.counts["/page"], 1)
