"""
Shared helpers for the tests of the Estimate Paper Quality pipeline.

This module has:
- Builders of the dataclasses that the tests share
- A fake of the lookup APIs client, with canned records
- A local HTTP server that stands in for the lookup APIs
- A builder of small PDFs

Import as:

import research.Estimate_academic_paper_quality.test.epq_test_utils as epqteuti
"""

import http.server
import logging
import os
import stat
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

import helpers.hio as hio
import pymupdf

import research.Estimate_academic_paper_quality.epq_bib_sources as epqbibso
import research.Estimate_academic_paper_quality.epq_types as epqtypes

_LOG = logging.getLogger(__name__)

# #############################################################################
# Builders
# #############################################################################


def make_record(
    *,
    source: str = "semantic_scholar",
    title: str = "Lazy Prices",
    authors: Tuple[str, ...] = ("Cohen, Lauren", "Malloy, Christopher"),
    year: int = 2020,
    record_type: str = "journal-article",
    doi: str = "10.1111/jofi.12885",
    arxiv_id: str = "",
    ssrn_id: str = "",
    s2_paper_id: str = "s2lazy",
    pdf_urls: Tuple[str, ...] = (),
) -> epqtypes.BibRecord:
    """
    Build a bibliographic record, by default the paper `Lazy Prices`.

    :return: record
    """
    return epqtypes.BibRecord(
        source=source,
        title=title,
        authors=authors,
        year=year,
        record_type=record_type,
        doi=doi,
        arxiv_id=arxiv_id,
        ssrn_id=ssrn_id,
        s2_paper_id=s2_paper_id,
        pdf_urls=pdf_urls,
    )


def make_evidence(
    *, quote: str = "Sharpe 1.4", location: str = "Table 4, p. 21"
) -> epqtypes.Evidence:
    """
    Build the evidence of a value.

    :return: evidence, verified
    """
    return epqtypes.Evidence(quote=quote, location=location, verified=True)


def make_feature(
    value: str, *, with_evidence: bool = True
) -> epqtypes.FeatureValue:
    """
    Build a feature value.

    :param value: free text, or `unknown`
    :param with_evidence: if True, add evidence
    :return: feature value
    """
    evidence = make_evidence() if with_evidence else None
    return epqtypes.FeatureValue(value=value, evidence=evidence)


def make_paper(
    *,
    base_name: str = "2020.Cohen_et_al.Lazy_Prices",
    year: int = 2020,
    line_nos: Tuple[int, ...] = (1,),
    download_inputs: Tuple[str, ...] = ("https://arxiv.org/pdf/2001.00001",),
    reference: str = (
        '2020, Cohen et al., "Lazy Prices" (https://arxiv.org/pdf/2001.00001)'
    ),
) -> epqtypes.Paper:
    """
    Build a paper, by default `Lazy Prices`.

    :return: paper
    """
    return epqtypes.Paper(
        key="lazy prices|cohen",
        aliases=("doi:10.1111/jofi.12885",),
        line_nos=line_nos,
        source="semantic_scholar",
        year=year,
        authors=("Cohen, Lauren", "Malloy, Christopher"),
        title="Lazy Prices",
        doi="10.1111/jofi.12885",
        arxiv_id="2001.00001",
        s2_paper_id="s2lazy",
        reference=reference,
        base_name=base_name,
        download_inputs=download_inputs,
    )


def make_facts() -> epqtypes.ExtractedFacts:
    """
    Build the facts of a paper with one Sharpe ratio.

    :return: facts
    """
    sharpe = epqtypes.LabeledFact(
        label="out of sample", value="1.4", evidence=make_evidence()
    )
    return epqtypes.ExtractedFacts(
        sharpe_ratios=(sharpe,),
        annual_returns=(),
        volatilities=(),
        max_drawdowns=(),
        sample_periods=(),
        universe=make_feature("US stocks"),
        data_assets=make_feature("unknown", with_evidence=False),
        data_frequency=make_feature("monthly"),
        data_fields=make_feature("unknown", with_evidence=False),
    )


def make_quality(*, rating: Optional[int] = 4) -> epqtypes.QualityRating:
    """
    Build a quality rating with known signals.

    :return: rating
    """

    def _signal(name: str, score: Optional[int]) -> epqtypes.SignalScore:
        return epqtypes.SignalScore(
            name=name,
            value="unknown" if score is None else "x",
            score=score,
            evidence=None,
            note="",
        )

    return epqtypes.QualityRating(
        rating=rating,
        page_length=_signal("page_length", 3),
        affiliation=_signal("affiliation", 5),
        citations=_signal("citations", None),
    )


def make_affiliation(
    *, institution: str = "Harvard University", kind: str = "university"
) -> epqtypes.Affiliation:
    """
    Build the affiliation of one author.

    :param institution: top-level institution as written in the paper
    :param kind: kind of institution
    :return: affiliation, with verified evidence
    """
    return epqtypes.Affiliation(
        author="Lauren Cohen",
        institution=institution,
        kind=kind,
        evidence=make_evidence(quote=institution, location="byline"),
    )


def make_paper_features(
    *, base_name: str = "2020.Cohen_et_al.Lazy_Prices"
) -> epqtypes.PaperFeatures:
    """
    Build the features of a paper, with one affiliation and 12 pages.

    :return: features
    """
    return epqtypes.PaperFeatures(
        base_name=base_name,
        features={
            "performance": make_feature("Sharpe 1.4, out of sample"),
            "recency": make_feature("2020"),
            "asset_class": make_feature("US equities"),
            "theoretical_domain": make_feature("unknown", with_evidence=False),
            "mechanism": make_feature("associational"),
        },
        facts=make_facts(),
        affiliations=(make_affiliation(),),
        n_pages=12,
        provenance=epqtypes.Provenance(
            model="test-model",
            prompt_sha1="abc",
            chars_sent=100,
            text_was_cut=False,
            cost_usd=0.04,
        ),
    )


def make_llm_payload() -> Dict[str, Any]:
    """
    Build the answer of the fake LLM for the paper `Lazy Prices`.

    The quote of `mechanism` is not in the PDF, so it stays unverified.

    :return: dict that the structured answer of the extractor accepts
    """
    unknown: Dict[str, Any] = {"value": "unknown", "evidence": None}
    evidence = {"quote": "Sharpe ratio of 1.4", "section": "Table 4"}
    return {
        "performance": {"value": "Sharpe 1.4", "evidence": evidence},
        "asset_class": {
            "value": "US equities",
            "evidence": {"quote": "CRSP stocks", "section": "Sec. 2"},
        },
        "theoretical_domain": unknown,
        "mechanism": {
            "value": "associational",
            "evidence": {"quote": "made up quote", "section": "Sec. 5"},
        },
        "sharpe_ratios": [
            {"label": "out of sample", "value": "1.4", "evidence": evidence}
        ],
        "annual_returns": [],
        "volatilities": [],
        "max_drawdowns": [],
        "sample_periods": [],
        "universe": unknown,
        "data_assets": unknown,
        "data_frequency": unknown,
        "data_fields": unknown,
        "affiliations": [
            {
                "author": "Lauren Cohen",
                "institution": "Harvard University",
                "kind": "university",
                "evidence": {"quote": "Harvard University", "section": "byline"},
            }
        ],
    }


# #############################################################################
# Fake of the lookup APIs client
# #############################################################################


class FakeBibSources(epqbibso.BibSources):
    """
    Client of the lookup APIs that answers from canned records.

    It stands in for the network, which is the external dependency.
    """

    def __init__(
        self,
        *,
        by_id: Optional[Dict[str, epqtypes.BibRecord]] = None,
        by_title: Optional[Dict[str, List[epqtypes.BibRecord]]] = None,
        unpaywall: Optional[Dict[str, str]] = None,
        url_meta: Optional[Dict[str, epqtypes.UrlMeta]] = None,
        citations: Optional[Dict[str, epqtypes.CitationCount]] = None,
    ) -> None:
        """
        Initialize the fake without any network setup.

        :param by_id: records by `<source>:<id>`, e.g., `s2:DOI:10.1/x`,
            `crossref:10.1/x`, `arxiv:1706.03762`
        :param by_title: records found by a search, by `<source>:<title>`
        :param unpaywall: PDF URLs by DOI
        :param url_meta: facts of a page, by URL
        :param citations: citation counts by DOI
        """
        self._by_id = by_id or {}
        self._by_title = by_title or {}
        self._unpaywall = unpaywall or {}
        self._url_meta = url_meta or {}
        self._citations = citations or {}
        self.calls: List[str] = []

    def lookup_s2(self, paper_id: str) -> Optional[epqtypes.BibRecord]:
        self.calls.append(f"lookup_s2 {paper_id}")
        return self._by_id.get(f"s2:{paper_id}")

    def search_s2(self, query: str, limit: int) -> List[epqtypes.BibRecord]:
        self.calls.append(f"search_s2 {query}")
        return self._by_title.get(f"s2:{query}", [])

    def lookup_crossref(self, doi: str) -> Optional[epqtypes.BibRecord]:
        self.calls.append(f"lookup_crossref {doi}")
        return self._by_id.get(f"crossref:{doi}")

    def search_crossref(
        self, query: str, limit: int
    ) -> List[epqtypes.BibRecord]:
        self.calls.append(f"search_crossref {query}")
        return self._by_title.get(f"crossref:{query}", [])

    def lookup_arxiv(self, arxiv_id: str) -> Optional[epqtypes.BibRecord]:
        self.calls.append(f"lookup_arxiv {arxiv_id}")
        return self._by_id.get(f"arxiv:{arxiv_id}")

    def lookup_unpaywall(self, doi: str) -> Optional[str]:
        self.calls.append(f"lookup_unpaywall {doi}")
        return self._unpaywall.get(doi)

    def fetch_url_meta(self, url: str) -> epqtypes.UrlMeta:
        self.calls.append(f"fetch_url_meta {url}")
        empty = epqtypes.UrlMeta(pdf_path="", doi="", title="", authors=())
        return self._url_meta.get(url, empty)

    def get_citation_count(
        self,
        *,
        s2_paper_id: str,
        doi: str,
        arxiv_id: str,
        refresh: bool = False,
    ) -> Optional[epqtypes.CitationCount]:
        self.calls.append(f"get_citation_count {doi}")
        return self._citations.get(doi)


# #############################################################################
# Local HTTP server
# #############################################################################


class LocalServer:
    """
    HTTP server on `127.0.0.1` that stands in for the lookup APIs.

    Each path has a queue of answers `(status, body, content type)`. The server
    sends them in order and repeats the last one.
    """

    def __init__(self, routes: Dict[str, List[Tuple[int, bytes, str]]]) -> None:
        """
        Initialize the server and start it on a free port.

        :param routes: queue of answers by URL path, without the query
        """
        self.routes = routes
        # Number of requests by path.
        self.counts: Dict[str, int] = {}
        # Headers of the last request, by path.
        self.headers: Dict[str, Dict[str, str]] = {}
        # Full path with the query of every request, in order.
        self.requests: List[str] = []
        server = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # pylint: disable=invalid-name
                path = self.path.split("?")[0]
                server.requests.append(self.path)
                server.counts[path] = server.counts.get(path, 0) + 1
                server.headers[path] = {
                    k.lower(): v for k, v in self.headers.items()
                }
                queue = server.routes.get(path)
                if not queue:
                    status, body, content_type = 404, b"{}", "application/json"
                else:
                    index = min(server.counts[path], len(queue)) - 1
                    status, body, content_type = queue[index]
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                # Keep the output of the tests quiet.
                pass

        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """
        Stop the server.
        """
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join()


def make_settings(url: str) -> epqbibso.HttpSettings:
    """
    Build the HTTP settings that point every API to a local server.

    :param url: URL of the local server
    :return: settings with no delays
    """
    return epqbibso.HttpSettings(
        s2_url=url,
        crossref_url=url,
        arxiv_url=url,
        unpaywall_url=url,
        s2_throttle_sec=0.0,
        crossref_throttle_sec=0.0,
        arxiv_throttle_sec=0.0,
        unpaywall_throttle_sec=0.0,
        web_throttle_sec=0.0,
        backoff_sec=(0.0, 0.0, 0.0, 0.0),
        timeout_sec=10.0,
    )


# #############################################################################
# Stub of the download tool
# #############################################################################


# Stub of the download tool. The input says what it does (no keyword: `good`):
# - `good`: write a valid PDF and a markdown with text
# - `html`: write an HTML page named `.pdf`, then fail
# - `scanned`: write a valid PDF and a markdown with almost no text
# - `convfail`: write a valid PDF, then fail
# - `fail`: write nothing and fail
# With `--clear_actions` it only converts an existing PDF.
STUB_TOOL = r"""
import argparse
import os
import shutil
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--input")
parser.add_argument("--output")
parser.add_argument("--clear_actions", action="store_true")
args, _ = parser.parse_known_args()
pdf_path = args.output + ".pdf"
md_path = args.output + ".md"
with open(os.environ["EPQ_STUB_CALLS"], "a") as f:
    f.write(args.input + " clear=" + str(args.clear_actions) + "\n")
if args.clear_actions:
    if "NOCONVERT" in open(pdf_path, errors="ignore").read():
        print("Error: conversion crashed", file=sys.stderr)
        sys.exit(1)
    open(md_path, "w").write("converted text " * 60)
    sys.exit(0)
if "fail" in args.input and "convfail" not in args.input:
    print("Error: HTTP 404 for the input", file=sys.stderr)
    sys.exit(1)
if "html" in args.input:
    open(pdf_path, "w").write("<html>landing page</html>")
    print("Error: cannot convert", file=sys.stderr)
    sys.exit(1)
if os.environ.get("EPQ_STUB_PDF"):
    shutil.copyfile(os.environ["EPQ_STUB_PDF"], pdf_path)
else:
    open(pdf_path, "w").write("%PDF-1.4 fake")
if "convfail" in args.input:
    print("Error: conversion crashed", file=sys.stderr)
    sys.exit(1)
text = "x" if "scanned" in args.input else "converted text " * 60
open(md_path, "w").write(text)
"""


def make_stub_tool(scratch: str) -> str:
    """
    Write the stub tool in the scratch space and make it executable.

    :param scratch: scratch dir of the test
    :return: path of the stub
    """
    path = os.path.join(scratch, "stub_tool.py")
    hio.to_file(path, f"#!{sys.executable}\n{STUB_TOOL}")
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    return path


# #############################################################################
# PDF builder
# #############################################################################


def create_pdf(
    path: str,
    page_texts: List[str],
    *,
    title_text: str = "",
    title_size: float = 24.0,
    meta_title: str = "",
) -> None:
    """
    Create a small PDF.

    :param path: path of the file
    :param page_texts: body text of each page, in 10 pt
    :param title_text: text in a large font at the top of the first page
    :param title_size: font size of the title text
    :param meta_title: title in the metadata of the PDF
    """
    doc = pymupdf.open()
    for i, page_text in enumerate(page_texts):
        page = doc.new_page()
        if i == 0 and title_text:
            page.insert_text((72, 72), title_text, fontsize=title_size)
        page.insert_text((72, 140), page_text, fontsize=10)
    if meta_title:
        doc.set_metadata({"title": meta_title})
    doc.save(path)
    doc.close()
