"""
Clients of the bibliographic lookup APIs.

This module gives the other components the records, the open-access links, and
the citation counts of Semantic Scholar, Crossref, arXiv, and Unpaywall. It also
reads the page or the PDF that a `url` line points to.

Each API is throttled and retried, and each answer is cached on disk, so that a
rerun makes no network call for a line already looked up. Bibliographic facts
come only from these sources, never from the memory of an LLM.

The design is in `docs/02.3-design.bib_sources.md`.

Import as:

import research.Estimate_academic_paper_quality.epq_bib_sources as epqbibso
"""

import dataclasses
import datetime
import hashlib
import json
import logging
import os
import re
import time
import urllib.parse
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import bs4
import feedparser
import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hprint as hprint
import requests

import research.Estimate_academic_paper_quality.epq_types as epqtypes

_LOG = logging.getLogger(__name__)

# #############################################################################
# Constants
# #############################################################################

# Fields to ask Semantic Scholar for.
_S2_FIELDS = (
    "title,year,authors,externalIds,openAccessPdf,citationCount,"
    "publicationDate,publicationTypes"
)

# Suffixes of a person name that are not part of the family name.
_NAME_SUFFIXES = ["jr", "sr", "ii", "iii", "iv"]

# Statuses that mean "retry later", with the answers `5xx`.
_RETRY_STATUS = 429

# Statuses that are a stable answer, so they are cached: found, or not found.
_STABLE_STATUSES = [200, 404]


@dataclasses.dataclass(frozen=True)
class HttpSettings:
    """
    Endpoints and timing of the HTTP calls.

    The defaults are the real APIs. A test points the endpoints to a local
    server and sets the delays to zero.
    """

    s2_url: str = "https://api.semanticscholar.org"
    crossref_url: str = "https://api.crossref.org"
    arxiv_url: str = "http://export.arxiv.org"
    unpaywall_url: str = "https://api.unpaywall.org"
    # Seconds to wait between two requests to the same API.
    s2_throttle_sec: float = 1.0
    crossref_throttle_sec: float = 1.0
    arxiv_throttle_sec: float = 3.0
    unpaywall_throttle_sec: float = 0.1
    web_throttle_sec: float = 0.5
    # Seconds to wait before each retry of a `429` or `5xx` answer.
    backoff_sec: Tuple[float, ...] = (2.0, 4.0, 8.0, 16.0)
    timeout_sec: float = 30.0


# Real endpoints and delays, the default of `BibSources`.
_DEFAULT_SETTINGS = HttpSettings()


# #############################################################################
# Names and parsers
# #############################################################################


def _is_name_suffix(token: str) -> bool:
    """
    Check if a token of a person name is a suffix, e.g., `Jr.` or `III`.

    :param token: token of a name
    :return: True if the token is not part of the family name
    """
    return token.strip().rstrip(".").lower() in _NAME_SUFFIXES


def name_to_family_first(name: str) -> str:
    """
    Convert an author name into the form `Family, Given`.

    A name that already has a comma is kept, except for a suffix after the
    comma. A suffix like `Jr.` is dropped.
    - `Lauren Cohen` -> `Cohen, Lauren`
    - `Robert J. Jackson, Jr.` -> `Jackson, Robert J.`

    :param name: author name, as `Given Family` or `Family, Given`
    :return: name as `Family, Given`, or the family name alone
    """
    parts = [part.strip() for part in name.split(",") if part.strip()]
    if len(parts) > 1 and _is_name_suffix(parts[1]):
        # The comma only separates a suffix: `Robert J. Jackson, Jr.`.
        parts = parts[:1]
    if len(parts) > 1:
        # Already `Family, Given`.
        result = ", ".join(parts[:2])
    else:
        tokens = parts[0].split() if parts else []
        while len(tokens) > 1 and _is_name_suffix(tokens[-1]):
            tokens.pop()
        if len(tokens) > 1:
            result = f"{tokens[-1]}, {' '.join(tokens[:-1])}"
        else:
            result = tokens[0] if tokens else ""
    return result


def get_family_name(author: str) -> str:
    """
    Get the family name of an author written as `Family, Given`.

    :param author: author name from a `BibRecord`
    :return: the part before the comma
    """
    return author.split(",")[0].strip()


def _get_ssrn_id_from_doi(doi: str) -> str:
    """
    Get the SSRN id from an SSRN DOI.

    :param doi: DOI, e.g., `10.2139/ssrn.5277078`
    :return: the SSRN id, or an empty string if it is not an SSRN DOI
    """
    match = re.match(r"^10\.2139/ssrn\.(\d+)$", doi.strip(), re.IGNORECASE)
    return match.group(1) if match else ""


def _get_arxiv_id_from_doi(doi: str) -> str:
    """
    Get the arXiv id from an arXiv DOI.

    :param doi: DOI, e.g., `10.48550/arXiv.1602.04938`
    :return: the arXiv id, or an empty string if it is not an arXiv DOI
    """
    match = re.match(r"^10\.48550/arxiv\.(.+)$", doi.strip(), re.IGNORECASE)
    return match.group(1) if match else ""


def _parse_s2(item: Dict[str, Any]) -> epqtypes.BibRecord:
    """
    Convert a Semantic Scholar paper into a record.

    :param item: JSON object of one paper
    :return: record with the fields that the resolver needs
    """
    external_ids = item.get("externalIds") or {}
    doi = external_ids.get("DOI") or ""
    open_access = item.get("openAccessPdf") or {}
    pub_types = item.get("publicationTypes") or []
    if "JournalArticle" in pub_types:
        record_type = "journal-article"
    else:
        record_type = pub_types[0].lower() if pub_types else ""
    record = epqtypes.BibRecord(
        source="semantic_scholar",
        title=re.sub(r"\s+", " ", item.get("title") or "").strip(),
        authors=tuple(
            name_to_family_first(author.get("name") or "")
            for author in item.get("authors") or []
        ),
        year=int(item.get("year") or 0),
        record_type=record_type,
        doi=doi,
        arxiv_id=external_ids.get("ArXiv") or _get_arxiv_id_from_doi(doi),
        ssrn_id=_get_ssrn_id_from_doi(doi),
        s2_paper_id=item.get("paperId") or "",
        pdf_urls=((open_access["url"],) if open_access.get("url") else ()),
    )
    return record


def _get_crossref_year(item: Dict[str, Any]) -> int:
    """
    Get the publication year of a Crossref work.

    :param item: JSON object of one work
    :return: year, 0 if the work has none
    """
    year = 0
    for key in ["issued", "published-print", "published-online", "created"]:
        date_parts = (item.get(key) or {}).get("date-parts") or [[None]]
        first_part = date_parts[0][0] if date_parts[0] else None
        if first_part:
            year = int(first_part)
            break
    return year


def _fix_caps(name: str) -> str:
    """
    Fix a name that Crossref gives in capitals.

    Some publishers deposit `COHEN`: a name in capitals is changed to
    `Cohen`, so that the reference and the file name read like the others.

    :param name: family name or given name
    :return: the name in title case if it is all capitals, else unchanged
    """
    return name.title() if name.isupper() else name


def _parse_crossref(item: Dict[str, Any]) -> epqtypes.BibRecord:
    """
    Convert a Crossref work into a record.

    :param item: JSON object of one work
    :return: record with the fields that the resolver needs
    """
    titles = item.get("title") or [""]
    authors = []
    for author in item.get("author") or []:
        family = _fix_caps(author.get("family") or "")
        given = _fix_caps(author.get("given") or "")
        name = f"{family}, {given}".strip(", ") if family else given
        authors.append(name)
    doi = item.get("DOI") or ""
    record = epqtypes.BibRecord(
        source="crossref",
        title=re.sub(r"\s+", " ", titles[0] or "").strip(),
        authors=tuple(authors),
        year=_get_crossref_year(item),
        record_type=item.get("type") or "",
        doi=doi,
        arxiv_id=_get_arxiv_id_from_doi(doi),
        ssrn_id=_get_ssrn_id_from_doi(doi),
        s2_paper_id="",
        pdf_urls=(),
    )
    return record


def _parse_arxiv(entry: Any) -> epqtypes.BibRecord:
    """
    Convert an entry of the arXiv Atom feed into a record.

    :param entry: entry parsed by `feedparser`
    :return: record with the fields that the resolver needs
    """
    # The id is a URL like `http://arxiv.org/abs/1602.04938v3`.
    arxiv_id = re.sub(r"^.*/abs/", "", entry.get("id", ""))
    arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
    published = entry.get("published", "")
    record = epqtypes.BibRecord(
        source="arxiv",
        title=re.sub(r"\s+", " ", entry.get("title", "")).strip(),
        authors=tuple(
            name_to_family_first(author.get("name", ""))
            for author in entry.get("authors", [])
        ),
        year=int(published[:4]) if published[:4].isdigit() else 0,
        record_type="posted-content",
        doi=entry.get("arxiv_doi", ""),
        arxiv_id=arxiv_id,
        ssrn_id="",
        s2_paper_id="",
        pdf_urls=(f"https://arxiv.org/pdf/{arxiv_id}",),
    )
    return record


# #############################################################################
# BibSources
# #############################################################################


class BibSources:
    """
    Throttled, retried, cached clients of the lookup APIs.
    """

    def __init__(
        self,
        *,
        email: str,
        s2_api_key: str,
        cache_dir: str,
        cache_mode: str = "",
        settings: HttpSettings = _DEFAULT_SETTINGS,
    ) -> None:
        """
        Initialize the clients.

        :param email: contact email, sent to Crossref and Unpaywall
            - An empty string skips Unpaywall
        :param s2_api_key: key of Semantic Scholar, "" for the shared pool
        :param cache_dir: dir that holds the cache of the answers
        :param cache_mode: one of `""`, `REFRESH_CACHE`, `DISABLE_CACHE`, or
            `HIT_CACHE_OR_ABORT`, with the meaning of `hcache_simple`
        :param settings: endpoints and delays
        """
        _LOG.debug(hprint.to_str("email cache_dir cache_mode"))
        hdbg.dassert_in(
            cache_mode,
            ["", "REFRESH_CACHE", "DISABLE_CACHE", "HIT_CACHE_OR_ABORT"],
            "Invalid cache mode",
        )
        self._email = email
        self._s2_api_key = s2_api_key
        self._cache_dir = cache_dir
        self._cache_mode = cache_mode
        self._settings = settings
        # Seconds between two requests, by API.
        self._throttle_sec = {
            "s2": settings.s2_throttle_sec,
            "crossref": settings.crossref_throttle_sec,
            "arxiv": settings.arxiv_throttle_sec,
            "unpaywall": settings.unpaywall_throttle_sec,
            "web": settings.web_throttle_sec,
        }
        # Time of the last request, by API.
        self._last_request: Dict[str, float] = {}
        # APIs that failed after all the retries: skipped for the rest of the
        # run, so that one API that is down does not cost the retries of every
        # line.
        self._failed_hosts: Set[str] = set()

    # ///////////////////////////////////////////////////////////////////////
    # HTTP and cache.
    # ///////////////////////////////////////////////////////////////////////

    def _throttle(self, host: str) -> None:
        """
        Wait so that two requests to the same API are not too close.

        :param host: key of the API in `self._throttle_sec`
        """
        last = self._last_request.get(host)
        if last is not None:
            wait = self._throttle_sec[host] - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
        self._last_request[host] = time.monotonic()

    def _build_headers(self, host: str) -> Dict[str, str]:
        """
        Build the headers of a request.

        :param host: key of the API
        :return: headers with the user agent and the key of Semantic Scholar
        """
        user_agent = "estimate_paper_quality/1.0"
        if self._email:
            user_agent += f" (mailto:{self._email})"
        headers = {"User-Agent": user_agent}
        if host == "s2" and self._s2_api_key:
            headers["x-api-key"] = self._s2_api_key
        return headers

    def _request(
        self, host: str, url: str, params: Dict[str, Any]
    ) -> requests.Response:
        """
        Send a GET request, retrying `429` and `5xx` answers with a backoff.

        A `200` and a `404` are returned. Any other answer raises after the
        retries, and the caller does not cache it.

        :param host: key of the API
        :param url: URL to get
        :param params: query parameters
        :return: response with status `200` or `404`
        """
        _LOG.debug(hprint.to_str("host url params"))
        if host in self._failed_hosts:
            raise RuntimeError(
                f"Request to '{url}' skipped: '{host}' failed earlier in this run"
            )
        backoff = self._settings.backoff_sec
        last_error = ""
        # True if the last failure can change on a retry: `429`, `5xx`, or a
        # network error. A `403` or a `400` cannot, and it does not disable the
        # API.
        retryable = False
        for attempt in range(len(backoff) + 1):
            self._throttle(host)
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers=self._build_headers(host),
                    timeout=self._settings.timeout_sec,
                )
            except (requests.ConnectionError, requests.Timeout) as e:
                # A network error is retried like a `5xx` answer.
                response = None
                last_error = type(e).__name__
                retryable = True
            if response is not None:
                if response.status_code in _STABLE_STATUSES:
                    return response
                last_error = f"HTTP {response.status_code}"
                retryable = (
                    response.status_code == _RETRY_STATUS
                    or response.status_code >= 500
                )
                if not retryable:
                    break
            if attempt < len(backoff):
                _LOG.warning(
                    "Request to '%s' failed with %s: retry in %s s",
                    url,
                    last_error,
                    backoff[attempt],
                )
                time.sleep(backoff[attempt])
        if retryable:
            # The API is down or limits us: stop asking it in this run.
            self._failed_hosts.add(host)
        raise RuntimeError(f"Request to '{url}' failed: {last_error}")

    def _cached(
        self, key: str, producer: Callable[[], Dict[str, Any]], *, refresh: bool
    ) -> Dict[str, Any]:
        """
        Return the cached answer for a key, or compute and cache it.

        :param key: text that identifies the request
        :param producer: function that makes the request on a cache miss
        :param refresh: if True, ignore the cached answer for this call
        :return: answer as a JSON dict
        """
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        path = os.path.join(self._cache_dir, "http", f"{digest}.json")
        mode = self._cache_mode
        use_cache = mode != "DISABLE_CACHE"
        ignore_cache = mode in ["REFRESH_CACHE", "DISABLE_CACHE"] or refresh
        if use_cache and not ignore_cache and os.path.isfile(path):
            entry: Dict[str, Any] = json.loads(hio.from_file(path))
            _LOG.debug("Cache hit for key='%s'", key)
            return entry
        if mode == "HIT_CACHE_OR_ABORT":
            raise ValueError(f"Cache miss for key='{key}'")
        entry = producer()
        if use_cache:
            hio.create_dir(os.path.dirname(path), incremental=True)
            hio.to_file(path, json.dumps(entry))
        return entry

    def _get(
        self,
        host: str,
        url: str,
        params: Dict[str, Any],
        *,
        refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Get a URL, with the cache, the throttle, and the retries.

        :param host: key of the API
        :param url: URL to get
        :param params: query parameters
        :param refresh: if True, ignore the cached answer for this call
        :return: dict with `status`, `body`, and `fetched_at`
        """
        key = f"GET|{url}|{json.dumps(params, sort_keys=True)}"

        def _produce() -> Dict[str, Any]:
            response = self._request(host, url, params)
            fetched_at = datetime.datetime.now(datetime.UTC)
            entry = {
                "status": response.status_code,
                "body": response.text,
                "fetched_at": fetched_at.strftime("%Y-%m-%d"),
            }
            return entry

        return self._cached(key, _produce, refresh=refresh)

    # ///////////////////////////////////////////////////////////////////////
    # Semantic Scholar.
    # ///////////////////////////////////////////////////////////////////////

    def lookup_s2(self, paper_id: str) -> Optional[epqtypes.BibRecord]:
        """
        Look up a paper in Semantic Scholar by id.

        :param paper_id: Semantic Scholar id, or `DOI:<doi>`, or `ARXIV:<id>`
        :return: record, `None` if Semantic Scholar does not have the paper
        """
        quoted = urllib.parse.quote(paper_id, safe=":/")
        url = f"{self._settings.s2_url}/graph/v1/paper/{quoted}"
        entry = self._get("s2", url, {"fields": _S2_FIELDS})
        record = None
        if entry["status"] == 200:
            record = _parse_s2(json.loads(entry["body"]))
        return record

    def search_s2(self, query: str, limit: int) -> List[epqtypes.BibRecord]:
        """
        Search Semantic Scholar by title.

        :param query: title or part of the title
        :param limit: maximum number of records
        :return: records in the order of relevance
        """
        url = f"{self._settings.s2_url}/graph/v1/paper/search"
        params = {"query": query, "limit": limit, "fields": _S2_FIELDS}
        entry = self._get("s2", url, params)
        records = []
        if entry["status"] == 200:
            data = json.loads(entry["body"]).get("data") or []
            records = [_parse_s2(item) for item in data]
        return records

    def get_citation_count(
        self,
        *,
        s2_paper_id: str,
        doi: str,
        arxiv_id: str,
        refresh: bool = False,
    ) -> Optional[epqtypes.CitationCount]:
        """
        Get the citation count of a paper from Semantic Scholar.

        The date of the count is the date of the cache entry, not the date of
        the run.

        :param s2_paper_id: Semantic Scholar id, "" if unknown
        :param doi: DOI, "" if unknown
        :param arxiv_id: arXiv id, "" if unknown
        :param refresh: if True, ignore the cached count
        :return: count with its source and date, `None` if Semantic Scholar
            does not have the paper
        """
        ids = []
        if s2_paper_id:
            ids.append(s2_paper_id)
        if doi:
            ids.append(f"DOI:{doi}")
        if arxiv_id:
            ids.append(f"ARXIV:{arxiv_id}")
        citation = None
        for paper_id in ids:
            quoted = urllib.parse.quote(paper_id, safe=":/")
            url = f"{self._settings.s2_url}/graph/v1/paper/{quoted}"
            entry = self._get(
                "s2", url, {"fields": "citationCount"}, refresh=refresh
            )
            if entry["status"] != 200:
                continue
            count = json.loads(entry["body"]).get("citationCount")
            if count is not None:
                citation = epqtypes.CitationCount(
                    count=int(count),
                    source="Semantic Scholar",
                    fetched_at=entry["fetched_at"],
                )
                break
        return citation

    # ///////////////////////////////////////////////////////////////////////
    # Crossref.
    # ///////////////////////////////////////////////////////////////////////

    def _crossref_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add the contact email to the parameters of a Crossref request.

        :param params: query parameters
        :return: parameters with `mailto` if there is an email
        """
        if self._email:
            params = {**params, "mailto": self._email}
        return params

    def lookup_crossref(self, doi: str) -> Optional[epqtypes.BibRecord]:
        """
        Look up a work in Crossref by DOI.

        :param doi: DOI
        :return: record, `None` if Crossref does not have the DOI
        """
        quoted = urllib.parse.quote(doi, safe="/")
        url = f"{self._settings.crossref_url}/works/{quoted}"
        entry = self._get("crossref", url, self._crossref_params({}))
        record = None
        if entry["status"] == 200:
            record = _parse_crossref(json.loads(entry["body"])["message"])
        return record

    def search_crossref(
        self, query: str, limit: int
    ) -> List[epqtypes.BibRecord]:
        """
        Search Crossref by title.

        :param query: title or part of the title
        :param limit: maximum number of records
        :return: records in the order of relevance
        """
        url = f"{self._settings.crossref_url}/works"
        params = self._crossref_params(
            {"query.bibliographic": query, "rows": limit}
        )
        entry = self._get("crossref", url, params)
        records = []
        if entry["status"] == 200:
            items = json.loads(entry["body"])["message"].get("items") or []
            records = [_parse_crossref(item) for item in items]
        return records

    # ///////////////////////////////////////////////////////////////////////
    # arXiv and Unpaywall.
    # ///////////////////////////////////////////////////////////////////////

    def lookup_arxiv(self, arxiv_id: str) -> Optional[epqtypes.BibRecord]:
        """
        Look up a paper in arXiv by id.

        :param arxiv_id: arXiv id without version, e.g., `1602.04938`
        :return: record, `None` if arXiv does not have the paper
        """
        url = f"{self._settings.arxiv_url}/api/query"
        entry = self._get("arxiv", url, {"id_list": arxiv_id})
        record = None
        if entry["status"] == 200:
            entries = feedparser.parse(entry["body"]).entries
            # An unknown id gives an entry titled `Error`.
            if entries and entries[0].get("title", "") != "Error":
                record = _parse_arxiv(entries[0])
        return record

    def lookup_unpaywall(self, doi: str) -> Optional[str]:
        """
        Find the open-access PDF of a DOI in Unpaywall.

        :param doi: DOI
        :return: URL of the PDF, `None` if there is no email or no PDF
        """
        if not self._email:
            # Unpaywall rejects a request with no real email.
            return None
        quoted = urllib.parse.quote(doi, safe="/")
        url = f"{self._settings.unpaywall_url}/v2/{quoted}"
        entry = self._get("unpaywall", url, {"email": self._email})
        pdf_url = None
        if entry["status"] == 200:
            location = json.loads(entry["body"]).get("best_oa_location") or {}
            pdf_url = location.get("url_for_pdf") or None
        return pdf_url

    # ///////////////////////////////////////////////////////////////////////
    # Page or PDF of a `url` line.
    # ///////////////////////////////////////////////////////////////////////

    def fetch_url_meta(self, url: str) -> epqtypes.UrlMeta:
        """
        Read the facts of the page or the PDF that a URL points to.

        A PDF is saved to `<cache_dir>/url_pdfs/<sha1>.pdf`, so that the
        resolver reads its title. An HTML page gives the `citation_*` meta tags
        that most publishers set.

        :param url: URL of a publisher page or a direct PDF
        :return: path of the saved PDF, or DOI, title, and authors of the page
        """

        def _produce() -> Dict[str, Any]:
            response = self._request("web", url, {})
            meta: Dict[str, Any] = {
                "pdf_path": "",
                "doi": "",
                "title": "",
                "authors": [],
            }
            if response.status_code == 200 and response.content[:4] == b"%PDF":
                digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
                pdf_path = os.path.join(
                    self._cache_dir, "url_pdfs", f"{digest}.pdf"
                )
                hio.create_dir(os.path.dirname(pdf_path), incremental=True)
                with open(pdf_path, "wb") as f:
                    f.write(response.content)
                meta["pdf_path"] = pdf_path
            elif response.status_code == 200:
                # Read the meta tags that most publishers set.
                soup = bs4.BeautifulSoup(response.text, "html.parser")
                dois = _get_meta_tags(soup, "citation_doi")
                titles = _get_meta_tags(soup, "citation_title")
                meta["doi"] = dois[0] if dois else ""
                meta["title"] = titles[0] if titles else ""
                meta["authors"] = _get_meta_tags(soup, "citation_author")
            return meta

        key = f"META|{url}"
        meta = self._cached(key, _produce, refresh=False)
        url_meta = epqtypes.UrlMeta(
            pdf_path=meta["pdf_path"],
            doi=meta["doi"],
            title=meta["title"],
            authors=tuple(name_to_family_first(a) for a in meta["authors"]),
        )
        return url_meta


def _get_meta_tags(soup: bs4.BeautifulSoup, name: str) -> List[str]:
    """
    Read the content of the `<meta name=...>` tags of an HTML page.

    :param soup: parsed page
    :param name: value of the `name` attribute, e.g., `citation_doi`
    :return: content of each tag with that name, in the order of the page
    """
    values = []
    for tag in soup.find_all("meta", attrs={"name": name}):
        content = str(tag.get("content", "")).strip()
        if content:
            values.append(content)
    return values
