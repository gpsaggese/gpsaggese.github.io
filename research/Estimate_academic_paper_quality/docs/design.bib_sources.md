# Design: Bib Sources

## Goal

- Give the other components the bibliographic records, the open-access links, and
  the citation counts of the lookup APIs (PRD `R3`, `R4`, `R16`)
  - One client per API: Semantic Scholar, Crossref, arXiv, Unpaywall, plus the
    fetch of the page or PDF of a `url` line
  - Throttle and retry each API, and cache each response on disk, so a rerun makes
    no network call for a line already looked up
- Where it fits: `resolver`, `pipeline` -> `bib_sources` -> lookup APIs
- Overview: `research/Estimate_academic_paper_quality/docs/design.architecture.md`
- Decisions: `research/Estimate_academic_paper_quality/docs/decisions.md`

## Diagram

```mermaid
C4Component
  title Bib Sources: Components
  Container_Boundary(b, "epq_bib_sources.py") {
    Component(api, "BibSources", "Python class", "Public lookups that return BibRecord")
    Component(parse, "_parse_s2() / _parse_crossref() / _parse_arxiv()", "Pure", "Raw JSON or Atom to BibRecord")
    Component(get, "_get()", "requests, hretry, simple_cache", "Throttled, retried, cached GET")
  }
  System_Ext(s2, "Semantic Scholar Graph API", "")
  System_Ext(cr, "Crossref REST API", "")
  System_Ext(ax, "arXiv API", "")
  System_Ext(uw, "Unpaywall API", "")
  System_Ext(web, "Publisher sites", "")
  Rel(api, get, "url, params")
  Rel(api, parse, "raw response")
  Rel(get, s2, "GET /graph/v1/paper/...")
  Rel(get, cr, "GET /works")
  Rel(get, ax, "GET /api/query")
  Rel(get, uw, "GET /v2/<doi>")
  Rel(get, web, "GET <url>")
```

## Files and Import Graph

- `research/Estimate_academic_paper_quality/epq_bib_sources.py`
- Imports `epq_types`, `requests`, `feedparser`, `bs4`, `helpers.hretry`,
  `helpers.hcache_simple`
- **Fact**: the download tool has private clients `_extract_arxiv_metadata()`,
  `_crossref_query()`, `_unpaywall_query()`, with `hretry.sync_retry()`; they
  return only year, authors, and title, with no DOI, no record type, no family
  names, and no citation count
- **Decision**: write new clients here instead of importing the private ones (D-23)
- **Fact**: `hcache_simple.simple_cache(cache_type="json")` caches the return value
  of a function on disk, keyed by its arguments, and `set_global_cache_mode()`
  switches every cached function to `REFRESH_CACHE`, `DISABLE_CACHE`, or
  `HIT_CACHE_OR_ABORT`

## One Request End to End

- `resolver` calls `sources.search_s2("Lazy Prices", limit=5)`:
  - `BibSources` builds the URL of `/graph/v1/paper/search` with the fields
    `title,year,authors,externalIds,openAccessPdf,citationCount,publicationDate`
  - `_get(url, params)`:
    - Returns the cached entry if one exists for `(url, params)`
    - Else waits for the throttle of the host, sends the GET, retries on `429` and
      `5xx` with backoff 2, 4, 8, 16 s
    - Caches the entry `{"status", "body", "fetched_at"}`: a `404` is cached too,
      since "not found" is a stable answer
    - Raises after the last retry: the error is not cached, so a rerun tries again
  - `_parse_s2()` turns each hit into a `BibRecord`
- `pipeline` calls `sources.get_citation_count(record)`:
  - Looks up Semantic Scholar by `paperId`, else by `DOI:` or `ARXIV:`
  - Returns `CitationCount(count, source="Semantic Scholar", fetched_at)`, where
    `fetched_at` comes from the cache entry, not from the clock of the run
  - Returns `None` when Semantic Scholar does not have the paper (D-16)
- `resolver` calls `sources.fetch_url_meta(url)` for a `url` line:
  - PDF answer (`%PDF` at the start): saves it to `state/cache/url_pdfs/<sha1>.pdf`
    and returns the path, so `resolver` reads its title
  - HTML answer: returns the `citation_doi`, `citation_title`, and
    `citation_author` meta tags, which most publishers set

## Configuration, Interfaces, and Known Limits

- Configuration:

  | Setting                          | Value                    | Why                                   |
  | :------------------------------- | :----------------------- | :------------------------------------ |
  | Throttle Semantic Scholar        | 1 request per second     | Limit of a keyed account              |
  | Throttle arXiv                   | 1 request per 3 seconds  | arXiv API terms of use                |
  | Throttle Crossref                | 1 request per second, `mailto=<email>` | Polite pool              |
  | `SEMANTIC_SCHOLAR_API_KEY`       | env var, optional        | Fewer `429` than the shared pool      |
  | `--email`                        | from `PipelineConfig`    | Unpaywall needs a real email          |
  | Cache dir                        | `<output_dir>/state/cache/` | The state lives in the output dir (D-18) |

- Interfaces:
  ```python
  class BibSources:
      def __init__(self, *, email: str, s2_api_key: str) -> None: ...
      def lookup_s2(self, paper_id: str) -> Optional[epqtypes.BibRecord]: ...
      def search_s2(self, query: str, limit: int) -> List[epqtypes.BibRecord]: ...
      def lookup_crossref(self, doi: str) -> Optional[epqtypes.BibRecord]: ...
      def search_crossref(self, query: str, limit: int) -> List[epqtypes.BibRecord]: ...
      def lookup_arxiv(self, arxiv_id: str) -> Optional[epqtypes.BibRecord]: ...
      def lookup_unpaywall(self, doi: str) -> Optional[str]: ...
      def fetch_url_meta(self, url: str) -> epqtypes.UrlMeta: ...
      def get_citation_count(
          self, record: epqtypes.BibRecord
      ) -> Optional[epqtypes.CitationCount]: ...
  ```
- `BibRecord` keeps the fields that `resolver` and `rater` need:
  `source`, `title`, `authors` (family names first), `year`, `record_type`, `doi`,
  `arxiv_id`, `ssrn_id`, `s2_paper_id`, `pdf_urls`
- Known limits:
  - The cache never expires: `--cache_mode REFRESH_CACHE` refetches everything
  - Without `SEMANTIC_SCHOLAR_API_KEY`, the shared pool often answers `429`, and a
    run of 100 lines takes longer

## Deliberately Not Built

- Google Scholar: it has no API and blocks scripts (D-16)
- OpenAlex, DBLP, or other sources: `R3` names arXiv, Crossref, and Semantic
  Scholar

## Open Questions

1. Check the current rate limits of Semantic Scholar and Crossref before the first
   run, since both change over time
