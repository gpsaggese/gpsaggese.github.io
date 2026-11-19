# Background: Semantic Scholar

- Goal of this file: explain what Semantic Scholar is, and how this pipeline uses it
- Main source: https://www.semanticscholar.org/product/api, read on 2026-11-03

# What It Is

- **Semantic Scholar**: a free search engine and citation graph of scientific papers
  - Web site: https://www.semanticscholar.org
  - Run by Ai2 (the Allen Institute for AI), a non-profit research institute
  - Ai2 describes it as a free, AI-driven tool to find and explore scientific
    literature
- Scope: papers of all fields, with a large share in computer science and biomedicine
- It is not a publisher: it indexes papers that other sites host
  - It stores the metadata, the abstract, the citations, and a link to an open-access
    PDF when one exists
  - It does not own the full text of most papers

# What Data It Holds

- **Paper**: title, year, venue, publication date, publication type, abstract
- **Authors**: names of the authors, with an id for each author
- **Citations**: which paper cites which paper, and the total citation count
- **Identifiers**: the same paper has several ids, and Semantic Scholar maps them
  - `paperId`: id of Semantic Scholar, a 40 character hex string
  - `externalIds`: DOI, arXiv id, PubMed id, DBLP id, and others
- **Open-access link**: `openAccessPdf`, a URL of a free PDF, if one is known
- **Extras**: SPECTER2 embeddings of the papers, and short machine summaries (TLDR)
  - The API page lists SPECTER2 embeddings
  - The pipeline uses none of the extras

# The API

- The Semantic Scholar API has three services
  - Academic Graph API: papers, authors, citations, and venues
  - Recommendations API: papers similar to a given paper
  - Datasets API: bulk download of the whole corpus
- The pipeline uses only the Academic Graph API
  - Base URL: `https://api.semanticscholar.org/graph/v1`
  - Docs: https://api.semanticscholar.org/api-docs/
- Endpoints that the pipeline calls:
  - `GET /paper/{id}`: one paper by id
    - `{id}` is a `paperId`, `DOI:<doi>`, or `ARXIV:<id>`
  - `GET /paper/search`: search by title text
- Fields that the pipeline asks for:
  - `title`, `year`, `authors`, `externalIds`, `openAccessPdf`, `citationCount`,
    `publicationDate`, `publicationTypes`
  - The citation lookup asks only for `citationCount`
- Formats: JSON in the answer, the key in the header `x-api-key`

# Rate Limits and the API Key

- Without a key, all users share one pool of 1,000 requests per second
  - The pool is shared with every anonymous user in the world, so one request can get
    `429 Too Many Requests` even when this pipeline sends very few
  - On 2026-11-02, the first search of every live run got `429`, also with a single
    `curl`
    ([`research/Estimate_academic_paper_quality/docs/07-progress.md`](/research/Estimate_academic_paper_quality/docs/07-progress.md))
- A key is free, and comes by email from the "Request an API key" form
  - Form: https://www.semanticscholar.org/product/api
  - The introductory limit of a key is 1 request per second on all endpoints
  - Never share the key and never commit it
- The pipeline reads the key from the environment variable `SEMANTIC_SCHOLAR_API_KEY`
  ```bash
  > export SEMANTIC_SCHOLAR_API_KEY=<key> ```
- The pipeline works without a key, with fewer lines resolved
  - Without a key, Semantic Scholar was blocked by `429`: Crossref alone resolved 18
    of 22 lines, and 20 of 22 after two fixes
  - The 2 lines left need Semantic Scholar

# How the Pipeline Uses It

- **Resolve stage**
  ([`research/Estimate_academic_paper_quality/resolve_papers.py`](/research/Estimate_academic_paper_quality/resolve_papers.py))
  - Searches a title with `/paper/search` and keeps the top 5 hits
  - Looks up a known DOI or arXiv id with `/paper/{id}`
  - The Semantic Scholar record is the canonical record of a paper: its title, year,
    and first author go into the reference, else Crossref, else arXiv
    ([`research/Estimate_academic_paper_quality/docs/03-decisions.md`](/research/Estimate_academic_paper_quality/docs/03-decisions.md)
    decision `D1`)
  - Why it is first: it merges the preprint and the journal version into one record
- **Download stage**
  ([`research/Estimate_academic_paper_quality/download_papers.py`](/research/Estimate_academic_paper_quality/download_papers.py))
  - `openAccessPdf` is one source of a PDF
  - Order of the sources: the local PDF of the line, the URL of the line, arXiv,
    `openAccessPdf` of Semantic Scholar, Unpaywall, then the SSRN page
- **Report stage**
  ([`research/Estimate_academic_paper_quality/report_papers.py`](/research/Estimate_academic_paper_quality/report_papers.py))
  - `get_citation_count()` is the only source of the citation count
  - It tries `paperId`, then `DOI:<doi>`, then `ARXIV:<id>`
  - If Semantic Scholar does not have the paper, the citation signal is `unknown` and
    is left out of the mean of the rating
    ([`research/Estimate_academic_paper_quality/docs/03-decisions.md`](/research/Estimate_academic_paper_quality/docs/03-decisions.md)
    decision `D16`)
  - The count is stored with its source and with the date of the fetch
- Client code:
  [`research/Estimate_academic_paper_quality/epq_bib_sources.py`](/research/Estimate_academic_paper_quality/epq_bib_sources.py)
  - Throttle: 1 request per second
  - Retry on `429` and `5xx`, with a wait of 2, 4, 8, then 16 seconds
  - Disk cache in `<output_dir>/state/cache/`: a rerun makes no call for a line that
    is cached, and a `404` is cached too
  - After all retries fail, the API is skipped for the rest of the run, and the other
    sources are used
- Design doc:
  [`research/Estimate_academic_paper_quality/docs/02.3-design.bib_sources.md`](/research/Estimate_academic_paper_quality/docs/02.3-design.bib_sources.md)

# Limits of the Data

- **Coverage gaps**: a paper can be missing, mainly a working paper that only SSRN
  hosts
  - A missing paper gives no citation count, so the rating uses fewer signals
- **Citation count is a snapshot**: it grows over time
  - The pipeline stores the date of the fetch next to the count
- **Counts differ between sources**: the count of Semantic Scholar is not the count
  of Google Scholar, Crossref, or OpenAlex
  - Compare counts only from one source
- **Wrong merges and splits**: the graph merges versions of a paper by itself, and
  can make mistakes
  - The pipeline checks the title and the author hints before it accepts a hit
    ([`research/Estimate_academic_paper_quality/docs/03-decisions.md`](/research/Estimate_academic_paper_quality/docs/03-decisions.md)
    decision `D3`)

# Alternatives Not Used

- **Google Scholar**: no API, and it blocks scripts
- **OpenAlex**: open citation graph with an API, not named by the requirements
- **Crossref `is-referenced-by-count`**: counts only citations among Crossref DOIs
- Decision record:
  [`research/Estimate_academic_paper_quality/docs/03-decisions.md`](/research/Estimate_academic_paper_quality/docs/03-decisions.md)
  decision `D16`

# Open Questions

1. What are the limits per endpoint, mainly for `/paper/search`?
   - The API docs page answered `429` to a fetch on 2026-11-03, so the numbers are
     not checked
2. How many papers does the corpus hold now?
3. Which sources feed the corpus (publishers, arXiv, PubMed, Crossref)?
4. Are the extras (TLDR, SPECTER2) still offered on the same terms?
5. Does the license of the API allow the use of the data in a published report?
   - Read the API License Agreement that is linked from
     https://www.semanticscholar.org/product/api
