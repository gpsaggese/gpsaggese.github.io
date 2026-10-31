# Estimate Paper Quality Decisions

- Every non-obvious choice of the design, why it was made, and what else was
  considered
- Newest at the bottom of each section
- Requirements:
  [`research/Estimate_academic_paper_quality/docs/01-prd.md`](/research/Estimate_academic_paper_quality/docs/01-prd.md)
- Design:
  [`research/Estimate_academic_paper_quality/docs/02-design.architecture.md`](/research/Estimate_academic_paper_quality/docs/02-design.architecture.md)

## Resolution

### `D1` (Semantic Scholar record is the canonical record)

- **Decision**: build the reference from the Semantic Scholar record; else from the
  Crossref record of type `journal-article`; else any Crossref record; else the arXiv
  record
- **Why**: Semantic Scholar merges the preprint and the journal version of a paper
  into one record with the DOI, the arXiv id, an open-access link, and the citation
  count, so one record serves
  [`R3`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r3-bibliographic-source),
  [`R4`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r4-download-link), and
  [`R16`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r16-paper-quality-rating)
- **Alternatives**: Crossref first (no merge of versions, no open-access link); the
  record that the line names (an SSRN line would show the SSRN year, not the journal
  year)
- **Status**: proposed

### `D2` (Paper identity is the normalized title plus the first author)

- **Decision**: `paper_key()` is the normalized title plus the last name of the first
  author; the DOI, the arXiv id, and the Semantic Scholar id are aliases
- **Why**: finance papers have an SSRN DOI, an NBER DOI, and a journal DOI for the
  same paper: a key on the DOI keeps three rows for one paper and breaks
  [`R6`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r6-duplicate-lines)
- **Alternatives**: the DOI; the Semantic Scholar id (missing for papers that only
  Crossref has)
- **Status**: proposed

### `D3` (A line matches when one distinct paper passes the title test)

- **Decision**: keep the candidates with title similarity at least `0.9` that share a
  last name with the author hints, group them by `paper_key()`, and match only when
  one group is left, or when exactly one group has similarity `1.0`
- **Why**:
  [`R5`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r5-unresolved-or-ambiguous-line)
  needs a rule that a test can check; a prefix counts as `0.95`, so a partial title
  ([`R1`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r1-input-file-format))
  still matches; the author hints of
  [`research/Estimate_academic_paper_quality/paper_list.md`](/research/Estimate_academic_paper_quality/paper_list.md)
  split papers with the same title
- **Alternatives**: take the top hit of Semantic Scholar (silent wrong matches); an
  LLM to pick the candidate
  ([`R3`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r3-bibliographic-source)
  forbids the LLM as a source)
- **Status**: proposed

### `D4` (The title of a PDF is the largest text of page 1)

- **Decision**: for a local PDF or a direct PDF URL, read the title from the spans of
  page 1 with the largest font, else from the PDF metadata, then look it up as a
  title line
- **Why**: **Fact**: a local PDF often has no metadata title (e.g., SSRN PDFs, per
  the comment in `_get_output_base_path()` of the download tool); the facts of the
  reference must still come from a lookup source
  ([`R3`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r3-bibliographic-source))
- **Alternatives**: the file name (often `ssrn_5277078.pdf`); an LLM on page 1
- **Status**: proposed

### `D5` (Each paper keeps an ordered list of download inputs)

- **Decision**: the order is the local PDF, the URL of the line, arXiv, the
  `openAccessPdf` of Semantic Scholar, Unpaywall, then the SSRN page; `fetcher` tries
  them in order until one gives a valid PDF
- **Why**: one link fails often (paywall, `403`, HTML page); a fallback raises the
  Download coverage metric; what the user gave comes first
- **Alternatives**: one link per paper (lower coverage); let the download tool pick
  from the DOI (it tries only Unpaywall)
- **Status**: proposed

### `D6` (A reference with no download link ends with ` ???`)

- **Decision**: with no public download input, the link is `<DOI URL> ???`, or `???`
  alone with no DOI
- **Why**:
  [`.claude/skills/references.rules.md`](/.claude/skills/references.rules.md) marks a
  link that does not work with ` ???`;
  [`R4`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r4-download-link)
  needs a mark
- **Alternatives**: an empty link; a new mark such as `(no PDF)`
- **Status**: proposed

## Download and Files

### `D7` (The pipeline computes the base name with a public helper)

- **Decision**: move `_format_base_filename()` of the download tool to
  `download_utils.format_base_filename()`; make it take the family name, drop
  suffixes like `Jr.`, and replace `.` `,` `'` with `_`; the pipeline passes the
  result as `--output`
- **Why**: **Fact**: the tool takes the last word of the first author and keeps `.`
  and `,` in the title, against
  [`.claude/skills/references.rules.md`](/.claude/skills/references.rules.md) `#
  Format to Use in File Names`
  ([`R9`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r9-file-base-name));
  one function keeps the name of the reference and of the files the same
- **Alternatives**: let the tool derive the name (it calls the APIs again, and the
  name can differ from the reference); import the private function (breaks the import
  rules)
- **Status**: proposed (changes the names of new files of the tool too)

### `D8` (Run the download tool as a subprocess, without `summarize`)

- **Decision**: run
  [`download_academic_paper_to_md.py`](/helpers_root/dev_scripts_helpers/download/download_academic_paper_to_md.py)
  with `--skip_action summarize`, one subprocess per paper and per download input
- **Why**: **Fact**: `summarize` is a default action and calls an LLM through
  OpenRouter, a cost that the pipeline does not need; a subprocess keeps the inline
  `uv` dependencies of the tool and turns its `dassert` into an exit code, so one
  failure does not stop the run
  ([`R10`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r10-failed-download-or-conversion))
- **Alternatives**: import the tool (private functions, shared process state); run
  `summarize` and use the summary (cost on every paper)
- **Status**: proposed

### `D9` (Check the `%PDF` header after a download)

- **Decision**: a downloaded file that does not start with `%PDF` is deleted and
  counts as a failed download input
- **Why**: **Fact**: the tool saves the body of any `200` answer as the PDF, so a
  landing page becomes an HTML file named `.pdf`, and the conversion then fails with
  an unclear reason
- **Alternatives**: fix the tool to check the content type (a change for all its
  users; can come later)
- **Status**: proposed

### `D10` (The user gives the output dir)

- **Decision**: `--output_dir` is required and has no default; the user decides if it
  is in the repo and if the files are committed
- **Why**: about 100 papers make 100 MB to 500 MB of PDFs; the user picked this
  option over a git-ignored dir of the project
- **Alternatives**: `research/Estimate_academic_paper_quality/runs/<input>/`,
  git-ignored; the same, committed
- **Status**: accepted

## Extraction

### `D11` (One structured LLM call per paper)

- **Decision**: one call of `hllm.get_structured_completion()` per paper returns all
  the features, the facts, and the affiliations, parsed into a pydantic model
- **Why**: the paper text is most of the tokens, so one call costs about a quarter of
  four calls; a parsed answer fails loud instead of giving bad JSON
- **Alternatives**: one call per feature; `hllm_cli.apply_llm()` with JSON in free
  text (no schema check)
- **Status**: proposed

### `D12` (The page of a quote comes from the PDF text)

- **Decision**: look for each quote in the text of each PDF page with `pymupdf`; when
  found, add `p. <N>` and mark `verified=True`; else keep the value with the section
  that the LLM gave and mark `verified=False`
- **Why**: **Fact**:
  [`convert_pdf_to_md.py`](/helpers_root/dev_scripts_helpers/documentation/convert_pdf_to_md.py)
  `remove_junk` deletes the `<!-- Page N
  -->` markers, so the markdown has no page; the search also catches a quote that the
  LLM made up
- **Alternatives**: keep the page markers (a change of the tool); turn an unverified
  value into `unknown` (drops right values that the PDF text breaks)
- **Status**: proposed

### `D13` (A long paper loses its references, then its middle)

- **Decision**: drop the reference list; if the text is still over `--max_chars`,
  keep the first 60% and the last 40% of the budget
- **Why**: finance papers often put the tables of results after the references, at
  the end; the middle has the least of the values of
  [`R15`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r15-extracted-facts)
- **Alternatives**: cut the end (loses the tables); send everything (cost and context
  limits)
- **Status**: proposed

### `D14` (The features file is JSON and is the rerun cache of `extract`)

- **Decision**: write `<BASE>.features.json` with a `schema_version`; `pipeline`
  skips `extract` for a paper whose file exists
- **Why**: one file serves
  [`R20`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r20-features-file)
  and
  [`R17`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r17-rerun-skips-extraction);
  JSON reads back with no parser of our own; the suffix collides with no output of
  the tool or of `paper.rate`
- **Alternatives**: markdown (easier to read, hard to read back); a separate cache
- **Status**: proposed

## Rating

### `D15` (Affiliation tiers come from a checked-in dated list)

- **Decision**:
  [`affiliation_tiers.yaml`](/research/Estimate_academic_paper_quality/affiliation_tiers.yaml)
  lists the entries of scores 5 and 4, with `as_of` and `source`, filled by hand from
  the QS site; the LLM only reads the names and the kind of each institution from the
  paper
- **Why**: [`.claude/skills/paper.rules.md`](/.claude/skills/paper.rules.md) forbids
  facts from memory; a list is deterministic and easy to audit; the user picked this
  option
- **Alternatives**: the LLM scores the affiliation with the table (the rank comes
  from memory)
- **Status**: accepted

### `D16` (Citations come from Semantic Scholar only)

- **Decision**: when Semantic Scholar does not have the paper, the citation signal is
  `unknown` and is left out of the mean
- **Why**: the rules name Google Scholar as the fallback, but it has no API and
  blocks scripts; the user picked this option
- **Alternatives**: OpenAlex `cited_by_count`; Crossref `is-referenced-by-count`
  (both are sources that the rules do not name)
- **Status**: accepted

### `D17` (Round half up with `math.floor(mean + 0.5)`)

- **Decision**: the rating is `math.floor(mean + 0.5)` of the known scores
- **Why**: **Fact**: Python `round()` rounds half to even (`round(2.5) == 2`),
  against "rounded half up" of
  [`R16`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r16-paper-quality-rating)
- **Alternatives**: `decimal.ROUND_HALF_UP` (same result, more code)
- **Status**: proposed

## Rerun and Output

### `D18` (All the state lives in the output dir)

- **Decision**: the outputs of the stages go to `<output_dir>/state/`, and every HTTP
  response of `bib_sources` is cached in `<output_dir>/state/cache/`
- **Why**: a rerun on the same dir makes no network call for old lines and no LLM
  call for old papers
  ([`R11`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r11-rerun-skips-converted-papers),
  [`R17`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r17-rerun-skips-extraction));
  each stage can run alone; one dir holds one run, so nothing leaks between lists
- **Alternatives**: a cache per stage (more code); a global cache in the repo (mixes
  runs)
- **Status**: proposed

### `D19` (`report.csv` has a row only for a paper with a features file)

- **Decision**: a paper that failed to download, convert, or extract is only in
  `issues.csv` and `references.md`
- **Why**:
  [`R12`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r12-feature-list)
  reports each converted paper; a row of `unknown` values hides the reason, which
  `issues.csv` gives
  ([`R19`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r19-issues-csv))
- **Alternatives**: one row per resolved paper, with `unknown` features
- **Status**: proposed

### `D20` (The normalized list also goes to `references.md`)

- **Decision**: write the references of all resolved papers, in
  [`R7`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r7-reverse-chronological-order)
  order, to `<output_dir>/references.md`
- **Why**: the use case "gets one clean list of references" includes the papers that
  failed to download, which are not in `report.csv`
- **Alternatives**: only the `reference` column of `report.csv`
- **Status**: proposed

## Code Organization

### `D21` (The loop of each stage catches errors per paper)

- **Decision**: the inner modules assert and raise; each stage of `pipeline` catches
  `Exception` per paper and records an `Issue` with the last line of the message
- **Why**:
  [`R5`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r5-unresolved-or-ambiguous-line)
  and
  [`R10`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r10-failed-download-or-conversion)
  need the run to go on; one catch point keeps the inner code free of error plumbing,
  per [`.claude/skills/architecture.rules.md`](/.claude/skills/architecture.rules.md)
- **Alternatives**: return codes in every module
- **Status**: proposed

### `D22` (Flat modules with the `epq_` prefix)

- **Decision**: put the script and the `epq_*.py` modules flat in
  [`research/Estimate_academic_paper_quality/`](/research/Estimate_academic_paper_quality/),
  tests in `test/`
- **Why**: seven small modules do not need a dir each; the prefix follows the
  `noesis_*.py` modules of [`research/Noesis/common/`](/research/Noesis/common/)
- **Alternatives**: one dir per component, as in
  [`research/Noesis/`](/research/Noesis/)
- **Status**: proposed

### `D23` (New API clients instead of the private ones of the download tool)

- **Decision**:
  [`epq_bib_sources.py`](/research/Estimate_academic_paper_quality/epq_bib_sources.py)
  has its own clients for arXiv, Crossref, Semantic Scholar, and Unpaywall
- **Why**: **Fact**: the clients of the tool are private and return only year,
  authors, and title; the resolver needs the DOI, the record type, the family names,
  the open-access links, and the citations, plus a disk cache
- **Alternatives**: make the clients of the tool public and extend them (a bigger
  change to a tool with other users)
- **Status**: proposed
