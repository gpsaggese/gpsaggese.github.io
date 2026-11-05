# Estimate Paper Quality Decisions

- Every non-obvious choice of the design, why it was made, and what else was
  considered
- Newest at the bottom of each section
- Requirements:
  [`research/Estimate_academic_paper_quality/docs/01-prd.md`](/research/Estimate_academic_paper_quality/docs/01-prd.md)
- Design:
  [`research/Estimate_academic_paper_quality/docs/02-design.architecture.md`](/research/Estimate_academic_paper_quality/docs/02-design.architecture.md)

## Resolution

### `D1` (Semantic Scholar Record Is the Canonical Record)

- **Decision**: build the reference from the Semantic Scholar record; else from the
  Crossref record of type `journal-article`; else any Crossref record; else the arXiv
  record
- **Why**: Semantic Scholar merges the preprint and the journal version of a paper
  into one record with the DOI, the arXiv id, an open-access link, and the citation
  count, so one record serves
  [`R3`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r3-bibliographic-source),
  [`R4`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r4-download-link),
  and
  [`R16`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r16-paper-quality-rating)
- **Alternatives**: Crossref first (no merge of versions, no open-access link); the
  record that the line names (an SSRN line would show the SSRN year, not the journal
  year)
- **Status**: proposed

### `D2` (Paper Identity Is the Normalized Title Plus the First Author)

- **Decision**: `paper_key()` is the normalized title plus the last name of the first
  author; the DOI, the arXiv id, and the Semantic Scholar id are aliases
- **Why**: finance papers have an SSRN DOI, an NBER DOI, and a journal DOI for the
  same paper: a key on the DOI keeps three rows for one paper and breaks
  [`R6`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r6-duplicate-lines)
- **Alternatives**: the DOI; the Semantic Scholar id (missing for papers that only
  Crossref has)
- **Status**: proposed

### `D3` (A Line Matches When One Distinct Paper Passes the Title Test)

- **Decision**: keep the candidates with title similarity at least `0.9` that share a
  last name with the author hints, group them by `paper_key()`, and match only when
  one group is left, or when exactly one group has similarity `1.0`
- **Why**:
  [`R5`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r5-unresolved-or-ambiguous-line)
  needs a rule that a test can check; a prefix counts as `0.95`, so a partial title
  ([`R1`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r1-input-file-format))
  still matches; the author hints of
  [`research/Estimate_academic_paper_quality/paper_list.csv`](/research/Estimate_academic_paper_quality/paper_list.csv)
  split papers with the same title
- **Alternatives**: take the top hit of Semantic Scholar (silent wrong matches); an
  LLM to pick the candidate
  ([`R3`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r3-bibliographic-source)
  forbids the LLM as a source)
- **Status**: proposed

### `D4` (The Title of a PDF Is the Largest Text of Page 1)

- **Decision**: for a local PDF or a direct PDF URL, read the title from the spans of
  page 1 with the largest font, else from the PDF metadata, then look it up as a
  title line
- **Why**: **Fact**: a local PDF often has no metadata title (e.g., SSRN PDFs, per
  the comment in `_get_output_base_path()` of the download tool); the facts of the
  reference must still come from a lookup source
  ([`R3`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r3-bibliographic-source))
- **Alternatives**: the file name (often `ssrn_5277078.pdf`); an LLM on page 1
- **Status**: proposed

### `D5` (Each Paper Keeps an Ordered List of Download Inputs)

- **Decision**: the order is the local PDF, the URL of the line, arXiv, the
  `openAccessPdf` of Semantic Scholar, Unpaywall, then the SSRN page; `fetcher` tries
  them in order until one gives a valid PDF
- **Why**: one link fails often (paywall, `403`, HTML page); a fallback raises the
  Download coverage metric; what the user gave comes first
- **Alternatives**: one link per paper (lower coverage); let the download tool pick
  from the DOI (it tries only Unpaywall)
- **Status**: proposed

### `D6` (A Reference with No Download Link Ends with ` ???`)

- **Decision**: with no public download input, the link is `<DOI URL> ???`, or `???`
  alone with no DOI
- **Why**:
  [`.claude/skills/references.rules.md`](/.claude/skills/references.rules.md) marks a
  link that does not work with ` ???`;
  [`R4`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r4-download-link)
  needs a mark
- **Alternatives**: an empty link; a new mark such as `(no PDF)`
- **Status**: proposed

### `D25` (The Parser Joins the Lines of a Wrapped Bullet)

- **Decision**: `parse_lines()` first joins a bullet that is wrapped on several
  lines, with a marker `-` or `*` and an indented continuation, into one entry,
  and numbers the entry with its first line
- **Why**: **Fact**:
  the former `research/Estimate_academic_paper_quality/paper_list.md` (now
  [`paper_list.csv`](/research/Estimate_academic_paper_quality/paper_list.csv))
  was wrapped at 85 chars, so the authors in parentheses are on the line after
  the title; parsed line by line, the authors are a `title` line that matches no
  paper, and
  [`R1`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r1-input-file-format)
  would fail on the real input
- **Alternatives**: require one paper per physical line (the user unwraps the
  file by hand); join by a rule on the parentheses (breaks for titles with
  parentheses)
- **Status**: proposed

### `D26` (The Resolver Is a Separate Executable That Reads and Writes CSV)

- **Decision**: `resolve_papers.py` reads an input CSV with a column `input` and
  writes a papers CSV and an issues CSV; the pipeline script of that time
  (`estimate_paper_quality.py`, replaced by the scripts of
  [`D27`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d27-the-pipeline-is-split-into-one-executable-per-stage))
  has no `resolve` stage, and reads the papers CSV with `--input` and the issues CSV with
  `--resolve_issues`
- **Why**: a person can open, sort, and fix the papers CSV before the costly stages
  (download, LLM); the resolver can run alone on a new list; a CSV needs no state
  dir, so [`D18`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d18-all-the-state-lives-in-the-output-dir)
  now covers the stages `download`, `extract`, and `report` only
- **Alternatives**: keep the stage `resolve` in the pipeline and add a second CLI that
  calls it (two paths to the same code); write `state/papers.json` and read a CSV
  only as an input (two formats for the same data)
- **Consequences**:
  - The input is not the markdown bullet list anymore: the joiner of
    [`D25`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d25-the-parser-joins-the-lines-of-a-wrapped-bullet)
    is not used by the executable, so a bullet list becomes a CSV first (e.g.,
    [`paper_list.csv`](/research/Estimate_academic_paper_quality/paper_list.csv))
  - The `line_no` of an issue is the row of the input CSV
  - A list is one CSV cell with the items joined by `|`: an item that holds `|` breaks
    on the read
- **Status**: superseded by
  [`D31`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d31-json-for-the-intermediate-files): the papers and the issues are
  JSON files, and the resolver writes them in an output dir

## Download and Files

### `D7` (The Pipeline Computes the Base Name with a Public Helper)

- **Decision**: move `_format_base_filename()` of the download tool to
  `download_utils.format_base_filename()`; make it take the family name, drop
  suffixes like `Jr.`, and replace `.` `,` `'` with `_`; the pipeline passes the
  result as `--output`
- **Why**: **Fact**: the tool takes the last word of the first author and keeps `.`
  and `,` in the title, against
  [`.claude/skills/references.rules.md`](/.claude/skills/references.rules.md)
  `# Format to Use in File Names`
  ([`R9`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r9-file-base-name));
  one function keeps the name of the reference and of the files the same
- **Alternatives**: let the tool derive the name (it calls the APIs again, and the
  name can differ from the reference); import the private function (breaks the import
  rules)
- **Status**: proposed (changes the names of new files of the tool too)

### `D8` (Run the Download Tool as a Subprocess, Without `summarize`)

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

### `D9` (Check the `%PDF` Header After a Download)

- **Decision**: a downloaded file that does not start with `%PDF` is deleted and
  counts as a failed download input
- **Why**: **Fact**: the tool saves the body of any `200` answer as the PDF, so a
  landing page becomes an HTML file named `.pdf`, and the conversion then fails with
  an unclear reason
- **Alternatives**: fix the tool to check the content type (a change for all its
  users; can come later)
- **Status**: proposed

### `D10` (The User Gives the Output Dir)

- **Decision**: `--output_dir` is required and has no default; the user decides if it
  is in the repo and if the files are committed
- **Why**: about 100 papers make 100 MB to 500 MB of PDFs; the user picked this
  option over a git-ignored dir of the project
- **Alternatives**: `research/Estimate_academic_paper_quality/runs/<input>/`,
  git-ignored; the same, committed
- **Consequences**: refined by
  [`D30`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d30-output_dir-for-several-files-and-output-for-one): all the
  executables take `--output_dir`, including the resolver
- **Status**: accepted

## Extraction

### `D11` (One Structured LLM Call Per Paper)

- **Decision**: one call of `hllm_cli.apply_llm()` per paper returns all the features,
  the facts, and the affiliations as JSON. The system prompt has the JSON schema of
  a pydantic model, and the script validates the answer with that model. The old
  `helpers.hllm` is obsolete and the pipeline does not use it
- **Why**: the paper text is most of the tokens, so one call costs about a quarter of
  four calls; a validated answer fails loud instead of giving bad JSON, and
  `hllm_cli` serves any provider of the `llm` library
- **Alternatives**: one call per feature; `hllm.get_structured_completion()`, which
  enforces the schema in the API but serves only OpenAI (decision of the user: use
  only `hllm_cli`)
- **Status**: accepted

### `D12` (The Page of a Quote Comes From the PDF Text)

- **Decision**: look for each quote in the text of each PDF page with `pymupdf`; when
  found, add `p. <N>` and mark `verified=True`; else keep the value with the section
  that the LLM gave and mark `verified=False`
- **Why**: **Fact**:
  [`convert_pdf_to_md.py`](/helpers_root/dev_scripts_helpers/documentation/convert_pdf_to_md.py)
    `remove_junk` deletes the `<!-- Page N
  -->` markers, so the markdown has no page; the search also catches a quote that the LLM made up
- **Alternatives**: keep the page markers (a change of the tool); turn an unverified
  value into `unknown` (drops right values that the PDF text breaks)
- **Status**: proposed

### `D13` (A Long Paper Loses Its References, Then Its Middle)

- **Decision**: drop the reference list; if the text is still over `--max_chars`,
  keep the first 60% and the last 40% of the budget
- **Why**: finance papers often put the tables of results after the references, at
  the end; the middle has the least of the values of
  [`R15`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r15-extracted-facts)
- **Alternatives**: cut the end (loses the tables); send everything (cost and context
  limits)
- **Status**: proposed

### `D14` (The Features File Is JSON and Is the Rerun Cache of `extract`)

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

### `D15` (Affiliation Tiers Come From a Checked-in Dated List)

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

### `D16` (Citations Come From Semantic Scholar Only)

- **Decision**: when Semantic Scholar does not have the paper, the citation signal is
  `unknown` and is left out of the mean
- **Why**: the rules name Google Scholar as the fallback, but it has no API and
  blocks scripts; the user picked this option
- **Alternatives**: OpenAlex `cited_by_count`; Crossref `is-referenced-by-count`
  (both are sources that the rules do not name)
- **Status**: accepted

### `D17` (Round Half Up with `math.floor(mean + 0.5)`)

- **Decision**: the rating is `math.floor(mean + 0.5)` of the known scores
- **Why**: **Fact**: Python `round()` rounds half to even (`round(2.5) == 2`),
  against "rounded half up" of
  [`R16`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r16-paper-quality-rating)
- **Alternatives**: `decimal.ROUND_HALF_UP` (same result, more code)
- **Status**: proposed

## Rerun and Output

### `D18` (All the State Lives in the Output Dir)

- **Decision**: the outputs of the stages go to `<output_dir>/`, with the names of
  [`D29`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d29-numbered-stage-files-in-the-output-dir), and every HTTP
  response of `bib_sources` is cached in `<output_dir>/state/cache/`
  - `state/` holds the cache and the logs of the download tool only: the issues of a
    stage are in `<output_dir>/NN-<stage>.issues.json`
- **Why**: a rerun on the same dir makes no network call for old lines and no LLM
  call for old papers
  ([`R11`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r11-rerun-skips-converted-papers),
  [`R17`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r17-rerun-skips-extraction));
  each stage can run alone; one dir holds one run, so nothing leaks between lists
- **Alternatives**: a cache per stage (more code); a global cache in the repo (mixes
  runs)
- **Status**: proposed (amended by
  [`D29`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d29-numbered-stage-files-in-the-output-dir): the stage issues left
  `state/`)

### `D19` (`report.json` Has a Row Only for a Paper with a Features File)

- **Decision**: a paper that failed to download, convert, or extract is only in
  `issues.json` and `references.md`
- **Why**:
  [`R12`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r12-feature-list)
  reports each converted paper; a row of `unknown` values hides the reason, which
  `issues.json` gives
  ([`R19`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r19-issues-file))
- **Alternatives**: one row per resolved paper, with `unknown` features
- **Status**: superseded by
  [`D35`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d35-the-report-has-one-row-per-paper-or-unresolved-line-with-its-issue): a failed paper has a row, with its issue

### `D20` (The Normalized List Also Goes to `references.md`)

- **Decision**: write the references of all resolved papers, in
  [`R7`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r7-reverse-chronological-order)
  order, to `<output_dir>/04-report.references.md`
- **Why**: the use case "gets one clean list of references" includes the papers that
  failed to download, which are not in `report.json`
- **Alternatives**: only the `reference` column of `report.json`
- **Status**: proposed

### `D24` (BibSources Keeps Its Own Disk Cache)

- **Decision**: `BibSources` stores each answer in
  `<output_dir>/state/cache/http/<sha1>.json`, with its own code, and reads the
  switch `--cache_mode` as `hcache_simple` does (`REFRESH_CACHE`,
  `DISABLE_CACHE`, `HIT_CACHE_OR_ABORT`); it does not use `simple_cache()`
- **Why**: **Fact**: `simple_cache()` keeps one process-wide dir and one in-memory
  dict by function name, and it writes the whole file of a function on each new
  answer; a client that takes its cache dir in the constructor has no global
  state, and a test points it to the scratch space; the cache entry also needs the
  date of the lookup, which `get_citation_count()` reports
  ([`R16`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r16-paper-quality-rating))
- **Alternatives**: `simple_cache()` on a module function (global state in the
  tests, one JSON file for all the answers)
- **Status**: proposed

### `D29` (Numbered Stage Files in the Output Dir)

- **Decision**: every file that a stage writes in `<output_dir>` is named
  `NN-<stage>.<what>.<ext>`
  - `NN` is the order of the executable: `resolve` is `01`, `download` `02`,
    `extract` `03`, and `rate` and `report` `04`, since both are stages of
    `report_papers.py`
  - `00-paper_list.csv` is the copy of the input CSV that `resolve_papers.py` makes
  - The issues of a stage are in the output dir, not in `state/`
  - The files of one paper stay in `papers/<BASE>.<ext>` with no number
  - `epq_paths.py` is the only place that names the files
- **Why**: a listing of the output dir reads in the order of the pipeline and shows
  which executable wrote each file; the old names (`papers.csv`,
  `state/issues.download.json`, `report.json`) had no order and mixed the results with
  the state; the writer and the reader of a file take the name from one getter
  (rule `ARCH3.6` in `.claude/skills/architecture.rules.md`)
- **Alternatives**: a subdir per stage (`02-download/`), which splits the files of one
  paper across 3 dirs; a number on the files of `papers/`, which makes long names,
  since the suffix (`.features.json`) already says the stage
- **Consequences**:
  - Old runs are not migrated: a rerun on an old dir writes the new files and does not
    read the old ones
  - In the docs, `report.json`, `issues.json`, and `references.md` are the short names of
    `04-report.report.json`, `04-report.issues.json`, and `04-report.references.md`
    ([`D35`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d35-the-report-has-one-row-per-paper-or-unresolved-line-with-its-issue) removed `04-report.issues.json`: `issues.json` now names the files
    `NN-<stage>.issues.json`)
  - Amends
    [`D18`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d18-all-the-state-lives-in-the-output-dir)
- **Status**: accepted

### `D30` (`--output_dir` for Several Files and `--output` for One)

- **Decision**: an executable that writes several files takes `--output_dir` only, and
  infers the names by
  [`D29`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d29-numbered-stage-files-in-the-output-dir)
  - An executable that writes one file takes either `--output_dir` (the name is
    inferred) or `-o/--output` (the path of the file), and exactly one of them
  - A later stage takes `--input`, and its default is the papers file in `--output_dir`
  - The 4 executables write several files, so none takes `--output` today
  - `resolve_papers.py` lost `--output` and `--issues_output`, and `report_papers.py`
    lost `--resolve_issues`
  - The parser of `resolve_papers.py` has `allow_abbrev=False`, so the old `--output`
    is an error and not a short form of `--output_dir`
- **Why**: the user decides where the files go
  ([`D10`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d10-the-user-gives-the-output-dir)) and the executable decides the
  names, so there is no file name to type or to get wrong; the stage `report` finds the
  issues of `resolve` without a flag; this is rule `ARCH3.7` in
  `.claude/skills/architecture.rules.md`
- **Alternatives**: keep `--output` for the papers file of the resolver and derive the
  issues file from it (the later stages need the same names, so they would need flags
  too); keep `--input` required (the path is typed 3 times per run)
- **Consequences**: the command
  `resolve_papers.py --output ~/papers/run_2026_10/papers.csv` is now
  `resolve_papers.py --output_dir ~/papers/run_2026_10`; refines
  [`D10`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d10-the-user-gives-the-output-dir)
- **Status**: accepted

### `D31` (JSON for the Intermediate Files)

- **Decision**: the papers list and the issues of every stage, `resolve` too, are JSON
  files with a `schema_version`
  - CSV stays for the input list that the user writes (amended by
    [`D34`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d34-the-report-is-json):
    `04-report.report.json` was a CSV when this decision was made)
  - `04-report.issues.json`, the issues of every stage, was JSON too (task 8): it had
    the layout of the files `NN-<stage>.issues.json`, so `epq_report.read_issues()`
    read it and `write_all_issues()` wrote it, sorted by stage and then by line
    (removed by [`D35`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d35-the-report-has-one-row-per-paper-or-unresolved-line-with-its-issue): the issues are in the rows of `04-report.report.json`)
  - `epq_report.write_papers()` and `read_papers()` replace the functions of the papers
    CSV; `write_issues_csv()` and `read_issues_csv()` are removed
- **Why**: a list (the authors, the aliases, the download inputs, the line numbers) is
  a JSON array, not a cell with items joined by `|` that breaks when an item holds
  `|`; the issues of all the stages have one format; people and programs both read
  JSON; this is rule `ARCH3.8` in `.claude/skills/architecture.rules.md`
- **Alternatives**: keep the papers CSV, which a person can sort and fix in a
  spreadsheet (the reason of
  [`D26`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d26-the-resolver-is-a-separate-executable-that-reads-and-writes-csv));
  write both (two formats for the same data)
- **Consequences**:
  - The papers file cannot be sorted in a spreadsheet, but it can be edited by hand
  - A file with another `schema_version` is rejected with a message that names the
    executable to run again
  - Supersedes
    [`D26`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d26-the-resolver-is-a-separate-executable-that-reads-and-writes-csv)
- **Status**: accepted

### `D33` (Each Stage Ends with a Summary in the Log and in a File)

- **Decision**: each stage ends with a summary that goes to the log and, with the
  same text, to `<output_dir>/NN-<stage>.summary.txt`
  - The summary has the items in the input, the items skipped, the items processed
    in the run, the items completed and failed, the items completed in total, the
    output files with the number of files written in the run, and the path of the
    issues file with the number of issues by status
  - The items are lines for `resolve` and papers for the other stages
  - A paper is in one group: done before (the output of an earlier run exists),
    no input (an earlier stage failed for it), completed, or failed
  - A dry run logs what it would process and writes no file
  - `epq_summary.py` formats and writes the text from a `StageSummary`, and each
    executable counts its items in its own loop
- **Why**: the old end of a stage was one line, e.g., `Download: 2 of 20 papers
  failed`, which did not say where the files and the issues are, or how many papers
  the run processed versus skipped; the file keeps the answer after the terminal is
  closed, and the stages run as subprocesses of `estimate_paper_quality.py`, so each
  one needs its own summary
- **Alternatives**: a JSON summary (the user reads this one, and no stage reads it);
  compute the counts from the files of the output dir after the run (it cannot tell
  the papers processed in the run from the papers done before); one summary at the
  end of `estimate_paper_quality.py` (it would not exist when the user runs a single
  executable)
- **Consequences**:
  - `_resolve()` takes `output_dir` and infers the papers file and the issues file,
    like the other stages
  - A rerun overwrites the summary of the stage, so the file describes the last run
  - Extends
    [`D29`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d29-numbered-stage-files-in-the-output-dir)
- **Status**: accepted

### `D34` (The Report Is JSON)

- **Decision**: `04-report.report.json` replaces `04-report.report.csv`: a JSON file
  with a `schema_version` and the key `papers`, which holds one entry per paper with a
  features file and a rating file (the key and the rows changed with
  [`D35`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d35-the-report-has-one-row-per-paper-or-unresolved-line-with-its-issue))
  - An entry has the keys that were the columns of the CSV, in the same order
  - An unknown rating or score is `null`, not the text `unknown`; a feature keeps its
    text, `unknown` included
  - The pipeline writes no CSV: `00-paper_list.csv` is a verbatim copy of the input
    that the user writes, so it stays CSV
  - The other files are not data tables and stay as they are: `NN-<stage>.summary.txt`
    (text for a person), `04-report.references.md`, and the PDF and the markdown of
    each paper; the HTTP cache is JSON already
- **Why**: a feature such as `performance` or `mechanism` is a long free text with
  commas and quotes, which a CSV shows as one hard-to-read line with doubled quotes; a
  score is a number, and JSON keeps it a number; every other output of the pipeline is
  JSON, so one format serves all (task 9)
- **Alternatives**: keep the CSV (the spreadsheet use is the reason of
  [`D31`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d31-json-for-the-intermediate-files)
  for the CSV report, which a person reads better as JSON); write both (two formats
  for the same data); also convert `00-paper_list.csv` (it is a copy of what the user
  wrote, and `resolve_papers.py` reads a CSV)
- **Consequences**:
  - The report is not a table in a spreadsheet: the user sorts it with `jq` or
    `pandas.read_json()`
  - A run dir made before this decision holds `04-report.report.csv`: run
    `report_papers.py --action report` to write the JSON file, and delete the CSV by
    hand, since the pipeline never deletes a file
  - Amends
    [`D31`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d31-json-for-the-intermediate-files)
- **Status**: accepted

### `D35` (The Report Has One Row per Paper or Unresolved Line, with Its Issue)

- **Decision**: `04-report.report.json` has one row for each paper and one row for
  each issue that no paper owns, e.g., an unresolved line, under the key `rows`; the
  file `04-report.issues.json` is removed (task 10)
  - A row has the keys `line_nos`, `status`, `reference`, the nine values of
    [`D34`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d34-the-report-is-json)
    (`paper_quality` to `citations_score`), `base_name`, and `issue`
  - `status` is `ok` for a paper with a features file, a rating file, and no issue;
    any other row is `failed`. A failed row keeps the values if the paper has the
    features file and the rating file, and the values are `null` if not
  - `issue` has the keys `stage`, `status`, `reason`, and `candidates`, and is `null`
    if no stage reported the row. A row that is `failed` with `issue: null` and no
    value means that an earlier stage did not run for the paper
  - `reference` is the reference of the paper, or, for a line that gave no paper, the
    text of the line; `base_name` is `null` for such a line
  - The rows of the papers come first, in the order of
    [`R7`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r7-reverse-chronological-order),
    then the rows of the other issues, by line number and stage
  - A paper that the resolver merged from many input lines is one row, and
    `line_nos` lists all the lines. So the rows cover every line of the input list
  - A paper with many issues has the issue of the earliest stage in its row, and a
    warning in the log; the files `NN-<stage>.issues.json` keep all of them
  - The files `NN-<stage>.issues.json` stay: they are how the stages talk to `report`
    ([`D26`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d26-the-resolver-is-a-separate-executable-that-reads-and-writes-csv))
  - Only `04-report.report.json` has `schema_version` 3: the other files stay at 2, so
    no features file needs a new LLM extraction
  - The summary of the stage `report` counts rows, not papers, and its issues file is
    `04-report.report.json`
- **Why**: to learn what happened to a line of the input list, the user read two
  files and joined them by hand; the counts did not match either (20 papers and 12
  issues, since 2 issues are lines that are no paper); one file with one row per input
  line answers "did this line work, and if not, why"
- **Alternatives**: objects `rating` and `features` inside a row (the flat keys keep
  the `jq` and `pandas.read_json()` use of
  [`D34`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d34-the-report-is-json));
  one row for each input line even when two lines are the same paper (the same data
  twice, and a sum over the rows counts a paper twice); a `status` column in the old
  report while `issues.json` stays (two files to join); a new `schema_version` for
  every file (it forces the user to extract again with the LLM)
- **Consequences**:
  - Supersedes
    [`D19`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d19-reportjson-has-a-row-only-for-a-paper-with-a-features-file),
    and amends
    [`D31`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d31-json-for-the-intermediate-files)
    and
    [`D34`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d34-the-report-is-json)
  - `epq_report.write_all_issues()` and `epq_paths.get_all_issues_path()` are removed,
    and `build_report_rows()` takes the issues
  - A run dir made before this decision holds `04-report.issues.json` and a report
    with the key `papers`: run `report_papers.py --action report` to write the new
    report, and delete the old file by hand, since the pipeline never deletes a file
- **Status**: accepted

## Code Organization

### `D21` (The Loop of Each Stage Catches Errors Per Paper)

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

### `D22` (Flat Modules with the `epq_` Prefix)

- **Decision**: put the script and the `epq_*.py` modules flat in
  [`research/Estimate_academic_paper_quality/`](/research/Estimate_academic_paper_quality/),
  tests in `test/`
- **Why**: seven small modules do not need a dir each; the prefix follows the
  `noesis_*.py` modules of [`research/Noesis/common/`](/research/Noesis/common/)
- **Alternatives**: one dir per component, as in
  [`research/Noesis/`](/research/Noesis/)
- **Status**: proposed

### `D23` (New API Clients Instead of the Private Ones of the Download Tool)

- **Decision**:
  [`epq_bib_sources.py`](/research/Estimate_academic_paper_quality/epq_bib_sources.py)
  has its own clients for arXiv, Crossref, Semantic Scholar, and Unpaywall
- **Why**: **Fact**: the clients of the tool are private and return only year,
  authors, and title; the resolver needs the DOI, the record type, the family names,
  the open-access links, and the citations, plus a disk cache
- **Alternatives**: make the clients of the tool public and extend them (a bigger
  change to a tool with other users)
- **Status**: proposed

### `D27` (The Pipeline Is Split Into One Executable Per Stage)

- **Decision**: replace the script `estimate_paper_quality.py` and the module
  `epq_pipeline.py` with one executable per stage, linked only by files on disk
  - `resolve_papers.py`: the input CSV to the papers file (a CSV when this decision
    was taken, JSON by
    [`D31`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d31-json-for-the-intermediate-files))
  - `download_papers.py`: the PDF and the markdown of each paper
  - `extract_features.py`: the features file of each paper, with one LLM call
  - `report_papers.py`: the actions `rate` and `report`, selected with
    `hselect_action`
  - `estimate_paper_quality.sh`: runs the four executables in order (a Python script
    since [`D32`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d32-the-pipeline-script-is-python-and-runs-the-stages-as-subprocesses))
  - Move the rating out of the extraction: `extract_features.py` writes the
    affiliations and the page count in the features file, and the action `rate` reads
    them and writes `papers/<BASE>.rating.json`
- **Why**: the rule `## Split a Pipeline Into Executables` in
  [`.claude/skills/architecture.rules.md`](/.claude/skills/architecture.rules.md)
  asks for one executable per stage when the stages differ
  - Dependencies: `download` needs `pymupdf`, Docker, and Chrome, `extract` needs the
    LLM client, `rate` and `report` need only HTTP and YAML
  - Runtimes: `download` and `extract` take minutes to hours, `rate` and `report` take
    seconds
  - Schedules: a paper is downloaded and extracted once, the citation counts grow, so
    `rate` runs again every month, and `report` runs whenever the user wants
  - The rating is pure code and costs no LLM call, but it was inside `extract`: a new
    tier list or new bands needed `--force_extract`, which pays the LLM again
  - `rate` and `report` are one executable because both are cheap and share their
    dependencies; the files between the two actions let `rate` run alone
- **Alternatives**: keep one script with the actions `download`, `extract`, `report` (it
  mixes stages with different dependencies and schedules); five executables, with
  `rate_papers.py` apart from `report_papers.py` (one more entry point for a stage of
  seconds with the dependencies of `report`); a Python script that calls the stages
  in one process (the `Bad` example of the rule, since the stages would share the
  process; a driver that runs them as subprocesses is [`D32`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d32-the-pipeline-script-is-python-and-runs-the-stages-as-subprocesses))
- **Consequences**:
  - `epq_pipeline.py` and `PipelineConfig` are gone: each script holds the loop of its
    stage and the checks at the boundary, and the catch of
    [`D21`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d21-the-loop-of-each-stage-catches-errors-per-paper)
    stays in each loop
  - `epq_paths.py` names the shared files and `epq_issues.py` builds the issue of a
    failed paper: both are leaf modules
  - The schema of the JSON files is version 2: the features file has `affiliations`
    and `n_pages` and has no `quality`; an old file fails to read with a message that
    names the stage to run again
  - New issue stage `rate`, with the statuses `rating_failed` and `rating_missing`, in
    `04-rate.issues.json` (`state/issues.rate.json` when this decision was taken, see
    [`D29`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d29-numbered-stage-files-in-the-output-dir))
  - The citation lookup moves from `extract` to `rate`: `--force_extract` does not
    refresh the citations anymore, `--refresh_citations` does, and `extract_features.py`
    needs no `BibSources`
  - `report.json` has a row only for a paper with a features file and a rating file
    ([`D19`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d19-reportjson-has-a-row-only-for-a-paper-with-a-features-file))
  - Every script has `--dry_run`; a dry run writes no file, so the next stage has no
    input and the driver cannot forward the flag (see [`D32`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d32-the-pipeline-script-is-python-and-runs-the-stages-as-subprocesses))
  - A test of the import graph fails on a script that imports another script
  - Extends
    [`D26`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d26-the-resolver-is-a-separate-executable-that-reads-and-writes-csv),
    which separated the resolver only
- **Status**: accepted

### `D28` (A Module With One Caller Lives in That Script)

- **Decision**: inline the three modules that only one script imports, and move their
  tests into the tests of that script
  - `epq_resolver.py` into `resolve_papers.py`
  - `epq_extractor.py` into `extract_features.py`, and `epq_extractor.prompt.md`
    becomes `extract_features.prompt.md`
  - `epq_rater.py` into `report_papers.py`
  - Keep as modules the code that two or more files import: `epq_types`,
    `epq_bib_sources`, `epq_fetcher`, `epq_report`, `epq_issues`, and `epq_paths`
  - The functions become private (`_` prefix), since only their script calls them
- **Why**: after
  [`D27`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d27-the-pipeline-is-split-into-one-executable-per-stage)
  each of the three modules had exactly one caller
  - A module with one caller shares nothing: it adds a file, an import alias, a
    layer in the import graph, and a doc, and a reader must open two files to follow
    one stage
  - The modules were inside the layer `Business logic` only to be below the
    scripts; the import graph keeps the other layers
- **Alternatives**: keep the modules, so that the pure logic (the rater has no I/O)
  is apart from the loop of the stage; this keeps the files above for no sharing. If a
  second file needs the code of one of the three, split that code out again
- **Consequences**:
  - `resolve_papers.py` has about 1,000 lines, `report_papers.py` about 800, and
    `extract_features.py` about 640
  - `rate_paper()` is `_compute_rating()`, since `report_papers.py` already has
    `_rate_paper()`, which writes the rating file of one paper
  - `extract_features()` is `_extract_features()`, and `LlmFunc` is `_LlmFunc`
  - `resolve_papers.py` imports `epq_fetcher`, and the test of the import graph
    allows it
  - `_resolve_lines()` has no caller in the script, since `_resolve()` loops on
    `_resolve_line()` with the catch of
    [`D21`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d21-the-loop-of-each-stage-catches-errors-per-paper):
    only the tests call it
  - The components `resolver`, `extractor`, and `rater` of the design docs are
    sections of their script, not files
  - Extends
    [`D22`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d22-flat-modules-with-the-epq_-prefix) and
    [`D27`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d27-the-pipeline-is-split-into-one-executable-per-stage)
- **Status**: accepted

### `D32` (The Pipeline Script Is Python and Runs the Stages as Subprocesses)

- **Decision**: replace `estimate_paper_quality.sh` with `estimate_paper_quality.py`, a
  Python driver that runs the four executables with `hsystem.system()`
  - One action per executable, named after it: `resolve_papers`, `download_papers`,
    `extract_features`, and `report_papers`, selected with `hselect_action` and run
    in this order
  - The flags are the long flags of the executables: `--input`, `--output_dir`,
    `--model`, `--force_extract`, `--refresh_citations`, `--dry_run`, and `-v`
    replace the `-i`, `-o`, `-m`, `-f`, and `-r` of the shell script
  - `--input` goes only to `resolve_papers.py`, `--output_dir` and `-v` go to every
    executable, and `--model`, `--force_extract`, and `--refresh_citations` go to
    the one that takes them
  - `--dry_run` logs the command of each executable and runs none of them
  - The script imports only `epq_paths`, and the test of the import graph allows
    only that
- **Why**: rule `ARCH3.5` in
  [`.claude/skills/architecture.rules.md`](/.claude/skills/architecture.rules.md)
  asks for a Python driver when users run only some of the steps or when the driver
  forwards `-v` or `--dry_run`
  - The shell script could not run a part of the pipeline, had no dry run, and had no
    test; the user ran an executable by hand to rerun one stage
  - Subprocesses keep the reason of
    [`D27`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d27-the-pipeline-is-split-into-one-executable-per-stage):
    each executable keeps its own `uv run` dependencies, runtime, and schedule, and
    the driver needs only `helpers`
- **Alternatives**: keep the shell script (no stage selection, no test); a Python
  script that imports the stages (the stages would share the process and the
  dependencies of `download`, `extract`, and `report` would meet in one environment);
  forward `--dry_run` to each executable, as `ARCH3.5` says (on a new output dir
  `resolve_papers.py --dry_run` writes no papers file, so `download_papers.py
  --dry_run` fails on a missing input and the dry run stops at the second stage)
- **Consequences**:
  - A call with the old flags `-o`, `-m`, `-f`, or `-r` fails with an error of
    `argparse`
  - A flag of a single executable, e.g., `--email` or `--min_delay_sec`, is not
    forwarded: the user calls that executable, then runs the others through the
    driver with `--skip_action`
  - `test_estimate_paper_quality.py` tests the command of each action and the dry run
  - Extends
    [`D27`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d27-the-pipeline-is-split-into-one-executable-per-stage)
- **Status**: accepted
