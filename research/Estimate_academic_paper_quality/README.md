# Estimate Academic Paper Quality

Planning docs for a pipeline that turns a list of academic papers into a CSV report
of candidate trading strategies.

- Input: a text file with one paper per line, as a title, URL, DOI, or local PDF
- Output: a reference list, paper markdown, and a CSV report of features per paper
- Users: quant researchers, who sort the CSV to pick the next papers to replicate
- Status: requirements and design only, no code exists yet
  - The planned CLI is `estimate_paper_quality.py`

## Structure of the Dir

| Directory | Description                                             |
| :-------- | :------------------------------------------------------ |
| `docs/`   | Requirements, design, and decision docs of the pipeline |

## Description of Files

| File                          | Description                                                                     | Cluster      |
| :---------------------------- | :------------------------------------------------------------------------------ | :----------- |
| `docs/decisions.md`           | Log of each design decision, with its reason, alternatives, and status          | Decisions    |
| `docs/design.architecture.md` | Overall design: components, import graph, data model, and requirements map      | Design       |
| `docs/design.bib_sources.md`  | Clients of the lookup APIs (Semantic Scholar, Crossref, arXiv, Unpaywall)       | Design       |
| `docs/design.extractor.md`    | One LLM call per paper that returns features, facts, affiliations, and quotes   | Design       |
| `docs/design.fetcher.md`      | Download and convert each paper with the download tool, skipping finished papers | Design       |
| `docs/design.pipeline.md`     | CLI, the four stages, the per-paper error loop, and the skip of finished work   | Design       |
| `docs/design.rater.md`        | Paper quality rating from page length, affiliation, and citation count          | Design       |
| `docs/design.report.md`       | Writers and readers of the output files: references, CSVs, and features files   | Design       |
| `docs/design.resolver.md`     | Line parsing, paper matching, dedup, reference building, and sorting            | Design       |
| `docs/prd.md`                 | Product requirements: inputs, features, outputs, non-goals, and success metrics | Requirements |
| `idea.md`                     | Original idea: the pipeline steps and the criteria to score each paper          | Idea         |
| `paper_list.md`               | Input list of papers, one per line, with authors in parentheses                 | Input        |

## Description of Architecture

- No code exists yet: the planned design is in `docs/design.architecture.md`
- The planned code is flat in this dir, with the `epq_` prefix (`D-22` in
  `docs/decisions.md`):
  - Script: `estimate_paper_quality.py` parses the CLI and calls `epq_pipeline.py`
  - Business logic: `epq_resolver.py`, `epq_extractor.py`, `epq_rater.py`
  - Infrastructure: `epq_bib_sources.py`, `epq_fetcher.py`, `epq_report.py`
  - Shared types: `epq_types.py`, with no project imports
- Imports point down the layers only, checked by a test of the import graph
