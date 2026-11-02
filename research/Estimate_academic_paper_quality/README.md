# Estimate Academic Paper Quality

Code and docs of a pipeline that turns a list of academic papers into a CSV report
of candidate trading strategies
- Input: a text file with one paper per line, as a title, URL, DOI, or local PDF
- Output: a reference list, paper markdown, and a CSV report of features per paper
- Users: quant researchers, who sort the CSV to pick the next papers to replicate
- Status: the code of the four stages and its tests are written, and the hand-filled
  tier list and the labeled sets are missing, see `docs/07-progress.md`
  - The CLI is `estimate_paper_quality.py`

## Structure of the Dir

| Directory | Description                                                      |
| :-------- | :--------------------------------------------------------------- |
| `docs/`   | Requirements, design, decision, plan, and progress docs          |
| `test/`   | Unit tests of the modules, and the helpers shared by the tests   |

## Description of Files

| File                              | Description                                                                      | Cluster      |
| :-------------------------------- | :------------------------------------------------------------------------------- | :----------- |
| `docs/01-prd.md`                  | Product requirements: inputs, features, outputs, non-goals, and success metrics  | Requirements |
| `docs/02-design.architecture.md`  | Overall design: components, import graph, data model, and requirements map       | Design       |
| `docs/02.1-design.pipeline.md`    | CLI, the four stages, the per-paper error loop, and the skip of finished work    | Design       |
| `docs/02.2-design.resolver.md`    | Line parsing, paper matching, dedup, reference building, and sorting             | Design       |
| `docs/02.3-design.bib_sources.md` | Clients of the lookup APIs (Semantic Scholar, Crossref, arXiv, Unpaywall)        | Design       |
| `docs/02.4-design.fetcher.md`     | Download and convert each paper with the download tool, skipping finished papers | Design       |
| `docs/02.5-design.extractor.md`   | One LLM call per paper that returns features, facts, affiliations, and quotes    | Design       |
| `docs/02.6-design.rater.md`       | Paper quality rating from page length, affiliation, and citation count           | Design       |
| `docs/02.7-design.report.md`      | Writers and readers of the output files: references, CSVs, and features files    | Design       |
| `docs/03-decisions.md`            | Log of each design decision, with its reason, alternatives, and status           | Decisions    |
| `docs/04-plan.md`                 | Versions and PRs to build, in order, with the open questions of each component   | Plans        |
| `docs/07-progress.md`             | What is done and learned: milestone status, findings, next tasks, and history    | Plans        |
| `idea.md`                         | Original idea: the pipeline steps and the criteria to score each paper           | Idea         |
| `paper_list.md`                   | Input list of papers, one per line, with authors in parentheses                  | Input        |

## Description of Architecture

- The design is in `docs/02-design.architecture.md`
- The code is flat in this dir, with the `epq_` prefix (`D22` in
  `docs/03-decisions.md`), and the tests are in `test/`:
  - Script: `estimate_paper_quality.py` parses the CLI and calls `epq_pipeline.py`,
    see `estimate_paper_quality.py.README.md`
  - Business logic: `epq_resolver.py`, `epq_extractor.py`, `epq_rater.py`
  - Infrastructure: `epq_bib_sources.py`, `epq_fetcher.py`, `epq_report.py`
  - Shared types: `epq_types.py`, with no project imports
- Imports point down the layers only, checked by a test of the import graph
- Other files: `epq_extractor.prompt.md` is the system prompt of the LLM, and
  `affiliation_tiers.yaml` is the dated tier list of the affiliation signal