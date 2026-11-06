# Docs

- All docs of
  [`research/Estimate_academic_paper_quality`](/research/Estimate_academic_paper_quality/),
  in reading order
- Start with
  [`research/Estimate_academic_paper_quality/README.md`](/research/Estimate_academic_paper_quality/README.md),
  then `01-prd.md`, then `02-design.architecture.md`
- The status of the work is in `07-progress.md`
- The README of each executable lives next to the code:
  - [`research/Estimate_academic_paper_quality/resolve_papers.py.README.md`](/research/Estimate_academic_paper_quality/resolve_papers.py.README.md)
  - [`research/Estimate_academic_paper_quality/download_papers.py.README.md`](/research/Estimate_academic_paper_quality/download_papers.py.README.md)
  - [`research/Estimate_academic_paper_quality/extract_features.py.README.md`](/research/Estimate_academic_paper_quality/extract_features.py.README.md)
  - [`research/Estimate_academic_paper_quality/report_papers.py.README.md`](/research/Estimate_academic_paper_quality/report_papers.py.README.md)

## Structure of the Dir

- All the docs sit at the top level of `docs/`: there is no subdir
- The number at the start of a name is the slot of the doc and gives the reading
  order, see `.claude/skills/project.rules.md` section "Name the Docs with a Slot
  Number"
  - `02.<N>` is the design doc of one component, in the order of the pipeline
  - The docs without a number come last in the table below
- The code that the design docs describe is flat in
  [`research/Estimate_academic_paper_quality/`](/research/Estimate_academic_paper_quality/),
  and the tests are in
  [`research/Estimate_academic_paper_quality/test/`](/research/Estimate_academic_paper_quality/test/)
- Four executables, one per stage, hold the code of the components `pipeline`,
  `resolver`, `extractor`, and `rater`: the docs `02.1`, `02.2`, `02.5`, and `02.6`
  describe components that live inside those scripts, not separate files, see
  [`D28`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d28-a-module-with-one-caller-lives-in-that-script)

## Description of Files

| File                         | Description                                                                                             | Cluster |
| :--------------------------- | :------------------------------------------------------------------------------------------------------ | :------ |
| `01-prd.md`                  | Product requirements: inputs, features, outputs, non-goals, and success metrics                         | Specs   |
| `02-design.architecture.md`  | Overall design: C4 diagrams, files, import graph, data model, and requirements map                      | Design  |
| `02.1-design.pipeline.md`    | The four executables, the files between them, the per-paper error loop, and the skip of finished work   | Design  |
| `02.2-design.resolver.md`    | Line parsing, paper matching, dedup, reference building, and sorting, in `resolve_papers.py`            | Design  |
| `02.3-design.bib_sources.md` | Clients of the lookup APIs (Semantic Scholar, Crossref, arXiv, Unpaywall), in `epq_bib_sources.py`      | Design  |
| `02.4-design.fetcher.md`     | Download and convert each paper with the download tool, in `epq_fetcher.py`                             | Design  |
| `02.5-design.extractor.md`   | One LLM call per paper that returns features, facts, affiliations, and quotes, in `extract_features.py` | Design  |
| `02.6-design.rater.md`       | Paper quality rating from page length, affiliation, and citation count, in `report_papers.py`           | Design  |
| `02.7-design.report.md`      | Writers and readers of the output files, in `epq_report.py`                                             | Design  |
| `03-decisions.md`            | Log of each design decision, with its reason, alternatives, and status                                  | Specs   |
| `04-plan.md`                 | Versions and PRs to build, in order, with the open questions of each component                          | Plans   |
| `07-progress.md`             | What is done and learned: milestone status, findings, next tasks, and history                           | Plans   |
