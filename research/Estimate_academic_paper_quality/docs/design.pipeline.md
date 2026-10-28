# Design: Pipeline

## Goal

- Run the four stages of the pipeline on one input file and one output dir
  - Parse and check the CLI flags, select the stages, call the other components
  - Loop on the papers, and turn a failure of one paper into an `Issue`, so the run
    goes on (PRD `R5`, `R10`)
  - Skip the work that a previous run already did (`R11`, `R17`)
- Where it fits: user -> `pipeline` -> `resolver`, `fetcher`, `extractor`, `rater`,
  `report`
- Overview: `research/Estimate_academic_paper_quality/docs/design.architecture.md`
- Decisions: `research/Estimate_academic_paper_quality/docs/decisions.md`

## Diagram

```mermaid
C4Component
  title Pipeline: Components
  Container_Boundary(p, "estimate_paper_quality.py + epq_pipeline.py") {
    Component(cli, "_parse() / _main()", "argparse, hselect_action", "Checks flags, builds PipelineConfig, runs the selected stages")
    Component(res, "run_resolve()", "Python", "Lines to papers and issues")
    Component(dl, "run_download()", "Python", "PDF and markdown per paper")
    Component(ex, "run_extract()", "Python", "Features file per converted paper")
    Component(rep, "run_report()", "Python", "references.md, report.csv, issues.csv")
  }
  Component_Ext(resolver, "epq_resolver", "", "")
  Component_Ext(fetcher, "epq_fetcher", "", "")
  Component_Ext(extractor, "epq_extractor", "", "")
  Component_Ext(rater, "epq_rater", "", "")
  Component_Ext(report, "epq_report", "", "")
  Component_Ext(bib, "epq_bib_sources", "", "")
  Rel(cli, res, "config")
  Rel(cli, dl, "config, papers")
  Rel(cli, ex, "config, papers")
  Rel(cli, rep, "config, papers, issues")
  Rel(res, resolver, "resolve_lines()")
  Rel(dl, fetcher, "is_fetched(), fetch_paper()")
  Rel(ex, extractor, "extract_features()")
  Rel(ex, rater, "rate_paper()")
  Rel(ex, bib, "get_citation_count()")
  Rel(rep, report, "write_references(), write_report(), write_issues()")
```

## Files and Import Graph

- `research/Estimate_academic_paper_quality/estimate_paper_quality.py`: script layer,
  `_parse()` and `_main(parser)` per `.claude/skills/script.rules.md`
- `research/Estimate_academic_paper_quality/epq_pipeline.py`: orchestration layer,
  `PipelineConfig` and one `run_<stage>()` per stage
- `estimate_paper_quality.py` imports `epq_pipeline`
- `epq_pipeline` imports every other `epq_*` module, `helpers.hselect_action`,
  `helpers.hllm`, `helpers.hllm_cost`, `helpers.hcache_simple`
- **Fact**: `helpers_root/dev_scripts_helpers/download/download_academic_paper_to_md.py`
  uses the same `hselect_action` pattern (`_VALID_ACTIONS`, `_DEFAULT_ACTIONS`,
  `mark_action()`), so the new script follows it

## One Request End to End

- `_main()`:
  - Checks at the boundary that `--input` exists and that `--output_dir` can be
    created, then creates `papers/` and `state/` under it
  - Sets the cache dir of `hcache_simple` to `<output_dir>/state/cache/` and applies
    `--cache_mode`
  - Builds `PipelineConfig`, then runs the selected stages in this order:
    `resolve`, `download`, `extract`, `report`
  - Passes the output of a stage to the next one; when a stage is not selected, the
    next stage reads that output from `state/` (D-18)
- `run_resolve(config)`:
  - Calls `epq_resolver.resolve_lines(lines, sources)` and gets papers and issues
  - Writes `state/papers.json` and `state/issues.resolve.json` with `epq_report`
- `run_download(config, papers)`:
  - For each paper: skips it if `epq_fetcher.is_fetched()` is true (`R11`)
  - Else calls `epq_fetcher.fetch_paper()` and turns a failed `FetchResult` into an
    `Issue` with status `no_download_link`, `download_failed`, or
    `conversion_failed`
  - Writes `state/issues.download.json`
- `run_extract(config, papers)`, for each paper:
  - Skips it if it has no markdown (the download issue already says why)
  - Skips it if its features file exists and `--force_extract` is not set (`R17`)
  - Reads the markdown and the text of each PDF page with `epq_fetcher`
  - Calls `epq_extractor.extract_features()` with an injected LLM function
  - Calls `epq_bib_sources.get_citation_count()`, with `REFRESH_CACHE` when
    `--force_extract` is set, so the count is fresh
  - Calls `epq_rater.rate_paper()` with the page count, the affiliations, the
    citation count, the year, and `config.today`
  - Builds the Recency feature from the year of the record, with the name of the
    source as evidence
  - Writes `papers/<BASE>.features.json` with `epq_report`
  - Logs the LLM cost of the paper and, at the end, the total
- `run_report(config, papers, issues)`:
  - Reads the features files and the issues of every stage
  - Writes `references.md`, `report.csv`, and `issues.csv`
- Error handling (D-21):
  - The inner modules assert with `hdbg.dassert_*` and raise
  - The loop of each stage catches `Exception` per paper and records an `Issue`
    with the last line of the message as the reason
  - An error outside the loop (e.g., a bad flag, a full disk) stops the run

## Configuration, Interfaces, and Known Limits

- Configuration:
  ```python
  @dataclasses.dataclass(frozen=True)
  class PipelineConfig:
      input_file: str
      output_dir: str
      # Contact for Unpaywall, "" to skip Unpaywall.
      email: str
      # LLM of `extract`, "" for the default model of `helpers.hllm`.
      model: str
      force_extract: bool
      # Max chars of markdown sent to the LLM (D-13).
      max_chars: int
      # Date used to annualize the citations, injected for tests.
      today: datetime.date
  ```
- Interfaces: CLI flags

  | Flag                      | Default           | Meaning                                         |
  | :------------------------ | :---------------- | :---------------------------------------------- |
  | `--input`                 | required          | Text file, one paper per line (`R1`)            |
  | `--output_dir`            | required          | Dir of all outputs and state (D-10)             |
  | `--email`                 | `""`              | Contact email for Unpaywall                     |
  | `--model`                 | `""`              | LLM for `extract`                               |
  | `--force_extract`         | off               | Extract again papers with a features file       |
  | `--max_chars`             | `300000`          | Cap on the markdown sent to the LLM             |
  | `--cache_mode`            | `""`              | `REFRESH_CACHE` refetches metadata and citations |
  | `--action`, `--skip_action`, `--clear_actions`, `--all_actions` | all 4 stages | Stage selection |
  | `-v`                      | `INFO`            | Log level                                       |

- Known limits:
  - Sequential: one paper at a time
  - A rerun must use the same `--output_dir`
  - A paper removed from the input list keeps its files in `papers/`, but leaves
    the report

## Deliberately Not Built

- A config file: the flags are few
- Parallel stages, retries of a failed paper in the same run: a rerun retries it

## Open Questions

1. Should `run_extract()` stop the run when the total LLM cost passes a cap (e.g.,
   `--max_cost_usd`)?
