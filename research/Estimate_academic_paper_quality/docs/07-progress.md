# Estimate Paper Quality Progress

## Status

| Milestone                              | Target date | Status                                      | PR   |
| :------------------------------------- | :---------- | :------------------------------------------ | :--- |
| `v0.1`: Normalized reference list      | TBD         | Code and tests written, not merged          | None |
| `v0.2`: Papers on disk                 | TBD         | Code and tests written, not merged          | None |
| `v0.3`: Report of candidate strategies | TBD         | Code written, not run on a live LLM         | None |
| `v0.4`: Measured pipeline              | TBD         | Not started: it needs the labeled sets      | None |

## Built

- Download tool: `format_base_filename()` in `download_utils.py`, called by
  `download_academic_paper_to_md.py` (`PR_H1`), see
  [`02-design.architecture.md`](/research/Estimate_academic_paper_quality/docs/02-design.architecture.md)
- Shared types (`PR_T1`) and the import graph test
- Resolver (`PR_R1`, `PR_R2`, `PR_R3`, `PR_R4`): see
  [`02.2-design.resolver.md`](/research/Estimate_academic_paper_quality/docs/02.2-design.resolver.md)
- Bib sources (`PR_B1`, `PR_B2`, `PR_B3`, `PR_B4`): see
  [`02.3-design.bib_sources.md`](/research/Estimate_academic_paper_quality/docs/02.3-design.bib_sources.md)
- Fetcher (`PR_F1`, `PR_F2`, `PR_F3`): see
  [`02.4-design.fetcher.md`](/research/Estimate_academic_paper_quality/docs/02.4-design.fetcher.md)
- Extractor (`PR_E1`, `PR_E2`): see
  [`02.5-design.extractor.md`](/research/Estimate_academic_paper_quality/docs/02.5-design.extractor.md)
- Rater (`PR_Q1`): see
  [`02.6-design.rater.md`](/research/Estimate_academic_paper_quality/docs/02.6-design.rater.md)
- Report (`PR_O1`, `PR_O2`): see
  [`02.7-design.report.md`](/research/Estimate_academic_paper_quality/docs/02.7-design.report.md)
- Pipeline and CLI (`PR_P1`, `PR_P2`, `PR_P3`, `PR_P5`, `PR_P6`): see
  [`02.1-design.pipeline.md`](/research/Estimate_academic_paper_quality/docs/02.1-design.pipeline.md)
- Not built: `PR_Q2` (the tier list needs a person), `PR_V1` and `PR_V2` (they
  need the labeled sets), and the unscheduled PRs
- Resolver as a separate executable (task 1 of `todos.md`, decision `D26`):
  `resolve_papers.py` reads an input CSV and writes a papers CSV and an issues CSV;
  the pipeline script of that time (`estimate_paper_quality.py`, replaced since by
  the entry below) lost the stage `resolve` and the flag `--email`, and
  read the papers CSV with `--input`. 168 tests pass, and a live run of the
  resolver on 3 rows worked with Semantic Scholar blocked (`429`)
- One executable per stage (`PR_P5`, decision `D27`): `download_papers.py`,
  `extract_features.py`, and `report_papers.py` (actions `rate` and `report`) replace
  `estimate_paper_quality.py` and `epq_pipeline.py`, and
  `estimate_paper_quality.sh` runs the four executables in order (replaced by the
  entry below). The rating moved
  out of the extraction into `papers/<BASE>.rating.json`, so a new tier list needs no
  new LLM call. 196 tests pass, each script runs `--help` through its `uv run`
  header, and the shell script ran on stubs of the four scripts
- Modules with one caller inlined (`PR_P6`, decision `D28`): `epq_resolver.py`,
  `epq_extractor.py`, and `epq_rater.py` are now code of `resolve_papers.py`,
  `extract_features.py`, and `report_papers.py`, and their tests are in the tests
  of the scripts. The prompt file is `extract_features.prompt.md`. 196 tests pass
- Python driver instead of the shell script (`PR_P5`, decision `D32`):
  `estimate_paper_quality.py` runs the four executables as subprocesses with
  `hsystem`, with one action per executable (`hselect_action`), so a user can run or
  skip one of them. `--dry_run` logs the command of each action and runs none, since
  a dry run of a stage needs the files of the stage before. 236 tests pass. A dry run
  on `paper_list.short.csv` printed the four commands, and a real run of
  `download_papers` on an empty output dir failed on the missing papers file and
  stopped with exit code 1. No live run of the full chain yet

## Findings

1. The real input does not have one paper per line: **Fact**:
   the former `paper_list.md` (now
   [`paper_list.csv`](/research/Estimate_academic_paper_quality/paper_list.csv))
   wrapped each bullet at 85 chars, so 37 physical lines are 22 entries. The parser joins
   the lines of a bullet
   ([`D25`](/research/Estimate_academic_paper_quality/docs/03-decisions.md#d25-the-parser-joins-the-lines-of-a-wrapped-bullet))
2. Semantic Scholar without a key answered `429` to the first search of every live
   run on 2026-11-02, also on a single `curl`. The pipeline now skips an API that
   failed after its retries, and uses the other sources
   - First run (S2 blocked, Crossref only, title in the query): 18 of 22 lines
     resolved in 68 s
   - After the fixes of findings 3 and 4: 20 of 22 lines resolved in 59 s
   - The 2 lines still unresolved are not in the first 5 results of Crossref:
     Semantic Scholar is needed for them
   - Data: the runs were made in a scratch dir, not kept
3. Crossref searches by title alone rank similar titles first: "Deep Learning in
   Asset Pricing" and "The 8-K Trading Gap" were not in the first 5 results. With the
   authors of the line in the query, both are found
4. Crossref gives some family names in capitals (`COHEN`, `BRADLEY`): they are
   changed to title case, so that the reference reads `Cohen et al.`
5. `hsystem.system(output_file=...)` keeps only stdout (`2>&1 >file`), and the error
   of the download tool is on stderr: the fetcher redirects both streams in the
   command. A test with a stub tool found this
6. The venv of the helpers has no `pymupdf`, `feedparser`, `bs4`, or `openai`: the
   script has a `uv run` header with its dependencies, like the download tool. The
   tests were run with the three first packages installed in a scratch dir, and
   `helpers.hllm` is imported inside `_make_llm_func()` of `extract_features.py` since
   `openai` is missing

7. The `download` stage works with the real download tool, with Docker running: for
   the arXiv id `1706.03762` it saved the PDF (2.2 MB) and the markdown (41 KB) as
   `2017.Vaswani_et_al.Attention_Is_All_You_Need` in about 6 s, and a rerun skipped
   the paper in 1 s
   ([`R8`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r8-pdf-and-markdown-files),
   [`R9`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r9-file-base-name),
   [`R11`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r11-rerun-skips-converted-papers))

8. Running `--help` of each executable through its `uv run` header found that the
   header of `extract_features.py` lacked `requests`, which `helpers.hllm_cost`
   imports; the tests did not show it, since they run in an environment that has all
   the packages. Each header lists only the packages of its own stage, so a stage does
   not install what another stage needs

## Next

1. Fill
   [`affiliation_tiers.yaml`](/research/Estimate_academic_paper_quality/affiliation_tiers.yaml)
   by hand from the QS site (`PR_Q2`): until then each affiliation scores by its
   kind, and a university scores 3
2. Run `resolve_papers.py --email` and then `estimate_paper_quality.py` with
   `SEMANTIC_SCHOLAR_API_KEY`, Docker, Chrome, and an LLM key: the `download` stage on an SSRN paper and the `extract` stage
   have not run on live services, and the action `rate` has not run on live Semantic Scholar
3. Label an input list and 20 papers (`PR_V1`, `PR_V2`), then measure the three
   success metrics and pick the default LLM
4. Split the work in PRs and merge, see
   [`04-plan.md`](/research/Estimate_academic_paper_quality/docs/04-plan.md)

## History

| Date       | Done                                                                                                                     |
| :--------- | :----------------------------------------------------------------------------------------------------------------------- |
| 2026-11-01 | Wrote `04-plan.md` (4 versions, 24 scheduled PRs) and this progress log                                                  |
| 2026-11-02 | Wrote the code of `v0.1` to `v0.3` and 156 tests in one pass; decisions `D24` and `D25`; ran `resolve` on the live APIs  |
| 2026-11-02 | Added a `Status` line to each PR entry of `04-plan.md`: 21 PRs written on branch `gp`, none merged into `main`           |
| 2026-11-02 | Split the resolver into `resolve_papers.py`, with CSV in and CSV out (`D26`); the pipeline reads the papers CSV          |
| 2026-11-02 | Split the pipeline into one executable per stage, with a rating file and a shell script (`D27`, `PR_P5`); 196 tests pass |
| 2026-11-02 | Inlined the resolver, extractor, and rater modules into the script that calls each (`D28`, `PR_P6`); 196 tests pass      |
| 2026-11-03 | Replaced the shell script with the Python driver `estimate_paper_quality.py` (`D32`, `PR_P5`); 236 tests pass            |
| 2026-11-04 | Wrote the report as JSON, `04-report.report.json`, so no output is a CSV (`D34`); 261 tests pass                         |
| 2026-11-05 | Merged `04-report.issues.json` into `04-report.report.json`: one row per paper or unresolved line (`D35`); 263 tests pass |

## Open Work

- None: no issue or PR is open. The code is committed on the branch `gp` and pushed, and
  it is not merged into `main`. `PR_H1` is in the helpers repo (commit `2f36c8ed`), and
  the `helpers_root` pointer of this repo is not moved yet
