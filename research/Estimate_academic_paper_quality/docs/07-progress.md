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
- Pipeline and CLI (`PR_P1`, `PR_P2`, `PR_P3`): see
  [`02.1-design.pipeline.md`](/research/Estimate_academic_paper_quality/docs/02.1-design.pipeline.md)
- Not built: `PR_Q2` (the tier list needs a person), `PR_V1` and `PR_V2` (they
  need the labeled sets), and the unscheduled PRs

## Findings

1. The real input does not have one paper per line: **Fact**:
   [`paper_list.md`](/research/Estimate_academic_paper_quality/paper_list.md) wraps
   each bullet at 85 chars, so 37 physical lines are 22 entries. The parser joins
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
   `helpers.hllm` is imported inside `make_llm_func()` since `openai` is missing

7. The `download` stage works with the real download tool, with Docker running: for
   the arXiv id `1706.03762` it saved the PDF (2.2 MB) and the markdown (41 KB) as
   `2017.Vaswani_et_al.Attention_Is_All_You_Need` in about 6 s, and a rerun skipped
   the paper in 1 s
   ([`R8`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r8-pdf-and-markdown-files),
   [`R9`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r9-file-base-name),
   [`R11`](/research/Estimate_academic_paper_quality/docs/01-prd.md#r11-rerun-skips-converted-papers))

## Next

1. Fill
   [`affiliation_tiers.yaml`](/research/Estimate_academic_paper_quality/affiliation_tiers.yaml)
   by hand from the QS site (`PR_Q2`): until then each affiliation scores by its
   kind, and a university scores 3
2. Run the full pipeline with `SEMANTIC_SCHOLAR_API_KEY`, `--email`, Docker, Chrome,
   and an LLM key: the `download` stage on an SSRN paper and the `extract` stage
   have not run on live services
3. Label an input list and 20 papers (`PR_V1`, `PR_V2`), then measure the three
   success metrics and pick the default LLM
4. Split the work in PRs and merge, see
   [`04-plan.md`](/research/Estimate_academic_paper_quality/docs/04-plan.md)

## History

| Date       | Done                                                                                                                       |
| :--------- | :------------------------------------------------------------------------------------------------------------------------- |
| 2026-11-01 | Wrote `04-plan.md` (4 versions, 24 scheduled PRs) and this progress log                                                    |
| 2026-11-02 | Wrote the code of `v0.1` to `v0.3` and 156 tests in one pass; decisions `D24` and `D25`; ran `resolve` on the live APIs  |

## Open Work

- None: no issue or PR is open. The code is not committed
