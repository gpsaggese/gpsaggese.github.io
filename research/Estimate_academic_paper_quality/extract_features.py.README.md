# `extract_features.py`

- Extracts the features of each downloaded paper with one LLM call per paper
- Input: the papers file that
  `research/Estimate_academic_paper_quality/resolve_papers.py` writes, by default
  `01-resolve.paper_list.json` in `--output_dir`, and the PDF and markdown files that
  `research/Estimate_academic_paper_quality/download_papers.py` writes
- Output: `papers/<BASE>.features.json` for each paper, with the features, the facts,
  the affiliations, the page count, and the cost of the call, and
  `03-extract.issues.json` with the papers that failed, in `--output_dir`
- Run it after `research/Estimate_academic_paper_quality/download_papers.py` and
  before `research/Estimate_academic_paper_quality/report_papers.py`. A rerun does
  not pay again for the papers that have a features file

## Examples

- Extract the features of the downloaded papers:

  ```bash
  > research/Estimate_academic_paper_quality/extract_features.py \
      --output_dir ~/papers/run_2026_10
  ```

- List the papers to extract, with no LLM call and no output file:

  ```bash
  > research/Estimate_academic_paper_quality/extract_features.py \
      --output_dir ~/papers/run_2026_10 \
      --dry_run
  ```

- Extract again the features of all the papers with another LLM:
  ```bash
  > research/Estimate_academic_paper_quality/extract_features.py \
      --output_dir ~/papers/run_2026_10 \
      --force_extract \
      --model <model>
  ```

## Configuration & Inputs

### Command-line Arguments

| Argument                                | Type | Default                 | Description                                                                               |
| :-------------------------------------- | :--- | :---------------------- | :---------------------------------------------------------------------------------------- |
| `-i`, `--input`                         | str  | `""`                    | Papers file of the stage resolve: empty is `01-resolve.paper_list.json` in `--output_dir` |
| `--output_dir`                          | str  | required                | Dir of all the outputs and of the state of the run                                        |
| `--model`                               | str  | `hllmcli.DEFAULT_MODEL` | LLM to call with the `llm` library, e.g., `gpt-4o-mini`                                   |
| `--force_extract`, `--no_force_extract` | flag | `--no_force_extract`    | Extract again the papers that have a features file                                        |
| `--max_chars`                           | int  | `300000`                | Cap on the characters of markdown sent to the LLM                                         |
| `--dry_run`                             | flag | false                   | Show the papers to extract, with no LLM call and no output file                           |
| `-v`                                    | str  | `INFO`                  | Logging level: `TRACE`, `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`                   |

### Environment Variables

- `OPENROUTER_KEY`: key of the default model, which needs the `llm-openrouter`
  plugin (the `uv run` header installs it)
- The key of the provider of another `--model`, as the `llm` library reads it, e.g.,
  `OPENAI_API_KEY` for `gpt-4o-mini`

### Other Requirements

- The script needs the files that
  `research/Estimate_academic_paper_quality/download_papers.py` writes. Without the
  PDF of a paper, the quotes of that paper stay unverified and its page count is
  unknown
- The first line of the file selects its dependencies with `uv run`

## Output & Side Effects

### Files Created

- The layout of the whole output dir is in the section "Output Dir" of
  [`research/Estimate_academic_paper_quality/README.md`](/research/Estimate_academic_paper_quality/README.md#output-dir)
- In `<output_dir>/papers/`: `<BASE>.features.json`, with a `schema_version`, the five
  features (`performance`, `recency`, `asset_class`, `theoretical_domain`, `mechanism`),
  the facts, the affiliations, `n_pages`, and the provenance (model, prompt hash,
  characters sent, cost in USD)
- `<output_dir>/03-extract.issues.json`: the papers that failed, read by
  `research/Estimate_academic_paper_quality/report_papers.py`
- `<output_dir>/03-extract.summary.txt`: the summary of the stage, with the same text
  as the end of the log (the counts of items, the time of the stage, the output files,
  and the path of the
  issues file). A rerun overwrites it. See the section "Summary of a Stage" in
  [`research/Estimate_academic_paper_quality/docs/02.1-design.pipeline.md`](/research/Estimate_academic_paper_quality/docs/02.1-design.pipeline.md)
- The script does not rate the papers: the rating is the stage `rate` of
  `research/Estimate_academic_paper_quality/report_papers.py`, so that a new tier
  list needs no new LLM call
- With `--dry_run` the script writes nothing

### Exit Codes

- `0`: Success. Papers that fail do not change it: they are issues for
  `research/Estimate_academic_paper_quality/report_papers.py`
- `1`: An assertion or an exception stopped the run, e.g., the papers file is not an
  existing file, or `--output_dir` has no `papers/` dir
- `2`: A missing required argument or an invalid flag

### Logging

- Logs go to the console, at the level of `-v`, and to a file
  `research/Estimate_academic_paper_quality/extract_features.py.log`, which each run
  overwrites
- The run logs a progress bar, a warning for each failed paper, the cost of each paper,
  and the summary of the stage, which has the total cost of the run

## Software Architecture

- The design of the extractor is in
  [`research/Estimate_academic_paper_quality/docs/02.5-design.extractor.md`](/research/Estimate_academic_paper_quality/docs/02.5-design.extractor.md).
  The extractor code is in this file, with no module of its own (decision `D28`)

### Data Flow

1. **Input**: `_main()` checks the inputs, builds an `_ExtractConfig`, and reads the
   `Paper` objects of the papers file with `epqrepor.read_papers()`. The papers file
   is `--input`, or `epqpaths.get_papers_path()` if `--input` is empty
2. **Read**: for each paper with a markdown file and no features file, the script
   reads the markdown, the text of each PDF page, and the page count
3. **Extract**: `_extract_features()` makes one LLM call and checks each
   quote against the text of the PDF
4. **Output**: `epqrepor.write_features_file()` writes the features file of the
   paper, and `epqrepor.write_issues()` writes `03-extract.issues.json` at the end of
   the run

### Key Functions

- `_parse() -> argparse.ArgumentParser`
  - Defines the flags, with the helpers for the boolean flag and `-v`
- `_main(parser) -> None`
  - Checks the inputs, builds the LLM function, and calls `_extract()`
- `_extract(papers, config, *, llm_func) -> List[Issue]`
  - Loops on the papers with a catch per paper, and writes the issues of the stage to
    the path of `epqpaths.get_issues_path(output_dir, "extract")`
- `_extract_paper(paper, config, *, llm_func) -> float`
  - Extracts one paper, writes its features file, and returns the cost of the call
- `_extract_features(md_text, page_texts, *, llm_func, max_chars, model) -> ExtractionResult`
  - Drops the references, cuts a long paper, makes the LLM call with the prompt
    `research/Estimate_academic_paper_quality/extract_features.prompt.md`, and checks
    each quote with `_locate_quote()`
- `_make_llm_func(model, *, force_refresh) -> _LlmFunc`
  - Builds the LLM function: `hllmcli.apply_llm()` with the model and the `library`
    backend bound
- `_get_system_prompt() -> str`
  - Reads the rules of `extract_features.prompt.md` and adds the JSON schema of
    `_LlmExtraction`
- `_parse_llm_answer(answer) -> _LlmExtraction`
  - Parses the JSON object of the answer, even if it has a code block around it, and
    validates it

### Design Patterns

- **Files between stages**: the script reads the papers file and the PDF and markdown
  files, and writes only files in `--output_dir`. It imports no other executable
  (decision `D27`)
- **Catch per item**: a failed paper becomes an issue and the run goes on (decision
  `D21`)
- **Injected dependencies**: `llm_func` is an argument of `_extract()`, so the tests use a
  fake LLM
- **The features file is the cache**: a paper with a features file is not extracted
  again, unless `--force_extract` is set
