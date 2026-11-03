# `extract_features.py`

- Extracts the features of each downloaded paper with one LLM call per paper
- Input: the papers CSV that `resolve_papers.py` writes, and the PDF and markdown files
  that `download_papers.py` writes
- Output: `papers/<BASE>.features.json` for each paper, with the features, the facts,
  the affiliations, the page count, and the cost of the call
- Run it after `download_papers.py` and before `report_papers.py`. A rerun does not pay
  again for the papers that have a features file

## Examples

- Extract the features of the downloaded papers:
  ```bash
  > research/Estimate_academic_paper_quality/extract_features.py \
      --input ~/papers/run_2026_10/papers.csv \
      --output_dir ~/papers/run_2026_10
  ```

- List the papers to extract, with no LLM call and no output file:
  ```bash
  > research/Estimate_academic_paper_quality/extract_features.py \
      --input ~/papers/run_2026_10/papers.csv \
      --output_dir ~/papers/run_2026_10 \
      --dry_run
  ```

- Extract again the features of all the papers with another LLM:
  ```bash
  > research/Estimate_academic_paper_quality/extract_features.py \
      --input ~/papers/run_2026_10/papers.csv \
      --output_dir ~/papers/run_2026_10 \
      --force_extract \
      --model <model>
  ```

## Configuration & Inputs

### Command-line Arguments

| Argument                                | Type | Default              | Description                                                          |
| :-------------------------------------- | :--- | :------------------- | :------------------------------------------------------------------- |
| `-i`, `--input`                         | str  | required             | Papers CSV that `resolve_papers.py` wrote                            |
| `--output_dir`                          | str  | required             | Dir of all the outputs and of the state of the run                   |
| `--model`                               | str  | `""`                 | LLM to call: empty is the default model of `helpers.hllm`            |
| `--force_extract`, `--no_force_extract` | flag | `--no_force_extract` | Extract again the papers that have a features file                   |
| `--max_chars`                           | int  | `300000`             | Cap on the characters of markdown sent to the LLM                    |
| `--dry_run`                             | flag | false                | Show the papers to extract, with no LLM call and no output file      |
| `-v`                                    | str  | `INFO`               | Logging level: `TRACE`, `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |

### Environment Variables

- `OPENAI_API_KEY`: key of the default provider of `helpers.hllm`
- `OPENROUTER_API_KEY`: key for a model that `helpers.hllm` serves through OpenRouter

### Other Requirements

- The script needs the files that `download_papers.py` writes. Without the PDF of a
  paper, the quotes of that paper stay unverified and its page count is unknown
- The first line of the file selects its dependencies with `uv run`

## Output & Side Effects

### Files Created

- In `<output_dir>/papers/`: `<BASE>.features.json`, with a `schema_version`, the five
  features (`performance`, `recency`, `asset_class`, `theoretical_domain`, `mechanism`),
  the facts, the affiliations, `n_pages`, and the provenance (model, prompt hash,
  characters sent, cost in USD)
- In `<output_dir>/state/`: `issues.extract.json`, the papers that failed, read by
  `report_papers.py`
- The script does not rate the papers: the rating is the stage `rate` of
  `report_papers.py`, so that a new tier list needs no new LLM call
- With `--dry_run` the script writes nothing

### Exit Codes

- `0`: Success. Papers that fail do not change it: they are issues for `report_papers.py`
- `1`: An assertion or an exception stopped the run, e.g., `--input` is not an
  existing file, or `--output_dir` has no `papers/` dir
- `2`: A missing required argument or an invalid flag

### Logging

- Logs go to the console, at the level of `-v`, and to a file
  `research/Estimate_academic_paper_quality/extract_features.py.log`, which each run
  overwrites
- The run logs a progress bar, a warning for each failed paper, the cost of each paper,
  and the total cost of the run

## Software Architecture

- The design of the extractor is in
  [`02.5-design.extractor.md`](/research/Estimate_academic_paper_quality/docs/02.5-design.extractor.md).
  The extractor code is in this file, with no module of its own (decision `D28`)

### Data Flow

1. **Input**: `_main()` checks the inputs, builds an `_ExtractConfig`, and reads the
   `Paper` objects of the papers CSV
2. **Read**: for each paper with a markdown file and no features file, the script
   reads the markdown, the text of each PDF page, and the page count
3. **Extract**: `_extract_features()` makes one LLM call and checks each
   quote against the text of the PDF
4. **Output**: `epq_report.write_features_file()` writes the features file of the paper

### Key Functions

- `_parse() -> argparse.ArgumentParser`
  - Defines the flags, with the helpers for the boolean flag and `-v`
- `_main(parser) -> None`
  - Checks the inputs, builds the LLM function, and calls `_extract()`
- `_extract(papers, config, *, llm_func, cost_tracker) -> List[Issue]`
  - Loops on the papers with a catch per paper, and writes the issues of the stage
- `_extract_paper(paper, config, *, llm_func, cost_tracker) -> None`
  - Extracts one paper and writes its features file
- `_extract_features(md_text, page_texts, *, llm_func, max_chars, model) -> ExtractionResult`
  - Drops the references, cuts a long paper, makes the LLM call with the prompt
    `extract_features.prompt.md`, and checks each quote with `_locate_quote()`
- `_make_llm_func(model) -> Tuple[_LlmFunc, LLMCostTracker]`
  - Builds the LLM function. It imports `helpers.hllm` only here

### Design Patterns

- **Files between stages**: the script reads the PDF and markdown files and writes only
  files in `--output_dir`. It imports no other executable (decision `D27`)
- **Catch per item**: a failed paper becomes an issue and the run goes on (decision
  `D21`)
- **Injected dependencies**: `llm_func` is an argument of `_extract()`, so the tests use a
  fake LLM
- **The features file is the cache**: a paper with a features file is not extracted
  again, unless `--force_extract` is set
