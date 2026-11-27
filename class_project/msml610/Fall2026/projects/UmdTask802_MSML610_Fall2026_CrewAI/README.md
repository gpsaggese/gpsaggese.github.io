# FRED Macro Briefing Crew

This project uses CrewAI to generate a short macroeconomic briefing from four Federal Reserve Economic Data (FRED) series. A three-agent workflow downloads and profiles the data, calculates the required statistics with deterministic Python tools, and writes the final briefing.

## Team

- [anoushkanand](https://github.com/anoushkanand)
- [upratham](https://github.com/upratham)
- [ROYSARANYA](https://github.com/ROYSARANYA)

## Data

The project uses dated CSV snapshots of these FRED series:

| Series | Description |
|---|---|
| `CPIAUCSL` | Consumer Price Index |
| `UNRATE` | Unemployment Rate |
| `FEDFUNDS` | Federal Funds Rate |
| `T10Y2Y` | 10-Year Minus 2-Year Treasury Spread |

The included snapshot was downloaded on November 4, 2026 and is stored under:

```text
data/fred/2026-11-04/
```

Saving the CSV files makes the calculations and evaluation reproducible.

## Workflow

The workflow uses three sequential CrewAI agents:

1. **FRED Data Researcher**
   - Downloads the four FRED series.
   - Reports the latest and prior-year values, observation dates, and missing-value counts.

2. **Macroeconomic Data Analyst**
   - Uses a custom Python tool to calculate:
     - Year-over-year CPI inflation
     - Change in unemployment
     - Change in the federal funds rate
     - Sign of the Treasury yield spread

3. **Macroeconomic Briefing Writer**
   - Writes a briefing using only the Researcher and Analyst outputs.
   - Includes observation dates and avoids unsupported numeric claims.

The deterministic calculations are implemented with pandas rather than relying on the language model for arithmetic.

## Project Files

| File | Purpose |
|---|---|
| `crewai_utils.py` | Reusable helpers for constructing agents, tasks, and sequential crews |
| `crewai_fred_tools.py` | CrewAI tools for downloading, profiling, and analyzing FRED data |
| `fred_tools.py` | Deterministic FRED download and calculation functions |
| `crewai.API.py` | Small two-agent CrewAI API example |
| `crewai.API.ipynb` | Tutorial notebook for the reusable CrewAI API |
| `crewai.example.py` | Three-agent FRED macro briefing application |
| `crewai.example.ipynb` | End-to-end FRED workflow notebook |
| `evaluation.py` | Single-LLM baseline and field-aware claim evaluator |
| `multi_snapshot_evaluation.py` | Cached multi-snapshot robustness experiment and aggregate reporting |
| `reporting.py` | Markdown summary table and FRED chart generation |
| `test/test_fred_pipeline.py` | Unit tests using synthetic FRED data |
| `test/test_docker_template.py` | Docker and notebook integration tests |

## Build the Docker Image

From this project directory:

```bash
export CSFY_DOCKER_ENGINE=docker
./docker_build.sh
```

## Configure Gemini

Set the API key in the current shell without saving it in the repository:

```bash
read -s "GEMINI_API_KEY?Paste Gemini API key: "
export GEMINI_API_KEY
echo
export CREWAI_MODEL="gemini/gemini-3.5-flash-lite"
```

The Docker command passes `GEMINI_API_KEY` and `CREWAI_MODEL` from the host environment into the container.

Model availability can change, so another Gemini model returned by the Gemini Models API may be used if necessary.

## Run the Crew

```bash
./docker_cmd.sh 'cd /git_root/class_project/msml610/Fall2026/projects/UmdTask802_MSML610_Fall2026_CrewAI && python crewai.example.py --data-dir data/fred --download-date 2026-11-04'
```

The generated briefing is saved to:

```text
outputs/fred_macro_briefing_2026-11-04.md
```

## Run the Baseline Evaluation

```bash
./docker_cmd.sh 'cd /git_root/class_project/msml610/Fall2026/projects/UmdTask802_MSML610_Fall2026_CrewAI && python evaluation.py --snapshot-dir data/fred/2026-11-04 --crew-brief outputs/fred_macro_briefing_2026-11-04.md'
```

This creates:

```text
outputs/single_llm_baseline.md
outputs/evaluation_report.json
```

## Evaluation Result

The original evaluator compared every extracted number against one combined list of valid values. This could incorrectly accept a number under the wrong economic context, and it did not penalize missing required claims.

The improved evaluator uses the following 12 labeled claims:

1. Latest CPI value
2. Prior-year CPI value
3. Year-over-year inflation
4. Latest unemployment rate
5. Prior-year unemployment rate
6. Change in unemployment
7. Latest federal funds rate
8. Prior-year federal funds rate
9. Change in the federal funds rate
10. Latest Treasury yield spread
11. Prior-year Treasury yield spread
12. Yield-curve sign

It reports accuracy among submitted claims, completeness of the required checklist, and the fraction of all required claims that are correct.

For the November 14, 2026 snapshot:

| Method | Correct | Submitted | Required | Accuracy | Completeness | Required-claim score |
|---|---:|---:|---:|---:|---:|---:|
| Three-agent CrewAI workflow | 12 | 12 | 12 | 1.00 | 1.00 | 1.00 |
| Single-LLM baseline | 12 | 12 | 12 | 1.00 | 1.00 | 1.00 |

The field-aware evaluation confirmed that both systems produced every required claim correctly in this run. The CrewAI workflow provides a deterministic and auditable calculation path, but it matched rather than outperformed the single-LLM baseline on this snapshot.

## Multi-Snapshot Evaluation

The robustness experiment evaluates four historical observation cutoffs representing different economic conditions:

| Cutoff | Economic condition |
|---|---|
| `2019-06-30` | Pre-pandemic conditions |
| `2020-05-31` | Pandemic unemployment shock |
| `2022-06-30` | High inflation |
| `2023-07-31` | High interest rates and an inverted yield curve |

Prepare the deterministic snapshots without making LLM calls:

```bash
./docker_cmd.sh 'cd /git_root/class_project/msml610/Fall2026/projects/UmdTask802_MSML610_Fall2026_CrewAI && python multi_snapshot_evaluation.py'
```

Run the CrewAI workflow and single-LLM baseline:

```bash
./docker_cmd.sh 'cd /git_root/class_project/msml610/Fall2026/projects/UmdTask802_MSML610_Fall2026_CrewAI && python multi_snapshot_evaluation.py --run-llm'
```

Generated model responses are cached by cutoff. Repeating the command reuses completed outputs unless `--force` is supplied.

| Method | Correct | Required | Mean accuracy | Mean completeness | Pooled required-claim score |
|---|---:|---:|---:|---:|---:|
| Three-agent CrewAI workflow | 48 | 48 | 1.00 | 1.00 | 1.00 |
| Single-LLM baseline | 48 | 48 | 1.00 | 1.00 | 1.00 |

Both methods correctly reported all 48 required claims across the four snapshots. The experiment therefore found no numerical-accuracy improvement from using the three-agent workflow over this strong baseline. CrewAI still provides a modular and auditable separation between data collection, deterministic calculation, and report generation.

The historical snapshots use the current FRED series truncated at each observation cutoff. They are not revision-aware ALFRED vintages.

## Export the Final Report

```bash
./docker_cmd.sh 'cd /git_root/class_project/msml610/Fall2026/projects/UmdTask802_MSML610_Fall2026_CrewAI && python reporting.py --snapshot-dir data/fred/2026-11-04 --briefing outputs/fred_macro_briefing_2026-11-04.md'
```

This creates:

```text
outputs/fred_macro_report.md
outputs/fred_macro_series.png
```

The final report contains the generated briefing, a summary table, and a four-panel line chart.

## Tests

Run the deterministic pipeline tests inside Docker:

```bash
./docker_cmd.sh 'cd /git_root/class_project/msml610/Fall2026/projects/UmdTask802_MSML610_Fall2026_CrewAI && PYTHONPATH=. python -m pytest -q test/test_fred_pipeline.py'
```

Run the Docker and notebook integration tests from the Mac host:

```bash
export CSFY_DOCKER_ENGINE=docker
python -m pytest -q test/test_docker_template.py
```

The 11 pipeline tests cover deterministic calculations, observation cutoffs, cutoff validation and propagation, field-aware evaluation, rounding, missing and incorrect claims, report export, snapshot caching, and aggregate reporting. They do not call FRED or Gemini. Notebook LLM execution is disabled by default.

## Limitations

- The historical experiments truncate the current FRED series and are not revision-aware ALFRED vintages.
- FRED data revisions may change results when snapshots are downloaded again.
- The evaluator checks claims against values recomputed from each saved snapshot.
- Four perfect snapshots do not establish that either method will always produce correct numbers.
- The experiment measures required-claim accuracy and completeness, not writing quality, latency, token usage, or cost.
- Gemini model availability, demand, and rate limits can change.
