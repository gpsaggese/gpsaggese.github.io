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
| `evaluation.py` | Single-LLM baseline and numeric-claim evaluator |
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

For the November 4, 2026 snapshot:

| Method | Correct numeric claims | Total numeric claims | Fraction correct |
|---|---:|---:|---:|
| Three-agent CrewAI workflow | 14 | 14 | 1.00 |
| Single-LLM baseline | 11 | 11 | 1.00 |

The multi-agent workflow and baseline tied on numeric accuracy in this run. The CrewAI workflow included more numeric claims and provided a deterministic, auditable calculation path, but it did not outperform the baseline on the specified fraction-correct metric for this snapshot.

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
./docker_cmd.sh 'cd /git_root/class_project/msml610/Fall2026/projects/UmdTask802_MSML610_Fall2026_CrewAI && python -m pytest -q test/test_fred_pipeline.py'
```

Run the Docker and notebook integration tests from the Mac host:

```bash
export CSFY_DOCKER_ENGINE=docker
python -m pytest -q test/test_docker_template.py
```

The pipeline tests do not call FRED or Gemini. Notebook LLM execution is disabled by default.

## Limitations

- FRED data revisions may change results for snapshots downloaded on another date.
- The numeric evaluator checks claims against values recomputed from the saved snapshot.
- A perfect score on one snapshot does not establish that either method will always produce correct numbers.
- Gemini model availability and rate limits can change.