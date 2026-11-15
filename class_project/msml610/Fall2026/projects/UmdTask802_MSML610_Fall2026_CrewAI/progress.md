# MSML610 Fall 2026 CrewAI Project Progress

## Project Links

GitHub Issue: https://github.com/gpsaggese/gpsaggese.github.io/issues/802

Skeleton Pull Request: https://github.com/gpsaggese/gpsaggese.github.io/pull/903

Research Log: https://docs.google.com/document/d/14TTelaXrCq2gIhQEe7YMnIZwCeTt4NVaI66ayCiBQxw/edit?tab=t.0

Supporting Data, Outputs, and Screenshots: https://drive.google.com/drive/folders/1yUhSuIRlE9PthALNjnYsgXe_H99Daxnu

## Project Objective

The project builds a three agent CrewAI workflow that converts four Federal Reserve Economic Data series into a short macroeconomic briefing about inflation, unemployment, interest rates, and the Treasury yield curve.

The Researcher collects and profiles the data. The Analyst performs deterministic calculations using custom tools. The Writer converts the verified results into a readable briefing.

The numeric accuracy of the CrewAI briefing is compared with a single language model baseline that receives the raw data profile without access to the custom calculation tools.

## October 29, 2026

### Project Setup

The team reviewed the project description and repository contribution requirements.

The standard class project template was copied into the required project directory. The initial structure included the Dockerfile, Docker helper scripts, requirements, API notebook, example notebook, tests, and project description.

The initial project structure was prepared separately from the implementation so that it could be reviewed as the first project checkpoint.

### Findings

The project must remain entirely within its assigned project directory.

The Docker environment, notebooks, source files, tests, and documentation must follow the standard class project conventions.

### Decision

Submit the project skeleton separately from the implementation.

### Next

Build the Docker image and install CrewAI and its required dependencies.

## November 4, 2026

### Docker and CrewAI Environment

The Docker image was built using Python 3.12.

CrewAI version 1.14.7 and Google Gemini support were added to the project requirements.

The Docker image name was converted to lowercase because Docker repository names cannot contain uppercase characters.

The Docker command script was updated to pass `GEMINI_API_KEY` and `CREWAI_MODEL` from the host environment into the container. Secret values are not stored in source files, notebooks, output files, or Git history.

### FRED Data Pipeline

The data pipeline was implemented in `fred_tools.py`.

The pipeline downloads and profiles the following FRED series:

1. `CPIAUCSL`, Consumer Price Index
2. `UNRATE`, Unemployment Rate
3. `FEDFUNDS`, Federal Funds Rate
4. `T10Y2Y`, Ten Year Minus Two Year Treasury Spread

Each profile contains the latest date, latest value, previous year date, previous year value, number of missing observations, and CSV path.

The initial snapshot was saved under `data/fred/2026-11-04`.

### Macro Calculation Tool

The deterministic calculation tool computes annual inflation, the change in unemployment, the change in the federal funds rate, the Treasury spread, and the sign of the yield curve.

The initial calculation produced the following results:

1. Annual inflation was 3.72 percent.
2. The unemployment rate was 4.20 percent.
3. The annual unemployment change was negative 0.20 percentage points.
4. The federal funds rate was 3.88 percent.
5. The annual federal funds rate change was negative 0.21 percentage points.
6. The Treasury spread was 0.32 percentage points.
7. The yield curve was positive.

### CrewAI Workflow

Three CrewAI agents were implemented.

The Researcher uses a custom tool to download and profile the FRED data.

The Analyst uses a custom tool to calculate the required macroeconomic statistics.

The Writer converts the verified data and calculations into a short briefing.

The agents run sequentially so that the output from one stage is available to the next stage.

The workflow successfully generated a briefing covering inflation, the labor market, interest rates, the yield curve, and the economic outlook.

### Evaluation

The evaluation system independently recalculates the expected statistics from the saved CSV snapshot using pandas.

Numeric claims are extracted from the generated briefing and checked against the independently calculated values within their displayed rounding precision.

A single language model baseline was also generated from the raw data profile without access to the custom tools.

The CrewAI briefing contained 14 evaluated numeric claims. All 14 claims were correct, producing an accuracy of 1.0.

The single language model baseline contained 11 evaluated numeric claims. All 11 claims were correct, also producing an accuracy of 1.0.

### Findings

The initial experiment resulted in a tie between the CrewAI workflow and the single language model baseline.

The CrewAI workflow did not beat the baseline in this run. However, it produced more evaluated numeric claims while maintaining perfect numeric accuracy.

Additional experiments are required before concluding that the CrewAI workflow is more accurate than the baseline.

### Reporting

The reporting program exports a Markdown report and a four panel chart containing the Consumer Price Index, unemployment rate, federal funds rate, and Treasury yield spread.

The generated datasets, reports, chart, evaluation results, and screenshots are stored in the shared Google Drive project folder linked above.

### Testing

Three pipeline tests were added and passed successfully. They verify the macroeconomic calculations, numeric claim evaluation, and final report export.

Five Docker template tests passed successfully. They verify the API notebook, example notebook, Docker shell, Docker build, and Docker command scripts.

Both `crewai.API.ipynb` and `crewai.example.ipynb` executed successfully inside the Docker environment.

### Decisions

Use dated FRED snapshots so that the same data can be used by the agents and the independent evaluation.

Use deterministic pandas calculations instead of relying entirely on language model arithmetic.

Pass API credentials through environment variables.

Keep downloaded data, generated outputs, binary files, and screenshots outside Git and store them in the shared Google Drive folder.

Report the evaluation tie honestly and conduct additional experiments later.

### Next

Prepare the research documentation and submit the initial project skeleton for review.

## November 11, 2026

### Skeleton Pull Request

Pull Request 903 was submitted as the first project checkpoint.

The pull request contained the standard project template, Docker infrastructure, boilerplate notebooks, and CrewAI project description.

Professor Saggese approved and merged the pull request into the upstream `master` branch.

### Findings

Separating the skeleton from the implementation created a clear review checkpoint.

The implementation can now be submitted through an incremental branch and pull request.

### Decision

Create the next implementation branch from the updated upstream `master` branch using the required incremental branch naming convention.

### Next

Create `UmdTask802_MSML610_Fall2026_CrewAI_1`, restore the local implementation, exclude generated data and outputs, and prepare the implementation pull request.

## November 12, 2026

### Research Documentation

A shared Google Drive project folder was created for the research materials.

The dated FRED snapshot, generated reports, evaluation results, chart, and supporting screenshots were uploaded to Google Drive.

A Google Doc research log was created with the project objective, system description, experiment results, visual evidence, findings, and next steps.

The Google Doc and project folder are linked at the beginning of this file.

### Implementation Branch

The local `master` branch was updated from `origin/master` after the skeleton pull request was merged.

The incremental branch `UmdTask802_MSML610_Fall2026_CrewAI_1` was created.

The local implementation was restored onto the incremental branch.

The project `.gitignore` was updated so that `data/`, `outputs/`, temporary files, logs, generated HTML files, secrets, and local environment files are not committed.

The original Git stash remains available as a temporary safety backup until the implementation has been committed and pushed successfully.

### Current Status

The Docker environment works successfully.

The FRED download and profiling pipeline is implemented.

The Researcher, Analyst, and Writer agents are implemented.

The sequential CrewAI workflow generates the macroeconomic briefing.

The numeric evaluation and single language model baseline are implemented.

The report and chart export are implemented.

The pipeline, notebook, and Docker tests pass.

The research evidence and generated artifacts are stored in Google Drive.

The implementation is being prepared for its first review.

### Remaining Work

Review the implementation files for correctness, formatting, secrets, and repository compliance.

Check the API notebook against the complete Milestone 2 requirements, including custom tools and task execution patterns.

Check that the example notebook runs the full workflow and exports the required briefing and chart.

Run additional evaluation experiments because the initial CrewAI and baseline comparison resulted in a tie.

Consider implementing the optional Skeptic agent.

Stage and inspect only the intended source files.

Commit and push the implementation branch.

Open the second project pull request and request review.
