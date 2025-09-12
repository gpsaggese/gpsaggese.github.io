**Description**

TaskWeaver is a code‑first agent framework that turns natural‑language requests into executable Python code. It plans tasks, generates Python to run them, executes in a managed runtime, and maintains in‑memory state across an interactive session. Plugins let you add domain‑specific tools so the agent can operate at a higher level over data (e.g., DataFrames) and external systems.

Technologies Used
TaskWeaver

- Code generation + safe execution loop (planner → codegen → executor → memory).
- Domain‑specific plugins and example‑driven templates to steer plans.
- Rich data handling (e.g., pandas DataFrames) and multi‑turn state.
- Extensible reflection/verification before/after running generated code.

---

### Project 1: Anomaly Detective for Stock Prices
**Difficulty**: 1 (Easy)

**Project Objective**:
Detect daily price anomalies for a chosen stock using a TaskWeaver agent that fetches data and runs a simple z‑score detector.

**Dataset Suggestions**:
- Dataset: Yahoo Finance Stock Prices (curated CSVs).
- Source: [Kaggle – Time Series Forecasting with Yahoo Stock Price](https://www.kaggle.com/datasets/arashnic/time-series-forecasting-with-yahoo-stock-price)

**Tasks**:
- Create a `pull_data` plugin to load a selected ticker CSV.
- Implement a `detect_anomalies` plugin (rolling mean/std + z‑scores).
- Ask the agent to “find anomalies in TSLA daily close,” approve the plan, and run.
- Return a short summary and save outputs (CSV + chart) to disk.

**Bonus Ideas (Optional)**:
- Add seasonal decomposition and compare anomalies to earnings dates.
- Support multiple tickers and rank which had the strongest anomalies.

---

### Project 2: Custom EDA on Netflix Titles
**Difficulty**: 2 (Medium)

**Project Objective**:
Have the agent produce an EDA plan and notebook for the Netflix Titles dataset (categoricals, cardinality, cross‑tabs, simple numeric profiles).

**Dataset Suggestions**:
- Dataset: Netflix Movies and TV Shows.
- Source: [Kaggle – Netflix Titles](https://www.kaggle.com/datasets/shivamb/netflix-shows)

**Tasks**:
- Build a `load_dataset` plugin to read the CSV and infer basic schema.
- Let the agent propose an EDA plan; you approve or edit before execution.
- Generate a Jupyter notebook (papermill/nbconvert) with results and comments.
- Produce a short markdown report with key findings.

**Bonus Ideas (Optional)**:
- Cross‑tab rating vs. genre; track releases by year.
- Pull IMDb ratings (optional tool) to correlate with runtime or genre.

---

### Project 3: Multi‑Indicator World Bank Analysis
**Difficulty**: 3 (Hard)

**Project Objective**:
Ingest multiple World Bank indicators (GDP, population, CO₂), align by country/year, and surface relationships + change points.

**Dataset Suggestions**:
- Dataset: World Development Indicators.
- Source: [World Bank DataBank – WDI](https://databank.worldbank.org/source/world-development-indicators)

**Tasks**:
- Plugins for: indicator download, alignment/joins, correlation, trend stats.
- Customize planning templates so the agent picks methods by feature type.
- Compile a notebook + narrative report with correlations and anomalies.
- Save artifacts (CSV/HTML/PNG) and a reproducible run config.

**Bonus Ideas (Optional)**:
- Add change‑point detection to flag structural breaks in GDP growth.
- Include a “confirmation” step before running large, long jobs.