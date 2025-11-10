**Description**

AG2 (formerly AutoGen) is an open‑source AgentOS for multi‑agent systems. It provides base agents (e.g., conversable assistants, user proxies), conversation patterns (group chats, sequential/nested flows), tool use and configurable human‑in‑the‑loop policies. Strong fit for planner/critic/executor setups.

Technologies Used
AG2

- Multi‑agent conversations with role‑specialized agents.
- Tool calling and optional code execution.
- Human‑in‑the‑loop policies and a `UserProxy` for approvals.
- Group chat managers and orchestration primitives.

---

### Project 1: Flower‑Classification Duo
**Difficulty**: 1 (Easy)

**Project Objective**:
Two agents: one trains a simple classifier on Iris; the other evaluates and reports.

**Dataset Suggestions**:
- Dataset: Iris.
- Source: [UCI – Iris](https://archive.ics.uci.edu/dataset/53/iris)

**Tasks**:
- Assistant loads data, splits train/test, trains SVM or k‑NN.
- Evaluator checks metrics, suggests tweaks.
- Run a short chat until accuracy stabilizes.

**Bonus Ideas (Optional)**:
- Add a critic agent that proposes hyperparameter grids.

---

### Project 2: Population‑Analysis Team
**Difficulty**: 2 (Medium)

**Project Objective**:
Loader, Analyst, and Reporter agents collaborate on world‑population data to surface trends and outliers.

**Dataset Suggestions**:
- Dataset: World Population by Country (to 2020).
- Source: [Kaggle – World Population by Countries](https://www.kaggle.com/datasets/muhammedtausif/world-population-by-countries)

**Tasks**:
- Loader ingests CSV and standardizes country names.
- Analyst computes growth rates, density, and regional aggregates.
- Reporter summarizes and saves a notebook + markdown brief.

**Bonus Ideas (Optional)**:
- Add a policy agent that checks for data‑quality issues before reporting.

---

### Project 3: Air‑Quality Advisory Network
**Difficulty**: 3 (Hard)

**Project Objective**:
Ingestion, forecasting, and policy agents collaborate to predict PM2.5 peaks and generate recommendations.

**Dataset Suggestions**:
- Dataset: Beijing Multi‑Site Air Quality.
- Source: [UCI – Beijing Multi‑Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijing%2Bmulti%2Bsite%2Bair%2Bquality%2Bdata)

**Tasks**:
- Ingestion agent loads/resamples time series.
- Modeling agent builds forecasts and flags peaks.
- Policy agent compares to thresholds and drafts guidance; user proxy approves.

**Bonus Ideas (Optional)**:
- Add a weather‑feature tool to improve forecasts.
