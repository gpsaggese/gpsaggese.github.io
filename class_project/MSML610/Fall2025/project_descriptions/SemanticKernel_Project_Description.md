**Description**

Semantic Kernel (SK) is a model‑agnostic SDK for building agents and composing LLMs with your own functions and services. It exposes plugins (native code, prompts, OpenAPI/MCP) that models can call, plus planning, memory and vector‑DB integrations. Works with C#, Python and Java.

Technologies Used
Semantic Kernel

- Function/plugin calling from models; prompt templates + skills.
- Planning to sequence function calls for multi‑step requests.
- Memory and vector‑DB connectors (Azure AI Search, Elasticsearch, etc.).
- Multi‑agent support and local model options (e.g., Ollama).

---

### Project 1: Function‑Calling Chatbot
**Difficulty**: 1 (Easy)

**Project Objective**:
Agent calls a Python function (plugin) to compute Iris summary stats and returns structured results.

**Dataset Suggestions**:
- Dataset: Iris.
- Source: [UCI – Iris](https://archive.ics.uci.edu/dataset/53/iris)

**Tasks**:
- Register a `@kernel_function` that loads CSV and returns per‑feature mean/std.
- Initialize a chat agent; ask for “summary stats for Iris.”
- Verify outputs across multiple models (local vs. hosted).

**Bonus Ideas (Optional)**:
- Add another function to output a small correlation table.

---

### Project 2: Housing Price Analyzer
**Difficulty**: 2 (Medium)

**Project Objective**:
Use SK planning + plugins to clean, analyze, and model a housing dataset; interpret coefficients and feature importance.

**Dataset Suggestions**:
- Dataset: Real Estate Valuation (regression).
- Source: [UCI – Real Estate Valuation](https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set)

**Tasks**:
- Plugins: data cleaning, correlation, and model training (linear regression/random forest).
- Planner sequences functions from “analyze housing prices.”
- Output a notebook + short report with coefficients/importance and validation scores.

**Bonus Ideas (Optional)**:
- Add a feature‑selection plugin (e.g., Lasso) and compare models.

---

### Project 3: Multi‑Agent Economic Report
**Difficulty**: 3 (Hard)

**Project Objective**:
Specialized SK agents (gatherer, analyst, presenter) collaborate to produce an economic brief on GDP/inflation/unemployment trends.

**Dataset Suggestions**:
- Dataset: World Development Indicators.
- Source: [World Bank DataBank – WDI](https://databank.worldbank.org/source/world-development-indicators)

**Tasks**:
- Gatherer pulls indicators via API; Analyst computes correlations and trends; Presenter writes narrative.
- Use SK memory to share intermediate results across agents.
- Export an HTML report with charts and takeaways.

**Bonus Ideas (Optional)**:
- Add a forecasting plugin for short‑term projections.