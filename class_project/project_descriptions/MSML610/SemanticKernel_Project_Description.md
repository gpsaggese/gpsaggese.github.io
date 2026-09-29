**Description**

Semantic Kernel (SK) is a model-agnostic SDK for building agents and composing LLMs
with your own functions and services. It exposes plugins (native code, prompts,
OpenAPI/MCP) that models can call, plus planning, memory and vector-DB integrations.
Works with C#, Python and Java.

Technologies Used
Semantic Kernel

- Function/plugin calling from models; prompt templates + skills
- Planning to sequence function calls for multi-step requests
- Memory and vector-DB connectors (Azure AI Search, Elasticsearch, etc.)
- Multi-agent support and local model options (e.g., Ollama)

### Fall2026

#### Tutorial

- Usual tutorial "Learn Semantic Kernel in 60 mins", following
  `.claude/skills/tutorials_in_60_mins.rules.md`
- Check if there is an existing tutorial and make it better
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Compare briefly with other agent frameworks (e.g., LangChain, LangGraph)
- Deliverables:
  - `semantic_kernel_utils.py`
  - `semantic_kernel.API.ipynb`
  - `semantic_kernel.example.ipynb`
- TODO(ai_gp): Improve this

#### Project: Multi-Agent Economic Report

- **Project Objective**: Specialized SK agents (gatherer, analyst, presenter)
  collaborate to produce an economic brief on GDP/inflation/unemployment trends
- **Dataset Suggestions**: World Development Indicators:
  [World Bank DataBank - WDI](https://databank.worldbank.org/source/world-development-indicators)
- **Tasks**:
  - **Define the Agents**: Gatherer pulls indicators via API; Analyst computes
    correlations and trends; Presenter writes narrative
  - **Share Memory**: Use SK memory to share intermediate results across agents
  - **Export the Report**: Export an HTML report with charts and takeaways
- **Bonus Ideas (Optional)**: Add a forecasting plugin for short-term projections
- TODO(ai_gp): Improve this

#### Milestones

- TODO(ai_gp): Add milestones related to the project

### Project 1: Function-Calling Chatbot

- **Difficulty**: 1 (Easy)
- **Project Objective**: Agent calls a Python function (plugin) to compute Iris
  summary stats and returns structured results
- **Dataset Suggestions**: Iris:
  [UCI - Iris](https://archive.ics.uci.edu/dataset/53/iris)
- **Tasks**:
  - Register a `@kernel_function` that loads CSV and returns per-feature mean/std
  - Initialize a chat agent; ask for "summary stats for Iris."
  - Verify outputs across multiple models (local vs. hosted)
- **Bonus Ideas (Optional)**: Add another function to output a small correlation
  table

### Project 2: Housing Price Analyzer

- **Difficulty**: 2 (Medium)
- **Project Objective**: Use SK planning + plugins to clean, analyze, and model a
  housing dataset; interpret coefficients and feature importance
- **Dataset Suggestions**: Real Estate Valuation (regression):
  [UCI - Real Estate Valuation](https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set)
- **Tasks**:
  - Plugins: data cleaning, correlation, and model training (linear
    regression/random forest)
  - Planner sequences functions from "analyze housing prices."
  - Output a notebook + short report with coefficients/importance and validation
    scores
- **Bonus Ideas (Optional)**: Add a feature-selection plugin (e.g., Lasso) and
  compare models
