# Description

Semantic Kernel (SK) is a model-agnostic SDK for building agents and composing LLMs
with your own functions and services, available for C#, Python, and Java. It solves
the problem of connecting an LLM to code and data: models call plugins (native code,
prompts, OpenAPI, MCP), and SK adds planning, memory, and vector-DB integrations. It
is worth a 60-minute tutorial because a plugin, an agent, and shared memory are
enough to build a multi-step workflow that runs on hosted or local models.

## Technologies Used

Semantic Kernel

- Function/plugin calling from models; prompt templates + skills
- Planning to sequence function calls for multi-step requests
- Memory and vector-DB connectors (Azure AI Search, Elasticsearch, etc.)
- Multi-agent support and local model options (e.g., Ollama)

# Tutorial

- Implement the tutorial "Learn Semantic Kernel in 60 mins", following
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - Build it with `.claude/skills/tutorial_in_60_mins.create/SKILL.md`
  - Follow the workflow in `tutorials/README.gp.md` and the quality principles in
    `tutorials/tutorials_checklist.md`
- Check the previous tutorials and projects, listed in the section
  `Existing Tutorials and Projects` of
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - No earlier tutorial or project uses Semantic Kernel, so read the closest agent
    work
  - Read the `README.md` of `tutorials/LangChain/`, `tutorials/LangGraph/`,
    `tutorials/Autogen/`, and `tutorials/tutorial_pydanticAI/`
  - Read the `README.md` of the Fall2025 CrewAI project
    - `class_project/msml610/Fall2025/projects/UmdTask111_Fall_2025_CrewAI_project_medium/`
- Create the project dir following the class instructions in
  `class_project/README.md`, section `Contribution to the Repo`
  - Start from `class_project/project_template`
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Compare briefly with `tutorials/LangChain/` and `tutorials/LangGraph/` from the
  agent orchestration point of view
- Deliverables:
  - `semantic_kernel_utils.py`
  - `semantic_kernel.API.ipynb`
  - `semantic_kernel.example.ipynb`

# Project

## Project 1 (Fall2026): SEC Filings Fundamentals Assistant

- **Project Objective**: An agent calls a plugin on SEC EDGAR XBRL data to answer
  questions on company fundamentals, with numbers that match the source and beat the
  numbers recalled by a no-plugin LLM
- **Dataset Suggestions**: EDGAR XBRL company facts, e.g., Apple at
  `https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json`
  - Documented in
    [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
  - SEC requires a `User-Agent` header with a contact email
  - Ticker to CIK map at `https://www.sec.gov/files/company_tickers.json`
- **Tasks**:
  - **Ingest the Company Facts**: Map the tickers of 5 companies to CIK with
    `company_tickers.json`, download their company facts JSON with the `User-Agent`
    header, cache it, and save a table with the ticker, the concept, the fiscal
    year, and the annual value
  - **Register the Plugin**: Register a `@kernel_function` that takes a ticker, a
    concept (e.g., `NetIncomeLoss`, `Assets`), and a fiscal year, and returns the
    annual value from the company facts JSON
  - **Initialize the Agent**: Initialize a `ChatCompletionAgent` with automatic
    function calling and ask questions such as "What was the net income of Apple in
    fiscal 2023?" for 5 companies
  - **Verify the Outputs**: Check that the numbers in 20 answers match the XBRL
    values across models (local vs. hosted), and compare with a baseline that
    answers from model memory without the plugin
  - **Report the Results**: Collect the exact-match rate and the median relative
    error of each model in a comparison table
- **Bonus Ideas (Optional)**: Add a function for derived ratios (net margin,
  debt-to-assets); store 10-K risk-factor text in a vector-store memory and answer
  qualitative questions with citations

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Ingest the Company Facts
  - Result: project dir created and container running, and the table of annual XBRL
    values for the 5 companies
- Milestone 2: API notebook
  - Project tasks: Register the Plugin, Initialize the Agent
  - Result: `semantic_kernel.API.ipynb` covering `Kernel`, a plugin with
    `@kernel_function`, a `ChatCompletionAgent` with automatic function calling, and
    a local and a hosted chat model
- Milestone 3: Example notebook
  - Project tasks: Register the Plugin, Initialize the Agent, Verify the Outputs,
    Report the Results
  - Result: `semantic_kernel.example.ipynb` running end to end and producing the
    comparison table

## Project 2: Multi-Agent Economic Report

- **Project Objective**: Specialized SK agents (gatherer, analyst, presenter)
  collaborate to produce an economic brief on GDP, inflation, and unemployment trends
  whose numbers match the source data
- **Dataset Suggestions**: World Development Indicators:
  [World Bank DataBank - WDI](https://databank.worldbank.org/source/world-development-indicators)
- **Tasks**:
  - **Ingest the Indicators**: The gatherer pulls GDP growth, inflation, and
    unemployment for a set of countries through the World Bank API into a table
  - **Define the Agents**: Create the gatherer, analyst, and presenter as
    `ChatCompletionAgent` objects, with plugins written as `@kernel_function`
    (indicator fetch, correlations and trends)
  - **Share Memory**: Use SK memory to share intermediate results across agents
  - **Evaluate the Report**: Compute the fraction of numeric claims in the narrative
    that match the source table, and compare a local model with a hosted model by
    that fraction and by latency
  - **Export the Report**: Export an HTML report with charts and takeaways
- **Bonus Ideas (Optional)**: Add a forecasting plugin for short-term projections

## Project 3: Loan Default Risk Triage

- **Project Objective**: Let an SK agent plan and run a credit risk workflow (clean,
  train, score, explain) from a natural-language request, and check that its
  approve, refer, or decline decisions match a plain scikit-learn pipeline and beat a
  no-plugin LLM baseline
- **Dataset Suggestions**: Default of Credit Card Clients (30,000 clients, default
  in the next month):
  [UCI - Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Tasks**:
  - **Build the Plugins**: Write `@kernel_function` plugins for data cleaning, default
    rate by segment (e.g., education and age band), model training (logistic
    regression and gradient boosting), and scoring one applicant as a probability of
    default
  - **Plan the Workflow**: Let the model sequence the plugin functions with
    `FunctionChoiceBehavior.Auto` from the request "assess the default risk of
    applicant 17 and explain the main drivers", and log the order of the calls
  - **Write the Prompt Template**: Write a prompt template that turns the score and
    the top coefficients into an approve, refer, or decline note with fixed
    probability thresholds
  - **Evaluate the Workflow**: Report ROC-AUC and precision-recall AUC of the scoring
    plugin on a stratified held-out set, the agreement of the agent decision with the
    direct scikit-learn decision on 100 test applicants, and the same agreement for a
    baseline LLM that sees the applicant row but has no plugins
  - **Write the Report**: Output a notebook and a short report with the coefficients,
    the feature importance, and the agreement table
- **Bonus Ideas (Optional)**: Add a threshold-tuning plugin that trades off approval
  rate and expected loss; compare a local model with a hosted model
