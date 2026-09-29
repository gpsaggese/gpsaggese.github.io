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
- Create `tutorials/SemanticKernel/`, since it does not exist yet
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

## Project 1: Multi-Agent Economic Report

- **Difficulty**: 2 (Medium)
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

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Ingest the Indicators
  - Result: `tutorials/SemanticKernel/` container running, and the indicator table
    for the chosen countries
- Milestone 2: API notebook
  - Project tasks: Define the Agents, Share Memory
  - Result: `semantic_kernel.API.ipynb` covering `Kernel`, plugins with
    `@kernel_function`, prompt templates, function calling, and memory
- Milestone 3: Example notebook
  - Project tasks: Define the Agents, Share Memory, Evaluate the Report, Export the
    Report
  - Result: `semantic_kernel.example.ipynb` running end to end and exporting the HTML
    report

## Project 2: Housing Price Analyzer

- **Difficulty**: 2 (Medium)
- **Project Objective**: Use SK planning and plugins to clean, analyze, and model a
  housing dataset, and interpret coefficients and feature importance
- **Dataset Suggestions**: Real Estate Valuation (regression):
  [UCI - Real Estate Valuation](https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set)
- **Tasks**:
  - **Build the Plugins**: Write plugins for data cleaning, correlation, and model
    training (linear regression or random forest)
  - **Plan the Workflow**: Let the planner sequence the plugin functions from the
    request "analyze housing prices"
  - **Evaluate the Models**: Report the validation scores of the models
  - **Write the Report**: Output a notebook and a short report with the coefficients
    and the feature importance
- **Bonus Ideas (Optional)**: Add a feature-selection plugin (e.g., Lasso) and
  compare models

## Project 3: Function-Calling Chatbot

- **Difficulty**: 1 (Easy)
- **Project Objective**: An agent calls a Python function (plugin) to compute Iris
  summary statistics and returns structured results
- **Dataset Suggestions**: Iris:
  [UCI - Iris](https://archive.ics.uci.edu/dataset/53/iris)
- **Tasks**:
  - **Register the Plugin**: Register a `@kernel_function` that loads the CSV and
    returns the per-feature mean and standard deviation
  - **Initialize the Agent**: Initialize a chat agent and ask for "summary stats for
    Iris"
  - **Verify the Outputs**: Verify that the outputs match pandas across multiple
    models (local vs. hosted)
  - **Report the Results**: Collect the outputs of each model in a comparison table
- **Bonus Ideas (Optional)**: Add another function to output a small correlation
  table
