# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # BambooAI API Tour
#
# A runnable, API-focused guide to BambooAI: what it is, how to configure it, and how to launch the conversation loop.
#
# How to use this notebook
# - Run top-to-bottom if you can.
# - Some cells call an LLM and may incur cost. You can still read the markdown safely without running.
#
# Related notebooks
# - `bambooai.example.ipynb` is a narrative, end-to-end walkthrough with more feature demos.
#

# %% [markdown]
# ## What BambooAI is
# BambooAI is an open-source, LLM-powered data analysis agent for pandas workflows. You ask questions in natural language, BambooAI plans the steps, generates or executes code, and returns tables or charts, depending on what you ask for.
#
# When to use it
# - You want an interactive, conversational way to explore a DataFrame.
# - You need automated code generation with error correction and iterative feedback loops.
# - You want analysis memory via a vector DB or semantic grounding via an ontology.
#
# Feature highlights
# - Natural language interface for data analysis with automatic Python generation.
# - Multi-step planning, error correction, and code editing loops.
# - Vector database integration for knowledge storage and semantic recall.
# - Ontology grounding via `.ttl` files for domain-specific semantics.
# - Web UI (Flask) and Jupyter notebook support.
#
# Model support
# - API providers: OpenAI, Google (Gemini), Anthropic, Groq, Mistral.
# - Local providers: Ollama and a selection of local models.
#

# %% [markdown]
# ## How BambooAI works (short form)
# 1. Initiation: start with a user question or prompt for one.
# 2. Task routing: decide between pure text responses or code generation.
# 3. User feedback: ask clarifying questions when ambiguity is detected.
# 4. Dynamic prompt build: assemble context, plan, and similar-task recall.
# 5. Debugging and execution: run generated code and auto-correct errors.
# 6. Results and knowledge base: rank answers and optionally store them in a vector DB.
#

# %% [markdown]
# ## Quick start (minimal usage)
# ```python
# from bambooai import BambooAI
# import pandas as pd
#
# df = pd.read_csv("testdata.csv")
# bamboo = BambooAI(df=df, planning=True, vector_db=False, search_tool=True)
# bamboo.pd_agent_converse()
# ```
#

# %% [markdown]
# ## How this notebook is organized
# - Environment and logging setup.
# - LLM configuration inspection.
# - Helper functions that wrap BambooAI’s API.
# - Environment sanity check + dataset load.
# - A minimal “hello world” run (for full E2E, see `bambooai.example.ipynb`).
# - Prompt cookbook (short version).
# - Sequential feature-focus walkthrough of each parameter (with custom prompts + “what to expect”).
# - Troubleshooting and cleanup notes.
#

# %% [markdown]
# ## 1) Setup and dependencies
#
# The BambooAI API relies on standard data science libraries plus `bambooai`, `plotly`, `pandas`, and `python-dotenv`. Make sure the dataset lives here and that your `.env` file defines `EXECUTION_MODE` before you execute the notebook.
#
# Data location
# - The default dataset path is `_DEFAULT_CSV = Path("testdata.csv")` in `Bambooai-blog/bambooai_utils.py`.
# - Override it with `--csv-path` (parser in `bambooai_utils.py`) or update `_DEFAULT_CSV` directly.
#
# Plot rendering (optional)
# - If interactive plots fail, set `PLOTLY_RENDERER=json` in your environment before running the imports cell.
#
# **This cell will:**
# - Load core imports and configure plotting defaults.
# - Add helper paths so `bambooai_utils.py` can be imported.
#

# %%
# Run this cell
import sys
# # %pip install -q qdrant-client
sys.path.insert(0, "/app/tutorials-Bambooai-blog")
import logging
import os
import sys
from pathlib import Path
from bambooai import BambooAI
from dotenv import load_dotenv
load_dotenv()
import numpy as np
import pandas as pd
import plotly.io as pio
import seaborn as sns

sys.path.insert(0, "/app/helpers_root")

plotly_renderer = os.getenv("PLOTLY_RENDERER", "jupyterlab")
pio.renderers.default = plotly_renderer
sns.set_style("whitegrid")
np.set_printoptions(suppress=True, precision=6)

load_dotenv()
logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)

_ROOT_DIR = Path.cwd()
if not (_ROOT_DIR / "helpers_root").exists():
    _ROOT_DIR = _ROOT_DIR.parent
_HELPERS_ROOT = _ROOT_DIR / "helpers_root"
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))
if str(_HELPERS_ROOT) not in sys.path:
    sys.path.insert(0, str(_HELPERS_ROOT))


# %% [markdown]
# ## 2) Installation and configuration
#
# At minimum you need:
# - `pip install bambooai`
# - API keys in `.env` for the LLM provider you choose.
#
# BambooAI reads its agent model settings from `LLM_CONFIG` (env var) or `LLM_CONFIG.json` in the working directory. If neither is present, it falls back to its package defaults. Prompt templates can be customized by creating `PROMPT_TEMPLATES.json` from the provided sample file.
#
# **This cell will:**
# - Inspect the active LLM configuration (if any).
#

# %%
# Run this cell
import json

config_env = os.getenv("LLM_CONFIG", "").strip()
config_path = Path("LLM_CONFIG.json")
config = None

if config_env:
    config = json.loads(config_env)
    source = "LLM_CONFIG env var"
elif config_path.exists():
    config = json.loads(config_path.read_text())
    source = "LLM_CONFIG.json"

if config:
    print(f"{source} found. Agent configs:")
    for agent in config.get("agent_configs", []):
        details = agent.get("details", {})
        print(f"- {agent.get('agent')}: {details.get('provider')}/{details.get('model')}")
else:
    print("No LLM_CONFIG found. BambooAI will use its package defaults (see BambooAI docs/config).")


# %% [markdown]
# ## Config reference (files)
# - `LLM_CONFIG.json` maps agents to models, providers, and parameters. Use `LLM_CONFIG.json` as a starting point, or set `LLM_CONFIG` in `.env` to inline the JSON.
# - Prompt templates can be overridden by providing `PROMPT_TEMPLATES.json` (created from `PROMPT_TEMPLATES_sample.json`) in the working directory.
# - Each run records a JSON log file (for example `logs/bambooai_run_log.json`) plus a consolidated log that tracks multiple runs.
#

# %% [markdown]
# ## Key parameters
# | Parameter | Type | Default | Impact |
# | --- | --- | --- | --- |
# | `df` | `pd.DataFrame` | `None` | Primary dataset for analysis. If not provided, BambooAI may attempt to source data from the internet or auxiliary datasets. |
# | `auxiliary_datasets` | `list[str]` | `None` | Additional datasets available during code execution. |
# | `max_conversations` | `int` | `4` | Number of user/assistant pairs retained in memory. |
# | `search_tool` | `bool` | `False` | Enables external search capability when needed. |
# | `planning` | `bool` | `False` | Enables multi-step planning for complex requests. |
# | `webui` | `bool` | `False` | Runs BambooAI as a Flask-based web app. |
# | `vector_db` | `bool` | `False` | Enables vector memory for recall or retrieval. |
# | `df_ontology` | `str` | `None` | Path to a `.ttl` ontology file for semantic grounding. |
# | `exploratory` | `bool` | `True` | Enables expert selection for query handling. |
# | `custom_prompt_file` | `str` | `None` | YAML file with custom prompt templates. |
#
# Vector DB and ontology notes
# - `vector_db=True` enables episodic memory. Pinecone and Qdrant are supported via `.env` configuration. When set to True, the model will first attempt to search its vector DB for previous conversation for clues to answer questions. If nothing is found, it attempts to reason on its own and answer. At the end of each output, BambooAI asks users to rank the solution it provided on a scale of 1-10 (10 being awesome and 1 being really bad). If you rank it pretty high (>6), the model will try to reference it for future conversations to learn from. 
#
# - Pinecone example env vars: `VECTOR_DB_TYPE=pinecone`, `PINECONE_API_KEY=...` (some versions also use `PINECONE_ENV`).
#
# - Qdrant example env vars: `VECTOR_DB_TYPE=qdrant`, `QDRANT_URL=...`, `QDRANT_API_KEY=...` (optional for local, required for cloud).
#
# - Pinecone embeddings are supported with `text-embedding-3-small` (OpenAI) or `all-MiniLM-L6-v2` (HF).
#
# - `df_ontology` expects a `.ttl` ontology file (RDF/OWL) that defines classes, properties, and relationships.
#

# %% [markdown]
# ## 3) API helper functions
#
# The BambooAI helpers are defined in `bambooai_utils.py`. The following cell prints each helper's docstring so you can quickly understand their responsibility.
#
# **This cell will:**
# - Print docstrings for the helper functions used by this notebook.
#

# %%
# Run this cell
from bambooai_utils import (
    _DEFAULT_CSV,
    _build_bamboo_agent,
    _load_dataframe,
    _run_agent,
    _setup_env,
    _parse,
    _resolve_execution_mode,
)

api_docs = {
    "_setup_env": _setup_env.__doc__,
    "_parse": _parse.__doc__,
    "_resolve_execution_mode": _resolve_execution_mode.__doc__,
    "_load_dataframe": _load_dataframe.__doc__,
    "_build_bamboo_agent": _build_bamboo_agent.__doc__,
    "_run_agent": _run_agent.__doc__,
}

for name, doc in api_docs.items():
    if doc:
        print(f"{name} docstring:\n{doc.strip()}\n")
    else:
        print(f"{name} has no docstring\n")

print(f"Default CSV path: {_DEFAULT_CSV}")


# %% [markdown]
# ## 4) EXECUTION_MODE and configuration requirements
#
# Our wrapper resolves `EXECUTION_MODE` as `args.execution_mode` or the environment variable `EXECUTION_MODE`. If both are empty, `_resolve_execution_mode` raises an assertion.
#
# Any non-empty value is accepted by the wrapper. Team convention is `local` or `api` (update to match your environment).
#
# **This cell will:**
# - Set `EXECUTION_MODE` inside the notebook and confirm the value.
#

# %%
# Run this cell
import os

os.environ["EXECUTION_MODE"] = "local"  # Update as needed
print("EXECUTION_MODE from env:", os.getenv("EXECUTION_MODE"))


# %% [markdown]
# ### What EXECUTION_MODE does
# - It controls where BambooAI executes generated code, based on your BambooAI setup.
# - Common values are `local` (run in-process) and `api` (run via a configured executor).
# - If you are unsure, it is recommended to start with `local`.
#

# %% [markdown]
# ## 5) Sanity check (environment + data)
#
# Use this quick check to confirm environment configuration and dataset readiness before running the agent.
#
# **This cell will:**
# - Print key env vars (masked).
# - Confirm the dataset path exists.
# - Load the dataframe and preview rows.
#

# %%
# Run this cell


def _mask(value: str) -> str:
    if not value:
        return "<not set>"
    if len(value) <= 6:
        return "*" * len(value)
    return f"{value[:3]}...{value[-2:]}"

keys = [
    "EXECUTION_MODE",
    "LLM_CONFIG",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "PINECONE_API_KEY",
]

print("Environment")
for key in keys:
    value = os.getenv(key, "")
    if key == "LLM_CONFIG":
        display_value = "<set>" if value else "<not set>"
    else:
        display_value = _mask(value)
    print(f"- {key}: {display_value}")

args = _parse().parse_args([])
csv_path = Path(args.csv_path) if args.csv_path else _DEFAULT_CSV
print("\nDataset")
print(f"- path: {csv_path}")
print(f"- exists: {csv_path.exists()}")

df = _load_dataframe(csv_path)
print(f"\nDataframe shape: {df.shape}")
display(df.head())


# %% [markdown]
# Expected output (healthy setup)
# - `EXECUTION_MODE` shows a masked non-empty value.
# - Dataset `exists: True`.
# - Dataframe shape has at least one row.
#

# %% [markdown]
# ## 6) Hello world (single prompt)
#
# This is the smallest interactive run. It builds an agent with minimal flags and starts the loop.
# When prompted, paste one simple question, then type `exit` or press Ctrl+D to stop.
#
# Cost note: This cell calls an LLM and may incur cost.
#
# **This cell will:**
# - Build a minimal agent.
# - Start the interactive loop.
#

# %%
# Run this cell
if "df" not in globals():
    csv_path = _DEFAULT_CSV
    df = _load_dataframe(csv_path)

bamboo_quick = _build_bamboo_agent(
    df,
    planning=False,
    vector_db=False,
    search_tool=False
)
print("BambooAI ready. When the loop starts, paste one prompt, then type 'exit' or press Ctrl+D to stop.")
_run_agent(bamboo_quick)


# %% [markdown]
# ### Full E2E Run
# For a full end-to-end workflow with dataset artifacts and a longer narrative, see `bambooai.example.ipynb`.
#

# %% [markdown]
# ## 7) Prompt cookbook (short)
#
# Use these examples to get quick wins. For a larger cookbook and narrative flow, see `bambooai.example.ipynb`.
#
# Basic EDA
# - "List the columns and their data types."
# - "Show summary stats for numeric columns and note any missing values."
#
# Visualization
# - "Plot a histogram of `monthly_spend_usd` with 30 bins and label axes."
#
# Advanced
# - "Detect anomalies in daily `monthly_spend_usd` using a 7-day rolling z-score; return flagged dates."
#

# %% [markdown]
# ## 8) Feature focus: parameters (sequential)
#
# This section walks through each BambooAI parameter (except `df` and `webui`, which are covered elsewhere) with a short prompt and expected behavior.
#

# %% [markdown]
# ### Feature focus: auxiliary_datasets
#
# Use auxiliary datasets when the primary dataframe needs enrichment (lookups, joins, mapping tables).
#
# Custom prompt
# - Join the auxiliary dataset on `country` and summarize average `monthly_spend_usd` by region.
#
# What to expect
# - The agent should load the auxiliary CSV and perform a join.
# - Output should include the joined fields and a grouped summary.
#
# Cost note: this cell calls an LLM and may incur cost.
#
# **This cell will:**
# - Create a small auxiliary dataset.
# - Build a BambooAI agent configured with `auxiliary_datasets`.
# - Start the interactive loop.
#

# %%
# Run this cell


if "df" not in globals():
    csv_path = _DEFAULT_CSV
    df = _load_dataframe(csv_path)

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)
aux_path = ARTIFACTS_DIR / "auxiliary_demo.csv"
aux_df = pd.DataFrame(
    {
        "country": ["US", "CA", "DE"],
        "region_label": ["North America", "North America", "Europe"],
    }
)
aux_df.to_csv(aux_path, index=False)
print("Wrote auxiliary dataset:", aux_path)

bamboo_aux = BambooAI(
    df=df,
    auxiliary_datasets=[str(aux_path)],
    planning=False,
    vector_db=False,
    search_tool=False,
)
print("Auxiliary datasets agent ready.")
_run_agent(bamboo_aux)


# %% [markdown]
# ### Feature focus: max_conversations
#
# This limits how much recent chat history BambooAI keeps in memory.
#
# Custom prompt
# - Earlier you listed the average monthly spend of europe and north america, how much was it?
#
# What to expect
# - With a low value (e.g., 1), the agent may forget older context and ask you to restate details.
# - With higher values, it should retain more prior turns.
#
# Cost note: this cell calls an LLM and may incur cost.
#
# **This cell will:**
# - Build a BambooAI agent with `max_conversations=1` to demonstrate short memory.
# - Start the interactive loop.
#

# %%
# Run this cell


if "df" not in globals():
    csv_path = _DEFAULT_CSV
    df = _load_dataframe(csv_path)

bamboo_short_memory = BambooAI(
    df=df,
    max_conversations=1,
    planning=False,
)
print("Agent ready with max_conversations=1.")
_run_agent(bamboo_short_memory)


# %% [markdown]
# ### Feature focus: search_tool
#
# Enable this when you want BambooAI to pull in external context from the web.
#
# Custom prompt
# - Find a short definition of `customer churn` and explain how it might map to our dataset.
#
# What to expect
# - If the search tool is configured, the agent should fetch external context and cite or summarize it.
# - If not configured, you may see a tool error or a warning.
#
# Cost note: this cell calls an LLM and may incur cost.
#
# **This cell will:**
# - Try to build an agent with `search_tool=True` and report any setup errors.
# - Start the interactive loop if initialization succeeds.
#

# %%
# Run this cell
if "df" not in globals():
    csv_path = _DEFAULT_CSV
    df = _load_dataframe(csv_path)

try:
    bamboo_search = _build_bamboo_agent(
        df,
        planning=False,
        vector_db=False,
        search_tool=True,
    )
    print("Search tool enabled agent ready.")
    _run_agent(bamboo_search)
except Exception as e:
    print("Search tool init failed. Check search tool availability and credentials.")
    print("Error:", e)


# %% [markdown]
# ### Feature focus: planning
#
# Planning helps BambooAI solve multi-step or ambiguous tasks by outlining a plan before executing code.
#
# Custom prompt
# - Compare revenue trends by region, identify the top 3 outliers, and explain possible causes.
#
# What to expect
# - The agent should produce a plan, then execute steps to answer.
# - For simple prompts, planning may add latency without changing results.
#
# Cost note: this cell calls an LLM and may incur cost.
#
# **This cell will:**
# - Build an agent with `planning=True`.
# - Start the interactive loop.
#

# %%
# Run this cell
if "df" not in globals():
    csv_path = _DEFAULT_CSV
    df = _load_dataframe(csv_path)

bamboo_planning = _build_bamboo_agent(
    df,
    planning=True,
    vector_db=False,
    search_tool=False,
)
print("Planning-enabled agent ready.")
_run_agent(bamboo_planning)


# %% [markdown]
# ### Feature focus: vector_db
#
# Vector DB enables memory and retrieval over prior conversations and documents.
#
# Custom prompt
# - "Using what you learned earlier, summarize the top 2 churn drivers."
#
# What to expect
# - With a configured vector DB, the agent can retrieve past context instead of re-deriving it.
# - Without proper credentials, initialization will fail.
#
# Cost note: this cell calls an LLM and may incur cost.
#
# **This cell will:**
# - Try to build an agent with `vector_db=True` and report any setup errors.
# - Start the interactive loop if initialization succeeds.
#

# %%
# Run this cell
if "df" not in globals():
    csv_path = _DEFAULT_CSV
    df = _load_dataframe(csv_path)

try:
    bamboo_vector = _build_bamboo_agent(
        df,
        planning=True,
        vector_db=True,
        search_tool=False,
    )
    print("Vector DB enabled agent ready.")
    _run_agent(bamboo_vector)
except Exception as e:
    print("Vector DB init failed. Check Pinecone/Qdrant env vars and credentials.")
    print("Error:", e)


# %% [markdown]
# ### Feature focus: df_ontology
#
# Ontology grounding provides schema-level meaning and constraints for columns and values.
#
# Custom prompt
# - Validate that `churned` and `has_premium` values match the ontology. Flag any invalid values.
#
# What to expect
# - The agent should reference ontology definitions and perform value checks.
# - If the ontology file is invalid, initialization may fail.
#
# Cost note: this cell calls an LLM and may incur cost.
#
# **This cell will:**
# - Create a tiny `.ttl` ontology.
# - Build an agent with `df_ontology`.
# - Start the interactive loop.
#

# %%
# Run this cell


if "df" not in globals():
    csv_path = _DEFAULT_CSV
    df = _load_dataframe(csv_path)

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)
ontology_path = ARTIFACTS_DIR / "mini_ontology.ttl"
ontology_path.write_text(
    """@prefix ex: <http://example.com/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:Customer a rdfs:Class .
ex:churned a rdfs:Property ;
  rdfs:domain ex:Customer ;
  rdfs:range xsd:boolean ;
  rdfs:label "churned" .
ex:has_premium a rdfs:Property ;
  rdfs:domain ex:Customer ;
  rdfs:range xsd:boolean ;
  rdfs:label "has_premium" .
ex:monthly_spend_usd a rdfs:Property ;
  rdfs:domain ex:Customer ;
  rdfs:range xsd:decimal ;
  rdfs:label "monthly_spend_usd" .
"""
)
print("Wrote ontology:", ontology_path)

bamboo_ontology = BambooAI(
    df=df,
    df_ontology=str(ontology_path),
    planning=True,
    exploratory=True,
)
print("Ontology grounded agent ready.")
_run_agent(bamboo_ontology)


# %% [markdown]
# ### Feature focus: exploratory
#
# Exploratory mode enables expert selection for query handling (e.g., routing to a specialist).
#
# Custom prompt
# - Analyze this dataset for churn drivers and suggest follow-up questions.
#
# What to expect
# - The agent may ask clarifying questions or choose a specialist persona before executing.
# - With `exploratory=False`, it should behave more directly without extra routing.
#
# Cost note: this cell calls an LLM and may incur cost.
#
# **This cell will:**
# - Build an agent with `exploratory=True`.
# - Start the interactive loop.
#

# %%
# Run this cell


if "df" not in globals():
    csv_path = _DEFAULT_CSV
    df = _load_dataframe(csv_path)

bamboo_exploratory = BambooAI(
    df=df,
    exploratory=True,
    planning=False,
)
print("Exploratory mode agent ready.")
_run_agent(bamboo_exploratory)


# %% [markdown]
# ### Feature focus: custom_prompt_file
#
# Custom prompts let you control response structure and tone.
#
# Custom prompt
# - Return a 3-bullet summary and a numbered action plan.
#
# What to expect
# - The agent should follow the style and structure defined in your prompt templates.
# - If the YAML file is missing or malformed, initialization may fail.
#
# Cost note: this cell calls an LLM and may incur cost.
#
# **This cell will:**
# - Create a minimal custom prompts YAML file.
# - Build an agent with `custom_prompt_file`.
# - Start the interactive loop.
#

# %%
# Run this cell


if "df" not in globals():
    
    csv_path = _DEFAULT_CSV
    df = _load_dataframe(csv_path)

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)
custom_prompt_path = ARTIFACTS_DIR / "custom_prompts.yaml"
custom_prompt_path.write_text(
    "# Placeholder prompts for BambooAI\n"
    "planner_prompt: \"You are a careful planner.\"\n"
    "code_prompt: \"Write concise pandas code.\"\n"
)
print("Wrote custom prompts:", custom_prompt_path)

bamboo_custom = BambooAI(
    df=df,
    custom_prompt_file=str(custom_prompt_path),
    planning=False,
    exploratory=True,
)
print("Custom prompt agent ready.")
_run_agent(bamboo_custom)


# %% [markdown]
# ## 9) Troubleshooting
#
# Common failures and fixes:
# - Assertion failure: Execution mode cannot be empty. Set `EXECUTION_MODE` in `.env` or in the notebook cell above.
# - CSV file does not exist or wrong path. Verify `--csv-path`, update `_DEFAULT_CSV`, or point to the correct file.
# - LLM config missing or auth errors. Ensure API keys are in `.env` and `LLM_CONFIG` or `LLM_CONFIG.json` is set correctly.
# - pandas read errors or empty df. Check CSV encoding, delimiter, and whether the file has rows.
# - Vector DB errors. Confirm Pinecone/Qdrant env vars and credentials.
# - Search tool errors. Confirm search tool availability and credentials.
#
# If the agent fails to start, re-run the sanity check cell and confirm environment settings before retrying.
#

# %% [markdown]
# ## 10) Cleanup and reset
# - Logs live under `logs/` and can be archived or deleted between runs.
# - To reset state, re-instantiate the agent. Some BambooAI versions also support `pd_agent_converse(action="reset")`.
# - If vector DB memory is enabled, use your provider’s tooling to clear stored records when needed.
#

# %% [markdown]
# ## 11) Optional: build agent without running (debugging)
#
# Use `_load_dataframe` and `_build_bamboo_agent` directly when you need to construct an agent programmatically without invoking `_run_agent`.
#
# **This cell will:**
# - Load the dataset.
# - Build the agent without starting the loop.
#

# %%
# Run this cell
csv_path = _DEFAULT_CSV
loaded_df = _load_dataframe(csv_path)
bamboo_agent = _build_bamboo_agent(loaded_df)

env_mode = os.getenv("EXECUTION_MODE", "<not set>")
print(f"Execution mode from environment: {env_mode}")
print(f"Loaded dataset shape: {loaded_df.shape}")
print(f"BambooAI agent ready: {type(bamboo_agent).__name__}")
print("\\nSample rows from the dataset:")
print(loaded_df.head())

