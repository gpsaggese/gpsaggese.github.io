# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: bambooaivenv (3.12.3)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # BambooAI API Reference
# 
# This notebook documents the reusable helper functions that keep the BambooAI workflow structured while pointing to the dataset and environment configuration that powers the conversation.
# 
# ## What you will learn
# 
# - How the environment and logging are initialized before BambooAI starts.
# - What each helper function does before the agent runs.
# - How to load data and build an agent without executing the conversation loop.

# %% [markdown]
# ## Setup and Dependencies
# 
# The BambooAI API relies on standard data science libraries plus `bambooai`, `plotly`, `pandas`, and `python-dotenv`. Make sure the `testdata.csv` file lives here and that your `.env` file defines `EXECUTION_MODE` before you execute the notebook.

# %%
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import plotly.io as pio
import seaborn as sns

pio.renderers.default = "jupyterlab"
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
# ## API helper functions
# 
# The BambooAI helpers are defined in `bambooai.py`. The following cell prints each helper's docstring so you can quickly understand their responsibility.

# %%
from bambooai import (
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
# ## Loading data and instantiating BambooAI
# 
# Use `_load_dataframe` and `_build_bamboo_agent` directly when you need to construct an agent programmatically. This cell demonstrates how the API prepares the data and the agent without invoking `_run_agent`.

# %%
csv_path = _DEFAULT_CSV
loaded_df = _load_dataframe(csv_path)
bamboo_agent = _build_bamboo_agent(loaded_df)

env_mode = os.getenv("EXECUTION_MODE", "<not set>")
print(f"Execution mode from environment: {env_mode}")
print(f"Loaded dataset shape: {loaded_df.shape}")
print(f"BambooAI agent ready: {type(bamboo_agent).__name__}")
print("\\nSample rows from the dataset:")
print(loaded_df.head(3).to_string(index=False))
print("\\nCall `_run_agent(bamboo_agent)` to start the conversation loop when you are ready.")

# %% [markdown]
# ## Pinecone-powered episodic memory
# 
# BambooAI exposes `vector_db` as an optional flag when instantiating the agent. Turning this on and providing Pinecone credentials enables the planner/executor agents to persist summaries and retrieve them later. The cells below sketch what that configuration looks like and how you can read from Pinecone to enrich the next prompt.

# %%
try:
    import pinecone
except ImportError:  # pragma: no cover
    pinecone = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

PINECONE_INDEX = os.getenv("PINECONE_INDEX", "bambooai-memory")
PINECONE_DIMENSION = int(os.getenv("PINECONE_DIMENSION", "1536"))

def init_pinecone_index():
    if not pinecone:
        _LOG.warning("Pinecone SDK is missing; vector memory will be disabled.")
        return None

    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENVIRONMENT")
    if not api_key or not environment:
        _LOG.warning("PINECONE_API_KEY or PINECONE_ENVIRONMENT not set; skipping memory index setup.")
        return None

    pinecone.init(api_key=api_key, environment=environment)
    if PINECONE_INDEX not in pinecone.list_indexes():
        pinecone.create_index(
            name=PINECONE_INDEX,
            dimension=PINECONE_DIMENSION,
            metric="cosine",
        )
    return pinecone.Index(PINECONE_INDEX)

def embed_text_for_memory(text: str) -> list[float]:
    if not OpenAI:
        _LOG.warning("OpenAI SDK is missing; returning a zero vector for Pinecone.")
        return [0.0] * PINECONE_DIMENSION
    client = OpenAI()
    response = client.embeddings.create(model="text-embedding-ada-002", input=text)
    return response.data[0].embedding

pinecone_index = init_pinecone_index()
pinecone_token = bool(pinecone_index)

if pinecone_token:
    bamboo_agent_with_memory = _build_bamboo_agent(loaded_df, vector_db=True)
    _LOG.info("Vector DB enabled; BambooAI will persist agent summaries into Pinecone.")
    # Demonstrate how Pinecone stores the last interaction
    sample_context = "Last user question about revenue growth trends."
    placeholder_vector = embed_text_for_memory(sample_context)
    pinecone_index.upsert([
        ("memory-example", placeholder_vector, {"query": sample_context, "source": "demo"})
    ])

    query_vector = embed_text_for_memory("How did we describe the last task?")
    matches = pinecone_index.query(vector=query_vector, top_k=2, include_metadata=True)
    print("Pinecone matches you can pass back into a planner/executor prompt:")
    for match in matches["matches"]:
        print(f"- {match['metadata']['query']} (score={match['score']:.3f})")
else:
    bamboo_agent_with_memory = None
    _LOG.info("No Pinecone index available; continuing without vector memory.")

print(
    "When `vector_db=True`, BambooAI automatically consults the configured vector DB."
    " Use the matches retrieved above to prime the planner or executor before you request a new plan."
)
