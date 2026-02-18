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
# # Complete BambooAI Conversation Example
# 
# This notebook walks through a real BambooAI run, including environment setup, data loading, and the `pd_agent_converse` execution loop.
# 
# ## Highlights
# 
# - Load the `testdata.csv` dataset that ships with this tutorial.
# - Initialize logging and enforce `EXECUTION_MODE` from `.env`.
# - Build the BambooAI agent and run the conversation with the pandas dataset.

# %% [markdown]
# ## Scenario and Dataset
# 
# We model a BambooAI planning workflow on the provided dataset. The dataset includes structured features for demonstration, and the `.env` file must expose `EXECUTION_MODE` so the agent can run with a known mode.

# %%
import logging
import os
import sys
from pathlib import Path

_ROOT_DIR = Path.cwd()
if not (_ROOT_DIR / "helpers_root").exists():
    _ROOT_DIR = _ROOT_DIR.parent
_HELPERS_ROOT = _ROOT_DIR / "helpers_root"
if str(_HELPERS_ROOT) not in sys.path:
    sys.path.insert(0, str(_HELPERS_ROOT))
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import pandas as pd

import helpers.hdbg as hdbg
from bambooai import (
    _DEFAULT_CSV,
    _build_bamboo_agent,
    _load_dataframe,
    _resolve_execution_mode,
    _run_agent,
    _setup_env,
)

hdbg.init_logger(verbosity=logging.INFO, use_exec_path=True)
_setup_env()
execution_mode = _resolve_execution_mode(os.getenv("EXECUTION_MODE", ""))
_LOG = logging.getLogger(__name__)
_LOG.info("Example execution mode: %s", execution_mode)
print("Environment and logging are configured.")

# %% [markdown]
# ## Load and Inspect the Dataset
# 
# Use `_load_dataframe` to read `testdata.csv`, confirm the shape, and preview a few rows before we run the agent.

# %%
csv_path = _DEFAULT_CSV
df = _load_dataframe(csv_path)

_LOG.info("Dataset loaded with shape %s", df.shape)
print("Preview of the first rows:")
print(df.head(5).to_string(index=False))

# %% [markdown]
# ## Instantiate BambooAI and Run the Conversation
# 
# Build the agent with planning enabled and execute `pd_agent_converse()` through `_run_agent`. This is the same loop that powers the BambooAI planning workflow in production.

# %%
bamboo_agent = _build_bamboo_agent(df)
_run_agent(bamboo_agent)

# %% [markdown]
# ## Next Steps
# 
# After the conversation finishes, inspect the logs for planning steps, review `bamboo_agent` attributes if needed, and rerun with alternative CSV files or configuration flags to explore different behaviors.
