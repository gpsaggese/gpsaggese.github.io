# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Deep Agents — API overview (used in `langchain.example.ipynb`)
#
# Deep Agents (the `deepagents` package used in this tutorial) is an optional layer that bundles a few "agent app" conveniences:
#
# - a ready-to-run agent loop (`create_deep_agent(...)`)
# - a toolbox (todos, filesystem tools, delegation to subagents)
# - pluggable **backends** (where state/files/stores live)
# - safety controls like sandboxing + HITL gates
#
# This notebook keeps Deep Agents coverage focused on the *public surface*:
# - `create_deep_agent(...)`
# - Backends: `FilesystemBackend`, `StateBackend`, `StoreBackend`, `CompositeBackend`
# - Subagents: `CompiledSubAgent`
# - HITL gates: `interrupt_on=...` and `Command(resume=...)`
#
# For the full DA1–DA8 walkthrough, see `langchain.example.ipynb`.
#

# %%
# This cell will:
# - Enable auto-reloading so edits are picked up without restarting the kernel.
# - Import the notebook utility library (deep_agents_API_utils.py).
# - Verify that deepagents is importable.
# %load_ext autoreload
# %autoreload 2

import deep_agents_API_utils as ut

version = ut.check_deepagents()
print("deepagents:", version)


# %% [markdown]
# This next cell shows how Deep Agents' **virtual filesystem** works.
#
# - The agent will refer to files like `/workspace/hello.txt`.
# - Under the hood, that maps to a real folder you can see locally: `./workspace/hello.txt`.
#
# Why this matters:
# - it keeps agent file access *contained* (good for safety)
# - it makes it easy to inspect what the agent wrote
#

# %%
# This cell will:
# - Run a Deep Agents demo (filesystem write/read via virtual FilesystemBackend).
# run_filesystem_demo is defined in deep_agents_API_utils.
paths, preview = ut.run_filesystem_demo(ut.get_chat_model())
print("hello.txt paths on disk:", paths)
print("final message preview:", preview)


# %% [markdown]
# Deep Agents also supports **human-in-the-loop (HITL) gating** for risky file operations via `interrupt_on=...`.
#
# In plain English:
# - the agent can *propose* an `edit_file`
# - execution pauses and emits an interrupt payload
# - you resume with an explicit decision (`approve` / `reject`)
#
# The cell below wires the guardrail and runs one tiny approve flow so you can see the interrupt lifecycle end-to-end.
#

# %%
# This cell will:
# - Run a Deep Agents HITL approve-flow demo.
# run_hitl_demo is defined in deep_agents_API_utils.
agent, out, interrupted = ut.run_hitl_demo(ut.get_chat_model())
print("interrupted:", interrupted)
print("agent type:", type(agent))
