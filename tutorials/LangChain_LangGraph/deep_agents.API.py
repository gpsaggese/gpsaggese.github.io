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
# Deep Agents (the `deepagents` package used in this tutorial) is an optional layer that bundles a few “agent app” conveniences:
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
# - Run a Deep Agents demo (filesystem/todos/subagents/HITL).
try:
    import deepagents  # type: ignore
    from deepagents import create_deep_agent  # type: ignore
    from deepagents.backends import (  # type: ignore
        FilesystemBackend,
    )

    print("deepagents:", getattr(deepagents, "__version__", "(unknown)"))
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "This section requires `deepagents`.\n"
        f"Import error: {type(e).__name__}: {str(e)[:200]}"
    )


# %% [markdown]
# This next cell shows how Deep Agents’ **virtual filesystem** works.
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
# - Run a Deep Agents demo (filesystem/todos/subagents/HITL).
root = Path(".").resolve()
Path("workspace").mkdir(parents=True, exist_ok=True)

backend = FilesystemBackend(root_dir=str(root), virtual_mode=True)
agent = create_deep_agent(model=ut.get_chat_model(), backend=backend)

out = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Call write_file with file_path='/workspace/hello.txt' and content='hello'. "
                    "Then call read_file on '/workspace/hello.txt' and return the content."
                ),
            }
        ]
    }
)

paths = sorted([str(p) for p in Path("workspace").rglob("hello.txt")])
print("hello.txt paths on disk:", paths)
print(
    "final message preview:", getattr(out["messages"][-1], "content", "")[:200]
)


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
# - Run a Deep Agents demo (filesystem/todos/subagents/HITL).
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

try:
    from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig
except ModuleNotFoundError as e:  # pragma: no cover
    raise RuntimeError(
        """This Deep Agents HITL demo needs the tutorial dependencies.

Run it from `tutorials/LangChain_LangGraph` with `requirements.txt` installed (or via Docker).
"""
    ) from e

root = Path(".").resolve()
backend = FilesystemBackend(root_dir=str(root), virtual_mode=True)
agent = create_deep_agent(
    model=ut.get_chat_model(),
    checkpointer=MemorySaver(),
    backend=backend,
    interrupt_on={
        "edit_file": InterruptOnConfig(allowed_decisions=["approve", "reject"])
    },
)

thread_id = "API_HITL_GUARDRAIL"
agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "write_file /workspace/hitl_api_demo.txt with 'line1\nline2\n'",
            }
        ]
    },
    config={"configurable": {"thread_id": thread_id}},
)
out = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "edit_file /workspace/hitl_api_demo.txt replace 'line2' with 'LINE2_APPROVED' then read_file /workspace/hitl_api_demo.txt",
            }
        ]
    },
    config={"configurable": {"thread_id": thread_id}},
)

interrupted = "__interrupt__" in out
print("interrupted:", interrupted)
if interrupted:
    out = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config={"configurable": {"thread_id": thread_id}},
    )
print("agent type:", type(agent))
