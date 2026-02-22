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
# # Description
#
# ## Learn LangChain in 60 Minutes — API notebook
#
# Welcome — if you’re new to LangChain/LangGraph, you’re in the right place.
#
# This notebook is the **“show me the pieces”** tour: lots of small, runnable snippets that build a mental model.
# When you’re ready for full workflows (agent loops, graphs, subagents, memory), jump to `langchain.example.ipynb`.

# %%
# !sudo /bin/bash -c "(source /venv/bin/activate; pip install --quiet jupyterlab-vim)"
# !jupyter labextension enable

# %% [markdown]
# ## APIs covered (parity with `langchain.example.ipynb`)
#
# A mental model before we start:
#
# - **LangChain** is the toolkit: prompts, models, tools, and composable building blocks ("runnables").
# - **LangGraph** is the orchestrator: stateful graphs, routing, checkpointing/memory, and **interrupts** for human‑in‑the‑loop (HITL).
# - **Deep Agents (`deepagents`)** is an optional, higher-level layer used later in this tutorial for “agent app” patterns
#   (filesystem tools, todos, subagents, sandboxing, and HITL gates).
#
# This notebook is a reference for the concrete APIs used in the examples notebook:
#
# - Models: `ChatOpenAI`, `ChatAnthropic` (configured via `.env`)
# - Prompts + LCEL: `ChatPromptTemplate`, `StrOutputParser`, composition with `|`
# - Runnables: `.invoke()`, `.batch()`, `.stream()`, `RunnableParallel`
# - Tools: `@tool` / `tool(...)`
# - Tool execution: `ToolNode`, `AIMessage.tool_calls`, `ToolMessage`
# - Injection: `InjectedState`, `InjectedStore`, `InMemoryStore`
# - Agents: `create_agent`, `AgentState`, `ToolRuntime`, `InjectedToolCallId`
# - LangGraph: `StateGraph`, `START`/`END`, reducers via `Annotated[..., reducer]`
# - HITL: `interrupt(...)` + `Command(resume=...)`
# - Deep Agents: `create_deep_agent`, `CompiledSubAgent`, backends, `interrupt_on=InterruptOnConfig(...)`
# - Notebooks as data: `nbformat`, `nbclient`, `papermill`
#
# If any of those names feel mysterious right now — perfect. We’ll introduce them with small examples.

# %% [markdown]
# # Imports
#
# We’ll do a tiny bit of setup before the fun parts.
# If the next cell errors with “No module named …”, that’s not you — it just means you’re not running with this tutorial’s pinned dependencies yet.
#

# %%
# This cell will:
# - Enable auto-reloading so edits are picked up without restarting the kernel.
# - Import the notebook utility library (`langchain.API_utils.py`) so all reusable functions and classes are available.
# %load_ext autoreload
# %autoreload 2

#import importlib.util as _ilu
import os
import sys
import importlib
from pathlib import Path as _Path


# def _require_import(module_name: str):
#     try:
#         return importlib.import_module(module_name)
#     except ModuleNotFoundError as e:
#         raise RuntimeError(
#             f"""Missing Python package {module_name!r}.

# This tutorial is meant to be run from `tutorials/LangChain_LangGraph` with its pinned dependencies.

# Quick fixes:
# - Docker (recommended): `cd tutorials/LangChain_LangGraph && docker compose up --build`
# - Local venv: `cd tutorials/LangChain_LangGraph && pip install -r requirements.txt`
# """
#         ) from e


# langchain = _require_import("langchain")
# langchain_core = _require_import("langchain_core")
# langgraph = _require_import("langgraph")

# # Load utility library for this notebook.
# _utils_path = _Path("langchain.API_utils.py")
# _spec = _ilu.spec_from_file_location("langchain_api_utils", str(_utils_path))
# _mod = _ilu.module_from_spec(_spec)
# sys.modules["langchain_api_utils"] = _mod
# _spec.loader.exec_module(_mod)

import langchain
import langchain_core
import langgraph

import langchain_API_utils as ut


# %%
import logging
import platform

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)
_LOG = logging.getLogger("learn_langchain.api")

# TODO(ai_gp): Move it to a function in *_utils.py and use print instead of _LOG.info
_LOG.info("python=%s", sys.version.split()[0])
_LOG.info("platform=%s", platform.platform())
_LOG.info("langchain=%s", getattr(langchain, "__version__", "unknown"))
_LOG.info("langchain_core=%s", getattr(langchain_core, "__version__", "unknown"))
_LOG.info("langgraph=%s", getattr(langgraph, "__version__", "unknown"))
_LOG.info("LLM_PROVIDER=%s", os.getenv("LLM_PROVIDER", "(unset)"))


# %% [markdown]
# ## Model (configured via `.env`)
#
# These notebooks are provider-agnostic: you pick a provider in `.env`, and the helper function builds the right chat model.
#
# Supported now:
# - `openai`
# - `anthropic`
# - optional `ollama` (install `langchain-ollama` first)
#
# Optional observability:
# - set `LANGSMITH_TRACING=true` (+ `LANGSMITH_API_KEY`) to trace runs in LangSmith
#
# If you don’t want to spend money while reading, you can skip the LLM-invoking cells — the markdown still explains what they’re doing.
#

# %%
# This cell will:
# - Load `.env` and check for optional LangSmith tracing.
# - `LlmConfig`, `load_llm_config`, and `get_chat_model` are defined in `langchain.API_utils`.
import os

from dotenv import load_dotenv

# LlmConfig, _require_env, load_llm_config, get_chat_model are in langchain.API_utils.

load_dotenv()

if os.getenv("LANGSMITH_TRACING", "").strip().lower() in {"1", "true", "yes"}:
    _LOG.info("LangSmith tracing requested (LANGSMITH_TRACING=true).")


# %%
# This cell will:
# - Instantiate the chat model from your `.env` configuration.
llm = ut.get_chat_model()
llm


# %% [markdown]
# ## Local dataset (`data/T1_slice.csv`)
#
# We’ll use a tiny local CSV (shipped with this tutorial) so the examples feel concrete.
#
# Two small conveniences happen in the next cell:
# - we load it into a Pandas DataFrame for prompt/context demos
# - we also copy it under `./workspace/data/` so filesystem tools can refer to it as `/workspace/data/T1_slice.csv`
#
# That “workspace” detail will matter once we get to sandboxed filesystem access.
#

# %%
# This cell will:
# - Load the local dataset into a Pandas DataFrame and prepare the time column.
# - Copy the dataset under `./workspace/data/` so Deep Agents can access it via `/workspace/...`.
from pathlib import Path
import shutil

import pandas as pd

DATASET_PATH = Path("data/T1_slice.csv").resolve()
df = pd.read_csv(DATASET_PATH)
TIME_COL = "Date/Time"
if TIME_COL in df.columns:
    df[TIME_COL] = pd.to_datetime(
        df[TIME_COL], format="%d %m %Y %H:%M", errors="coerce"
    )

# Make the dataset visible to Deep Agents filesystem tools under `/workspace/...`.
WORKSPACE_DIR = Path("workspace").resolve()
WORKSPACE_DATA_DIR = WORKSPACE_DIR / "data"
WORKSPACE_DATA_DIR.mkdir(parents=True, exist_ok=True)
WORKSPACE_DATASET_PATH = WORKSPACE_DATA_DIR / "T1_slice.csv"
if not WORKSPACE_DATASET_PATH.exists():
    shutil.copyfile(str(DATASET_PATH), str(WORKSPACE_DATASET_PATH))
df.head(5)


# %%
# This cell will:
# - Build compact, JSON-serializable metadata and sample rows to pass into prompts.
# build_dataset_meta is defined in langchain.API_utils.
DATASET_META = ut.build_dataset_meta(df)
DATASET_META


# %% [markdown]
# ## LCEL: prompt | model | parser
#
# LCEL (LangChain Expression Language) is a **pipe** syntax for composing steps.
# If you’ve used Unix pipes (`a | b | c`), it’s the same vibe:
#
# - build a prompt
# - call a model
# - parse the result
#
# Key pieces in this notebook:
# - `ChatPromptTemplate` (how we structure instructions + user input)
# - `StrOutputParser` (turn a chat message into a plain string)
# - composition with `|` (build a reusable “chain”)
#
# As you run the next cell, focus on the *shape* of the pipeline more than the exact prompt wording.
#

# %%
# This cell will:
# - Demonstrate a small part of the API surface used in the examples notebook.
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a concise tutor. Answer clearly."),
        ("human", "{question}"),
    ]
)

chain = prompt | llm | StrOutputParser()
chain.invoke({"question": "Explain LCEL in one sentence."})


# %% [markdown]
# ## Runnables: invoke / batch / stream / RunnableParallel
#
# A “runnable” is anything you can **call**.
# LangChain standardizes that with a few common methods:
#
# - `.invoke(input)` → one input, one output
# - `.batch([inputs])` → many inputs at once (often more efficient)
# - `.stream(input)` → yield partial outputs as they arrive
# - `RunnableParallel(...)` → run independent chains side-by-side and combine the results
#
# When you’re learning, it helps to treat runnables like functions — except they can be composed and configured.
#

# %%
# This cell will:
# - Demonstrate a small part of the API surface used in the examples notebook.
from langchain_core.runnables import RunnableParallel

summary_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You write crisp summaries."),
        ("human", "Summarize in 3 bullets:\n\n{text}"),
    ]
)
risks_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You list caveats."),
        ("human", "List 3 risks/caveats:\n\n{text}"),
    ]
)

summary_chain = summary_prompt | llm | StrOutputParser()
risks_chain = risks_prompt | llm | StrOutputParser()

parallel = RunnableParallel(summary=summary_chain, risks=risks_chain)
parallel.invoke(
    {"text": "LangChain provides composable building blocks for LLM apps."},
    config={"max_concurrency": 2},
)


# %%
# This cell will:
# - Use `ToolNode` to execute tool calls inside a graph.
questions = [
    {"question": "What is a tool in LangChain?"},
    {"question": "What is ToolNode in LangGraph?"},
    {"question": "What does InjectedState do?"},
]
chain.batch(questions, return_exceptions=True, config={"max_concurrency": 3})


# %%
# This cell will:
# - Demonstrate a small part of the API surface used in the examples notebook.
chunks = []
for chunk in chain.stream(
    {"question": "Give me a 2-bullet explanation of RunnableParallel."}
):
    chunks.append(chunk)
final = "".join(chunks)
final[:300] + ("..." if len(final) > 300 else "")


# %% [markdown]
# ## Tools: `@tool` + ToolNode execution
#
# A *tool* is a normal Python function with a schema.
# The LLM can “ask” to call a tool (with arguments), and your code executes it.
#
# Two ways you’ll see tools used:
#
# 1) **Directly** (call the function yourself)
# 2) **Inside a graph** via `ToolNode` (LangGraph executes any requested tool calls and feeds results back)
#
# If you’re new: don’t worry about the message formats yet. Focus on the story:
# "model asks for tool" → "we run tool" → "tool returns data" → "model continues".
#

# %%
# This cell will:
# - Build and compile a `StateGraph` (a small LangGraph workflow).
# - Use `ToolNode` to execute tool calls inside a graph.
# mean, zscore, ToolState are defined in langchain.API_utils.
from langchain_core.messages import AIMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode

tool_node = ToolNode([ut.mean, ut.zscore])

g = StateGraph(ut.ToolState)
g.add_node("tools", tool_node)
g.add_edge(START, "tools")
g.add_edge("tools", END)
graph = g.compile()

tool_calls = [
    {
        "name": "mean",
        "args": {"xs": [1, 2, 3, 4]},
        "id": "t1",
        "type": "tool_call",
    },
    {
        "name": "zscore",
        "args": {"xs": [9, 10, 10], "x": 10},
        "id": "t2",
        "type": "tool_call",
    },  # error (std=0)
]

out = graph.invoke({"messages": [AIMessage(content="", tool_calls=tool_calls)]})
[
    type(m).__name__ + ":" + (getattr(m, "content", "")[:80])
    for m in out["messages"]
]


# %% [markdown]
# ## InjectedState: runtime-only args (system-owned)
#
# Sometimes a tool needs access to *system-owned* context that the model shouldn’t be allowed to spoof.
#
# `InjectedState` is the pattern for that:
# - your tool signature includes an injected parameter
# - LangGraph supplies it at runtime (not from the model’s JSON arguments)
#
# Think of it like dependency injection:
# - model controls: normal tool inputs
# - system controls: injected inputs (state, stores, call IDs)
#

# %%
# This cell will:
# - Build and compile a `StateGraph` (a small LangGraph workflow).
# - Use `ToolNode` to execute tool calls inside a graph.
# dataset_brief, InjectedStateState are defined in langchain.API_utils.
import json

from langchain_core.messages import AIMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode

tool_node = ToolNode([ut.dataset_brief])
g = StateGraph(ut.InjectedStateState)
g.add_node("tools", tool_node)
g.add_edge(START, "tools")
g.add_edge("tools", END)
graph = g.compile()

state_in: ut.InjectedStateState = {
    "dataset_meta": DATASET_META,
    "messages": [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "dataset_brief",
                    "args": {
                        "question": "What columns exist and what is the sampling frequency?"
                    },
                    "id": "t1",
                    "type": "tool_call",
                }
            ],
        )
    ],
}
out = graph.invoke(state_in)
json.loads(out["messages"][-1].content)


# %% [markdown]
# ## InjectedStore: injected persistent store handle
#
# A store is a place to keep small bits of information across calls (like preferences, cached results, or “facts we’ve already extracted”).
#
# `InjectedStore` lets a tool receive a store handle **without** the model being able to fabricate it.
#
# In this tutorial we use `InMemoryStore` for simplicity, but the pattern generalizes to other persistence layers.
#

# %%
# This cell will:
# - Build and compile a `StateGraph` (a small LangGraph workflow).
# - Use `ToolNode` to execute tool calls inside a graph.
# save_pref, load_pref, StoreState are defined in langchain.API_utils.
from langchain_core.messages import AIMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
tool_node = ToolNode([ut.save_pref, ut.load_pref])
g = StateGraph(ut.StoreState)
g.add_node("tools", tool_node)
g.add_edge(START, "tools")
g.add_edge("tools", END)
graph = g.compile(store=store)

out1 = graph.invoke(
    {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "save_pref",
                        "args": {
                            "user_id": "u1",
                            "key": "freq_hint",
                            "value": "1min",
                        },
                        "id": "t1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    }
)
out2 = graph.invoke(
    {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "load_pref",
                        "args": {"user_id": "u1", "key": "freq_hint"},
                        "id": "t2",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    }
)
out1["messages"][-1].content, out2["messages"][-1].content


# %% [markdown]
# ## Agent APIs used in `langchain.example.ipynb`
#
# An *agent* is a loop: the model looks at the conversation + available tools, chooses an action, and repeats until it’s done.
#
# In this tutorial we use a helper, `create_agent(...)`, to build a tool-calling agent quickly.
# Later, in the examples notebook, you’ll see the same ideas expressed as explicit LangGraph loops.
#
# If you ever feel confused, this heuristic helps:
# - **LangChain agent helpers** get you started fast.
# - **LangGraph** is what you reach for when you want full control (state, routing, memory, HITL).
#

# %%
# This cell will:
# - Create a tool-calling agent using `create_agent(...)`.
# utc_now is defined in langchain.API_utils.
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

agent = create_agent(
    model=llm,
    tools=[ut.utc_now],
    system_prompt="Use tools when a tool can answer the question more reliably than guessing.",
)
out = agent.invoke(
    {
        "messages": [
            HumanMessage(content="Call utc_now and return the exact value.")
        ]
    }
)
[(type(m).__name__, getattr(m, "content", "")[:120]) for m in out["messages"]][
    -4:
]


# %% [markdown]
# ## Tool-calling output contract (reproducible handoff)
#
# A practical pattern from production-style graph tutorials: ask agents to return a **reproducible call snippet** after using tools.
#
# Why this helps:
# - humans can verify what happened
# - downstream automation can replay behavior
# - handoffs between teammates become less ambiguous
#

# %%
# This cell will:
# - Create a tool-calling agent with an explicit output contract.
# - Ask for UTC time and require a reproducible Python snippet in the final answer.
contract_agent = create_agent(
    model=llm,
    tools=[ut.utc_now],
    system_prompt=(
        "When time is requested, call utc_now. "
        "In your final answer, include a fenced python block with the exact tool call used."
    ),
)
contract_out = contract_agent.invoke(
    {
        "messages": [
            HumanMessage(content="What is the current UTC time? Use your tool.")
        ]
    }
)
print(getattr(contract_out["messages"][-1], "content", ""))


# %% [markdown]
# ## Advanced agent tool plumbing: `AgentState`, `ToolRuntime`, `InjectedToolCallId`
#
# This section is here for when you’re ready to peek “under the hood”.
#
# The high-level story:
# - tool calls happen inside a conversation
# - each tool call has an ID
# - LangGraph/LangChain pass runtime helpers so tools can update state and emit the right `ToolMessage`
#
# If it feels advanced on a first read, that’s normal — the goal is to make the concepts *available*, not to memorize them.
#

# %%
# This cell will:
# - Create a tool-calling agent using `create_agent(...)`.
# CustomState and extract_facts are built by make_custom_state_and_tool() in langchain.API_utils.
import json

from langchain.agents import create_agent

CustomState, extract_facts = ut.make_custom_state_and_tool()

supervisor = create_agent(
    llm,
    tools=[extract_facts],
    system_prompt="First call extract_facts, then summarize the returned facts.",
    state_schema=CustomState,
)

state = supervisor.invoke(
    {
        "messages": [
            {"role": "user", "content": "Text: LangGraph supports interrupts."}
        ],
        "user_prefs": {"tone": "formal"},
        "facts": [],
    }
)
{
    "facts": state.get("facts"),
    "last": getattr(state["messages"][-1], "content", "")[:160],
}


# %% [markdown]
# ## Human-in-the-loop building block: `interrupt(...)` + resume
#
# Sometimes an agent should *pause* and ask a human before doing something risky:
# - deleting a file
# - sending an email
# - running a trade
# - making an irreversible change
#
# LangGraph’s low-level building block for this is `interrupt(value)`:
#
# - The first time a node calls `interrupt(...)`, execution **stops** and the graph returns an `__interrupt__` payload.
# - To continue, you call the graph again with `Command(resume=...)`.
# - When the graph resumes, the node is **re-executed**, and `interrupt(...)` returns the human’s choice.
#
# In the next cell we create a tiny file in `tmp_runs/hitl/` and only delete it if the human approves.
#

# %%
# This cell will:
# - Build and compile a `StateGraph` (a small LangGraph workflow).
# - Demonstrate human-in-the-loop control using `interrupt(...)` and resume.
# HITLState, propose_delete, do_delete are defined in langchain.API_utils.
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

builder = StateGraph(ut.HITLState)
builder.add_node("propose", ut.propose_delete)
builder.add_node("delete", ut.do_delete)
builder.add_edge(START, "propose")
builder.add_edge("propose", "delete")
builder.add_edge("delete", END)
graph = builder.compile(checkpointer=MemorySaver())

tmp_dir = Path("tmp_runs/hitl").resolve()
tmp_dir.mkdir(parents=True, exist_ok=True)
victim = tmp_dir / "victim.txt"
victim.write_text("delete me", encoding="utf-8")

thread_id = "HITL_API_DEMO"
out1 = graph.invoke(
    {"target_path": str(victim), "decision": ""},
    config={"configurable": {"thread_id": thread_id}},
)
pending = (
    out1.get("__interrupt__", [])[0].value if "__interrupt__" in out1 else None
)
out2 = graph.invoke(
    Command(resume="approve"), config={"configurable": {"thread_id": thread_id}}
)
{"pending": pending, "victim_exists_after": victim.exists()}


# %% [markdown]
# ## Notebook ops: nbformat + nbclient + artifacts + papermill
#
# Notebooks are just JSON documents.
# That means you can:
# - generate them (`nbformat`)
# - execute them programmatically (`nbclient`)
# - collect outputs and errors
# - parameterize runs (`papermill`)
#
# Why include this in a LangChain/LangGraph tutorial?
# Because “agents that write and run notebooks” is a surprisingly practical workflow for data work.
# We’ll keep the demos safe: everything writes under `tmp_runs/`.
#

# %%
# This cell will:
# - Demonstrate notebook operations (write/execute/parameterize notebooks).
from pathlib import Path

import nbformat
from nbformat import validate
from nbclient import NotebookClient

run_dir = Path("tmp_runs").resolve()
run_dir.mkdir(parents=True, exist_ok=True)

nb = nbformat.v4.new_notebook()
nb.cells = [
    nbformat.v4.new_markdown_cell("# nbclient smoke test"),
    nbformat.v4.new_code_cell("x = 2 + 3\nprint(x)"),
    nbformat.v4.new_code_cell("import math\nprint(math.sqrt(81))"),
]
validate(nb)

in_path = run_dir / "smoke_in.ipynb"
out_path = run_dir / "smoke_out.ipynb"
nbformat.write(nb, str(in_path))

nb2 = nbformat.read(str(in_path), as_version=4)
client = NotebookClient(
    nb2, resources={"metadata": {"path": str(run_dir)}}, timeout=60
)
client.execute()
nbformat.write(nb2, str(out_path))

str(out_path)


# %% [markdown]
# ### Write a notebook via a tool (from a spec)
#
# We’ll build a tiny notebook in memory (a title + a code cell), then write it to disk.
#
# This is the first building block for “notebook automation” — generating a notebook artifact from a structured spec.
#

# %%
# This cell will:
# - Demonstrate notebook operations (write/execute/parameterize notebooks).
# write_notebook is defined in langchain.API_utils.
spec = {
    "cells": [
        {"type": "markdown", "source": "# Tool-written notebook"},
        {"type": "code", "source": "print('ok')"},
    ]
}
ut.write_notebook.invoke({"spec": spec, "out_rel": "demo/tool_hello.ipynb"})


# %% [markdown]
# ### Notebook ops as tools + secure injected workspace (ToolNode)
#
# Here we treat notebook operations as **tools** inside a LangGraph workflow.
#
# The important idea:
# - tools can be powerful (file access, execution)
# - so we often want a *controlled* workspace root
#
# You’ll see us use an injected workspace directory so the graph can safely read/write only where we intend.
#

# %%
# This cell will:
# - Build and compile a `StateGraph` (a small LangGraph workflow).
# - Use `ToolNode` to execute tool calls inside a graph.
# - Demonstrate notebook operations (write/execute/parameterize notebooks).
# nb_write, nb_run, nb_extract_errors, nb_extract_artifacts, nb_list_files, ToolGraphState
# are defined in langchain.API_utils.
from pathlib import Path

from langchain_core.messages import AIMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode

workspace = Path("tmp_runs/ipynb_tools_workspace").resolve()
workspace.mkdir(parents=True, exist_ok=True)

tool_node = ToolNode(
    [
        ut.nb_write,
        ut.nb_run,
        ut.nb_extract_errors,
        ut.nb_extract_artifacts,
        ut.nb_list_files,
    ]
)
g = StateGraph(ut.ToolGraphState)
g.add_node("tools", tool_node)
g.add_edge(START, "tools")
g.add_edge("tools", END)
graph = g.compile()

spec = {
    "cells": [
        {"type": "markdown", "source": "# Tool-made notebook"},
        {"type": "code", "source": "print('hello')"},
    ]
}

# IMPORTANT: Tool calls in a single ToolNode are not a dependency graph.
# Execute dependent operations in separate invocations for deterministic behavior.

out1 = graph.invoke(
    {
        "workspace_dir": str(workspace),
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "nb_write",
                        "args": {"spec": spec, "out_rel": "demo/in.ipynb"},
                        "id": "t1",
                        "type": "tool_call",
                    },
                ],
            )
        ],
    }
)

out2 = graph.invoke(
    {
        "workspace_dir": str(workspace),
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "nb_run",
                        "args": {
                            "in_rel": "demo/in.ipynb",
                            "out_rel": "demo/out.executed.ipynb",
                            "timeout_s": 60,
                        },
                        "id": "t2",
                        "type": "tool_call",
                    },
                ],
            )
        ],
    }
)

out3 = graph.invoke(
    {
        "workspace_dir": str(workspace),
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "nb_list_files",
                        "args": {},
                        "id": "t3",
                        "type": "tool_call",
                    },
                ],
            )
        ],
    }
)

(
    out1["messages"][-1].content,
    out2["messages"][-1].content,
    out3["messages"][-1].content[:200],
)


# %% [markdown]
# ### Execute notebooks + collect errors
#
# We’ll execute a notebook programmatically and capture:
# - stdout
# - execution errors (if any)
#
# This is a friendly way to build “run this notebook and report back” pipelines.
#

# %%
# This cell will:
# - Demonstrate notebook operations (write/execute/parameterize notebooks).
# extract_errors is defined in langchain.API_utils.
from pathlib import Path

import nbformat
from nbformat import validate
from nbclient import NotebookClient

run_dir = Path("tmp_runs/execute").resolve()
run_dir.mkdir(parents=True, exist_ok=True)

# Notebook that errors.
nb_err = nbformat.v4.new_notebook()
nb_err.cells = [
    nbformat.v4.new_markdown_cell("# Intentional error"),
    nbformat.v4.new_code_cell("print('before')"),
    nbformat.v4.new_code_cell("1/0"),
    nbformat.v4.new_code_cell("print('after')"),
]
validate(nb_err)
in_path = run_dir / "error_in.ipynb"
out_path = run_dir / "error_out.executed.ipynb"
nbformat.write(nb_err, str(in_path))

nb = nbformat.read(str(in_path), as_version=4)
client = NotebookClient(
    nb,
    timeout=60,
    allow_errors=True,
    resources={"metadata": {"path": str(run_dir)}},
)
client.execute()
nbformat.write(nb, str(out_path))

ut.extract_errors(nb)


# %% [markdown]
# ### Extract artifacts from executed notebooks (stdout + inline images)
#
# Executed notebooks can contain rich outputs (plots, tables, HTML).
#
# We’ll show a simple approach to pull a couple useful artifacts out of the executed notebook:
# - printed output
# - embedded images
#

# %%
# This cell will:
# - Demonstrate notebook operations (write/execute/parameterize notebooks).
import base64
import json

run_dir = Path("tmp_runs/artifacts").resolve()
run_dir.mkdir(parents=True, exist_ok=True)

nb = nbformat.v4.new_notebook()
nb.cells = [
    nbformat.v4.new_markdown_cell("# Artifact notebook"),
    nbformat.v4.new_code_cell("print('hello from stdout')"),
    nbformat.v4.new_code_cell(
        "import matplotlib.pyplot as plt\n"
        "plt.plot([0,1,2],[0,1,4])\n"
        "plt.title('inline')\n"
        "plt.show()\n"
    ),
]
in_nb = run_dir / "artifacts_in.ipynb"
executed_nb = run_dir / "artifacts.executed.ipynb"
nbformat.write(nb, str(in_nb))

nb2 = nbformat.read(str(in_nb), as_version=4)
NotebookClient(
    nb2, timeout=120, resources={"metadata": {"path": str(run_dir)}}
).execute()
nbformat.write(nb2, str(executed_nb))

out_dir = run_dir / "out"
out_dir.mkdir(parents=True, exist_ok=True)
manifest = []

for i, cell in enumerate(nb2.cells):
    if cell.get("cell_type") != "code":
        continue
    for j, out in enumerate(cell.get("outputs", [])):
        if out.get("output_type") == "stream":
            txt = out.get("text", "")
            p = out_dir / f"cell_{i}_stream_{j}.txt"
            p.write_text(txt if isinstance(txt, str) else "".join(txt))
            manifest.append({"cell": i, "kind": "stream", "path": str(p)})
        if out.get("output_type") in ("display_data", "execute_result"):
            data = out.get("data", {})
            if "text/plain" in data:
                t = data["text/plain"]
                p = out_dir / f"cell_{i}_text_{j}.txt"
                p.write_text(t if isinstance(t, str) else "".join(t))
                manifest.append(
                    {"cell": i, "kind": "text/plain", "path": str(p)}
                )
            if "image/png" in data:
                b64 = data["image/png"]
                b = base64.b64decode(
                    b64 if isinstance(b64, str) else "".join(b64)
                )
                p = out_dir / f"cell_{i}_img_{j}.png"
                p.write_bytes(b)
                manifest.append({"cell": i, "kind": "image/png", "path": str(p)})

(out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
{
    "executed_nb": str(executed_nb),
    "n_artifacts": len(manifest),
    "manifest": str(out_dir / "manifest.json"),
}


# %% [markdown]
# ### Filesystem artifacts (notebooks that write files)
#
# Sometimes notebooks produce *real files* (CSVs, images, model outputs).
#
# In the next cell we execute a notebook that writes files into a run directory, then list what it produced.
# Everything stays under `tmp_runs/`.
#

# %%
# This cell will:
# - Demonstrate notebook operations (write/execute/parameterize notebooks).

run_dir = Path("tmp_runs/writes_files").resolve()
run_dir.mkdir(parents=True, exist_ok=True)

nb = nbformat.v4.new_notebook()
nb.cells = [
    nbformat.v4.new_markdown_cell("# Writes files"),
    nbformat.v4.new_code_cell(
        "import csv\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "rows = [(i, i*i) for i in range(5)]\n"
        "with open('table.csv', 'w', newline='') as f:\n"
        "    w = csv.writer(f)\n"
        "    w.writerow(['x','y'])\n"
        "    w.writerows(rows)\n"
        "\n"
        "xs = [r[0] for r in rows]\n"
        "ys = [r[1] for r in rows]\n"
        "plt.plot(xs, ys)\n"
        "plt.title('y=x^2')\n"
        "plt.savefig('plot.png', dpi=120)\n"
        "print('wrote table.csv and plot.png')\n"
    ),
]

in_nb = run_dir / "writes_files.ipynb"
out_nb = run_dir / "writes_files.executed.ipynb"
nbformat.write(nb, str(in_nb))

nb2 = nbformat.read(str(in_nb), as_version=4)
NotebookClient(
    nb2, timeout=120, resources={"metadata": {"path": str(run_dir)}}
).execute()
nbformat.write(nb2, str(out_nb))

sorted([p.name for p in run_dir.iterdir() if p.is_file()])


# %% [markdown]
# ### Parameterized runs (Papermill)
#
# Papermill is a simple way to run the *same* notebook with different parameters.
#
# This is useful for:
# - experiments
# - scheduled reports
# - batch runs over multiple inputs
#
# We’ll do a tiny demo so you can see the mechanics.
#

# %%
# This cell will:
# - Demonstrate notebook operations (write/execute/parameterize notebooks).
import papermill as pm

run_dir = Path("tmp_runs/papermill").resolve()
run_dir.mkdir(parents=True, exist_ok=True)

nb = nbformat.v4.new_notebook()
nb.cells = [
    nbformat.v4.new_markdown_cell("# Papermill demo"),
    nbformat.v4.new_code_cell(
        "# Parameters\nx = 1\ny = 2", metadata={"tags": ["parameters"]}
    ),
    nbformat.v4.new_code_cell("print({'x': x, 'y': y, 'x_plus_y': x + y})"),
]
nb.metadata["kernelspec"] = {
    "name": "python3",
    "display_name": "Python 3",
    "language": "python",
}

in_nb = run_dir / "pm_in.ipynb"
out_nb = run_dir / "pm_out.ipynb"
nbformat.write(nb, str(in_nb))

pm.execute_notebook(
    str(in_nb),
    str(out_nb),
    parameters={"x": 10, "y": 32},
    cwd=str(run_dir),
    kernel_name="python3",
)
str(out_nb)


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
