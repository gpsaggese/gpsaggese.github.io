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
# ## Learn LangChain in 60 Minutes — examples notebook
#
# This is the **end-to-end** companion to `langchain.API.ipynb`.
#
# If you’re brand new, it can help to skim the API notebook first so the names feel familiar.
# Then come back here for the patterns that make things “click” in real apps.
#
# What you’ll build (incrementally):
# - a tool-calling agent loop
# - LangGraph workflows: state, routing, reducers, and a ReAct loop from scratch
# - subagents + subgraphs (composition)
# - memory boundaries via checkpointers
# - human-in-the-loop interrupts + resume
# - Deep Agents demos (todos/filesystem/subagents/HITL/sandboxing)
#
# Same note as the API notebook: some cells call an LLM (cost). It’s always okay to pause, read, and only run what you’re comfortable with.
#

# %% [markdown]
# # Imports
#
# This notebook shares the same setup pattern as `langchain.API.ipynb`.
#
# Run from `tutorials/LangChain_LangGraph` so local paths and helper utilities resolve exactly as written.
#

# %%
# %load_ext autoreload
# %autoreload 2

import os
import sys
import importlib


def _require_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise RuntimeError(
            f"""Missing Python package {module_name!r}.

This tutorial is meant to be run from `tutorials/LangChain_LangGraph` with its pinned dependencies.

Quick fixes:
- Docker (recommended): `cd tutorials/LangChain_LangGraph && docker compose up --build`
- Local venv: `cd tutorials/LangChain_LangGraph && pip install -r requirements.txt`
"""
        ) from e


langchain = _require_import("langchain")
langchain_core = _require_import("langchain_core")
langgraph = _require_import("langgraph")


# %%
# This cell will:
# - Configure logging and print environment/version info for debugging.
import logging
import platform

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)
_LOG = logging.getLogger("learn_langchain.examples")

_LOG.info("python=%s", sys.version.split()[0])
_LOG.info("platform=%s", platform.platform())
_LOG.info("langchain=%s", getattr(langchain, "__version__", "unknown"))
_LOG.info("langchain_core=%s", getattr(langchain_core, "__version__", "unknown"))
_LOG.info("langgraph=%s", getattr(langgraph, "__version__", "unknown"))
_LOG.info("LLM_PROVIDER=%s", os.getenv("LLM_PROVIDER", "(unset)"))


# %% [markdown]
# ## Deep Agents (E7: DA1 → DA8)
#
# Deep Agents is a compact “agent app” layer used in this tutorial.
# It wraps common patterns (todos, filesystem tools, delegation) so you can focus on *behavior* rather than plumbing.
#
# We’ll walk through DA1 → DA8 in small steps:
# - DA1: Hello deep agent
# - DA2: planning tool (`write_todos`)
# - DA3: filesystem surface (`write_file` / `read_file`)
# - DA4: backends matrix (State vs Filesystem vs Store)
# - DA5: dict-based subagents (delegate via `task`)
# - DA6: `CompiledSubAgent` (delegate to a compiled runnable)
# - DA7: HITL gates (`interrupt_on` + `Command(resume=...)`)
# - DA8: sandboxing (`FilesystemBackend(virtual_mode=True)`)
#
# Two safety notes:
# - filesystem demos write under `workspace/` and `tmp_runs/` in this folder
# - HITL gates are what prevent “silent” edits when you want a human to approve
#

# %%
# This cell will:
# - Run a Deep Agents demo (filesystem/todos/subagents/HITL).
try:
    import deepagents  # type: ignore
    from deepagents import CompiledSubAgent, create_deep_agent  # type: ignore
    from deepagents.backends import (  # type: ignore
        CompositeBackend,
        FilesystemBackend,
        StateBackend,
        StoreBackend,
    )

    print("deepagents:", getattr(deepagents, "__version__", "(unknown)"))
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Deep Agents section requires `deepagents`.\n"
        "Run this notebook via the provided Docker setup or install tutorial deps.\n"
        f"Import error: {type(e).__name__}: {str(e)[:200]}"
    )


# %%
# This cell will:
# - Run the next step of the end-to-end example.
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage


def _all_tool_calls(messages: list[Any]) -> list[dict[str, Any]]:
    """
    Collect tool call dicts emitted by `AIMessage` objects in `messages`.
    """
    calls: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                if isinstance(tc, dict):
                    calls.append(tc)
    return calls


def _tool_outputs(messages: list[Any], tool_name: str) -> list[Any]:
    """
    Return tool message contents for `tool_name`.
    """
    outs: list[Any] = []
    for m in messages:
        if isinstance(m, ToolMessage) and getattr(m, "name", None) == tool_name:
            outs.append(m.content)
    return outs


def _as_text(x: Any) -> str:
    """
    Convert `x` to a readable string for printing.
    """
    if isinstance(x, str):
        return x
    return (
        json.dumps(x, indent=2, ensure_ascii=False)
        if isinstance(x, (dict, list))
        else str(x)
    )


def _read_file_text(read_result: Any) -> str:
    """
    Extract plain text from a `read_file` tool output.
    """
    if read_result is None:
        return ""
    if isinstance(read_result, str):
        return read_result
    if isinstance(read_result, list):
        parts = [str(x) for x in read_result]
        return "\n".join(parts)
    if isinstance(read_result, dict):
        content = read_result.get("content")
        if isinstance(content, list):
            return "\n".join([str(x) for x in content])
        if isinstance(content, str):
            return content
        text = read_result.get("text")
        if isinstance(text, str):
            return text
    return _as_text(read_result)


def _extract_bullets(text: str) -> list[str]:
    """
    Extract Markdown-style bullet lines from `text`.
    """
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("- ") or s.startswith("* "):
            lines.append("- " + s[2:].strip())
            continue
        if s.startswith("•"):
            lines.append("- " + s.lstrip("•").strip())
            continue
        if len(s) >= 3 and s[0].isdigit() and s[1] in (".", ")"):
            lines.append("- " + s[2:].strip())
            continue
        if (
            len(s) >= 4
            and s[0].isdigit()
            and s[1].isdigit()
            and s[2] in (".", ")")
        ):
            lines.append("- " + s[3:].strip())
            continue
    return lines


def build_dataset_context() -> str:
    """
    Build a compact dataset context string for prompts, if `DATASET_META` exists.
    """
    meta = globals().get("DATASET_META", None)
    if not isinstance(meta, dict):
        return "Dataset context: (not loaded)."
    cols = meta.get("columns", [])
    sample_rows = meta.get("sample_rows", [])
    time_col = meta.get("time_col", None)
    freq = meta.get("freq", None)
    parts = [
        f"path={meta.get('path')}",
        f"tool_path={meta.get('tool_path')}",
        f"n_rows={meta.get('n_rows')}",
        f"n_cols={meta.get('n_cols')}",
        f"time_col={time_col}",
        f"freq={freq}",
        f"columns={cols}",
        f"sample_rows={sample_rows}",
    ]
    return "Dataset context:\\n" + "\\n".join(parts)


DATASET_CONTEXT = build_dataset_context()


# %% [markdown]
# ### DA1 — Hello, deep agent
#
# Goal: create the smallest possible Deep Agent and ask it a simple question.
#
# You’re looking for:
# - a normal chat response
# - plus evidence that the agent is *tool-capable* (even if it doesn’t call a tool yet)
#

# %%
# This cell will:
# - Run a Deep Agents demo (filesystem/todos/subagents/HITL).
llm_da = get_chat_model()
agent = create_deep_agent(model=llm_da)
out = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "In 3 bullets, explain what a deep agent is useful for.",
            }
        ]
    }
)
print("state keys:", sorted(list(out.keys()))[:20])
print(
    "final message preview:", getattr(out["messages"][-1], "content", "")[:240]
)


# %% [markdown]
# ### DA2 — Planning tool: `write_todos`
#
# Goal: show that the agent can produce a structured plan.
#
# This is a gentle entry point into “agentic” behavior:
# - break a task into steps
# - keep the steps visible
# - iterate as you go
#

# %%
# This cell will:
# - Run a Deep Agents demo (filesystem/todos/subagents/HITL).
agent = create_deep_agent(model=get_chat_model())
prompt = (
    "Before doing anything else, call write_todos with 5 EDA tasks for THIS dataset. "
    "Mark the first task as in_progress.\n\n"
    f"{DATASET_CONTEXT}\n\nIf you need raw rows, use read_file {DATASET_META.get('tool_path')}."
)
out = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
todos = out.get("todos", None)
print("todos channel present:", todos is not None)
print(_as_text(todos)[:800])


# %% [markdown]
# ### DA3 — Filesystem surface: `write_file` / `read_file`
#
# Goal: let the agent create and read files in the sandboxed workspace.
#
# We’ll keep everything under `/workspace/...` so it maps to `./workspace/...` on disk.
# That way you can inspect files with your own eyes.
#

# %%
# This cell will:
# - Run a Deep Agents demo (filesystem/todos/subagents/HITL).
root = Path(".").resolve()
Path("workspace").mkdir(parents=True, exist_ok=True)
agent = create_deep_agent(
    model=get_chat_model(),
    backend=FilesystemBackend(root_dir=str(root), virtual_mode=True),
)
prompt = (
    "Use filesystem tools:\n"
    "1) write_file /workspace/notes.md with EXACTLY 6 lines, each starting with '- ', of EDA checks for THIS dataset\n"
    "2) read_file /workspace/notes.md\n"
    "3) Then respond."
    "\n\n"
    f"{DATASET_CONTEXT}\n\nIf you need raw rows, use read_file {DATASET_META.get('tool_path')}."
)
out = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
reads = _tool_outputs(out["messages"], "read_file")
print("n read_file tool outputs:", len(reads))
paths = sorted([str(p) for p in Path("workspace").rglob("notes.md")])
print("notes.md paths on disk:", paths)

if paths:
    read_txt = Path(paths[0]).read_text(encoding="utf-8")
else:
    read_txt = (
        _read_file_text(reads[-1])
        if reads
        else _as_text(getattr(out["messages"][-1], "content", "(missing)"))
    )

bullets = _extract_bullets(read_txt)
print("two bullets:", bullets[:2])
if len(bullets) < 2:
    print("raw preview:", read_txt[:400])


# %% [markdown]
# ### DA4 — Backends matrix: State vs Filesystem vs Store
#
# Goal: see how different kinds of “memory” plug in.
#
# - **State**: ephemeral, per-run updates
# - **Filesystem**: durable artifacts (notes, reports, code)
# - **Store**: structured persistence (facts, preferences)
#
# In real projects you usually use a mix.
#

# %%
# This cell will:
# - Run a Deep Agents demo (filesystem/todos/subagents/HITL).
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

ckpt = MemorySaver()

NOTE_STATE = "/workspace/notes.md"
NOTE_FS = "/workspace/notes.md"
NOTE_STORE = "/memories/notes.md"


def run(agent, thread_id: str, user_msg: str):
    """
    Invoke `agent` with `thread_id` for persistence/scoping tests.
    """
    return agent.invoke(
        {"messages": [{"role": "user", "content": user_msg}]},
        config={"configurable": {"thread_id": thread_id}},
    )


# 1) State backend (thread-scoped)
agent_state = create_deep_agent(model=get_chat_model(), checkpointer=ckpt)
run(
    agent_state,
    "STATE_A",
    f"write_file {NOTE_STATE} with 'hello from STATE thread A'",
)
outA = run(agent_state, "STATE_A", f"read_file {NOTE_STATE}")
outB = run(
    agent_state, "STATE_B", f"read_file {NOTE_STATE} (if missing, say so)"
)
state_reads_a = _tool_outputs(outA["messages"], "read_file")
state_reads_b = _tool_outputs(outB["messages"], "read_file")
print(
    "StateBackend thread A read:",
    _as_text(state_reads_a[-1])[:120] if state_reads_a else "(no tool output)",
)
print(
    "StateBackend thread B read:",
    _as_text(state_reads_b[-1])[:120] if state_reads_b else "(no tool output)",
)

# 2) Filesystem backend (disk, cross-thread)
root = Path("tmp_runs/deepagents/fs_root").resolve()
root.mkdir(parents=True, exist_ok=True)
agent_fs = create_deep_agent(
    model=get_chat_model(),
    checkpointer=ckpt,
    backend=FilesystemBackend(root_dir=str(root), virtual_mode=True),
)
run(agent_fs, "FS_A", f"write_file {NOTE_FS} with 'hello from FS thread A'")
outA = run(agent_fs, "FS_A", f"read_file {NOTE_FS}")
outB = run(agent_fs, "FS_B", f"read_file {NOTE_FS}")
fs_reads_a = _tool_outputs(outA["messages"], "read_file")
fs_reads_b = _tool_outputs(outB["messages"], "read_file")
print(
    "FilesystemBackend thread A read:",
    _as_text(fs_reads_a[-1])[:120] if fs_reads_a else "(no tool output)",
)
print(
    "FilesystemBackend thread B read:",
    _as_text(fs_reads_b[-1])[:120] if fs_reads_b else "(no tool output)",
)
print("fs root_dir:", root)

# 3) Store backend via CompositeBackend (cross-thread under /memories/)
store = InMemoryStore()
composite_backend = lambda rt: CompositeBackend(
    default=StateBackend(rt),
    routes={"/memories/": StoreBackend(rt)},
)
agent_store = create_deep_agent(
    model=get_chat_model(),
    checkpointer=ckpt,
    backend=composite_backend,
    store=store,
)
run(
    agent_store,
    "STORE_A",
    f"write_file {NOTE_STORE} with 'hello from STORE thread A'",
)
outA = run(agent_store, "STORE_A", f"read_file {NOTE_STORE}")
outB = run(agent_store, "STORE_B", f"read_file {NOTE_STORE}")
store_reads_a = _tool_outputs(outA["messages"], "read_file")
store_reads_b = _tool_outputs(outB["messages"], "read_file")
print(
    "StoreBackend thread A read:",
    _as_text(store_reads_a[-1])[:120] if store_reads_a else "(no tool output)",
)
print(
    "StoreBackend thread B read:",
    _as_text(store_reads_b[-1])[:120] if store_reads_b else "(no tool output)",
)


# %% [markdown]
# ### DA5 — Dict subagents: delegate via `task`
#
# Goal: delegate to specialized subagents.
#
# Think of subagents as teammates:
# - each gets a clear job
# - each can have its own tools or style
# - the main agent stays focused on coordination
#

# %%
# This cell will:
# - Run a Deep Agents demo (filesystem/todos/subagents/HITL).
from langgraph.checkpoint.memory import MemorySaver

ckpt = MemorySaver()
profiler_subagent = {
    "name": "profile-agent",
    "description": "Produces an executive EDA profile: summary + next analyses.",
    "system_prompt": (
        "You are an EDA profiling specialist.\n"
        "Use only the dataset context provided in the task message.\n"
        "Do not claim to have read files unless the message includes their contents.\n"
        "Output:\n"
        "- 2-bullet executive summary\n"
        "- 2 next analyses\n"
        "Keep it concise."
    ),
    "tools": [],
}
root = Path(".").resolve()
Path("workspace").mkdir(parents=True, exist_ok=True)
dataset_on_disk = Path("workspace/data/T1_slice.csv")
print(
    "dataset exists on disk:", dataset_on_disk.exists(), "path:", dataset_on_disk
)

agent = create_deep_agent(
    model=get_chat_model(),
    checkpointer=ckpt,
    backend=FilesystemBackend(root_dir=str(root), virtual_mode=True),
    subagents=[profiler_subagent],  # type: ignore[list-item]
    name="main-agent",
)
prompt = (
    "Delegate to the profile-agent using task(), then present the final result.\n"
    "Important: delegate / subagent."
    "\n\n"
    f"{DATASET_CONTEXT}\n\nIf you need raw rows, use read_file {DATASET_META.get('tool_path')}."
)
out = agent.invoke(
    {"messages": [{"role": "user", "content": prompt}]},
    config={"configurable": {"thread_id": "DA5"}},
)
calls = _all_tool_calls(out["messages"])
print("tool calls:", [c.get("name") for c in calls])
print(
    "final message preview:", getattr(out["messages"][-1], "content", "")[:240]
)


# %% [markdown]
# ### DA6 — `CompiledSubAgent`: delegate to a compiled runnable
#
# Goal: treat a LangGraph workflow as a callable “subagent.”
#
# This is the bridge between the two worlds:
# - LangGraph gives you structured control
# - Deep Agents gives you ergonomic delegation
#
# You’ll see how a compiled graph can be slotted in as a worker.
#

# %%
# This cell will:
# - Create a tool-calling agent using `create_agent(...)`.
# - Run a Deep Agents demo (filesystem/todos/subagents/HITL).
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

ckpt = MemorySaver()
specialized_prompt = (
    "You are a strict hypothesis generator for EDA.\n"
    "Use only the dataset context provided in the task message.\n"
    "Do not claim to have read files unless the message includes their contents.\n"
    "Return:\n"
    "- 2 plausible hypotheses (bullets)\n"
    "- For each: 1 concrete test/plot\n"
    "Be concise."
)
compiled_worker = create_agent(
    model=get_chat_model(), tools=[], system_prompt=specialized_prompt
)

hypothesis_agent = CompiledSubAgent(
    name="hypothesis-agent",
    description="Generates hypotheses and concrete tests/plots for EDA.",
    runnable=compiled_worker,
)
root = Path(".").resolve()
Path("workspace").mkdir(parents=True, exist_ok=True)
dataset_on_disk = Path("workspace/data/T1_slice.csv")
print(
    "dataset exists on disk:", dataset_on_disk.exists(), "path:", dataset_on_disk
)

agent = create_deep_agent(
    model=get_chat_model(),
    checkpointer=ckpt,
    backend=FilesystemBackend(root_dir=str(root), virtual_mode=True),
    subagents=[hypothesis_agent],
    name="main-agent",
)
prompt = (
    "Delegate to the hypothesis-agent using task(), then summarize results.\n"
    "Important: delegate / hypothesis."
    "\n\n"
    f"{DATASET_CONTEXT}\n\nIf you need raw rows, use read_file {DATASET_META.get('tool_path')}."
)
out = agent.invoke(
    {"messages": [{"role": "user", "content": prompt}]},
    config={"configurable": {"thread_id": "DA6"}},
)
calls = _all_tool_calls(out["messages"])
print("tool calls:", [c.get("name") for c in calls])
print(
    "final message preview:", getattr(out["messages"][-1], "content", "")[:240]
)


# %% [markdown]
# ### DA7 — HITL gates: `interrupt_on` + `Command(resume=...)`
#
# This is the “seatbelt” moment.
#
# We configure the agent so that certain tools (like editing files) *cannot complete* without a human decision.
# Instead of silently writing, the agent run produces an interrupt payload that you approve/reject.
#
# If you’re building anything that touches real systems, this pattern is worth memorizing.
#

# %%
# This cell will:
# - Run a Deep Agents demo (filesystem/todos/subagents/HITL).
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

ckpt = MemorySaver()
thread_id = "DA7_NOTEBOOK"

root = Path(".").resolve()
Path("workspace").mkdir(parents=True, exist_ok=True)

agent = create_deep_agent(
    model=get_chat_model(),
    checkpointer=ckpt,
    backend=FilesystemBackend(root_dir=str(root), virtual_mode=True),
    interrupt_on={
        "edit_file": InterruptOnConfig(allowed_decisions=["approve", "reject"])
    },
)

# Step 1: create a file (no interrupt)
agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "write_file /workspace/notes.md with 'line1\\nline2\\n'",
            }
        ]
    },
    config={"configurable": {"thread_id": thread_id}},
)

# Step 2: attempt an edit (expect interrupt), then resume programmatically
out = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "edit_file /workspace/notes.md replace 'line2' with 'LINE2_EDITED' then read_file /workspace/notes.md",
            }
        ]
    },
    config={"configurable": {"thread_id": thread_id}},
)
has_intr = "__interrupt__" in out
print("interrupted:", has_intr)
if has_intr:
    intr0 = out["__interrupt__"][0]
    print(
        "interrupt payload preview:",
        _as_text(getattr(intr0, "value", intr0))[:240],
    )
    out = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config={"configurable": {"thread_id": thread_id}},
    )

# Confirm edited content via a fresh read
out2 = agent.invoke(
    {"messages": [{"role": "user", "content": "read_file /workspace/notes.md"}]},
    config={"configurable": {"thread_id": thread_id}},
)
read_outs = _tool_outputs(out2["messages"], "read_file")
paths = sorted([str(p) for p in Path("workspace").rglob("notes.md")])
print("notes.md paths on disk:", paths)

if paths:
    read_txt = Path(paths[0]).read_text(encoding="utf-8")
else:
    read_txt = (
        _read_file_text(read_outs[-1])
        if read_outs
        else _as_text(getattr(out2["messages"][-1], "content", ""))
    )

print(read_txt.replace("\\n", "\\\\n")[:200])


# %% [markdown]
# ### DA8 — Sandboxing: `FilesystemBackend(virtual_mode=True)`
#
# Sandboxing is about containment.
#
# With `virtual_mode=True`, the agent’s filesystem tools operate in a *virtual* root (like `/workspace/...`).
# That virtual path maps to a real directory you can inspect (`./workspace/...`).
#
# In practice, this gives you two benefits:
# - you can safely let an agent work with files without giving it your whole machine
# - you can review exactly what it wrote
#

# %%
# This cell will:
# - Run a Deep Agents demo (filesystem/todos/subagents/HITL).
from langgraph.checkpoint.memory import MemorySaver

ckpt = MemorySaver()
thread_id = "DA8_NOTEBOOK"
root = Path("tmp_runs/deepagents/sandbox_root").resolve()
root.mkdir(parents=True, exist_ok=True)

# Create a secret outside the sandbox root_dir (in the notebook workdir).
outside_secret = Path("tmp_runs/deepagents/secret_outside_sandbox.txt")
outside_secret.parent.mkdir(parents=True, exist_ok=True)
outside_secret.write_text("SUPER_SECRET=do_not_leak\\n", encoding="utf-8")

backend = FilesystemBackend(root_dir=str(root), virtual_mode=True)
agent = create_deep_agent(
    model=get_chat_model(), checkpointer=ckpt, backend=backend
)


# Note: filesystem tool calls are chosen by the model. For debugging, print the actual
# tool call args the agent produced, since it may choose a different path than requested.
def _print_tool_calls(state: dict, label: str) -> None:
    """
    Print tool call names and args emitted by the model.
    """
    calls = _all_tool_calls(state.get("messages", []))
    simplified = [{"name": c.get("name"), "args": c.get("args")} for c in calls]
    print(f"{label} tool_calls:", simplified)


# Safe file inside the sandbox.
out_ok = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Call write_file on /workspace/ok.txt with content 'safe'. Then call read_file on /workspace/ok.txt. Do not read any other paths.",
            }
        ]
    },
    config={"configurable": {"thread_id": thread_id}},
)
ok_reads = _tool_outputs(out_ok["messages"], "read_file")
ok_txt = (
    _read_file_text(ok_reads[-1])
    if ok_reads
    else _as_text(getattr(out_ok["messages"][-1], "content", ""))
)
print("ok read preview:", ok_txt[:80])
_print_tool_calls(out_ok, "ok")

# Attempt path traversal escape.
out_env = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Call read_file on /workspace/../tmp_runs/deepagents/secret_outside_sandbox.txt exactly. Return the raw result (error or content). Do not substitute any other path.",
            }
        ]
    },
    config={"configurable": {"thread_id": thread_id}},
)
escape_reads = _tool_outputs(out_env["messages"], "read_file")
print(
    "escape attempt output preview:",
    _as_text(escape_reads[-1])[:160] if escape_reads else "(no tool output)",
)
_print_tool_calls(out_env, "escape")

# Attempt to read a host path.
out_hosts = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Call read_file on /etc/hosts exactly. Return the raw result (error or content). Do not substitute any other path.",
            }
        ]
    },
    config={"configurable": {"thread_id": thread_id}},
)
hosts_reads = _tool_outputs(out_hosts["messages"], "read_file")
print(
    "/etc/hosts attempt output preview:",
    _as_text(hosts_reads[-1])[:160] if hosts_reads else "(no tool output)",
)
_print_tool_calls(out_hosts, "hosts")

print("sandbox root_dir:", root)
print(
    "sandbox files:",
    sorted([str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()])[
        :20
    ],
)

