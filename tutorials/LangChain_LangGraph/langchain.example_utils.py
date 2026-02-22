"""
Utility library for langchain.example.ipynb.

All reusable functions and classes extracted from the notebook live here,
organised in the same order as the notebook sections they belong to.
Shared helpers (LlmConfig, get_chat_model, build_dataset_meta, HITLState, …)
are loaded from langchain.API_utils and re-exported so callers only need to

Import as:

import tutorials.LangChain_LangGraph.langchain.example_utils as tlllexut
"""

import importlib
import importlib.util
import json
import logging
import math
import pathlib
from typing import Annotated, Any, List, TypedDict

from langchain_core.tools import tool
from langgraph.graph.message import add_messages

_LOG = logging.getLogger(__name__)

# ##############################################################################
# Bootstrap: load shared helpers from langchain.API_utils
# ##############################################################################

_api_utils_path = (
    pathlib.Path(__file__).resolve().parent / "langchain.API_utils.py"
)
_spec = importlib.util.spec_from_file_location(
    "_langchain_api_utils", str(_api_utils_path)
)
_api = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_api)

# Re-export shared helpers so callers only need to import this module.
LlmConfig = _api.LlmConfig
get_chat_model = _api.get_chat_model
build_dataset_meta = _api.build_dataset_meta
HITLState = _api.HITLState
propose_delete = _api.propose_delete
do_delete = _api.do_delete
mean = _api.mean
utc_now = _api.utc_now


# ##############################################################################
# Imports / Setup
# ##############################################################################


def _require_import(module_name: str):
    """
    Import `module_name` or raise a user-friendly RuntimeError.
    """
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


# ##############################################################################
# Basic tools (math + time)
# ##############################################################################


@tool
def sqrt(x: float) -> float:
    """
    Return sqrt(x); raises ValueError for negative input.
    """
    x = float(x)
    if x < 0:
        raise ValueError("x must be >= 0")
    return math.sqrt(x)


# ##############################################################################
# LangGraph: reducers (accumulate evidence)
# ##############################################################################


def add_list(old: List[str], new: List[str]) -> List[str]:
    """
    Reducer that concatenates two evidence lists.
    """
    return old + new


# #############################################################################
# ReducerState
# #############################################################################


class ReducerState(TypedDict):
    evidence: Annotated[List[str], add_list]


def make_dataset_nodes(df):
    """
    Return (find_missingness, find_outliers) graph node functions bound to
    `df`.
    """

    def find_missingness(_: ReducerState) -> dict:
        """
        Compute missingness findings from `df`.
        """
        miss = (df.isna().mean() * 100).sort_values(ascending=False)
        top = miss.head(3)
        evidence = [
            f"missingness: {idx} has {val:.2f}% missing"
            for idx, val in top.items()
        ]
        return {"evidence": evidence}

    def find_outliers(_: ReducerState) -> dict:
        """
        Compute a simple outlier finding using z-scores on one numeric column
        of `df`.
        """
        numeric_cols = [c for c in df.columns if c != "Date/Time"]
        col = None
        if "Wind Speed (m/s)" in df.columns:
            col = "Wind Speed (m/s)"
        elif numeric_cols:
            col = numeric_cols[0]
        if not col:
            return {"evidence": ["outliers: no numeric columns found"]}
        s = df[col].astype(float)
        mu = float(s.mean())
        sigma = float(s.std(ddof=0))
        if sigma == 0.0:
            return {
                "evidence": [
                    f"outliers: std({col}) is 0, cannot compute z-scores"
                ]
            }
        z = ((s - mu) / sigma).abs()
        idx = int(z.idxmax())
        ts = None
        if "Date/Time" in df.columns:
            ts = df.loc[idx, "Date/Time"]
        evidence = [
            f"outliers: max |z| for {col} at row={idx} ts={ts}"
            f" value={s.loc[idx]:.3f} z={z.loc[idx]:.2f}"
        ]
        return {"evidence": evidence}

    return find_missingness, find_outliers


# ##############################################################################
# ReAct loop: model node + ToolNode
# ##############################################################################


# #############################################################################
# RS
# #############################################################################


class RS(TypedDict):
    messages: Annotated[list, add_messages]


def make_call_model(llm, tools):
    """
    Return a graph node that calls `llm` bound to `tools`.
    """

    def call_model(state: RS) -> dict:
        """
        Call the model with bound tools and append the AI message.
        """
        bound = llm.bind_tools(tools)
        ai = bound.invoke(state["messages"])
        return {"messages": [ai]}

    return call_model


def needs_tools(state: RS) -> str:
    """
    Route to tools if the last AI message contains tool calls.
    """
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "end"


# ##############################################################################
# Subagents utilities
# ##############################################################################


def _last_text(result: dict) -> str:
    """
    Return the final message text/content from an agent result state.
    """
    msg = result["messages"][-1]
    return (
        getattr(msg, "text", None) or getattr(msg, "content", None) or str(msg)
    )


# ##############################################################################
# Subgraphs (graph-as-node composition)
# ##############################################################################


# #############################################################################
# SubState
# #############################################################################


class SubState(TypedDict):
    raw: str
    parsed: dict
    formatted: str


def parse_node(state: SubState) -> dict:
    """
    Parse `key: value` lines from `state['raw']` into a dict.
    """
    raw = state["raw"]
    parsed = {}
    for line in raw.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            parsed[k.strip()] = v.strip()
    return {"parsed": parsed}


def format_node(state: SubState) -> dict:
    """
    Format the parsed fields as a bullet list.
    """
    parsed = state.get("parsed", {})
    lines = [f"- {k}: {v}" for k, v in parsed.items()]
    return {"formatted": "Parsed fields:\n" + "\n".join(lines)}


# #############################################################################
# ParentState
# #############################################################################


class ParentState(TypedDict):
    user_text: str
    result: str


def make_call_subgraph(subgraph):
    """
    Return a graph node function that invokes `subgraph` and projects its
    output.
    """

    def call_subgraph(state: ParentState) -> dict:
        """
        Call `subgraph` and project its formatted output into the parent state.
        """
        out = subgraph.invoke({"raw": state["user_text"]})
        return {"result": out["formatted"]}

    return call_subgraph


# ##############################################################################
# Shared vs private memory boundaries (checkpointers)
# ##############################################################################


# #############################################################################
# CSub
# #############################################################################


class CSub(TypedDict):
    n: int


def bump(state: CSub) -> dict:
    """
    Increment a counter used to demonstrate subgraph memory behavior.
    """
    return {"n": state.get("n", 0) + 1}


# #############################################################################
# P
# #############################################################################


class P(TypedDict):
    mode: str
    sub_n: int


def make_call_sub(sub_shared, sub_private):
    """
    Return a graph node that calls the shared or private subgraph.
    """

    def call_sub(state: P) -> dict:
        """
        Call the shared or private subgraph depending on `state['mode']`.
        """
        if state["mode"] == "shared":
            out = sub_shared.invoke({"n": state.get("sub_n", 0)})
            return {"sub_n": out["n"]}
        out = sub_private.invoke(
            {"n": 0},
            config={"configurable": {"thread_id": "SUBGRAPH_THREAD"}},
        )
        return {"sub_n": out["n"]}

    return call_sub


def make_run_twice(parent):
    """
    Return a function that invokes `parent` graph twice and returns both
    sub-counters.
    """

    def run_twice(mode: str):
        """
        Invoke the parent graph twice and return the two observed sub-counters.
        """
        out1 = parent.invoke(
            {"mode": mode, "sub_n": 0},
            config={"configurable": {"thread_id": f"PARENT_{mode}"}},
        )
        out2 = parent.invoke(
            {"mode": mode, "sub_n": out1["sub_n"]},
            config={"configurable": {"thread_id": f"PARENT_{mode}"}},
        )
        return out1["sub_n"], out2["sub_n"]

    return run_twice


# ##############################################################################
# Deep Agents (DA): helper utilities
# ##############################################################################


def _all_tool_calls(messages: list[Any]) -> list[dict[str, Any]]:
    """
    Collect tool call dicts emitted by `AIMessage` objects in `messages`.
    """
    from langchain_core.messages import AIMessage

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
    from langchain_core.messages import ToolMessage

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
        return "\n".join(str(x) for x in read_result)
    if isinstance(read_result, dict):
        content = read_result.get("content")
        if isinstance(content, list):
            return "\n".join(str(x) for x in content)
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


def build_dataset_context(meta: dict | None = None) -> str:
    """
    Build a compact dataset context string for prompts from `meta`.
    """
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
    return "Dataset context:\n" + "\n".join(parts)


def _print_tool_calls(state: dict, label: str) -> None:
    """
    Print tool call names and args emitted by the model.
    """
    calls = _all_tool_calls(state.get("messages", []))
    simplified = [{"name": c.get("name"), "args": c.get("args")} for c in calls]
    print(f"{label} tool_calls:", simplified)
