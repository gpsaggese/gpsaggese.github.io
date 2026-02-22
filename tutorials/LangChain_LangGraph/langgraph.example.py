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
# ## LangGraph: StateGraph (hello)
#
# LangGraph is a way to express workflows as a **stateful graph**.
#
# If you’ve ever thought “I wish this agent had a clear structure and memory,” this is the tool.
#
# A tiny checklist as you read the next cell:
# - What does “state” look like? (a dict / TypedDict)
# - What are the nodes? (functions that read + return updates)
# - How do edges determine what runs next?
#

# %%
# This cell will:
# - Build and compile a `StateGraph` (a small LangGraph workflow).
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# #############################################################################
# S
# #############################################################################


class S(TypedDict):
    n: int
    msg: str


def inc(state: S) -> dict:
    """Increment `state['n']` by 1."""
    return {"n": state.get("n", 0) + 1}


def set_msg(state: S) -> dict:
    """Set `state['msg']` to a string derived from the current counter."""
    return {"msg": f"n={state.get('n', 0)}"}


g = StateGraph(S)
g.add_node("inc", inc)
g.add_node("msg", set_msg)
g.add_edge(START, "inc")
g.add_edge("inc", "msg")
g.add_edge("msg", END)
graph = g.compile()

graph.invoke({"n": 0, "msg": ""})


# %% [markdown]
# ## LangGraph: conditional routing
#
# Graphs get interesting when the next step depends on state.
#
# In this section you’ll see:
# - a node returns an update
# - a router looks at state and chooses the next node
#
# This is the foundation for “if the model asked for a tool, run tools; otherwise, finish.”
#

# %%
# This cell will:
# - Build and compile a `StateGraph` (a small LangGraph workflow).
from typing import Literal


# #############################################################################
# R
# #############################################################################


class R(TypedDict):
    flag: bool
    out: str


def a(state: R) -> dict:
    """Write a marker output for the `A` branch."""
    return {"out": "path=A"}


def b(state: R) -> dict:
    """Write a marker output for the `B` branch."""
    return {"out": "path=B"}


def route(state: R) -> Literal["a", "b"]:
    """Route based on the boolean `state['flag']`."""
    return "a" if state.get("flag") else "b"


g = StateGraph(R)
g.add_node("a", a)
g.add_node("b", b)
g.add_conditional_edges(START, route, {"a": "a", "b": "b"})
g.add_edge("a", END)
g.add_edge("b", END)
graph = g.compile()

graph.invoke({"flag": True, "out": ""}), graph.invoke({"flag": False, "out": ""})


# %% [markdown]
# ## LangGraph: reducers (accumulate evidence)
#
# Reducers are how you *accumulate* state across steps.
#
# Common uses:
# - collect “evidence” across iterations
# - build up a list of intermediate results
# - append messages rather than overwrite
#
# In the next cell, focus on how state updates combine rather than replace.
#

# %%
# This cell will:
# - Build and compile a `StateGraph` (a small LangGraph workflow).
from typing import Annotated, List


def add_list(old: List[str], new: List[str]) -> List[str]:
    """Reducer that concatenates two evidence lists."""
    return old + new


# #############################################################################
# ReducerState
# #############################################################################


class ReducerState(TypedDict):
    evidence: Annotated[List[str], add_list]


def find_missingness(_: ReducerState) -> dict:
    """Compute missingness findings from the local dataset."""
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    top = miss.head(3)
    evidence = [
        f"missingness: {idx} has {val:.2f}% missing" for idx, val in top.items()
    ]
    return {"evidence": evidence}


def find_outliers(_: ReducerState) -> dict:
    """Compute a simple outlier finding using z-scores on one numeric column."""
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
            "evidence": [f"outliers: std({col}) is 0, cannot compute z-scores"]
        }
    z = ((s - mu) / sigma).abs()
    idx = int(z.idxmax())
    ts = None
    if "Date/Time" in df.columns:
        ts = df.loc[idx, "Date/Time"]
    evidence = [
        f"outliers: max |z| for {col} at row={idx} ts={ts} value={s.loc[idx]:.3f} z={z.loc[idx]:.2f}"
    ]
    return {"evidence": evidence}


g = StateGraph(ReducerState)
g.add_node("missingness", find_missingness)
g.add_node("outliers", find_outliers)
g.add_edge(START, "missingness")
g.add_edge("missingness", "outliers")
g.add_edge("outliers", END)
graph = g.compile()

graph.invoke({"evidence": []})["evidence"]


# %% [markdown]
# ## ReAct loop from scratch: model node + ToolNode
#
# ReAct (“Reason + Act”) is a simple pattern:
#
# - the model thinks about what to do
# - if it needs information, it calls a tool
# - it repeats until it can answer
#
# Here we build that loop explicitly with LangGraph:
# - a model node that proposes tool calls
# - a `ToolNode` that executes them
# - routing logic that decides whether to continue looping
#
# This is one of the best places to pause and say: “Ah — *this* is what an agent really is.”
#

# %%
# This cell will:
# - Build and compile a `StateGraph` (a small LangGraph workflow).
# - Use `ToolNode` to execute tool calls inside a graph.
from typing import Annotated as Ann
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# #############################################################################
# RS
# #############################################################################


class RS(TypedDict):
    messages: Ann[list, add_messages]


tools = [utc_now, mean, sqrt]
tool_node = ToolNode(tools)


def call_model(state: RS) -> dict:
    """Call the model with bound tools and append the AI message."""
    bound = llm.bind_tools(tools)
    ai = bound.invoke(state["messages"])
    return {"messages": [ai]}


def needs_tools(state: RS) -> str:
    """Route to tools if the last AI message contains tool calls."""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "end"


g = StateGraph(RS)
g.add_node("model", call_model)
g.add_node("tools", tool_node)
g.add_edge(START, "model")
g.add_conditional_edges("model", needs_tools, {"tools": "tools", "end": END})
g.add_edge("tools", "model")
graph = g.compile()

out = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="Compute mean([1,2,3,4,10]) and sqrt(49). Also tell me the current UTC time."
            )
        ]
    }
)
[(type(m).__name__, getattr(m, "content", "")[:120]) for m in out["messages"]][
    -4:
]


# %% [markdown]
# ## Subagents: supervisor + worker tools
#
# A helpful pattern is to split responsibilities:
# - a **supervisor** decides what needs doing
# - **workers** do specialized tasks (often via tools)
#
# This keeps each piece simpler and makes debugging much easier.
#

# %%
# This cell will:
# - Create a tool-calling agent using `create_agent(...)`.
from langchain.tools import tool as lc_tool


def _last_text(result: dict) -> str:
    """
    Return the final message text/content from an agent result state.
    """
    msg = result["messages"][-1]
    return (
        getattr(msg, "text", None) or getattr(msg, "content", None) or str(msg)
    )


# E6.1: supervisor calls a worker wrapped as a tool
worker_agent = create_agent(
    llm,
    tools=[],
    system_prompt=(
        "You are a summarization specialist.\n"
        "Given text, return:\n"
        "- 1 sentence summary\n"
        "- 3 bullet key points\n"
        "Return only the summary + bullets."
    ),
)


@lc_tool(
    "summarize_text",
    description="Summarize long text into a short summary + 3 bullet points.",
)
def summarize_text(text: str) -> str:
    """
    Summarize `text` using the worker agent and return a plain string.
    """
    return _last_text(
        worker_agent.invoke({"messages": [{"role": "user", "content": text}]})
    )


supervisor = create_agent(
    llm,
    tools=[summarize_text],
    system_prompt="If asked to summarize, call summarize_text and return the tool output.",
)

out = supervisor.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Summarize: LangChain provides building blocks for LLM apps.",
            }
        ]
    }
)
_last_text(out)


# %% [markdown]
# ## Subagents: ToolRuntime state + Command(update=...)
#
# Sometimes you want a tool to do more than return a value — you want it to update graph state.
#
# In LangGraph that’s expressed with `Command(update=...)`.
#
# You’ll also see `ToolRuntime`, which gives the tool access to useful runtime context (like the current state).
#

# %%
# This cell will:
# - Create a tool-calling agent using `create_agent(...)`.
import json
from typing_extensions import Annotated as TxAnnotated

from langchain.agents import AgentState
from langchain.tools import ToolRuntime, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command


# #############################################################################
# CustomState
# #############################################################################


class CustomState(AgentState):
    user_prefs: dict
    facts: list[str]


worker = create_agent(
    llm, tools=[], system_prompt="Rewrite text. Return only rewritten text."
)


@lc_tool(
    "rewrite_with_prefs",
    description="Rewrite text following preferences from supervisor state.",
)
def rewrite_with_prefs(
    text: str, runtime: ToolRuntime[None, CustomState]
) -> str:
    """
    Rewrite `text` using supervisor preferences available via `runtime.state`.
    """
    tone = runtime.state.get("user_prefs", {}).get("tone", "neutral")
    result = worker.invoke(
        {
            "messages": [
                {"role": "system", "content": f"Tone must be: {tone}."},
                {"role": "user", "content": text},
            ]
        }
    )
    return _last_text(result)


fact_worker = create_agent(
    llm, tools=[], system_prompt='Return ONLY JSON: {"facts": ["..."]}'
)


@lc_tool(
    "extract_facts",
    description="Extract facts and update supervisor state via Command(update=...).",
)
def extract_facts(
    text: str, tool_call_id: TxAnnotated[str, InjectedToolCallId]
) -> Command:
    """
    Extract facts and store them in the supervisor state via `Command(update=...)`.
    """
    raw = _last_text(
        fact_worker.invoke({"messages": [{"role": "user", "content": text}]})
    )
    try:
        facts = list(json.loads(raw).get("facts", []))
    except Exception:
        facts = [raw]
    return Command(
        update={
            "facts": facts,
            "messages": [
                ToolMessage(
                    content=f"Stored {len(facts)} facts.",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


supervisor = create_agent(
    llm,
    tools=[rewrite_with_prefs, extract_facts],
    system_prompt="Use rewrite_with_prefs for rewrite requests; use extract_facts for 'read and explain'.",
    state_schema=CustomState,
)

out1 = supervisor.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Rewrite: please send me the report by tonight.",
            }
        ],
        "user_prefs": {"tone": "formal"},
        "facts": [],
    }
)
out2 = supervisor.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Read this and explain it: LangGraph supports interrupts.",
            }
        ],
        "user_prefs": {"tone": "neutral"},
        "facts": [],
    }
)

{"rewrite": _last_text(out1), "facts_updated": out2.get("facts")}


# %%
# This cell will:
# - Create a tool-calling agent using `create_agent(...)`.
# E6.2: two subagents (date normalization + email drafting)
date_agent = create_agent(
    llm,
    tools=[],
    system_prompt='Return ONLY JSON: {"normalized": "...", "notes": "..."}',
)


@lc_tool(
    "normalize_datetime",
    description="Normalize informal date/time mentions into an explicit format. Returns JSON.",
)
def normalize_datetime(request: str) -> str:
    """
    Normalize an informal date/time request using a specialized subagent.
    """
    return _last_text(
        date_agent.invoke({"messages": [{"role": "user", "content": request}]})
    )


email_agent = create_agent(
    llm,
    tools=[],
    system_prompt="Draft a short professional email body. Return only the email body.",
)


@lc_tool(
    "draft_email_body",
    description="Draft a concise professional email body for a user request.",
)
def draft_email_body(request: str) -> str:
    """
    Draft a short professional email body for `request`.
    """
    return _last_text(
        email_agent.invoke({"messages": [{"role": "user", "content": request}]})
    )


sup = create_agent(
    llm,
    tools=[normalize_datetime, draft_email_body],
    system_prompt="Pick the right tool for the user's intent.",
)

a = sup.invoke(
    {
        "messages": [
            {"role": "user", "content": "Normalize: next Tuesday at 2pm."}
        ]
    }
)
b = sup.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Write an email to my professor asking for a 2-day extension.",
            }
        ]
    }
)
{"normalize_datetime": _last_text(a), "draft_email_body": _last_text(b)}


# %%
# This cell will:
# - Create a tool-calling agent using `create_agent(...)`.
# E6.3: context isolation (noisy worker, clean supervisor)
@lc_tool(
    "generate_noise",
    description="Generate a long string to simulate noisy intermediate work.",
)
def generate_noise(n_chars: int) -> str:
    """
    Generate a long string used to simulate irrelevant intermediate work.
    """
    return "X" * int(n_chars)


noisy_worker_agent = create_agent(
    llm,
    tools=[generate_noise],
    system_prompt=(
        "You MUST call generate_noise with n_chars=8000 exactly once, then ignore it.\n"
        "Return ONLY a concise 2-sentence answer."
    ),
)


@lc_tool(
    "noisy_worker",
    description="Do a task in an isolated context and return a concise final answer.",
)
def noisy_worker(task: str) -> str:
    """
    Run `task` in an isolated subagent context and return the final answer.
    """
    return _last_text(
        noisy_worker_agent.invoke(
            {"messages": [{"role": "user", "content": task}]}
        )
    )


sup = create_agent(
    llm,
    tools=[noisy_worker],
    system_prompt="Call noisy_worker for the user's request.",
)
out = sup.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Explain in plain English what 'context isolation' means in subagents.",
            }
        ]
    }
)
_last_text(out)


# %%
# This cell will:
# - Create a tool-calling agent using `create_agent(...)`.
# - Demonstrate notebook operations (write/execute/parameterize notebooks).
# E6.6: parallel tool calls (one AI turn emits multiple tool calls)
sum_agent = create_agent(
    llm,
    tools=[],
    system_prompt="Summarize in 2 sentences. Return only the summary.",
)
act_agent = create_agent(
    llm,
    tools=[],
    system_prompt="Extract action items as bullets. Return only bullets.",
)
reply_agent = create_agent(
    llm,
    tools=[],
    system_prompt="Draft a short reply email. Return only the email body.",
)


@lc_tool("sub_summarize", description="Summarize the text in 2 sentences.")
def sub_summarize(text: str) -> str:
    """
    Summarize `text` in 2 sentences.
    """
    return _last_text(
        sum_agent.invoke({"messages": [{"role": "user", "content": text}]})
    )


@lc_tool(
    "sub_action_items", description="Extract action items as bullet points."
)
def sub_action_items(text: str) -> str:
    """
    Extract action items from `text` as bullet points.
    """
    return _last_text(
        act_agent.invoke({"messages": [{"role": "user", "content": text}]})
    )


@lc_tool(
    "sub_draft_reply",
    description="Draft a short email reply addressing the content.",
)
def sub_draft_reply(text: str) -> str:
    """
    Draft a short email reply based on `text`.
    """
    return _last_text(
        reply_agent.invoke({"messages": [{"role": "user", "content": text}]})
    )


sup = create_agent(
    llm,
    tools=[sub_summarize, sub_action_items, sub_draft_reply],
    system_prompt="Use tools as needed and return a clean final response.",
)

email_thread = (
    "Call ALL THREE tools (sub_summarize, sub_action_items, sub_draft_reply). "
    "Text: We need to ship the notebook execution feature by Friday. Please confirm papermill works."
)
out = sup.invoke({"messages": [{"role": "user", "content": email_thread}]})
_last_text(out)


# %% [markdown]
# ## Subgraphs (graph-as-node composition)
#
# A subgraph is just a graph you treat like a node.
#
# This is how you build larger systems without everything becoming one giant tangle:
# - small graph for “extract facts”
# - small graph for “summarize”
# - parent graph that composes them
#

# %%
# This cell will:
# - Build and compile a `StateGraph` (a small LangGraph workflow).
# #############################################################################
# SubState
# #############################################################################
class SubState(TypedDict):
    raw: str
    parsed: dict
    formatted: str


def parse_node(state: SubState) -> dict:
    """Parse `key: value` lines from `state['raw']` into a dict."""
    raw = state["raw"]
    parsed = {}
    for line in raw.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            parsed[k.strip()] = v.strip()
    return {"parsed": parsed}


def format_node(state: SubState) -> dict:
    """Format the parsed fields as a bullet list."""
    parsed = state.get("parsed", {})
    lines = [f"- {k}: {v}" for k, v in parsed.items()]
    return {"formatted": "Parsed fields:\n" + "\n".join(lines)}


sub = StateGraph(SubState)
sub.add_node("parse", parse_node)
sub.add_node("format", format_node)
sub.add_edge(START, "parse")
sub.add_edge("parse", "format")
sub.add_edge("format", END)
subgraph = sub.compile()


# #############################################################################
# ParentState
# #############################################################################


class ParentState(TypedDict):
    user_text: str
    result: str


def call_subgraph(state: ParentState) -> dict:
    """Call `subgraph` and project its formatted output into the parent state."""
    out = subgraph.invoke({"raw": state["user_text"]})
    return {"result": out["formatted"]}


parent = StateGraph(ParentState)
parent.add_node("worker", call_subgraph)
parent.add_edge(START, "worker")
parent.add_edge("worker", END)
parent_graph = parent.compile()

parent_graph.invoke(
    {
        "user_text": "name: Indro\nrole: ML engineer\nlocation: Kolkata",
        "result": "",
    }
)["result"]


# %% [markdown]
# ## Shared vs private memory boundaries (checkpointers)
#
# Checkpointers are how LangGraph remembers state across runs.
#
# This section helps answer:
# - what does `thread_id` do?
# - when do two runs share memory vs start fresh?
# - how do you keep different users/sessions isolated?
#
# If you’ve ever debugged an agent that “remembered the wrong thing,” this is the antidote.
#

# %%
# This cell will:
# - Build and compile a `StateGraph` (a small LangGraph workflow).
from langgraph.checkpoint.memory import MemorySaver


# #############################################################################
# CSub
# #############################################################################


class CSub(TypedDict):
    n: int


def bump(state: CSub) -> dict:
    """Increment a counter used to demonstrate subgraph memory behavior."""
    return {"n": state.get("n", 0) + 1}


sub_builder = StateGraph(CSub)
sub_builder.add_node("bump", bump)
sub_builder.add_edge(START, "bump")
sub_builder.add_edge("bump", END)

sub_shared = sub_builder.compile()
sub_private = sub_builder.compile(checkpointer=MemorySaver())


# #############################################################################
# P
# #############################################################################


class P(TypedDict):
    mode: str
    sub_n: int


def call_sub(state: P) -> dict:
    """Call the shared or private subgraph depending on `state['mode']`."""
    if state["mode"] == "shared":
        out = sub_shared.invoke({"n": state.get("sub_n", 0)})
        return {"sub_n": out["n"]}
    out = sub_private.invoke(
        {"n": 0}, config={"configurable": {"thread_id": "SUBGRAPH_THREAD"}}
    )
    return {"sub_n": out["n"]}


parent_builder = StateGraph(P)
parent_builder.add_node("call_sub", call_sub)
parent_builder.add_edge(START, "call_sub")
parent_builder.add_edge("call_sub", END)
parent = parent_builder.compile(checkpointer=MemorySaver())


def run_twice(mode: str):
    """Invoke the parent graph twice and return the two observed sub-counters."""
    out1 = parent.invoke(
        {"mode": mode, "sub_n": 0},
        config={"configurable": {"thread_id": f"PARENT_{mode}"}},
    )
    out2 = parent.invoke(
        {"mode": mode, "sub_n": out1["sub_n"]},
        config={"configurable": {"thread_id": f"PARENT_{mode}"}},
    )
    return out1["sub_n"], out2["sub_n"]


run_twice("shared"), run_twice("private")


# %% [markdown]
# ## Human-in-the-loop gate (interrupt + resume)
#
# HITL isn’t about slowing you down — it’s about making powerful agents *safe*.
#
# In this section, the graph will:
# - pause with an interrupt payload
# - wait for a human decision
# - resume using `Command(resume=...)`
#
# If you’re wondering “where does the UI come from?” — great question.
# LangGraph gives you the **primitive** (interrupt + resume). You can surface that in a notebook, a web app, Slack, etc.
#

# %%
# This cell will:
# - Build and compile a `StateGraph` (a small LangGraph workflow).
# - Demonstrate human-in-the-loop control using `interrupt(...)` and resume.
from pathlib import Path
from typing import Literal as Lit

from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver


# #############################################################################
# HITLState
# #############################################################################


class HITLState(TypedDict):
    target_path: str
    decision: Lit["approve", "reject", ""]


def propose_delete(state: HITLState) -> dict:
    """Emit an interrupt asking for approval to delete `state['target_path']`."""
    payload = {
        "action": "delete_file",
        "target_path": state["target_path"],
        "message": "Approve deletion?",
    }
    decision = interrupt(payload)
    return {"decision": decision}


def do_delete(state: HITLState) -> dict:
    """Delete the file if the prior interrupt decision was `approve`."""
    if state["decision"] != "approve":
        return {}
    p = Path(state["target_path"])
    if p.exists() and p.is_file():
        p.unlink()
    return {}


builder = StateGraph(HITLState)
builder.add_node("propose", propose_delete)
builder.add_node("delete", do_delete)
builder.add_edge(START, "propose")
builder.add_edge("propose", "delete")
builder.add_edge("delete", END)

hitl_graph = builder.compile(checkpointer=MemorySaver())

tmp_dir = Path("tmp_runs").resolve()
tmp_dir.mkdir(parents=True, exist_ok=True)
victim = tmp_dir / "victim.txt"
victim.write_text("delete me", encoding="utf-8")

thread_id = "HITL_NOTEBOOK_DEMO"
out1 = hitl_graph.invoke(
    {"target_path": str(victim), "decision": ""},
    config={"configurable": {"thread_id": thread_id}},
)
pending = (
    out1.get("__interrupt__", [])[0].value if "__interrupt__" in out1 else None
)
pending


# %%
# This cell will:
# - Run the next step of the end-to-end example.
out2 = hitl_graph.invoke(
    Command(resume="approve"), config={"configurable": {"thread_id": thread_id}}
)
victim.exists()

