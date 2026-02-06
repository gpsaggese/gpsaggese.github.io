# `langchain.example.ipynb` index

This notebook shows end-to-end examples that mirror the E0–E7 section labels used throughout these notebooks.

Covered (with runnable snippets):

- Agent loop via `create_agent` + tool calling (E3)
- Middleware hook (`before_agent`) pattern (E3)
- LangGraph patterns: `StateGraph`, conditional routing, reducers (E4)
- ReAct loop “from scratch”: model node + `ToolNode` loop (E4)
- Subgraphs and memory boundaries (E6)
- HITL interrupts + resume via `Command` (E6)
- Deep agents: todos, filesystem tools, backends, subagents, HITL gates, sandboxing (E7)

The notebook assumes a provider configured via `.env` (see `.env.example`).
