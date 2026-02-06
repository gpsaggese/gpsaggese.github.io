# `langchain.API.ipynb` index

This notebook is a runnable, minimal tour of the APIs used throughout this tutorial folder.

Covered (with runnable snippets):

- LCEL basics: `ChatPromptTemplate`, `StrOutputParser`, `prompt | model | parser`
- Runnables: `.invoke()`, `.batch()`, `.stream()`, `RunnableParallel`
- Tools: `@tool`, calling tools directly, tool schemas
- LangGraph tool execution: `ToolNode`, direct tool-call format, ToolMessage outputs
- Injection patterns: `InjectedState`, `InjectedStore` (+ `InMemoryStore`)
- Notebook ops: `nbformat` writing + `nbclient` execution (see the “Notebook ops” section in the notebook, and `notebooks/` for small examples)
- Deep agents: todos, filesystem tools, backends, subagents, HITL gates, sandboxing (E7)

This notebook assumes a provider configured via `.env` (see `.env.example`).
