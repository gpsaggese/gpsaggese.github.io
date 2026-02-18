# `langchain.API.ipynb` index

This notebook is the “what are these pieces?” companion to the examples notebook.
It introduces APIs in small runnable chunks, with heavier concepts (HITL, injected state/store, sandboxing) explained before code.

## Section map

- **Setup + model factory**
  - `.env`-driven model selection (`openai` / `anthropic`, optional `ollama`)
  - reproducibility knobs (`LLM_TEMPERATURE`, retries/timeouts)
- **LCEL + runnables**
  - `ChatPromptTemplate`, `StrOutputParser`, `prompt | model | parser`
  - `.invoke()`, `.batch()`, `.stream()`, `RunnableParallel`
- **Tools + ToolNode**
  - `@tool` function schema
  - direct tool-call execution through LangGraph `ToolNode`
- **Injection patterns (human-safe defaults)**
  - `InjectedState` for system-owned runtime context
  - `InjectedStore` + `InMemoryStore` for persisted values
- **Agent APIs**
  - `create_agent`, `AgentState`, `ToolRuntime`, `InjectedToolCallId`
  - reproducible tool-call output contract pattern
- **HITL primitive**
  - `interrupt(...)` + `Command(resume=...)` with checkpointed thread state
- **Notebook operations**
  - write notebooks via `nbformat`
  - execute via `nbclient`
  - parameterize via `papermill`
  - extract execution artifacts
- **Deep Agents API surface**
  - `create_deep_agent`, subagents, backends, HITL gates, sandboxing

## Notes for first-time readers

- If a section feels advanced, read the markdown first and run cells one by one.
- LLM-calling cells can cost money; it is safe to skim and selectively execute.
- Filesystem examples write under `workspace/` and `tmp_runs/` in this folder.
