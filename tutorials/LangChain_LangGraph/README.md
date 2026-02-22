# Learn LangChain + LangGraph in 60 Minutes
This folder is a hands-on, beginner-friendly tutorial for building agentic
workflows with `langchain`, `langgraph`, and `deepagents`.

## What This Folder Teaches
- `langchain.API.ipynb`: API building blocks (LCEL, runnables, tools,
  `ToolNode`, injected state/store, notebook ops, Deep Agents API surface)
- `langchain.example.ipynb`: end-to-end patterns (agent loops, routing,
  reducers, ReAct loop, subagents/subgraphs, HITL, sandboxing, docs-RAG mini
  pipeline)

Coverage map (E0–E7 labels used in the notebooks):

- E0–E2: core APIs and composition in `langchain.API.ipynb`
- E3–E4: agent + graph orchestration patterns in `langchain.example.ipynb`
- E5.\*: notebook automation patterns in `langchain.API.ipynb`
- E6.\*: subagents/subgraphs/HITL in `langchain.example.ipynb`
- E7.\*: Deep Agents workflows in both notebooks

## `langchain.API.ipynb`

- This notebook is the “what are these pieces?” companion to the examples notebook.
- It introduces APIs in small runnable chunks, with heavier concepts (HITL,
  injected state/store, sandboxing) explained before code.

### Section map

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

- LLM-calling cells can cost money; it is safe to skim and selectively execute.
- Filesystem examples write under `workspace/` and `tmp_runs/` in this folder.

## `langchain.example.ipynb`

- This notebook is the end-to-end walkthrough: “how the building blocks become an
  agent workflow.”

### Section map

- **Data grounding**
  - local dataset load + quick EDA for concrete prompts
- **Docs-RAG mini pipeline**
  - markdown docs → chunking → vector store → retrieval chain
  - incremental update pattern via checksum snapshots
- **Agent loop basics**
  - `create_agent` + tool calls + message loop behavior
- **LangGraph control flow**
  - `StateGraph`, conditional routing, reducers
  - ReAct loop from scratch (`model` + `ToolNode`)
- **Composition patterns**
  - supervisor/worker subagents
  - `Command(update=...)` state updates
  - graph-as-node subgraphs
  - shared vs private checkpointer boundaries
- **Human-in-the-loop**
  - interrupts and resume flow with thread IDs
- **Deep Agents (DA1–DA8)**
  - todos, filesystem, backend matrix, dict subagents, compiled subagents
  - HITL edit gates (`interrupt_on`)
  - sandboxing with `FilesystemBackend(virtual_mode=True)`

### Notes for first-time readers

- Run in order on first pass; later sections depend on helper functions/state
  from earlier ones.
- The tutorial intentionally keeps examples small and explicit rather than fully
  abstracted.
- For production, add stronger auth, persistent stores, tighter tool policies,
  and observability.

## Mental Model
- **LangChain**: prompt/model/tool primitives and runnable composition.
- **LangGraph**: stateful orchestration (`StateGraph`), routing,
  memory/checkpointing, interrupts.
- **Deep Agents**: higher-level "agent app" layer (todos, filesystem tools,
  delegation, HITL gates, sandboxing).

If you feel "I see code but I don't know what layer I'm in," use this heuristic:

- Writing prompts/tools/chains → LangChain
- Wiring nodes/edges/state/memory → LangGraph
- Assembling a packaged assistant experience → Deep Agents

## Quick Start (Docker)
- Run
  ```bash
  > cd tutorials/LangChain_LangGraph
  > docker_build.sh
  # Run.
  > cp .env.example .env
  # fill provider + key(s) in `.env`
  > docker_jupyter.sh
  ```

- Open `http://localhost:8888/lab`.

## Suggested Learning Path
1. Run `langchain.API.ipynb` through tools + `ToolNode` + injection patterns.
2. Run `langchain.example.ipynb` through ReAct/subgraphs/HITL.
3. Focus on Deep Agents DA7/DA8 for safe file-edit flows (`interrupt_on` +
   sandboxing).
4. Revisit docs-RAG section for incremental index refresh patterns.

## Architecture + Limitations
- This tutorial favors clarity over production hardening.
- Many cells call live LLM APIs (cost + latency).
- Deep Agents demos are sandboxed to `workspace/` and `tmp_runs/`, but
  token/password-less Jupyter in `docker-compose.yml` is only for trusted local
  environments
