# `langchain.example.ipynb` index

This notebook is the end-to-end walkthrough: “how the building blocks become an agent workflow.”

## Section map

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

## Notes for first-time readers

- Run in order on first pass; later sections depend on helper functions/state from earlier ones.
- The tutorial intentionally keeps examples small and explicit rather than fully abstracted.
- For production, add stronger auth, persistent stores, tighter tool policies, and observability.
