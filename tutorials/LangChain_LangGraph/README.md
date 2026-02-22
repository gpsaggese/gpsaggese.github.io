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

## Mental Model First
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
```bash
cd tutorials/LangChain_LangGraph
cp .env.example .env
# fill provider + key(s) in `.env`
docker compose up --build
```

Open `http://localhost:8888/lab`.

## Quick Start (Local Venv)
```bash
cd tutorials/LangChain_LangGraph
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

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
  environments.
- For production, add stronger auth, persistent stores, audit logging, and
  explicit policy checks around tool calls.
