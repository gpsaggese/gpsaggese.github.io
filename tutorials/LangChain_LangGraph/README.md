<!-- toc -->

- [Learn LangChain + LangGraph in 60 Minutes](#learn-langchain--langgraph-in-60-minutes)
  * [What this folder teaches](#what-this-folder-teaches)
  * [Mental model first](#mental-model-first)
  * [Project files](#project-files)
  * [Quick start (Docker)](#quick-start-docker)
  * [Quick start (local venv)](#quick-start-local-venv)
  * [Suggested learning path](#suggested-learning-path)
  * [Architecture + limitations](#architecture--limitations)

<!-- tocstop -->

# Learn LangChain + LangGraph in 60 Minutes

This folder is a hands-on, beginner-friendly tutorial for building agentic workflows with `langchain`, `langgraph`, and `deepagents`.

## What this folder teaches

- `langchain.API.ipynb`: API building blocks (LCEL, runnables, tools, `ToolNode`, injected state/store, notebook ops, Deep Agents API surface)
- `langchain.example.ipynb`: end-to-end patterns (agent loops, routing, reducers, ReAct loop, subagents/subgraphs, HITL, sandboxing, docs-RAG mini pipeline)

Coverage map (E0–E7 labels used in the notebooks):
- E0–E2: core APIs and composition in `langchain.API.ipynb`
- E3–E4: agent + graph orchestration patterns in `langchain.example.ipynb`
- E5.*: notebook automation patterns in `langchain.API.ipynb`
- E6.*: subagents/subgraphs/HITL in `langchain.example.ipynb`
- E7.*: Deep Agents workflows in both notebooks

## Mental model first

- **LangChain**: prompt/model/tool primitives and runnable composition.
- **LangGraph**: stateful orchestration (`StateGraph`), routing, memory/checkpointing, interrupts.
- **Deep Agents**: higher-level “agent app” layer (todos, filesystem tools, delegation, HITL gates, sandboxing).

If you feel “I see code but I don’t know what layer I’m in,” use this heuristic:
- writing prompts/tools/chains → LangChain
- wiring nodes/edges/state/memory → LangGraph
- assembling a packaged assistant experience → Deep Agents

## Project files

- `langchain.API.ipynb`: API tour with runnable snippets
- `langchain.example.ipynb`: end-to-end workflows
- `langchain.API.md`: API section index
- `langchain.example.md`: workflow section index
- `langchain_utils.py`: docs-RAG + incremental update utilities used by notebook examples
- `.env.example`: provider/tracing/embeddings config template
- `requirements.txt`: pinned dependencies for this tutorial
- `Dockerfile`, `docker-compose.yml`, `docker_*.sh`, `run_jupyter.sh`: container workflow helpers

## Quick start (Docker)

```bash
cd tutorials/LangChain_LangGraph
cp .env.example .env
# fill provider + key(s) in `.env`
docker compose up --build
```

Open `http://localhost:8888/lab`.

## Quick start (local venv)

```bash
cd tutorials/LangChain_LangGraph
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

## Suggested learning path

1. Run `langchain.API.ipynb` through tools + `ToolNode` + injection patterns.
2. Run `langchain.example.ipynb` through ReAct/subgraphs/HITL.
3. Focus on Deep Agents DA7/DA8 for safe file-edit flows (`interrupt_on` + sandboxing).
4. Revisit docs-RAG section for incremental index refresh patterns.

## Architecture + limitations

- This tutorial favors clarity over production hardening.
- Many cells call live LLM APIs (cost + latency).
- Deep Agents demos are sandboxed to `workspace/` and `tmp_runs/`, but token/password-less Jupyter in `docker-compose.yml` is only for trusted local environments.
- For production, add stronger auth, persistent stores, audit logging, and explicit policy checks around tool calls.
