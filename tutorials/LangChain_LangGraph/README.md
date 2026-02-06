<!-- toc -->

- [Learn LangChain in 60 Minutes](#learn-langchain-in-60-minutes)
  * [What you’ll learn (mapped to the E0–E7 labels used in these notebooks)](#what-youll-learn-mapped-to-the-e0e7-labels-used-in-these-notebooks)
  * [Project files](#project-files)
  * [Docker: build + run JupyterLab](#docker-build--run-jupyterlab)
  * [Running the notebooks](#running-the-notebooks)
  * [Provider configuration](#provider-configuration)

<!-- tocstop -->

# Learn LangChain in 60 Minutes

This folder is a “Learn X in 60 Minutes” tutorial. If you want the writing guide we use for these tutorials, see:
`https://github.com/causify-ai/tutorials/blob/master/docs/all.learn_X_in_60_minutes.how_to_guide.md`

## What you’ll learn (mapped to the E0–E7 labels used in these notebooks)

- `langchain.API.ipynb`: E0–E2 + E5 + E7 (LCEL, runnables, tools, ToolNode, InjectedState/InjectedStore, notebook ops, deep agents)
- `langchain.example.ipynb`: E3–E6 + E7 (agent loop, LangGraph patterns, reducers, subagents, subgraphs/HITL, deep agents)

Coverage checklist (by the E0–E7 section labels used in the notebooks):
- E0–E2: covered in `langchain.API.ipynb`
- E3–E4: covered in `langchain.example.ipynb`
- E5.1–E5.9: covered in `langchain.API.ipynb` (with runnable nbformat/nbclient/papermill demos)
- E6.1–E6.9: covered in `langchain.example.ipynb` (subagents + subgraphs + HITL)
- E7.*: covered in **both** notebooks (deep agents: todos, filesystem tools, backends, subagents, HITL gates, sandboxing)

Both notebooks assume a provider is configured via a `.env` file (see `.env.example`).

## Project files

- `langchain.API.ipynb`: basic LangChain/LangGraph APIs used throughout this tutorial
- `langchain.example.ipynb`: end-to-end examples (agent loop, graphs, subagents, HITL, deep agents)
- `langchain.API.md`: quick index of APIs used
- `langchain.example.md`: quick index of example workflows
- `requirements.txt`: pinned deps for this tutorial
- `Dockerfile`: installs deps for JupyterLab
- `docker-compose.yml`: starts JupyterLab with the repo mounted at `/app`

## Docker: build + run JupyterLab

From this directory:

```bash
cd tutorials/LangChain_LangGraph
cp .env.example .env
# Edit `.env` and set your API key(s).
docker compose up --build
```

Then open:

- `http://localhost:8888/lab`

Notes:
- The compose file disables the Jupyter token/password for convenience. Don’t use this on an untrusted network.
- The repo is mounted into the container (`/app`), so edits on your host are reflected instantly.

## Running the notebooks

Open these two notebooks and run them top-to-bottom:

- `langchain.API.ipynb`
- `langchain.example.ipynb`

Notes:
- Running these notebooks will call your configured LLM provider and may incur costs
- The notebooks include a small local dataset at `data/T1_slice.csv`
- Deep Agents filesystem demos write files under `workspace/` and `tmp_runs/deepagents/`

## Provider configuration

Set env vars in `.env` (loaded via `docker-compose.yml`) and restart the container. See `.env.example` for the supported variables.
