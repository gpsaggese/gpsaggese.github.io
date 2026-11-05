# AgenticEDA

Autonomous exploratory data analysis powered by multi-loop parallel subagents. Upload a dataset, get a live notebook with investigations, a narrative story report, and a chat agent to drill deeper.

## What it does

- **Upload & go**: CSV, Excel, JSON, Parquet, log files — drop a file and hit Run
- **Multi-loop investigation**: N parallel subagents per loop, M loops with convergence detection
- **Vision-in-the-loop**: subagents see their own plots (multimodal feedback) and reason about visual patterns
- **Knowledge graph**: findings accumulate across loops with confidence scoring, cross-referencing, and contradiction detection
- **Live notebook**: cells stream in real-time, can be executed against the session kernel, and preserve outputs across refreshes
- **Narrative story**: LLM-synthesized report with executive summary, investigation sections, plot galleries, and cross-notebook cell citations
- **Chat investigations**: ask follow-up questions that spawn background subagents, show complete/failed/timeout status, update the KG, and append to the story
- **Export**: PDF (IEEE format via LaTeX/tectonic) and Markdown

## Architecture

```
Frontend (Next.js)              Backend (FastAPI)              Agent (Python)
┌────────────────────┐         ┌───────────────────┐         ┌──────────────────────┐
│ Notebook (multi-tab)│   WS   │ Session, Kernel,  │         │ Main agent loop      │
│ Story + KaTeX math │◄──────►│ Stream, Story,    │◄───────►│   ├─ Initial EDA      │
│ Chat sidebar       │  REST  │ Chat, Run, History│         │   ├─ Hypothesis gen   │
│ Agent activity log │         │ Kernel pool mgr   │         │   ├─ N subagents (mp) │
│ Progress bar       │         │                   │         │   ├─ KG accumulation  │
└────────────────────┘         └───────────────────┘         │   └─ LLM conclusions  │
                                                              └──────────────────────┘
```

## How it works

A run is a single call to `run_agent` (`src/agent/eda_agent.py`), started by
`POST /api/run/{session_id}` in a background thread. It executes three phases in
order.

### 1. Initial EDA pass

The agent walks an ordered checklist defined in `src/agent/goals.py`: load, inspect
dtypes, describe, parse datetime, audit and handle missing values, distributions,
time series, seasonality, rolling stats, outliers, correlations, train/test split,
summary.

Each goal carries a `should_skip(state)` predicate evaluated against live state, so
steps drop out when they do not apply. Time-series goals are skipped when no usable
time axis was parsed. Correlations are skipped when fewer than two numeric columns
exist.

Cell source comes from deterministic generators in `src/agent/code_templates.py`,
not from the LLM. After execution the output goes to `interpret_output` for a
one-sentence finding, with any plot passed along as an image. The result then goes
to `decide_next_step`, which may issue at most one follow-up cell per goal.

Cell errors are retried twice with LLM-generated corrections that see the prior
error text. If both attempts fail, the cell is replaced with a markdown skip note
and the run continues.

### 2. Investigation loops

Each loop runs hypothesis generation, parallel investigation, and accumulation.

Hypothesis generation (`src/agent/hypothesis.py`) proposes candidates from current
findings plus knowledge-graph context. Each candidate is checked against the graph
by similarity. One that matches an existing hypothesis above 0.75 similarity, where
confidence already exceeds 0.8, is discarded without being re-tested.

Investigation dispatches up to `max_subagents` hypotheses in parallel. Before
dispatch the cleaned dataframe is written to `.cache/df_clean.parquet`, and each
subagent is allocated its own kernel from `backend/services/kernel_pool.py`, seeded
from that file. If the checkpoint is unavailable, subagents share the main kernel.

Subagents run as `multiprocessing.Process` instead of threads, which allows a wedged
worker to be terminated once it passes its deadline. Each subagent runs its own
adaptive loop (`src/agent/subagent.py`): generate one cell, execute it, feed the
output back into the conversation, decide the next cell. Plot images are included in
that feedback when a cell produces one. The loop stops when the model sets `done` or
the cell budget is reached. Common failures such as a missing import are repaired by
pattern match without an LLM call.

Events flow from child processes over a shared queue to a drainer thread that
forwards them to the WebSocket. Subagent cells stream into their own notebook tabs
while the run continues. A subagent that fails or exceeds its deadline is reported
with that status and does not block the others.

Accumulation ingests each result into the knowledge graph as a typed node carrying
confidence, evidence cell IDs, and plot references. Conclusions that share columns
then reinforce each other.

A loop ends the phase early if it is not the first and produced no result above 0.5
confidence.

### 3. Synthesis

A final LLM call cross-references the accumulated findings, flags contradictions
between investigations, and writes numbered conclusions.
`src/reporting/story_builder.py` turns the graph into `story.json`, containing an
executive summary, per-investigation sections, and plot artifacts. That file backs
both the web story view and the PDF export.

### Steering

Instructions submitted while a run is in progress are queued by
`backend/services/steering_service.py` and read only at safe checkpoints: between
goals, after a cell completes, and before follow-ups. They are never applied
mid-execution. Once read, they are appended to the prompt context for subsequent
LLM calls and echoed into the notebook.

### Plot handling

Cells emit plots as `application/vnd.plotly.v1+json` specs built as plain
dictionaries via `emit_plot_spec`; the frontend renders them interactively. Plotly
is not a Python dependency. `src/reporting/plot_contract.py` defines the spec shape.
Matplotlib figures are captured as inline PNG and are what the agent passes back to
vision-capable models.

## Prerequisites

- Python 3.12 recommended
- Node.js 22 recommended for the frontend
- An LLM API key for the provider selected in `.env`
- Optional: Docker and Docker Compose for containerized runs

## Environment setup

Create a root `.env` file before starting the backend. The backend loads this file automatically.

Linux:

```bash
cp .env.example .env
```

macOS:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Windows Command Prompt:

```bat
copy .env.example .env
```

Minimum OpenAI configuration:

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-5.4-nano-2026-03-17
LLM_TIMEOUT=60
LLM_MAX_RETRIES=2
OPENAI_API_KEY=sk-...
```

Supported provider values and required keys:

| `LLM_PROVIDER` | Required env vars |
| --- | --- |
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `google`, `gemini`, `google_genai` | `GOOGLE_API_KEY` |
| `openai_compatible` | `OPENAI_COMPAT_BASE_URL`, `OPENAI_COMPAT_API_KEY` |
| `azure_openai_v1` | `AZURE_OPENAI_BASE_URL`, `AZURE_OPENAI_API_KEY` |

Optional model controls:

```env
LLM_TEMP=0.2
EDA_AGENT_MODEL=gpt-5.4-nano-2026-03-17
EDA_GATE_MODEL=gpt-5.4-nano-2026-03-17
EDA_SUBAGENT_MODEL=gpt-5.4-nano-2026-03-17
```

Frontend API configuration is usually not needed for local development. When the frontend runs on `localhost:3000`, it automatically calls `http://localhost:8000/api`.

Set this only when the backend API is somewhere else:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

For local frontend development, put `NEXT_PUBLIC_API_URL` in `frontend/.env.local` or export it in the shell before running `npm run dev`.

## Run locally

Start the backend from the repo root.

Linux:

```bash
python3 -m pip install -r requirements.txt
PYTHONPATH=. uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

macOS:

```bash
python3 -m pip install -r requirements.txt
PYTHONPATH=. uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Windows PowerShell:

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH="."
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Windows Command Prompt:

```bat
python -m pip install -r requirements.txt
set PYTHONPATH=.
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Start the frontend in a second terminal:

Linux:

```bash
cd frontend
npm install
npm run dev
```

macOS:

```bash
cd frontend
npm install
npm run dev
```

Windows PowerShell:

```powershell
cd frontend
npm install
npm run dev
```

Windows Command Prompt:

```bat
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`, upload a dataset, configure subagents/loops/depth, and click Run EDA.

## Run with Docker

The Docker Compose setup reads backend environment variables from the root `.env` file.

Linux:

```bash
cp .env.example .env
# edit .env and add your provider API key
docker compose up --build
```

macOS:

```bash
cp .env.example .env
# edit .env and add your provider API key
docker compose up --build
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
# edit .env and add your provider API key
docker compose up --build
```

Windows Command Prompt:

```bat
copy .env.example .env
REM edit .env and add your provider API key
docker compose up --build
```

Compose starts two services. The backend is published on
`http://localhost:8000` and the frontend on `http://localhost:3000`. Open the
frontend to use the app.

The browser resolves the API origin at runtime via `inferApiBase()` in
`frontend/src/lib/backend.ts`, so no API URL needs to be configured for this
layout. Set `NEXT_PUBLIC_API_URL` at image build time only when the backend is
not reachable from the browser's origin.

Session data persists in the `sessions` Docker volume across restarts.

The backend image builds on both `linux/amd64` and `linux/arm64`. Tectonic is
installed by direct download with the release asset selected from
`dpkg --print-architecture`, because upstream publishes a musl build for aarch64
and a gnu build for x86_64.

## Project structure

```
src/
├── agent/
│   ├── eda_agent.py          # Main loop: goal checklist, investigation loops, synthesis
│   ├── goals.py              # Ordered EDA checklist with skip predicates
│   ├── code_templates.py     # Deterministic cell-source generators
│   ├── reasoning.py          # Output interpretation (multimodal) + next-step decisions
│   ├── hypothesis.py         # Hypothesis generation and dedup against the graph
│   ├── subagent.py           # Adaptive per-hypothesis investigation loop
│   ├── subagent_worker.py    # Process-safe worker (kernel connection via file)
│   ├── knowledge_graph.py    # Typed nodes/edges, confidence scoring, evidence chains
│   └── state.py              # AgentState carried through the run
├── config/config.py          # get_chat_model(): multi-provider LLM factory
├── ingest/file_loader.py     # CSV/Excel/JSON/Parquet loading
├── reporting/
│   ├── story_builder.py      # Graph to story.json (sections, captions, plots)
│   ├── plot_contract.py      # Plot artifact spec shared with the frontend
│   └── versioning.py         # Notebook and story snapshot history
└── chat/chat_agent.py        # Follow-up Q&A over the knowledge graph

backend/
├── app.py                    # FastAPI application, router mounting, CORS
├── Dockerfile                # Backend image (builds on amd64 and arm64)
├── routers/
│   ├── run.py                # Starts run_agent in a background thread
│   ├── chat.py               # Chat + hypothesis investigation events/status
│   ├── notebook.py           # Notebook fetch/patch with output normalization
│   ├── kernel.py             # Session kernel status and code-cell execution
│   ├── story.py              # Story fetch, regenerate, PDF/Markdown export
│   ├── stream.py             # WebSocket event streaming
│   ├── session.py            # Upload, session management
│   └── history.py            # Version history endpoints
└── services/
    ├── kernel_manager.py     # IPython kernel lifecycle + cross-process execution
    ├── kernel_pool.py        # Multi-kernel allocation for parallel subagents
    ├── session_manager.py    # Session directories on disk
    ├── steering_service.py   # Queue for mid-run user instructions
    └── history_service.py    # Delegates to src/reporting/versioning.py

frontend/src/
├── stores/                   # Zustand (notebook, story, chat, session)
├── hooks/                    # useAgentStream (WS event routing), useChat, useKernel
├── components/
│   ├── notebook/             # NotebookPane, NotebookCell, CellOutput, ThinkingBlock
│   ├── story/                # StoryPane, StorySectionCard (KaTeX + cross-notebook citations)
│   ├── chat/                 # ChatSidebar, ChatInput, ExecutionPlan
│   └── layout/               # AgentActivityBadge, NotebookTabs
└── app/session/[id]/page.tsx # Main session page

docs/
└── design.md                 # Architecture design notes
```

## Tech stack

- **Agent**: Python, `multiprocessing` for parallel subagents
- **Backend**: FastAPI, Jupyter kernel client, WebSocket streaming
- **Frontend**: Next.js, React, Zustand, Tailwind CSS, KaTeX, react-markdown
- **LLM**: configurable (OpenAI, Anthropic, Google, Azure, OpenAI-compatible)
- **PDF**: tectonic (LaTeX) with IEEEtran document class

LangChain is used as a provider abstraction only. `get_chat_model()` builds a
client from `langchain-core` plus the adapter package for the configured
provider, and the agent calls `.invoke()` on it with `SystemMessage` and
`HumanMessage`. Control flow, state, and looping are implemented directly in
`src/agent/`, so there are no chains, prompt templates, output parsers, or tool
binding. Prompts are f-strings and structured responses are parsed with
`json.loads`.

`requirements.txt` includes `scipy`, `statsmodels`, and `scikit-learn` even
though no module under `src/` imports them. The agent generates Python that runs
in the Jupyter kernel and uses these libraries; `scipy.stats` is named
explicitly in the subagent prompt. Removing them causes generated cells to fail
at execution time.
