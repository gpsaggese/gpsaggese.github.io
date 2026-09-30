# AgenticEDA — Backend Notes

This backend serves a real-time agentic EDA product. A user uploads a dataset;
an agent writes and executes notebook cells live, streaming every action to the
frontend over a WebSocket.

> **Note:** this file previously documented a staged `--mode` pipeline built on
> LangGraph (`src/pipeline.py` plus per-phase modules under `src/ingest/`,
> `src/quality_handling/`, `src/univariate_analysis/`, and others). That code
> was never reachable from the running server and has been removed. See
> "History" at the bottom.

## Entrypoint

```
docker-compose.yml → backend/Dockerfile → uvicorn backend.app:app
  → backend/routers/run.py:_run_agent_in_thread
    → src.agent.eda_agent.run_agent
```

`run_agent` executes in a background thread per session. There is no CLI
entrypoint.

## Package Layout

- `backend/` — FastAPI app: routers, services, Pydantic models.
- `src/agent/` — the agent loop. This is where the product logic lives.
- `src/config/config.py` — `get_chat_model()`, the multi-provider LLM factory.
- `src/chat/chat_agent.py` — follow-up Q&A over the knowledge graph.
- `src/ingest/file_loader.py` — CSV/Excel/Parquet loading.
- `src/reporting/` — story building, plot contract, version history.

## Agent Loop Shape

`run_agent` (`src/agent/eda_agent.py`) has three sections:

1. **Goal checklist** — iterates `build_goal_checklist()` from
   `src/agent/goals.py`. Each goal has a `should_skip(state)` predicate, so
   time-series steps are skipped when no usable time column was parsed. Cell
   sources come from `src/agent/code_templates.py` (deterministic) with LLM
   interpretation and at most one follow-up per goal via
   `src/agent/reasoning.py`.

2. **Investigation loops** — up to `max_loops` rounds. Each round generates
   hypotheses (`src/agent/hypothesis.py`), deduplicates them against the
   knowledge graph, dispatches up to `max_subagents` in parallel, then ingests
   results into the graph. A round stops early if it produced no findings above
   0.5 confidence.

3. **Conclusion synthesis** — one LLM call over the accumulated graph.

## Subagent Parallelism

Subagents run as `multiprocessing.Process`, not threads
(`eda_agent.py` around the `mp.Process` construction). Two constraints worth
preserving:

- Results must be read off `result_queue` **before** `p.join()`. On macOS the
  queue is a pipe with a limited buffer; if the child's `put()` blocks on a full
  pipe it can never exit, and `join()` hangs forever.
- Processes are used partly so a wedged subagent can be `terminate()`d. A thread
  cannot be killed, so switching to threads would lose the timeout guarantee.

Each subagent gets its own kernel from `backend/services/kernel_pool.py`, seeded
from a parquet checkpoint of the cleaned dataframe. If the checkpoint is
missing, subagents fall back to the main kernel.

## State

`src/agent/state.py` holds `AgentState`, a mutable dataclass threaded through
the whole run. Almost every field is a primitive; the exception is
`knowledge_graph`, which holds a live `KnowledgeGraph`. That class has
`to_dict()`/`from_dict()`, so the state is serializable with a small custom
encoder if checkpointing is ever added.

Note that serializing `AgentState` does not make a run resumable on its own —
the Jupyter kernel holds `df` in memory, and restoring agent state would leave a
dangling reference. The parquet checkpoint is the seed of a real resume path.

Be careful about size: `node.metadata["plot_images"]` stores base64 PNGs. Any
checkpointing scheme should write images to the session dir and store paths
instead.

## Knowledge Graph

`src/agent/knowledge_graph.py` accumulates typed nodes (`fact`, `hypothesis`,
`evidence`, `conclusion`) with weighted edges. It handles confidence scoring
from statistical evidence, hypothesis dedup by similarity, supersession, and
cross-investigation reinforcement when findings share columns. It is also what
the chat agent and story builder read from.

## Streaming and Steering

Everything the agent does is pushed through the `push_event` callback, forwarded
to the client by `backend/routers/stream.py`. Event types include `cell_write`,
`cell_executing`, `cell_output`, `cell_error`, `thinking`, `phase_transition`,
`plan_update`, and the subagent lifecycle events.

User steering is queued by `backend/services/steering_service.py` and drained
only at safe checkpoints — between goals, after cells, before follow-ups — never
mid-execution.

## LLM Access

`get_chat_model()` supports `openai`, `openai_compatible`, `azure_openai_v1`,
`anthropic`, and `google`. It is `lru_cache`d, so config changes need a process
restart.

LangChain is used **only** as a provider shim: the adapter classes, the
`SystemMessage`/`HumanMessage` types, and `.invoke()`. There are no chains, no
prompt templates, no output parsers, no tool binding. Prompts are f-strings and
structured output is hand-parsed (`json.loads` after stripping markdown fences).
Keep it that way or adopt the abstractions deliberately — don't mix.

## Plots

"plotly" in this codebase is a **MIME type, not a library**. Cells emit
`application/vnd.plotly.v1+json` specs built as plain dicts via
`emit_plot_spec` in `src/agent/code_templates.py`; the frontend renders them.
Plotly is deliberately not a Python dependency. `src/reporting/plot_contract.py`
defines the spec shape.

## Dependencies

`requirements.txt` carries `scipy`, `statsmodels`, and `scikit-learn` even
though no file under `src/` imports them. The agent *generates* Python that runs
in the kernel and reaches for these. Removing them breaks generated cells at
runtime, not at import time — which is much harder to notice. `scipy.stats` is
named explicitly in the subagent prompt.

## History

The original design was a sequential 11-stage LangGraph pipeline driven by
`src/main.py --mode <stage>`, where `--mode` meant "run up to this stage". The
live product replaced it with the agent loop above. The pipeline, its stage
modules, and its two test files were removed once confirmed unreachable from
`backend.app`. If you want that history, it is in the git log before this
commit.
