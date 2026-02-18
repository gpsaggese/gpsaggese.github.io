# BambooAI: Local Agentic Workflows with Minimal Code
*Because “it works on my machine” is not a methodology.*

## Overview
Teams want faster analysis, cleaner workflows, and reusable outputs, but the path from “raw data” to “decision-ready insight” is still annoyingly manual. Modern AI tools can help, but many solutions trade one problem for another: hidden complexity, non-reproducible runs, and opaque outputs that are hard to trust or rerun.

This document describes how BambooAI targets the gap between experimentation and reproducible workflows by keeping the work structured, transparent, and repeatable.

## Why BambooAI exists
Most data work does not fail because people cannot write Python. It fails because the workflow around the analysis is fragile and expensive:

- You repeat the same setup work for every dataset (profiling, cleaning, baseline plots, exports).
- “The result” lives inside a notebook that nobody else can reproduce.
- Environments drift and things break across machines.
- You get answers, but not the **steps**, **code**, and **artifacts** that make the work reusable.

BambooAI targets that gap: it lets you interact with data in natural language while still producing **structured plans**, **executable code**, and **saved outputs**, with an emphasis on repeatable workflows rather than one-off chat responses.

## What BambooAI is (in one sentence)
BambooAI is an open-source, multi-agent library for conversational data discovery and analysis that can generate and execute Python code for analysis and visualization, with optional planning, self-healing error correction, semantic grounding via ontologies, and episodic memory via a vector database.

## How BambooAI actually works
BambooAI is built around an agentic workflow. Instead of one monolithic prompt, different specialized agents handle different parts of the job (for example: planning, code generation, selection/routing, and more). You can configure which models are used for which agents via `LLM_CONFIG.json`.

At a high level, a typical run looks like this:

1. **Interpret the request** in natural language (what the user wants).
2. **Optional planning step** for complex tasks (what should happen, in what order).
3. **Generate executable Python code** for analysis and visualization.
4. **Execute the code** and produce outputs.
5. **Self-heal** if execution fails (debug and correct the code).
6. **Save artifacts** (datasets, results, logs) so work can be reused.

This “plan → code → run → fix → save” loop is why BambooAI feels more like a workflow engine than a chat interface.

## What makes BambooAI different

### Multi-agent design, configurable by role
BambooAI uses a multi-agent setup where agents can be configured with different models and parameters, enabling specialization (for example, one agent for planning, another for code). This is controlled via `LLM_CONFIG.json`.

### Optional planning for complex tasks
You can enable a planning agent so BambooAI proposes steps before execution, which improves auditability and makes it easier to review or steer the workflow.

### Self-healing execution
BambooAI can correct errors during code execution, which is crucial in real analysis where datasets rarely behave nicely.

### Semantic grounding with dataframe ontologies (semantic memory)
You can provide a `.ttl` ontology to ground BambooAI in your domain and data relationships, improving accuracy and relevance of generated solutions.

### Learning from successful work via a vector database (episodic memory)
BambooAI can store highly-rated successful solutions in a vector database and retrieve them later for similar tasks. It supports Pinecone and Qdrant (including local Qdrant).

### Web UI + Jupyter support
BambooAI supports both notebook workflows and a web UI.

## Recommended repo structure for a blog project
This structure keeps the project clean and makes it obvious what is configuration, what is runtime output, and what is your own work:

```text
bambooai/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── README.md
│
├── web_app/
│   ├── .env
│   ├── LLM_CONFIG.json
│   ├── storage/
│   ├── temp/
│   └── logs/
│
├── notebooks/
│   └── bambooai_experiments.ipynb
│
└── datasets/
    └── generated/
        └── ... (auto-created during runs)
```
This aligns with the repository’s web app folder layout and configuration pattern (including sample LLM_CONFIG and prompt template files).

## A concrete mini-example (from my notebook)
To keep this grounded, here’s the simplest prompt I used in my uploaded notebook:

> “Give a summary of the data and perform EDA on it.”

```python
import pandas as pd
from bambooai import BambooAI
from dotenv import load_dotenv

load_dotenv()

df = pd.read_csv("your_dataset.csv")

bamboo = BambooAI(
    df=df,
    planning=True,     # optional planning agent
    vector_db=False,   # episodic memory off for minimal setup
    search_tool=False  # keep runs dataset-local
)

bamboo.pd_agent_converse()
```

Even with a basic request like this, BambooAI behaved like an **agentic workflow**, not a single-shot chatbot. Under the hood, multiple agents effectively take turns doing specialized work. In my run, you could see the impact of four key roles:

### Planner agent (defines the workflow before touching code)
After the prompt, the **planner** broke the task into a sensible sequence: summarize the dataset (shape, columns, missingness, distributions), identify important numeric vs categorical fields, and outline standard EDA checks and visuals.

This matters because it keeps the workflow auditable. Instead of jumping straight into code, it establishes the intent and steps up front.

### Executor agent (runs the actual work)
Once the plan was in place, the **executor** moved from intent to action by generating and running analysis steps. In the notebook output, this translated into:

- computing summary statistics and basic profiling
- producing standard EDA visuals (distributions, counts, etc.)
- performing checks like correlations (where applicable)
- writing outputs and intermediate artifacts to disk (so results weren’t trapped inside the notebook)

This “execute and persist artifacts” behavior is the difference between a demo and something you can reuse.

### Code reviewer agent (catches issues and nudges the code toward correctness)
Real-world EDA is messy. In my run, some operations naturally required guardrails (for example: correlations should only be computed on numeric columns, not strings/categorical fields). That’s where a **code reviewer** role becomes valuable: it enforces correctness patterns and prevents the workflow from collapsing when the dataset isn’t perfectly clean.

Instead of failing and forcing manual fixes, the workflow used review and correction behavior as part of the loop: generate → run → detect issue → refine.

### Summarizer agent (turns raw outputs into readable takeaways)
After execution, the **summarizer** produced a human-readable interpretation of what happened: high-level dataset summary, key distributions, and practical takeaways from the EDA.

This is crucial for communication: stakeholders usually want conclusions and highlights, while analysts want code and artifacts. BambooAI tries to deliver both.

## Why this mini-example matters
The prompt was intentionally simple. The goal was to validate the workflow behavior:

- Plan first (so you can inspect intent)
- Execute (so results are concrete)
- Review and correct (so it doesn’t fall apart on real data)
- Summarize (so outputs are communicable)
- Persist artifacts (so work is reusable)

## Closing thoughts
BambooAI’s appeal is that it supports a disciplined workflow:

- Configurable multi-agent execution
- Optional planning for complex tasks
- Executable code generation and execution
- Self-healing error correction
- Semantic grounding via ontologies
- Memory via vector databases
- Web UI plus notebook support
- Local-first reproducibility with Docker

That combination is what makes it useful beyond demos, and why it’s worth documenting as a workflow pattern rather than a single notebook run.
