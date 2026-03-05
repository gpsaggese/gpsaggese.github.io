# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CrewAI API Overview
#
# CrewAI is an open-source Python framework for building **role-playing,
# autonomous AI agent crews**. Each agent has a defined role, goal, and
# backstory; agents collaborate on tasks and can call custom tools.
#
# This notebook walks through the key building blocks:
# - **LLM** – how to configure a language model (local Ollama or cloud)
# - **Agent** – defining a role-playing agent
# - **Task** – assigning work to an agent
# - **Crew** – assembling agents and tasks into a pipeline
# - **Tools** – extending agents with custom Python functions
# - **Process** – sequential vs. hierarchical execution

# %% [markdown]
# ## Setup

# %%
import logging
import os

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool

import crewai_utils as tcrwuti

logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)

# %% [markdown]
# ## LLM Configuration
#
# CrewAI uses **LiteLLM** under the hood, so any provider supported by
# LiteLLM works out of the box.
#
# - **Local (Ollama):** prefix the model with `"ollama/"` and point
#   `base_url` at the Ollama server.
# - **OpenAI:** set `OPENAI_API_KEY` and use `"gpt-4o"`, etc.
#
# For this tutorial we use the **local Ollama** path so no API key is
# required.

# %%
# Local Ollama LLM – no API key needed.
llm = tcrwuti.get_local_llm(
    model="ollama/gemma3:latest",
    base_url="http://host.docker.internal:11434",
    temperature=0.2,
)
print(f"LLM model: {llm.model}")

# %% [markdown]
# ## Agents
#
# An **Agent** is a role-playing entity with:
# - `role` – job title / persona
# - `goal` – what the agent is trying to achieve
# - `backstory` – background that shapes the agent's behaviour
# - `llm` – the language model to use
# - `tools` – list of callable tools (optional)
# - `verbose` – print execution traces (useful for debugging)

# %%
# A simple summarisation agent.
summarizer = Agent(
    role="Summarizer",
    goal="Produce concise bullet summaries of provided text.",
    backstory="A careful analyst who only outputs essential points.",
    llm=llm,
    verbose=True,
)
print(f"Agent role: {summarizer.role}")

# %%
# An agent that uses tools.
data_analyst = Agent(
    role="Data Analyst",
    goal="Perform lightweight EDA on local CSVs via tools.",
    backstory="Prefers precise, minimal outputs. Uses tools exactly as requested.",
    tools=tcrwuti.EDA_TOOLS,
    llm=llm,
    verbose=True,
)
print(f"Agent tools: {[t.name for t in data_analyst.tools]}")

# %% [markdown]
# ## Tasks
#
# A **Task** describes the work to be done:
# - `description` – detailed instructions for the agent
# - `expected_output` – what a correct answer looks like
# - `agent` – which agent executes the task
#
# Tasks can also declare `context` (a list of upstream Tasks whose output
# is injected into the description at runtime).

# %%
# Create a sample text file for the summariser.
os.makedirs("data", exist_ok=True)
sample_text = (
    "CrewAI lets you define agents with roles, goals, and tools, "
    "then assign tasks. This demo reads this file and outputs a "
    "3-bullet summary."
)
with open("data/sample.txt", "w") as fh:
    fh.write(sample_text)

summarise_task = Task(
    description=(
        f"Summarize the following text into exactly 3 concise bullet "
        f"points:\n\n{sample_text}"
    ),
    expected_output="Exactly three bullet points.",
    agent=summarizer,
)
print("Task created:", summarise_task.description[:60], "...")

# %% [markdown]
# ## Crew
#
# A **Crew** assembles agents and tasks:
# - `agents` – list of Agent instances
# - `tasks` – ordered list of Task instances
# - `process` – `Process.sequential` (default) or `Process.hierarchical`
# - `verbose` – print crew-level logs
#
# Call `crew.kickoff()` to run the pipeline.

# %%
crew = Crew(
    agents=[summarizer],
    tasks=[summarise_task],
    process=Process.sequential,
    verbose=True,
)
# Kick off the crew and capture the result.
result = crew.kickoff()
print("\n=== RESULT ===\n", result)

# %% [markdown]
# ## Tools
#
# Tools extend an agent's capabilities with custom Python functions.
# Decorate any function with `@tool` from `crewai.tools`:
# - The **docstring** becomes the tool description shown to the LLM.
# - Arguments must be type-annotated; CrewAI auto-generates the schema.

# %%
@tool
def word_count(text: str) -> str:
    """Count the number of words in the provided text. Returns a string."""
    count = len(text.split())
    return f"Word count: {count}"


# Attach the tool to a new agent and run a quick test.
counter_agent = Agent(
    role="Word Counter",
    goal="Count words in any text using the word_count tool.",
    backstory="A precise counter that always uses its tool.",
    tools=[word_count],
    llm=llm,
    verbose=True,
)

count_task = Task(
    description="Use the word_count tool on: 'Hello world this is CrewAI'",
    expected_output="Word count as an integer.",
    agent=counter_agent,
)

counter_crew = Crew(
    agents=[counter_agent],
    tasks=[count_task],
    process=Process.sequential,
    verbose=True,
)
count_result = counter_crew.kickoff()
print("\n=== COUNT RESULT ===\n", count_result)

# %% [markdown]
# ## Process: Sequential vs Hierarchical
#
# ### Sequential (default)
# Tasks run **one after another** in the order listed. The output of each
# task can optionally be injected into the next via `context`.
#
# ### Hierarchical
# A **manager agent** (auto-created or explicitly set via `manager_llm`)
# plans which agent tackles each task and in what order. Use this when the
# workflow is complex or not fully determined upfront.

# %%
# Example: two-task sequential pipeline with context passing.
task_a = Task(
    description="List exactly 3 facts about Python programming language.",
    expected_output="Three bullet points about Python.",
    agent=summarizer,
)

task_b = Task(
    description=(
        "Given the facts above, write one sentence explaining "
        "why Python is popular."
    ),
    expected_output="One sentence.",
    agent=summarizer,
    context=[task_a],  # inject task_a output into task_b description
)

pipeline_crew = Crew(
    agents=[summarizer],
    tasks=[task_a, task_b],
    process=Process.sequential,
    verbose=True,
)
pipeline_result = pipeline_crew.kickoff()
print("\n=== PIPELINE RESULT ===\n", pipeline_result)

# %% [markdown]
# ## Memory and Context Sharing
#
# CrewAI supports **short-term memory** (shared within a run) and
# **long-term memory** (persisted across runs using embeddings).
#
# Enable memory by passing `memory=True` to the Crew constructor.
# The default embedding backend is OpenAI; switch to a local embedder
# via the `embedder` parameter for fully offline use.

# %%
# Memory-enabled crew example (uses in-process short-term store).
memory_crew = Crew(
    agents=[summarizer],
    tasks=[summarise_task],
    process=Process.sequential,
    memory=True,   # enables short-term shared memory
    verbose=False,
)
print("Memory crew created with memory=True")
