# Learn CrewAI in 60 Minutes
## Structured Multi Agent AI Systems with Planning, Execution, and Critique

---

# 1. Goal

This tutorial provides everything needed to become familiar with **CrewAI** in 60 minutes.

CrewAI is a framework for building **collaborative multiagent AI systems** where agents with specialized roles plan, execute, critique, and refine tasks iteratively.

By the end, you will move from:

Single LLM prompts → Structured autonomous AI systems

---

# 2. What This Tutorial Provides

## Hands on Experience

- Working Docker container
- Reproducible notebook (`tutorial_crewai.ipynb`)
- End to end runnable example
- Multiagent planning loop

## Conceptual Understanding

You will understand:

- What CrewAI is
- What problem it solves
- Native API structure
- When to use multiagent systems
- Alternatives and trade-offs

## Practical Application

You can:

- Build a Planner -> Worker -> Critic loop
- Execute notebook code through agents
- Implement retry logic
- Track explicit state

## Self Sufficiency

Everything runs inside Docker.
All dependencies are pinned.
No local environment conflicts.

## Reproducibility

- Notebook runs end to end
- Dependencies are managed via the container environment
- Takes ~ 3 minutes to execute

---

# 3. Structure

## Setup

- Clone repository
- Build Docker container
- Start Jupyter
- Verify environment

## API Exploration

Work through:

`tutorial_crewai.ipynb`

Learn:

- Creating agents
- Defining tasks
- Running a crew
- How to give an agent tools
- How an agent calls Python functions

---

# 4. Docker Container

The Docker container:

- Installs CrewAI and dependencies
- Installs nbclient
- Installs Jupyter
- Pins all versions
- Runs notebook tests

Build container:

```bash
docker compose build
docker compose up
```

# 5. What is CrewAI?

CrewAI is a Python framework for building structured multi agent AI systems.

Instead of:
 
```bash
prompt -> response
```

CrewAI enables:

```bash
Planner -> Worker -> Critic -> Update -> Repeat
```
It supports role specialization, task delegation, tool usage, explicit coordination, iterative refinement

# 6. What Problem Does It Solve?

Single LLM calls lack iterative correction, error recovery, explicit state, structured decomposition, and deterministic ochestration.

CrewAI introduces modular agents, explicit tasks, structured workflows, and observable state transitions.

# 7. Native API Overview
Core abstractions: 
Agent: 

```bash
Agent(
    role="Planner",
    goal="Plan next notebook step",
    backstory="Expert AI planner"
)
```

Task:

```bash
Task(
    description="Execute notebook cell",
    agent=worker
)
```

Crew:

```bash
Crew(
    agents=[planner, worker, critic],
    tasks=[task]
)
```

# 8. State Management

State includes: Objective, Execution history, Notebook outputs, Error traces, Completion flag

Explicit state ensures: Reproducibility, Debuggability, Deterministic behavior, Observability

Avoid hidden memory.