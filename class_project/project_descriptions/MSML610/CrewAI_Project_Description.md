# Description

CrewAI is a lean Python framework (built from scratch) for orchestrating "crews" of
role-based agents and event-driven "flows". It solves the problem of splitting a
complex task across specialist agents, with low-level control when needed. It is
worth a 60-minute tutorial because a working multi-agent pipeline takes a few dozen
lines, and the same primitives scale from a three-agent brief to parallel workflows.

## Technologies Used

CrewAI

- Role-based agents (researcher, analyst, writer, etc.) with custom tools
- Crews for teamwork; Flows for fine-grained orchestration
- Sequential/parallel tasks with automatic dependency handling
- High-performance execution; prompt and tool customization

# Tutorial

- Usual tutorial "Learn CrewAI in 60 mins", following
  `.claude/skills/tutorials_in_60_mins.rules.md`
- Start from the existing `tutorials/CrewAI/` and make it better
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Compare with `tutorials/LangChain/`, `tutorials/LangGraph/`, and
  `tutorials/Autogen/` from the multi-agent orchestration point of view, and do the
  same clean up for them if you want
- Deliverables:
  - `crewai_utils.py`
  - `crewai.API.ipynb`
  - `crewai.example.ipynb`

# Project

## Project 1: Iris EDA Crew

- **Difficulty**: 1 (Easy)
- **Project Objective**: A 3-agent crew (Researcher, Analyst, Writer) performs EDA on
  Iris and ships a short brief whose numbers match the data
- **Dataset Suggestions**: [UCI - Iris](https://archive.ics.uci.edu/dataset/53/iris)
- **Tasks**:
  - **Load the Data**: Give the Researcher a custom tool that loads Iris and returns
    a profile (shape, types, missing values, class balance)
  - **Define the Crew**: Define the Researcher, Analyst, and Writer as `Agent`
    objects with role, goal, and backstory, and one `Task` each with an expected
    output
  - **Run the Crew**: Run the `Crew` with `Process.sequential` through `kickoff()`,
    with the Analyst calling a custom tool for per-class statistics and correlations
  - **Evaluate the Brief**: Recompute the statistics with pandas and report the
    fraction of numeric claims in the brief that match within rounding
  - **Export the Brief**: Export the write-up as a Markdown file with a table and a
    pairplot
- **Bonus Ideas (Optional)**: Add a Visualizer agent for quick charts

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Load the Data
  - Result: `tutorials/CrewAI/` container running, and the Iris profile table
    produced by the custom tool
- Milestone 2: API notebook
  - Project tasks: Define the Crew, Run the Crew
  - Result: `crewai.API.ipynb` covering role-based agents with custom tools, and
    sequential and parallel tasks
- Milestone 3: Example notebook
  - Project tasks: Define the Crew, Run the Crew, Evaluate the Brief, Export the
    Brief
  - Result: `crewai.example.ipynb` running end to end and exporting the brief

## Project 2: NBA Stats Workflow

- **Difficulty**: 2 (Medium)
- **Project Objective**: Crew analyzes NBA player stats for a chosen season and
  writes storylines about top performers
- **Dataset Suggestions**: NBA Player Stats (seasonal)
  - [Basketball-Reference - 2024-25 Per-Game](https://www.basketball-reference.com/leagues/NBA_2025_per_game.html)
  - [Kaggle - 2024/25 Player Stats](https://www.kaggle.com/datasets/eduardopalmieri/nba-player-stats-season-2425)
- **Tasks**:
  - **Ingest and Clean the Data**: Engineer agent fetches and cleans the player stats
  - **Compute Leaders**: Analyst computes category leaders and advanced metrics
  - **Write Storylines**: Storyteller writes highlights for the top performers
  - **Orchestrate with a Flow**: Parallelize the tasks via a `Flow` and merge the
    results at the end
- **Bonus Ideas (Optional)**: Add a Scout agent to analyze rookies vs. veterans

## Project 3: Energy Consumption Orchestrator

- **Difficulty**: 3 (Hard)
- **Project Objective**: Crew analyzes household electric power consumption and
  recommends energy-saving actions
- **Dataset Suggestions**: Individual Household Electric Power Consumption
  - [UCI - Household Electric Power Consumption](https://archive.ics.uci.edu/ml/datasets/individual%2Bhousehold%2Belectric%2Bpower%2Bconsumption)
- **Tasks**:
  - **Detect Peaks and Trends**: Time-Series Analyst detects peaks and trends
  - **Group Sub-Metering**: Device Specialist groups the sub-metering channels
  - **Draft Actions**: Recommender drafts energy-saving actions
  - **Report Savings**: Combine the outputs into a report with estimated cost savings
- **Bonus Ideas (Optional)**: Add weather features to explain daily variations
