**Description**

CrewAI is a lean Python framework (built from scratch) for orchestrating “crews” of role‑based agents and event‑driven “flows.” It emphasizes speed, simple ergonomics, and low‑level control when you need it, making it easy to split complex tasks across specialists.

Technologies Used
CrewAI

- Role‑based agents (researcher, analyst, writer, etc.) with custom tools.
- Crews for teamwork; Flows for fine‑grained orchestration.
- Sequential/parallel tasks with automatic dependency handling.
- High‑performance execution; prompt and tool customization.

---

### Project 1: Crew‑Based Iris Analysis
**Difficulty**: 1 (Easy)

**Project Objective**:
A 3‑agent crew (Researcher, Analyst, Writer) performs EDA on Iris and ships a short brief.

**Dataset Suggestions**:
- Dataset: Iris.
- Source: [UCI – Iris](https://archive.ics.uci.edu/dataset/53/iris)

**Tasks**:
- Researcher loads and profiles data; Analyst computes stats; Writer drafts summary.
- Run the crew once; export the write‑up.

**Bonus Ideas (Optional)**:
- Add a Visualizer agent for quick charts.

---

### Project 2: NBA Stats Workflow
**Difficulty**: 2 (Medium)

**Project Objective**:
Crew analyzes NBA player stats for a chosen season and writes storylines about top performers.

**Dataset Suggestions**:
- Dataset: NBA Player Stats (seasonal).
- Sources: [Basketball‑Reference – 2024‑25 Per‑Game](https://www.basketball-reference.com/leagues/NBA_2025_per_game.html) or [Kaggle – 2024/25 Player Stats](https://www.kaggle.com/datasets/eduardopalmieri/nba-player-stats-season-2425)

**Tasks**:
- Engineer agent fetches/cleans data; Analyst computes leaders and advanced metrics; Storyteller writes highlights.
- Parallelize tasks via a Flow; merge results at the end.

**Bonus Ideas (Optional)**:
- Add a Scout agent to analyze rookies vs. veterans.

---

### Project 3: Energy Consumption Orchestrator
**Difficulty**: 3 (Hard)

**Project Objective**:
Crew analyzes household electric power consumption and recommends energy‑saving actions.

**Dataset Suggestions**:
- Dataset: Individual Household Electric Power Consumption.
- Source: [UCI – Household Electric Power Consumption](https://archive.ics.uci.edu/ml/datasets/individual%2Bhousehold%2Belectric%2Bpower%2Bconsumption)

**Tasks**:
- Time‑Series Analyst detects peaks and trends; Device Specialist groups sub‑metering; Recommender drafts actions.
- Combine outputs into a report with estimated cost savings.

**Bonus Ideas (Optional)**:
- Add weather features to explain daily variations.