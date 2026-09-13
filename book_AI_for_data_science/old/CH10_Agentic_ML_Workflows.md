# Chapter 10: Agentic ML Workflows

## 1. LLM Agents and Tool Use

### 1.1 What Makes an LLM Agent

- An LLM agent is an LLM that can:
  - **Reason**: decompose complex goals into sub-tasks
  - **Act**: call tools (code execution, database queries, web search, API calls)
  - **Observe**: receive tool results and update its plan
  - **Iterate**: repeat the reason-act-observe loop until the goal is achieved
- Single-turn prompting = one response; agents = multi-step, goal-directed loops
- Key property: agents can take actions with real-world side effects

### 1.2 Agent Architectures

- **ReAct** (Reasoning + Acting): agent alternates between reasoning steps
  (Thought:) and action steps (Action:) until the task is complete
  - Transparent: the reasoning trace shows why each tool was called
  - Well-suited for data science tasks with multiple investigation steps
- **Plan-and-Execute**: agent first creates a complete plan, then executes it
  step by step
  - Better for long-horizon tasks where the full sequence is known upfront
- **Reflection**: agent critiques its own outputs and iterates
  - Reflexion framework: agent writes a verbal self-critique after each failure

### 1.3 Tools for Data Science Agents

- Relevant tools for ML agents:
  - Python REPL: execute code and observe output
  - Database connector: query SQL or graph databases
  - Search: retrieve documentation or prior work
  - File I/O: read datasets, write results
  - MLflow API: log experiments, retrieve model metrics
- Agent frameworks: LangChain, Griptape, AutoGPT, Haystack agents

- TUTORIAL: Langchain and Neo4j - Build a tool-using LLM agent that can query a
  database, run Python code, and search documentation; demonstrate how agents
  decompose a complex data task into tool calls

- TUTORIAL: ReAct - Implement a ReAct-style agent that interleaves reasoning and
  action for a data science task; compare its performance to a single-shot
  prompt on multi-step problems

- TUTORIAL: txtai - Build a semantic search tool for an ML agent that retrieves
  relevant notebooks and documentation from a local knowledge base; integrate
  the search tool into a LangChain agent

---

## 2. Agentic Data Pipelines

### 2.1 When Agents Replace Fixed Pipelines

- Fixed pipelines: deterministic sequence of steps defined at design time
- Agentic pipelines: agent decides which steps to execute based on observations
  - Useful when:
    - The optimal processing path depends on data characteristics
    - Multiple data sources need to be combined dynamically
    - The pipeline needs to handle unexpected data quality issues
  - Risks:
    - Less predictable than fixed pipelines
    - Harder to debug and monitor
    - May take suboptimal actions in novel situations

### 2.2 Agentic Data Collection and Transformation

- Data collection agent:
  - Decides which APIs to query based on the analysis goal
  - Handles authentication, pagination, and rate limiting
  - Adapts to API changes without code updates
- Transformation agent:
  - Proposes and applies transformations based on data quality assessment
  - Validates transformations by comparing summary statistics before/after

### 2.3 Practical Boundaries of Agentic Pipelines

- Agents are reliable for bounded, well-defined tasks:
  - "Download data from API X, clean it using these rules, and save as parquet"
- Agents are unreliable for open-ended tasks:
  - "Analyze this dataset and build the best possible model"
  - Without constraints, agents may make poor choices or loop indefinitely
- Best practice: break large agentic tasks into bounded sub-tasks with
  human checkpoints at each stage

- TUTORIAL: Griptape - Build an agentic data pipeline where an AI agent decides
  which data sources to query, transforms the results, and writes a summary
  report without human intervention

- TUTORIAL: AutoGPT - Demonstrate an autonomous agent completing a bounded ML
  task end-to-end; review each agent decision to understand where human
  oversight is still required

- TUTORIAL: Haystack - Build an agentic retrieval-and-generation pipeline where
  an agent selects the most relevant data sources based on the query type and
  routes to the appropriate retrieval strategy

---

## 3. Agentic Experiment Orchestration

### 3.1 Closing the Experiment Loop

- Traditional experiment workflow: human designs experiment -> runs it ->
  analyzes results -> designs next experiment
- Agentic loop: agent designs, runs, analyzes, and proposes the next experiment
  automatically
  - Requires: programmatic access to experiment tracking (MLflow, W&B)
  - Requires: the ability to execute training scripts and retrieve results
  - Human role: approve agent's proposed next experiment before execution

### 3.2 Automated Research Exploration

- Use case: explore a large hyperparameter space or architecture search space
  more efficiently than grid search
  - Agent observes partial results and skips configurations that are unlikely
    to be optimal (analogous to Bayesian optimization but with reasoning)
- Risk: confirmation bias in agent reasoning; agent may over-invest in a
  promising-looking direction that is actually a local optimum
- Mitigation: require the agent to maintain a "diversity budget" that forces
  exploration

### 3.3 Agent Self-Improvement via Reflection

- Reflexion: agent writes a verbal reflection after each failed attempt,
  stores it in memory, and uses it to avoid repeating the same mistake
  - Applied to ML: agent reflects on why a model failed (e.g., "data leakage
    was present because I did not split before fitting the scaler") and
    avoids the same error in subsequent runs

- TUTORIAL: mlflow - Build an agent that reads experiment results from MLflow
  and automatically proposes and launches the next experiment based on current
  findings

- TUTORIAL: Reflexion - Apply the Reflexion framework to an agent that debugs a
  failing ML pipeline; demonstrate how self-reflection improves success rate
  across multiple agent runs

- TUTORIAL: SWE-agent - Use SWE-agent to autonomously debug and fix a failing
  data pipeline; review the agent's step-by-step reasoning to understand its
  diagnostic process

---

## 4. Self-Improving ML Systems

### 4.1 Active Learning: Humans in the Loop

- Active learning: the model queries the human for labels on the examples it
  is most uncertain about
  - Key insight: not all labels are equally valuable; uncertain examples are
    more informative
  - Reduces labeling cost while maintaining model quality
- Query strategies:
  - Uncertainty sampling: query examples where model confidence is lowest
  - Query by committee: query examples where multiple models disagree
  - Expected model change: query examples that would change model parameters
    most

### 4.2 Online Learning and Continuous Adaptation

- Online learning = update the model parameters as each new example arrives
  - No full retraining required
  - Adapts to distribution shifts automatically
- River: pure online ML library; every estimator has `learn_one()` and
  `predict_one()` methods
- Use cases: recommendation systems, fraud detection, real-time pricing

### 4.3 Bandit-Based Experimentation

- Multi-armed bandit: allocate traffic between model variants based on observed
  performance
  - Unlike A/B testing, bandits adapt: more traffic goes to the better variant
    as evidence accumulates
- Ax platform: supports bandit-based adaptive experimentation

- TUTORIAL: Ax - Build a self-improving pipeline where Ax automatically
  schedules experiments based on observed performance; use AI to interpret the
  optimization trajectory and suggest termination criteria

- TUTORIAL: modAL - Implement an active learning loop where the model queries
  human labels only for the most uncertain examples; use AI to generate the
  query strategy from a description of the labeling cost constraints

- TUTORIAL: River - Build an online learning system with River that continuously
  updates a fraud detection model on incoming transactions; use AI to select the
  appropriate concept drift detector and retraining threshold
