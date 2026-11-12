# Chapter 1: AI-Augmented Data Science

## 1. The New ML Lifecycle

### 1.1 From Manual to AI-Assisted Workflows

- Traditional ML lifecycle = sequential, manual phases: data collection,
  cleaning, modeling, evaluation, deployment
- AI augmentation = inserting LLMs and agents at each phase to accelerate
  iteration speed and reduce human toil
- Key shift: data scientists move from writing code to reviewing and guiding
  AI-generated code
  - Human role becomes: problem framing, validation, judgment
  - AI role becomes: code generation, pattern matching, boilerplate execution

### 1.2 Phases of the Modern ML Lifecycle

- Data acquisition -> EDA -> feature engineering -> model selection -> training
  -> evaluation -> deployment -> monitoring
- Each phase now has an AI co-pilot counterpart:
  - Data acquisition: AI suggests relevant datasets, scraping strategies
  - EDA: AI generates summary statistics, surface anomalies, proposes
    visualizations
  - Feature engineering: AI proposes transformations based on domain knowledge
  - Model selection: AI benchmarks multiple algorithms
  - Monitoring: AI flags drift and regression automatically

### 1.3 Human-in-the-Loop Principles

- AI output requires human validation at every gate
  - Hallucinations in code: syntactically correct but semantically wrong
  - Statistical errors: AI may misapply methods (e.g., data leakage)
- Roles in the new lifecycle:
  - **Prompter**: frames the problem clearly for the AI
  - **Reviewer**: evaluates AI suggestions critically
  - **Integrator**: assembles AI-generated components into coherent pipelines

- TUTORIAL: mlflow - Track experiments generated with AI assistance; compare
  AI-suggested models vs. baseline; version datasets and parameters

- TUTORIAL: DVC - Version control datasets and models in AI-assisted pipelines;
  reproduce AI-generated experiment runs

---

## 2. LLM Capabilities and Limitations for Data Work

### 2.1 What LLMs Can Do Well

- Code generation: writing pandas transformations, sklearn pipelines, plot
  routines
- Documentation and explanation: summarizing datasets, describing model
  behavior, writing docstrings
- Debugging assistance: identifying likely errors from stack traces and error
  messages
- Brainstorming: proposing feature ideas, modeling strategies, evaluation
  metrics

### 2.2 LLM Limitations for Data Science

- Context window constraints: LLMs cannot ingest entire large datasets or long
  notebooks
  - Workaround: pass summaries, schemas, and samples instead of raw data
- Hallucination of APIs and functions: LLMs may reference deprecated or
  non-existent library methods
  - Workaround: always test generated code; lock library versions
- Lack of causal reasoning: LLMs pattern-match, they do not understand causal
  structure
  - Workaround: apply domain knowledge to validate feature engineering
- No runtime state: LLMs do not execute code and cannot observe actual data
  distributions
  - Workaround: share EDA outputs (describe(), head(), value_counts()) in
    context

### 2.3 Prompt Engineering for Data Work

- Effective prompt patterns for data science tasks:
  - Provide schema + sample rows + task description
  - Ask for code with inline comments explaining each step
  - Request multiple approaches ranked by tradeoff
- Iterative refinement: treat LLM interaction as a dialogue, not a one-shot
  query
  - First pass: generate candidate solution
  - Second pass: review and ask for improvements
  - Third pass: adapt to actual data characteristics

- TUTORIAL: llm - Use the `llm` CLI to query models programmatically from
  notebooks; chain prompts for multi-step data tasks; compare outputs across
  models

- TUTORIAL: Anthropic MCP - Use the Model Context Protocol to give LLMs
  structured access to data tools, APIs, and file systems; build agents that
  retrieve and process data autonomously

- TUTORIAL: Ollama Python - Run local LLMs for private data work; avoid sending
  sensitive data to external APIs; integrate local inference into Python
  notebooks

---

## 3. The Tooling Landscape

### 3.1 LLM Interfaces and Frameworks

- Direct API access: OpenAI API, Anthropic API, Google Gemini API
- Orchestration frameworks: wrap LLM calls with memory, routing, and tool use
  - LangChain: chaining LLM calls with retrievers, memory, and agents
  - LlamaIndex: indexing and querying documents with LLMs

- TUTORIAL: Langchain and Neo4j - Build a data question-answering pipeline;
  connect a graph database to an LLM for schema-aware querying

- TUTORIAL: LlamaIndex - Index a corpus of data science documentation; build a
  RAG (retrieval-augmented generation) assistant that answers questions about
  datasets

### 3.2 AI-Assisted Coding Environments

- IDE integrations: GitHub Copilot, Cursor, VS Code + LLM extensions
- Notebook-native assistants: JupyterAI, marimo AI integration
- Key patterns for effective use:
  - Keep cells small and focused so the AI has clear context
  - Use docstrings and comments to guide AI completion
  - Commit frequently so AI suggestions can be reverted

- TUTORIAL: Papermill - Parameterize notebooks for AI-driven batch execution;
  run the same analysis across many dataset slices automatically

### 3.3 Agentic Tools

- Agents = LLMs that can call tools, execute code, and iterate toward a goal
  autonomously
- Relevant for data science:
  - Data scraping agents
  - Self-debugging code agents
  - AutoML agents that benchmark models
- Current state: agents are useful for bounded tasks; require human supervision
  for open-ended work

- TUTORIAL: AutoGPT - Demonstrate an agentic workflow for a bounded data task
  (e.g., download a dataset, describe it, and train a baseline model); review
  agent decisions at each step

- TUTORIAL: Griptape - Build an AI-powered data pipeline application; wire
  together tools for data loading, transformation, and summarization with
  structured AI reasoning

### 3.4 Model Hubs and Ecosystem

- HuggingFace: central hub for pretrained models, datasets, and demos
  - Use for: embedding models, NLP preprocessing, domain-specific models
- Model selection strategy: choose smallest model that meets accuracy threshold
  - Balance: accuracy vs. inference cost vs. latency

- TUTORIAL: HuggingFace - Load a pretrained embedding model; use it to represent
  tabular text features; compare embeddings vs. TF-IDF for a classification task

---

## 4. Engineering Principles for AI-Assisted Work

### 4.1 Reproducibility

- AI-assisted code is still code: apply the same reproducibility standards
  - Pin library versions in `requirements.txt`
  - Set random seeds explicitly
  - Log all parameters and data snapshots
- LLM-generated code is non-deterministic: document the prompt used to generate
  each artifact
  - Store prompts alongside code in version control

- TUTORIAL: DVC - Demonstrate full pipeline reproducibility; show how to replay
  an AI-assisted experiment end-to-end from raw data to evaluation metrics

### 4.2 Modularity and Testability

- Structure AI-generated code as functions and classes, not monolithic notebooks
  - Enables unit testing of individual components
  - Enables reuse across projects
- Test AI-generated code as rigorously as hand-written code
  - Property-based testing for data transformations
  - Regression tests against known-good outputs

- TUTORIAL: hypothesis - Apply property-based testing to AI-generated data
  transformation functions; automatically discover edge cases that break the
  generated code

- TUTORIAL: Pydantic - Validate inputs and outputs of AI-generated data pipeline
  stages; catch schema violations early before they propagate downstream

### 4.3 Documentation and Communication

- AI can generate documentation from code but humans must validate accuracy
  - AI-generated docstrings may misrepresent edge cases
  - AI-generated READMEs may omit critical setup steps
- Best practice: use AI to draft, human to finalize

- TUTORIAL: DocsGPT - Build a Q&A system over internal project documentation;
  demonstrate how AI can answer questions about a codebase using RAG

### 4.4 Ethics and Accountability

- AI-augmented work does not reduce human accountability
  - The data scientist is responsible for model outputs, not the AI
  - Bias in AI-generated features must be audited by the human
- Key principle: AI amplifies existing skills; it does not replace judgment
- Checklist for AI-assisted projects:
  - Review all AI-generated transformations for statistical correctness
  - Audit feature engineering for protected attribute leakage
  - Validate model outputs on held-out data with human-designed metrics

### 4.5 Workflow Integration

- Practical integration pattern for day-to-day data science:
  - Start each task with a clear problem statement (input for AI prompting)
  - Use AI to generate a first-pass implementation
  - Review and refactor with domain knowledge applied
  - Commit, document, and test before moving to next task
- Avoid: accepting AI output without understanding it
  - The "vibe coding" failure mode: code runs but logic is wrong

- TUTORIAL: hydra-core - Manage configuration for AI-assisted experiments;
  sweep over hyperparameter grids generated by AI suggestions; reproduce any
  run from its config file

- TUTORIAL: Faker - Generate synthetic datasets for testing AI-generated
  pipelines; validate that transformations behave correctly on controlled,
  known-good data before applying to real data
