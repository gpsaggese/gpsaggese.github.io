# Chapter 2: AI-Assisted Python and ML Coding

## 1. LLM-Assisted Code Generation

### 1.1 The Code Generation Workflow

- Core idea: describe intent in natural language, receive working Python code as
  output
- Effective inputs for code generation:
  - Schema description (column names, types, cardinality)
  - Sample rows or `df.head()` output pasted into the prompt
  - Explicit statement of goal: transform, filter, aggregate, model
- Common generated artifacts:
  - Data loading and parsing scripts
  - Pandas transformation chains
  - sklearn pipelines end-to-end
  - Plotting routines with seaborn/matplotlib

### 1.2 Code Generation Patterns for ML

- Scaffold generation: AI writes the full skeleton, human fills in domain
  specifics
  - Example: generate a cross-validation loop with early stopping; adapt
    hyperparameters manually
- Boilerplate elimination: AI handles repetitive setup code
  - argparse definitions, logging setup, file I/O wrappers
- API translation: convert pseudocode or R/MATLAB idioms to Python
  - Example: translate a statistical formula into numpy/scipy calls

- TUTORIAL: llm - Use `llm` to generate sklearn pipeline code from a dataset
  description; compare outputs from different models (GPT-4, Claude, local
  Ollama); iterate with follow-up prompts to refine the pipeline

- TUTORIAL: Anthropic MCP - Use the Model Context Protocol to give the LLM
  direct access to a Python environment; generate code that is immediately
  executed and results fed back into the conversation for iterative refinement

### 1.3 Low-Code and AutoML Code Generation

- AI can generate low-code ML workflows that run end-to-end with minimal
  boilerplate
- Distinction between AutoML and AI-generated code:
  - AutoML: automated search over algorithms and hyperparameters
  - AI-generated code: human-readable pipeline that the data scientist can
    inspect and modify

- TUTORIAL: pycaret - Generate a complete ML comparison pipeline for a tabular
  dataset using PyCaret; use AI to explain each step of the generated code;
  extend the pipeline with a custom transformer

### 1.4 Code Generation Pitfalls

- Silent correctness errors: code runs without exceptions but produces wrong
  answers
  - Data leakage: scaler fitted on full dataset before split
  - Wrong axis: mean computed across rows instead of columns
  - Index misalignment: pandas index not reset after filtering
- Stale API usage: LLMs trained on older data may generate deprecated calls
  - Mitigation: specify library version in the prompt
- Non-idiomatic code: AI code is often verbose or bypasses vectorization
  - Mitigation: ask explicitly for "vectorized pandas" or "numpy-only" solution

---

## 2. Code Review and Debugging with AI

### 2.1 AI-Assisted Code Review

- Use LLMs to review code for correctness, not just style
  - Ask: "Review this transformation for data leakage"
  - Ask: "Does this pipeline have any statistical errors?"
  - Ask: "What edge cases does this function not handle?"
- Structured review prompts:
  - Provide the function + a description of the data it will receive
  - Request a numbered list of potential issues
  - Ask for a corrected version after reviewing the issues

### 2.2 Debugging with LLMs

- Debugging workflow:
  1. Paste the full traceback and the relevant code block
  2. Include the data schema and sample that triggered the error
  3. Ask for root cause analysis, not just the fix
- Common ML debugging tasks that LLMs handle well:
  - Shape mismatch errors in numpy/PyTorch
  - Pandas SettingWithCopyWarning root cause
  - Serialization errors in model saving
- LLM limitations in debugging:
  - Cannot inspect actual runtime state; needs representative input
  - May suggest workarounds rather than root cause fixes

- TUTORIAL: SWE-agent - Demonstrate an autonomous debugging agent that reads
  a failing test, identifies the root cause in a data pipeline, and proposes a
  fix; review the agent's reasoning trace step by step

### 2.3 Static Analysis and Linting Integration

- AI complements, but does not replace, static analysis tools
  - Ruff/Flake8: syntax and style errors (fast, deterministic)
  - Pyright/mypy: type errors (fast, deterministic)
  - AI review: semantic and statistical errors (slow, probabilistic)
- Recommended workflow:
  - Run linter first to eliminate trivial issues
  - Then prompt AI with clean code for deeper review

- TUTORIAL: hypothesis - Use property-based testing to automatically find bugs
  in AI-generated data transformation code; define invariants (e.g., output
  row count equals input, no NaN introduced) and let Hypothesis find
  counterexamples

### 2.4 Iterative Debugging Loops

- Treat AI debugging as a conversation:
  - Iteration 1: provide error + code -> get hypothesis about root cause
  - Iteration 2: test hypothesis -> share result with AI
  - Iteration 3: confirm fix or escalate to a different approach
- When to stop and debug manually:
  - After 3 AI iterations without convergence
  - When AI begins to contradict its earlier suggestions
  - When the error requires runtime inspection (pdb/ipdb)

---

## 3. Documentation and Testing

### 3.1 AI-Generated Documentation

- AI can draft documentation faster than humans but requires human validation
  - Docstrings: AI excels at describing parameters and return types
  - README sections: AI can summarize purpose and usage examples
  - Inline comments: AI explains what code does; human verifies it is correct
- Quality check for AI-generated docstrings:
  - Verify that described behavior matches actual implementation
  - Ensure edge cases are mentioned (not just the happy path)
  - Confirm that parameter types and return types are accurate

- TUTORIAL: DocsGPT - Index a data science project's documentation; build an
  internal Q&A system where team members query the docs in natural language;
  demonstrate how this reduces onboarding time

### 3.2 AI-Assisted Test Generation

- LLMs can generate test scaffolds that cover:
  - Happy path with typical inputs
  - Edge cases: empty DataFrame, single-row input, all-null column
  - Type and shape assertions
- Workflow for AI test generation:
  1. Provide the function signature and docstring
  2. Ask for 5 unit tests covering different scenarios
  3. Review tests for correctness and add missing scenarios

- TUTORIAL: hypothesis - Generate property-based tests for ML preprocessing
  functions; use strategies to generate random DataFrames; assert invariants
  that must hold for any valid input

- TUTORIAL: Faker - Generate realistic synthetic data for test fixtures; create
  DataFrames with plausible distributions for names, dates, prices, and
  categorical fields; use in pytest fixtures shared across test files

### 3.3 Schema Validation as Living Documentation

- Schema validation serves dual purpose: runtime safety and documentation
  - Defines expected types and constraints explicitly in code
  - Catches breaking changes when upstream data changes schema
- Pydantic as schema documentation:
  - Define data models that describe inputs to each pipeline stage
  - Serve as machine-readable contracts between pipeline components

- TUTORIAL: Pydantic - Define Pydantic models for the input and output of each
  stage in an ML pipeline; demonstrate validation errors caught at runtime;
  show how models serve as living documentation that stays in sync with code

### 3.4 Notebook Documentation Patterns

- Notebooks are the primary documentation artifact in data science
- AI-assisted notebook documentation:
  - Generate markdown cells that explain code cells
  - Summarize findings from output cells in prose
  - Create section headers that map to the analysis narrative
- Best practices:
  - Every notebook should have a "Purpose and Scope" cell at the top
  - Every major code block should have a preceding markdown cell with intent
  - Conclusions cell at the bottom with key findings

- TUTORIAL: Papermill - Use AI to generate a parameterized notebook template;
  execute the same analysis notebook across multiple datasets automatically;
  collect results across runs into a summary DataFrame

---

## 4. Prompt Engineering for Data Scientists

### 4.1 Anatomy of an Effective Data Science Prompt

- Components of a high-quality prompt:
  - **Role**: "You are an expert Python data scientist"
  - **Context**: data schema, column descriptions, size, problem domain
  - **Task**: specific transformation, model, or analysis to implement
  - **Constraints**: library versions, performance requirements, output format
  - **Examples**: sample input/output pairs when the task is ambiguous
- Principle: the more context provided, the less hallucination

### 4.2 Prompt Patterns for Common Tasks

- **Schema-to-code pattern**: paste `df.dtypes` and `df.describe()` and ask for
  a specific transformation
  ```
  Given a DataFrame with these columns and statistics: [paste output]
  Write a function that normalizes all numeric columns using min-max scaling,
  handling NaN values by imputing with the column median before scaling.
  ```
- **Refactor pattern**: ask AI to improve existing code
  ```
  Refactor this function to be vectorized (no Python loops) and handle
  edge cases for empty DataFrames:
  [paste code]
  ```
- **Explain pattern**: ask AI to explain unfamiliar code
  ```
  Explain this code line by line, describing what each operation does to
  the DataFrame and why:
  [paste code]
  ```

### 4.3 Context Management Strategies

- LLM context windows are limited: manage context deliberately
  - Send schemas, not full DataFrames
  - Send `df.describe()` output, not raw data
  - Summarize previous conversation steps when continuing a long session
- Multi-turn strategies:
  - Break large tasks into subtasks; prompt for each independently
  - Use chain-of-thought: ask AI to reason before generating code
  - Ask for pseudocode first, then implementation

- TUTORIAL: Langchain and Neo4j - Build a conversational data assistant that
  maintains multi-turn context about a dataset; use memory to preserve schema
  information across multiple queries without re-sending it each time

### 4.4 Prompt Templates for ML Pipelines

- Reusable prompt templates reduce repetition and improve consistency
  - Create templates for: EDA summary, feature engineering, model comparison,
    evaluation report
  - Parameterize templates with dataset name, target variable, and constraints
- Template example for model selection:
  ```
  Dataset: {dataset_description}
  Target variable: {target} (type: {target_type})
  Constraints: {constraints}
  Task: Recommend 3 ML algorithms ranked by expected performance.
  For each, provide: (1) why it fits this problem, (2) key hyperparameters
  to tune, (3) sklearn code to instantiate and cross-validate.
  ```

- TUTORIAL: hydra-core - Manage prompt templates as configuration files; version
  control different prompt variants; run systematic experiments to compare
  prompt effectiveness across model families

### 4.5 Evaluating AI-Generated Code Quality

- Correctness checklist for AI-generated ML code:
  - Train/test split happens before any fitting
  - No target variable in feature set
  - Categorical encoding fitted only on train, applied to test
  - Evaluation metric matches the business objective
  - Random seeds set for reproducibility
- Automated quality checks:
  - Run the code on a small synthetic dataset to verify it executes
  - Compare output statistics to expected ranges
  - Write at least one unit test before committing

- TUTORIAL: SDV (Synthetic Data Vault) - Generate synthetic tabular data that
  mimics real dataset statistics; use it to test AI-generated pipelines safely
  without exposing sensitive data; validate that pipelines produce correct
  output distributions on controlled synthetic inputs
