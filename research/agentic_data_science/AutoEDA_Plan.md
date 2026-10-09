# AutoEDA Agent - Implementation Plan

## Overview

This plan breaks down the AutoEDA agent specification into EPICs and Issues for
team implementation. The agent is a Jupyter-native, agentic framework for
autonomous data analysis using LangGraph for orchestration.

These EPICs were also tracked as real GitHub issues on project board
[AutoML-v1.3 - AgenticEDA](https://github.com/orgs/causify-ai/projects/116)
(16 items, snapshot 2026-09-17). This doc is now self-sufficient — the live
project does not need to be checked separately (see the Appendix at the
bottom for the full cross-reference). **Callout:** all real merged AutoEDA
code lives in `causify-ai/tutorials/agentic_eda/`, not `helpers/hagentic_eda/`
as assumed throughout EPICs 1-7 below — see Issue 1.2's status note.

---

## EPIC 0: Research & Framework Selection

**Goal:** Survey the state of the art in agentic EDA and pick the
orchestration framework before building.

**Owner:** andresryes, PranavShashidhara, madhurlak0810, indrayudd,
HarshitGadge, aangelo9

### Issue 0.1: Investigate SOTA in Agentic EDA
**Status:** 🔶 Research done in GitHub comments
(https://github.com/causify-ai/tutorials/issues/666), never consolidated
into a standalone doc as the issue requested.
- Papers found: VLDB cross-domain automated-EDA survey
  (https://www.vldb.org/pvldb/vol18/p5086-zhu.pdf), QUIS
  (https://arxiv.org/html/2410.10270v1), InsightPilot
  (https://www.microsoft.com/en-us/research/wp-content/uploads/2023/12/InsightPilot.pdf)
  wrapping XInsight, QuickInsights, MetaInsight; benchmark DataSciBench
  (https://datascibench.github.io)
- Framework decision: **LangGraph** (confirms EPIC 1/3 choice below)
- Proposed architecture: LLM Planner + Schema Profiler -> Insight Engine
  (QuickInsights -> MetaInsight -> XInsight) -> Final Report Builder
- Proposed v0 scope: multivariate time series input, notebook output
- **Acceptance Criteria:**
  - ✅ Framework selected
  - ~~Consolidated write-up~~ — not done, findings live only in issue comments

---

## EPIC 1: Foundation & Infrastructure Setup

**Goal:** Establish the core infrastructure and dependencies needed for the AutoEDA agent.

**Owner:** DevOps/Infrastructure Team

### Issue 1.1: Set up LangGraph and LangChain integration
**Status:** 🔶 Proven via a standalone tutorial
(https://github.com/causify-ai/tutorials/issues/609, done, merged
[PR #614](https://github.com/causify-ai/tutorials/pull/614)) but never
integrated as a `helpers` dependency/module as this issue specifies. That
tutorial's directory (`tutorial_langgraph/`) was later deleted; surviving
LangGraph example code lives in `tutorials/langchain_reference/graphs/` and
`tutorials/agentic_eda/simple_pemdas_agent/`. Schema-parsing counterpart:
https://github.com/causify-ai/helpers/issues/986 has two unmerged draft PRs
([#989](https://github.com/causify-ai/helpers/pull/989),
[#991](https://github.com/causify-ai/helpers/pull/991)); duplicate spec at
https://github.com/causify-ai/tutorials/issues/655 has no work against it.
- Add LangGraph and LangChain to project dependencies
- Create base module: `helpers/lang_graph_setup.py`
- Document version compatibility and requirements
- **Acceptance Criteria:**
  - Dependencies installed and tested
  - Example LangGraph state and graph work correctly

### Issue 1.2: Define core module structure for AutoEDA
**Status:** ✅ Done, but at a different location than planned. Real
consolidated code is at `causify-ai/tutorials/agentic_eda/` (via
https://github.com/causify-ai/helpers/issues/997, done, and
https://github.com/causify-ai/tutorials/issues/628, done, merged
[PR #641](https://github.com/causify-ai/tutorials/pull/641)) — `import
helpers.hagentic_eda` was never achieved; the `helpers/hagentic_eda/`
package below was never created.
- Create `helpers/hagentic_eda/` package with submodules
- Submodules: `state.py`, `graph.py`, `tools.py`, `prompts.py`, `utils.py`
- Create `__init__.py` with public exports
- **Acceptance Criteria:**
  - Module imports correctly
  - All submodules are discoverable

### Issue 1.3: Create integration tests setup
**Status:** ✅ CI/test plumbing done at the repo level via
https://github.com/causify-ai/helpers/issues/998 (done, merged
[PR #638](https://github.com/causify-ai/tutorials/pull/638), copied
`helpers`' reusable CI workflows into `tutorials/.github/workflows/`); the
fixtures/sample-dataset tasks below are not done.
- Add pytest fixtures for AutoEDA agent testing
- Set up test data directory with sample datasets
- Create utility functions for test assertions
- **Acceptance Criteria:**
  - Fixtures work with pytest
  - Sample datasets load correctly

---

## EPIC 2: Agent State Management

**Goal:** Design and implement the agent state schema for tracking analysis progress, dataset context, and execution history.

**Owner:** Data Engineering Team

**Status:** ⬜ Not started.

### Issue 2.1: Define AgentState schema with Pydantic
**File:** `helpers/hagentic_eda/state.py`
- Create `AgentState` class with fields:
  - `conversation_history`: List of messages
  - `dataset_context`: DataFrame metadata
  - `analysis_phase`: Current phase (setup, cleaning, analysis, reporting)
  - `generated_cells`: List of executed notebook cells
  - `error_log`: Error tracking for self-correction
- Use Pydantic for validation
- **Acceptance Criteria:**
  - State instantiates correctly
  - All fields have proper type hints
  - Validation catches invalid inputs

### Issue 2.2: Define supporting schema classes
**File:** `helpers/hagentic_eda/state.py`
- Create `DataFrameInfo`: shape, columns, dtypes, null_counts, sample_values
- Create `ColumnInfo`: name, dtype, detected_type, statistics, issues
- Create `NotebookCell`: id, type (code/markdown), content, result, error
- **Acceptance Criteria:**
  - Classes serialize/deserialize correctly
  - Example instances created successfully

### Issue 2.3: Implement state serialization utilities
**File:** `helpers/hagentic_eda/state.py`
- Create `serialize_state()` function for persistence
- Create `deserialize_state()` function for recovery
- Test with sample state objects
- **Acceptance Criteria:**
  - State round-trips correctly (serialize -> deserialize)
  - JSON output is readable

### Issue 2.4: Add state management utilities
**File:** `helpers/hagentic_eda/state.py`
- Implement `update_dataset_context()` to infer schema
- Implement `log_error()` for error tracking
- Implement `add_cell_result()` for cell execution tracking
- Write unit tests
- **Acceptance Criteria:**
  - Unit tests pass
  - Methods update state correctly

---

## EPIC 3: LangGraph Agent Orchestration

**Goal:** Define and implement the LangGraph state machine that orchestrates agent reasoning, tool use, and code generation.

**Owner:** Agent Core Team

**Status:** ⬜ Not started against this exact design.
[PR #989](https://github.com/causify-ai/helpers/pull/989)'s `graph.py`
(from helpers#986, unmerged) is the closest real analog.

### Issue 3.1: Define graph nodes
**File:** `helpers/hagentic_eda/graph.py`
- Implement `agent_node`: Main reasoning with LLM
- Implement `tools_node`: Execute LangChain tools
- Implement `generate_code_node`: Extract code from agent response
- Implement `update_context_node`: Update dataset understanding from results
- Implement `handle_error_node`: Process and fix execution errors
- **Acceptance Criteria:**
  - All nodes implement proper signatures
  - State transitions work correctly

### Issue 3.2: Define graph edges and routing logic
**File:** `helpers/hagentic_eda/graph.py`
- Create `should_use_tools()` conditional function
- Create `should_generate_code()` conditional function
- Create `should_handle_error()` conditional function
- Connect: START -> agent -> tools/generate_code/END
- Connect: tools -> agent, generate_code -> INTERRUPT, RESUME -> update_context
- **Acceptance Criteria:**
  - All edges connect correctly
  - Conditional routing works with test inputs

### Issue 3.3: Implement interrupt/resume for execution feedback
**File:** `helpers/hagentic_eda/graph.py`
- Implement `interrupt_after` configuration for generate_code node
- Add resume logic to handle execution results
- Test interrupt/resume cycle
- **Acceptance Criteria:**
  - Graph interrupts after code generation
  - Resume with execution results updates state

### Issue 3.4: Set up state persistence with checkpointing
**File:** `helpers/hagentic_eda/graph.py`
- Configure memory-based checkpointing
- Create `save_checkpoint()` and `load_checkpoint()` utilities
- Test state recovery from checkpoints
- **Acceptance Criteria:**
  - Checkpoints save/load correctly
  - State persists across sessions

### Issue 3.5: Create graph visualization and documentation
**File:** `helpers/hagentic_eda/graph.py`
- Add method to export graph structure (Mermaid diagram)
- Generate ASCII diagram of state machine
- Document node responsibilities and transitions
- **Acceptance Criteria:**
  - Diagram renders correctly
  - Documentation is clear

### Issue 3.6: User Interaction System for EDA Approval
**Status:** ⬜ Not started (https://github.com/causify-ai/tutorials/issues/662).
Loosely related: `tutorials/agentic_eda/agentui/` (AgenTUI chat interface),
but not scoped specifically to EDA-plan approval.
- Interactive plan presentation
- User feedback incorporation
- Approval workflow management
- Change request handling
- Version history of analyses
- **Acceptance Criteria:**
  - User can review/approve/modify the proposed EDA plan before execution

---

## EPIC 4: System Prompt Engineering

**Goal:** Design comprehensive system prompts that guide agent behavior across analysis phases.

**Owner:** AI Research Team

### Issue 4.1: Design system prompt template structure
**File:** `helpers/hagentic_eda/prompts.py`
- Create base prompt with sections:
  - Role and capabilities
  - Workflow phases (setup, cleaning, analysis, reporting)
  - Code generation guidelines (style, imports, comments)
  - Context injection points
  - Output format requirements
- **Acceptance Criteria:**
  - Template renders without errors
  - All sections present and meaningful

### Issue 4.2: Implement context formatting functions
**File:** `helpers/hagentic_eda/prompts.py`
- Create `format_dataframe_context()` for current data state
- Create `format_error_context()` for error recovery
- Create `format_analysis_history()` for conversation tracking
- Test with sample data and errors
- **Acceptance Criteria:**
  - Formatted context is concise and complete
  - Functions handle edge cases (empty data, no errors, etc.)

### Issue 4.3: Design phase-specific prompt variations
**File:** `helpers/hagentic_eda/prompts.py`
- Phase 1: "Read Schema and Infer Types" prompt
- Phase 2: "Propose EDA Plan" prompt
- Phase 3: "Run Full Analysis" prompt
- Phase 4: "Generate Report" prompt
- **Acceptance Criteria:**
  - Each phase has distinct guidance
  - Phase transitions are clear

### Issue 4.4: Create prompt testing framework
**File:** `helpers/hagentic_eda/prompts.py`
- Write unit tests for prompt formatting
- Add smoke tests with LLM (token counting, context length)
- Create example prompts with annotations
- **Acceptance Criteria:**
  - Tests pass
  - Prompts under context limits

### Issue 4.5: Few-Shot Prompt Collection for EDA
**Status:** ⬜ Not started (https://github.com/causify-ai/tutorials/issues/658).
- 10+ complete EDA examples (code + narrative), across data types/domains
- Include common edge cases and solutions
- **Acceptance Criteria:**
  - Collection is directly usable as few-shot context for the agent

---

## EPIC 5: Tool Definitions and Integration

**Goal:** Define tools the agent can invoke for data inspection, suggestions, and analysis.

**Owner:** Data Science & Helpers Integration Team

### Issue 5.1: Implement inspection tools
**Status:** 🔶 Attempted, not merged. helpers#850 "Reorg all the EDA files"
(https://github.com/causify-ai/helpers/issues/850) added
`dev_scripts_helpers/documentation/generate_EDA_context.py` in
[PR #979](https://github.com/causify-ai/helpers/pull/979), but the PR was
closed without merging. The script now lives instead at
`tutorials/agentic_eda/generate_EDA_context.py`.
**File:** `helpers/hagentic_eda/tools.py`
- `get_dataframe_info(df)`: Returns schema, shape, samples
- `get_column_statistics(df, column)`: Distribution, outliers, null count
- `detect_data_issues(df)`: Identify type mismatches, missing values
- Use `@tool` decorator from LangChain
- Write docstrings for LLM understanding
- **Acceptance Criteria:**
  - Tools callable by agent
  - Output useful and actionable

### Issue 5.2: Implement suggestion tools
**File:** `helpers/hagentic_eda/tools.py`
- `suggest_cleaning_steps(detected_issues)`: Recommend data prep
- `suggest_visualizations(column_types)`: Recommend charts/plots
- `suggest_features(dataset_info)`: Feature engineering ideas
- **Acceptance Criteria:**
  - Suggestions are specific and data-driven
  - Agent can act on recommendations

### Issue 5.3: Implement analysis tools
**File:** `helpers/hagentic_eda/tools.py`
- `run_statistical_test(df, test_type, columns)`: Hypothesis testing
- `check_correlations(df, method)`: Pearson/Spearman correlation
- Pydantic input validation for all tools
- **Acceptance Criteria:**
  - Tools validate inputs
  - Results formatted clearly

### Issue 5.4: Integrate with helpers modules
**File:** `helpers/hagentic_eda/tools.py`
- Leverage `hdataframe.py` for DataFrame operations
- Leverage `hpandas.py` for pandas utilities
- Leverage `hplot.py` (if exists) for visualization suggestions
- Document integration points
- **Acceptance Criteria:**
  - Tools use helpers functions
  - No code duplication

### Issue 5.5: Create tool registry and documentation
**File:** `helpers/hagentic_eda/tools.py`
- Create `TOOL_REGISTRY` dict mapping tool names to functions
- Generate tool documentation for LLM
- Create examples for each tool
- **Acceptance Criteria:**
  - Registry complete
  - Agent can discover all tools

---

## EPIC 6: Data Type Analysis Modules

**Goal:** Implement specialized analysis for different data types (time series, categorical, scalar, cross-variable).

**Owner:** Analytics Team (split by data type)

**Status:** ⬜ Not started. tutorials#657
(https://github.com/causify-ai/tutorials/issues/657, "Data Type Specific
Analysis Templates") is the umbrella real issue for Issues 6.1-6.4 — same
four template groups, no PR or comments.

### Issue 6.0: Advanced Data Type Inference (Owner: HarshitGadge)
**Status:** ⬜ Not started (https://github.com/causify-ai/tutorials/issues/656,
exact owner match with real assignee HarshitGadge).
- Time series patterns, categorical (incl. high-cardinality), text
  characteristics, semantic types (email/phone/address), column relationships
- **Acceptance Criteria:**
  - Detection accurate; feeds Issues 6.1-6.4's routing logic

### Issue 6.1: Time Series Analysis Module (Owner: Pranav + Harshit)
**Status:** ⬜ Not started as specified; loosely related informal prototype
at `tutorials/agentic_eda/intermediate_v0_timeseries_agent/`.
**File:** `helpers/hagentic_eda/analysis_timeseries.py`
- `analyze_time_series_autocorr()`: ACF/PACF plots
- `detect_seasonality()`: Decomposition and seasonal detection
- `detect_trend()`: Trend analysis
- `rolling_statistics()`: Moving mean/std
- `detect_change_points()`: Change point detection (optional)
- **Acceptance Criteria:**
  - Functions handle missing data gracefully
  - Visualizations are clear

### Issue 6.2: Categorical Analysis Module (Owner: Sai)
**Status:** ⬜ Not started —
https://github.com/causify-ai/helpers/issues/992 ("Categorical variables
agent for AutoEDA", assignees protocorn/srinivassaitangudu, exact owner
match), no PR, no comments.
**File:** `helpers/hagentic_eda/analysis_categorical.py`
- `analyze_categorical_distribution()`: Frequency counts, cardinality
- `time_series_categorical()`: Category distribution over time
- `crosstab_analysis()`: Association between variables
- `create_categorical_plots()`: Visualizations
- **Acceptance Criteria:**
  - Handles high-cardinality variables
  - Output is interpretable

### Issue 6.3: Scalar (Numeric) Analysis Module (Owner: Madhur)
**Status:** ⬜ Not started; part of tutorials#657's Scalar Analysis template group.
**File:** `helpers/hagentic_eda/analysis_scalar.py`
- `analyze_distribution()`: Histogram, KDE, normality tests
- `detect_outliers()`: IQR, z-score, isolation forest methods
- `generate_summary_stats()`: Mean, median, std, quantiles
- `correlation_matrix()`: Pearson and Spearman
- `pairwise_scatter_plots()`: Visualization
- **Acceptance Criteria:**
  - Outlier detection works with different methods
  - Statistics are accurate

### Issue 6.4: Cross-Variable Analysis Module (Owner: Sahil + Sai)
**Status:** ⬜ Not started; part of tutorials#657's Cross-variable Analysis template group.
**File:** `helpers/hagentic_eda/analysis_cross_variable.py`
- `correlate_time_series()`: Correlation between time series
- `categorical_numeric_interaction()`: Groups and aggregations
- `conditional_distributions()`: Conditional analysis
- `feature_interaction_analysis()`: Interaction detection
- **Acceptance Criteria:**
  - Handles mixed data types
  - Insights are actionable

### Issue 6.5: Unified Analysis Orchestrator
**Status:** ⬜ Not started — no direct GitHub issue.
**File:** `helpers/hagentic_eda/analysis.py`
- Create `AnalysisOrchestrator` class
- Route to appropriate analysis module based on detected types
- Aggregate results into unified report
- **Acceptance Criteria:**
  - All data types analyzed
  - Results combined logically

---

## EPIC 7: Code Generation and Execution

**Goal:** Enable the agent to generate and execute Python code in notebook cells with error recovery.

**Owner:** Execution Engine Team

**Status:** ⬜ Not started, no direct real issue.

### Issue 7.1: Code generation from agent responses
**File:** `helpers/hagentic_eda/code_generation.py`
- `extract_code_blocks()`: Parse agent response for code
- `validate_code()`: Syntax check and security checks
- `format_code()`: Apply style conventions
- **Acceptance Criteria:**
  - Handles multiple code blocks
  - Blacklist checks for dangerous commands

### Issue 7.2: Notebook cell management
**File:** `helpers/hagentic_eda/notebook_management.py`
- `create_notebook_cell()`: Generate cell object
- `insert_cell()`: Add to notebook programmatically
- `execute_cell()`: Run code in kernel
- `capture_output()`: Get stdout, stderr, results
- **Acceptance Criteria:**
  - Cells execute correctly
  - Output captured accurately

### Issue 7.3: Error handling and recovery
**File:** `helpers/hagentic_eda/code_generation.py`
- `parse_error()`: Extract error type and message
- `generate_fix()`: Create corrective code
- `retry_with_fix()`: Run corrective code
- **Acceptance Criteria:**
  - Errors detected accurately
  - Agent can recover from common errors

### Issue 7.4: Cell execution tracking
**File:** `helpers/hagentic_eda/execution_tracking.py`
- Track execution history: timestamps, duration, status
- Store results and metadata
- Generate execution reports
- **Acceptance Criteria:**
  - Tracking is complete and accurate

---

## EPIC 8: Jupyter Integration, Frontend & Service Layer

**Goal:** Create the browser-to-kernel bridge, JupyterLab extension, and a service layer around the agent.

**Owner:** Frontend/Integration Team

**Note:** This requires JupyterLab extension development (TypeScript) which is a major undertaking. Consider this phase 2.

**Status:** ⬜ Not started.

### Issue 8.1: Server Extension (Python)
- Backend service hosting LangGraph agent
- WebSocket server for communication
- Kernel interaction layer

### Issue 8.2: Frontend Extension (TypeScript)
- JupyterLab extension initialization
- UI for interacting with agent
- Notebook cell manipulation through `@jupyterlab/notebook`

### Issue 8.3: Communication Bridge
- JSON message format specification
- Bidirectional WebSocket handling
- Synchronization between UI and kernel state

### Issue 8.4: FastAPI Microservice for AutoEDA
**Status:** ⬜ Not started (https://github.com/causify-ai/tutorials/issues/659).
- REST endpoints: schema validation, analysis planning, execution
  management, result retrieval
- Auth + rate limiting, async processing, health monitoring, OpenAPI docs
- **Acceptance Criteria:**
  - All endpoints implemented and documented

---

## EPIC 9: Testing & Quality Assurance

**Goal:** Comprehensive testing framework for the AutoEDA agent.

**Owner:** QA Team

**Status:** 🔶 Repo-level CI is done (helpers#998 -> merged
[tutorials PR #638](https://github.com/causify-ai/tutorials/pull/638));
AutoEDA-specific unit/integration/benchmark/validation tests below are not
started.

### Issue 9.1: Unit tests for core modules
- Test state management (EPIC 2)
- Test graph logic (EPIC 3)
- Test tools (EPIC 5)
- Test analysis modules (EPIC 6)
- Target: 85%+ code coverage
- **Acceptance Criteria:**
  - All unit tests pass
  - Coverage threshold met

### Issue 9.2: Integration tests for full workflows
- Test end-to-end agent execution
- Test with sample datasets (time series, categorical, scalar)
- Test error recovery and state persistence
- **Acceptance Criteria:**
  - Workflows complete successfully
  - State persists correctly

### Issue 9.3: Performance and stress testing
- Test with large datasets
- Measure agent latency
- Identify bottlenecks
- **Acceptance Criteria:**
  - Performance within acceptable bounds
  - No memory leaks

### Issue 9.4: Test dataset curation
- Create sample CSV, Parquet, Feather files
- Create schema files (JSON/YAML)
- Document dataset characteristics
- **Acceptance Criteria:**
  - Datasets cover all data types
  - Easy to use in tests

### Issue 9.5: AutoEDA Benchmarking System
**Status:** ⬜ Not started (https://github.com/causify-ai/tutorials/issues/660);
research seed found via Issue 0.1 (DataSciBench benchmark).
- Curated benchmark datasets
- Metrics: insight accuracy, coverage completeness, code quality, report clarity
- Comparison with SOTA solutions; performance tracking over time
- **Acceptance Criteria:**
  - Metrics reproducible and automatable

### Issue 9.6: Out-of-Sample Validation System
**Status:** ⬜ Not started (https://github.com/causify-ai/tutorials/issues/661).
- Time-based splitting for time series; stratified splitting for categorical data
- Insight consistency checking across splits; overfitting detection; CV support
- **Acceptance Criteria:**
  - Splits correct per data type; consistency/overfitting checks automated

---

## EPIC 10: Documentation & Examples

**Goal:** Create comprehensive documentation and example notebooks.

**Owner:** Documentation Team

**Status:** ⬜ Not started. Seed research already exists informally — see Issue 0.1.

### Issue 10.1: API documentation
- Docstrings for all public functions
- Generate Sphinx/MkDocs docs
- Publish to GitHub Pages
- **Acceptance Criteria:**
  - All public APIs documented
  - Examples in docstrings

### Issue 10.2: Tutorials and guides
- Getting started guide
- Architecture overview document
- Step-by-step tutorial notebooks
- Advanced usage guide
- **Acceptance Criteria:**
  - Tutorials follow project conventions
  - Examples run successfully

### Issue 10.3: Example AutoEDA analysis
- Create example notebook showing full workflow
- Demonstrate on Kaggle datasets (stock prices, Netflix, etc.)
- Annotate key insights
- **Acceptance Criteria:**
  - Notebook is clear and runnable
  - Insights are meaningful

---

## Implementation Roadmap

### Phase 0: Research (done informally)
- SOTA survey and framework choice (LangGraph) already happened in
  tutorials#666's comment thread — see Issue 0.1. Never written up as the
  standalone doc the issue asked for.

### Phase 1: Foundation (Weeks 1-3)
- EPIC 1: Infrastructure setup
- EPIC 2: State management
- EPIC 3: LangGraph orchestration
- Deliverable: Basic graph structure compiling and running

### Phase 2: Intelligence (Weeks 4-6)
- EPIC 4: System prompts
- EPIC 5: Tools
- EPIC 6: Analysis modules
- Deliverable: Agent can perform basic analysis

### Phase 3: Execution & Integration (Weeks 7-9)
- EPIC 7: Code generation and execution
- EPIC 9: Testing (unit + integration)
- Deliverable: End-to-end agent workflow with sample datasets

### Phase 4: Polish & Documentation (Weeks 10-12)
- EPIC 8: Jupyter integration (if time permits)
- EPIC 9: Performance testing
- EPIC 10: Documentation and examples
- Deliverable: Production-ready agent with comprehensive docs

### Phase 5: Future Work
- Advanced features (text analysis, clustering, etc.)
- Full JupyterLab extension
- Cloud deployment options

---

## Success Criteria (End State)

1. **Functional Agent**: Autonomously analyzes datasets end-to-end
2. **State Management**: Persists progress and recovers from errors
3. **Extensible**: New analysis types can be added easily
4. **Well-Tested**: 85%+ code coverage, integration tests pass
5. **Well-Documented**: API docs, tutorials, and examples provided
6. **Production-Ready**: Error handling, logging, and monitoring in place

---

## Team Assignments Summary

| EPIC | Owner | Skills |
|------|-------|--------|
| 0 | Research | SOTA survey, framework evaluation |
| 1 | DevOps | Infrastructure, CI/CD |
| 2 | Data Eng | Python, Pydantic, Data structures |
| 3 | Agent Core | LangGraph, Python, State machines |
| 4 | AI Research | Prompt engineering, LLM understanding |
| 5 | Data Science + Helpers | Statistics, Data analysis, Integration |
| 6 | Analytics (split) | Domain expertise per data type |
| 7 | Execution | Python, Jupyter, Error handling |
| 8 | Frontend | TypeScript, JupyterLab, WebSockets |
| 9 | QA | Testing, pytest, Performance |
| 10 | Docs | Technical writing, Examples |

---

## Appendix: GitHub Project 116 Cross-Reference

Snapshot of https://github.com/orgs/causify-ai/projects/116
("AutoML-v1.3 - AgenticEDA", 16 items, captured 2026-09-17) — the
authoritative status source; the live project's Status column is often
stale (some "In Progress" issues had closed-unmerged PRs; #666 showed
"Todo" despite real research in its comments).

| GitHub Issue | Maps to | Status | Location |
|---|---|---|---|
| [helpers#850](https://github.com/causify-ai/helpers/issues/850) | Issue 5.1 | 🔶 Attempted, not merged | [PR #979](https://github.com/causify-ai/helpers/pull/979) closed unmerged; superseded by `tutorials/agentic_eda/generate_EDA_context.py` |
| [helpers#986](https://github.com/causify-ai/helpers/issues/986) | Issue 1.1 | 🔶 Two unmerged drafts | [PR #989](https://github.com/causify-ai/helpers/pull/989) (`langgraph/`), [PR #991](https://github.com/causify-ai/helpers/pull/991) (`autoeda/`) |
| [helpers#992](https://github.com/causify-ai/helpers/issues/992) | Issue 6.2 | ⬜ Not started | — |
| [tutorials#609](https://github.com/causify-ai/tutorials/issues/609) | Issue 1.1 | ✅ Done, later deleted | merged [PR #614](https://github.com/causify-ai/tutorials/pull/614); dir removed in cleanup commit `4fbaff56`; surviving code in `tutorials/langchain_reference/graphs/`, `tutorials/agentic_eda/simple_pemdas_agent/` |
| [helpers#997](https://github.com/causify-ai/helpers/issues/997) | Issue 1.2 | ✅ Done (different location) | `tutorials/agentic_eda/` |
| [helpers#998](https://github.com/causify-ai/helpers/issues/998) | Issue 1.3 | ✅ Done | merged [PR #638](https://github.com/causify-ai/tutorials/pull/638); `tutorials/.github/workflows/` |
| [tutorials#628](https://github.com/causify-ai/tutorials/issues/628) | Issues 1.2 / 1.3 | ✅ Done | merged [PR #641](https://github.com/causify-ai/tutorials/pull/641); `tutorials/agentic_eda/` |
| [tutorials#655](https://github.com/causify-ai/tutorials/issues/655) | Issue 1.1 (dup of helpers#986) | ⬜ Not started | — |
| [tutorials#656](https://github.com/causify-ai/tutorials/issues/656) | Issue 6.0 | ⬜ Not started | — |
| [tutorials#657](https://github.com/causify-ai/tutorials/issues/657) | Issues 6.1-6.4 | ⬜ Not started | related prototype `tutorials/agentic_eda/intermediate_v0_timeseries_agent/` |
| [tutorials#658](https://github.com/causify-ai/tutorials/issues/658) | Issue 4.5 | ⬜ Not started | — |
| [tutorials#659](https://github.com/causify-ai/tutorials/issues/659) | Issue 8.4 | ⬜ Not started | — |
| [tutorials#660](https://github.com/causify-ai/tutorials/issues/660) | Issue 9.5 | ⬜ Not started (research seed in #666) | — |
| [tutorials#661](https://github.com/causify-ai/tutorials/issues/661) | Issue 9.6 | ⬜ Not started | — |
| [tutorials#662](https://github.com/causify-ai/tutorials/issues/662) | Issue 3.6 | ⬜ Not started | loosely related `tutorials/agentic_eda/agentui/` |
| [tutorials#666](https://github.com/causify-ai/tutorials/issues/666) | Issue 0.1 | 🔶 Research done, not consolidated | comment thread only |

**Biggest divergence:** all real merged AutoEDA code lives in
`causify-ai/tutorials/agentic_eda/`, not `helpers/hagentic_eda/` as assumed
throughout EPICs 1-7.
