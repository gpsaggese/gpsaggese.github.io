# AutoEDA Agent Specification

## Goals

The AutoEDA agent is designed to:
- Discover meaningful patterns, anomalies, and relationships in data.
- Generate a Jupyter notebook using internal helper functions to perform the analysis.
- Produce a structured, readable report summarizing insights.
- Operate in an interactive loop with the user for feedback and approval.

## Inputs

- **Schema File**: JSON or YAML describing the structure and types of the dataset.
- **Data File**: CSV, Parquet, or Feather containing the actual data to analyze.

## Agentic Loop

### 1. Read Schema 
https://github.com/causify-ai/helpers/issues/986
- Parse the schema file to extract:
  - Column names
  - Declared data types
  - Metadata such as primary keys, time indices, target variables Madhur

### 2. Infer Data Types and Semantics
- Cross-check declared types with actual data sample
- Auto-detect feature types:
  - Time series
  - Categorical
  - Scalar/numeric
  - Text (future extension) Harshit

### 3. Propose EDA Plan
- Based on inferred types, generate a plan for EDA
- Select relevant helper functions from a predefined library
- Sample a subset of the data to run lightweight diagnostics
- Present the proposed plan to the user for confirmation

Time series analysis: Pranav Harshit
Categorical Variables: Sai
https://github.com/causizy-ai/helpers/issues/992
Scalar Variables:  Madhur 
Cross Variable Analysis: Sahil + Sai

### 4. Run Full Analysis
- Upon user confirmation, apply the selected analyses on the full dataset
- Save visualizations and intermediate outputs to files Sahil + Pranav

### 5. Generate Outputs
- **Jupyter Notebook**:
  - Clean, modular code using helper functions
  - Well-commented with markdown explanations
- **Report**:
  - Summarized insights
  - Key charts and findings
  - High-level interpretation of statistical outputs Sai + Madhur

## Analysis Templates by Data Type

### Time Series Variables
- Check for missing data patterns
- Compute min / max timestamps
- Univariate analysis:
  - Autocorrelation
  - Seasonal decomposition
  - Spectral analysis (e.g., FFT)
- Trend and seasonality detection
- Rolling statistics
- Change point detection (optional)
Eg: https://www.kaggle.com/datasets/arashnic/time-series-forecasting-with-yahoo-stock-price

### Categorical Variables
- Frequency counts
- Cardinality check
- Distribution over time (if applicable)
- Crosstab and association with other variables
Eg.: https://www.kaggle.com/datasets/shivamb/netflix-shows

### Scalar (Numeric) Variables
- Distribution plots (histograms, KDE)
- Outlier detection (IQR, z-score, isolation forest)
- Descriptive statistics
- Correlation matrix (Pearson, Spearman)
- Pairwise scatter plots
- Mutual information with target (if any)
Eg: https://www.kaggle.com/datasets/muhammedtausif/world-population-by-countries?select=world-population-by-country-2020.csv

### Cross-variable Analysis
- Correlation between time series
- Interactions between categorical and numeric variables
- Grouped aggregations (e.g., mean by category)
- Conditional distributions
Eg: https://www.kaggle.com/datasets/vikrishnan/boston-house-prices


## Helper Function Library (to be implemented or imported)
- `plot_distribution()`
- `plot_correlation_matrix()`
- `analyze_missing_values()`
- `analyze_time_series_autocorr()`
- `generate_summary_stats()`
- `detect_outliers()`
- `cross_tabulate()`
- `mutual_correlation_analysis()`
- `create_visual_report()`

## Additional Features (Optional / Future Work)
- Text column summarization (e.g., word frequencies)
- Clustering and dimensionality reduction
- Automatic insight scoring or ranking
- Configurable EDA depth based on user role or dataset size

# **High-level plan**

First things to do

- Create tutorials for the basic packages (langchain, CrewAI)
- Create tutorials on how to programmatically use Jupyter notebook
- Indro

# **Research Plan: AgenticEDA**

**Objective:** Develop a Jupyter-native, agentic framework for autonomous data analysis, utilizing a JupyterLab Extension architecture for direct kernel interaction.

- Utilize the modules (helpers/csfy) to provide functionalities without reinventing the wheel and data standardizations.

## **Infrastructure Core**

*Focus: Establishing the browser-to-kernel bridge and the basic "Write-Execute-Read" loop.*

### **v0.1: The Extension Skeleton**

* **Goal:** Establish the communication pipeline between the User Interface (Frontend) and the Agent (Backend).
* **Deliverables:**
  * **Server Extension (Python):** Set up the backend service that hosts the LangGraph agent and connects to LLM APIs.
  * **Frontend Extension (TypeScript):** Initialize the JupyterLab extension using `@jupyterlab/notebook` and `@jupyterlab/application`.
  * **Communication Bridge:** Implement WebSocket handling to pass JSON messages (User Prompt \<-\> Agent Code) between the frontend and server.

### **The "Write-Execute-Read” Loop**

* **Goal:** Enable the agent to mechanically interact with the notebook cells.
* **Deliverables:**
  * **The Writer:** Implement `INotebookTracker` logic in the frontend to programmatically insert new code cells populated with content from the agent.
  * **The Executor:** Implement `CodeCell.execute()` triggers to run the inserted cells on the user's active kernel.
  * **The Reader:** Implement an `IOPub` listener that captures execution results (`stdout`, `stderr`/tracebacks, and `execute_result`) and sends them back to the agent for inspection.

###  **State Management**

* **Goal:** Robust session handling and self-correction.
* **Deliverables:**
  * **SharedState Object:** Define the schema for passing variables/plans between graph nodes (e.g., maintaining a list of "known columns" or "previous errors").
  * **Code Logic:** Implement the "Code \-\> Error \-\> Fix" loop. If the "Reader" captures a traceback, the agent automatically generates a correction cell.
  * **Guard Rails:** Implement a blacklist for destructive commands (e.g., `os.system`, `shutil.rmtree`) before code is sent to the frontend.

## **Intelligence & Heuristics**

*Focus: Implementing specific agent roles and the heuristic logic layers.*

1. ### **Cleaning the data:**

   **Goal:** Intelligent data preparation using Hybrid Heuristics.

2. ### **Analyze the data**

   **Goal:** Semantic understanding of features, plots, checks and statistical analysis
- Develop an understanding of what has to done
- Get a sense of the data
- Identify target variable(s)

3. ###  **The "Engineer" (AutoML Agent)**

   **Goal:** Baseline modeling integration.
- The model needs to have a sen

## **User Experience & Optimization**

*Focus: Reducing friction and handling large contexts.*

### **Interaction Polish**

* **Goal:** Full integration.
* **Deliverables:**

###  **If circumstances permit: Context Optimization**

**Goal:** Handling long sessions.

ISSUE: Do X and Y

# **Summary**

- Implementation plan for the LangGraph-based agent that performs autonomous data analysis
- Covers agent state schema, graph definition, system prompts, and tool definitions

# **Agent Core \- LangGraph Integration**

- Goal: Implement the LangGraph-based agent that performs autonomous data analysis

## **Agent State Schema**

- Agent state must track:
  - Conversation history
  - Dataset understanding (columns, types, issues)
  - Generated cells and their results
  - Analysis phase and progress
  - Error tracking for self-correction

### **Scope**

- Define core state schema: a class with state definition by conversation, dataset context, cell tracking, analysis progress, and process control
- Define supporting schemas:
  - `DataFrameInfo`: columns, dtypes, shape, sample values, null counts
  - `NotebookCell`: id, type, content, result, error
  - `ColumnInfo`: name, dtype, statistics, detected issues
- Implement state serialization for persistence

### **Expected outcomes**

- State schema definitions
- Supporting model definitions
- State serialization utilities
- State migration strategy for schema changes

### **Requirements**

- State captures all necessary information
- Annotated fields accumulate correctly
- State serializable for persistence
- Schema documented with examples

## **LangGraph Definition**

- The agent graph orchestrates reasoning, tool use, code generation, and result processing with interrupt points for kernel execution

### **Scope**

- Define graph nodes:

  - `agent`: Main reasoning node \- decides next action
  - `tools`: Execute LangChain tools (inspection, suggestions)
  - `generate_code`: Extract code from agent response
  - `handle_error`: Process execution errors, generate fixes
  - `update_context`: Update dataset understanding from results


- Define edges and routing:

  - START \-\> agent
  - agent \-\> tools (if tool call)
  - agent \-\> generate\_code (if code generation)
  - agent \-\> END (if complete)
  - tools \-\> agent
  - generate\_code \-\> INTERRUPT (wait for execution)
  - \[resume\] \-\> update\_context
  - update\_context \-\> agent (continue) or handle\_error (if error)
  - handle\_error \-\> agent


- Implement interrupt/resume for execution feedback loop

- Configure checkpointing for state persistence

### **Technical Requirements**

- LangGraph StateGraph with typed state
- interrupt\_after for execution synchronization
- Conditional routing based on state
- Memory-based checkpointing

### **Expected outcomes**

- Graph definition with all nodes
- Routing functions
- Interrupt/resume handling
- Graph visualization for documentation

### **Requirements**

- Graph compiles without errors
- All routing conditions covered
- Interrupt/resume works correctly
- State persists across interrupts

## **System Prompt Engineering**

- System prompt guides agent behavior across analysis phases
- Must be structured for:
  - Consistent code generation style
  - Phase-appropriate actions
  - Error recovery guidance
  - Output format compliance

### **Scope**

- Design prompt template with sections:

  - Role and capabilities
  - Workflow phases and transitions
  - Code generation guidelines
  - Current context injection
  - Output format requirements


- Implement context injection

- Define phase-specific guidance

- Include examples for code style

### **Deliverables**

- System prompt template
- Context formatting functions
- Phase-specific prompt variations
- Prompt testing framework

### **Acceptance Criteria**

- Agent generates consistent code style
- Phase transitions guided correctly
- Error recovery prompts effective
- Prompt under context window limits

## **Tool Definitions**

- Tools provide structured actions the agent can take beyond code generation
- Should leverage existing helpers/ modules where applicable

### **Scope**

- Define inspection tools:

  - `get_dataframe_info`: Schema, shape, samples
  - `get_column_statistics`: Detailed stats per column
  - `detect_data_issues`: Missing values, outliers, type mismatches


- Define suggestion tools:

  - `suggest_cleaning_steps`: Based on detected issues
  - `suggest_visualizations`: Based on column types
  - `suggest_features`: Feature engineering recommendations


- Define analysis tools:

  - `run_statistical_test`: Hypothesis testing
  - `check_correlations`: Correlation analysis


- Integrate with helpers/ modules:

  - `hdataframe.py` utilities
  - `hpandas.py` utilities

### **Technical Requirements**

- LangChain @tool decorator
- Pydantic input validation
- Clear docstrings for LLM understanding
- Error handling with informative messages

### **Expected Outcomes**

- Tool implementations
- Tool registry
- Integration with helpers modules
- Tool documentation
- Tools callable by agent
- Input validation prevents errors
- Outputs useful for agent reasoning
- Helpers integration working

Goal: You want to design an AutoEDA agent to:
- Discover meaningful patterns, anomalies, and relationships in data.
- Generate a Jupyter notebook using internal helper functions to perform the analysis.
- Produce a structured, readable report summarizing insights.
- Operate in an interactive loop with the user for feedback and approval.

Create a plan broken down in EPICs and Issues
