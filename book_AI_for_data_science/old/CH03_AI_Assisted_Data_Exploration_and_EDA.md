# Chapter 3: AI-Assisted Data Exploration and EDA

## 1. Automated Exploratory Data Analysis

### 1.1 What Automated EDA Does

- Traditional EDA = manual process: write summary statistics, plot distributions,
  check correlations, identify missing values — takes hours to days
- Automated EDA tools generate this analysis in seconds by profiling every
  column, computing pairwise statistics, and rendering interactive reports
- AI augmentation adds a layer on top: interpret the profile, flag anomalies,
  and recommend next steps for cleaning and modeling
- Key benefit: data scientists can spend time on judgment, not mechanical
  computation

### 1.2 Profiling Tools and Workflows

- `ydata-profiling` (formerly pandas-profiling): generates an HTML report with
  distributions, correlations, missing-value maps, and data type summaries
  - Use `ProfileReport(df, explorative=True)` for deep analysis
  - Enable `minimal=True` for large datasets to avoid memory issues
- `Dataprep`: faster profiling with `create_report()`, optimized for large
  DataFrames using Dask under the hood
- Workflow: run profile -> paste summary to LLM -> ask for interpretation and
  next steps

### 1.3 AI Interpretation of EDA Outputs

- Send the text output of EDA (not raw data) to an LLM:
  - Share the list of columns with high missing rates
  - Share the correlation matrix as a CSV
  - Share the list of suspicious value counts (rare categories, outlier bins)
- Effective prompts for EDA interpretation:
  - "These 5 columns have >30% missing values. What imputation strategy would
    you recommend for each based on the data type and correlation pattern?"
  - "This correlation matrix shows X and Y are 0.98 correlated. Should I drop
    one and which?"
- LLM limitations in EDA: cannot inspect the actual data distribution without
  the summary; always share `df.describe()` and `df.value_counts()` outputs

- TUTORIAL: YData-profiling - Generate a comprehensive automated EDA report for
  a tabular dataset; use AI to interpret the profile and suggest next steps for
  data cleaning and feature engineering

- TUTORIAL: Dataprep - Use the `create_report()` function to explore a dataset
  automatically; compare the AI-generated EDA summary to manual analysis

- TUTORIAL: Modin - Use Modin as a drop-in pandas replacement to accelerate EDA
  on multi-million-row datasets; benchmark EDA pipeline speed with and without
  Modin on a large CSV file

---

## 2. AI-Assisted Data Visualization

### 2.1 Visualization Recommendation

- AI can recommend chart types based on data characteristics:
  - Continuous target + continuous feature -> scatter plot or hexbin
  - Categorical feature + continuous target -> box plot or violin
  - Time index + numeric values -> line chart with anomaly overlay
  - High-cardinality categorical -> bar chart sorted by frequency
- Prompting pattern: paste `df.dtypes` and `df.describe()` and ask "what
  visualizations should I create first?"

### 2.2 Declarative Visualization with AI

- Declarative libraries (Altair, Vega-Lite) are well-suited for AI generation
  because the chart spec is readable and editable text
  - AI can generate a full Altair JSON spec from a description
  - Human reviews and tweaks the spec without rewriting from scratch
- Grammar of graphics: chart = data + marks + encodings + transformations
  - Each component maps directly to prompt language

### 2.3 Interactive and Streaming Visualization

- Jupyter widgets and Streamlit enable interactive EDA dashboards
- Use case: build a one-click EDA dashboard that any stakeholder can use
  without Python knowledge
- AI role: generate the dashboard code; human customizes layout and filters

- TUTORIAL: Altair - Build declarative statistical visualizations guided by AI
  prompts; generate chart recommendations for each variable pair in a dataset

- TUTORIAL: wandb - Log and visualize training metrics and dataset statistics;
  use the AI-assisted panel builder to surface patterns across experiment runs

- TUTORIAL: Streamlit - Build an interactive EDA dashboard with Streamlit;
  use AI to generate the dashboard code from a dataset description; deploy
  locally and share with stakeholders

---

## 3. Data Quality Assessment at Scale

### 3.1 Defining Data Quality

- Data quality dimensions:
  - **Completeness**: proportion of non-null values per column
  - **Consistency**: no contradictory values across related columns
  - **Accuracy**: values fall within valid ranges
  - **Timeliness**: data is not stale relative to the business process
  - **Uniqueness**: no duplicate rows or keys
- AI role: generate quality rules from schema descriptions and business context

### 3.2 Expectation-Based Validation

- Great Expectations workflow:
  1. Define an expectation suite: rules about what the data should satisfy
  2. Run a validation checkpoint against incoming data
  3. Generate a data docs report with pass/fail for each expectation
- AI generates expectations from natural language:
  - "The `age` column should be between 0 and 120"
  - "The `user_id` column should be unique and non-null"
  - "The `revenue` column should be positive"

### 3.3 Schema Enforcement

- Pandera: define schemas as code; validate DataFrames at runtime
  - Schema defines column types, ranges, and custom checks
  - Catches breaking changes when upstream data evolves
- Integration with CI: run pandera checks as part of data pipeline testing

- TUTORIAL: Great Expectations - Define data quality expectations for a
  production dataset; run validation suites and use AI to interpret failed
  checks and suggest remediation steps

- TUTORIAL: pandera - Write schema validation rules for a pandas DataFrame; use
  AI to generate pandera schemas from existing datasets automatically

- TUTORIAL: Pyjanitor - Build a data quality pipeline using pyjanitor's
  method-chaining API; use AI to generate the cleaning chain from a description
  of data issues found in the EDA report

---

## 4. Schema Inference and Documentation

### 4.1 Schema Inference from Raw Data

- Automated schema inference: tools infer column types, nullability, and
  constraints from sample data
  - Benefits: reduces manual schema definition effort
  - Risks: inference errors on edge cases (e.g., numeric IDs inferred as
    continuous features)
- AI can post-process inferred schemas: add human-readable column descriptions,
  flag ambiguous columns for human review

### 4.2 Data Catalog and Discovery

- Data catalog = central registry of datasets, schemas, and lineage
  - Enables data scientists to discover relevant datasets without tribal knowledge
  - Reduces time spent searching for data
- Modern catalogs: Amundsen, DataHub, Marquez
  - Amundsen: metadata + search + governance
  - DataHub: lineage-aware metadata with stream ingestion
  - Marquez: open lineage standard for pipeline metadata

### 4.3 AI-Enhanced Documentation

- AI can generate column descriptions from column names + sample values
  - Example: column `cust_rev_30d` + sample values [12.5, 0.0, 450.0] ->
    "Customer revenue in the last 30 days (USD)"
- AI can generate dataset-level README from the schema and context
- Human review step: verify generated descriptions against business definitions

- TUTORIAL: Amundsen - Set up a data discovery catalog; ingest table metadata
  and use AI to generate human-readable descriptions for each column and table

- TUTORIAL: DataHub - Build a metadata platform for a multi-source data
  pipeline; demonstrate how AI can surface relevant datasets and lineage
  information during exploratory analysis

- TUTORIAL: Marquez - Track data lineage for an EDA pipeline with Marquez;
  use AI to generate lineage annotations from pipeline code descriptions
