# Chapter 9: AI-Powered Monitoring and Drift Detection

## 1. Data Drift Detection

### 1.1 What Is Data Drift

- Data drift = the statistical distribution of input features changes over time
  after model deployment
  - **Covariate shift**: input distribution P(X) changes; P(Y|X) unchanged
  - **Prior probability shift**: target distribution P(Y) changes
  - **Concept drift**: the relationship P(Y|X) changes; model's learned
    mapping becomes incorrect
- Root causes:
  - Seasonal patterns: user behavior in summer vs. winter
  - Data pipeline changes: upstream data source schema evolution
  - External events: economic shocks, regulatory changes, product launches

### 1.2 Statistical Drift Tests

- Univariate tests for numeric features:
  - Kolmogorov-Smirnov test: compares CDFs of reference vs. current distribution
  - Population Stability Index (PSI): binned comparison, commonly used in finance
  - Wasserstein distance: earth mover's distance between distributions
- Categorical features: chi-squared test or Jensen-Shannon divergence
- Multivariate drift: detect changes in joint distribution using embedding-based
  methods or Maximum Mean Discrepancy (MMD)

### 1.3 AI-Assisted Drift Diagnosis

- After detecting drift, AI helps diagnose the cause:
  - "Feature X drifted significantly. The distribution shifted from mainly
    values 0-10 to values 50-100. What could cause this change in a retail
    transaction system?"
  - AI suggests: pricing change, new customer segment, data pipeline bug
- Prompting pattern: share the drift report (column name, drift statistic,
  distribution summary before/after) -> ask for root cause hypotheses

- TUTORIAL: YData-profiling - Generate baseline and current data profiles; use
  AI to compare them and identify columns with significant distribution shift

- TUTORIAL: River - Implement online drift detection on a data stream using
  River's drift detectors; use AI to interpret drift signals and decide when to
  trigger retraining

- TUTORIAL: stumpy - Use stumpy's matrix profile to detect recurring patterns
  and structural breaks in streaming feature data; use AI to distinguish
  seasonal drift from concept drift in the motif analysis

---

## 2. Model Performance Monitoring

### 2.1 Performance Monitoring Challenges

- Ground truth delay: for many problems, the true label arrives long after the
  prediction was made
  - Loan default: know in 12 months
  - Fraud: may know in days or weeks
  - Recommendation: know when the user interacts
- Proxy metrics: use model confidence scores or surrogate signals as early
  indicators of degradation
- Monitoring strategy:
  - Alert on input drift immediately (no label needed)
  - Alert on proxy metric degradation as a leading indicator
  - Alert on true performance metrics when labels become available

### 2.2 Monitoring Systems

- MLflow model monitoring: log predictions and ground truth together; query
  with SQL to compute rolling accuracy metrics
- Weights & Biases monitoring dashboards: real-time charts of key metrics;
  alerting via email or Slack when thresholds are crossed
- Evidently AI: open-source monitoring dashboards for data and model drift with
  pre-built reports

### 2.3 Retraining Triggers

- Retraining strategies:
  - **Scheduled retraining**: retrain every N days regardless of performance
  - **Performance-triggered retraining**: retrain when accuracy drops below threshold
  - **Drift-triggered retraining**: retrain when input distribution shifts
    significantly
- AI generates the retraining trigger policy from SLA descriptions:
  - "We need 95% precision. If the rolling weekly precision drops below 93%,
    trigger retraining."

- TUTORIAL: mlflow - Set up model performance monitoring by logging predictions
  and ground truth labels to MLflow; use AI to write SQL queries that detect
  performance degradation

- TUTORIAL: wandb - Build a model monitoring dashboard in Weights and Biases;
  alert on accuracy drops; use AI to diagnose whether degradation is caused by
  drift or a deployment issue

- TUTORIAL: sktime - Monitor a time series forecasting model with sktime;
  compute rolling forecast errors and use AI to determine when residual patterns
  indicate model degradation

---

## 3. LLM-Assisted Incident Response

### 3.1 The Incident Response Workflow

- ML incident = unexpected degradation in model performance or data quality
  that requires immediate investigation
- Response workflow:
  1. Alert triggered by monitoring system
  2. On-call data scientist investigates root cause
  3. Short-term mitigation: roll back to previous model, fall back to rule-based
     system
  4. Root cause analysis: find the upstream cause
  5. Long-term fix: retrain, update pipeline, fix data source

### 3.2 AI as Incident Responder

- AI accelerates incident response:
  - Correlates alert with recent data pipeline changes
  - Suggests likely root causes based on alert pattern
  - Drafts incident report from monitoring data
- Knowledge graph of incidents: historical incidents stored in a graph with
  causes, symptoms, and resolutions
  - LangChain + Neo4j: query the incident knowledge base with natural language

### 3.3 Automated Alert Triage

- Pass monitoring summaries to an LLM for triage:
  - Input: drift alert + performance alert + recent data changelog
  - Output: structured incident report with severity, likely cause, and
    recommended actions
- Reduces mean time to resolution (MTTR) by shortlisting hypotheses

- TUTORIAL: Langchain and Neo4j - Build an incident response assistant that
  queries a knowledge graph of past incidents; use AI to match current anomalies
  to historical root causes and suggest remediation steps

- TUTORIAL: llm - Automate alert triage by passing monitoring summaries to an
  LLM; generate a structured incident report with severity, likely cause, and
  recommended actions

- TUTORIAL: Apache Kafka - Build a real-time alert processing pipeline with
  Kafka; consume model prediction events and monitoring metrics; use AI to
  generate the consumer code that detects anomalous prediction patterns

---

## 4. Observability at Scale

### 4.1 ML Observability vs. Monitoring

- Monitoring = tracking known metrics with predefined thresholds
- Observability = ability to understand any internal state of the system from
  its external outputs
  - Key principle: you cannot monitor what you did not anticipate; observability
    lets you investigate unexpected failures
- ML observability requires logging:
  - Every prediction with its input features and model version
  - Model confidence scores and uncertainty estimates
  - Data lineage: where did each input feature come from?

### 4.2 Data Lineage for Observability

- Data lineage = graph that tracks the provenance of every dataset and
  transformation in the pipeline
  - When model performance drops, trace back through the lineage to find which
    upstream change caused it
- DataHub: metadata platform that ingests and displays lineage across complex
  multi-source pipelines
- Dagster: asset graph makes lineage explicit; freshness and SLA monitoring
  built in

### 4.3 Large-Scale Observability Infrastructure

- Challenges at scale:
  - Logging millions of predictions per day without impacting serving latency
  - Storing and querying large prediction logs efficiently
  - Sampling strategies: log every Nth prediction or log only anomalous ones
- Architecture pattern:
  - Model server -> Kafka -> streaming processor -> time series database ->
    monitoring dashboard

- TUTORIAL: DataHub - Integrate DataHub with a production ML pipeline to track
  data lineage and model metadata; use AI to surface which upstream data changes
  could explain a model degradation

- TUTORIAL: Dagster - Use Dagster's asset observability features to monitor data
  freshness and quality across a multi-step pipeline; use AI to generate
  freshness SLAs from pipeline descriptions

- TUTORIAL: Marquez - Set up Marquez as an OpenLineage metadata server for a
  multi-step ML pipeline; use AI to query the lineage graph and identify
  the root cause of a simulated data quality incident
