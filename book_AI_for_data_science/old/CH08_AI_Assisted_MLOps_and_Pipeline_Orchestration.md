# Chapter 8: AI-Assisted MLOps and Pipeline Orchestration

## 1. AI-Assisted Pipeline Code Generation

### 1.1 From Notebook to Pipeline

- Notebooks are great for exploration but poor for production:
  - No dependency management between cells
  - Hard to test individual steps
  - Not reproducible without running top to bottom
- Pipeline frameworks enforce structure:
  - Nodes: individual functions that transform data
  - Edges: dependencies between nodes (output of one is input of next)
  - Catalog: declarative definition of all data inputs and outputs
- AI role: scaffold the pipeline structure from a notebook; generate node
  functions from existing cell code; create the catalog definition

### 1.2 Declarative Pipeline Frameworks

- **Kedro**: data science project framework with catalog, node, and pipeline
  abstractions
  - AI generates kedro nodes from Python functions
  - Catalog YAML defines data sources declaratively
- **Prefect**: workflow orchestration with retry logic, caching, and alerting
  - Flows are Python functions decorated with `@flow` and `@task`
  - AI adds failure handling and caching based on pipeline description
- **Luigi**: batch workflow management; tasks with explicit `requires()` and
  `output()` dependencies

### 1.3 Code Generation Patterns for Pipelines

- Scaffold pattern: AI generates the full pipeline skeleton from a description
  of the data flow
  - Input: "Ingest CSV -> clean missing values -> engineer features -> train
    model -> evaluate -> save model"
  - Output: Kedro pipeline with 5 nodes and catalog entries
- Review pattern: AI reviews an existing pipeline for missing error handling,
  missing logging, and potential data leakage

- TUTORIAL: Kedro - Use AI to scaffold a Kedro data pipeline from a natural
  language description of the data flow; review the generated nodes and
  connections; run the pipeline end-to-end

- TUTORIAL: Prefect - Generate a Prefect workflow from an existing notebook
  using AI; add retry logic, caching, and notifications suggested by the AI
  review of the pipeline's failure modes

- TUTORIAL: Luigi - Build a Luigi batch pipeline with multiple dependent stages;
  use AI to generate the task definitions from a textual description of the
  data flow; add email notification on failure

---

## 2. Containerization and Deployment with AI

### 2.1 Containerization for ML Models

- Docker containers package the model, runtime, and dependencies into a
  portable unit
  - Eliminates "works on my machine" problems
  - Enables reproducible serving environments
- Key Dockerfile elements for ML:
  - Base image: Python version, GPU support (CUDA)
  - Dependency installation: requirements.txt
  - Model artifacts: copy serialized model into the image
  - Entrypoint: start the serving process

### 2.2 Model Serving Patterns

- **Online inference**: real-time REST API serving individual predictions
  - BentoML: packages model + preprocessing as a service; generates Dockerfile
  - FastAPI: lightweight REST API framework; AI generates endpoint code
- **Batch inference**: scheduled scoring of large datasets
  - Kedro or Prefect pipelines
- **Streaming inference**: process events as they arrive
  - Kafka consumer + online model

### 2.3 AI-Assisted Deployment Code

- AI generates BentoML service definitions:
  - "Create a BentoML service that accepts a JSON payload with 5 numeric
    features and returns the fraud probability"
- AI generates Dockerfiles from requirements:
  - "Write a Dockerfile for a Python 3.11 sklearn model serving with FastAPI"
- AI generates infrastructure-as-code (Terraform, Kubernetes manifests) from
  deployment descriptions

- TUTORIAL: bentoml - Package a trained model as a BentoML service; use AI to
  generate the service definition and Dockerfile; deploy locally and test the
  REST API

- TUTORIAL: Dagster - Build a Dagster asset graph for an ML pipeline; use AI to
  add partitioning and freshness policies appropriate for the update cadence

- TUTORIAL: fastapi - Build a model serving REST API with FastAPI; use AI to
  generate the endpoint code, request/response schemas, and OpenAPI
  documentation; containerize with Docker

---

## 3. CI/CD for ML with AI Assistance

### 3.1 ML-Specific CI/CD Challenges

- Traditional CI/CD tests code correctness; ML CI/CD must also test:
  - Data quality: does incoming data meet schema expectations?
  - Model performance: does the retrained model meet the performance threshold?
  - Data and model drift: has the data distribution shifted since last training?
- ML CD (continuous delivery) = automatically retrain and deploy when data
  changes or performance degrades

### 3.2 Pipeline CI/CD Tools

- DVC pipelines: define training stages in `dvc.yaml`; DVC reruns only the
  stages whose inputs have changed; integrates with GitHub Actions
- Kubeflow pipelines: define multi-step ML workflows on Kubernetes; GPU
  scheduling and parallel training
- Meltano: ELT platform for data ingestion; integrates dbt, Airflow, and
  Singer-based connectors for end-to-end ML data pipelines

### 3.3 AI in CI/CD

- AI generates the CI pipeline YAML from a description of the workflow:
  - "Write a GitHub Actions workflow that runs data validation, model training,
    and performance evaluation on every push to main"
- AI reviews the CI pipeline for missing steps:
  - "What failure modes does this pipeline not handle?"
- AI generates test suites for pipeline components

- TUTORIAL: DVC - Set up a DVC-based CI pipeline that retrains and evaluates a
  model on every data update; use AI to generate the pipeline stages from a
  description of the training workflow

- TUTORIAL: Kubeflow - Deploy a Kubeflow pipeline for model training and serving
  on Kubernetes; use AI to generate the pipeline YAML definition from a
  high-level description

- TUTORIAL: Meltano - Build an ELT data ingestion pipeline with Meltano; use AI
  to configure the Singer taps and targets for ingesting data from an external
  API into a data warehouse

---

## 4. Experiment Tracking and Model Registry

### 4.1 The Need for a Model Registry

- Model registry = versioned catalog of trained models
  - Maps model version to the experiment that produced it
  - Tracks the model's lifecycle: staging -> production -> archived
  - Provides audit trail: who approved this model, when, and why?
- Required for regulated industries (finance, healthcare) and any production
  ML system

### 4.2 Promoting Models Through the Registry

- Promotion criteria: model performance on held-out dataset >= threshold
  - AI generates the promotion criteria from business requirements:
    - "The production model must have AUC-ROC >= 0.85 on the last month's data"
  - AI writes the evaluation script that checks the criteria before promotion

### 4.3 Lineage and Reproducibility

- Full lineage: data version -> code version -> experiment -> model version
- LakeFS: Git-like version control for data lakes
  - Enables branching and merging of datasets
  - Links a model version to the exact data snapshot used for training
- Metaflow: human-centric workflow framework; tracks namespace-isolated
  experiments across team members

- TUTORIAL: mlflow - Set up an MLflow Model Registry; promote models through
  staging, production, and archived states; use AI to write the evaluation
  criteria for each transition gate

- TUTORIAL: Metaflow - Build a Metaflow workflow with branching experiment
  paths; use the namespace system to isolate experiments across team members

- TUTORIAL: LakeFS - Version a training dataset in LakeFS; link the model
  version to the exact data branch used for training; demonstrate rollback to
  a previous data version and model retraining
