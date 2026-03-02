# Comparison of Data Science Agents

## Description

- **Data Science Agents** are AI-powered autonomous systems that combine large
  language models (LLMs) with planning, tool use, and memory to perform
  end-to-end data science workflows — from data loading and cleaning through
  modeling and evaluation — with minimal human intervention
- These agents span a spectrum from single-purpose AutoML tools (e.g.,
  AutoGluon, PyCaret) to fully conversational notebook assistants (e.g.,
  Jupyter AI, ChatGPT Advanced Data Analysis) and multi-agent frameworks (e.g.,
  Microsoft AutoGen, CrewAI) that coordinate specialized sub-agents
- Key capabilities include automated EDA, feature engineering, algorithm
  selection, hyperparameter tuning, SHAP-based explainability, and natural-
  language code generation for tabular, time-series, and NLP data tasks
- Agents differ significantly in autonomy level, reproducibility, local vs.
  cloud execution, interpretability of generated code, and the quality of the
  final models they produce — making rigorous head-to-head comparison valuable
- Most tools expose a Python SDK or CLI, making them accessible in standard
  Jupyter/Colab environments without specialized hardware
- This project teaches students to critically evaluate AI tooling rather than
  accept vendor claims, building skills in experimental design, benchmarking,
  and meta-analysis of ML pipelines

| Type | Name | Description | Website | Strength |
|---|---|---|---|---|
| General coding agent | Devin (Cognition AI) | Fully autonomous software engineer agent that plans, writes, executes, debugs and iterates on projects | https://cognition.ai | End-to-end autonomy |
| Terminal coding agent | Open Interpreter | Runs code locally from natural language — manipulates files, data, and notebooks | https://openinterpreter.com | Direct local execution |
| Notebook agent | Data Interpreter (ChatGPT Advanced Data Analysis) | Upload data → automatic cleaning, analysis, modeling, and visualization | https://chat.openai.com | Fast exploratory analysis |
| AutoML agent | AutoGluon | Automated model selection, feature engineering, and tuning pipelines | https://auto.gluon.ai | Strong tabular ML performance |
| Experiment agent | PyCaret | Low-code ML experimentation platform with automated comparisons | https://pycaret.org | Rapid benchmarking |
| Multi-agent research system | Microsoft AutoGen | Agents collaborate to plan experiments, write code, and critique results | https://github.com/microsoft/autogen | Research workflows |
| Agent framework | CrewAI | Structured teams of agents performing analysis tasks collaboratively | https://github.com/joaomdmoura/crewai | Modular workflows |
| Data analysis agent | PandasAI | Natural-language interface for pandas data analysis | https://pandas-ai.com | Simple business analytics |
| Workflow agent | LangGraph | Stateful agent graphs for long-running analytical pipelines | https://langchain-ai.github.io/langgraph | Persistent reasoning loops |
| Notebook automation | Jupyter AI | AI assistant integrated directly inside notebooks | https://jupyter.org/ai | Familiar DS environment |

## Project Objective

Design and execute a controlled empirical study that benchmarks at least three
Data Science Agent tools across multiple real-world datasets and task types.
The project aims to answer: *Which agents produce the best models, most
readable code, and most useful insights — and under what conditions?*
Students will select agents from different categories (AutoML, notebook
assistant, multi-agent), apply each to the same datasets and tasks, and
systematically compare performance metrics, code quality, runtime, and
explainability of the generated outputs.

## Dataset Suggestions

- **Heart Disease Prediction (UCI / Kaggle)**
  - Source: Kaggle — UCI Heart Disease Dataset
  - URL: https://www.kaggle.com/datasets/redwankarimsony/heart-disease-uci
  - Contains: 14 clinical features (age, cholesterol, chest pain type, etc.)
    with a binary target indicating presence of heart disease; ~300 rows
  - Access: Free Kaggle account required; download via `kaggle datasets
    download` CLI or direct CSV link; no authentication token needed for
    manual download

- **NYC Yellow Taxi Trip Records**
  - Source: NYC Open Data / TLC Trip Record Data
  - URL: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
  - Contains: Pick-up/drop-off timestamps, GPS coordinates, trip distance,
    fare amount, tip, and passenger count; monthly Parquet files (~millions
    of rows — use one month's subset)
  - Access: Fully public, no authentication; direct Parquet download links
    available on the page; recommend sampling 50k rows for laptop use

- **Air Quality — OpenAQ**
  - Source: OpenAQ public API
  - URL: https://api.openaq.org/v2/measurements (REST, no key required for
    basic access)
  - Contains: Real-time and historical PM2.5, PM10, NO₂, O₃, CO readings
    from thousands of global monitoring stations with timestamps and GPS
  - Access: Free tier with no API key; query by city, parameter, and date
    range; returns JSON easily loaded with `requests` + `pandas`

- **Amazon Product Reviews — HuggingFace Datasets**
  - Source: HuggingFace Hub — `McAuley-Lab/Amazon-Reviews-2023`
  - URL: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
  - Contains: Product ratings (1–5 stars), review text, verified purchase
    flag, product category; load a small subset (e.g., "All_Beauty", ~500k
    rows) with `datasets.load_dataset()`
  - Access: Free, no authentication; streamed or downloaded via
    `datasets` library

## Tasks

- **Environment Setup & Agent Installation**: Install and configure at least
  three chosen agents (e.g., AutoGluon, PyCaret, Jupyter AI) in a shared Colab
  or conda environment and document version pinning for reproducibility
- **Baseline EDA Comparison**: Run each agent's automated EDA feature on all
  datasets; compare the visualizations, statistical summaries, and anomaly
  reports generated by each tool
- **Automated Modeling & Benchmarking**: Apply each agent to a supervised
  learning task per dataset (binary classification, regression, and sentiment
  scoring); record held-out accuracy, F1/RMSE, and wall-clock training time
- **Code Quality Review**: Examine the code generated or executed by each
  agent; assess readability, modularity, presence of comments, and whether the
  code can be re-run independently of the agent
- **Explainability & Reasoning Analysis**: Extract feature importance rankings
  or SHAP values from each agent's output and compare how well each agent
  explains *why* its model makes predictions
- **Error & Failure Mode Analysis**: Deliberately feed each agent a dataset
  with missing values or class imbalance and document how each handles or
  reports these data quality issues
- **Summary Scorecard**: Build a comparative table/dashboard scoring each agent
  across all tasks using a rubric that weights accuracy, speed, code quality,
  and explainability

## Bonus Ideas

- **Custom Evaluation Rubric**: Design a weighted scoring rubric for agent
  comparison (e.g., 40 % model performance, 30 % code quality, 20 %
  explainability, 10 % runtime) and discuss trade-offs in weighting choices
- **Multi-Agent Pipeline**: Use AutoGen or CrewAI to create a collaborative
  pipeline where one agent does EDA, another selects a model, and a third
  writes the evaluation report — then compare this pipeline to a single-agent
  approach
- **LLM-as-Judge**: Use an open LLM (e.g., via HuggingFace Inference API or
  Ollama locally) to automatically score the narrative explanations produced by
  each agent for clarity and correctness
- **Cost & Carbon Estimate**: If cloud-based agents are included, estimate API
  call costs and compute energy usage using tools like CodeCarbon
  (https://codecarbon.io) and discuss sustainability trade-offs
- **Adversarial Input**: Submit intentionally mislabeled or corrupted data to
  each agent and evaluate robustness — does the agent detect the problem or
  silently produce bad results?

## Useful Resources

- **AutoGluon Documentation** — Tabular prediction quickstart and benchmarks:
  https://auto.gluon.ai/stable/tutorials/tabular/tabular-quick-start.html
- **PyCaret Documentation** — Compare models and AutoML workflow:
  https://pycaret.gitbook.io/docs/
- **Jupyter AI GitHub** — Installation guide and supported LLM backends:
  https://github.com/jupyterlab/jupyter-ai
- **Microsoft AutoGen GitHub** — Multi-agent conversation examples including
  data science workflows:
  https://github.com/microsoft/autogen
- **OpenML Benchmark Suite** — Curated tabular datasets and standardized
  evaluation protocols for AutoML comparison studies:
  https://www.openml.org/search?type=benchmark
