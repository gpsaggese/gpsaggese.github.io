# Comparison of Data Science Agents

## Status

- **Status:**: in_progress
- **Complete Specs:**: 80%
- **Assignee:**: TBD

## Core Idea

- Design and execute a controlled empirical study that benchmarks at least three
  Data Science Agent tools across multiple real-world datasets and task types
  - Select agents from different categories: AutoML, notebook assistant,
    multi-agent
  - Apply each agent to the same datasets and tasks
  - Systematically compare the generated outputs on:
    - Performance metrics
    - Code quality
    - Runtime
    - Explainability
- **Data Science Agents** are AI-powered autonomous systems
  - They combine large language models (LLMs) with planning, tool use, and memory
  - They perform end-to-end data science workflows, from data loading and
    cleaning through modeling and evaluation, with minimal human intervention
- Agents span a spectrum:
  - Single-purpose AutoML tools (e.g., AutoGluon, PyCaret)
  - Fully conversational notebook assistants (e.g., Jupyter AI, ChatGPT Advanced
    Data Analysis)
  - Multi-agent frameworks that coordinate specialized sub-agents (e.g.,
    Microsoft AutoGen, CrewAI)
- Key capabilities:
  - Automated EDA
  - Feature engineering
  - Algorithm selection
  - Hyperparameter tuning
  - SHAP-based explainability
  - Natural-language code generation for tabular, time-series, and NLP data tasks
- Agents differ significantly in:
  - Autonomy level
  - Reproducibility
  - Local vs. cloud execution
  - Interpretability of generated code
  - Quality of the final models they produce
  - This makes rigorous head-to-head comparison valuable
- Most tools expose a Python SDK or CLI
  - They are accessible in standard Jupyter/Colab environments without
    specialized hardware
- The project teaches students to critically evaluate AI tooling rather than
  accept vendor claims
  - It builds skills in experimental design, benchmarking, and meta-analysis of
    ML pipelines

## Formalization

### Candidate Agents

| Type                        | Name                                              | Description                                                                                            | Website                                  | Strength                      |
| :-------------------------- | :------------------------------------------------ | :----------------------------------------------------------------------------------------------------- | :--------------------------------------- | :---------------------------- |
| General coding agent        | Devin (Cognition AI)                              | Fully autonomous software engineer agent that plans, writes, executes, debugs and iterates on projects | https://cognition.ai                     | End-to-end autonomy           |
| Terminal coding agent       | Open Interpreter                                  | Runs code locally from natural language: manipulates files, data, and notebooks                        | https://openinterpreter.com              | Direct local execution        |
| Notebook agent              | Data Interpreter (ChatGPT Advanced Data Analysis) | Upload data -> automatic cleaning, analysis, modeling, and visualization                               | https://chat.openai.com                  | Fast exploratory analysis     |
| AutoML agent                | AutoGluon                                         | Automated model selection, feature engineering, and tuning pipelines                                   | https://auto.gluon.ai                    | Strong tabular ML performance |
| Experiment agent            | PyCaret                                           | Low-code ML experimentation platform with automated comparisons                                        | https://pycaret.org                      | Rapid benchmarking            |
| Multi-agent research system | Microsoft AutoGen                                 | Agents collaborate to plan experiments, write code, and critique results                               | https://github.com/microsoft/autogen     | Research workflows            |
| Agent framework             | CrewAI                                            | Structured teams of agents performing analysis tasks collaboratively                                   | https://github.com/joaomdmoura/crewai    | Modular workflows             |
| Data analysis agent         | PandasAI                                          | Natural-language interface for pandas data analysis                                                    | https://pandas-ai.com                    | Simple business analytics     |
| Workflow agent              | LangGraph                                         | Stateful agent graphs for long-running analytical pipelines                                            | https://langchain-ai.github.io/langgraph | Persistent reasoning loops    |
| Notebook automation         | Jupyter AI                                        | AI assistant integrated directly inside notebooks                                                      | https://jupyter.org/ai                   | Familiar DS environment       |

## Key Examples

- **[Example 1]**: [Concrete scenario illustrating the idea]
- **[Example 2]**: [Second scenario, possibly from a different domain]
- **[Example 3]**: [Edge case or failure mode]

## Questions

1. Which agents produce the best models, most readable code, and most useful
   insights, and under what conditions?
2. [Open question 2: what would a proof or counterexample look like?]
3. [Provocative implication: if true, what does this change?]

## Research Topics

- **Custom evaluation rubric**: design a weighted scoring rubric for agent
  comparison
  - E.g., 40% model performance, 30% code quality, 20% explainability, 10%
    runtime
  - Discuss trade-offs in weighting choices
- **Multi-agent pipeline**: use AutoGen or CrewAI to create a collaborative
  pipeline
  - One agent does EDA
  - Another agent selects a model
  - A third agent writes the evaluation report
  - Compare this pipeline to a single-agent approach
- **LLM-as-judge**: use an open LLM (e.g., via HuggingFace Inference API or
  Ollama locally) to automatically score the narrative explanations produced by
  each agent for clarity and correctness
- **Cost and carbon estimate**: if cloud-based agents are included, estimate API
  call costs and compute energy usage using tools like CodeCarbon
  (https://codecarbon.io)
  - Discuss sustainability trade-offs
- **Adversarial input**: submit intentionally mislabeled or corrupted data to each
  agent and evaluate robustness
  - Does the agent detect the problem or silently produce bad results?

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: select and download the datasets
  - **Heart Disease Prediction (UCI / Kaggle)**
    - Source: Kaggle: UCI Heart Disease Dataset
    - URL: https://www.kaggle.com/datasets/redwankarimsony/heart-disease-uci
    - Contains: 14 clinical features (age, cholesterol, chest pain type, etc.)
      with a binary target indicating presence of heart disease; ~300 rows
    - Access: Free Kaggle account required; download via
      `kaggle datasets download` CLI or direct CSV link; no authentication token
      needed for manual download
  - **NYC Yellow Taxi Trip Records**
    - Source: NYC Open Data / TLC Trip Record Data
    - URL: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
    - Contains: Pick-up/drop-off timestamps, GPS coordinates, trip distance, fare
      amount, tip, and passenger count; monthly Parquet files (~millions of rows:
      use one month's subset)
    - Access: Fully public, no authentication; direct Parquet download links
      available on the page; recommend sampling 50k rows for laptop use
  - **Air Quality: OpenAQ**
    - Source: OpenAQ public API
    - URL: https://api.openaq.org/v2/measurements (REST, no key required for
      basic access)
    - Contains: Real-time and historical PM2.5, PM10, NO2, O3, CO readings from
      thousands of global monitoring stations with timestamps and GPS
    - Access: Free tier with no API key; query by city, parameter, and date
      range; returns JSON easily loaded with `requests` + `pandas`
  - **Amazon Product Reviews: HuggingFace Datasets**
    - Source: HuggingFace Hub: `McAuley-Lab/Amazon-Reviews-2023`
    - URL: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
    - Contains: Product ratings (1-5 stars), review text, verified purchase flag,
      product category; load a small subset (e.g., "All_Beauty", ~500k rows) with
      `datasets.load_dataset()`
    - Access: Free, no authentication; streamed or downloaded via `datasets`
      library

- Milestone 2: environment setup and agent installation
  - Install and configure at least three chosen agents (e.g., AutoGluon, PyCaret,
    Jupyter AI) in a shared Colab or conda environment
  - Document version pinning for reproducibility

- Milestone 3: baseline EDA comparison
  - Run each agent's automated EDA feature on all datasets
  - Compare the visualizations, statistical summaries, and anomaly reports
    generated by each tool

- Milestone 4: automated modeling and benchmarking
  - Apply each agent to a supervised learning task per dataset: binary
    classification, regression, and sentiment scoring
  - Record held-out accuracy, F1/RMSE, and wall-clock training time

- Milestone 5: code quality review
  - Examine the code generated or executed by each agent
  - Assess readability, modularity, presence of comments, and whether the code
    can be re-run independently of the agent

- Milestone 6: explainability and reasoning analysis
  - Extract feature importance rankings or SHAP values from each agent's output
  - Compare how well each agent explains _why_ its model makes predictions

- Milestone 7: error and failure mode analysis
  - Deliberately feed each agent a dataset with missing values or class imbalance
  - Document how each agent handles or reports these data quality issues

- Milestone 8: summary scorecard
  - Build a comparative table/dashboard scoring each agent across all tasks
  - Use a rubric that weights accuracy, speed, code quality, and explainability

## References

- **AutoGluon Documentation**: Tabular prediction quickstart and benchmarks:
  https://auto.gluon.ai/stable/tutorials/tabular/tabular-quick-start.html
- **PyCaret Documentation**: Compare models and AutoML workflow:
  https://pycaret.gitbook.io/docs/
- **Jupyter AI GitHub**: Installation guide and supported LLM backends:
  https://github.com/jupyterlab/jupyter-ai
- **Microsoft AutoGen GitHub**: Multi-agent conversation examples including data
  science workflows: https://github.com/microsoft/autogen
- **OpenML Benchmark Suite**: Curated tabular datasets and standardized evaluation
  protocols for AutoML comparison studies:
  https://www.openml.org/search?type=benchmark
