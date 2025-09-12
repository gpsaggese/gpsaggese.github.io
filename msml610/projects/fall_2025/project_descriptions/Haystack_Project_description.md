**Description**

Haystack is an open‑source framework for retrieval pipelines and agentic workflows. Agents decide when to call tools (retrievers, aggregators, functions), keep memory, and run multi‑step reasoning. Strong for production RAG and tool‑use orchestration.

Technologies Used
Haystack

- Agent with tool‑calling + ToolInvoker execution.
- Pipeline architecture (branching/looping) and vector DB integrations.
- Works with many LLM providers and multimodal components.
- Deployable with observability/logging options.

---

### Project 1: Titanic Q&A Agent
**Difficulty**: 1 (Easy)

**Project Objective**:
Answer NL questions about Titanic passengers with a tiny RAG/agent setup.

**Dataset Suggestions**:
- Dataset: Titanic passenger data.
- Source: [Kaggle – Titanic Competition](https://www.kaggle.com/competitions/titanic)

**Tasks**:
- Load the CSV into a document store (or table tool) with row‑level docs.
- Add a query tool that filters/aggregates (e.g., survivors by class/gender).
- Configure an Agent to decide when to call the tool vs. answer directly.

**Bonus Ideas (Optional)**:
- Add chart generation for survival rates by Pclass/Sex.

---

### Project 2: COVID‑19 RAG Pipeline
**Difficulty**: 2 (Medium)

**Project Objective**:
Daily‑report retrieval + aggregation tools; agent generates a trend summary for a country.

**Dataset Suggestions**:
- Dataset: Johns Hopkins CSSE COVID‑19.
- Source: [GitHub – CSSEGISandData/COVID‑19](https://github.com/CSSEGISandData/COVID-19)

**Tasks**:
- Ingest daily CSVs; tag docs with date/country.
- Tools: aggregation by date; moving‑average computation.
- Agent plans multi‑step reasoning and composes the narrative.

**Bonus Ideas (Optional)**:
- Add vaccination time series and compare pre/post trends.

---

### Project 3: Multimodal Consumer Sentiment Analyzer
**Difficulty**: 3 (Hard)

**Project Objective**:
Combine text reviews with user photos; agent extracts themes, sentiment, and image captions, then merges insights.

**Dataset Suggestions**:
- Dataset: Yelp Open Dataset (reviews + photos).
- Source: [Yelp – Open Dataset](https://business.yelp.com/data/resources/open-dataset/)

**Tasks**:
- Load reviews; store photos/links separately; add multimodal embedders.
- Tools: sentiment analysis, topic modeling, image captioning.
- Agent orchestrates tool calls and compiles a report per category.

**Bonus Ideas (Optional)**:
- Add self‑critique step for the agent to refine outputs.