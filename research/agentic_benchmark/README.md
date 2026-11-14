# Benchmarking Agentic Reasoning for Data Science
### DATA605 Spring 2026 | University of Maryland
### Author: Shobha Gupta | Issue #435

---

## What Is This Project? (For Non-Technical Readers)

Imagine you hire three different assistants to analyze medical data and predict heart disease.

- **Assistant 1 (Single Agent — like ChatGPT):** Reads the data once, builds a model, gives you the answer. Done. No second opinions.
- **Assistant 2 (AutoGluon):** Tries every possible mathematical approach automatically and picks the best one. Fast but blind to obvious mistakes.
- **Assistant 3 (This Project — LangGraph):** Has a team of 5 specialists who work in a loop. If one specialist makes a mistake, a reviewer catches it and sends the work back for correction.

This project builds Assistant 3 and proves it produces more reliable, robust results — especially when the data is messy or corrupted.

---

## The Problem With Single-Pass AI Systems

When ChatGPT or similar tools analyze data, they do it once and report whatever result they get. If the data has problems — corrupt values, misleading features, class imbalance — the system reports wrong results with full confidence.

**Real example from this project:**
When analyzing NYC Taxi data, a naive system reported 94% accuracy. Sounds great. But our system's Quality Reviewer noticed that one feature (`payment_type`) was directly leaking the answer — cash riders never tip, card riders almost always do. Remove that feature and accuracy drops to 56%. The single-pass system never caught this. Ours did.

---

## Description

This project moves beyond evaluating third-party "black box" tools to engineering a custom, stateful multi-agent system using LangGraph. While standard agents (like ChatGPT) follow linear, one-shot processes, this research builds a cyclic architecture where agents can plan, execute, critique, and self-correct. By developing an internal "Analyst-Reviewer" loop, the project explores the frontier of Agentic Reasoning — testing whether a structured graph of specialized agents can outperform monolithic AI models in reliability, code quality, and handling "adversarial" or "noisy" data science tasks.

| Type | Name | Description | Website | Strength |
|---|---|---|---|---|
| Notebook agent | ChatGPT Advanced Data Analysis | Upload data → automatic cleaning, analysis, modeling | https://chat.openai.com | Fast exploratory analysis |
| AutoML agent | AutoGluon | Automated model selection, feature engineering, tuning | https://auto.gluon.ai | Strong tabular ML performance |
| Multi-agent research system | Microsoft AutoGen | Agents collaborate to plan experiments and critique results | https://github.com/microsoft/autogen | Research workflows |
| Workflow agent | LangGraph | Stateful agent graphs for long-running analytical pipelines | https://langchain-ai.github.io/langgraph | Persistent reasoning loops |

---

## Project Objective

The primary goal is to benchmark the efficacy of stateful multi-agent orchestration against single-agent and AutoML baselines. This project aims to answer:

- Can a cyclic multi-agent graph (LangGraph) significantly reduce hallucinations and logical errors compared to single-agent assistants?
- Does a "Reviewer" node in an agentic workflow produce more reliable results than one-shot generation?
- How do different agent architectures (Linear vs. Cyclic vs. AutoML) recover when faced with corrupted or ambiguous data?

---

## How The System Works (The Hospital Analogy)

Think of it like a hospital quality control process:

Patient Data → Triage Nurse → Lab Technician → Doctor → Senior Reviewer
↓
"This diagnosis looks wrong"
↓
Send back to Lab Technician or Doctor
↓
Final Report (only when correct)

### The 5 Nodes

**Node 1 — The Planner (Triage Nurse)**
Analyzes the dataset schema and sets the strategy. Identifies problem type (classification vs regression), risks (imbalance, missing values, noisy data), and recommends an approach. Uses Llama 3.2 to reason about the data.

**Node 2 — The EDA Analyst (Lab Technician)**
Performs exploratory data analysis, cleans data, and identifies outliers. Handles missing values using clinically appropriate methods — mode for categorical columns, median for continuous. Generates visualizations for understanding the data.

**Node 3 — The ML Architect (Doctor)**
Selects algorithms (Random Forest, Logistic Regression, Gradient Boosting), trains models, and picks the best one by F1 score. Generates SHAP values to explain why the model makes each prediction.

**Node 4 — The Quality Reviewer (Senior Reviewer — The Star)**
Acts as the scientific guardrail. Checks:
- **Recall:** Is the model catching real cases or just predicting the majority class?
- **SHAP Dominance:** Is one feature suspiciously dominant? That signals data leakage.
- **Routing Decision:** retry_ml → back to ML Architect, retry_eda → back to EDA Analyst, pass → Report Writer

**Node 5 — The Report Writer**
Synthesizes the final journey — documenting results, errors caught, corrections made, and key findings with SHAP explanations.

---

## The Quality Reviewer Rules (Guardrails)

### Statistical Logic Gate
- **Leakage Detection:** SHAP dominance check — if one feature accounts for >95% of importance, flag as leakage
- **Imbalance Audit:** Rejects models that only report accuracy for imbalanced datasets — recall must be checked
- **Impossible Values:** Flags unrealistic data points (negative taxi fares, negative trip durations) for re-cleaning

### Explainability Gate
- **SHAP Consistency:** Verifies feature importance aligns with EDA findings
- **Logical Grounding:** Rejects generic explanations in favor of data-backed insights

---

## Datasets Used

### Heart Disease (303 patients, 14 clinical features)
- **Source:** Kaggle — UCI Heart Disease Dataset
- **URL:** https://www.kaggle.com/datasets/redwankarimsony/heart-disease-uci
- **Target:** Binary — presence of heart disease (0/1)
- **Challenge:** Missing values in ca and thal (critical clinical features)
- **Key finding:** `thalach` (max heart rate) and `cp` (chest pain) are top predictors

### NYC Yellow Taxi (2.96M trips → 50K sample)
- **Source:** NYC TLC Trip Record Data
- **URL:** https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- **Target:** Binary — did the passenger tip? (0/1)
- **Challenge:** 333,403 impossible values removed (negative fares, negative durations)
- **Key finding:** payment_type caused data leakage — AUC dropped from 0.94 to 0.56 after removal

### Amazon Beauty Reviews (701K reviews → 50K sample)
- **Source:** HuggingFace Hub — McAuley-Lab/Amazon-Reviews-2023
- **URL:** https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- **Target:** Binary — Positive (4-5 stars) vs Negative (1-3 stars)
- **Challenge:** 71/29 class imbalance
- **Key finding:** Title length is the strongest predictor of review sentiment

---

## Benchmark Results

### Clean Data Performance

| Dataset | LangGraph F1 | Single Agent F1 | AutoGluon F1 | LangGraph Wins? |
|---|---|---|---|---|
| Heart Disease | **88.14%** | 88.14% | 73.08% | ✅ vs AutoGluon |
| NYC Taxi | **89.41%** | 88.42% | 89.27% | ✅ |
| Amazon Reviews | **83.27%** | 76.75% | 82.74% | ✅ |

### Under Adversarial Conditions
(15% label flipping + 5% impossible values injected)

| Dataset | LangGraph F1 | Single Agent F1 | Advantage |
|---|---|---|---|
| Heart Disease | **68.00%** | 58.33% | +16.6% |
| NYC Taxi | **83.56%** | 81.28% | +2.8% |
| Amazon Reviews | **78.82%** | 70.49% | +11.8% |

**Key insight:** On clean data, systems perform similarly. Under adversarial conditions, LangGraph degrades significantly less because the Quality Reviewer catches errors before they reach the final report.

---

## Project Structure

notebooks/
├── langgraph_agent.API.ipynb        # Technical reference — all nodes documented
├── langgraph_agent.example.ipynb   # Step-by-step walkthrough for new users
├── Heart_Disease_eda.png            # EDA visualization
├── Heart_Disease_shap.png           # SHAP feature importance
├── NYC_Taxi_eda.png
├── NYC_Taxi_shap.png
├── Amazon_Reviews_eda.png
└── Amazon_Reviews_shap.png
requirements.txt                     # All dependencies with pinned versions
Dockerfile                           # Reproducible containerized environment
docker_build.sh                      # Build script
docker_name.sh                       # Image naming configuration

---

## How To Run

### Prerequisites
1. Install Ollama: https://ollama.com/download
2. Pull Llama 3.2: `ollama pull llama3.2`
3. Install dependencies: `pip install -r requirements.txt`

### Run With Docker
```bash
bash docker_build.sh
docker run -p 8888:8888 gpsaggese/langgraph_agentic_ds
```

### Run Locally
1. Start Ollama in terminal: `ollama serve`
2. Open Jupyter: `jupyter notebook`
3. Open `notebooks/langgraph_agent.example.ipynb`
4. Run all cells from top to bottom

---

## Tasks Completed

1. Environment Setup — Version pinning for reproducibility
2. Graph Construction — StateGraph with conditional edges in LangGraph
3. Benchmarking — All competitors run against all 3 datasets
4. Adversarial Reliability Test — Label flipping and impossible value injection
5. Interpretability — SHAP analysis with Quality Reviewer validation

![Architecture diagram of the LangGraph multi-agent workflow](Benchmarkingworkflow.png)
---

## Useful Resources
- **LangGraph Documentation:** https://langchain-ai.github.io/langgraph
- **AutoGluon Documentation:** https://auto.gluon.ai/stable/tutorials/tabular/tabular-quick-start.html
- **Microsoft AutoGen GitHub:** https://github.com/microsoft/autogen
- **OpenML Benchmark Suite:** https://www.openml.org/search?type=benchmark
