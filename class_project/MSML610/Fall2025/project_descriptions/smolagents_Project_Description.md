**Description**

smolagents is a minimalist Python library for building agents that either write and execute Python (CodeAgents) or call tools via structured messages (ToolCallingAgents). It keeps abstractions thin, integrates with the Hugging Face Hub, and can run code in sandboxes like E2B or Docker. Good for compact, code‑centric AutoEDA flows.

Technologies Used
smolagents

- Tiny codebase; quick to stand up CodeAgents that “think in code.”
- ToolCallingAgents for JSON/tool‑calling workflows.
- Model‑agnostic (Hub, OpenAI/Anthropic, local) and modality‑agnostic.
- Optional sandboxed execution (E2B/Docker) for safe Python runs.

---

### Project 1: Iris Code Agent
**Difficulty**: 1 (Easy)

**Project Objective**:
A CodeAgent that loads Iris and returns summary stats by species.

**Dataset Suggestions**:
- Dataset: Iris.
- Source: [UCI – Iris](https://archive.ics.uci.edu/dataset/53/iris)

**Tasks**:
- Create a CodeAgent with a prompt that writes Python (pandas/numpy).
- Compute means/variances per feature and format the output table.
- Save a CSV of the summary to `/outputs`.

**Bonus Ideas (Optional)**:
- Add pairwise correlations or a quick train/test split + accuracy.

---

### Project 2: Customer‑Churn Investigator
**Difficulty**: 2 (Medium)

**Project Objective**:
Two agents: a CodeAgent for data analysis and a ToolCallingAgent to fetch definitions/context. Goal: EDA on churn drivers.

**Dataset Suggestions**:
- Dataset: Telco Customer Churn.
- Source: [Kaggle – Telco Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Tasks**:
- CodeAgent loads CSV, cleans data, computes churn by contract type/tenure.
- ToolCallingAgent answers “what does X column mean?” via web lookup (optional).
- Combine outputs into a concise analyst brief.

**Bonus Ideas (Optional)**:
- Fit a simple logistic regression and report top coefficients.

---

### Project 3: Multimodal Pollution Analyst
**Difficulty**: 3 (Hard)

**Project Objective**:
Analyze Beijing air‑quality time series and relate pollution spikes to weather; optionally incorporate satellite/image features.

**Dataset Suggestions**:
- Dataset: Beijing Multi‑Site Air Quality.
- Source: [UCI – Beijing Multi‑Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijing%2Bmulti%2Bsite%2Bair%2Bquality%2Bdata)

**Tasks**:
- CodeAgent loads hourly pollutant + weather data, cleans and resamples.
- Decompose series, compute correlations, flag smog events.
- (Optional) Add a vision model step if you bring in images.

**Bonus Ideas (Optional)**:
- Forecast next‑day PM2.5 and evaluate with MAE/MAPE.
