# Benchmarking Data Science Agents: A Comparative Study

## Description

- Data science benchmarks are structured evaluation frameworks that measure how
  well AI agents perform tasks like data analysis, machine learning engineering,
  and multi-step reasoning — mimicking the real workflow of a data scientist
- These benchmarks vary in scope from narrow library-level coding tasks
  (pandas/NumPy) to full end-to-end ML pipelines, making direct comparisons
  non-trivial and scientifically interesting
- Leaderboards from benchmarks like SWE-bench, GAIA, and MLE-Bench are
  publicly available and continuously updated, providing rich, real-world
  performance data on state-of-the-art LLM agents
- A unified comparison across benchmarks reveals capability gaps — an agent
  may excel at isolated coding tasks but fail at multi-tool orchestration or
  long-horizon planning
- Understanding benchmark design choices (task style, evaluation metric,
  difficulty) is as important as the scores themselves, since poor benchmark
  design can mislead model selection in production settings
- This project combines data collection, visualization, and critical analysis to
  produce an empirical, reproducible study of where AI agents succeed and fail
  as data scientists

## Project Objective

The goal of this project is to systematically compare publicly available data
science benchmarks for AI agents — including DataSciBench, DSBench, MLE-Bench,
GAIA, and SWE-bench — by collecting evaluation results, analyzing benchmark
design dimensions (task style, metric, difficulty), and producing visualizations
that reveal capability gaps across frontier LLM agents. Students will build a
reproducible analytical pipeline that ingests leaderboard data, applies
clustering and correlation analysis, and surfaces actionable insights about
which benchmarks best predict real-world data science competence.

## Dataset Suggestions

| Benchmark | What it evaluates | Task style | Why it matters | URL |
|---|---|---|---|---|
| **DataSciBench (2025)** | Analytical reasoning + metric judgement + coding | Open-ended analysis questions | Tests if an agent can *think like a data scientist* | https://arxiv.org/abs/2502.13897 |
| **DSBench (ICLR 2025)** | Full DS workflow | Multi-step Kaggle-like projects | Closest to real end-to-end analyst work | https://github.com/LiqiangJing/DSBench |
| **DS-Bench (code DS)** | Library-level DS programming | Isolated coding tasks | Measures practical pandas/NumPy competence | https://arxiv.org/abs/2505.15621 |
| **TAM-Bench** | AutoML competition performance | Submission-score optimization | Autonomous model-building agents | https://arxiv.org/abs/2509.09321 |
| **MLE-Bench** | ML engineering pipelines | Training + infra + evaluation | Production ML engineer capability | https://github.com/openai/mle-bench |
| **TheAgentCompany** | Workplace task execution | Multi-tool workflows | Long-horizon job-like behavior | https://github.com/TheAgentCompany/TheAgentCompany |
| **SWE-bench** | Complex coding reliability | Debugging real repos | Technical reliability proxy | https://www.swebench.com/ |
| **GAIA** | Multi-step reasoning | Tool-use reasoning problems | Planning intelligence for agents | https://huggingface.co/gaia-benchmark |
| **MSC-Bench** | Tool orchestration | Multi-tool coordination | Whether agent can operate tools coherently | https://arxiv.org/abs/2510.19423 |

- **GAIA Benchmark Leaderboard (HuggingFace)**
  - Source: HuggingFace Datasets / HuggingFace Spaces leaderboard
  - URL: https://huggingface.co/datasets/gaia-benchmark/GAIA and
    https://huggingface.co/spaces/gaia-benchmark/leaderboard
  - Contains: 466 multi-step reasoning questions at three difficulty levels,
    plus model accuracy scores from dozens of submitted agents
  - Access: Fully public; no authentication needed; dataset loadable via
    `datasets` library

- **SWE-bench Leaderboard (princeton-nlp / HuggingFace)**
  - Source: Princeton NLP Group via HuggingFace Datasets
  - URL: https://huggingface.co/datasets/princeton-nlp/SWE-bench and
    https://www.swebench.com/
  - Contains: 2,294 real GitHub issues from 12 Python repos; model patch
    success rates tracked on a public leaderboard
  - Access: Fully public; dataset loadable via `datasets` library; leaderboard
    results scraped or fetched from swebench.com JSON endpoints

- **DSBench Tasks and Evaluation Results (GitHub)**
  - Source: LiqiangJing/DSBench GitHub repository
  - URL: https://github.com/LiqiangJing/DSBench
  - Contains: 466 data analysis tasks and 74 ML modeling tasks derived from
    Kaggle competitions; includes ground-truth answers and reported model
    scores for GPT-4, Claude, and Gemini variants
  - Access: Fully public; clone the repo or download CSVs directly from the
    repository

- **MLE-Bench Results and Task Metadata (GitHub)**
  - Source: OpenAI / openai/mle-bench GitHub repository
  - URL: https://github.com/openai/mle-bench
  - Contains: 75 Kaggle competition tasks repurposed as ML engineering
    challenges; includes agent performance metrics (medal rate, competition
    score percentile) for several OpenAI and open-source agents
  - Access: Fully public; task metadata and evaluation results available in
    the repository's `results/` directory

## Tasks

- **Data Collection and Schema Design**: Download leaderboard results and task
  metadata from the four datasets above; design a unified schema (benchmark
  name, model name, score, task category, difficulty) and store in a clean
  CSV or SQLite database
- **Benchmark Taxonomy Analysis**: Classify each benchmark across dimensions
  such as task style (coding, reasoning, pipeline), evaluation metric
  (accuracy, patch rate, score percentile), and required tools; produce a
  structured comparison table and heatmap
- **Cross-Benchmark Score Correlation**: For models that appear in multiple
  benchmark leaderboards, compute pairwise Pearson/Spearman correlations
  between scores to identify which benchmarks measure overlapping vs.
  complementary skills
- **Capability Gap Visualization**: Build scatter plots and radar charts
  showing how frontier models (GPT-4o, Claude 3.5, Gemini 1.5) rank
  differently across benchmarks, highlighting where strong models are
  surprisingly weak
- **Difficulty and Task-Type Breakdown**: Stratify benchmark results by
  difficulty tier or task category and analyze whether agent rankings are
  stable across easy vs. hard tasks
- **Reproducibility Audit**: Select one small benchmark subset (e.g., 10–20
  GAIA level-1 questions) and attempt to replicate reported results using a
  free-tier API (e.g., Groq + Llama-3, or HuggingFace Inference API); report
  discrepancies and potential causes

## Bonus Ideas

- **Meta-benchmark Score Aggregation**: Design a weighted composite score that
  combines results from all benchmarks to produce a single "data science
  agent ranking" and compare it to existing rankings like LMSYS Chatbot Arena
- **Benchmark Saturation Analysis**: Check whether top benchmark scores have
  plateaued over time using date-stamped leaderboard snapshots; use linear
  regression to project when benchmarks will be "solved"
- **Task Difficulty Prediction**: Train a simple classifier (logistic
  regression or gradient boosting) to predict whether a given model will pass
  a specific task type, using benchmark metadata as features
- **Cost-Performance Tradeoff**: Combine public model pricing data (e.g., from
  OpenRouter or provider pricing pages) with benchmark scores to compute a
  score-per-dollar metric across models
- **Design Your Own Mini-Benchmark**: Create 10–15 novel data science tasks
  not covered by existing benchmarks (e.g., debugging a broken pandas pipeline,
  interpreting a seaborn plot), run them against a free model, and evaluate
  systematically

## Useful Resources

- GAIA Benchmark paper and dataset:
  https://huggingface.co/datasets/gaia-benchmark/GAIA
- SWE-bench official site and leaderboard: https://www.swebench.com/
- DSBench GitHub repository: https://github.com/LiqiangJing/DSBench
- MLE-Bench GitHub repository: https://github.com/openai/mle-bench
- DataSciBench paper (arXiv 2025): https://arxiv.org/abs/2502.13897
