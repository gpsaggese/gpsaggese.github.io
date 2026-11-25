# Agentic Benchmark: Fair Comparison Results

Author: Shobha Gupta | DATA605 Spring 2026 | Issue #435

## Question

Does a multi-agent LangGraph pipeline with a Reviewer node produce more reliable results than a single-pass pipeline and AutoGluon, especially when the data is corrupted?

## Why this folder exists

My first results said LangGraph won. When I audited them, the comparison was not fair: the systems used different models and different splits, the corruption had no random seed, the test set was tiny, and models were chosen by F1, which rewards predicting everyone as positive. I rebuilt the benchmark so every system is tested the same way, and the numbers below come from that rebuild. The older claims (for example "only LangGraph detected leakage" and "+16.6% under noise") are not supported and should be ignored.

## Setup

- Systems: always positive (baseline), single agent, graph without reviewer, graph with reviewer, AutoGluon (60 second limit).
- Every system gets the same stratified split, the same features and the same candidate models. The model is chosen by cross-validated MCC on training data only.
- Corruption is seeded and applied to training data only: label flips (5, 15, 30%) plus impossible values (2, 5, 5%). The corrupted rows are recorded, so the Reviewer can be scored as a detector.
- Seeds: Heart Disease 10, Amazon Reviews 5, NYC Taxi 5. Tables show mean and standard deviation.
- MCC is the Matthews Correlation Coefficient. 0 means chance and 1 means perfect. An always-positive model scores 0.

## Results (MCC, mean ± standard deviation)

The single agent and the graph without the reviewer use the same models on the same data, so their scores are identical.

| Dataset | Label noise | Single agent / graph, no reviewer | Graph + reviewer | AutoGluon |
|---|---|---|---|---|
| Heart Disease | 0% | 0.646 ± 0.084 | 0.638 ± 0.106 | 0.642 ± 0.093 |
| Heart Disease | 5% | 0.654 ± 0.102 | 0.632 ± 0.073 | 0.623 ± 0.096 |
| Heart Disease | 15% | 0.651 ± 0.106 | 0.562 ± 0.133 | 0.605 ± 0.098 |
| Heart Disease | 30% | 0.504 ± 0.102 | 0.443 ± 0.199 | 0.468 ± 0.140 |
| Amazon Reviews | 0% | 0.586 ± 0.018 | 0.586 ± 0.018 | 0.588 ± 0.021 |
| Amazon Reviews | 5% | 0.574 ± 0.015 | 0.571 ± 0.009 | 0.581 ± 0.020 |
| Amazon Reviews | 15% | 0.549 ± 0.015 | 0.550 ± 0.026 | 0.573 ± 0.027 |
| Amazon Reviews | 30% | 0.519 ± 0.024 | 0.517 ± 0.028 | 0.517 ± 0.059 |
| NYC Taxi (payment_type dropped) | 0% | 0.209 ± 0.039 | 0.191 ± 0.043 | 0.139 ± 0.067 |
| NYC Taxi (payment_type dropped) | 15% | 0.212 ± 0.041 | 0.183 ± 0.063 | 0.125 ± 0.051 |
| NYC Taxi (payment_type kept) | 0% | 0.911 ± 0.006 | 0.911 ± 0.006 | not run |
| NYC Taxi (payment_type kept) | 15% | 0.913 ± 0.007 | 0.913 ± 0.007 | not run |
| Heart Disease (leak injected) | 0% | 1.000 ± 0.000 | 0.650 ± 0.076 | not run |

Balanced accuracy and specificity for every row are in `results/`.

## How well the Reviewer found the corrupted data

| Dataset | Impossible values found | Flipped labels found | Notes |
|---|---|---|---|
| Heart Disease | all (precision 1.0, recall 1.0) | precision 0.31 to 0.63, recall 0.09 to 0.48 | Also drops some good rows on clean data |
| Amazon Reviews | all | none | Label check never fired |
| NYC Taxi | all | none (precision 0.86, recall 0.17 with payment_type kept) | |
| Injected leak (Heart) | not applicable | not applicable | Leak flagged in 10 of 10 seeds |

## What this shows

1. On accuracy there is no clear winner. LangGraph, the single agent and AutoGluon are within each other's spread on Heart Disease and Amazon, and the simple models beat AutoGluon on taxi.
2. The Reviewer does not improve accuracy. On Heart Disease with 15% noise it is slightly worse (0.562 vs 0.651), because its label check removes good rows from a training set of about 240.
3. The Reviewer is useful for data integrity. It finds every injected impossible value, and it flagged an injected leaked feature in all 10 seeds. The other systems scored a false 1.0 on that test, and the Reviewer reported the true 0.65.
4. It misses the natural `payment_type` leak in NYC Taxi. Keeping that one column lifts every system from MCC 0.21 to 0.91, and no system removed it.
5. Rules route the graph. The LLM only writes the plan note and the summary, so these runs do not test LLM decision making. In the LLM run on Amazon the results match the rules run exactly.

## Limits

- The single agent is a stand-in for a one-pass assistant. I did not run ChatGPT.
- The leak test uses an injected feature that is easy to spot by design.
- AutoGluon used a 60 second limit and the medium quality preset.
- The Reviewer's impossible-value bounds are written by hand per dataset.
- A redesigned Reviewer that must prove a fix helps before applying it is still in development and is not in these results.

## Files

- `agentic_bench.py`: the benchmark harness (splits, systems, Reviewer, corruption, metrics).
- `run_heart.py`, `run_heart_ag.py`, `run_amazon_ag.py`, `run_taxi.py`, `run_taxi_ag.py`, `run_benchmark.py`, `run_leak.py`: one script per experiment.
- `results/`: the raw result CSVs the tables above come from.

## How to run

1. Start Ollama if you want LLM summaries: `ollama serve`, then `ollama pull llama3.2`.
2. Install: `pip install langgraph langchain-ollama scikit-learn pandas numpy vaderSentiment autogluon.tabular`
3. Put the data where the scripts expect it (Heart Disease CSV, NYC Taxi parquet, Amazon Beauty CSV) and edit the paths at the top of each script.
4. Run one script at a time, for example `python run_heart.py`. Results are written to `outputs/`.
