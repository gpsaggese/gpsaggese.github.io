---
title: "W&B Weave in 60 Minutes"
draft: true
authors:
    - rbubeck
    - MitaliPurwar21
date: 2026-11-18
description: Learn how to trace and evaluate machine learning models with W&B Weave.
categories:
    - Machine Learning
    - AI Tools
---

TL;DR: Learn how to trace, evaluate, and compare machine learning models using
W&B Weave in 60 minutes, with hands-on examples covering model observability,
evaluation, and Federal Reserve monetary policy analysis.

<!-- more -->

## Tutorial in 30 Seconds

**W&B Weave** is a framework from Weights & Biases for tracing, evaluating,
and understanding AI and machine learning applications.

Key capabilities:

- **Tracing**: Decorate Python functions with `@weave.op` to record inputs,
  outputs, execution times, and nested function calls.
- **Versioned datasets**: Use `weave.Dataset` to organize evaluation examples
  and track dataset versions.
- **Models**: Define reusable prediction interfaces with `weave.Model`.
- **Scorers**: Implement custom evaluation metrics using functions or
  `weave.Scorer` classes.
- **Evaluations**: Use `weave.Evaluation` to compare model performance across
  a common dataset.
- **Observability**: Inspect individual predictions, evaluation results, and
  execution traces in the Weave interface.

Traditional model evaluation often produces aggregate metrics without
preserving the execution details behind individual predictions. Weave
connects evaluation results to model calls, datasets, and traces, making
it easier to investigate errors and compare experiments.

This tutorial's goal is to show you in 60 minutes:

- The core APIs of W&B Weave, including operations, datasets, models,
  scorers, and evaluations.
- How to compare two versions of a monetary policy text classifier using
  synthetic examples.
- How to evaluate machine learning models on a real financial text dataset
  and connect their predictions to downstream financial analysis.

## Official References

- [W&B Weave Documentation](https://weave-docs.wandb.ai/)
- [W&B Weave GitHub Repository](https://github.com/wandb/weave)

## Tutorial Content

This tutorial includes the code, notebooks, and Docker configuration in
[tutorials/Weave](https://github.com/gpsaggese/gpsaggese.github.io/tree/master/tutorials/Weave).

- [`README.md`](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/tutorials/Weave/README.md):
  Instructions and setup for the tutorial environment.
- A Docker system to build and run the environment using our standardized
  approach.
- [`weave.API.ipynb`](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/tutorials/Weave/weave.API.ipynb):
  Tutorial notebook focusing on core Weave building blocks.
    - Trace Python functions and inspect nested calls.
    - Publish and version evaluation datasets and models.
    - Define custom scorers and compare keyword-based classifiers.
    - Inspect evaluation results in the Weave interface.
- [`weave.example.ipynb`](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/tutorials/Weave/weave.example.ipynb):
  End-to-end Federal Open Market Committee (FOMC) tone classification and
  Treasury yield analysis.
    - Train baseline and class-balanced TF-IDF logistic regression models.
    - Evaluate both models on 496 labeled FOMC sentences using accuracy
      and macro-F1.
    - Score the monetary policy tone of 84 FOMC statements from 2015
      through 2024.
    - Align announcements with changes in two-year Treasury yields.
    - Compare Spearman correlations and quantify uncertainty using a
      paired bootstrap.
- [`weave_utils.py`](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/tutorials/Weave/weave_utils.py):
  Shared utility functions for tracing, classification, evaluation,
  financial data processing, and statistical analysis.

## Application Results

The application compares two classifiers using the same training and
evaluation datasets. The balanced classifier adjusts class weights to
account for the imbalance between dovish, hawkish, and neutral sentences.

| Metric | Baseline | Balanced |
|--------|----------|----------|
| Macro-F1 | 0.5180 | 0.5825 |
| Accuracy | 60.08% | 60.28% |
| Spearman correlation | 0.1691 | 0.1773 |

The balanced classifier improves macro-F1 by approximately 12.4% relative
to the baseline, while overall accuracy remains similar.

Both classifiers produce weak positive correlations between consecutive
changes in statement tone and changes in two-year Treasury yields.
Neither correlation is statistically significant at the 5% level.

The paired bootstrap produces a 95% confidence interval of approximately
[-0.1814, 0.1928] for the difference between the two correlations.

These results illustrate an important distinction: improving classification
performance does not necessarily improve the usefulness of predictions
for a downstream financial application. The financial analysis is
exploratory and does not establish a causal relationship.

## Data Sources

- [FOMC Communication Dataset](https://huggingface.co/datasets/gtfintechlab/fomc_communication)
- [Federal Reserve FOMC Statements](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm)
- [U.S. Treasury Interest Rate Statistics](https://home.treasury.gov/resource-center/data-chart-center/interest-rates)