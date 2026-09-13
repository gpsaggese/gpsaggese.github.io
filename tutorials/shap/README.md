# SHAP Tutorial

Learn how to explain machine learning model predictions in 60 minutes using SHAP
(SHapley Additive exPlanations).

## Quick Start

- From the root of the repository:
  ```bash
  > cd tutorials/shap
  > ./docker_build.sh
  > ./docker_jupyter.sh
  ```

- Open your browser to http://localhost:8888 and work through the notebooks in
  order:
  1. `01.API.shap.ipynb`: core SHAP API — Explainer, Explanation, and plots

- For Docker build system details see the
  [project template README](../../class_project/project_template/docker_scripts.README.md)

## Introduction

- Model explainability answers a simple question: _why did the model make this
  prediction?_

- Modern models (gradient boosting, random forests, neural nets) are accurate
  but opaque. You get a number out, with no reason attached. SHAP recovers that
  reason, either for the model as a whole (global) or for a single prediction
  (local).

- Use SHAP when:
  - You must justify a decision to a regulator, customer, or domain expert (e.g.,
    loan denial, medical diagnosis)
  - You want to debug a model that performs well on metrics but behaves strangely
  - You want to check that the model relies on sensible features, not on leakage
    or spurious correlations
  - You need to build trust with stakeholders before deploying

## Core Concepts

- SHAP decomposes any prediction into per-feature contributions that sum to the
  output:
  ```
  prediction[i] = base_value + sum(shap_values[i])
  ```

- This decomposition is grounded in cooperative game theory (Shapley values from
  Shapley 1953). SHAP values are the unique attribution satisfying three axioms:
  efficiency, dummy, and symmetry (no other attribution satisfying all three
  exists).

- Two organizing axes:
  - Global vs local:
    - Global: explains the model's overall behavior across all data (bar plot,
      beeswarm)
    - Local: explains one specific prediction (waterfall plot)
  - Model-specific vs model-agnostic:
    - Model-specific: `LinearExplainer` (linear models), `TreeExplainer` (trees)
    - Model-agnostic: `KernelExplainer` (any callable)

## Key Abstractions

- **`Explainer`**: wraps a trained model and computes Shapley values
  - `shap.LinearExplainer`: exact SHAP values for linear models
  - `shap.TreeExplainer`: fast exact SHAP for tree-based models
  - `shap.KernelExplainer`: model-agnostic approximation for any callable

- **`Explanation`**: the result object with three arrays:
  - `.values` shape `(n_samples, n_features)`: per-feature SHAP contributions
  - `.base_values` shape `(n_samples,)`: model's average prediction (baseline)
  - `.data` shape `(n_samples, n_features)`: original input feature values

- **Plots**: visualize feature contributions
  - `shap.plots.waterfall`: single-prediction breakdown (local)
  - `shap.plots.bar`: mean absolute SHAP ranking (global)
  - `shap.plots.beeswarm`: distribution of SHAP values per feature (global)
  - `shap.plots.scatter`: feature value vs SHAP contribution (feature-level)

## Tutorial Content

- This tutorial includes all the code, notebooks, and Docker container in
  [tutorials/shap](https://github.com/gpsaggese/umd_classes/tree/master/tutorials/shap):
  - `01.API.shap.ipynb`: guided exploration of the SHAP API — Explainer,
    Explanation, and plots using a linear regression model on toy data
  - `01.API.shap.py`: Jupytext percent-format mirror of the notebook
  - A Docker system to build and run the environment using the standardized
    approach

## Official References

- SHAP:
  - [docs](https://shap.readthedocs.io/)
  - [GitHub](https://github.com/shap/shap)
  - [paper](https://arxiv.org/abs/1705.07874)

## Changelog

- 2026-06-29: Initial release
