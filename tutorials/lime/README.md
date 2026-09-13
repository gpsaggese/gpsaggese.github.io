# Model Explainability Tutorial

Learn how to explain machine learning models in 60 minutes using interpretable
models, SHAP, LIME, permutation importance, and counterfactuals.

## Quick Start

- From the root of the repository:
  ```bash
  > cd tutorials/ml_explainability
  > ./docker_build.sh
  > ./docker_jupyter.sh
  ```

- Open your browser to http://localhost:8888 and work through the notebooks in
  order:
  1. `explainability_shap_lime.API.ipynb`: core API of each explainability tool
  2. `explainability_shap_lime.example.ipynb`: end-to-end explanation of a model
     trained on a real dataset

- For Docker build system details see the
  [project template README](../../class_project/project_template/docker_scripts.README.md)

## Introduction

- Model explainability answers a simple question: _why did the model make this
  prediction?_

- Modern models (gradient boosting, random forests, neural nets) are accurate
  but opaque. You get a number out, with no reason attached. Explainability
  techniques recover that reason, either for the model as a whole (global) or for
  a single prediction (local).

- Use explainability when:
  - You must justify a decision to a regulator, customer, or domain expert (e.g.,
    loan denial, medical diagnosis)
  - You want to debug a model that performs well on metrics but behaves strangely
  - You want to check that the model relies on sensible features, not on leakage
    or spurious correlations
  - You need to build trust with stakeholders before deploying

- Do not reach for explainability when a simple, inherently interpretable model
  (linear regression, a small decision tree) already meets your accuracy target.
  In that case the model explains itself.

- This tutorial covers five complementary approaches:
  - Interpretable models (`sklearn.linear_model`, `pygam`)
  - Permutation importance (`sklearn.inspection`)
  - SHAP (`shap`)
  - LIME (`lime`)
  - Counterfactual explanations (`dice-ml`)

- Other tools in this space:
  - `eli5`: unified API over several explainers
  - `interpret` (Microsoft): glassbox models and dashboards
  - `captum`: gradient-based attribution for PyTorch models
  - `alibi`: counterfactuals, anchors, and trust scores

## Official References

- SHAP:
  - [docs](https://shap.readthedocs.io/),
  - [GitHub](https://github.com/shap/shap),
  - [paper](https://arxiv.org/abs/1705.07874)
- LIME:
  - [GitHub](https://github.com/marcotcr/lime),
  - [paper](https://arxiv.org/abs/1602.04938)
- pyGAM:
  - [docs](https://pygam.readthedocs.io/)
- scikit-learn inspection:
  - [permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
- DiCE:
  - [docs](https://interpret.ml/DiCE/),
  - [GitHub](https://github.com/interpretml/DiCE)

## Prerequisites

- Comfort with Python and `numpy` / `pandas`
- Basic familiarity with training a model in `scikit-learn`
- No prior knowledge of explainability methods is assumed

## Core Concepts

- Two axes organize every method in this tutorial:
  - Global vs local:
    - Global: explains the model's overall behavior across all data
    - Local: explains one specific prediction
  - Model-specific vs model-agnostic:
    - Model-specific: exploits model internals (e.g., linear coefficients, tree
      structure)
    - Model-agnostic: treats the model as a black box, probing it with perturbed
      inputs

- A third idea, feature attribution, runs through SHAP and LIME: split a
  prediction into per-feature contributions that sum to the output. SHAP grounds
  this split in game theory (Shapley values), giving it consistency guarantees
  that ad-hoc methods lack.
## Choosing a Technique

| Technique | Scope | Model | Best for |
| :-------- | :---- | :---- | :------- |
| Linear / GAM | Global | Specific | Inherently interpretable models |
| Permutation importance | Global | Agnostic | Fast feature ranking |
| SHAP | Global and local | Agnostic (fast for trees) | Trustworthy attributions |
| LIME | Local | Agnostic | Quick single-instance, text and images |
| Counterfactuals | Local | Agnostic | Actionable recourse |

- A practical workflow:
  - Start with permutation importance to rank features globally
  - Use SHAP for reliable local and global attributions
  - Reach for LIME on text or images, or for a fast second opinion
  - Add counterfactuals when users need to know what to change

## Tutorial Content

- This tutorial includes all the code, notebooks, and Docker container in
  [tutorials/ml_explainability](https://github.com/gpsaggese/umd_classes/tree/master/tutorials/ml_explainability):
  - `explainability_shap_lime.API.ipynb`: core API of each explainability tool
  - `explainability_shap_lime.example.ipynb`: end-to-end explanation of a model
    on a real dataset
  - `explainability_shap_lime_utils.py`: shared helper functions
  - A Docker system to build and run the environment using the standardized
    approach

## Changelog

- 2026-06-29: Initial release
