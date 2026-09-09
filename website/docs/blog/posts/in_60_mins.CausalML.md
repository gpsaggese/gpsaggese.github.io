---
title: "CausalML in 60 Minutes"
draft: false
authors:
    - MohammedSyed
    - gpsaggese
date: 2026-04-10
description:
categories:
    - AI Research
    - Data Science
---

TL;DR: Learn how to estimate heterogeneous treatment effects using CausalML in
60 minutes with hands-on examples including individual-level causal inference on
CDC diabetes health data.

<!-- more -->

## Tutorial in 30 Seconds

**CausalML** is an open-source Python library from Uber for causal machine
learning, providing a suite of meta-learner algorithms to estimate
individualized treatment effects from observational data.

Key capabilities:

- **Meta-learners**: S, T, X, R, and DR-Learner algorithms for heterogeneous
  treatment effect estimation
- **Uplift modeling**: Identify who benefits most from a treatment or
  intervention
- **Robustness checks**: Placebo tests, sensitivity analysis, and estimator
  comparisons built in
- **Scikit-learn compatible**: Integrates with any sklearn-compatible base
  learner

This tutorial's goal is to show you in 60 minutes:

- The basic API of CausalML (an open-source library for causal machine
  learning)
- Concrete examples of using CausalML to estimate who benefits most from
  physical activity using CDC BRFSS diabetes health data

## Official References

- [CausalML: A Python Package for Uplift Modeling and Causal ML](https://causalml.readthedocs.io/)
- [GitHub repo](https://github.com/uber/causalml)
- [Original Paper](https://arxiv.org/abs/2002.11631)

## Tutorial Content

This tutorial includes all the code, notebooks, and Docker containers in
[tutorials/CausalML_Diabetes_Study](https://github.com/gpsaggese/gpsaggese.github.io/tree/master/tutorials/CausalML_Diabetes_Study)

- [`README.md`](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/tutorials/CausalML_Diabetes_Study/README.md): Instructions and setup for the tutorial environment
- A Docker system to build and run the environment using our standardized
  approach
- [`CausalML.API.ipynb`](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/tutorials/CausalML_Diabetes_Study/CausalML.API.ipynb): Tutorial notebook focusing on the CausalNavigator API and meta-learner configurations
- [`CausalML.example.ipynb`](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/tutorials/CausalML_Diabetes_Study/CausalML.example.ipynb): Advanced end-to-end causal inference example
    - Loads and preprocesses the CDC BRFSS diabetes dataset (250,000+ respondents)
    - Checks causal assumptions (overlap/positivity) using propensity score analysis
    - Estimates individualized treatment effects (CATE) using the X-Learner
    - Visualizes heterogeneity across age, income, and health status subgroups
    - Validates results with placebo tests, estimator comparisons, and sensitivity
      analysis
- [`causalml_utils.py`](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/tutorials/CausalML_Diabetes_Study/causalml_utils.py): Utility functions and the `CausalNavigator` wrapper class
