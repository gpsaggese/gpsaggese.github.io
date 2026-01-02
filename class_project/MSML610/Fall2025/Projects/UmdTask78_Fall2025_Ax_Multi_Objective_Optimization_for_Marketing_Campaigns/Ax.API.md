# Ax - Multi-Objective Optimization for Marketing Campaigns

- Student: Damian Calabresi
- Course: MSML610 - Fall 2025
- UMD ID: 121332271

# Table of Contents

- [Overview](#overview)
  - [Bayesian Optimization](#bayesian-optimization)
  - [Adaptive Experimentation](#adaptive-experimentation)
- [Ax API](#ax-api)
  - [Overview](#overview-1)
  - [API Reference](#api-reference)
- [Wrapper Layer](#wrapper-layer)
- [References](#references)

# 1. Overview

Adaptive Experimentation (Ax) is an open-source tool created and maintained by The Meta Adaptive Experimentation Team initially for internal use and open to the public.

Ax is a platform to optimize almost any type of experiment. It's well suited when the experiment complies with a few characteristics:

- The **benefit** or **outcome** can be measured or quantified.
- The **benefit** or **outcome** cannot be calculated from the **inputs**. There is no formula that can be used to calculate the **benefit** or **outcome** from the **inputs**. The only way to know the result is to run an experiment.
- The **cost** of running an experiment is elevated and the number of experiments has to be reduced to a minimum.

## 1.1. Bayesian Optimization

Ax optimization algorithms are based on Bayesian Optimization. Internal implementation is based on [BoTorch](https://botorch.org/) a library for Bayesian Optimization built on top of PyTorch.

## 1.2. Adaptive Experimentation

Adaptive Experimentation is a technique to optimize experiments based on the results of the previous experiments. It is a way to find the best configuration of the experiment parameters to maximize the **benefit** or **outcome** while minimizing the **cost** of running the experiment.

The basic adaptive experimentation flow works as follows (2):
- Configure your optimization experiment, defining the space of values to search over, objective(s), constraints, etc.
- Suggest new trials, to be evaluated one at a time or in a parallel (a “batch”)
- Evaluate the suggested trials by executing the black box function and reporting the results back to the optimization algorithm
- Repeat steps 2 and 3 until a stopping condition is met or the evaluation budget is exhausted

Bayesian optimization, one of the most effective forms of adaptive experimentation, intelligently balances tradeoffs between exploration (learning how new parameterizations perform) and exploitation (refining parameterizations previously observed to be good).

# 2. Ax API

## 2.1. Overview

The following diagram shows the main components of the Ax API and the steps to run an Adaptive Experiment:

![Ax API](images/ax-api.png)

The library is organized around the following main components:

- **Client**: The main module that will create an Experiment, return the Trials to be evaluated and store the results.
- **Experiment**: The entity that contains the configuration and all the information about the current experiment.
- **OptimizationConfig**: Defines the optimization problem, the inputs, outputs, and constraints.
- **Trial**: A new trial is generated on each iteration. It contains the inputs to be evaluated (Also called Arm)
- **GenerationStrategy**: The entity that generates the new inputs to be evaluated. It leverages different strategies on different stages (Sobol, Transfer-Learning BayesOpt, BoTorch)

![image](images/ax-api-experiment.png)

For more information, see the [Ax Glossary](https://ax.dev/docs/glossary)

## 2.2. API Reference

Only the most important modules and methods are documented here. For more detailed information, see the [Ax API Reference](https://ax.readthedocs.io/en/stable/)

- `Client`
  - `configure_experiment`: Receives an ExperimentConfig. Creates the Experiment Object. No value is returned, the Client is stateful.
  - `configure_optimization`: Receives the `objective` expression and outcome constraints. Outcome constraints are optional and define when the optimization should stop.
  - `get_next_trials`: Receives the number of trials to generate in parallel. Returns a list of Trials. Each Trial will contain a suggested input to be evaluated.
  - `complete_trial`: Receives the result of the evaluation of the Trial. Ax will update the Experiment with the result. Metadata can be attached for future reference.
  - `get_best_parameterization`: Returns the Trial or input values that maximized the objective function.
  - `compute_analyses`: Renders multiple visualizations of the experiment results. The visualization methods to render can be passed as a parameter.
  - `get_pareto_frontier`: If the Experiment has been defined with two or more objectives, this method returns the list of tuples which are part of the Pareto frontier.
- `IRunner`: Abstract class that can be implemented to define a experiment that's delegated to an external service to be completed asynchronously. Defines the methods `run_trial` and `poll_trial`.
- `IMetric`: Abstract class that works together with the `IRunner` object to retrieve the results from an external service.

# 3. Wrapper Layer

The Ax API is a high-level API that abstracts the users from the complexity of BoTorch. It provides a simple interface to configure and run an experiment. For this reason, no wrapper layer is implemented on top of it.

# 4. References

1. [Ax - Why Ax?](https://ax.dev/docs/why-ax)
2. [Ax - Adaptive Experimentation](https://ax.dev/docs/intro-to-ae)
3. [BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization](https://arxiv.org/abs/1910.06403)