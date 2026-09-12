# Summary

## Title
- From Data to Decisions: Building Decision Systems with Probabilistic Causal
  Reasoning

- Reasoning under Uncertainty: Causal Machine Learning for Decision Making

## Target Audience
- Senior ML engineers and data scientists with a statistics and probabilistic ML
  background who build production decision systems
- Working knowledge of causal basics (DAGs, SCMs, do-calculus) assumed

## Short TOC
- The sequence of the parts in the books are:
  - Motivation
    - 01, Introduction
    - 02, Why Decisions, Not Predictions
    - 03, The Cost of Ignoring Causality, Uncertainty and Dynamics
  - Advanced Modeling Theory & Tools
    - 04, Knowledge Representation
    - 05, Probabilistic ML
    - 06, Causal ML
  - Data
    - 07, Building Causal Knowledge
    - 08, Causal data pipelines
  - Decision-Making Theory & Tools
    - 09, Decision Theory Foundations
    - 10, Taxonomy of Decision-Making Problems and Algorithms
    - 11, Simple Decisions
    - 12, Complex Decisions
    - 13, Agentic Causal Reasoning
  - Implementation, Deployment, & Governance
    - 14, Building Stakeholder Alignment
    - 15, Deployment, Monitoring, and Adaptation
    - 16, Trust, Explainability, Fairness, and Governance

## All Lesson Materials
// From ./generate_all_tocs.sh

- `data605/all_tocs.md`
- `data605/lectures_source/*.smd`

- `msml610/all_tocs.md`
- `msml610/lectures_source/*.smd`

- `book.Agentic_AI/all_tocs.md`
- `book.Agentic_AI/lectures_source/*.smd`

- `book_springer/all_tocs.md`
- `book_springer/lectures_source/*.smd`

## Chapter Templates and Invariants

- Follow `.claude/skills/book.rules.md` for the Chapter Template (Goals, Topics,
  TODO, Slides, Lesson Materials, Notes) and Roadmap section conventions used
  throughout this file

# Roadmap

// https://docs.google.com/spreadsheets/d/1dU3crReWWLcSG8jI4jTvA4430-yMkqvdOEXEIbmktPQ/edit?gid=831837256#gid=831837256

| Chap                                        |  Slides       | Criticize | Tutorial | Book |
| --------------------------------------------|  ------------ | --------- | ---------| -----|
|                                             |               |           |          |      |
| **Motivation**                              |               |           |          |      |
| 01. Why Decisions, Not Predictions          | 50%          |           |          |      |
| 02. The Cost of Ignoring Causality          |              |           |          |      |
| **Advanced Modeling Theory & Tools**        |               |           |          |      |
| 04. Knowledge Representation                |               |           |          |      |
| 05. Probalistic ML                          |               |           |          |      |
| 06. Causal ML                               |               |           |          |      |
| **Data**                                    |               |           |          |      |
| 07. Building Causal Knowledge               |               |           |          |      |
| 08. Causal data pipelines                   |               |           |          |      |
| **Decision-Making Theory & Tools**          |               |           |          |      |
| 09. Decision Theory Foundations             |               |           |          |      |
| 10. Taxonomy of Decision-Making Problems    |               |           |          |      |
| 11. Simple Decisions                        |               |           |          |      |
| 12. Complex Decisions                       |               |           |          |      |
| 13. Agentic Causal Reasoning                |               |           |          |      |
| **Implementation, Deployment, & Governance**|               |           |          |      |
| 14. Building Stakeholder Alignment          |               |           |          |      |
| 15. Deployment, Monitoring, and Adaptation  |               |           |          |      |
| 16. Trust, Explainability, Fairness, and Governance |       |           |          |      |

| Slides                                                                       | Typst | Slides | Criticize | Tutorial |
| ---------------------------------------------------------------------------- | ----- | -------| --------- | ---------|
|                                                                              |       |        |           |          |
| `msml610/lectures_source/Lesson00-Class.smd`                                 | yes   |        |           |          |
| `msml610/lectures_source/Lesson01.1-AI_and_Machine_Learning.smd`             |       |        |           |          |
| `msml610/lectures_source/Lesson01.2-The_Foundations_of_AI.smd`               |       |        |           |          |
| `msml610/lectures_source/Lesson01.3-Brief_History_of_AI.smd`                 |       |        |           |          |
| `msml610/lectures_source/Lesson02.1-A_Map_of_Machine_Learning.smd`           |       |        |           |          |
| `msml610/lectures_source/Lesson02.2-ML_Paradigms.smd`                        |       |        |           |          |
| `msml610/lectures_source/Lesson02.3-ML_Techniques_Input_Processing.smd`      |       |        |           |          |
| `msml610/lectures_source/Lesson02.4-ML_Techniques_Model_Learning.smd`        |       |        |           |          |
| `msml610/lectures_source/Lesson02.5-ML_Techniques_Model_Evaluation.smd`      |       |        |           |          |
| `msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd`    |       |        |           |          |
| `msml610/lectures_source/Lesson03.1-Knowledge_representation.smd`            | yes   | 80%    | 50%       |          |
| `msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd` | yes   | 80%    | -         |          |
| `msml610/lectures_source/Lesson03.3-Non_classical_logics.smd`                | yes   | 80%    | -         |          |
| `msml610/lectures_source/Lesson04.1-Models.smd`                              |       |        |           |          |
| `msml610/lectures_source/Lesson04.2-Models.smd`                              |       |        |           |          |
| `msml610/lectures_source/Lesson04.3-Models.smd`                              |       |        |           |          |
| `msml610/lectures_source/Lesson05.1-Learning_Theory.smd`                     |       |        |           | 70%      |
| `msml610/lectures_source/Lesson05.2-Overfitting.smd`                         |       |        |           | 70%      |
| `msml610/lectures_source/Lesson05.3-Learn_Validation.smd`                    |       |        |           | -        |
| `msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd`                   | yes   | 80%    |           | 70%      |
| `msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd`             | yes   | 80%    |           | 70%      |
| `msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd`  | yes   | 80%    |           | 70%      |
| `msml610/lectures_source/Lesson07.2-Posterior_Based_Decisions.smd`           |       |        |           | 70%      |
| `msml610/lectures_source/Lesson07.3-Hierarchical_Models.smd`                 |       |        |           | 70%      |
| `msml610/lectures_source/Lesson07.4-Generalized_Linear_Models.smd`           |       |        |           | 70%      |
| `msml610/lectures_source/Lesson07.5-Bayesian_Model_Comparison.smd`           |       |        |           | 70%      |
| `msml610/lectures_source/Lesson08.1-Causal_AI_intro.smd`                     | yes   | 80%    |           |          |
| `msml610/lectures_source/Lesson08.2-Causal_AI_concepts.smd`                  | yes   | 80%    |           |          |
| `msml610/lectures_source/Lesson08.3-Causal_AI_in_business.smd`               | yes   | 80%    |           |          |
| `msml610/lectures_source/Lesson08.4-Causal_networks.smd`                     | yes   | 80%    |           | 60%      |
| `msml610/lectures_source/Lesson08.5-Do_calculus.smd`                         | yes   | 70%    |           | 60%      |
| `msml610/lectures_source/Lesson08.6-Causal_inference_intro.smd`              | yes   | 70%    |           | 60%      |
| `msml610/lectures_source/Lesson08.7-Causal_experiments.smd`                  | yes   | 70%    |           | 60%      |
| `msml610/lectures_source/Lesson08.8.Causal_Linear_Regression.smd`            | yes   | 70%    |           | 60%      |
| `msml610/lectures_source/Lesson08.9-Effect_heterogeneity_and_Metalearners.smd` | yes   | 70%  |           | 60%      |
| `msml610/lectures_source/Lesson08.X-Causal_inference.smd`                    
| `msml610/lectures_source/Lesson09.1-Reasoning_over_time.smd`                 
| `msml610/lectures_source/Lesson09.2-Hidden_Markov_Models.smd`                
| `msml610/lectures_source/Lesson09.3-Multi_Armed_Bandits.smd`                 | yes   | 10%    |           | 10%      |
| `msml610/lectures_source/Lesson09.4-gh_Filter.smd`                           |       | 70%    |           | 60%      |
| `msml610/lectures_source/Lesson09.5-Kalman_Filter.smd`                       |       | 70%    |           | 60%      |
| `msml610/lectures_source/Lesson09.6-Dynamic_Bayesian_Networks.smd`           
| `msml610/lectures_source/Lesson09.7-Advanced_Bandits.smd`                    | yes   | 50%    |           | 0%       |
| `msml610/lectures_source/Lesson10.1-Timeseries_forecasting.smd`              | ?
| `msml610/lectures_source/Lesson10.2-Causal_Inference_for_Time_Series.smd`    | ?
| `msml610/lectures_source/Lesson11.1-Decision_Making_with_Causal_Models.smd`  | ?
| `msml610/lectures_source/Lesson11.2-Probabilistic_deep_learning.smd`         | Move 
| `msml610/lectures_source/Lesson12.1-Reinforcement_learning.smd`              | ?     | 70%    |           | 60%      |
| `msml610/lectures_source/Lesson12.2-Causal_Discovery.smd`                    | ?     | 70%    |           | 60%      |
| `msml610/lectures_source/Lesson13.1-Explainability.smd`                      | ?     | 70%    |           | where?   |

- [ ] Apply `msml610/lectures_source/prompt.convert_to_typst.md` to all the files
  in `msml610/lectures_source/Lesson*`
- [ ] `/slides.lint_incrementally`
- [ ] `/slides.fix_formatting`

## `book_springer` Tutorials

> find book_springer/tutorials -name *.ipynb
```
book_springer/tutorials/Lesson10_01_q_learning/q_learning.ipynb
```

## MSML610 Tutorials

> find msml610/tutorials -name *.ipynb | sort
```
msml610/tutorials/L05_statistical_learning/L05_01_01_hoeffding_inequality.ipynb
msml610/tutorials/L05_statistical_learning/L05_01_02_bin_analogy_ml.ipynb
msml610/tutorials/L05_statistical_learning/L05_01_03_vc_dimension.ipynb
msml610/tutorials/L05_statistical_learning/L05_01_04_growth_function.ipynb
msml610/tutorials/L05_statistical_learning/L05_02_01_bias_variance.ipynb
msml610/tutorials/L05_statistical_learning/L05_02_02_overfitting.ipynb
msml610/tutorials/L06_bayesian_networks/L06_01_exact_inference.ipynb
msml610/tutorials/L06_bayesian_networks/L06_02_approximate_inference.ipynb
msml610/tutorials/L07_prob_programming/L07_01_bayesian_coin.ipynb
msml610/tutorials/L07_prob_programming/L07_02_probabilistic_programming.ipynb
msml610/tutorials/L07_prob_programming/L07_02_robust_modeling.ipynb
msml610/tutorials/L07_prob_programming/L07_03_hierarchical_models.ipynb
msml610/tutorials/L07_prob_programming/L07_04_generalized_linear_models.ipynb
msml610/tutorials/L07_prob_programming/L07_05_evaluating_models.ipynb
msml610/tutorials/L08_causal_inference/L08_04_01_causal_inference.ipynb
msml610/tutorials/L08_causal_inference/L08_04_02_causal_inference.ipynb
msml610/tutorials/L08_causal_inference/L08_04_05_propensity_score.ipynb
msml610/tutorials/L08_causal_inference/L08_04_07_metalearners.ipynb
msml610/tutorials/L08_causal_inference/L08_04_08_difference_in_difference.ipynb
msml610/tutorials/L09_kalman_filter/L09_04_gh_filter.ipynb
msml610/tutorials/L09_kalman_filter/L09_05_01_discrete_bayes_dog.ipynb
msml610/tutorials/L09_kalman_filter/L09_05_02_univariate_kalman_filter.ipynb
msml610/tutorials/L09_kalman_filter/L09_05_03_multivariate_kalman_filter.ipynb
msml610/tutorials/L09_kalman_filter/L09_05_04_non_linear_kalman_filter.ipynb
msml610/tutorials/L09_multi_armed_bandits/L09_03_02_multi_armed_bandits.ipynb
msml610/tutorials/L10_causal_discovery/L10_2_causal_discovery.ipynb
msml610/tutorials/L12_reinforcement_learning/L12_01_gridworld_4x3.ipynb
msml610/tutorials/L12_reinforcement_learning/L12_02_gridworld_4x3_gymnasium.ipynb
```

## TODOs
- Incorporate ./notes/math.Probabilistic_programming_for_hackers.DavidsonPilon.2017.smd

# Detailed TOC

# Part I: Why Businesses Need Decisions, Not Predictions

## 01: Introduction

### Goals
- Frame the core premise: ML systems must optimize for decision value, not
  prediction accuracy
- Introduce the Decision Pipeline Framework (data → prediction → causal effect →
  policy)
- Explain why causal and probabilistic reasoning for business decision-making

### Topics
- The Philosophy and Motivation
  - Why prediction accuracy alone fails to drive business value
  - Decision-centric framing: from data science to decision science
  - Role of causal reasoning and probabilistic models in decision systems
- The Decision Pipeline Framework
  - From raw data to prediction to causal effect estimation to utility-maximizing
    policy
  - Feedback loops and learning from outcomes
  - Why each stage matters and what breaks when stages are skipped
- How This Book Is Organized
  - Five-part structure: motivation, theory, data, algorithms,
    deployment/governance
  - Intended audience and prerequisites
  - How chapters build on each other and fit into the decision pipeline

- Focus on intuition over math (unless necessary)
- Emphasize realistic assumptions and numerical methods
  - Analytical solutions are so 1800s
- Interactive Jupyter notebook tutorials for hands-on approach

### Slides
- `book_springer/lectures_source/Lesson01.1_Introduction.smd`

### Lesson Materials
- `book_springer/lectures_source/Lesson01.1_Introduction.smd`
  - [70%]: Decision Pipeline Framework (data→prediction→causal effect→policy,
    feedback loops, why stages matter), book organization (five-part
    structure, audience/prerequisites, chapter progression), intuition-first
    focus, Jupyter tutorials
- `msml610/lectures_source/Lesson00-Class.smd`
  - [25%]: "Invariants of a Class Lecture" matches intuition-over-math focus,
    realistic-assumptions/numerical-methods emphasis, Jupyter tutorials
- Not covered
  - [10%]: Role of probabilistic models specifically in decision systems
    (only causal reasoning is developed); the book's own five-part
    structure, audience, and prerequisites beyond the book_springer decks
    themselves

## 02: Why Decisions, Not Predictions

### Goals
- Show the gap: High prediction accuracy ≠ good business value (ROI)
- Distinguish prediction from decision
- Expose four critical failure modes: ignoring causality, uncertainty
  quantification, business objectives, and dynamics

### Topics
- Why Decisions, Not Predictions
- The Cost of Ignoring Causality
  - When Correlation Misleads
  - Structural Failure Modes
- The Cost of Ignoring Uncertainty
- The Cost of Ignoring the Business Objective
  - Proxies, Costs, and Trade-offs
  - From Scores to Actionable Decisions
- The Cost of Ignoring Dynamics and Feedback
  - A World That Reacts
  - Missing Exploration and Long Horizons
- Why This Matters
  - Roadmap

### Slides
- `book_springer/lectures_source/Lesson02.1_From_Data_Science_To_Decision_Science.smd`

### Lesson Materials
- `book_springer/lectures_source/Lesson02.1_From_Data_Science_To_Decision_Science.smd`
  - [97%]: Chapter's own slide deck — near 1:1 with the topic list: why
    decisions not predictions, cost of ignoring causality (confounding,
    colliders, Simpson's paradox), cost of ignoring uncertainty (point
    estimates, aleatoric/epistemic, overfitting), cost of ignoring the
    business objective (Goodhart's law, asymmetric losses, black-box
    scores), cost of ignoring dynamics/feedback (shift, performativity,
    exploration, long-horizon effects), roadmap
- `msml610/lectures_source/Lesson11.1-Decision_Making_with_Causal_Models.smd`
  - [40%]: Prediction vs. decision pipelines, Simpson's paradox/confounding,
    aleatoric vs. epistemic uncertainty, exploration via VOI/EVPI/EVSI and
    causal bandits; missing structural failure modes, feedback/performativity
- `msml610/lectures_source/Lesson08.1-Causal_AI_intro.smd`
  - [35%]: Prediction vs. decision, correlation vs. causation ladder,
    Simpson's paradox/confounding, concept/covariate/label shift,
    feedback-loop examples in recommenders and criminal justice
- `msml610/lectures_source/Lesson08.2-Causal_AI_concepts.smd`
  - [20%]: Ladder of causation, data-science-vs-decision-science comparison,
    causal-vs-predictive questions, descriptive/predictive/causal/decision
    roadmap
- `msml610/lectures_source/Lesson08.4-Causal_networks.smd`
  - [15%]: Confounder/collider examples overlapping "When Correlation
    Misleads" and part of "Structural Failure Modes"; no uncertainty,
    objective, or dynamics content
- Not covered
  - [3%]: Book-structural, self-referential content only present in the
    book's own deck — "Why AI Projects Fail" framing and the explicit Part
    II–V roadmap synthesis

## 03: Handling Causality, Uncertainty, Business Objectives, and Dynamics

### Goals
- Combine causality and probability to find true effects
- Quantify uncertainty from evidence to decision bounds
- Encode business goals as utility-based decision functions

### Topics
- Causal Models and Effect Identification
  - Moving beyond correlation: confounding, mediators, and spurious paths in
    observational data
  - Structural causal models (SCMs) and DAGs: formalizing mechanisms
  - Identification criteria: backdoor, frontdoor, do-calculus
- Probabilistic Uncertainty Quantification
  - Bayesian inference: priors, posteriors, posterior predictive distributions
  - Epistemic uncertainty over model; aleatoric irreducible noise
  - Decision-relevant bounds: confidence intervals, credible intervals,
    thresholds
- Business Objectives and Decision Rules
  - Utility functions: preferences, risk attitudes, multi-criteria trade-offs
  - Cost asymmetries: false-positive and false-negative costs
  - From predictions to actions: translating scores into decisions
- Dynamics, Feedback, and Performativity
  - Temporal causality: feedback loops linking past decisions to future causes
  - Performativity: how decisions change the systems they operate on
  - Sequential reasoning: multi-period consequences and adaptation
// TODO(ai_gp): Check this

### Slides 
- `book_springer/lectures_source/Lesson02.2_Integrating_Causality_And_Probability_in_ML.smd`
- `book_springer/lectures_source/Lesson02.3_Integrating_Business_Objective_And_Real_World_Dynamics.smd`

### Lesson Materials
- `book_springer/lectures_source/Lesson02.2_Integrating_Causality_And_Probability_in_ML.smd`
  - [55%]: Causal models and effect identification in full (confounders,
    mediators, colliders, SCMs/DAGs, do-operator, backdoor/front-door, IV,
    potential outcomes) plus probabilistic uncertainty quantification
    (posteriors vs. point estimates, epistemic vs. aleatoric, variational
    inference, calibration/conformal prediction)
- `book_springer/lectures_source/Lesson02.3_Integrating_Business_Objective_And_Real_World_Dynamics.smd`
  - [50%]: Business objectives and decision rules in full (utility
    functions, expected utility, risk preferences, cost asymmetries,
    multi-objective trade-offs, decision networks) plus most of
    dynamics/feedback (feedback loops, Granger causality,
    exploration-exploitation, Bellman equation); performativity present
    only implicitly, not named
- `msml610/lectures_source/Lesson11.1-Decision_Making_with_Causal_Models.smd`
  - [40%]: Deeper treatment of utility functions, expected utility, risk
    preferences, multi-criteria trade-offs, decision networks, prior
    elicitation, and aleatoric-vs-epistemic uncertainty with credible
    intervals
- `msml610/lectures_source/Lesson08.4-Causal_networks.smd`
  - [25%]: Causal DAGs, structural causal models, confounder/mediator/
    moderator variable types with worked examples
- `msml610/lectures_source/Lesson08.5-Do_calculus.smd`
  - [20%]: Do-operator, backdoor/front-door adjustment, the three formal
    rules of do-calculus
- `msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd`
  - [15%]: Bayes' theorem, priors and conjugate priors, confidence vs.
    credible intervals
- Not covered
  - [8%]: No file gives "performativity" an explicit, formal named
    treatment (only appears in chapter 02's deck); posterior predictive
    distributions and explicit decision thresholds receive only glancing
    coverage

# Part II: Advanced Modeling Theory & Tools

## 04: Knowledge Representation

### Goals
- Understand representation as a foundation for reasoning
- Master formal knowledge representation schemes
- Integrate symbolic and probabilistic reasoning

### Topics
- Representation as a Foundation for Decisions
  - Why representation choice shapes what questions a model can answer
  - Symbolic, sub-symbolic, procedural, declarative, and natural-language
    representations: trade-offs in expressiveness and tractability
- Formal Knowledge Representation
  - Propositional and first-order logic, entailment, and inference: symbolic
    reasoning foundations
  - Non-monotonic reasoning, default logic, and open- vs. closed-world
    assumptions
  - Rule-based and knowledge-based agents; grounding symbols in the world
  - Ontologies, description logics, RDF/SPARQL, semantic networks, and
    knowledge graphs: organizing domain knowledge into reusable structures
- Graphical Models for Uncertainty
  - Why logic fails under uncertainty
  - Bayesian networks: definition and structure
- Causal Graphical Models
  - Causal networks (causal DAGs) and the ladder of causation: association,
    intervention, and counterfactual
  - Structural causal models (SCMs): definition
- Integrating Logic and Causality
  - Markov logic networks: combining symbolic reasoning with probabilistic
    inference
  - From correlation to causal reasoning: what graphs add beyond raw data
  - Choosing a representation: when logic, probability, or causal graphs fit
    best

### TODO
- Add `## Structural Causal Model`

### Slides
- `book_springer/lectures_source/Lesson04.1_Knowledge_Representation.smd`

### Lesson Materials
- `msml610/lectures_source/Lesson03.1-Knowledge_representation.smd`
  - [45%]: Representation foundations, expressiveness/tractability
    trade-offs, propositional/FOL basics, entailment/inference, rule-based
    and knowledge-based agents, grounding, ontologies
- `msml610/lectures_source/Lesson03.3-Non_classical_logics.smd`
  - [35%]: Non-monotonic reasoning, default logic, open- vs. closed-world
    assumption, description logics (OWL), RDF/SPARQL, semantic
    networks/knowledge graphs
- `msml610/lectures_source/Lesson08.4-Causal_networks.smd`
  - [20%]: Causal (Bayesian) networks vs. non-causal, causal DAG properties,
    full ladder of causation, structural causal model definition
- `msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd`
  - [20%]: Why logic fails under uncertainty, Bayesian network definition
    and structure (DAG, CPTs, conditional independence)
- `msml610/lectures_source/Lesson03.2-Propositional_and_first_order_logic.smd`
  - [20%]: In-depth propositional and first-order logic (syntax, semantics,
    inference rules, satisfiability)
- `msml610/lectures_source/Lesson09.2-Hidden_Markov_Models.smd`
  - [8%]: Markov logic networks (first-order logic + Markov random fields,
    weighted formulas, joint distribution)
- Not covered
  - [5%]: A genuine decision framework for "when logic, probability, or
    causal graphs fit best" is only thinly present; what causal graphs add
    beyond correlational data is treated only implicitly

## 05: Probabilistic ML

### Goals
- Master Bayesian updating: priors, posteriors, and approximate inference methods
- Build Bayesian generative models: networks, regression, and hierarchical models
- Turn posterior uncertainty into decisions, comparisons, and model choices

### Topics
- Bayesian Inference Foundations
  - Bayesian updating: turning priors into posteriors as evidence arrives,
    versus the frequentist view
  - Approximate inference: sampling, variational methods, and MCMC for
    posteriors with no closed form
- Bayesian Generative Models
  - Bayesian networks: structure, semantics, and exact vs. approximate
    inference
  - Regression and hierarchical models: posterior-based uncertainty over
    parameters, pooled across groups
- Uncertainty in Predictions
  - Posterior predictive distributions and checks: averaging over model
    uncertainty, validating fit
  - Epistemic vs. aleatoric uncertainty; robust inference and effect-size
    comparison across groups
- Bayesian Decision-Making
  - Expected utility and loss functions: turning posterior uncertainty into
    a decision
  - Savage-Dickey ratios, credible intervals, and prior choice for
    communicating results
- Model Comparison and Selection
  - Balancing fit and complexity: Occam's razor, overfitting, and the
    bias-variance trade-off
  - Information criteria, cross-validation, ensembles, and Bayes factors for
    choosing among models

### Slides
- `book_springer/lectures_source/Lesson05.1_Probabilistic_ML.smd`

### Lesson Materials
- `msml610/lectures_source/Lesson07.2-Posterior_Based_Decisions.smd`
  - [45%]: Savage-Dickey ratio, Region of Practical Equivalence, loss
    functions/expected utility, posterior predictive checks, robust
    inference, effect-size comparison
- `book_springer/lectures_source/Lesson05.1_Probabilistic_ML.smd`
  - [40%]: Chapter's own outline touching nearly every sub-topic, but each
    bullet is a single line pointing to msml610 sources rather than a
    standalone treatment
- `msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.smd`
  - [35%]: Bayesian network semantics, exact inference (enumeration,
    variable elimination), Markov blankets, approximate inference
    (rejection/importance sampling, MCMC)
- `msml610/lectures_source/Lesson07.5-Bayesian_Model_Comparison.smd`
  - [30%]: Occam's razor, overfitting/bias-variance trade-off, information
    criteria, cross-validation, Bayesian model selection/averaging, Bayes
    factors
- `msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd`
  - [30%]: Bayesian updating, frequentist vs. Bayesian views, prior choice,
    credible vs. confidence intervals, probabilistic programming (PyMC)
- `msml610/lectures_source/Lesson06.1-Bayesian_Networks.smd`
  - [15%]: Bayesian network definitions/structure, conditional independence,
    CPTs, belief updating
- `msml610/lectures_source/Lesson07.3-Hierarchical_Models.smd`
  - [15%]: Hierarchical/multilevel models, pooled vs. unpooled, hyper-priors,
    shrinkage across groups
- `msml610/lectures_source/Lesson07.4-Generalized_Linear_Models.smd`
  - [12%]: Bayesian linear/logistic regression, generalized linear models,
    posterior-based uncertainty over parameters
- `msml610/lectures_source/Lesson11.2-Probabilistic_deep_learning.smd`
  - [10%]: Explicit aleatoric-vs-epistemic uncertainty, variational
    inference, MCMC/HMC for Bayesian neural nets (deep-learning framing)
- `msml610/lectures_source/Lesson05.2-Overfitting.smd`
  - [8%]: Classical bias-variance decomposition and overfitting
    (non-Bayesian framing)
- Not covered
  - [12%]: No lecture gives a rigorous general-Bayesian (non-deep-learning)
    treatment of variational inference (mean-field, ELBO/KL derivation), or
    develops epistemic-vs-aleatoric uncertainty beyond a label

## 06: Causal ML

### Goals
- Represent causal assumptions with DAGs and structural causal models
- Use do-calculus to identify interventional effects from observational data
- Apply backdoor, frontdoor, and IV methods; know when identification fails

### Topics
- Causal Graphical Models
  - Structural causal models and DAGs: formal specification of generative
    mechanisms and causal order
  - Confounders, mediators, and colliders; the ladder of causation from
    association to counterfactuals
- Do-Calculus and Interventional Reasoning
  - The do-operator: distinguishing intervention from passive observation, and
    the three rules of do-calculus
  - From interventions to potential outcomes: individual, average, and
    conditional treatment effects
- Identification Criteria
  - Backdoor and frontdoor criteria for identifying effects under
    confounding and mediation
  - D-separation and graph structure; confounding bias and propensity-score
    adjustment in practice
- Advanced Identification Methods
  - Instrumental variables and natural experiments for breaking confounding
    via exogenous variation
  - Sensitivity analysis for unmeasured confounding, and the limits of
    identifiability from observational data

### Slides
- `book_springer/lectures_source/Lesson06.1_Causal_ML.smd`

### Lesson Materials
- `book_springer/lectures_source/Lesson02.2_Integrating_Causality_And_Probability_in_ML.smd`
  - [55%]: Ladder of causation, SCMs, confounders/mediators/colliders, DAG
    building, do-operator, backdoor/front-door with worked examples, instrumental
    variables, potential outcomes, treatment effects
- `msml610/lectures_source/Lesson08.6-Causal_inference_intro.smd`
  - [45%]: Potential outcomes, treatment effects, confounding-bias equation,
    Simpson's paradox, formal d-separation, backdoor adjustment, SCM definition,
    chain/fork/collider structures
- `msml610/lectures_source/Lesson08.5-Do_calculus.smd`
  - [40%]: Do-operator, RCTs, backdoor criteria/adjustment, front-door adjustment
    worked example, formal three rules of do-calculus
- `msml610/lectures_source/Lesson08.4-Causal_networks.smd`
  - [30%]: SCM (sprinkler example), causal DAGs, observed/unobserved and
    endogenous/exogenous variables, mediator/moderator/confounder/collider,
    ladder-of-causation worked example
- `msml610/lectures_source/Lesson08.8.Causal_Linear_Regression.smd`
  - [25%]: Confounding via regression, FWL theorem, adjustment via controls,
    propensity-score estimation/matching/IPW — best match for
    "propensity-score adjustment in practice"
- `msml610/lectures_source/Lesson10.2-Causal_Inference_for_Time_Series.smd`
  - [20%]: Instrumental variables (exogenous-variation intuition, complier
    limitation), natural experiments (draft-lottery example), time-varying
    unobserved confounders
- `msml610/lectures_source/Lesson12.2-Causal_Discovery.smd`
  - [15%]: Identifiability (Markov equivalence/CPDAG), faithfulness, brief
    sensitivity-analysis bullet in DAG validation
- `book_springer/lectures_source/Lesson06.1_Causal_ML.smd`
  - [10%]: Chapter's own outline — headers map 1:1 onto all four topics but
    content is source pointers only, no prose or examples
- `book.Agentic_AI/lectures_source/Lesson15.1-Causal_Reasoning_Agents.smd`
  - [8%]: Brief references to confounders, positivity/no-unmeasured-
    confounding, sensitivity analysis, and causal DAGs inside agent-prompt
    templates; no formal identification methods
- Not covered
  - [15%]: No file gives a full formal treatment of sensitivity analysis for
    unmeasured confounding (e.g., Rosenbaum bounds, E-values); the three
    do-calculus rules and rigorous d-separation are never presented
    together in one place; IV/natural experiments are covered only in a
    time-series-specific framing

# Part III: Data

## 07: Building Probabilistic and Causal Knowledge

### Goals
- Elicit causal structure from experts and encode it as a validated DAG
- Distinguish confounders, mediators, moderators, and colliders in a graph
- Learn causal structure from data and reason about time-dependent causality

### Topics
- Eliciting Causal Knowledge from Experts
  - Structured, step-by-step elicitation of outcomes, interventions, and
    drivers from stakeholders
  - Handling disagreement in hybrid teams; eliciting and validating priors
- Building Causal DAGs
  - From a business question to a graph: choosing variables and encoding
    assumed mechanisms
  - Specifying edges and validating the resulting DAG against a worked example
- Variable Types and Relationships
  - Observed vs. unobserved and endogenous vs. exogenous variables
  - Confounders, moderators, colliders, and fork structures; how each role
    biases or clarifies effect estimation
- Causal Discovery from Data
  - Formal discovery problem; observational vs. interventional data
  - Constraint-based and score-based algorithm families
  - Combining automated discovery with domain knowledge, and validating the
    resulting DAG
- Temporal Structure in Causal Models
  - Why time series causality differs from the cross-sectional case
  - Feedback loops, simultaneity, and non-stationarity
  - When temporal structure helps identification, and when it misleads it

### Slides
- `book_springer/lectures_source/Lesson07.1_Building_Probabilistic_And_Causal_Knowledge.smd`

### Lesson Materials
- `msml610/lectures_source/Lesson08.4-Causal_networks.smd`
  - [55%]: Building causal DAGs (definition, directed/acyclic properties,
    worked examples) and full coverage of variable types (observed/
    unobserved, endogenous/exogenous, mediator, moderator, confounder,
    collider, fork/chain/inverted-fork)
- `msml610/lectures_source/Lesson12.2-Causal_Discovery.smd`
  - [24%]: Formal discovery definition, observational vs. interventional
    data, constraint-based/score-based/functional/interventional algorithm
    families, combining discovery with domain knowledge, DAG validation
- `msml610/lectures_source/Lesson10.2-Causal_Inference_for_Time_Series.smd`
  - [23%]: Why time-series causality differs from cross-sectional, feedback
    loops/simultaneity, non-stationarity, when temporal structure helps vs.
    misleads identification
- `msml610/lectures_source/Lesson08.3-Causal_AI_in_business.smd`
  - [20%]: Structured elicitation steps (outcomes, interventions, drivers),
    hybrid-team roles, Step 4 (build causal DAG) with a worked example
- `msml610/lectures_source/Lesson11.1-Decision_Making_with_Causal_Models.smd`
  - [7%]: Eliciting and validating priors (quantile elicitation, pairwise
    judgments, historical calibration, elicitation pitfalls)
- Not covered
  - [15%]: No file gives dedicated treatment of handling disagreement in
    hybrid teams as conflict resolution (only team-roles/process material
    exists); no end-to-end worked example of specifying edges and
    validating a DAG as a methodology

## 08: Causal Data Pipelines

### Goals
- Trace how data collection and selection choices bias causal estimates
- Detect distribution shift and proxy decay before they corrupt decisions
- Run pre-flight checks so causal assumptions hold before modeling begins

### Topics
- Data Collection and Its Biases
  - Collection design: how sampling and measurement choices shape the data
  - Selection mechanisms: who and what gets captured versus missed
  - Confounder introduction: biases built into the acquisition process itself
- Selection Bias and Incomplete Populations
  - Missing data mechanisms: MCAR, MAR, and MNAR patterns
  - Subgroup representation: historical data often misses critical populations
  - Impact on causal inference: incomplete data breaks identifiability
    assumptions
- Distribution Shift and Covariate Mismatch
  - Training vs. production: causal effects fail to hold in new environments
  - Concept drift: non-stationary relationships that change over time
  - Detecting shift: monitoring input distributions for meaningful changes
- Measurement Error and Proxy Validity
  - Proxy quality: does the proxy actually capture the true construct?
  - Attenuation: measurement error systematically weakens causal effect
    estimates
  - Correction methods: debiasing estimates under known measurement error
- Pre-Flight Data Quality Checks
  - Confounder balance: checking covariate distributions match assumptions
  - Missingness and outlier detection before modeling begins
  - Robust inference: methods that tolerate imperfect, noisy data

### Slides
- `book_springer/lectures_source/Lesson08.1_Causal_Data_Pipelines.smd`

### Lesson Materials
- `msml610/lectures_source/Lesson08.6-Causal_inference_intro.smd`
  - [35%]: Confounder introduction (confounding-bias equation), surrogate
    confounding/proxy issues, selection-bias mechanisms (collider,
    mediator, survivorship, self-selection), positivity/unconfoundedness
- `book_springer/lectures_source/Lesson15.1_Deployment_Monitoring_And_Adaptation.smd`
  - [30%]: Why assumptions break in production, taxonomy of assumptions to
    monitor, directly/indirectly testable assumptions, assumption-
    monitoring dashboard — strong match for shift detection
- `msml610/lectures_source/Lesson08.8.Causal_Linear_Regression.smd`
  - [25%]: Positivity (definition, IPW validity, positivity-bias
    trade-off), propensity score for selection bias, comparison with
    regression — matches confounder balance and selection-bias pre-flight
    checks
- `msml610/lectures_source/Lesson10.2-Causal_Inference_for_Time_Series.smd`
  - [20%]: Non-stationarity/trends, time-varying unobserved confounders
    (concept-drift-adjacent), instrumental variables/natural experiments as
    correction strategies
- `data605/lectures_source/Lesson07.2-Data_Wrangling.smd`
  - [20%]: Single-source missing-data handling, univariate/multivariate/
    time-series outlier detection
- `msml610/lectures_source/Lesson02.3-ML_Techniques_Input_Processing.smd`
  - [15%]: Handling outliers and missing data, noise removal — generic ML
    data-cleaning, not causal-framed
- `msml610/lectures_source/Lesson02.6-ML_Techniques_How_To_Do_Research.smd`
  - [15%]: Sampling bias (non-random sampling, undercoverage, survivorship,
    self-selection) and mitigation
- `data605/lectures_source/Lesson02.3-Data_Pipelines.smd`
  - [10%]: General data-engineering content (ingestion, ETL/ELT/EtLT, data
    cleaning) relevant only to generic collection design
- `msml610/lectures_source/Lesson08.1-Causal_AI_intro.smd`
  - [8%]: "Problem 6: Distribution Shift" — concept/covariate/label shift
    defined
- Not covered
  - [20%]: MCAR/MAR/MNAR missing-data taxonomy is not formalized anywhere;
    measurement-error attenuation and debiasing/correction methods are not
    covered in depth; confounder-balance diagnostics (e.g., standardized
    mean differences) and general "robust inference" are not treated as
    distinct topics

# Part IV: Decision-Making Theory & Tools

## 09: Decision Theory Foundations

### Topics
- Von Neumann-Morgenstern Axioms
  - Rational choice axioms: completeness, transitivity, continuity, and
    independence
  - Utility existence theorem: rational preferences representable as a utility
    function
  - Why expected utility is the mathematically correct decision framework
- Utility Functions in Practice
  - Eliciting preferences: methods for extracting stakeholder objectives
  - Multi-criteria trade-offs: combining value functions toward Pareto
    optimality
  - Designing utility functions while avoiding common specification pitfalls
- Subjective Expected Utility
  - Belief plus preference: combining uncertainty with stakeholder values
  - Bayesian updating: revising beliefs as new information arrives
  - Decision rules: how to choose actions under posterior uncertainty
- Influence Diagrams and Decision Networks
  - Graphical representation: nodes for decisions, uncertainties, and values
  - Backward induction: solving decision networks for optimal policies
  - Value of information: computing the worth of additional observations
- Risk Preferences and Utility Curvature
  - Risk aversion: decreasing marginal utility of wealth
  - Risk neutrality: a linear utility function over outcomes
  - Risk seeking: increasing marginal utility, rare and context-dependent

### Slides
- N/A

### Lesson Materials
- `msml610/lectures_source/Lesson11.1-Decision_Making_with_Causal_Models.smd`
  - [95%]: Utility functions, expected utility principle, risk preferences and
    visualizing risk aversion, multi-criteria trade-offs, decision networks,
    Bayesian decision rules, prior elicitation
- `msml610/lectures_source/Lesson12.1-Reinforcement_learning.smd`
  - [75%]: Maximum Expected Utility principle, utility of states/policies,
    Bellman equation: expected-utility framework applied to sequential
    decisions
- `msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd`
  - [65%]: Bayes' theorem, priors, Bayesian vs. frequentist updating —
    foundation for subjective expected utility
- `msml610/lectures_source/Lesson07.2-Posterior_Based_Decisions.smd`
  - [55%]: Posterior-based decision rules via loss functions, ROPE,
    Savage-Dickey ratio: decision rules under posterior uncertainty; not
    risk/utility content
- `msml610/lectures_source/Lesson09.3-Multi_Armed_Bandits.smd`
  - [50%]: Sequential decision-making, Bayesian bandits, value-of-information
    framing: peripheral to axiomatic utility core
- Not covered
  - [55%]: Formal VNM axioms (completeness, transitivity, continuity,
    independence) and the utility-existence theorem, stakeholder
    preference-elicitation methodology beyond priors

## 10: Taxonomy of Decision-Making Problems and Algorithms

### Topics
- From Causal Effects to Expected Value
  - Treatment effect estimates translated directly into decision value
  - Utility encoding: mapping causal effects onto a decision-relevant scale
  - Decision rule: choosing the action that maximizes expected utility
- Bayesian Decision-Making and Value of Information
  - Posterior-based choices: using the full distribution, not point estimates
  - EVPI/EVSI: value of perfect information versus sample information
  - Sequential decisions: adapting choices as new information arrives
- Exploration vs. Exploitation
  - Thompson sampling and UCB: balancing uncertainty against performance
  - Contextual bandits: making decisions conditional on observed context
  - Bayesian optimization: acquisition functions guiding efficient search
- Counterfactual Decision Analysis
  - Alternative scenarios: "What if we had chosen differently?"
  - Causal reasoning about the consequences of past decisions
  - Regret analysis: evaluating decision quality after the fact
- Robustness Under Misspecification
  - Decisions made under uncertainty about model correctness
  - Minimax and distributional robustness against unknown unknowns
  - Sensitivity analysis: how much conclusions change with assumptions

### Slides
- `book_springer/lectures_source/Lesson10.1_Taxonomy_of_Decision_Problems.smd`
- `book_springer/lectures_source/Lesson10.01.algo_info.md`
- `book_springer/lectures_source/Lesson10.01.algo_to_problem_table.md`
- `book_springer/lectures_source/Lesson10.01.problem_to_algo_table.md`
- `book_springer/lectures_source/Lesson10.01.tutorial_plan.md`

### Lesson Materials
- `msml610/lectures_source/Lesson11.1-Decision_Making_with_Causal_Models.smd`
  - [95%]: Causal effects to expected value, EVPI/EVSI, Bayesian optimization,
    causal multi-armed bandits, exploration vs. exploitation
- `msml610/lectures_source/Lesson09.3-Multi_Armed_Bandits.smd`
  - [90%]: Thompson sampling, UCB, epsilon-greedy, contextual bandits, regret
    bounds
- `msml610/lectures_source/Lesson12.1-Reinforcement_learning.smd`
  - [80%]: MDPs, value/policy iteration, off-line vs. on-line MDP solving
- `msml610/lectures_source/Lesson08.7-Causal_experiments.smd`
  - [75%]: A/B testing design, decision framework for experiment vs. observe,
    hybrid experimental-causal approaches
- `msml610/lectures_source/Lesson07.2-Posterior_Based_Decisions.smd`
  - [65%]: Posterior-based decision rules via loss functions and ROPE: using
    full posterior distribution, not point estimates
- `msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd`
  - [45%]: Bayesian updating fundamentals, priors: foundational only
- _Not covered_
  - [50%]: Minimax/distributional robustness against model misspecification,
    formal sensitivity analysis, advanced acquisition-function design

## 11: Simple Decisions

### Topics
- Bayesian Inference for Decisions
  - Reasoning over model uncertainty: Bayesian networks, variational inference,
    MCMC
  - Encoding prior knowledge: updating beliefs as evidence arrives
  - Deciding under partial information: posterior uncertainty plus decision
    rules
- Sequential Inference with Hidden State
  - Tracking unobserved state: HMMs, Kalman filters, particle filters
  - Partially observable decision problems (POMDPs): acting without full
    observability
  - Belief states: a sufficient statistic for decisions over hidden state
- Bandit Algorithms
  - Exploration-exploitation tradeoff: learning about actions vs. exploiting
    known-good ones
  - Algorithm family: epsilon-greedy, UCB, Thompson sampling, contextual bandits
  - Regret: minimizing cumulative loss while learning
- Planning and Search
  - Deterministic planning: A\* search and model predictive control (MPC)
  - Stochastic and adversarial planning: Monte Carlo tree search (MCTS), minimax
  - Finite-horizon assumption: near-optimal sequences under known, bounded
    dynamics
- Value-Based Reinforcement Learning
  - Value functions: Q-learning, SARSA, value and policy iteration
  - Deep value-based methods: DQN and function approximation for large states
  - Delayed rewards: learning environment structure from sparse feedback

### Slides
- `book_springer/lectures_source/Lesson11.1_Simple_Decisions.smd`

### Lesson Materials
- `msml610/lectures_source/Lesson12.1-Reinforcement_learning.smd`
  - [95%]: MDPs, POMDPs, belief-state transitions, value/Q-learning,
    temporal-difference learning, policy search
- `msml610/lectures_source/Lesson09.3-Multi_Armed_Bandits.smd`
  - [95%]: Epsilon-greedy, UCB, Thompson sampling, contextual bandits, regret
    analysis
- `msml610/lectures_source/Lesson09.1-Reasoning_over_time.smd`
  - [85%]: Sequential inference over hidden state: filtering, prediction,
    smoothing, Viterbi/most-likely-explanation
- `msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.smd`
  - [70%]: Bayesian inference, belief updating, posterior-based reasoning
- `msml610/lectures_source/Lesson07.2-Posterior_Based_Decisions.smd`
  - [55%]: Loss-function-based decision rules under posterior uncertainty
- Not covered
  - [35%]: Deterministic/stochastic/adversarial planning and search (A\*, MPC,
    MCTS, minimax): no lecture in any of the four courses covers classical
    search/planning; deep Q-network specifics

## 12: Complex Decisions

### Topics
- Policy-Based and Actor-Critic Methods
  - Direct policy parameterization via gradient ascent: REINFORCE, PPO, TRPO
  - Actor-critic stability: combining value and policy in A2C, A3C, DDPG, TD3,
    SAC
  - High-dimensional and continuous action spaces where value-based methods
    struggle
- Planning with Learned Models and Deep Search
  - Model-based planning: Dyna-Q style imagined rollouts accelerate learning
  - Neural networks combined with tree search: AlphaGo, AlphaZero, MuZero
  - World models: learned dynamics enabling long-horizon reasoning
- Hierarchical and Modular Decision-Making
  - Decomposing complex decisions into subtasks, options, and temporal
    abstraction
  - Hierarchical abstractions that reduce the effective planning horizon
  - Knowledge reuse: transferring learned subpolicies across related problems
- Offline and Batch Learning
  - Learning from logged data with no live environment interaction
  - Distribution mismatch: divergence between behavior and learned policies
  - Algorithm family: batch Q-learning, offline RL, and CQL
- Multi-Agent and Game-Theoretic Algorithms
  - Reasoning about decisions when other agents are also acting
  - Equilibrium and coordination: Nash equilibrium solvers (CFR), QMIX, CommNet
  - Competitive and cooperative learning: MAPPO, MAAC, MADDPG

### Slides
- `book_springer/lectures_source/Lesson12.1_Complex_Decisions.smd`

### Lesson Materials
- `msml610/lectures_source/Lesson12.1-Reinforcement_learning.smd`
  - [95%]: Model-based vs. model-free RL, policy search, causal RL, structural
    causal models for MDPs, counterfactual credit assignment, deconfounding
    offline/off-policy data
- `book_cs_refreshers/lectures_source/Lesson95.Refresher_game_theory.smd`
  - [50%]: Nash equilibrium, zero-sum games and the minimax theorem, cooperative
    game theory, mechanism design, multi-agent RL (conceptual foundations, not
    QMIX/MAPPO/CFR specifics)
- `msml610/lectures_source/Lesson11.1-Decision_Making_with_Causal_Models.smd`
  - [55%]: Robustness under misspecification (aleatoric/epistemic uncertainty),
    exploration vs. exploitation in a causal-decision frame
- `msml610/lectures_source/Lesson08.7-Causal_experiments.smd`
  - [55%]: Policy evaluation and off-policy learning: logged-data learning
    without live interaction
- `msml610/lectures_source/Lesson10.2-Causal_Inference_for_Time_Series.smd`
  - [35%]: Temporal causal structure, feedback loops/simultaneity, VARs —
    causal-inference methodology, not policy-learning algorithms
- `msml610/lectures_source/Lesson09.1-Reasoning_over_time.smd`
  - [35%]: Markov process/state-transition fundamentals: loosely supports
    temporal abstraction, no hierarchical-RL content
- Not covered
  - [50%]: Policy-gradient/actor-critic algorithm details (REINFORCE, PPO, TRPO,
    A2C/A3C, DDPG/TD3/SAC), AlphaGo/AlphaZero/MuZero-style neural+tree search,
    hierarchical RL/options, offline RL algorithm specifics (CQL), deep MARL
    algorithms (QMIX, MAPPO, MADDPG, CFR)

## 13: Agentic Causal Reasoning

### Topics
- Causal Reasoning in LLMs
  - Pattern-based limits: why foundation models struggle with counterfactuals
  - Knowledge vs. reasoning: knowing facts does not mean reasoning causally
  - In-context learning: eliciting causal reasoning through careful prompting
- Structured Prompting and SCM-Augmented Agents
  - Chain-of-thought and tree-of-thought: causal decomposition and verification
  - Embedding causal world models directly into agent architectures
  - Model-based reasoning: planning using explicit causal models
- Tool-Use and Causal Simulation
  - External tools: calculators, simulators, and retrieval systems
  - Causal simulation: planning by running forward models
  - Grounding reasoning: connecting abstract plans to executable actions
- Trustworthy and Adaptive Agentic AI
  - Transparency and fairness in agent decision-making
  - Performativity: decisions that change the world they act on
  - Adaptive learning: updating models based on observed outcomes
- Probabilistic Integration and Online Discovery
  - Hybrid reasoning: combining symbolic causal models with learned
    probabilities
  - Uncertainty quantification maintained over causal structure itself
  - Online discovery: learning causal DAGs from live deployment data

### Slides
- N/A

### Lesson Materials
- `book.Agentic_AI/lectures_source/Lesson15.1-Causal_Reasoning_Agents.smd`
  - [98%]: LLM causal-reasoning strengths/limits, chain-of-thought
    causal-prompting frameworks, integrating causal DAGs with LLM workflows,
    causal agent architectures, causal MDPs and planning under causal
    uncertainty, transparency/fairness/robustness/safety through causal
    constraints
- `book.Agentic_AI/lectures_source/Lesson05.1-Reasoning_Memory_and_Planning.smd`
  - [60%]: World models for planning: LLM-as-world-model, WebDreamer "simulate
    before you act" (forward-model causal simulation)
- `book.Agentic_AI/lectures_source/Lesson04.1-LLM_Reasoning.smd`
  - [55%]: Chain-of-thought/tree-of-thought variants, self-consistency: general
    reasoning scaffolding, not causal-specific decomposition
- `book.Agentic_AI/lectures_source/Lesson07.1-Tool_use_and_retrieval.smd`
  - [45%]: Tool-use/retrieval architecture, grounding on enterprise knowledge —
    the tool-use half of the topic, no causal-simulation content
- `book.Agentic_AI/lectures_source/Lesson01.1-What_Is_An_Agentic_AI.smd`
  - [30%]: Generic perceive-plan-act agent architecture and tool/environment
    taxonomy: background context only
- `book.Agentic_AI/lectures_source/Lesson09.1_Post_training_and_verifiable_agents.smd`
  - [25%]: Verification, reward hacking: tangential to trustworthy-agent topic,
    not causal/fairness/performativity specific
- `book.Agentic_AI/lectures_source/Lesson03.1-History_of_LLM_Agents.smd`
  - [20%]: ReAct and agent history: background context
- Not covered
  - [30%]: Online causal-discovery algorithms learning DAGs from live deployment
    data, formal performativity mathematics, hybrid symbolic-probabilistic
    uncertainty quantified over causal structure itself

# Part V: Implementation, Deployment, Governance

## 14: Building Stakeholder Alignment

### Topics
- Eliciting and Refining Causal Knowledge
  - Structured methods: interviews, surveys, and consensus-building workshops
  - Handling disagreement: reconciling conflicting views among domain experts
  - Iterative refinement: cycling between expert input and data validation
- Communicating Causal Assumptions
  - Audience targeting: tailoring the message for experts, operations,
    leadership
  - Precision without jargon: explaining technical concepts accessibly
  - DAG visualization: making causal structures visible and open to debate
- Sensitivity Analysis as Communication Tool
  - Robustness demonstration: showing decisions hold across plausible
    assumptions
  - Threshold identification: finding the breaking points in assumptions
  - Building stakeholder confidence through transparent, thorough analysis
- Intervention Design and Risk Alignment
  - Lever identification: determining which variables can actually be controlled
  - Trade-offs: weighing costs, side effects, and feasibility of options
  - Cost asymmetry: aligning stakeholders on acceptable error rates
- Stakeholder Buy-In Before Deployment
  - Model adequacy review: does the model capture the business problem?
  - Assumption sign-off: stakeholders formally endorse the causal structure
  - Go/no-go decision: assessing readiness for production deployment

### Slides
- N/A

### Lesson Materials
- `msml610/lectures_source/Lesson08.1-Causal_AI_intro.smd`
  - [90%]: Causal AI workflow with hybrid teams, "Roles in Hybrid Teams," Step 4
    (Build Causal DAG), stakeholder alignment in business deployment
- `msml610/lectures_source/Lesson12.2-Causal_Discovery.smd`
  - [85%]: Using domain knowledge as constraints, combining discovery with
    expert judgment, validating a discovered DAG
- `msml610/lectures_source/Lesson11.1-Decision_Making_with_Causal_Models.smd`
  - [88%]: Communicating uncertainty to stakeholders; prior elicitation methods
    and pitfalls (structured elicitation of causal/prior knowledge)
- `msml610/lectures_source/Lesson08.4-Causal_networks.smd`
  - [75%]: Building a causal DAG, mediator/moderator/confounder types,
    documenting causal assumptions for review
- `msml610/lectures_source/Lesson07.2-Posterior_Based_Decisions.smd`
  - [60%]: Loss-function elicitation, Region of Practical Equivalence: framing
    acceptable error/cost trade-offs for stakeholder sign-off
- `msml610/lectures_source/Lesson08.4-Causal_networks.smd`
  - [45%]: Causal DAG structures (chains, forks, colliders, d-separation),
    confounding bias, backdoor adjustment: supports DAG
    visualization/communication only
- Not covered
  - [45%]: Formal structured elicitation methods (interviews, surveys, consensus
    workshops), disagreement-resolution protocols among experts, formal go/no-go
    sign-off procedures

## 15: Deployment, Monitoring, and Adaptation

### Topics
- From Notebook to Production
  - Infrastructure: moving models from experimentation to operational systems
  - Data pipelines: integrating real-time data streams into serving
  - Latency and throughput requirements for production performance
- Cost-Aware Rollout and Experimentation
  - Phased rollout, shadow mode, and canary analysis to manage risk
  - A/B testing vs. observational methods, combined in hybrid approaches
  - Continuous experimentation: sequential testing and incremental policy
    improvement
- Monitoring Causal Assumptions
  - Directly testable signals: effect stability and proxy validity
  - Indirectly testable signals: sensitivity bounds and negative controls
  - Dashboards: alerting when assumptions begin to break down
- Heterogeneous Deployment, Versioning, and Rollback
  - Stratified rollout across segments with different treatment effects
  - Error budgets and model versioning to track changes over time
  - Rollback procedures triggered when failures are detected
- Feedback, Adaptation, and Technical Debt
  - Learning from deployed decisions and adapting to concept drift
  - Iterative refinement of models and assumptions over time
  - System reliability, documentation, and management of legacy components

### Slides
- `book_springer/lectures_source/Lesson15.1_Deployment_Monitoring_And_Adaptation.smd`

### Lesson Materials
- `book_springer/lectures_source/Lesson15.1_Deployment_Monitoring_And_Adaptation.smd`
  - [98%]: Notebook-to-production gap, serving patterns, phased rollout
    (shadow/canary/ramp), guardrail metrics, assumption-monitoring taxonomy
    (directly/indirectly testable/domain-assessed), sensitivity analysis and
    E-values, negative controls, versioning, error budgets, rollback, feedback
    loops, performativity, technical debt
- `msml610/lectures_source/Lesson08.7-Causal_experiments.smd`
  - [85%]: A/B test design, power analysis, SRM, novelty/primacy effects,
    multi-armed bandits, experiment-vs-observe decision framework, off-policy
    evaluation
- `msml610/lectures_source/Lesson12.1-Reinforcement_learning.smd`
  - [75%]: Policy iteration, model-based/model-free RL, active/passive RL,
    off-policy deconfounding, causal generalization across environments
- `msml610/lectures_source/Lesson10.2-Causal_Inference_for_Time_Series.smd`
  - [65%]: Non-stationarity/trend detection, feedback loops and simultaneity,
    time-varying unobserved confounders
- `msml610/lectures_source/Lesson08.9-Effect_heterogeneity_and_Metalearners.smd`
  - [65%]: CATE, effect heterogeneity, cumulative-gain curves: basis for
    heterogeneous deployment and stratified rollout targeting
- `data605/lectures_source/Lesson02.3-Data_Pipelines.smd`
  - [55%]: ETL/ELT paradigms, workflow orchestration, data ingestion —
    production pipeline infrastructure
- Not covered
  - [20%]: Vendor-specific monitoring/dashboard tooling, organization-specific
    runbook templates

## 16: Trust, Explainability, Fairness, and Governance

### Topics
- Building Trust and Explainability
  - Transparency and justified confidence earned through evidence
  - Causal vs. statistical explainability: mechanisms versus SHAP and LIME
  - Serve-time explanations: mechanism-based and tailored to the audience
- Identifying and Responding to Failures
  - Runtime detection: root causes in misspecification and distribution drift
  - Diagnosis versus proactive monitoring of production behavior
  - Emergency response: rapid detection followed by rollback
- Fairness Under Deployment
  - Disparate impact and fairness monitoring across subgroups
  - Heterogeneous effects: balancing the fairness-robustness tradeoff
  - Planning ahead for effect variation across different groups
- Human Oversight and Governance
  - Escalation and override: human-in-the-loop review protocols
  - Trust calibration: knowing when to rely on versus override the system
  - Audit trails: documentation, versioning, and impact assessment
- Regulatory Compliance and Guardrails
  - GDPR, the EU AI Act, and other fairness standards
  - Documentation and audit preparation, with legal coordination as needed
  - Guardrails: safety constraints and graceful degradation under failure

### Slides
- N/A

### Lesson Materials
- `msml610/lectures_source/Lesson13.1-Explainability.smd`
  - [95%]: Causal vs statistical explainability, SHAP, LIME, counterfactual
    explanations, faithfulness/stability of explanations
- `book.Agentic_AI/lectures_source/Lesson15.1-Causal_Reasoning_Agents.smd`
  - [90%]: Counterfactual fairness, path-specific effects, causal definitions of
    fairness/discrimination, causal constraints for fair models,
    safety-through-causal-constraints (guardrails, harmful-outcome prevention),
    transparency for trustworthy autonomy, human-in-the-loop override, causal
    monitoring/adaptation
- `msml610/lectures_source/Lesson08.4-Causal_networks.smd`
  - [85%]: Causal mechanisms, mediation analysis, direct vs indirect effects —
    foundation for path-specific fairness reasoning
- `msml610/lectures_source/Lesson08.1-Causal_AI_intro.smd`
  - [70%]: Interpretability/explainability techniques, causal AI for trustworthy
    and transparent systems
- `book.Agentic_AI/lectures_source/Lesson01.1-What_Is_An_Agentic_AI.smd`
  - [55%]: Human-in-the-loop vs full autonomy trade-offs, audit and
    accountability, error-catching before real-world consequences
- `msml610/lectures_source/Lesson08.9-Effect_heterogeneity_and_Metalearners.smd`
  - [35%]: Effect heterogeneity (CATE) across subgroups: relevant to detecting
    disparate treatment effects, though it defines no fairness metrics
- Not covered
  - [55%]: Regulatory compliance specifics (GDPR, EU AI Act provisions), formal
    audit-trail/documentation standards, legal coordination processes

# Appendix
