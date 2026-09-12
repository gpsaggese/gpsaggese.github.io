// class_scripts/create_book_toc_from_slides.py --input book.Causal_Probabilistic_ML/book_map.md --max_level 2 --in_place
// Current TOC: book.Causal_Probabilistic_ML/book_toc.md
// Official TOC: ~/src/notes1/book.manning.Causal_Probabilistic_Machine_Learning/manning.proposal_v3.toc.md

// Apply '/Users/saggese/src/umd_classes1/helpers_root/.claude/skills/slides.rules.md:117:## Use Bold for Slide Sections' to ...

# Part I: Understanding Probabilistic and Causal ML

## 1: The Need for Probabilistic and Causal Machine Learning

### TODOs
- [ ] Finish slides (80%, 66)
- [ ] Add / integrate probabilistic / explainable
- [ ] Add "small data"
  - /Users/saggese/src/csfy1/blog/docs/posts/AI_for_Optimal_Decision-Making.md
  - /Users/saggese/src/csfy1/blog/docs/posts/Cracking_the_Long_Tail_of_Data_Science_Problems.md
  - /Users/saggese/src/csfy1/blog/docs/posts/Data_Is_Dumb_And_Thats_Why_Causality_Matters.md
- [ ] Keep tables in the text
- [ ] Implement in tikz the pic
  - Use inkspace / svg from msml610/lectures_source/Lesson01-1.aux.md
- [ ] Move Causal AI in business in a separate chapter
- [ ] Read papers
- [ ] Read books

### Lessons
- `msml610/lectures_source/Lesson08.1-Causal_AI_intro.txt`

### Current TOC
// `msml610/lectures_source/Lesson08.1-Causal_AI_intro.txt`
- Introduction and Motivation (7)
  - Background (3)
  - What ML Systems Can and Cannot Tell You (4)
- Why Causal AI Matters (15)
  - Problems with Traditional AI (9)
  - Optimization vs. Inference vs. Decision Theory (2)
  - The Cost of Ignoring Causality (4)
- Causal AI Fundamentals (10)
  - The Ladder of Causation (4)
  - Correlation vs Causation Models (3)
  - Data Science vs. Decision Science (3)
- Causal AI in Business (18)
  - Business Context and Motivation (3)
  - The Causal AI Workflow (11)
  - Explainability and Interpretability (4)

### Target TOC
- The Problem: When Prediction Fails
- What ML Systems Can and Cannot Tell You
- The Ladder of Causation
- Data Science vs. Decision Science
- Tools and Tutorials — Introduction to causal DAGs using real-world examples
- Summary

### Tutorials

### Related packages
- DoWhy (6,600): Causal inference using graphical models
- PyWhy (2,400): Python ecosystem for causal inference
- Azua (1,400): Causal decision-making framework

### Related books
- [B001] Agrawal et al., "Prediction Machines", 2018
- [B002] Pearl et al., "The Book of Why", 2018
- [B003] Huyen, "Designing Machine Learning Systems", 2022
- [B039] Hurwitz & Thompson, "Causal Artificial Intelligence", 2024
- [B040] Brynjolfsson & McAfee, "Machine, Platform, Crowd", 2021
- [B041] Angrist & Pischke, "Probability, Statistics, and Causal Inference", 2020

### Related Papers
- [P001] Ribeiro et al., "Why Should I Trust You? Explaining the Predictions of
  Any Classifier", 2016 https://arxiv.org/pdf/1602.04938.pdf
- [P002] Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions",
  2017 https://arxiv.org/pdf/1705.07874.pdf
- [P003] Taori et al., "Data Feedback Loops: Model-driven Amplification of
  Dataset Biases", 2020 https://proceedings.mlr.press/v202/taori23a/taori23a.pdf
- [P004] Imbens & Wooldridge, "Recent Developments in the Econometrics of Program
  Evaluation", 2018 https://www.nber.org/papers/w24318
- [P005] Pearl, "Simpson's Paradox, Confounding, and Collapsibility", 1999
  https://bayes.cs.ucla.edu/BOOK-2K
- [P006] Kaddour et al., "Challenges and Opportunities with Causal Discovery
  Algorithms", 2023 https://www.nature.com/articles/s41598-020-59669-x
- [P007] Zhang et al., "A Survey on Causal Inference", 2020
  https://arxiv.org/pdf/2002.05209.pdf
- [P008] Peters et al., "Causality: Models, Learning, and Inference", 2023
  https://arxiv.org/pdf/2012.13993.pdf
- [P009] Pearl, "The Seven Pillars of Causal Reasoning with Reflections on
  Machine Learning", 2021
  https://cacm.acm.org/research/seven-pillars-of-causal-reasoning-with-reflections-on-machine-learning/
- [P010] Joshi et al., "Towards Realistic Counterfactual Explanations with
  Contrastive Pertinent Features", 2019 https://arxiv.org/pdf/1906.04957.pdf
- [P011] Peters et al., "Elements of Causal Inference: Foundations and Learning
  Algorithms", 2020
  https://mitpress.mit.edu/9780262037310/elements-of-causal-inference/

## 2: Bayesian Networks and Probabilistic Reasoning

### TODOs
- [ ] Finish slides (80%, 77)
- [x] Make sure to use definitions and other tags
- [ ] Check slides for mistakes
- [ ] Improve graphics and visuals
- [ ] Generate book
- [ ] Finalize tutorial

### Lessons
- `msml610/lectures_source/Lesson06.1-Bayesian_Networks.txt`
- `msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.txt`

### Current TOC
// `msml610/lectures_source/Lesson06.1-Bayesian_Networks.txt`
- Logic-Based AI Under Uncertainty (10)
  - Problem (3)
  - Solution (7)
- Probabilistic Reasoning (18)
  - Conditional Independence (5)
  - Bayesian Networks (13)

// `msml610/lectures_source/Lesson06.2-Using_Bayesian_Networks.txt`
- Semantics of Bayesian Networks (8)
- Constructing a Bayesian Network (34)
- Exact Inference in Bayesian Networks (4)
- Approximate Inference in Bayesian Networks (28)

### Target TOC
- Probability and Conditional Independence
- Bayesian Networks
- Constructing a Bayesian Network
- Exact and Approximate Inference
- Tools and Tutorials — Implementing Bayesian Networks in PyMC
- Summary

### Tutorials
- pgmpy
- DoWhy, CausalML, CausalNex, gcastle

### Related packages
- pgmpy (5,000): Probabilistic Graphical Models (Bayesian Networks, inference)
- Pomegranate (3,600): Probabilistic modeling library
- CausalNex (3,000): Causal reasoning with Bayesian Networks
- bnlearn (1,900): Causal discovery with Bayesian networks
- CausalGraphicalModels (1,500): Toolkit for causal graphs in Python

### Related books
- [B004] Koller et al., "Probabilistic Graphical Models", 2009
- [B005] Jensen et al., "Bayesian Networks and Decision Graphs", 2007
- [B006] Bishop, "Pattern Recognition and Machine Learning", 2006

## 3: Causal DAGs and Structural Models

### TODOs
- [ ] Finish slides (80%, 77)
- [ ] Make sure to use definitions and other tags
- [ ] Check slides for mistakes
- [ ] Improve graphics and visuals
- [ ] Generate book
- [ ] Finalize tutorial

### Lessons
- `msml610/lectures_source/Lesson08.3-Do_Calculus.txt`

### Current TOC
// `msml610/lectures_source/Lesson08.3-Do_Calculus.txt`
  - Intervention and Counterfactuals (5)
  - Randomized Controlled Trial (3)
  - Backdoor Adjustment (6)
  - Frontdoor Adjustment (5)
  - Do-Calculus (3)

### Target TOC
- Causal vs. Observational Graphs
- Structural Causal Models
- Special Variable Types (mediators, moderators, confounders, colliders)
- Building Causal DAGs from Domain Knowledge
- Tools and Tutorials — Building and visualizing causal DAGs
- Summary

### Tutorials
- PyMC
- arviz
- xarray

### Related packages
- Dagitty (1,500): DAG creation and causal effect identification
- CausalGraphicalModels (1,500): Toolkit for causal graphs in Python
- Tetrad (1,100): Suite for causal model discovery and analysis
- Geminos (500): Causal diagram generation and analysis

### Related books
- [B007] Pearl, "Causality", 2009
- [B008] Peters et al., "Elements of Causal Inference", 2017
- [B009] Morgan et al., "Counterfactuals and Causal Inference", 2014

## 4: From Causal Models to Code

### TODOs
- [ ] Finish slides
- [ ] Make sure to use definitions and other tags
- [ ] Check slides for mistakes
- [ ] Improve graphics and visuals
- [ ] Generate book
- [ ] Merge the slides into a single one
- [ ] Finalize tutorial

### Lessons
- `msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.txt`
- `msml610/lectures_source/Lesson07.2-Posterior_Based_Decisions.txt`
- `msml610/lectures_source/Lesson07.3-Hierarchical_Models.txt`
- `msml610/lectures_source/Lesson07.4-Generalized_Linear_Models.txt`
- `msml610/lectures_source/Lesson07.5-Bayesian_Model_Comparison.txt`

### Current TOC
// `msml610/lectures_source/Lesson07.1-Intro_to_Probabilistic_Programming.txt`
- Concepts (12)
- Coin Example (18)
  - Analytical Approach (7)
  - Frequentist vs Bayesian (6)
  - Probabilistic Programming (5)

// `msml610/lectures_source/Lesson07.2-Posterior_Based_Decisions.txt`
- Posterior-Based Decisions (34)
  - Chemical Shift: Example (5)
  - Posterior Predictive Checks (10)
  - Groups Comparison (5)

// `msml610/lectures_source/Lesson07.3-Hierarchical_Models.txt`
- Hierarchical Models (22)

// `msml610/lectures_source/Lesson07.4-Generalized_Linear_Models.txt`
- Generalized Linear Models (29)
  - Simple Linear Model (15)
  - Logistic Regression (8)
  - Multiple Linear Regression (6)

// `msml610/lectures_source/Lesson07.5-Bayesian_Model_Comparison.txt`
- Bayesian Model Comparison (39)
  - The Balance Between Simplicity and Accuracy (3)
  - Measures of Predictive Accuracy (11)
  - Bayesian Model Selection and Ensemble (2)
  - Bayesian Hypothesis Testing (7)
  - Regularizing Priors (2)

### Target TOC
- Bayesian Inference in Practice
- Generalized Linear Models
- Hierarchical Models
- Posterior-Based Decisions
- Tools and Tutorials — Posterior workflows in PyMC
- Summary

### Tutorials
- `msml610/tutorials/L07_prob_programming/L07_01_bayesian_coin.ipynb`
- `msml610/tutorials/L07_prob_programming/L07_02_probabilistic_programming.ipynb`
- `msml610/tutorials/L07_prob_programming/L07_02_robust_modeling.ipynb`
- `msml610/tutorials/L07_prob_programming/L07_03_hierarchical_models.ipynb`
- `msml610/tutorials/L07_prob_programming/L07_04_generalized_linear_models.ipynb`
- `msml610/tutorials/L07_prob_programming/L07_05_evaluating_models.ipynb`

### Related packages
- pyro (8,200): Probabilistic programming on PyTorch
- PyMC3 (7,900): Bayesian statistical modeling framework
- TensorFlow Probability (3,800): Probabilistic reasoning library
- Numpyro (3,200): Probabilistic programming with NumPy/JAX
- PyStan (2,400): Python interface to Stan
- CmdStanPy (1,300): Python interface to CmdStan

### Related books
- [B010] Imbens et al., "Causal Inference for Statistics, Social, and Biomedical
  Sciences", 2015
- [B011] Sutton, "Python Causal Analysis", 2024
- [B012] Burkov, "Machine Learning Engineering", 2020

# Part II: Estimating Causal Effects

## 5: Interventions, Experiments, and Adjustments

### TODOs
- [ ] Finish slides
- [ ] Make sure to use definitions and other tags
- [ ] Check slides for mistakes
- [ ] Improve graphics and visuals
- [ ] Generate book
- [ ] Finalize tutorial

### Lessons
- `msml610/lectures_source/Lesson08.3-Do_Calculus.txt`

### Current TOC
// `msml610/lectures_source/Lesson08.3-Do_Calculus.txt`
  - Intervention and Counterfactuals (5)
  - Randomized Controlled Trial (3)
  - Backdoor Adjustment (6)
  - Frontdoor Adjustment (5)
  - Do-Calculus (3)

### Target TOC
- Interventions and Counterfactuals
- Randomized Controlled Trials
- Observational Adjustment
- Do-Calculus
- Tools and Tutorials — Using DoWhy for causal inference
- Summary

### Tutorials
- DoWhy, CausalML, CausalNex, gcastle

### Related packages
- DoWhy (6,600): Causal inference using graphical models
- CausalImpact (5,600): Causal inference for intervention analysis
- CausalML (4,800): Uplift modeling and causal inference
- EconML (4,600): ML-based causal effect estimation

### Related books
- [B013] Rosenbaum, "Observation and Experiment", 2017
- [B014] Angrist et al., "Mostly Harmless Econometrics", 2008
- [B015] Gerber et al., "Field Experiments", 2012

## 6: Causal Identification and Estimation

### TODOs
- [ ] Finish slides
- [ ] Make sure to use definitions and other tags
- [ ] Check slides for mistakes
- [ ] Improve graphics and visuals
- [ ] Generate book
- [ ] Finalize tutorial

### Lessons
- `msml610/lectures_source/Lesson08.4.txt`

### Current TOC
// `msml610/lectures_source/Lesson08.4.txt`
- Introduction to Causal Inference (50)
- Randomized Experiments and Stats Review (8)
- Graphical Causal Models (34)
- The Unreasonable Effectiveness of Linear Regression (20)
  - Feature Selection Dilemma
- Propensity score (20)
- Effect heterogeneity (18)
- Metalearners (26)
- Difference-in-differences (5)
  - Canonical Difference-in-Differences (5)
  - Identification Assumption
- Synthetic control
- Geo and switchback experiments
- Non-compliance and instruments
- Next steps

### Target TOC
- The Identification Problem
- Classical Strategies (instrumental variables, RDD, DiD)
- Matching and Propensity Scores
- Modern Causal Forests
- Sensitivity to Unmeasured Confounding
- Case Study: Healthcare Treatment Effects
- Tools and Tutorials — EconML and CausalML
- Summary

### Tutorials
- DoWhy, CausalML, CausalNex, gcastle
- `msml610/tutorials/L08_causal_inference/L08_04_01_causal_inference.ipynb`
- `msml610/tutorials/L08_causal_inference/L08_04_02_causal_inference.ipynb`
- `msml610/tutorials/L08_causal_inference/L08_04_05_propensity_score.ipynb`
- `msml610/tutorials/L08_causal_inference/L08_04_07_metalearners.ipynb`
- `msml610/tutorials/L08_causal_inference/L08_04_08_difference_in_difference.ipynb`

### Related packages
- DoWhy (6,600): Causal inference using graphical models
- CausalML (4,800): Uplift modeling and causal inference
- EconML (4,600): ML-based causal effect estimation
- causal-learn (3,200): Causal discovery and inference toolkit
- CausalPy (1,200): Causal effect estimation and visualization
- CausalInference (1,100): Statistical causal inference methods

### Related books
- [B016] Angrist et al., "Mastering 'Metrics", 2014
- [B017] Wooldridge, "Econometric Analysis of Cross Section and Panel Data", 2010
- [B018] Huntington-Klein, "The Effect", 2021

## 7: Explainability and Causal Attribution

### TODOs
- [x] Make sure to use definitions and other tags
- [ ] IN PROGRESS: Finalize tutorial
- [ ] Check slides for mistakes
- [ ] Improve graphics and visuals
- [ ] Generate book

### Lessons
- `msml610/lectures_source/Lesson13.1-Explainability.txt`

### Current TOC

### Target TOC
- Explanation vs. Causality
- Explanation Methods (SHAP, LIME)
- The Critical Gap: Feature Importance Is Not Causality
- Causal Attribution
- Decision Support
- Tools and Tutorials — SHAP, LIME, DiCE, and DoWhy
- Summary

### Tutorials
- tutorials/shap
- tutorials/lime

### Related packages
- SHAP → shap package with real datasets
- LIME → lime package with tabular & text examples
- Permutation Importance → sklearn.inspection
- Counterfactuals → alibi or dice-ml
- DoWhy (6,600): Causal inference using graphical models
- CausalNex (3,000): Causal reasoning with Bayesian Networks
- CausalGraphicalModels (1,500): Toolkit for causal graphs in Python

### Related books
- [B019] Molnar, "Interpretable Machine Learning", 2022
- [B020] Kearns et al., "The Ethical Algorithm", 2019
- [B021] Kohavi et al., "Trustworthy Online Controlled Experiments", 2020

## ?: Probabilistic Inference for Time Series

### Lessons
- `msml610/lectures_source/Lesson09.1-Reasoning_over_time.txt`
- `msml610/lectures_source/Lesson09.2-Hidden_Markov_Models.txt`
- `msml610/lectures_source/Lesson09.4-gh_Filter.txt`
- `msml610/lectures_source/Lesson09.5-Kalman_Filter.txt`
- `msml610/lectures_source/Lesson09.6-Dynamic_Bayesian_Networks.txt`

### Target TOC

### TODOs
- [ ] Review / reorg

## 8: Causal Inference for Time Series

### TODOs
- [ ] Finish slides (40%, 79 + ...)
- [x] Make sure to use definitions and other tags
- [ ] Check slides for mistakes
- [ ] Improve graphics and visuals
- [ ] Generate book
- [ ] Finalize tutorial

### Lessons
- `msml610/lectures_source/Lesson10.1-Timeseries_forecasting.txt`
- `msml610/lectures_source/Lesson10.2-Causal_Inference_for_Time_Series.txt`

### Current TOC
// `msml610/lectures_source/Lesson10.1-Timeseries_forecasting.txt`
- Time Series (21)
  - Basic definition (9)
  - Time series operators (6)
  - Time series decomposition (6)
- Classical Methods (33)
  - Simple Models for Stochastic Process (3)
  - Autoregressive models (11)
  - Moving average models (5)
  - ARMA(p, q) process (6)
  - ARCH model (8)
- Modern Approaches (10)
- Special techniques for time series modeling (14)

// `msml610/lectures_source/Lesson10.2-Causal_Inference_for_Time_Series.txt`
- Time Series vs. Cross-Sectional Causality (10)
  - Temporal Causal Structures (3)
  - Challenges Specific to Time Series (5)
  - When Temporal Structure Helps and When It Misleads (2)
- Granger Causality (7)
  - Definition and Intuition (3)
  - Assumptions and Limitations (2)
  - Practical Examples (2)
- Interrupted Time Series (ITS) (8)
  - Design and Estimation (4)
  - ITS and Regression Discontinuity (2)
  - Applications in Causal Inference (2)
- Difference-in-Differences (DiD) (10)
  - Parallel Trends Assumption (5)
  - Estimation and Robustness (2)
  - Extensions: Multiple Time Periods (3)
- Synthetic Control Methods (9)
  - Constructing a Counterfactual from Donor Series (2)
  - Weighted Combinations and Optimal Weights (3)
  - When Synthetic Control Succeeds and Fails (4)

### Target TOC
- Time Series vs. Cross-Sectional Causality
- Granger Causality
- Interrupted Time Series
- Difference-in-Differences
- Synthetic Control Methods
- Tools and Tutorials — CausalImpact and CausalPy
- Summary

### Tutorials
- Prophet, FilterPy, tsfresh, GluonTS
- `msml610/tutorials/L09_kalman_filter`

### Related packages
- TiMINo (3,600): Time-series causal discovery under independent noise
  assumptions
- orbit (2,000): Bayesian time series models
- HMMlearn (1,600): Hidden Markov Models with sklearn API
- BETS (350): Time-series causal network inference using elastic net regression

### Related books
- [B022] Hamilton, "Time Series Analysis", 1994
- [B023] Hyndman et al., "Forecasting: Principles and Practice", 2021
- [B024] Molak, "Causal Inference and Discovery in Python", 2023

## 9: A/B Testing and Experimentation

### Lessons
- `msml610/lectures_source/Lesson09.3-Multi_Armed_Bandits.txt`

### Current TOC
// `msml610/lectures_source/Lesson09.3-Multi_Armed_Bandits.txt`
- Introduction (24)
- Algorithms (30)
- Bayesian Approaches (40)

### Target TOC
- Randomization as Causal Identification
- A/B Test Design
- Beyond Standard A/B Tests (switchbacks, multi-armed bandits)
- Sequential Decision-Making
- Tools and Tutorials — CausalML and CausalPy
- Summary

### Tutorials
- `msml610/tutorials/L09_multi_armed_bandits`

### Related packages
- Vowpal Wabbit (8,600): High-performance online learning with contextual bandit
  support
- contextualbandits (1,700): Python implementations of contextual bandit
  algorithms
- MABWiser (280): Contextual and non-contextual multi-armed bandit library
- PyXAB (200): Research-focused library for X-armed bandits

### Related books
- [B021] Kohavi et al., "Trustworthy Online Controlled Experiments", 2020
- [B025] Siroker et al., "A/B Testing", 2013
- [B026] Thomke, "Experimentation Works", 2020

## 10: Causal Discovery

### Lessons
- `msml610/lectures_source/Lesson12.1-Causal_Discovery.txt`

### Current TOC
// `msml610/lectures_source/Lesson12.1-Causal_Discovery.txt`
- The Discovery Problem (11)
  - Inferring Causal Structure from Observational Data (4)
  - Identifiability and Causal Sufficiency (5)
  - Practical Limitations (2)
- When to Use Discovery vs Domain Knowledge (4)
  - Discovery as Hypothesis Generation (2)
  - Combining Discovery with Domain Knowledge (2)
- Discovery Algorithm Families (22)
- Challenges and Validation (8)

### Target TOC
- When Discovery Works and When It Doesn't
- Discovery vs. Domain Knowledge
- Discovery Algorithms (PC, FCI, GES, NOTEARS, LiNGAM)
- Validation and Refutation
- Tools and Tutorials — causal-learn and LiNGAM
- Summary

### Tutorials
- From
  - IN PROGRESS: tutorials/dowhy
  - causal-learn
  - CDT
  - LiNGAM

### Related packages
- causal-learn (3,200): Causal discovery and inference toolkit
- Causal Discovery Toolbox (3,100): Framework for discovering causal structure
  from observational data
- gCastle (2,400): Toolkit for causal structure learning and trustworthy AI
- bnlearn (1,900): Causal discovery with Bayesian networks
- LiNGAM (1,400): Discovery of linear non-Gaussian causal models
- Tetrad (1,100): Suite for causal model discovery and analysis

### Related books
- [B007] Pearl, "Causality", 2009
- [B008] Peters et al., "Elements of Causal Inference", 2017
- [B004] Koller et al., "Probabilistic Graphical Models", 2009
- [B027] Spirtes et al., "Causation, Prediction, and Search", 2000

### Related Papers
- [P012] Zanga et al., "A Survey on Causal Discovery: Theory and Practice", 2023
  https://arxiv.org/pdf/2305.10032
- [P013] Glymour et al., "Review of Causal Discovery Methods Based on Graphical
  Models", 2020 https://par.nsf.gov/servlets/purl/10125762
- [P014] Guo et al., "A survey on causal inference", 2015
  https://qiniu.pattern.swarma.org/attachment/A%20Survey%20on%20Causal%20Inference.pdf
- [P015] Meek, "Causal inference and causal explanation with background
  knowledge", 2006 https://arxiv.org/pdf/1302.4972
- [P016] Maathuis et al., "Estimating high-dimensional intervention effects from
  observational data", 2012
- [P017] Kalisch et al., "Causal inference using invariant prediction", 2010
- [P018] Zhang et al., "On the identifiability of the post-nonlinear causal
  model", 2009 https://arxiv.org/pdf/1205.2599
- [P019] Loh et al., "Causal discovery with identifiable nonlinear models", 2018
- [P020] Uhler et al., "Geometries of faithfulness in graphical models", 2017
  https://arxiv.org/abs/1608.00191
- [P021] Gamella et al., "Active learning of linear causal models", 2021
  https://arxiv.org/abs/2006.05690
- [P022] Wang et al., "RL-based structural equation modeling", 2019
- [P023] Mani et al., "Learning the structure of causal models with experimental
  and observational data", 2013
- [P024] Solus et al., "Causal discovery with unknown interventions", 2019
- [P025] Squires et al., "Active structure learning of causal DAGs", 2018
  https://arxiv.org/abs/2011.00641
- [P026] Sachs et al., "Causal protein-signaling networks derived from
  multiparameter single-cell data", 2015
- [P027] Castelo et al., "Structural, syntactic, and statistical issues in
  learning Bayesian networks from data", 2008
- [P028] Zheng et al., "DAGs with NO TEARS: Continuous Optimization for Structure
  Learning", 2018
  https://papers.nips.cc/paper/8157-dags-with-no-tears-continuous-optimization-for-structure-learning.pdf
