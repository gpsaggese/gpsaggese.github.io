# Tetrad Tutorial

<!-- toc -->
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Knowledge Object](#knowledge-object)
- [TetradSearch](#tetradsearch)
- [Tetrad with JPype](#tetrad-with-jpype)
- [Simulating Data with Tetrad](#simulating-data-with-tetrad)
<!-- tocstop -->

## Introduction
### Overview
Tetrad is a Java library of various interfaces for exploring causal explanations of data. With the py-tetrad python interface and JPype, we can access all of Tetrad's functionality, from creating graphs of relationships within data to simulating data from constructed models.

### Goals
By the end of this tutorial, you'll be able to:
- Understand the basic functionality and usage of Tetrad
- Set up JPype for use with py-tetrad
- Create relationship graphs using Tetrad
- Explore causal explanations of data

## Prerequisites
Py-tetrad requires:
- Java JDK 21+
- Python 3.12+
- JPype library
- PyTorch
To ensure these are met, please refer to the README for instructions on how to build and run the Docker container.

## Knowledge Object
Tetrad's Knowledge object allows for prior knowledge to be input as constraints on the model. This is done by stratifying the features into tiers, requiring certain edges, and forbidding other edges.

Knowledge tiers allow us to separate our variables into a timeline where variables in higher-numbered tiers are unable to affect those of lower-numbered tiers. In this way, we can constrain the model against finding certain backward relationships.

Required edges are those that we know from prior knowledge must be included in the model. Conversely, forbidden edges are those that we know from prior knowledge cannot exist within the model. Specifying these helps to constrain the model such that it may converge more accurately. 

## TetradSearch
The TetradSearch class is a wrapper for many searches in Tetrad that simplifies calling various search algorithms. It allows one to set scoring algorithms, parameters, and run searches on datasets. TetradSearch calls code from the java A full list of the available search algorithms, scoring algorithms, and parameters can be found in the [official Tetrad manual](https://htmlpreview.github.io/?https:///github.com/cmu-phil/tetrad/blob/development/tetrad-lib/src/main/resources/docs/manual/index.html#knowledge_box). I will describe the algorithms used in this project and their arguments here, but there are many more available algorithms in the Tetrad library.

### FCI
The FCI algorithm can be run using the TetradSearch wrapper. It is a constraint-based algorithm that produces a partial ancestral graph (PAG) comprised of conditional independence relationships. Input data may be continuous, discrete, or mixed. This algorithm is compatible with time series data by setting the time lag parameter on TetradSearch via `set_time_lag`.

**Function**: `TetradSearch.run_fci`

**Arguments:**
- `depth`: integer - The depth of search where -1 indicates no limit
- `stable_fas`: boolean - Whether the stable version of the Peter-Clark (PC) adjacency search should be used. This can help limit the effect that the order of variables has on the end result, especially in high-dimensional data.
- `max_disc_path_length`: integer - The maximum discriminating path length. Finding a discriminating path can be computationally expensive. -1 indicates no maximum path length. 
- `complete_rule_set_used`: boolean - Indicates whether the full FCI ruleset should be used. If false, a simpler ruleset guaranteeing arrow completeness is used. If true, the full ruleset is used and guarantees tail completeness as well. 
- `guarantee_pag`: boolean - Whether the search should guarantee a legal partial ancestral graph (PAG) as output. This defaults to false and comes with the risk that the search may output a PAG despite certain assumptions not being met. 

### RFCI
The RFCI algorithm can be run using the TetradSearch wrapper. It is a modification to the FCI algorithm above that is faster and nearly as informative.

**Function**: `TetradSearch.run_rfci()`

**Arguments:**
- `depth`: integer - The depth of search where -1 indicates no limit.
- `stable_fas`: boolean - Whether the stable version of the Peter-Clark (PC) adjacency search should be used.
- `max_disc_path_length`: integer - The maximum discriminating path length. -1 indicates no maximum path length. 
- `complete_rule_set_used`: boolean - Indicates whether the full FCI ruleset should be used.

### FGES
The FGES algorithm can be run using the TetradSearch wrapper. FGES is an optimized and parallelized Bayesian algorithm based on Greedy Equivalent Search. This algorithm requires a scoring algorithm like SEM BIC, BDEU, or Conditional Gaussian be set through TetradSearch. FGES can be run on continuous, discrete, or mixed data. It produces a Complete Partially Directed Acyclic Graph (CPDAG), which is an equivalent class of Directed Acyclic Graphs (DAGs).

FGES search also outputs a score for the overall graph and each individual node. These scores, produced by the designated scoring algorithm, represent the comparitive accuracy of the model. A higher score indicates a better model. Scores for individual nodes indicate how well-founded their position in the model is. 

**Function**: `TetradSearch.run_fges()`

**Scoring Algorithms**:
- `use_sem_bic`: Use SEM BIC for scoring. This is for continuous datasets.
- `use_bdeu`: Use BDEU for scoring. This is for discrete datasets.
- `use_conditional_gaussian_score`: Use Conditional Gaussian for scoring. This is for mixed datasets.

**Arguments:**
- `symmetric_first_step`: integer - Whether the search should first calculate both directions of a relationship (i.e. X->Y and Y<-X) and using the higher score before proceeding with the search.
- `max_degree`: integer - The maximum degree of any node in the graph. -1 indicates no limit.
- `parallelized`: boolean - Whether the search should be parallelized.
- `faithfulness_assumed`: boolean - Whether one-edge faithfulness should be assumed. This means that if X and Y are independent, then it is assumed that X and Y are independent given Z.

### IMaGES and IMaGES-BOSS
The IMaGES algorithm can be run using JPype, creating a Parameters object, and converting your data to a Tetrad DataModel. IMaGES is based on FGES but for multiple datasets. 

IMaGES-BOSS uses the BOSS algorithm in place of FGES. This is particularly designed for continuous variables. The BOSS (Best Order Score Search) algorithm is based on the Greedy Sparsest Perumutation (GSP) algorithm and assumes causal sufficiency. Both algorithms output a CPDAG. 

**Function**: `Images.search()` and `ImagesBoss.search()`

**Arguments:**
- `dataSets` - List\<DataModel\>, A list of Tetrad DataModel objects containing the data.
- `parameters` - Parameters, An object for holding parameters to be passed to Tetrad's Java libraries. Some available parameters include:
    * `NUM_RUNS`: integer - The number of runs the search should perform >= 1.
    * `RANDOM_SELECTION_SIZE`: integer - The number of datasets that should be taken per random sample.
    * `SYMMETRIC_FIRST_STEP`: integer - Whether the search should first calculate both directions of a relationship (i.e. X->Y and Y<-X) and using the higher score before proceeding with the search.
    * `MAX_DEGREE`: integer - The maximum degree of any node in the graph. -1 indicates no limit.
    * `PARALLELIZED`: boolean - Whether the search should be parallelized.
    * `FAITHFULNESS_ASSUMED`: boolean - Whether one-edge faithfulness should be assumed.

## Tetrad with [JPype](https://jpype.readthedocs.io/en/latest/index.html)
Some of Tetrad's functionality is not included directly in py-tetrad and must instead be called through JPype. JPype is a python module that provides access to Java from Python. Since Tetrad is a Java library, JPype provides the ability to run Tetrad via Python code. This allows full access to all of Tetrad's search algorithms and their parameters as demonstrated by the IMaGES example in the associated tetrad.API.ipynb notebook.

## Simulating Data with Tetrad
### Simulating Data with Corresponding True Causal Graphs
The py-tetrad package includes several methods for simulating datasets and their corresponding true causal graphs. Continuous, discrete, or mixed datasets may be simulated with input parameters including:
- sample size 
- number of measured variables
- number of latent variables
- average degree of the graph
For simulating continuous data, there are the additional available input parameters:
- high and low ends of the coefficient ranges for the underlying data distribution
- high and low ends of the variance ranges for the underlying data distribution
For simulating discrete data, there are the additional available input parameters:
- minimum number of categories for discrete variables
- maximum number of categories for discrete variables
Finally, for mixed datasets one can specify:
- the percentage of variables that are discrete
Using JPype to access Tetrad directly allows for even more control over the simulated data, including setting the random seed for reproducibility. 

### Simulating Data with a Causal Perceptron Network
The CausalPerceptronNetwork class provided by py-tetrad enables us to simulate data from an existing graph. In this way, one can input a graph created from another search algorithm (I.E. FGES) and simulate data based upon that graph. The CausalPerceptronNetwork inputs the following parameters:
- a graph
- number of samples
- noise distribution on the graph nodes
- number of hidden dimensions
- input scale
- activation module
- nonlinearity 
- probability of discrete variables
- range of number of categories in discrete variables
- seed
