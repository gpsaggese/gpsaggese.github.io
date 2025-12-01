# Tetrad Tutorial

<!-- toc -->
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Tetrad's Knowledge Object](#tetrad's-knowledge-object)
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

## Tetrad's Knowledge Object
Tetrad's Knowledge object allows for prior knowledge to be input as constraints on the model. This is done by stratifying the features into tiers, requiring certain edges, and forbidding other edges.

Knowledge tiers allow us to separate our variables into a timeline where variables in higher-numbered tiers are unable to affect those of lower-numbered tiers. In this way, we can constrain the model against finding certain backward relationships.

Required edges are those that we know from prior knowledge must be included in the model. Conversely, forbidden edges are those that we know from prior knowledge cannot exist within the model. Specifying these helps to constrain the model such that it may converge more accurately. 

## TetradSearch
The TetradSearch class is a wrapper for many searches in Tetrad that simplifies calling various search algorithms. It allows one to set scoring algorithms, parameters, and run searches on datasets. TetradSearch calls code from the java A full list of the available search algorithms, scoring algorithms, and parameters can be found in the [official Tetrad manual](https://htmlpreview.github.io/?https:///github.com/cmu-phil/tetrad/blob/development/tetrad-lib/src/main/resources/docs/manual/index.html#knowledge_box).

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
