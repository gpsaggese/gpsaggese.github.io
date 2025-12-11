# causal-learn API Tutorial

This document provides a comprehensive guide to the causal-learn library API, focusing on its core components for causal discovery and causal effect estimation.

<!-- toc -->

- [Introduction](#introduction)
- [Installation](#installation)
- [Core Components](#core-components)
  * [Causal Discovery Algorithms](#causal-discovery-algorithms)
  * [Causal Effect Estimation](#causal-effect-estimation)
  * [Visualization](#visualization)
- [API Reference](#api-reference)
  * [PC Algorithm](#pc-algorithm)
  * [GES Algorithm](#ges-algorithm)
  * [FCI Algorithm](#fci-algorithm)
  * [Structural Equation Modeling](#structural-equation-modeling)
- [Wrapper Functions](#wrapper-functions)
- [Examples](#examples)

<!-- tocstop -->

## Introduction

causal-learn is a Python library designed for causal inference and discovery. It provides algorithms for:
- **Causal Discovery**: Identifying causal relationships from observational data
- **Causal Effect Estimation**: Quantifying the magnitude of causal effects
- **Graphical Models**: Working with Directed Acyclic Graphs (DAGs)
- **Structural Equation Modeling**: Estimating causal effects using SEM

This tutorial covers the native API of causal-learn and the lightweight wrapper layer built on top of it for this project.

## Installation

```bash
pip install causal-learn
```

Additional dependencies:
```bash
pip install numpy pandas matplotlib networkx scikit-learn tensorflow keras
```

## Core Components

### Causal Discovery Algorithms

causal-learn provides several algorithms for causal discovery:

1. **PC Algorithm** - Constraint-based causal discovery
2. **GES Algorithm** - Score-based causal discovery
3. **FCI Algorithm** - Handles latent confounders

### Causal Effect Estimation

Methods for estimating causal effects:
- Regression adjustment
- Structural Equation Modeling (SEM)
- Instrumental variables
- Matching methods

### Visualization

Tools for visualizing causal graphs:
- NetworkX integration for DAG visualization
- Matplotlib-based graph plotting
- Interactive visualizations

## API Reference

### PC Algorithm

The PC algorithm is a constraint-based method for causal discovery.

**Native API:**
```python
from causallearn.search.ConstraintBased.PC import pc

# Run PC algorithm
cg = pc(data, alpha=0.05, indep_test='fisherz')
```

**Parameters:**
- `data`: numpy array or pandas DataFrame
- `alpha`: Significance level for independence tests
- `indep_test`: Independence test method ('fisherz', 'chisq', 'gsq', etc.)

**Returns:**
- `cg`: CausalGraph object representing the discovered causal structure

### GES Algorithm

The GES algorithm uses score-based search for causal discovery.

**Native API:**
```python
from causallearn.search.ScoreBased.GES import ges

# Run GES algorithm
cg = ges(data, score_func='local_score_BIC')
```

**Parameters:**
- `data`: numpy array or pandas DataFrame
- `score_func`: Scoring function ('local_score_BIC', 'local_score_BDeu', etc.)

**Returns:**
- `cg`: CausalGraph object

### FCI Algorithm

The FCI algorithm extends PC to handle latent confounders.

**Native API:**
```python
from causallearn.search.ConstraintBased.FCI import fci

# Run FCI algorithm
cg = fci(data, alpha=0.05, indep_test='fisherz')
```

**Parameters:**
- `data`: numpy array or pandas DataFrame
- `alpha`: Significance level
- `indep_test`: Independence test method

**Returns:**
- `cg`: CausalGraph object (may include bidirected edges for latent confounders)

### Structural Equation Modeling

SEM is used to estimate causal effects from a known or discovered causal structure.

**Native API:**
```python
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.search.ConstraintBased.PC import pc

# Discover structure first
cg = pc(data, alpha=0.05)

# Then estimate effects using regression or SEM
# (Implementation details depend on specific SEM library used)
```

## Wrapper Functions

Our wrapper layer provides simplified interfaces for common causal inference tasks.

### `discover_causal_structure()`

Wrapper function for causal discovery with automatic algorithm selection.

```python
from utils.utils_post_processing import discover_causal_structure

# Discover causal structure
graph, edges = discover_causal_structure(
    data=processed_df,
    algorithm='PC',
    alpha=0.05,
    variables=['inflation', 'unemployment', 'wage_growth']
)
```

### `estimate_causal_effects()`

Wrapper function for estimating causal effects using SEM.

```python
from utils.utils_post_processing import estimate_causal_effects

# Estimate causal effects
effects = estimate_causal_effects(
    data=processed_df,
    causal_graph=graph,
    treatment='inflation',
    outcome='wage_growth',
    method='SEM'
)
```

### `visualize_causal_graph()`

Wrapper function for visualizing causal DAGs.

```python
from utils.utils_post_processing import visualize_causal_graph

# Visualize causal graph
visualize_causal_graph(
    graph=graph,
    output_path='outputs/causal_graphs/main_dag.png',
    title='Causal Structure: Economic Factors → Employment Outcomes'
)
```

## Examples

### Example 1: Basic Causal Discovery

**Note**: First download the dataset by running `python data/download_data.py`

```python
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc

# Load data (after downloading via download_data.py)
data = pd.read_csv('data/economic_data.csv', parse_dates=['date'])

# Select numeric columns for causal discovery
variables = ['unemployment_rate', 'inflation_rate', 'wage_growth', 'gdp_growth']
numeric_data = data[variables].dropna()

# Run PC algorithm
cg = pc(numeric_data.values, alpha=0.05, indep_test='fisherz')

# Get adjacency matrix
adj_matrix = cg.G.graph
print("Causal structure discovered:")
print(adj_matrix)
```

### Example 2: Causal Effect Estimation

```python
from utils.utils_post_processing import estimate_causal_effects

# Estimate effect of inflation on wage growth
effect = estimate_causal_effects(
    data=processed_df,
    causal_graph=discovered_graph,
    treatment='inflation_rate',
    outcome='wage_growth',
    method='SEM'
)

print(f"Causal effect: {effect['coefficient']:.4f}")
print(f"95% CI: [{effect['ci_lower']:.4f}, {effect['ci_upper']:.4f}]")
```

### Example 3: Temporal Causal Discovery

```python
from utils.utils_post_processing import rolling_window_causal_discovery

# Discover causal structure over rolling windows
results = rolling_window_causal_discovery(
    data=time_series_df,
    window_size=24,  # 24 months
    algorithm='PC',
    alpha=0.05
)

# Results contain causal graphs for each time window
for window, graph in results.items():
    print(f"Window {window}: {len(graph.edges)} edges discovered")
```

## Key Design Decisions

1. **Algorithm Selection**: PC algorithm is used as default for its balance of accuracy and interpretability
2. **Independence Testing**: Fisher's Z-test is preferred for continuous variables
3. **Graph Representation**: NetworkX is used for graph manipulation and visualization
4. **Effect Estimation**: SEM is preferred over simple regression for handling confounders

## Best Practices

1. **Data Preprocessing**: Ensure data is properly cleaned and time-aligned before causal discovery
2. **Significance Levels**: Use alpha=0.05 as default, but adjust based on sample size
3. **Validation**: Always validate discovered structures with domain knowledge
4. **Visualization**: Visualize causal graphs to understand relationships
5. **Sensitivity Analysis**: Test robustness of results to different parameters

## References

- [causal-learn Documentation](https://causal-learn.readthedocs.io/)
- [PC Algorithm Paper](https://www.jmlr.org/papers/volume8/kalisch07a/kalisch07a.pdf)
- [GES Algorithm Paper](https://www.jmlr.org/papers/volume3/chickering02b/chickering02b.pdf)
- [Structural Equation Modeling](https://en.wikipedia.org/wiki/Structural_equation_modeling)

