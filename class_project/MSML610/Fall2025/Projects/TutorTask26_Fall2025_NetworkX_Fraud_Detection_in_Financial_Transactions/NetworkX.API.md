# Table of Contents
- [NetworkX Graph Analysis Demonstrations](#networkx-graph-analysis-demonstrations)
    - [Function: `nx.DiGraph()`](#function-nxdigraph)
    - [Function: `nx.Graph()`](#function-nxgraph)
    - [Function: `nx.degree_centrality()`](#function-nxdegree_centrality)
    - [Function: `nx.betweenness_centrality()`](#function-nxbetweenness_centrality)
    - [Function: `nx.pagerank()`](#function-nxpagerank)
    - [Function: `nx.clustering()`](#function-nxclustering)
    - [Function: `nx.density()`](#function-nxdensity)
    - [Function: `nx.connected_components()`](#function-nxconnected_components)
- [Wrapper Functions for Fraud Detection](#wrapper-functions-for-fraud-detection)
    - [Wrapper Function: `create_transaction_graph()`](#wrapper-function-create_transaction_graph)
    - [Wrapper Function: `find_fraud_nodes()`](#wrapper-function-find_fraud_nodes)
    - [Wrapper Function: `detect_fraud_rings()`](#wrapper-function-detect_fraud_rings)
    - [Wrapper Function: `find_hub_accounts()`](#wrapper-function-find_hub_accounts)

# NetworkX Graph Analysis Demonstrations

This documentation provides detailed examples and explanations for core functionalities in NetworkX: graph creation, centrality measures, network properties, and fraud detection wrappers.

---

## Function: `nx.DiGraph()`

### Purpose
Creates a directed graph data structure where edges have direction (A→B is different from B→A).

### Software Layer
Data Structure Layer - provides the foundation for representing networks with directional relationships.

### Arguments
- No required arguments for basic initialization
- Optional: `incoming_graph_data` - can initialize from edge list or other graph formats

### Example Usage
```python
import networkx as nx

# Create empty directed graph
G = nx.DiGraph()

# Add nodes
G.add_node("Alice")
G.add_node("Bob")

# Add directed edge
G.add_edge("Alice", "Bob", amount=100)
```

### Output
Empty directed graph object ready for adding nodes and edges.

### Use Case
Essential for modeling financial transactions where money flows in one direction (sender → receiver).

---

## Function: `nx.Graph()`

### Purpose
Creates an undirected graph where edges are bidirectional (A—B means A connects to B and B connects to A).

### Software Layer
Data Structure Layer - represents symmetric relationships.

### Arguments
- No required arguments for initialization

### Example Usage
```python
import networkx as nx

# Create undirected graph
G = nx.Graph()
G.add_edge("Alice", "Bob")
```

### Output
Empty undirected graph object.

### Use Case
Used for symmetric relationships like friendships or mutual connections.

---

## Function: `nx.degree_centrality()`

### Purpose
Computes degree centrality for all nodes, measuring importance based on the number of connections (normalized).

### Software Layer
Analysis Layer - computes network importance metrics.

### Arguments
- `G`: NetworkX graph object

### Example Usage
```python
import networkx as nx

G = nx.DiGraph()
G.add_edges_from([("A", "B"), ("A", "C"), ("B", "C")])

degree_cent = nx.degree_centrality(G)
print(degree_cent)
# {'A': 1.0, 'B': 0.5, 'C': 0.5}
```

### Output
Dictionary mapping nodes to degree centrality scores (0 to 1).

### Use Case
Identifies hub accounts in transaction networks - accounts with many connections are potential distribution points for stolen funds.

---

## Function: `nx.betweenness_centrality()`

### Purpose
Measures how often a node appears on the shortest paths between other nodes - identifies bridge or intermediary positions.

### Software Layer
Analysis Layer - computes path-based importance.

### Arguments
- `G`: NetworkX graph object
- `k` (optional): Number of nodes to sample for approximation

### Example Usage
```python
betweenness = nx.betweenness_centrality(G)
# Returns: {'A': 0.0, 'B': 0.333, 'C': 0.0}
```

### Output
Dictionary mapping nodes to betweenness centrality scores.

### Use Case
Identifies money mule accounts that act as intermediaries to layer transactions and hide money trails.

---

## Function: `nx.pagerank()`

### Purpose
Implements Google's PageRank algorithm - ranks nodes based on importance of their connections (important if connected to other important nodes).

### Software Layer
Analysis Layer - iterative importance ranking.

### Arguments
- `G`: NetworkX graph object
- `alpha` (optional): Damping factor (default: 0.85)
- `max_iter` (optional): Maximum iterations

### Example Usage
```python
pagerank = nx.pagerank(G)
# Returns: {'A': 0.373, 'B': 0.342, 'C': 0.285}
```

### Output
Dictionary mapping nodes to PageRank scores (sum to 1.0).

### Use Case
Identifies terminal accounts in fraud schemes - accounts receiving funds from many sources have high PageRank.

---

## Function: `nx.clustering()`

### Purpose
Computes clustering coefficient measuring how interconnected a node's neighbors are (tendency to form triangles).

### Software Layer
Property Analysis Layer - measures local connectivity patterns.

### Arguments
- `G`: NetworkX graph object
- `nodes` (optional): Specific nodes to compute

### Example Usage
```python
clustering = nx.clustering(G)
avg_clustering = nx.average_clustering(G)
```

### Output
Dictionary mapping nodes to clustering coefficients (0 to 1).

### Use Case
Fraud rings show high clustering - members frequently transact with each other forming tight-knit groups.

---

## Function: `nx.density()`

### Purpose
Calculates graph density - ratio of actual edges to possible edges.

### Software Layer
Property Analysis Layer - computes global connectivity.

### Arguments
- `G`: NetworkX graph object

### Example Usage
```python
density = nx.density(G)
# Returns: 0.667 (for 3 nodes with 2 edges)
```

### Output
Float between 0 and 1.

### Use Case
Fraud networks are typically much denser than normal transaction networks - fraud rings have density >0.5 while normal networks have <0.01.

---

## Function: `nx.connected_components()`

### Purpose
Finds separate connected subgraphs - groups of nodes that can reach each other but are isolated from other groups.

### Software Layer
Component Analysis Layer - identifies network structure.

### Arguments
- `G`: NetworkX undirected graph

### Example Usage
```python
components = list(nx.connected_components(G))
# Returns: [{'A', 'B', 'C'}, {'D', 'E'}]
```

### Output
Generator of sets, each containing nodes in one component.

### Use Case
Isolated components may represent separate fraud operations - helps identify distinct fraud rings.

---

# Wrapper Functions for Fraud Detection
Wrapper functions are basically the native functions only, we just have put few conditional statements to handle the edge cases and exceptions. All functions skeleton can be found in utils.py.

## Wrapper Function: `create_transaction_graph()`

### Software Layer
- Native API Used: NetworkX `DiGraph()`, `add_edge()`
- This wrapper converts pandas DataFrames into NetworkX graphs specifically for transaction analysis.

### Purpose
Abstracts the process of converting tabular transaction data into a graph representation suitable for network analysis and fraud detection.

### Function Definition

```python
def create_transaction_graph(df, directed=True):
    """
    Creates a NetworkX graph from transaction DataFrame.
    
    Args:
        df (pd.DataFrame): Transaction data with 'nameOrig', 'nameDest', 'amount', 'isFraud'
        directed (bool): Whether to create directed graph (default: True)
    
    Returns:
        nx.DiGraph or nx.Graph: Transaction network
    """
```

### Design Decisions

- Uses `nx.DiGraph()` by default because financial transactions have clear direction
- Preserves transaction attributes (amount, type, fraud label) as edge attributes
- Logs graph statistics for monitoring
- Handles multiple transactions between same users

### Example Usage

```python
from utils import create_transaction_graph
import pandas as pd

df = pd.read_csv("transactions.csv")
G = create_transaction_graph(df, directed=True)
print(f"Created graph with {G.number_of_nodes()} nodes")
```

---

## Wrapper Function: `find_fraud_nodes()`

### Software Layer
- Native API Used: `G.edges(data=True)` to iterate through edges
- Wrapper identifies all nodes involved in fraudulent transactions

### Purpose
Quickly identifies the set of all accounts that participated in at least one fraudulent transaction, providing ground truth for validation.

### Function Definition

```python
def find_fraud_nodes(G):
    """
    Identifies all nodes involved in fraudulent transactions.
    
    Returns:
        set: Nodes that participated in fraud
    """
```

### Design Decisions

- Returns a set for O(1) membership testing
- Includes both senders and receivers of fraud transactions
- Uses edge attribute `fraud` to check transaction labels
- Logs count for monitoring

### Example Usage

```python
from utils import find_fraud_nodes

fraud_nodes = find_fraud_nodes(G)
print(f"Found {len(fraud_nodes)} fraudulent accounts")

if 'C00123' in fraud_nodes:
    print("C00123 is involved in fraud")
```

---

## Wrapper Function: `detect_fraud_rings()`

### Software Layer
- Native API Used: `nx.connected_components()` for community detection, `nx.density()` for tightness measurement
- Combines multiple NetworkX functions to identify suspicious communities

### Purpose
Detects potential fraud rings by finding tightly-connected communities with high internal transaction density and fraud involvement.

### Function Definition

```python
def detect_fraud_rings(G, min_size=3):
    """
    Detects potential fraud rings (dense communities).
    
    Args:
        min_size (int): Minimum community size
    
    Returns:
        list: Suspicious communities with metadata
    """
```

### Design Decisions

- Converts to undirected graph for community detection (relationships matter, not direction)
- Uses connected components to find isolated groups
- Calculates density for each component
- Flags communities with density >0.3 or containing fraud
- Requires minimum size threshold (default 3 members)

### Example Usage

```python
from utils import detect_fraud_rings

rings = detect_fraud_rings(G, min_size=3)
for ring in rings:
    print(f"Ring: {ring['size']} members, Density: {ring['density']:.2f}")
```

---

## Wrapper Function: `find_hub_accounts()`

### Software Layer
- Native API Used: `G.degree()` to get node connectivity
- Statistical analysis wrapper over NetworkX degree function

### Purpose
Identifies accounts with unusually high connectivity using statistical thresholds - potential distribution hubs in fraud schemes.

### Function Definition

```python
def find_hub_accounts(G, threshold_percentile=90):
    """
    Identifies accounts with unusually high degree.
    
    Args:
        threshold_percentile (int): Percentile cutoff
    
    Returns:
        list: Hub accounts with degrees
    """
```

### Design Decisions

- Uses percentile thresholds instead of absolute values (adapts to network scale)
- Default 90th percentile flags top 10% most connected accounts
- Returns list of (node, degree) tuples for inspection
- Logs count for monitoring

### Example Usage

```python
from utils import find_hub_accounts

hubs = find_hub_accounts(G, threshold_percentile=95)
print(f"Found {len(hubs)} hub accounts")
for node, degree in hubs[:5]:
    print(f"  {node}: {degree} connections")
```

---

## Summary

**Native NetworkX provides:**
- Graph data structures (DiGraph, Graph)
- Centrality algorithms (degree, betweenness, PageRank)
- Network property calculations (density, clustering, components)
- Visualization capabilities

**My wrappers provide:**
- Easy DataFrame-to-graph conversion
- Fraud-specific detection functions
- Statistical threshold-based anomaly detection
- Pre-configured workflows for common fraud analysis tasks