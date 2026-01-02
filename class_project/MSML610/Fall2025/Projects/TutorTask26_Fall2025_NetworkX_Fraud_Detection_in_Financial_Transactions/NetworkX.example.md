# Table of Contents

### Project Overview
- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
- [Project Workflow](#project-workflow)

### Graph Construction
- [Function: `create_transaction_graph(df, directed=True)`](#function-create_transaction_graphdf-directedtrue)

### Network Analysis
- [Function: `analyze_network_structure(G)`](#function-analyze_network_structureg)
- [Function: `compute_degree_centrality(G)`](#function-compute_degree_centralityg)
- [Function: `compute_betweenness_centrality(G)`](#function-compute_betweenness_centralityg)
- [Function: `compute_pagerank(G)`](#function-compute_pagerankg)
- [Function: `get_top_central_nodes(G, metric, top_n)`](#function-get_top_central_nodesg-metric-top_n)

### Fraud Detection
- [Function: `find_fraud_nodes(G)`](#function-find_fraud_nodesg)
- [Function: `find_hub_accounts(G, threshold_percentile)`](#function-find_hub_accountsg-threshold_percentile)
- [Function: `find_intermediary_accounts(G, threshold_percentile)`](#function-find_intermediary_accountsg-threshold_percentile)
- [Function: `detect_fraud_rings(G, min_size)`](#function-detect_fraud_ringsg-min_size)

### Risk Scoring
- [Function: `compute_fraud_risk_score(G, node)`](#function-compute_fraud_risk_scoreg-node)
- [Function: `rank_nodes_by_fraud_risk(G, top_n)`](#function-rank_nodes_by_fraud_riskg-top_n)

### Visualization
- [Function: `visualize_network(G, highlight_fraud)`](#function-visualize_networkg-highlight_fraud)
- [Function: `visualize_subgraph(G, nodes, title)`](#function-visualize_subgraphg-nodes-title)
- [Function: `plot_degree_distribution(G)`](#function-plot_degree_distributiong)

-------------------------------------------------------------------------------------------------------------------------------------

# Introduction

## Project Overview

This project demonstrates how to detect fraudulent transactions in financial networks using **NetworkX graph analysis**. Traditional fraud detection examines transactions in isolation, but fraudsters often work in coordinated groups. By modeling transactions as a network graph, we can reveal hidden relationships and patterns that expose coordinated fraud schemes.

**Time Required:** 60 minutes

**Primary Tool:** NetworkX (graph analysis library)

**Key Insight:** Relationships between accounts are harder to disguise than individual transactions.

---

## Dataset Description

### PaySim Synthetic Financial Dataset

PaySim is a synthetic dataset that simulates mobile money transactions based on real financial logs from an African mobile money service. It contains realistic fraud patterns including account takeovers, money mules, and coordinated fraud rings.

**Key Columns:**
- `step`: Time unit (1 step = 1 hour of simulation time)
- `type`: Transaction type (TRANSFER, PAYMENT, CASH_OUT, DEBIT, CASH_IN)
- `amount`: Transaction amount in local currency
- `nameOrig`: Customer who initiated the transaction
- `oldbalanceOrg`: Origin account balance before transaction
- `newbalanceOrig`: Origin account balance after transaction
- `nameDest`: Destination customer/merchant
- `oldbalanceDest`: Destination balance before transaction
- `newbalanceDest`: Destination balance after transaction
- `isFraud`: Ground truth fraud label (1=fraudulent, 0=legitimate)
- `isFlaggedFraud`: Business rule flag for transfers >200,000

**Dataset Characteristics:**
- Highly imbalanced (~0.1-5% fraud rate)
- Multiple transaction types
- Fraud concentrated in TRANSFER and CASH_OUT types
- Coordinated behavior between fraudulent accounts

**Download:** [PaySim Dataset on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)

---

## Project Workflow

### Phase 1: Graph Construction
Convert transaction DataFrame into NetworkX directed graph where:
- **Nodes** = User accounts
- **Edges** = Transactions (with attributes: amount, type, fraud label)

### Phase 2: Network Analysis
Compute network properties and centrality measures:
- Degree centrality (hub accounts)
- Betweenness centrality (intermediaries)
- PageRank (important receivers)
- Graph density and clustering

### Phase 3: Fraud Detection
Identify suspicious patterns using network analysis:
- Nodes involved in fraudulent transactions
- Hub accounts with unusual connectivity
- Intermediary accounts bridging network parts
- Fraud rings (dense communities)

### Phase 4: Risk Scoring
Rank accounts by fraud risk using network-based features:
- Connectivity patterns
- Fraud involvement ratio
- Network position

### Phase 5: Visualization
Create interpretable visualizations:
- Full network with fraud highlighting
- Fraud subnetworks
- Comparison of fraud vs normal patterns

-------------------------------------------------------------------------------------------------------------------------------------

# Graph Construction

## Function: `create_transaction_graph(df, directed=True)`

### Purpose

Converts a pandas DataFrame of financial transactions into a NetworkX graph representation. This is the foundational step that transforms tabular transaction data into a network structure suitable for graph analysis.

---

### Arguments

- `df (pd.DataFrame)`: Transaction data containing columns:
  - `nameOrig`: Source account
  - `nameDest`: Destination account
  - `amount`: Transaction amount
  - `isFraud`: Fraud label (0 or 1)
  - `type`: Transaction type (TRANSFER, PAYMENT, etc.)
- `directed (bool)`: If True, creates a directed graph (default: True)

---

### Design Decisions

- **Directed Graph**: We use `nx.DiGraph()` because financial transactions have clear direction (sender → receiver). This allows us to distinguish between incoming and outgoing transaction patterns.
- **Edge Attributes**: Each transaction becomes an edge with attributes (amount, type, fraud label) preserved for later analysis.
- **Multiple Edges**: If the same two users transact multiple times, each transaction creates a separate edge in the multigraph structure.

---

### Example Usage

```python
from networkx_utils import create_transaction_graph
import pandas as pd

# Load transaction data
df = pd.read_csv('transactions.csv')

# Create directed graph
G = create_transaction_graph(df, directed=True)

print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
```

### Output

A NetworkX DiGraph object with:
- Nodes representing user accounts
- Directed edges representing transactions
- Edge attributes: `amount`, `fraud`, `type`

### Sample Output
```
Created graph: 200 nodes, 5000 edges
```

### Use Case

This graph serves as the foundation for all subsequent network analysis, enabling:
- Centrality computation
- Community detection
- Fraud pattern identification
- Network visualization

-------------------------------------------------------------------------------------------------------------------------------------

# Network Analysis

## Function: `analyze_network_structure(G)`

### Purpose

Computes comprehensive network statistics to understand the overall structure and connectivity of the transaction network. This provides a high-level overview before diving into detailed fraud detection.

---

### Arguments

- `G (nx.Graph or nx.DiGraph)`: NetworkX graph representing the transaction network

---

### Design Decisions

- Provides multiple complementary metrics:
  - **Size metrics**: Number of nodes and edges
  - **Connectivity**: Density, average degree, component structure
  - **Validation**: Checks if network is connected
- Returns structured dictionary for easy inspection and logging
- Handles both directed and undirected graphs appropriately

---

### Example Usage

```python
from networkx_utils import analyze_network_structure

stats = analyze_network_structure(G)

print("Network Statistics:")
for key, value in stats.items():
    print(f"  {key}: {value}")
```

### Output

Dictionary containing network statistics:
```python
{
    'nodes': 200,
    'edges': 5000,
    'density': 0.0025,
    'avg_degree': 50.0,
    'is_connected': False,
    'num_components': 15
}
```

### Interpretation

- **Low density** (< 0.01): Sparse network, typical of financial networks
- **Multiple components**: Indicates separate transaction groups
- **Average degree**: Typical user connectivity level
- **Not fully connected**: Expected in real-world transaction networks

### Use Case

- Initial network assessment
- Quality check after graph construction
- Baseline for comparing fraud vs normal subnetworks

---

## Function: `compute_degree_centrality(G)`

### Purpose

Calculates degree centrality for all nodes in the graph. Degree centrality measures the importance of a node based on the number of connections it has, normalized by the maximum possible connections.

---

### Arguments

- `G (nx.Graph or nx.DiGraph)`: NetworkX graph

---

### Design Decisions

- Uses NetworkX's built-in `degree_centrality()` for efficiency
- Returns normalized scores (0 to 1) for fair comparison across networks
- For directed graphs, considers total degree (in + out)

---

### Example Usage

```python
from networkx_utils import compute_degree_centrality

degree_cent = compute_degree_centrality(G)

# Get top 5 nodes
top_5 = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:5]
for node, score in top_5:
    print(f"{node}: {score:.4f}")
```

### Output

Dictionary mapping each node to its degree centrality score:
```python
{
    'C00045': 0.1508,
    'C00123': 0.0954,
    'C00089': 0.0804,
    ...
}
```

### Interpretation

- **High degree centrality** (>0.1): Hub accounts with many connections
- **Low degree centrality** (<0.01): Peripheral accounts with few connections
- **Fraud signal**: Unusually high degree may indicate distribution hubs for stolen funds

### Use Case

- Identify hub accounts for investigation
- Compare centrality of fraud vs normal accounts
- Input feature for fraud risk scoring

---

## Function: `compute_betweenness_centrality(G)`

### Purpose

Calculates betweenness centrality, which measures how often a node appears on the shortest paths between other nodes. High betweenness indicates intermediary or "bridge" positions in the network.

---

### Arguments

- `G (nx.Graph or nx.DiGraph)`: NetworkX graph

---

### Design Decisions

- Uses NetworkX's `betweenness_centrality()` implementation
- Normalized scores (0 to 1) for interpretability
- **Note**: Computationally expensive for large graphs (O(n³)), so consider sampling for networks with >1000 nodes

---

### Example Usage

```python
from networkx_utils import compute_betweenness_centrality

betweenness = compute_betweenness_centrality(G)

# Find top intermediaries
top_intermediaries = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
for node, score in top_intermediaries:
    print(f"{node}: {score:.4f}")
```

### Output

Dictionary mapping nodes to betweenness centrality scores:
```python
{
    'C00234': 0.0876,
    'C00156': 0.0654,
    'C00078': 0.0432,
    ...
}
```

### Interpretation

- **High betweenness** (>0.05): Accounts that bridge different network parts
- **Low betweenness** (~0): Peripheral accounts not on shortest paths
- **Fraud signal**: Money mule accounts often have high betweenness (layering transactions)

### Use Case

- Identify intermediary accounts used for money laundering
- Detect accounts that connect otherwise separate fraud groups
- Prioritize investigation of strategic network positions

---

## Function: `compute_pagerank(G)`

### Purpose

Calculates PageRank scores for all nodes. PageRank measures importance based on the structure of incoming connections - nodes are important if they receive connections from other important nodes.

---

### Arguments

- `G (nx.DiGraph)`: NetworkX directed graph

---

### Design Decisions

- Uses Google's PageRank algorithm via NetworkX
- Default damping factor: 0.85 (standard value)
- Particularly effective for directed graphs (follows money flow)
- Iterative algorithm converges to stable scores

---

### Example Usage

```python
from networkx_utils import compute_pagerank

pagerank = compute_pagerank(G)

# Top receivers by importance
top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
for node, score in top_pagerank:
    print(f"{node}: {score:.4f}")
```

### Output

Dictionary mapping nodes to PageRank scores (sum to 1.0):
```python
{
    'C00456': 0.0234,
    'C00789': 0.0198,
    'C00123': 0.0156,
    ...
}
```

### Interpretation

- **High PageRank**: Accounts receiving funds from many active sources
- **Low PageRank**: Accounts with few or unimportant incoming connections
- **Fraud signal**: Terminal accounts in fraud schemes often have high PageRank

### Use Case

- Identify final destination accounts in money flow
- Detect accounts receiving from many fraud sources
- Complement degree centrality (which doesn't weight connection importance)

---

## Function: `get_top_central_nodes(G, metric='degree', top_n=10)`

### Purpose

Convenience function to retrieve the top N most central nodes by a specified centrality metric. Simplifies the common task of finding and ranking the most important nodes.

---

### Arguments

- `G (nx.Graph or nx.DiGraph)`: NetworkX graph
- `metric (str)`: Centrality type - 'degree', 'betweenness', or 'pagerank'
- `top_n (int)`: Number of top nodes to return (default: 10)

---

### Design Decisions

- Wraps centrality computation and sorting in one call
- Returns list of (node, score) tuples for easy iteration
- Automatically selects appropriate centrality function

---

### Example Usage

```python
from networkx_utils import get_top_central_nodes

# Get top 10 hubs by degree
top_hubs = get_top_central_nodes(G, metric='degree', top_n=10)
print("Top Hub Accounts:")
for node, score in top_hubs:
    print(f"  {node}: {score:.4f}")

# Get top 5 intermediaries
top_bridges = get_top_central_nodes(G, metric='betweenness', top_n=5)
```

### Output

List of tuples: `[(node_id, centrality_score), ...]`
```python
[
    ('C00045', 0.1508),
    ('C00123', 0.0954),
    ('C00089', 0.0804),
    ...
]
```

### Use Case

- Quick identification of key accounts
- Comparing different centrality metrics
- Generating investigation priority lists

-------------------------------------------------------------------------------------------------------------------------------------

# Fraud Detection

## Function: `find_fraud_nodes(G)`

### Purpose

Identifies all nodes (accounts) that participated in at least one fraudulent transaction. This creates a ground-truth set for validation and serves as the starting point for deeper fraud pattern analysis.

---

### Arguments

- `G (nx.Graph or nx.DiGraph)`: NetworkX graph with edge attribute `fraud`

---

### Design Decisions

- Iterates through all edges checking the `fraud` attribute
- Returns a set (not list) for efficient membership testing
- Includes both senders and receivers of fraudulent transactions
- Logs the count of fraud-involved nodes for monitoring

---

### Example Usage

```python
from networkx_utils import find_fraud_nodes

fraud_nodes = find_fraud_nodes(G)

print(f"Accounts involved in fraud: {len(fraud_nodes)}")
print(f"Fraud rate: {len(fraud_nodes) / G.number_of_nodes() * 100:.2f}%")

# Check if specific account is fraudulent
if 'C00123' in fraud_nodes:
    print("C00123 is involved in fraud!")
```

### Output

Set of node IDs involved in fraudulent transactions:
```python
{'C00045', 'C00123', 'C00234', 'C00456', ...}
```

### Sample Output
```
Found 45 nodes involved in fraud
```

### Use Case

- Validation: Check if detected suspicious accounts match known fraudsters
- Precision/Recall calculation for fraud detection methods
- Ground truth for training or evaluating fraud models
- Subgraph extraction for fraud network analysis

---

## Function: `find_hub_accounts(G, threshold_percentile=90)`

### Purpose

Identifies accounts with unusually high connectivity (degree) that may represent distribution hubs in fraud schemes. Hub accounts often receive stolen funds and distribute them to multiple accounts to obscure the money trail.

---

### Arguments

- `G (nx.Graph or nx.DiGraph)`: NetworkX graph
- `threshold_percentile (int)`: Percentile cutoff for flagging (default: 90)

---

### Design Decisions

- Uses statistical thresholds rather than absolute values (adapts to network scale)
- Default 90th percentile flags top 10% most connected accounts
- Returns list of (node, degree) tuples for inspection
- Logs count of detected hubs

---

### Example Usage

```python
from networkx_utils import find_hub_accounts

# Find top 10% most connected accounts
hubs = find_hub_accounts(G, threshold_percentile=90)

print(f"Detected {len(hubs)} hub accounts")
print("\nTop 5 Hubs:")
for node, degree in hubs[:5]:
    print(f"  {node}: {degree} connections")
```

### Output

List of (node, degree) tuples above threshold:
```python
[
    ('C00045', 75),
    ('C00123', 68),
    ('C00234', 62),
    ...
]
```

### Sample Output
```
Found 20 hub accounts (>90th percentile)
```

### Interpretation

- **Legitimate hubs**: High-volume traders, payment processors
- **Fraudulent hubs**: Distribution accounts spreading stolen funds
- **Red flag**: Hub + recent account creation + high-value transactions

### Use Case

- Prioritize investigation of highly connected accounts
- Detect distribution patterns in fraud schemes
- Monitor for sudden connectivity spikes (account takeover signal)

---

## Function: `find_intermediary_accounts(G, threshold_percentile=90)`

### Purpose

Identifies accounts with high betweenness centrality that serve as bridges or intermediaries in the transaction network. These accounts are often money mules used to layer transactions and obscure the origin of stolen funds.

---

### Arguments

- `G (nx.Graph or nx.DiGraph)`: NetworkX graph
- `threshold_percentile (int)`: Percentile cutoff (default: 90)

---

### Design Decisions

- Focuses on betweenness centrality (measures bridging, not just volume)
- Statistical threshold adapts to network structure
- Complementary to hub detection (different fraud roles)

---

### Example Usage

```python
from networkx_utils import find_intermediary_accounts

intermediaries = find_intermediary_accounts(G, threshold_percentile=90)

print(f"Detected {len(intermediaries)} intermediary accounts")
print("\nTop 5 Intermediaries:")
for node, score in intermediaries[:5]:
    print(f"  {node}: Betweenness = {score:.4f}")
```

### Output

List of (node, betweenness_score) tuples:
```python
[
    ('C00234', 0.0876),
    ('C00156', 0.0654),
    ('C00078', 0.0432),
    ...
]
```

### Sample Output
```
Found 18 intermediary accounts
```

### Interpretation

- **High betweenness**: Account sits on many shortest paths between others
- **Fraud pattern**: Money mules receive funds then forward to other accounts
- **Layering**: Multiple intermediaries create complex trails

### Use Case

- Identify money mule accounts
- Trace layered transaction chains
- Detect sophisticated laundering schemes
- Prioritize investigation of bridge accounts

---

## Function: `detect_fraud_rings(G, min_size=3)`

### Purpose

Detects potential fraud rings by identifying tightly-connected communities in the transaction network. Fraud rings are groups of accounts that frequently transact with each other, often showing coordinated behavior.

---

### Arguments

- `G (nx.Graph or nx.DiGraph)`: NetworkX graph
- `min_size (int)`: Minimum community size to flag (default: 3)

---

### Design Decisions

- Converts to undirected graph for community detection (relationships matter, not direction)
- Uses connected components to find isolated groups
- Calculates density for each component (tight groups = higher density)
- Flags communities with:
  - High density (>0.3)
  - Any fraudulent transactions
  - Minimum size threshold met

---

### Example Usage

```python
from networkx_utils import detect_fraud_rings

rings = detect_fraud_rings(G, min_size=3)

print(f"Detected {len(rings)} suspicious communities\n")

for i, ring in enumerate(rings[:3], 1):
    print(f"Ring {i}:")
    print(f"  Size: {ring['size']} members")
    print(f"  Density: {ring['density']:.3f}")
    print(f"  Contains fraud: {ring['has_fraud']}")
    print(f"  Members: {list(ring['nodes'])[:5]}")
    print()
```

### Output

List of dictionaries, each representing a suspicious community:
```python
[
    {
        'nodes': {'C00123', 'C00234', 'C00345', 'C00456'},
        'size': 4,
        'density': 0.667,
        'has_fraud': True
    },
    ...
]
```

### Sample Output
```
Found 8 suspicious communities
```

### Interpretation

- **High density** (>0.5): Members heavily interconnected (everyone transacts with everyone)
- **Isolated**: Community doesn't connect to main network
- **Fraud involvement**: At least one fraudulent transaction within ring
- **Pattern**: Coordinated accounts working together

### Use Case

- Detect coordinated fraud operations
- Identify all members of fraud rings for investigation
- Visualize fraud network structure
- Prevent future fraud by monitoring ring members

-------------------------------------------------------------------------------------------------------------------------------------

# Risk Scoring

## Function: `compute_fraud_risk_score(G, node)`

### Purpose

Computes a fraud risk score for a single node based on its network position and fraud involvement. This score combines multiple signals into a single metric for prioritizing investigations.

---

### Arguments

- `G (nx.Graph or nx.DiGraph)`: NetworkX graph
- `node (str)`: Node ID to score

---

### Design Decisions

- **Scoring formula**: Combines two components
  - 40% weight: Degree centrality (high connectivity is suspicious)
  - 60% weight: Fraud involvement ratio (edges with fraud / total edges)
- Returns score between 0 and 1 (higher = riskier)
- Returns 0 if node doesn't exist in graph

---

### Example Usage

```python
from networkx_utils import compute_fraud_risk_score

# Score a specific account
risk = compute_fraud_risk_score(G, 'C00123')
print(f"Fraud risk for C00123: {risk:.3f}")

# Score multiple accounts
accounts = ['C00123', 'C00234', 'C00345']
for account in accounts:
    risk = compute_fraud_risk_score(G, account)
    print(f"{account}: {risk:.3f}")
```

### Output

Float between 0 and 1:
```python
0.734  # High risk
0.156  # Low risk
0.000  # No risk (node not in graph)
```

### Interpretation

- **0.0-0.3**: Low risk (normal account behavior)
- **0.3-0.6**: Medium risk (investigate if other red flags present)
- **0.6-1.0**: High risk (priority investigation)

### Scoring Breakdown

For node 'C00123' with:
- Degree centrality: 0.15
- 8 fraud edges out of 10 total edges

Score = 0.4 × 0.15 + 0.6 × (8/10) = 0.06 + 0.48 = **0.54**

### Use Case

- Prioritize investigation resources
- Real-time risk assessment for new transactions
- Threshold-based alerting (e.g., flag if score > 0.6)
- Complement rule-based fraud detection

---

## Function: `rank_nodes_by_fraud_risk(G, top_n=20)`

### Purpose

Ranks all nodes in the graph by fraud risk score and returns the top N riskiest accounts. This creates an investigation priority list for fraud analysts.

---

### Arguments

- `G (nx.Graph or nx.DiGraph)`: NetworkX graph
- `top_n (int)`: Number of top risky nodes to return (default: 20)

---

### Design Decisions

- Computes risk score for every node using `compute_fraud_risk_score()`
- Sorts in descending order (highest risk first)
- Returns manageable list (default 20) for actionable investigation
- Logs completion for monitoring

---

### Example Usage

```python
from networkx_utils import rank_nodes_by_fraud_risk

# Get top 20 riskiest accounts
risky_accounts = rank_nodes_by_fraud_risk(G, top_n=20)

print("Top 10 Risky Accounts:")
print("="*50)
for i, (node, score) in enumerate(risky_accounts[:10], 1):
    print(f"{i:2d}. {node}: Risk Score = {score:.3f}")
```

### Output

List of (node, risk_score) tuples sorted by risk:
```python
[
    ('C00234', 0.873),
    ('C00456', 0.791),
    ('C00123', 0.754),
    ...
]
```

### Sample Output
```
Ranked top 20 nodes by fraud risk

Top 10 Risky Accounts:
==================================================
 1. C00234: Risk Score = 0.873
 2. C00456: Risk Score = 0.791
 3. C00123: Risk Score = 0.754
 ...
```

### Use Case

- Generate daily investigation lists
- Automated alert generation for high-risk accounts
- Performance tracking (how many top-ranked are actually fraudulent?)
- Resource allocation for fraud investigation teams

-------------------------------------------------------------------------------------------------------------------------------------

# Visualization

## Function: `visualize_network(G, highlight_fraud=True, figsize=(12, 8), layout='spring')`

### Purpose

Creates a visual representation of the transaction network with optional fraud highlighting. Visualization makes fraud patterns immediately visible and supports investigation workflows.

---

### Arguments

- `G (nx.Graph or nx.DiGraph)`: NetworkX graph to visualize
- `highlight_fraud (bool)`: If True, color fraud edges red (default: True)
- `figsize (tuple)`: Figure size in inches (default: (12, 8))
- `layout (str)`: Layout algorithm - 'spring', 'circular', or 'kamada_kawai' (default: 'spring')

---

### Design Decisions

- **Spring layout**: Force-directed positioning (most aesthetically pleasing)
- **Fraud highlighting**: Red edges for fraud, gray for normal
- **Node coloring**: Light blue for all nodes (extensible to color by risk)
- **Arrows**: Show transaction direction in directed graphs
- **Labels**: Show for small graphs (<50 nodes), hide for large graphs (readability)

---

### Example Usage

```python
from networkx_utils import visualize_network

# Basic visualization with fraud highlighting
visualize_network(G, highlight_fraud=True)

# Larger figure with circular layout
visualize_network(G, highlight_fraud=True, figsize=(16, 12), layout='circular')

# No fraud highlighting
visualize_network(G, highlight_fraud=False)
```

### Output

Matplotlib figure showing:
- Nodes as circles (user accounts)
- Edges as arrows (transactions)
- Red edges indicate fraudulent transactions
- Gray edges indicate normal transactions

### Visual Interpretation

- **Red edge clusters**: Fraud rings or fraud-heavy regions
- **Isolated red components**: Self-contained fraud operations
- **Hub nodes**: Large nodes with many connections
- **Network structure**: Reveals community structure and connectivity patterns

### Use Case

- Investigation: Visually trace fraud patterns
- Presentations: Show fraud detection results to stakeholders
- Validation: Confirm detection algorithms are working
- Pattern discovery: Identify new fraud tactics visually

---

## Function: `visualize_subgraph(G, nodes, title="Subgraph")`

### Purpose

Extracts and visualizes a subgraph containing only specified nodes and their connections. This focuses attention on specific accounts of interest without the clutter of the full network.

---

### Arguments

- `G (nx.Graph or nx.DiGraph)`: Full NetworkX graph
- `nodes (list)`: List of node IDs to include in subgraph
- `title (str)`: Plot title (default: "Subgraph")

---

### Design Decisions

- Uses `G.subgraph()` to extract induced subgraph
- Maintains fraud highlighting from full network
- Same visualization style as `visualize_network()` for consistency
- Automatically sized for readability

---

### Example Usage

```python
from networkx_utils import visualize_subgraph

# Visualize specific fraud ring
fraud_ring_members = ['C00123', 'C00234', 'C00345', 'C00456']
visualize_subgraph(G, fraud_ring_members, title="Detected Fraud Ring")

# Visualize top risky accounts and their connections
risky_accounts = [node for node, _ in rank_nodes_by_fraud_risk(G, top_n=10)]
visualize_subgraph(G, risky_accounts, title="Top 10 Risky Accounts")
```

### Output

Matplotlib figure showing only the specified subgraph with fraud highlighting and custom title.

### Use Case

- Investigate specific fraud rings in detail
- Present case evidence (focus on relevant accounts)
- Compare multiple suspicious communities side-by-side
- Drill down from network overview to specific patterns

---

## Function: `plot_degree_distribution(G)`

### Purpose

Plots the degree distribution of the network as a histogram. The degree distribution reveals the network's structural properties and helps identify unusual connectivity patterns.

---

### Arguments

- `G (nx.Graph or nx.DiGraph)`: NetworkX graph

---

### Design Decisions

- Logarithmic y-axis to handle power-law distributions
- 30 bins for good granularity
- Clean styling with grid for readability

---

### Example Usage

```python
from networkx_utils import plot_degree_distribution

# Plot degree distribution
plot_degree_distribution(G)
```

### Output

Histogram showing:
- X-axis: Node degree (number of connections)
- Y-axis: Frequency (log scale)
- Typical pattern: Power law (few hubs, many low-degree nodes)

### Interpretation

- **Power-law distribution**: Few nodes with many connections, most with few
- **Outliers**: Extremely high-degree nodes warrant investigation
- **Comparison**: Fraud vs normal networks often have different distributions

### Use Case

- Network characterization
- Anomaly detection (nodes far from typical degree)
- Validation of fraud detection (do detected fraudsters have unusual degrees?)
- Academic: Understand network topology

