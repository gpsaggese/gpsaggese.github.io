# -----------------------------------------------------------------------------
# Importing Modules
# -----------------------------------------------------------------------------

# Generic Imports
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Logging Imports
import logging
from loguru import logger
logging.getLogger('networkx').setLevel(logging.WARNING)

# NetworkX Imports
import networkx as nx

# Visualization Imports
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# Graph Creation
# -----------------------------------------------------------------------------

def create_transaction_graph(df, directed=True):
    """
    Creates a NetworkX graph from transaction DataFrame.
    
    Args:
        df (pd.DataFrame): Transaction data with 'nameOrig', 'nameDest', 'amount', 'isFraud', 'type'
        directed (bool): Whether to create directed graph (default: True)
    
    Returns:
        nx.DiGraph or nx.Graph: Transaction network graph
    """
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    for _, row in df.iterrows():
        G.add_edge(
            row['nameOrig'], 
            row['nameDest'],
            amount=row['amount'],
            fraud=row['isFraud'],
            type=row['type']
        )
    
    logger.info(f"Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

# -----------------------------------------------------------------------------
# Centrality Measures
# -----------------------------------------------------------------------------

def compute_degree_centrality(G):
    """
    Computes degree centrality for all nodes.
    
    Args:
        G (nx.Graph): NetworkX graph
    
    Returns:
        dict: Dictionary mapping nodes to degree centrality scores
    """
    degree_cent = nx.degree_centrality(G)
    logger.info("Degree centrality computed")
    return degree_cent

def compute_betweenness_centrality(G):
    """
    Computes betweenness centrality for all nodes.
    
    Args:
        G (nx.Graph): NetworkX graph
    
    Returns:
        dict: Dictionary mapping nodes to betweenness centrality scores
    """
    betweenness = nx.betweenness_centrality(G)
    logger.info("Betweenness centrality computed")
    return betweenness

def compute_pagerank(G):
    """
    Computes PageRank for all nodes.
    
    Args:
        G (nx.Graph): NetworkX graph
    
    Returns:
        dict: Dictionary mapping nodes to PageRank scores
    """
    pagerank = nx.pagerank(G)
    logger.info("PageRank computed")
    return pagerank

def get_top_central_nodes(G, metric='degree', top_n=10):
    """
    Returns top N nodes by centrality metric.
    
    Args:
        G (nx.Graph): NetworkX graph
        metric (str): 'degree', 'betweenness', or 'pagerank'
        top_n (int): Number of top nodes to return
    
    Returns:
        list: List of (node, score) tuples
    """
    if metric == 'degree':
        scores = nx.degree_centrality(G)
    elif metric == 'betweenness':
        scores = nx.betweenness_centrality(G)
    elif metric == 'pagerank':
        scores = nx.pagerank(G)
    else:
        logger.error(f"Unknown metric: {metric}")
        return []
    
    top_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top_nodes

# -----------------------------------------------------------------------------
# Network Properties
# -----------------------------------------------------------------------------

def compute_clustering_coefficient(G):
    """
    Computes clustering coefficient for all nodes.
    
    Args:
        G (nx.Graph): NetworkX graph
    
    Returns:
        dict: Dictionary mapping nodes to clustering coefficients
    """
    clustering = nx.clustering(G.to_undirected())
    logger.info("Clustering coefficients computed")
    return clustering

def compute_graph_density(G):
    """
    Computes overall graph density.
    
    Args:
        G (nx.Graph): NetworkX graph
    
    Returns:
        float: Graph density (0 to 1)
    """
    density = nx.density(G)
    return density

def analyze_network_structure(G):
    """
    Analyzes overall network structure and returns comprehensive statistics.
    
    Args:
        G (nx.Graph): NetworkX graph
    
    Returns:
        dict: Dictionary containing network statistics
    """
    stats = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': nx.density(G),
        'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
        'is_connected': nx.is_weakly_connected(G) if G.is_directed() else nx.is_connected(G),
        'num_components': nx.number_weakly_connected_components(G) if G.is_directed() else nx.number_connected_components(G)
    }
    
    logger.info("Network structure analyzed")
    return stats

# -----------------------------------------------------------------------------
# Fraud Detection
# -----------------------------------------------------------------------------

def find_fraud_nodes(G):
    """
    Identifies all nodes involved in fraudulent transactions.
    
    Args:
        G (nx.Graph): NetworkX graph
    
    Returns:
        set: Set of nodes involved in fraud
    """
    fraud_nodes = set()
    for u, v, data in G.edges(data=True):
        if data.get('fraud', 0) == 1:
            fraud_nodes.add(u)
            fraud_nodes.add(v)
    
    logger.info(f"Found {len(fraud_nodes)} nodes involved in fraud")
    return fraud_nodes

def find_hub_accounts(G, threshold_percentile=90):
    """
    Identifies accounts with unusually high degree (hub accounts).
    
    Args:
        G (nx.Graph): NetworkX graph
        threshold_percentile (int): Percentile threshold for flagging
    
    Returns:
        list: List of (node, degree) tuples for hub accounts
    """
    degrees = dict(G.degree())
    threshold = np.percentile(list(degrees.values()), threshold_percentile)
    
    hubs = [(node, deg) for node, deg in degrees.items() if deg > threshold]
    logger.info(f"Found {len(hubs)} hub accounts (>{threshold_percentile}th percentile)")
    return hubs

def find_intermediary_accounts(G, threshold_percentile=90):
    """
    Identifies intermediary accounts with high betweenness centrality.
    
    Args:
        G (nx.Graph): NetworkX graph
        threshold_percentile (int): Percentile threshold
    
    Returns:
        list: List of (node, betweenness_score) tuples
    """
    betweenness = nx.betweenness_centrality(G)
    threshold = np.percentile(list(betweenness.values()), threshold_percentile)
    
    intermediaries = [(node, score) for node, score in betweenness.items() 
                     if score > threshold]
    logger.info(f"Found {len(intermediaries)} intermediary accounts")
    return intermediaries

def detect_fraud_rings(G, min_size=3):
    """
    Detects potential fraud rings (tightly-connected communities).
    
    Args:
        G (nx.Graph): NetworkX graph
        min_size (int): Minimum community size to flag
    
    Returns:
        list: List of dictionaries containing community information
    """
    # Convert to undirected for community detection
    G_undirected = G.to_undirected()
    
    # Find connected components
    components = list(nx.connected_components(G_undirected))
    
    suspicious = []
    for component in components:
        if len(component) >= min_size:
            subgraph = G.subgraph(component)
            density = nx.density(subgraph)
            
            # Check if any edge in this component is fraud
            has_fraud = any(data.get('fraud', 0) == 1 
                          for _, _, data in subgraph.edges(data=True))
            
            if density > 0.3 or has_fraud:
                suspicious.append({
                    'nodes': component,
                    'size': len(component),
                    'density': density,
                    'has_fraud': has_fraud
                })
    
    logger.info(f"Found {len(suspicious)} suspicious communities")
    return suspicious

# -----------------------------------------------------------------------------
# Risk Scoring
# -----------------------------------------------------------------------------

def compute_fraud_risk_score(G, node):
    """
    Computes fraud risk score for a single node based on network position.
    
    Args:
        G (nx.Graph): NetworkX graph
        node (str): Node ID to score
    
    Returns:
        float: Risk score between 0 and 1
    """
    if node not in G:
        return 0
    
    # Compute degree centrality
    degree_cent = nx.degree_centrality(G).get(node, 0)
    
    # Check fraud involvement
    fraud_edges = sum(1 for _, _, d in G.edges(node, data=True) if d.get('fraud', 0) == 1)
    total_edges = G.degree(node)
    fraud_ratio = fraud_edges / total_edges if total_edges > 0 else 0
    
    # Weighted score: 40% degree centrality, 60% fraud involvement
    score = 0.4 * degree_cent + 0.6 * fraud_ratio
    
    return score

def rank_nodes_by_fraud_risk(G, top_n=20):
    """
    Ranks all nodes by fraud risk score.
    
    Args:
        G (nx.Graph): NetworkX graph
        top_n (int): Number of top risky nodes to return
    
    Returns:
        list: List of (node, risk_score) tuples sorted by risk
    """
    scores = {node: compute_fraud_risk_score(G, node) for node in G.nodes()}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    logger.info(f"Ranked top {top_n} nodes by fraud risk")
    return ranked

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def visualize_network(G, highlight_fraud=True, figsize=(12, 8), layout='spring'):
    """
    Visualizes the transaction network with optional fraud highlighting.
    
    Args:
        G (nx.Graph): NetworkX graph
        highlight_fraud (bool): Whether to highlight fraud edges in red
        figsize (tuple): Figure size
        layout (str): Layout algorithm - 'spring', 'circular', or 'kamada_kawai'
    """
    plt.figure(figsize=figsize)
    
    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42, k=0.5)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    else:
        pos = nx.kamada_kawai_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=300, alpha=0.7)
    
    # Draw edges
    if highlight_fraud:
        fraud_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('fraud', 0) == 1]
        normal_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('fraud', 0) == 0]
        
        nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color='gray', 
                              alpha=0.3, arrows=True, arrowsize=10)
        nx.draw_networkx_edges(G, pos, edgelist=fraud_edges, edge_color='red', 
                              width=2, alpha=0.8, arrows=True, arrowsize=15)
    else:
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5, arrows=True)
    
    # Draw labels for small graphs
    if G.number_of_nodes() < 50:
        nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title('Transaction Network (Red = Fraud)', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_subgraph(G, nodes, title="Subgraph"):
    """
    Visualizes a subgraph containing only specified nodes.
    
    Args:
        G (nx.Graph): NetworkX graph
        nodes (list): List of node IDs to include
        title (str): Plot title
    """
    subgraph = G.subgraph(nodes)
    visualize_network(subgraph, highlight_fraud=True, figsize=(10, 8))
    plt.title(title, fontsize=14)

def plot_degree_distribution(G):
    """
    Plots the degree distribution of the network.
    
    Args:
        G (nx.Graph): NetworkX graph
    """
    degrees = [G.degree(n) for n in G.nodes()]
    
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Degree', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Degree Distribution', fontsize=14)
    plt.yscale('log')
    plt.grid(alpha=0.3)
    plt.show()

def plot_centrality_distribution(centrality_scores, metric_name):
    """
    Plots distribution of centrality scores.
    
    Args:
        centrality_scores (dict): Dictionary of node: score mappings
        metric_name (str): Name of the centrality metric
    """
    plt.figure(figsize=(10, 6))
    plt.hist(list(centrality_scores.values()), bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel(f'{metric_name} Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'{metric_name} Distribution', fontsize=14)
    plt.grid(alpha=0.3)
    plt.show()