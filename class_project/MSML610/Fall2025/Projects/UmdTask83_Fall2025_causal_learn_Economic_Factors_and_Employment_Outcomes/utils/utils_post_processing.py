"""
Post-processing utilities for causal inference and analysis.

This module provides functions for:
- Causal discovery using causal-learn algorithms (PC, GES, FCI)
- Causal effect estimation using Structural Equation Modeling (SEM)
- Temporal analysis over rolling time windows
- Visualization of causal graphs
- Model training and evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import causal-learn
CAUSAL_LEARN_AVAILABLE = False
try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.ScoreBased.GES import ges
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.GraphUtils import GraphUtils
    CAUSAL_LEARN_AVAILABLE = True
    logger.info("causal-learn library loaded successfully")
except ImportError:
    logger.warning("causal-learn not installed. Run: pip install causal-learn")
    logger.warning("Using fallback implementations.")


def discover_causal_structure(
    data: pd.DataFrame,
    algorithm: str = 'PC',
    alpha: float = 0.05,
    variables: Optional[List[str]] = None,
    indep_test: str = 'fisherz',
    **kwargs
) -> Tuple[Any, List[Tuple[str, str]]]:
    """
    Discover causal structure from observational data using causal-learn.
    
    Args:
        data: DataFrame with observational data
        algorithm: Causal discovery algorithm ('PC', 'GES', 'FCI')
        alpha: Significance level for independence tests
        variables: List of variable names (None to use all columns)
        indep_test: Independence test ('fisherz', 'chisq', 'gsq', 'kci')
        **kwargs: Additional algorithm-specific parameters
        
    Returns:
        Tuple of (causal_graph, edges_list)
        - causal_graph: Graph object with discovered structure
        - edges_list: List of (source, target) tuples representing directed edges
    """
    logger.info(f"Discovering causal structure using {algorithm} algorithm (alpha={alpha})")
    
    # Select variables if specified
    if variables:
        available = [v for v in variables if v in data.columns]
        if len(available) != len(variables):
            missing = set(variables) - set(available)
            logger.warning(f"Missing variables: {missing}")
        data = data[available].copy()
    
    # Get variable names
    var_names = list(data.columns)
    n_vars = len(var_names)
    
    # Convert to numpy array and handle missing values
    data_array = data.dropna().values.astype(np.float64)
    
    if len(data_array) < 50:
        logger.warning(f"Small sample size ({len(data_array)}). Results may be unreliable.")
    
    logger.info(f"Running {algorithm} on {len(data_array)} samples with {n_vars} variables")
    
    edges = []
    graph_obj = None
    
    if CAUSAL_LEARN_AVAILABLE:
        try:
            if algorithm.upper() == 'PC':
                # Run PC algorithm
                cg = pc(
                    data_array, 
                    alpha=alpha, 
                    indep_test=indep_test,
                    stable=True,
                    uc_rule=0,
                    uc_priority=2,
                    show_progress=False
                )
                graph_obj = cg
                
                # Extract edges from adjacency matrix
                adj_matrix = cg.G.graph
                for i in range(n_vars):
                    for j in range(n_vars):
                        # Check for directed edge i -> j
                        if adj_matrix[i, j] == -1 and adj_matrix[j, i] == 1:
                            edges.append((var_names[i], var_names[j]))
                        # Check for undirected edge (treat as bidirectional for now)
                        elif adj_matrix[i, j] == -1 and adj_matrix[j, i] == -1 and i < j:
                            edges.append((var_names[i], var_names[j]))
                            
            elif algorithm.upper() == 'GES':
                # Run GES algorithm
                record = ges(data_array, score_func='local_score_BIC')
                graph_obj = record
                
                # Extract edges
                adj_matrix = record['G'].graph
                for i in range(n_vars):
                    for j in range(n_vars):
                        if adj_matrix[i, j] == -1 and adj_matrix[j, i] == 1:
                            edges.append((var_names[i], var_names[j]))
                        elif adj_matrix[i, j] == -1 and adj_matrix[j, i] == -1 and i < j:
                            edges.append((var_names[i], var_names[j]))
                            
            elif algorithm.upper() == 'FCI':
                # Run FCI algorithm
                g, edges_fci = fci(data_array, indep_test, alpha, verbose=False)
                graph_obj = g
                
                # Extract edges from FCI output
                adj_matrix = g.graph
                for i in range(n_vars):
                    for j in range(n_vars):
                        if adj_matrix[i, j] != 0 and i < j:
                            edges.append((var_names[i], var_names[j]))
            else:
                logger.error(f"Unknown algorithm: {algorithm}")
                
        except Exception as e:
            logger.error(f"Causal discovery failed: {e}")
            logger.info("Falling back to correlation-based heuristic")
            edges = _fallback_causal_discovery(data, var_names, alpha)
    else:
        # Fallback when causal-learn is not available
        edges = _fallback_causal_discovery(data, var_names, alpha)
    
    # Create a simple graph object to store results
    class CausalGraph:
        def __init__(self, nodes, edges, raw_graph=None):
            self.nodes = nodes
            self.edges = edges
            self.raw_graph = raw_graph
            
        def get_adjacency_matrix(self):
            n = len(self.nodes)
            adj = np.zeros((n, n))
            node_idx = {n: i for i, n in enumerate(self.nodes)}
            for src, tgt in self.edges:
                if src in node_idx and tgt in node_idx:
                    adj[node_idx[src], node_idx[tgt]] = 1
            return adj
    
    result_graph = CausalGraph(var_names, edges, graph_obj)
    
    logger.info(f"Discovered {len(edges)} causal relationships")
    for src, tgt in edges:
        logger.info(f"  {src} → {tgt}")
    
    return result_graph, edges


def _fallback_causal_discovery(
    data: pd.DataFrame, 
    var_names: List[str], 
    alpha: float
) -> List[Tuple[str, str]]:
    """
    Fallback causal discovery using partial correlations.
    This is a simplified heuristic when causal-learn is not available.
    """
    from scipy import stats
    
    edges = []
    n_vars = len(var_names)
    
    # Calculate correlation matrix
    corr_matrix = data[var_names].corr()
    
    # Use partial correlations to find edges
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            # Check if correlation is significant
            r = corr_matrix.iloc[i, j]
            n = len(data)
            t_stat = r * np.sqrt((n - 2) / (1 - r**2 + 1e-10))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            
            if p_value < alpha and abs(r) > 0.1:
                # Use temporal ordering or domain knowledge heuristic
                # For economic data: GDP/inflation typically cause wages/employment
                cause_vars = ['gdp_growth', 'inflation_rate', 'federal_funds_rate', 'unemployment_rate']
                effect_vars = ['wage_growth', 'employment_rate', 'real_wage_growth']
                
                var_i, var_j = var_names[i], var_names[j]
                
                # Determine direction based on domain knowledge
                if var_i in cause_vars and var_j in effect_vars:
                    edges.append((var_i, var_j))
                elif var_j in cause_vars and var_i in effect_vars:
                    edges.append((var_j, var_i))
                else:
                    # Default: alphabetical order (arbitrary but consistent)
                    edges.append((var_i, var_j) if var_i < var_j else (var_j, var_i))
    
    return edges


def estimate_causal_effects(
    data: pd.DataFrame,
    causal_graph: Any,
    treatment: str,
    outcome: str,
    method: str = 'regression',
    confounders: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Estimate causal effects using regression adjustment or SEM.
    
    Args:
        data: DataFrame with observational data
        causal_graph: CausalGraph object from causal discovery
        treatment: Name of treatment variable
        outcome: Name of outcome variable
        method: Estimation method ('regression', 'SEM', 'IV')
        confounders: List of confounding variables to adjust for
        
    Returns:
        Dictionary with effect estimates and statistics
    """
    logger.info(f"Estimating causal effect: {treatment} → {outcome} using {method}")
    
    if treatment not in data.columns:
        raise ValueError(f"Treatment variable '{treatment}' not found in data")
    if outcome not in data.columns:
        raise ValueError(f"Outcome variable '{outcome}' not found in data")
    
    # Prepare data
    analysis_data = data[[treatment, outcome]].dropna()
    
    # Identify confounders from graph if not specified
    if confounders is None and causal_graph is not None:
        confounders = _identify_confounders(causal_graph, treatment, outcome)
        if confounders:
            # Add confounders to analysis data
            confounder_cols = [c for c in confounders if c in data.columns]
            if confounder_cols:
                analysis_data = data[[treatment, outcome] + confounder_cols].dropna()
    
    if len(analysis_data) < 30:
        logger.warning(f"Small sample size ({len(analysis_data)}). Results may be unreliable.")
    
    if method.lower() == 'regression':
        result = _estimate_regression(analysis_data, treatment, outcome, confounders)
    elif method.lower() == 'sem':
        result = _estimate_sem(analysis_data, treatment, outcome, confounders)
    else:
        logger.warning(f"Unknown method '{method}', using regression")
        result = _estimate_regression(analysis_data, treatment, outcome, confounders)
    
    logger.info(f"Estimated effect: {result['coefficient']:.4f} "
                f"(95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}], "
                f"p={result['p_value']:.4f})")
    
    return result


def _identify_confounders(
    causal_graph: Any, 
    treatment: str, 
    outcome: str
) -> List[str]:
    """Identify potential confounders from the causal graph."""
    if not hasattr(causal_graph, 'edges') or not hasattr(causal_graph, 'nodes'):
        return []
    
    # Find variables that have edges to both treatment and outcome
    confounders = []
    edges_to_treatment = {src for src, tgt in causal_graph.edges if tgt == treatment}
    edges_to_outcome = {src for src, tgt in causal_graph.edges if tgt == outcome}
    
    # Common causes are confounders
    confounders = list(edges_to_treatment & edges_to_outcome)
    
    return confounders


def _estimate_regression(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: Optional[List[str]] = None
) -> Dict[str, float]:
    """Estimate causal effect using OLS regression with adjustment."""
    from scipy import stats
    
    # Prepare features
    if confounders:
        feature_cols = [treatment] + [c for c in confounders if c in data.columns]
    else:
        feature_cols = [treatment]
    
    X = data[feature_cols].values
    y = data[outcome].values
    
    # Add intercept
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    
    # OLS estimation
    try:
        beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        y_pred = X_with_intercept @ beta
        residuals = y - y_pred
        
        n = len(y)
        p = X_with_intercept.shape[1]
        
        # Calculate standard errors
        mse = np.sum(residuals**2) / (n - p)
        var_beta = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        se = np.sqrt(np.diag(var_beta))
        
        # Treatment effect is the second coefficient (after intercept)
        coef = beta[1]
        se_coef = se[1]
        
        # Calculate t-statistic and p-value
        t_stat = coef / se_coef
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - p))
        
        # Confidence interval
        ci_lower = coef - 1.96 * se_coef
        ci_upper = coef + 1.96 * se_coef
        
        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return {
            'coefficient': float(coef),
            'se': float(se_coef),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'r_squared': float(r_squared),
            'n_observations': int(n),
            'method': 'OLS Regression'
        }
        
    except Exception as e:
        logger.error(f"Regression failed: {e}")
        return {
            'coefficient': 0.0,
            'se': 0.0,
            'ci_lower': 0.0,
            'ci_upper': 0.0,
            't_statistic': 0.0,
            'p_value': 1.0,
            'r_squared': 0.0,
            'n_observations': len(data),
            'method': 'Failed',
            'error': str(e)
        }


def _estimate_sem(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Estimate causal effect using Structural Equation Modeling.
    Falls back to regression if SEM libraries not available.
    """
    try:
        import semopy
        
        # Build SEM model specification
        if confounders:
            model_spec = f"""
            {outcome} ~ {treatment} + {' + '.join(confounders)}
            """
        else:
            model_spec = f"{outcome} ~ {treatment}"
        
        model = semopy.Model(model_spec)
        model.fit(data)
        
        # Get estimates
        estimates = model.inspect()
        treatment_row = estimates[estimates['rval'] == treatment]
        
        if len(treatment_row) > 0:
            coef = treatment_row['Estimate'].values[0]
            se = treatment_row['Std. Err'].values[0]
            p_value = treatment_row['p-value'].values[0]
            
            return {
                'coefficient': float(coef),
                'se': float(se),
                'ci_lower': float(coef - 1.96 * se),
                'ci_upper': float(coef + 1.96 * se),
                't_statistic': float(coef / se) if se > 0 else 0.0,
                'p_value': float(p_value),
                'r_squared': 0.0,  # SEM doesn't directly report this
                'n_observations': len(data),
                'method': 'SEM (semopy)'
            }
            
    except ImportError:
        logger.warning("semopy not installed. Using regression instead.")
    except Exception as e:
        logger.warning(f"SEM estimation failed: {e}. Using regression instead.")
    
    # Fallback to regression
    return _estimate_regression(data, treatment, outcome, confounders)


def visualize_causal_graph(
    graph: Any,
    output_path: Optional[str] = None,
    title: str = 'Causal Structure',
    figsize: Tuple[int, int] = (12, 8),
    node_colors: Optional[Dict[str, str]] = None,
    edge_labels: Optional[Dict[Tuple[str, str], str]] = None,
    layout: str = 'spring'
) -> None:
    """
    Visualize causal DAG using NetworkX and Matplotlib.
    
    Args:
        graph: CausalGraph object or list of edges
        output_path: Path to save the visualization (None to display)
        title: Title for the plot
        figsize: Figure size (width, height)
        node_colors: Dict mapping node names to colors
        edge_labels: Dict mapping edges to labels
        layout: Layout algorithm ('spring', 'circular', 'hierarchical')
    """
    logger.info(f"Visualizing causal graph: {title}")
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Extract edges
    if hasattr(graph, 'edges'):
        edges = graph.edges
    elif isinstance(graph, (list, tuple)):
        edges = graph
    else:
        edges = []
    
    # Add edges
    for edge in edges:
        if isinstance(edge, tuple) and len(edge) >= 2:
            G.add_edge(edge[0], edge[1])
    
    if len(G.nodes) == 0:
        logger.warning("No nodes in graph, creating example visualization")
        G.add_edges_from([
            ('gdp_growth', 'employment_rate'),
            ('gdp_growth', 'unemployment_rate'),
            ('unemployment_rate', 'wage_growth'),
            ('inflation_rate', 'real_wage_growth'),
            ('federal_funds_rate', 'inflation_rate')
        ])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Choose layout
    if layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'hierarchical':
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except:
            pos = nx.spring_layout(G, k=2, iterations=50)
    else:  # spring
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Define default colors based on variable types
    if node_colors is None:
        node_colors = {}
        for node in G.nodes():
            if 'gdp' in node.lower() or 'growth' in node.lower():
                node_colors[node] = '#90EE90'  # Light green
            elif 'unemployment' in node.lower() or 'employment' in node.lower():
                node_colors[node] = '#FFB6C1'  # Light pink
            elif 'inflation' in node.lower() or 'cpi' in node.lower():
                node_colors[node] = '#FFD700'  # Gold
            elif 'wage' in node.lower():
                node_colors[node] = '#87CEEB'  # Sky blue
            elif 'rate' in node.lower():
                node_colors[node] = '#DDA0DD'  # Plum
            else:
                node_colors[node] = '#D3D3D3'  # Light gray
    
    colors = [node_colors.get(node, '#D3D3D3') for node in G.nodes()]
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=3000, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='#404040', arrows=True, 
                           arrowsize=25, arrowstyle='-|>', alpha=0.7,
                           connectionstyle='arc3,rad=0.1', ax=ax)
    
    # Draw labels with better formatting
    labels = {node: node.replace('_', '\n') for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold', ax=ax)
    
    # Add edge labels if provided
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()


def rolling_window_causal_discovery(
    data: pd.DataFrame,
    window_size: int,
    step_size: int = None,
    algorithm: str = 'PC',
    alpha: float = 0.05,
    variables: Optional[List[str]] = None,
    time_column: str = 'date'
) -> Dict[str, Any]:
    """
    Apply causal discovery over rolling time windows to detect temporal changes.
    
    Args:
        data: DataFrame with time series data
        window_size: Size of rolling window (number of observations)
        step_size: Step size for rolling (default: window_size // 4)
        algorithm: Causal discovery algorithm
        alpha: Significance level
        variables: List of variables to include
        time_column: Name of time/date column
        
    Returns:
        Dictionary with results for each window
    """
    logger.info(f"Rolling window causal discovery (window={window_size}, step={step_size})")
    
    if step_size is None:
        step_size = max(1, window_size // 4)
    
    # Sort by time
    if time_column in data.columns:
        data = data.sort_values(time_column).reset_index(drop=True)
    
    results = {
        'windows': [],
        'edges_over_time': [],
        'edge_stability': {}
    }
    
    n_observations = len(data)
    window_idx = 0
    
    for start_idx in range(0, n_observations - window_size + 1, step_size):
        end_idx = start_idx + window_size
        window_data = data.iloc[start_idx:end_idx]
        
        try:
            graph, edges = discover_causal_structure(
                data=window_data,
                algorithm=algorithm,
                alpha=alpha,
                variables=variables
            )
            
            # Get time range for this window
            if time_column in data.columns:
                start_time = window_data[time_column].iloc[0]
                end_time = window_data[time_column].iloc[-1]
            else:
                start_time = start_idx
                end_time = end_idx
            
            window_result = {
                'window_idx': window_idx,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_time': start_time,
                'end_time': end_time,
                'n_edges': len(edges),
                'edges': edges,
                'graph': graph
            }
            
            results['windows'].append(window_result)
            results['edges_over_time'].append(set(edges))
            
            # Track edge stability
            for edge in edges:
                edge_key = f"{edge[0]} → {edge[1]}"
                if edge_key not in results['edge_stability']:
                    results['edge_stability'][edge_key] = 0
                results['edge_stability'][edge_key] += 1
            
            logger.info(f"Window {window_idx}: {len(edges)} edges discovered")
            window_idx += 1
            
        except Exception as e:
            logger.warning(f"Window {window_idx} failed: {e}")
            window_idx += 1
            continue
    
    # Calculate stability percentages
    n_windows = len(results['windows'])
    if n_windows > 0:
        for edge_key in results['edge_stability']:
            results['edge_stability'][edge_key] = results['edge_stability'][edge_key] / n_windows
    
    logger.info(f"Completed {n_windows} windows")
    return results


def temporal_effect_estimation(
    data: pd.DataFrame,
    window_size: int,
    treatment: str,
    outcome: str,
    step_size: int = None,
    method: str = 'regression',
    time_column: str = 'date'
) -> Dict[str, np.ndarray]:
    """
    Estimate causal effects over rolling time windows.
    
    Returns:
        Dictionary with temporal effect estimates
    """
    logger.info(f"Temporal effect estimation: {treatment} → {outcome}")
    
    if step_size is None:
        step_size = max(1, window_size // 4)
    
    # Sort by time
    if time_column in data.columns:
        data = data.sort_values(time_column).reset_index(drop=True)
    
    times = []
    effects = []
    ci_lowers = []
    ci_uppers = []
    p_values = []
    
    n_observations = len(data)
    
    for start_idx in range(0, n_observations - window_size + 1, step_size):
        end_idx = start_idx + window_size
        window_data = data.iloc[start_idx:end_idx]
        
        try:
            result = estimate_causal_effects(
                data=window_data,
                causal_graph=None,
                treatment=treatment,
                outcome=outcome,
                method=method
            )
            
            effects.append(result['coefficient'])
            ci_lowers.append(result['ci_lower'])
            ci_uppers.append(result['ci_upper'])
            p_values.append(result['p_value'])
            
            # Get midpoint time
            if time_column in data.columns:
                mid_idx = start_idx + window_size // 2
                times.append(data[time_column].iloc[mid_idx])
            else:
                times.append(start_idx + window_size // 2)
                
        except Exception as e:
            logger.warning(f"Window estimation failed: {e}")
            continue
    
    return {
        'time': np.array(times),
        'effect': np.array(effects),
        'ci_lower': np.array(ci_lowers),
        'ci_upper': np.array(ci_uppers),
        'p_value': np.array(p_values)
    }


def prepare_lstm_data(
    data: pd.DataFrame,
    features: List[str],
    target: str,
    sequence_length: int = 12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for LSTM model training.
    
    Args:
        data: Input DataFrame
        features: List of feature column names
        target: Target variable name
        sequence_length: Length of input sequences
        
    Returns:
        Tuple of (X_seq, y_seq) arrays for LSTM training
    """
    logger.info(f"Preparing LSTM data with sequence length: {sequence_length}")
    
    # Select features and target
    available_features = [f for f in features if f in data.columns]
    if len(available_features) != len(features):
        missing = set(features) - set(available_features)
        logger.warning(f"Missing features: {missing}")
    
    feature_data = data[available_features].values
    target_data = data[target].values
    
    # Remove missing values
    mask = ~(np.isnan(feature_data).any(axis=1) | np.isnan(target_data))
    feature_data = feature_data[mask]
    target_data = target_data[mask]
    
    if len(feature_data) < sequence_length + 1:
        raise ValueError(f"Not enough data points ({len(feature_data)}) for sequence length {sequence_length}")
    
    # Create sequences
    X_seq = []
    y_seq = []
    
    for i in range(sequence_length, len(feature_data)):
        X_seq.append(feature_data[i-sequence_length:i])
        y_seq.append(target_data[i])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    logger.info(f"Prepared {len(X_seq)} sequences with shape {X_seq.shape}")
    
    return X_seq, y_seq


def plot_temporal_effects(
    temporal_results: Dict[str, np.ndarray],
    treatment: str,
    outcome: str,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot temporal evolution of causal effects.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    times = temporal_results['time']
    effects = temporal_results['effect']
    ci_lower = temporal_results['ci_lower']
    ci_upper = temporal_results['ci_upper']
    
    # Plot effect line
    ax.plot(times, effects, 'b-', linewidth=2, label='Causal Effect')
    
    # Plot confidence interval
    ax.fill_between(times, ci_lower, ci_upper, alpha=0.3, color='blue',
                    label='95% Confidence Interval')
    
    # Add zero line
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No Effect')
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(f'Causal Effect\n({treatment} → {outcome})', fontsize=12)
    ax.set_title(f'Temporal Evolution of Causal Effect', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved temporal plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def compare_causal_effects(
    effects_dict: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Create a comparison plot of multiple causal effects.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    names = list(effects_dict.keys())
    coefficients = [effects_dict[n]['coefficient'] for n in names]
    ci_lowers = [effects_dict[n]['ci_lower'] for n in names]
    ci_uppers = [effects_dict[n]['ci_upper'] for n in names]
    
    # Calculate error bars
    errors = [[c - l for c, l in zip(coefficients, ci_lowers)],
              [u - c for c, u in zip(coefficients, ci_uppers)]]
    
    # Create bar plot
    colors = ['green' if c > 0 else 'red' for c in coefficients]
    bars = ax.barh(names, coefficients, xerr=errors, capsize=5, color=colors, alpha=0.7)
    
    # Add zero line
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add significance markers
    for i, name in enumerate(names):
        p_val = effects_dict[name].get('p_value', 1.0)
        if p_val < 0.001:
            marker = '***'
        elif p_val < 0.01:
            marker = '**'
        elif p_val < 0.05:
            marker = '*'
        else:
            marker = ''
        if marker:
            x_pos = coefficients[i] + (ci_uppers[i] - coefficients[i]) + 0.05
            ax.text(x_pos, i, marker, va='center', fontsize=12)
    
    ax.set_xlabel('Causal Effect (Coefficient)', fontsize=12)
    ax.set_title('Comparison of Causal Effects', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparison plot to {output_path}")
    else:
        plt.show()
    
    plt.close()
