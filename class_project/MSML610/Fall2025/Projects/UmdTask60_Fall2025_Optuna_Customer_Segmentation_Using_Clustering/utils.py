"""
Project: Customer Segmentation Using Clustering with Optuna

This module provides utilities for customer segmentation using clustering algorithms
optimized with Optuna's hyperparameter search framework.
"""

import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import optuna
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# DATA LOADING AND PREPROCESSING


def load_data(filepath):
    """
    Load Online Retail dataset from Excel file.
    
    Args:
        filepath (str): Path to Excel file
        
    Returns:
        pd.DataFrame: Raw transaction data
    """
    df = pd.read_excel(filepath)
    return df


def clean_data(df):
    """
    Clean and preprocess transaction data.
    
    Steps:
        1. Remove duplicate transactions
        2. Standardize text fields (StockCode, Description)
        3. Remove service codes (non-product transactions)
        4. Remove zero/negative prices
        5. Cap outliers at 99th percentile to reduce noise
        6. Calculate revenue per transaction
        7. Parse dates for temporal analysis
        8. Filter valid transactions only
    
    Args:
        df (pd.DataFrame): Raw transaction data
        
    Returns:
        pd.DataFrame: Cleaned transaction data
    """
    # Remove exact duplicates
    df = df.drop_duplicates()
    
    # Standardize text columns for consistency
    df['StockCode'] = df['StockCode'].astype(str).str.strip().str.upper()
    df['Description'] = df['Description'].astype(str).str.strip().str.title()
    
    # Remove service codes (POST, DISC, etc.) - these are adjustments, not products
    service_codes = ['POST', 'D', 'C2', 'M', 'BANK', 'PADS', 'DOT']
    for code in service_codes:
        df = df[~df['StockCode'].str.contains(code, na=False)]
    
    # Remove invalid prices - business rule: price must be positive
    df = df[df['UnitPrice'] > 0]
    
    # Cap outliers at 99th percentile to reduce impact of data entry errors
    # This prevents extreme values from dominating clustering
    q99_qty = df['Quantity'].quantile(0.99)
    q99_price = df['UnitPrice'].quantile(0.99)
    df['Quantity'] = df['Quantity'].clip(upper=q99_qty)
    df['UnitPrice'] = df['UnitPrice'].clip(upper=q99_price)
    
    # Calculate revenue for RFM analysis
    df['Revenue'] = df['Quantity'] * df['UnitPrice']
    
    # Parse dates for recency calculation
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Final filters: positive quantity, valid customer ID
    df = df[df['Quantity'] > 0]
    df = df.dropna(subset=['CustomerID', 'InvoiceDate'])
    df = df.reset_index(drop=True)
    
    return df



# FEATURE ENGINEERING


def compute_rfm(df):
    """
    Compute RFM (Recency, Frequency, Monetary) features for customer segmentation.
    
    RFM is a proven framework for customer value analysis:
        - Recency: Days since last purchase (lower = more engaged)
        - Frequency: Number of unique transactions (higher = more loyal)
        - Monetary: Total revenue generated (higher = more valuable)
    
    Args:
        df (pd.DataFrame): Cleaned transaction data
        
    Returns:
        pd.DataFrame: Customer-level RFM features
    """
    # Reference date: 1 day after last transaction in dataset
    # This ensures all recency values are positive
    ref_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    # Aggregate transactions to customer level
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (ref_date - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',                               # Frequency
        'Revenue': 'sum'                                      # Monetary
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    return rfm



# OPTUNA OPTIMIZATION


def optimize_kmeans(X, n_trials=100):
    """
    Optimize K-Means hyperparameters using Optuna with silhouette score.
    
    Silhouette score is used as the objective because:
        - Industry standard metric for cluster quality
        - Measures both separation and cohesion
        - Range [-1, 1], higher is better
        - More reliable than composite metrics
    
    Hyperparameters optimized:
        - n_clusters: Number of clusters (3-8 range based on business needs)
        - init: Initialization method (k-means++ vs random)
        - n_init: Number of initializations (10-30 for stability)
    
    Note: No pruning for K-Means since it converges in single fit.
    
    Args:
        X (np.ndarray): Scaled feature matrix (n_customers, n_features)
        n_trials (int): Number of Optuna trials
        
    Returns:
        optuna.Study: Completed optimization study
    """
    
    def objective(trial):
        # Suggest hyperparameters
        n_clusters = trial.suggest_int('n_clusters', 3, 8)
        init = trial.suggest_categorical('init', ['k-means++', 'random'])
        n_init = trial.suggest_int('n_init', 10, 30)
        
        # Train model
        model = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            random_state=42
        )
        labels = model.fit_predict(X)
        
        # Evaluate using silhouette score
        score = silhouette_score(X, labels)
        
        return score
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return study


def optimize_dbscan(X, n_trials=50):
    """
    Optimize DBSCAN hyperparameters using Optuna with pruning.
    
    DBSCAN challenges:
        - Sensitive to eps (neighborhood radius) and min_samples
        - Can produce many noise points if parameters are poor
        - May fail to find valid clusters
    
    Optimization strategy:
        - Search eps from 0.3 to 2.5 (scaled feature space)
        - Search min_samples from 3 to 10
        - Prune trials with too many noise points (>50%)
        - Prune trials with < 2 clusters
    
    Pruning is appropriate here because we can detect invalid
    clustering early without computing silhouette.
    
    Args:
        X (np.ndarray): Scaled feature matrix
        n_trials (int): Number of Optuna trials
        
    Returns:
        optuna.Study: Completed optimization study
    """
    
    def objective(trial):
        eps = trial.suggest_float('eps', 0.3, 2.5, step=0.1)
        min_samples = trial.suggest_int('min_samples', 3, 10)
        
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        
        # Check cluster validity
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        noise_ratio = n_noise / len(labels)
        
        # Prune if degenerate solution
        if n_clusters < 2 or noise_ratio > 0.5:
            raise optuna.TrialPruned()
        
        # Evaluate only non-noise points
        mask = labels != -1
        score = silhouette_score(X[mask], labels[mask])
        
        return score
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return study


def optimize_hierarchical(X, n_trials=100):
    """
    Optimize Hierarchical Clustering hyperparameters using Optuna.
    
    Hierarchical clustering advantages:
        - Deterministic (no random initialization)
        - Provides dendrogram for interpretation
        - Works well with spherical and non-spherical clusters
    
    Hyperparameters optimized:
        - n_clusters: Number of clusters to cut dendrogram (3-8)
        - linkage: Distance metric between clusters
            - ward: Minimize variance (best for K-Means-like data)
            - complete: Minimize max distance (compact clusters)
            - average: Balance between ward and complete
    
    Note: No pruning since hierarchical clustering completes in single fit.
    
    Args:
        X (np.ndarray): Scaled feature matrix
        n_trials (int): Number of Optuna trials
        
    Returns:
        optuna.Study: Completed optimization study
    """
    
    def objective(trial):
        n_clusters = trial.suggest_int('n_clusters', 3, 8)
        linkage = trial.suggest_categorical('linkage', ['ward', 'complete', 'average'])
        
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )
        labels = model.fit_predict(X)
        
        # Evaluate using silhouette score
        score = silhouette_score(X, labels)
        
        return score
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return study



# EVALUATION


def evaluate_clusters(X, labels, name):
    """
    Compute standard clustering evaluation metrics.
    
    Metrics:
        - Silhouette: [-1, 1], higher = better separation
        - Davies-Bouldin: [0, inf), lower = better (more compact/separated)
        - Calinski-Harabasz: [0, inf), higher = better (ratio of between/within variance)
    
    Args:
        X (np.ndarray): Feature matrix
        labels (np.ndarray): Cluster assignments
        name (str): Algorithm name
        
    Returns:
        dict: Evaluation metrics
    """
    # Only filter for DBSCAN (which has -1 noise labels)
    if -1 in labels:
        mask = labels != -1
        X_valid = X[mask]
        labels_valid = labels[mask]
    else:
        X_valid = X
        labels_valid = labels
    
    silh = silhouette_score(X_valid, labels_valid)
    db = davies_bouldin_score(X_valid, labels_valid)
    ch = calinski_harabasz_score(X_valid, labels_valid)
    n_clusters = len(set(labels_valid))
    
    return {
        'algorithm': name,
        'silhouette': silh,
        'davies_bouldin': db,
        'calinski_harabasz': ch,
        'n_clusters': n_clusters
    }


def compare_algorithms(results):
    """
    Compare algorithms using BOTH Silhouette and Davies-Bouldin.

    Args:
        results (list of dict): Output from evaluate_clusters() or manual dicts 
                                with keys: 'algorithm', 'silhouette', 
                                'davies_bouldin'.

    Returns:
        dict: Winning algorithm metrics.
    """
    df = pd.DataFrame(results)

    # Normalize Silhouette
    df['silh_norm'] = (df['silhouette'] - df['silhouette'].min()) / \
                      (df['silhouette'].max() - df['silhouette'].min() + 1e-9)

    # Normalize DB using the UNDERSCORE name
    db_norm_raw = (df['davies_bouldin'] - df['davies_bouldin'].min()) / \
                  (df['davies_bouldin'].max() - df['davies_bouldin'].min() + 1e-9)
    df['db_norm'] = 1 - db_norm_raw

    # Composite: 60% silhouette, 40% DB
    df['composite_score'] = 0.6 * df['silh_norm'] + 0.4 * df['db_norm']

    winner_idx = df['composite_score'].idxmax()
    winner = df.loc[winner_idx].to_dict()

    print("\nALGORITHM COMPARISON (Multi-Metric)")
    print(f"{'Algorithm':<15} {'Silhouette':<12} {'Davies-Bouldin':<15} {'Composite':<10}")
    for _, row in df.iterrows():
        mark = " ← WINNER" if row['algorithm'] == winner['algorithm'] else ""
        print(f"{row['algorithm']:<15} {row['silhouette']:<12.4f} "
              f"{row['davies_bouldin']:<15.4f} {row['composite_score']:<10.4f}{mark}")

    return winner

# VISUALIZATION

def plot_pca_clusters(X, labels, title="Customer Segments - PCA Projection"):
    """
    Visualize clusters in 2D using PCA projection.
    
    PCA reduces 3D RFM space to 2D for visualization while preserving
    maximum variance. Shows cluster separation and overlap.
    
    Args:
        X (np.ndarray): Feature matrix
        labels (np.ndarray): Cluster assignments
        title (str): Plot title
        
    Returns:
        matplotlib.figure.Figure: Plot figure
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_labels = sorted(set(labels))
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:
            # DBSCAN noise points
            color = [0, 0, 0, 0.5]
            marker = 'x'
            label_name = 'Noise'
            zorder = 1
        else:
            marker = 'o'
            label_name = f'Cluster {label}'
            zorder = 2
        
        mask = labels == label
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=[color],
            label=label_name,
            marker=marker,
            s=60,
            alpha=0.7,
            edgecolors='k',
            linewidths=0.5,
            zorder=zorder
        )
    
    ax.set_xlabel(
        f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
        fontsize=12,
        fontweight='bold'
    )
    ax.set_ylabel(
        f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
        fontsize=12,
        fontweight='bold'
    )
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    return fig


def plot_rfm_distributions(rfm):
    """
    Plot distributions of RFM features to understand data characteristics.
    
    Args:
        rfm (pd.DataFrame): RFM features
        
    Returns:
        matplotlib.figure.Figure: Plot figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Recency
    axes[0].hist(rfm['Recency'], bins=50, alpha=0.7, color='#3498db', edgecolor='black')
    axes[0].set_title('Recency Distribution', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Days Since Last Purchase', fontsize=10)
    axes[0].set_ylabel('Frequency', fontsize=10)
    axes[0].grid(alpha=0.3, axis='y')
    
    # Frequency
    axes[1].hist(rfm['Frequency'], bins=50, alpha=0.7, color='#2ecc71', edgecolor='black')
    axes[1].set_title('Frequency Distribution', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Number of Orders', fontsize=10)
    axes[1].set_ylabel('Frequency', fontsize=10)
    axes[1].grid(alpha=0.3, axis='y')
    
    # Monetary
    axes[2].hist(rfm['Monetary'], bins=50, alpha=0.7, color='#e74c3c', edgecolor='black')
    axes[2].set_title('Monetary Distribution', fontweight='bold', fontsize=12)
    axes[2].set_xlabel('Total Revenue (£)', fontsize=10)
    axes[2].set_ylabel('Frequency', fontsize=10)
    axes[2].grid(alpha=0.3, axis='y')
    
    plt.suptitle('RFM Feature Distributions', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def plot_cluster_profiles(rfm, labels, title="Cluster Profiles Heatmap"):
    """
    Visualize cluster characteristics using heatmap of average RFM values.
    
    Normalized heatmap shows relative differences between clusters.
    Annotations show actual mean values for business interpretation.
    
    Args:
        rfm (pd.DataFrame): RFM features
        labels (np.ndarray): Cluster assignments
        title (str): Plot title
        
    Returns:
        matplotlib.figure.Figure: Plot figure
    """
    rfm_copy = rfm.copy()
    rfm_copy['Cluster'] = labels
    
    # Remove noise points if present
    rfm_copy = rfm_copy[rfm_copy['Cluster'] != -1]
    
    # Compute mean RFM per cluster
    profiles = rfm_copy.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    
    # Normalize for heatmap (0-1 scale)
    profiles_norm = (profiles - profiles.min()) / (profiles.max() - profiles.min())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        profiles_norm.T,
        annot=profiles.T.round(1),
        fmt='g',
        cmap='YlOrRd',
        cbar_kws={'label': 'Normalized Value'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('RFM Metric', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    return fig



# OPTUNA VISUALIZATION WRAPPERS


def plot_optuna_history(study, title="Optimization History"):
    """Wrapper for Optuna's optimization history plot."""
    import optuna.visualization as vis
    fig = vis.plot_optimization_history(study)
    fig.update_layout(title=title)
    return fig


def plot_optuna_param_importance(study, title="Hyperparameter Importance"):
    """Wrapper for Optuna's parameter importance plot."""
    import optuna.visualization as vis
    fig = vis.plot_param_importances(study)
    fig.update_layout(title=title)
    return fig


def plot_optuna_parallel_coordinate(study, title="Parallel Coordinate Plot"):
    """Wrapper for Optuna's parallel coordinate plot."""
    import optuna.visualization as vis
    fig = vis.plot_parallel_coordinate(study)
    fig.update_layout(title=title)
    return fig
