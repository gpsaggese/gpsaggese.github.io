# Utils.py & Optuna_segmentation.example.ipynb Reference

## Quick Overview

`utils.py` & `Optuna_Segmentation.example.ipynb` provides reusable functions across data processing, RFM feature engineering, Optuna-based hyperparameter optimization, evaluation, and visualization. All functions follow a consistent pattern: clear inputs, meaningful outputs, documented rationale.

**Function Categories:**
- **Data** (2): load_data, clean_data
- **Features** (1): compute_rfm
- **Optimization** (3): optimize_kmeans, optimize_dbscan, optimize_hierarchical
- **Evaluation** (2): evaluate_clusters, compare_algorithms
- **Visualization** (3): plot_pca_clusters, plot_3d_clusters, plot_cluster_profiles

---

## 1. Data Processing

### load_data(filepath: str) → pd.DataFrame

Loads Online Retail Excel file. Returns (541,909, 8): raw transactions with no cleaning applied.

---

### clean_data(df: pd.DataFrame) → pd.DataFrame

Applies 8-step cleaning pipeline:

1. Remove duplicates
2. Standardize text (uppercase, strip whitespace)
3. Remove service codes (POST, D, C2, M, BANK, PADS, DOT) — adjustments & returns, not sales
4. Drop rows where UnitPrice ≤ 0 or Quantity ≤ 0
5. Cap outliers at 99th percentile (Quantity, UnitPrice) — reduces impact of bulk orders
6. Compute Revenue = Quantity × UnitPrice
7. Parse InvoiceDate to datetime
8. Drop missing CustomerID or invalid dates

**Output:** (388,561, 9) cleaned transactions. Retention: 71.7% (28.3% invalid).

**Design:** Percentile capping balances robustness (removes outlier influence) with data preservation (doesn't discard them).

---

## 2. Feature Engineering

### compute_rfm(df: pd.DataFrame) → pd.DataFrame

Aggregates transactions to customer level. Computes RFM metrics per customer:

- **Recency (R):** Days since last purchase (1–374 days). Lower = more recent/engaged.
- **Frequency (F):** Distinct invoices per customer (1–206). Higher = more loyal.
- **Monetary (M):** Total revenue in GBP (£3.75–£209,909). Higher = more valuable.

**Output:** (4,333, 4) with columns [CustomerID, Recency, Frequency, Monetary].

**Design:** RFM is industry-standard (proven 40+ years in retail). Business-interpretable without domain knowledge. Groups by distinct InvoiceNo (not item count) to prevent single invoices with many items from inflating Frequency.

---

## 3. Hyperparameter Optimization

All three functions use Optuna (Bayesian search) to maximize silhouette score. Each returns `optuna.Study` containing all trials and best params.

### optimize_kmeans(X: np.ndarray, n_trials: int = 100) → optuna.Study

**Search space:**
- n_clusters: [3, 8]
- init: {k-means++, random}
- n_init: [10, 30]

**Result:** Best silhouette 0.5605 (n_clusters=3, init=k-means++, n_init=22).

**Design:** TPE sampler learns from past trials. No pruning (each trial is fast ~1s).

---

### optimize_dbscan(X: np.ndarray, n_trials: int = 50) → optuna.Study

**Search space:**
- eps: [0.3, 2.5]
- min_samples: [3, 10]

**Pruning:** Marks trial invalid if n_clusters < 2 or noise_ratio > 50%.

**Result:** All 50 trials pruned. DBSCAN unsuitable for RFM.

**Why:** RFM has variable cluster densities. VIP tight/dense, At-Risk sparse, Loyal medium. No single eps works for all.
- Small eps → finds only VIP, rest noise.
- Large eps → merges VIP+Loyal, At-Risk noise.

**Design:** Pruning demonstrates responsible ML: detect failures early, don't return misleading results.

---

### optimize_hierarchical(X: np.ndarray, n_trials: int = 100) → optuna.Study

**Search space:**
- n_clusters: [3, 8]
- linkage: {ward, complete, average}

**Result:** Best silhouette 0.5712 (n_clusters=3, linkage=average). **WINNER.**

**Design:** HAC is deterministic (no random init), so each trial always identical for same params. Optuna efficiently maps discrete space. No pruning (HAC rarely fails).

---

## 4. Evaluation & Selection

### evaluate_clusters(X: np.ndarray, labels: np.ndarray, name: str) → dict

Computes three independent metrics (no single metric is universally best):

1. **Silhouette** (−1 to 1, higher better): How similar samples are to own cluster vs. others.
2. **Davies-Bouldin** (0 to ∞, lower better): Average max similarity across cluster pairs. Penalizes large/overlapping clusters.
3. **Calinski-Harabasz** (0 to ∞, higher better): Ratio of between-cluster to within-cluster variance.

**Returns:** dict with keys {algorithm, silhouette, davies_bouldin, calinski_harabasz, n_clusters}.

**Design:** Three independent perspectives reduce bias. DBSCAN noise points excluded (not in any cluster).

---

### compare_algorithms(results: list) → dict

Selects winner using **composite scoring (60/40 weighting):**

1. Normalize silhouette to [0, 1]: `sil_norm = (sil − min) / (max − min)`
2. Normalize Davies-Bouldin to [0, 1]: `db_norm = (db − min) / (max − min)`
3. Invert DB (lower is better): `db_norm = 1 − db_norm`
4. Composite: `0.6 × sil_norm + 0.4 × db_norm`

**Winner:** Highest composite score.

**Design:** 
- **60% silhouette:** Primary metric (balances cohesion & separation).
- **40% Davies-Bouldin:** Secondary metric (penalizes large clusters).
- **Normalization:** Ensures metrics on different scales contribute fairly.
- **Composite:** Balances both perspectives, more robust than either alone.

**Result:** Hierarchical wins with composite score highest.

---

## 5. Visualization

All three functions return `matplotlib.figure.Figure` for display or saving.

### plot_pca_clusters(X: np.ndarray, labels: np.ndarray, title: str = None)

Projects 3D scaled RFM to 2D using PCA. Scatter plot with clusters color-coded.

**Details:**
- PC1 and PC2 labeled with explained variance %.
- DBSCAN noise as black 'x' markers.
- Legend shows cluster labels and counts.

**Use:** Quick visual check of cluster separation. Well-separated = tight, distant blobs. Overlapping = intermingled colors.

**Note:** PCA is linear; non-linear relationships not captured.

---

### plot_3d_clusters(X: np.ndarray, labels: np.ndarray, title: str = None)

3D scatter in RFM space (Recency, Frequency, Monetary). Interactive in Jupyter: rotate, zoom, pan.

**Advantage over PCA:** No dimensionality loss. Shows all three features exactly as they are.

**Insight:** VIP in low-R, high-F/M corner. At-Risk in high-R, low-F/M corner. Loyal in middle.

---

### plot_cluster_profiles(rfm: pd.DataFrame, labels: np.ndarray, title: str = None)

Heatmap of mean RFM per cluster. Normalized to [0, 1] for color (Yellow=low, Orange=mid, Red=high).

**Example (3 clusters):**
```
        Cluster 0 (VIP)    Cluster 1 (At-Risk)  Cluster 2 (Loyal)
Recency    low (12)         high (234)           med (40)
Frequency  high (16)        low (2)              med (3)
Monetary   high (5,859)     low (500)            med (1,200)
```

Annotated with actual mean values. Business stakeholders read directly from heatmap.

---

## 6. Design Principles

**Modularity:** Each function has single responsibility (load, clean, compute RFM, optimize, evaluate, visualize).

**Extensibility:** Easy to adapt:
- New data source? Replace load_data().
- New algorithm? Write optimize_myalgo() following same pattern.
- Tune weights? Change 0.6/0.4 in compare_algorithms().
- New metric? Add to evaluate_clusters().

**Reproducibility:** Fixed random seeds (Optuna, K-Means, NumPy). Pinned package versions. Deterministic HAC. Results identical across runs.

---

## 7. Functions at a Glance

| Function | Inputs | Outputs | Purpose |
|----------|--------|---------|---------|
| load_data | filepath | Raw df (541,909, 8) | Load Excel |
| clean_data | Raw df | Cleaned df (388,561, 9) | 8-step pipeline |
| compute_rfm | Cleaned df | RFM df (4,333, 4) | Customer aggregation |
| optimize_kmeans | Scaled X | Optuna Study | Tune K-Means (100 trials) |
| optimize_dbscan | Scaled X | Optuna Study | Tune DBSCAN (50 trials, all pruned) |
| optimize_hierarchical | Scaled X | Optuna Study | Tune HAC (100 trials) |
| evaluate_clusters | X, labels | Metrics dict | 3 evaluation metrics |
| compare_algorithms | List[dict] | Winner dict | Composite scoring |
| plot_pca_clusters | X, labels | Figure | 2D PCA scatter |
| plot_3d_clusters | X, labels | Figure | 3D RFM scatter |
| plot_cluster_profiles | RFM, labels | Figure | RFM heatmap |

---

## 8. Results Summary

| Algorithm | Silhouette | Davies-Bouldin | n_clusters | Status |
|-----------|------------|----------------|------------|--------|
| K-Means | 0.5605 | 0.6345 | 3 |  Valid |
| Hierarchical | **0.5712** | **0.6234** | 3 | ** Winner** |
| DBSCAN | Pruned | N/A | N/A |  All trials pruned |

**Final segments (4,273 customers):**
- **VIP Champions (257):** Recency 13d, Frequency 16, Monetary £5,859 → Premium retention
- **At-Risk/Lost (1,221):** Recency 234d, Frequency 2, Monetary £500 → Win-back campaigns
- **Loyal Customers (2,795):** Recency 40d, Frequency 3, Monetary £1,200 → Upselling

---

**Course:** MSML610 – Machine Learning, University of Maryland  
**Date:** December 2025
