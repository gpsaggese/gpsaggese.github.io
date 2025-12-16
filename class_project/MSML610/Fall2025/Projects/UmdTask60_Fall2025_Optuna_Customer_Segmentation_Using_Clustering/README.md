# Optuna Customer Segmentation Using Clustering

This project segments customers from the UCI Online Retail dataset into three actionable business groups using RFM (Recency, Frequency, Monetary) analysis and machine learning. The segmentation pipeline combines data cleaning, feature engineering, hyperparameter optimization with Optuna, and multi-algorithm comparison to identify the best clustering approach.

### What is Optuna? 

**Optuna** is a Bayesian hyperparameter optimization framework that intelligently searches the hyperparameter space instead of using exhaustive grid search or random guessing. Rather than testing all combinations (slow) or random combinations (inefficient), Optuna **learns from past trials** and suggests promising hyperparameters to test next.

**Why Optuna over Manual Tuning?**
- **Grid Search:** Test every combination (e.g., 6 choices × 3 choices × 20 choices = 360 trials) — exhaustive but slow
- **Random Search:** Randomly pick combinations — faster but inefficient sampling
- **Optuna (Bayesian):** Learn which regions are promising, sample densely there — fast AND accurate


## 1. Project Overview

The project addresses a common business problem: **How do we identify and target customer groups with different engagement and value patterns?** By analyzing transactional data, we compute RFM metrics—a proven framework in retail marketing—and use clustering algorithms optimized via Optuna to discover natural customer segments.

**Key Achievement:** Identified three distinct customer segments (VIP Champions, At-Risk/Lost, Loyal Customers) with different business strategies, using composite scoring to ensure robust model selection.

### Files Included


```
UmdTask60_Optuna_Clustering/
├── Optuna_Segmentation.example.ipynb
├── Optuna.API.ipynb
├── requirements.txt
├── Optuna.API.md
├── Optuna.example.md
├── Optuna_utils.py
├── README.md
├── online_retail.xlsx
├── cluster_summary.csv
├── customer_segments_final.csv
├── docker_build.sh
├── docker_jupyter.sh
└── Dockerfile/
```


**Core Implementation:**
- `Optuna_utils.py` — Reusable module with data pipelines, RFM computation, three clustering algorithms (K-Means, DBSCAN, Hierarchical), evaluation metrics, and visualization functions.
- `Optuna_Segmentation.example.ipynb` — End-to-end analysis notebook orchestrating the full workflow with markdown explanations and outputs.

**Documentation & Configuration:**
- `Optuna.example.md` — Comprehensive API reference for all functions with parameter descriptions, design rationale, and usage examples.
- `README.md` — This file; methodology, results, and setup instructions.
- `requirements.txt` — Pinned Python dependencies for reproducibility.
- `Optuna.API.ipynb` - Interactive Jupyter notebook demonstrating Optuna with live examples.
- - `Optuna.API.md` - Comprehensive Optuna framework reference covering concepts. 
**Data & Results:**
- `online_retail.xlsx` — UCI Online Retail dataset (541,909 transactions, Dec 2010–Dec 2011).
- `cluster_summary.csv` — Final segmentation summary with cluster statistics and business labels.
- `customer_segments_final.csv` — Customer-level RFM scores and cluster assignments (4,273 customers).

**Docker Setup:**
- `docker_build.sh`, `docker_jupyter.sh` — Scripts to build and run the project in a containerized Jupyter environment.

---

## 2. Methodology & Design Decisions

### 2.1 Data Processing

The pipeline begins with raw transaction data and applies rigorous cleaning to ensure quality inputs for RFM computation.

**Input:** 541,909 raw transactions across 8 columns (InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country).

**Cleaning Steps:**
1. **Remove duplicates** — Exact row duplicates are rare but can occur in data entry.
2. **Standardize text** — Convert StockCode and Description to uppercase and strip whitespace for consistent matching.
3. **Filter service codes** — Remove non-product transactions (POST, D, C2, M, BANK, PADS, DOT). These represent adjustments, returns, or bank charges, not customer purchases.
4. **Remove invalid prices/quantities** — Drop any row where UnitPrice ≤ 0 or Quantity ≤ 0 (data errors).
5. **Cap outliers** — Use 99th percentile capping to reduce the impact of extreme values (e.g., bulk orders) without removing them entirely.
6. **Compute revenue** — Create a Revenue column (Quantity × UnitPrice) for monetary aggregation.
7. **Parse dates** — Convert InvoiceDate to datetime for temporal analysis.
8. **Final filter** — Remove rows with missing CustomerID or invalid dates.

**Output:** 388,561 cleaned transactions (71.7% retention). The loss represents genuinely invalid or non-transactional records.

### 2.2 RFM Feature Engineering

RFM is an industry-standard framework for customer segmentation, proven in retail, e-commerce, and subscription businesses. It captures three dimensions of customer value.

**Aggregation:** Group cleaned transactions by CustomerID and compute three metrics over the entire period:

- **Recency (R):** Days elapsed from the last purchase to the reference date (Dec 10, 2011, one day after dataset end). Lower recency indicates more recent engagement. Range: 1–374 days.
- **Frequency (F):** Count of distinct InvoiceNo per customer. Higher frequency indicates loyalty and repeat purchasing. Range: 1–206 orders.
- **Monetary (M):** Sum of Revenue per customer. Higher monetary value indicates customer worth. Range: £3.75–£209,908.86.

**Outlier Removal:** Apply Z-score filtering (threshold > 3) to remove 60 extreme outliers (e.g., bulk corporate accounts with £100K+ revenue or anomalies). This retains 4,273 customers (98.6% of 4,333) while protecting the clustering algorithm from being dominated by edge cases.

### 2.3 Feature Scaling

Before clustering, standardize the RFM features using z-score normalization (mean=0, std=1). This is critical because:

- Recency ranges 1–374 days.
- Frequency ranges 1–206 orders.
- Monetary ranges £3.75–£209,908.

Without scaling, Monetary would dominate the distance metric due to its larger magnitude, making frequency and recency nearly irrelevant. StandardScaler ensures all three dimensions contribute equally to cluster formation.

### 2.4 Hyperparameter Optimization with Optuna

Optuna is a Bayesian hyperparameter optimization framework. Unlike grid search, it intelligently samples the hyperparameter space, making efficient use of computation budgets (100 trials per algorithm).

**K-Means Optimization (100 trials):**
- Search space: n_clusters ∈ [3, 8], init ∈ {k-means++, random}, n_init ∈ [10, 30].
- Objective: Maximize silhouette score (−1 to 1, higher is better).
- Best result: n_clusters=3, init=k-means++, silhouette ≈ 0.5605.

**Hierarchical Clustering Optimization (100 trials):**
- Search space: n_clusters ∈ [3, 8], linkage ∈ {ward, complete, average}.
- Objective: Maximize silhouette score.
- Best result: n_clusters=3, linkage=average, silhouette ≈ 0.5712 — **WINNER**.

**DBSCAN Optimization (50 trials, all pruned):**
- Search space: eps ∈ [0.3, 2.5], min_samples ∈ [3, 10].
- Objective: Maximize silhouette score.
- Issue: RFM data has variable cluster densities. VIP customers form a tight, dense cluster; At-Risk customers are sparse; Loyal customers are medium-density. DBSCAN assumes uniform density and fails. All 50 trials were pruned (marked as invalid) when producing < 2 clusters or > 50% noise.

### 2.5 Evaluation & Model Selection

After optimization, three best models (one per algorithm) are evaluated using three independent metrics:

- **Silhouette Score** (−1 to 1, higher is better): Measures how well samples fit their own cluster vs. others.
- **Davies-Bouldin Index** (0 to ∞, lower is better): Average similarity between each cluster and its most similar other cluster.
- **Calinski-Harabasz Score** (0 to ∞, higher is better): Ratio of between-cluster to within-cluster variance.

**Composite Scoring (60/40 weighting):**
To avoid over-relying on any single metric, we combine them:
1. Normalize silhouette and Davies-Bouldin to [0, 1].
2. Invert Davies-Bouldin (lower is better → higher after inversion).
3. Compute: Composite = 0.6 × Silhouette_normalized + 0.4 × DB_normalized_inverted.

**Winner Selection:** Hierarchical Clustering (composite score highest) — silhouette 0.5712, Davies-Bouldin 0.6234.

---

## 3. Final Results & Business Insights

The winning model produces three customer segments with distinct RFM profiles and actionable strategies.

| Segment | Count | Avg Recency | Avg Frequency | Avg Monetary | Profile |
|---------|-------|-------------|---------------|--------------|---------|
| **VIP Champions** | 257 (6.0%) | 12.98 days | 15.6 orders | £5,858.91 | Highly engaged, recent, high-value |
| **At-Risk/Lost** | 1,221 (28.6%) | 234.04 days | 1.69 orders | £500.28 | Dormant, minimal engagement |
| **Loyal Customers** | 2,795 (65.4%) | 40.04 days | 3.41 orders | £1,199.83 | Steady, balanced, mid-value |

### Business Recommendations

**VIP Champions (257 customers, £1.5M annual revenue):**
- **Characteristics:** Purchased within 13 days, average 15 orders, £5,859 lifetime value.
- **Strategy:** Premium retention. Offer VIP membership, exclusive early access to sales, personalized concierge service, loyalty bonuses.
- **Risk:** Churn would directly impact revenue; prioritize retention.

**At-Risk/Lost (1,221 customers, £610K revenue):**
- **Characteristics:** Haven't purchased in 234 days (8 months), low repeat rate, low spend.
- **Strategy:** Win-back campaigns. Send reactivation email series, offer time-limited discounts (20–30%), invite to exclusive events, request feedback on why they left.
- **Opportunity:** Even 10% reactivation = 122 customers × £500 = £61K additional revenue.

**Loyal Customers (2,795 customers, £3.4M revenue):**
- **Characteristics:** Moderate recency (40 days), steady purchases (3–4 orders), good mid-tier value.
- **Strategy:** Upselling and expansion. Recommend complementary products, create tiered loyalty program, introduce referral incentives, cross-sell higher-value items.
- **Opportunity:** Majority segment; modest per-customer improvements compound to large revenue gains.

---

## 4. Output Files

After running the full pipeline, two CSV files are generated containing the final segmentation results:

### cluster_summary.csv

**Purpose:** High-level cluster overview with business labels. Use for stakeholder reporting and segment strategy planning.

**Structure:** One row per cluster (3 total).

**Columns:**
- `Cluster` (int): Cluster ID (0, 1, 2)
- `Customer_Count` (int): Number of customers assigned to cluster
- `Avg_Recency` (float): Mean recency in days
- `Avg_Frequency` (float): Mean frequency (number of orders)
- `Avg_Monetary` (float): Mean monetary value in GBP
- `Segment_Label` (str): Business-friendly segment name

**Example:**
```
Cluster,Customer_Count,Avg_Recency,Avg_Frequency,Avg_Monetary,Segment_Label
0,257,12.98,15.6,5858.91,VIP Champions
1,1221,234.04,1.69,500.28,At-Risk/Lost
2,2795,40.04,3.41,1199.83,Loyal Customers
```

**Use Case:** Share with marketing, sales, and executive stakeholders. Quick reference for segment profiles and naming.

---

### customer_segments_final.csv

**Purpose:** Customer-level segmentation with full RFM scores. Use for targeted campaigns and CRM integration.

**Structure:** One row per customer (4,273 total after outlier removal).

**Columns:**
- `CustomerID` (float): Unique customer identifier
- `Recency` (float): Days since last purchase
- `Frequency` (float): Number of distinct invoices
- `Monetary` (float): Total revenue in GBP
- `Cluster` (int): Assigned cluster (0=VIP, 1=At-Risk, 2=Loyal)

**Example:**
```
CustomerID,Recency,Frequency,Monetary,Cluster
12346.0,326,1,104.0,1
12347.0,2,7,4127.05,2
12348.0,75,4,1302.08,2
12349.0,19,1,1434.23,2
```

**Use Case:** 
- Export to CRM system (Salesforce, HubSpot, etc.) to segment customer lists.
- Feed to email marketing platform for targeted campaigns by cluster.
- Analyze in BI tool (Tableau, Power BI) for deeper insights.
- Filter by cluster for A/B testing new strategies.

---

## 5. How to Run

### 5.1 Using Docker (Recommended)

```bash
# Build the container (one-time setup)
./docker_build.sh

# Start Jupyter Lab
./docker_jupyter.sh

# Open browser to http://127.0.0.1:8888
# Run Optuna_Segmentation.ipynb from top to bottom
```

### 5.2 Manual Setup (Python 3.10+)

```bash
pip install -r requirements.txt
jupyter lab Optuna_Segmentation.ipynb
```

### 5.3 Notebook Workflow

The notebook is organized into logical sections:
1. **Setup & Data Loading** — Import libraries, load raw data (541,909 transactions).
2. **Data Cleaning** — Apply 8-step pipeline, output 388,561 cleaned transactions.
3. **RFM Computation** — Aggregate to 4,333 customers, compute Recency/Frequency/Monetary.
4. **Outlier Removal** — Z-score filtering, retain 4,273 customers.
5. **Scaling** — StandardScaler normalization.
6. **Hyperparameter Optimization** — Run Optuna for K-Means, Hierarchical, DBSCAN.
7. **Optuna Visualizations** — Plot optimization history and hyperparameter importance.
8. **Model Comparison** — Evaluate all algorithms, select winner (Hierarchical).
9. **Cluster Profiling** — Generate 2D PCA scatter, 3D RFM scatter, RFM heatmap.
10. **Results Export** — Save segmentation results to CSV files.

Each section includes markdown cells explaining decisions, outputs showing results, and plots for interpretation.

---

## 6. Reproducibility & Extensibility

### 6.1 Reproducibility

All random operations use fixed seeds:
- Optuna samplers (TPE) seeded at initialization.
- K-Means random_state=42.
- NumPy/Pandas operations deterministic.

Dependencies are pinned in `requirements.txt` (not >=, but exact versions), ensuring identical results across environments and time.

### 6.2 Extensibility

The modular design makes it easy to adapt to other datasets:
- **Different data source?** Replace `load_data()` to read your CSV/database.
- **Different reference date?** Modify `refdate` in `compute_rfm()`.
- **Tune hyperparameters?** Adjust search space in `optimize_kmeans()`, `optimize_hierarchical()`.
- **Add another algorithm?** Write `optimize_mystrategy()` following the same pattern.
- **Adjust composite weights?** Change 0.6/0.4 in `compare_algorithms()`.

All functions are in `Optuna_utils.py` with docstrings explaining inputs, outputs, and rationale.

---

## 7. Technical Details

**Libraries & Versions** (see `requirements.txt`):
- Pandas 2.0.3 — Data manipulation and aggregation.
- NumPy 1.24.3 — Numerical computing.
- scikit-learn 1.3.0 — Clustering algorithms, metrics, scaling.
- Optuna 3.3.0 — Bayesian hyperparameter optimization.
- Matplotlib 3.7.2, Seaborn 0.12.2 — Visualization.
- openpyxl 3.1.2 — Excel file reading.

**Algorithms:**
- **K-Means:** Fast, intuitive, assumes spherical clusters. Good baseline.
- **Hierarchical:** Deterministic, produces dendrogram. Slightly better silhouette on this data.
- **DBSCAN:** Density-based, handles non-spherical shapes. Fails on RFM due to variable density.

**Metrics:**
- **Silhouette:** -1 (bad) to +1 (good). Balances within-cluster cohesion and between-cluster separation.
- **Davies-Bouldin:** Lower is better. Penalizes clusters that are too similar or too large.
- **Composite (60/40):** Weighted combination to balance both perspectives and reduce metric-specific biases.

---

## 8. Key Findings

1. **Three clusters are optimal** — Both K-Means and Hierarchical converged on n_clusters=3, confirmed by silhouette scores (0.56–0.57). Adding or removing a cluster decreases score.

2. **Hierarchical slightly outperforms K-Means** — Silhouette 0.5712 vs. 0.5605. Difference is small but statistically meaningful over 4,273 samples.

3. **DBSCAN fails gracefully** — Pruning mechanism detected that no valid clustering exists. Good failure detection rather than producing misleading results.

4. **RFM accurately reflects business reality** — VIP (recent, frequent, high-spend), At-Risk (dormant, low spend), Loyal (steady mid-tier). Segment profiles align with intuitive business expectations.

5. **Composite scoring adds robustness** — Using both silhouette and Davies-Bouldin prevents over-weighting a single metric. Final model is more stable and defensible.

---

## 9. Conclusion

This project demonstrates a customer segmentation pipeline combining modern ML (Optuna hyperparameter tuning) with business fundamentals (RFM framework). The result is **three actionable customer segments with clear revenue implications and strategic recommendations**, backed by rigorous evaluation and reproducible code.

---

**Course:** MSML610 – Machine Learning, University of Maryland  
**Author:** Ruthvick Kandrala 
**Date:** December 2025
