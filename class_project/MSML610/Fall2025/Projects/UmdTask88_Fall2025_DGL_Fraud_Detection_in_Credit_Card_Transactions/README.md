###Fraud Detection in Credit Card Transactions
##Graph-Based Fraud Detection Pipeline

#Overview

This project builds a heterogeneous bipartite graph of
transactions ↔ account proxies ("users") from the
IEEE-CIS Fraud Detection
 dataset.
The goal is to detect fraudulent transactions by leveraging both tabular features and relational structure between entities.

Component	Description
Data Pipeline	Joined train_transaction + train_identity into a cleaned parquet (merged.parquet).
Feature Engineering	Added transaction-level and account-level aggregated features (time of day, log-amount, rolling stats).
Graph Construction	Built a HeteroData object with transaction ↔ account nodes and reverse edges (data/artifacts/hetero_graph.pt).
Tabular Baseline	Logistic Regression (balanced) trained on Phase 1 features; metrics logged in data/artifacts/baseline_metrics.json.
Automation & Testing	Makefile targets for data→features→graph→baseline, plus Pytest smoke tests for graph counts & split integrity.
📊 Current Results
Metric	Validation (3 days hold-out)
Precision	~ 0.6 – 0.8 (expected range after leakage fix)
Recall	~ 0.6 – 0.8
PR-AUC	0.70 ± 0.05
(actual numbers in data/artifacts/baseline_metrics.json)	

#Next Steps

Leakage-free account features

Re-compute account aggregates using train-only data before merging to validation.

Graph Neural Network (GNN) Modeling

Implement GraphSAGE / GAT for transaction-node classification.

Compare GNN vs. tabular baselines on precision, recall, PR-AUC.

Temporal Encoding (Optional Booster)

Add time-aware node/edge embeddings to capture behavior drift.

Benchmark & Error Analysis

Study false positives/negatives; derive business insights like Precision@K.

MLOps Packaging

Dockerize pipeline, integrate ClearML or W&B for reproducible experiments.

#Project Structure
configs/         – YAML configs for paths & settings  
data/raw/        – Original Kaggle CSVs  
data/processed/  – Cleaned & feature-ready parquet files  
data/artifacts/  – Graph + metrics artifacts  
src/             – Modular Python package (data, features, graph, models, tests)  
notebooks/       – Prototyping / exploratory notebooks  
Makefile         – One-command pipeline orchestration  

#Tech Stack

Python 3.11 | Pandas | PyTorch | PyTorch Geometric | Scikit-learn | Make | PyYAML

#Inspiration / Use Case

Financial institutions like Capital One and Chase monitor high-volume transaction streams.
Graph-based modeling captures behavioral relationships (e.g., shared device, IP, or card proxy) that traditional tabular models miss—making it ideal for real-time fraud detection and risk scoring.

#Timeline Snapshot
Phase	Focus	Target
1	Data pipeline + baseline + graph build	✅ Completed
2	GNN modeling + evaluation	
3	Deployment / MLOps integration
