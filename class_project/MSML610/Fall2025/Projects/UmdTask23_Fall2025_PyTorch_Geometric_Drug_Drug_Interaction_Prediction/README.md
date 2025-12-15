Drug–Drug Interaction Prediction using Graph Neural Networks
Project Overview

This project focuses on predicting potential drug–drug interactions (DDIs) using graph-based deep learning. Each drug is represented by its molecular structure, and interactions are predicted using a Graph Neural Network (GNN) implemented with PyTorch Geometric.

In addition to the GNN model, a traditional machine learning baseline using Morgan fingerprints + Logistic Regression is implemented for comparison. This allows us to evaluate whether graph-based learning provides meaningful benefits over standard cheminformatics approaches.

Dataset

Source: Kaggle – Drug–Drug Interaction dataset

File used: db_drug_interactions_2.csv

Columns:

Drug 1

Drug 2

Interaction Description

Each row represents a known interaction between two drugs (positive samples).

Methodology
1. SMILES Retrieval

Drug names are mapped to SMILES strings using PubChem.

A local SMILES cache is maintained to avoid repeated API calls.

Drug names are normalized (lowercase, stripped) to ensure consistent lookups.

2. Graph Construction

SMILES strings are converted into molecular graphs using RDKit.

Nodes: Atoms

Edges: Chemical bonds

Node features: Basic atomic properties

Graphs are processed using PyTorch Geometric.

3. Dataset Construction

Positive samples (label = 1): Known interacting drug pairs.

Negative samples (label = 0): Randomly sampled drug pairs not present in known interactions.

Dataset is shuffled and split into:

Train

Validation

Test (stratified by label)

4. Model Architecture (GNN)

Encoder: Graph Attention Network (GAT)

Each drug graph is encoded independently.

Drug embeddings are combined and passed through a prediction head.

Output: interaction probability.

5. Class Imbalance Handling

The dataset is imbalanced. To address this:

pos_weight is used in BCEWithLogitsLoss

This improves recall and PR-AUC performance, which is critical for DDI prediction.

6. Baseline Model

A classical ML baseline is implemented:

Features: Morgan fingerprints (ECFP)

Model: Logistic Regression (via SGD)

This serves as a reference point to justify the use of GNNs.

Evaluation Metrics

Models are evaluated using:

ROC-AUC

PR-AUC (primary metric due to class imbalance)

Results
Graph Neural Network (GAT)
Metric	Score
Test ROC-AUC	~0.69
Test PR-AUC	~0.66
Baseline: Morgan Fingerprints + Logistic Regression
Metric	Score
Test ROC-AUC	~0.81
Test PR-AUC	~0.79
Discussion

The baseline model performs strongly due to handcrafted chemical fingerprints.

The GNN model, while slightly weaker in this setup, learns directly from molecular structure, offering better extensibility.

With more tuning, pretraining, or additional chemical features, GNNs can outperform traditional methods.

Including both models strengthens the project by demonstrating comparative analysis, not just raw performance.

Project Structure
.
├── Drug_example.ipynb        # End-to-end experiment notebook
├── Drug_utils.py             # GNN models, training, evaluation
├── utils_data_io.py          # Data loading and SMILES cache
├── utils_post_processing.py  # Metrics and helpers
├── cache/
│   └── smiles_cache.pkl
├── db_drug_interactions_2.csv
└── README.md

Technologies Used

Python

PyTorch

PyTorch Geometric

RDKit

scikit-learn

PubChem API

How to Run

Place db_drug_interactions_2.csv in the project directory.

Run Drug_example.ipynb from top to bottom.

SMILES will be fetched automatically (cached for future runs).

Training, evaluation, and baseline comparison will execute end-to-end.

Key Takeaways

Demonstrates graph-based learning on molecular data

Handles class imbalance properly

Includes strong baseline comparison

Designed for clarity, reproducibility, and academic rigor