**Description**

PyTorch Geometric is a library built on top of PyTorch that facilitates deep learning
on irregularly structured data, particularly graphs. It provides efficient
implementations of various graph neural network (GNN) architectures, along with data
loaders and utilities for handling graph-structured data.

Technologies Used
PyTorch Geometric

- Support for various graph neural network architectures (e.g., GCN, GAT)
- Efficient data handling for large graphs
- Predefined datasets and utilities for graph processing
- Built-in support for message passing and graph convolutions

### Fall2026

#### Tutorial

- Usual tutorial "Learn PyTorch Geometric in 60 mins", following
  `.claude/skills/tutorials_in_60_mins.rules.md`
- Check if there is an existing tutorial and make it better
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Deliverables:
  - `pytorch_geometric_utils.py`
  - `pytorch_geometric.API.ipynb`
  - `pytorch_geometric.example.ipynb`

#### Project: Fraud Detection in Financial Transactions

- **Project Objective**: The aim is to detect fraudulent transactions in a financial
  network by classifying transaction nodes, optimizing for precision and recall
  metrics
- **Dataset Suggestions**: Utilize the "Credit Card Fraud Detection" dataset from
  [Kaggle](https://www.kaggle.com/datasets/dalpozz/creditcard-fraud)
- **Tasks**:
  - **Construct the Graph**: Construct a transaction graph where nodes represent
    accounts and edges represent transactions
  - **Learn Node Embeddings**: Use node embeddings from a GraphSAGE model to capture
    transaction patterns
  - **Train a Classifier**: Train a classifier (e.g., logistic regression) on the
    embeddings to identify fraudulent transactions
  - **Evaluate the Model**: Evaluate the model using precision, recall, and F1 score
    metrics
- **Bonus Ideas (Optional)**: Implement a semi-supervised learning approach to
  improve fraud detection; analyze the model's performance on different transaction
  types
- TODO(ai_gp): Improve this

#### Milestones

- TODO(ai_gp): Add milestones related to the project

### Project 1: Social Network Analysis for Community Detection

- **Difficulty**: 1 (Easy)
- **Project Objective**: The goal is to identify and visualize communities within a
  social network graph by learning node embeddings and clustering them into groups,
  optimizing for modularity as a measure of community structure
- **Dataset Suggestions**: Use the "Facebook Social Network" dataset available on
  Kaggle
- **Tasks**:
  - Load and preprocess the social network graph using PyTorch Geometric utilities
  - Train a Graph Convolutional Network (GCN) to learn node embeddings
  - Apply a clustering algorithm (e.g., k-means) to group nodes into communities
  - Evaluate clustering quality using modularity and visualize communities in the
    graph
- **Bonus Ideas (Optional)**: Experiment with different GNN architectures (e.g., GAT)
  for embeddings; try other clustering methods (spectral clustering, DBSCAN) and
  compare results

### Project 3: Drug-Drug Interaction Prediction

- **Difficulty**: 3 (Hard)
- **Project Objective**: The goal is to predict potential interactions between drugs
  based on their molecular structures represented as graphs, optimizing for the
  accuracy of predictions
- **Dataset Suggestions**: Use the "Drug-Drug Interaction" dataset from
  [Kaggle](https://www.kaggle.com/datasets/andrews124/drug-drug-interaction)
- **Tasks**:
  - Transform molecular structures into graph representations using atom and bond
    information
  - Implement a Graph Neural Network (GNN) like Graph Attention Network (GAT) to
    learn drug representations
  - Train the model to predict interactions between drug pairs and evaluate using
    ROC-AUC scores
  - Analyze feature importance to understand which molecular features contribute to
    interactions
- **Bonus Ideas (Optional)**: Explore transfer learning with pre-trained GNN models
  on similar datasets; conduct a comparative analysis with traditional machine
  learning methods for interaction prediction
