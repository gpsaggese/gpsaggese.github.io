# Description

PyTorch Geometric is a library built on top of PyTorch for deep learning on
irregularly structured data, particularly graphs. It solves the problem of training
graph neural networks (GNNs) efficiently, with implementations of common GNN layers,
mini-batch loaders for large graphs, and ready-to-use graph datasets. It is worth a
60-minute tutorial because the message-passing abstraction lets a student go from a
raw graph to a trained node classifier in a few dozen lines.

## Technologies Used

PyTorch Geometric

- Support for various graph neural network architectures (e.g., GCN, GAT)
- Efficient data handling for large graphs
- Predefined datasets and utilities for graph processing
- Built-in support for message passing and graph convolutions

# Tutorial

- Implement the tutorial "Learn PyTorch Geometric in 60 mins", following
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - Build it with `.claude/skills/tutorial_in_60_mins.create/SKILL.md`
  - Follow the workflow in `tutorials/README.gp.md` and the quality principles in
    `tutorials/tutorials_checklist.md`
- Check the previous tutorials and projects, listed in the section
  `Existing Tutorials and Projects` of
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - Read the `README.md` of the Fall2025 PyTorch Geometric project, and reuse what
    is good
    - `class_project/msml610/Fall2025/projects/UmdTask23_Fall2025_PyTorch_Geometric_Drug_Drug_Interaction_Prediction/`
  - Read the `README.md` of the Fall2025 DGL and NetworkX fraud detection projects
    for the related graph tools
    - `class_project/msml610/Fall2025/projects/UmdTask88_Fall2025_DGL_Fraud_Detection_in_Credit_Card_Transactions/`
    - `class_project/msml610/Fall2025/projects/TutorTask26_Fall2025_NetworkX_Fraud_Detection_in_Financial_Transactions/`
- Create `tutorials/PyTorch_Geometric/`, since it does not exist yet
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Deliverables:
  - `pytorch_geometric_utils.py`
  - `pytorch_geometric.API.ipynb`
  - `pytorch_geometric.example.ipynb`

# Project

## Project 1: Fraud Detection in Financial Transactions

- **Difficulty**: 2 (Medium)
- **Project Objective**: Detect illicit transactions in a financial transaction graph
  by classifying transaction nodes, optimizing for precision and recall on the
  illicit class, and measure how much the graph structure adds over a features-only
  baseline
- **Dataset Suggestions**:
  [Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set),
  loaded with `torch_geometric.datasets.EllipticBitcoinDataset`
- **Tasks**:
  - **Load the Transaction Graph**: Load the graph, inspect nodes, edges, features,
    and labels, and mask the nodes with unknown labels
  - **Define the Problem**: Classify each transaction as illicit or licit, train on
    the early time steps and test on the later ones, and train a features-only
    logistic regression baseline
  - **Train GraphSAGE**: Train a GraphSAGE model built with `SAGEConv` and
    `NeighborLoader`, and fit a logistic regression on the learned node embeddings
  - **Evaluate the Models**: Compute precision, recall, F1, and the precision-recall
    AUC on the illicit class for the baseline, GraphSAGE, and the embedding
    classifier
  - **Visualize the Embeddings**: Project the node embeddings to 2D with t-SNE
    colored by label, and plot the precision-recall curves of the models
- **Bonus Ideas (Optional)**: Use the unlabeled nodes with a semi-supervised
  approach; build a k-nearest-neighbor graph on the tabular
  [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
  dataset and compare with Elliptic

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Load the Transaction Graph
  - Result: `tutorials/PyTorch_Geometric/` container running, and a table with the
    number of nodes, edges, and labels per time step
- Milestone 2: API notebook
  - Project tasks: Train GraphSAGE
  - Result: `pytorch_geometric.API.ipynb` covering `Data`, built-in datasets,
    `GCNConv` and `SAGEConv` message passing, and `NeighborLoader` on a small graph
- Milestone 3: Example notebook
  - Project tasks: Define the Problem, Train GraphSAGE, Evaluate the Models,
    Visualize the Embeddings
  - Result: `pytorch_geometric.example.ipynb` running end to end

## Project 2: Social Network Analysis for Community Detection

- **Difficulty**: 1 (Easy)
- **Project Objective**: Identify and visualize communities within a social network
  graph by learning node embeddings and clustering them into groups, optimizing for
  modularity as a measure of community structure
- **Dataset Suggestions**:
  [Facebook Social Circles (ego-Facebook)](https://snap.stanford.edu/data/ego-Facebook.html)
- **Tasks**:
  - **Load the Graph**: Load and preprocess the social network graph using PyTorch
    Geometric utilities
  - **Learn Node Embeddings**: Train a Graph Convolutional Network (GCN) to learn
    node embeddings
  - **Cluster the Nodes**: Apply a clustering algorithm (e.g., k-means) to group
    nodes into communities
  - **Evaluate the Communities**: Evaluate clustering quality using modularity and
    visualize communities in the graph
- **Bonus Ideas (Optional)**: Experiment with different GNN architectures (e.g., GAT)
  for embeddings; try other clustering methods (spectral clustering, DBSCAN) and
  compare results

## Project 3: Drug-Drug Interaction Prediction

- **Difficulty**: 3 (Hard)
- **Project Objective**: Predict potential interactions between drugs based on their
  molecular structures represented as graphs, optimizing for the accuracy of
  predictions
- **Dataset Suggestions**:
  [TDC DrugBank DDI](https://tdcommons.ai/multi_pred_tasks/ddi/) (drug pairs with
  SMILES strings)
- **Tasks**:
  - **Build Molecular Graphs**: Transform the SMILES of each drug into a graph with
    atom and bond features, e.g., with `torch_geometric.utils.from_smiles`
  - **Build Labeled Pairs**: Take the interacting pairs as positives and sample
    non-interacting pairs as negatives
  - **Learn Drug Representations**: Implement a Graph Attention Network (`GATConv`)
    to learn a representation of each drug, and train a pair classifier on top
  - **Evaluate the Model**: Evaluate the predictions using ROC-AUC and
    precision-recall AUC
  - **Analyze Feature Importance**: Analyze which molecular features contribute to
    interactions
- **Bonus Ideas (Optional)**: Explore transfer learning with pre-trained GNN models
  on similar datasets; conduct a comparative analysis with traditional machine
  learning methods for interaction prediction
