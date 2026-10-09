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
- Create the project dir following the class instructions in
  `class_project/README.md`, section `Contribution to the Repo`
  - Start from `class_project/project_template`
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Deliverables:
  - `pytorch_geometric_utils.py`
  - `pytorch_geometric.API.ipynb`
  - `pytorch_geometric.example.ipynb`

# Project

## Project 1 (Fall2026): Sector Discovery in a Stock Correlation Graph

- **Project Objective**: Recover the market sectors of S&P 500 stocks by learning
  node embeddings on a return-correlation graph and clustering them, optimizing for
  the agreement with the GICS sectors (normalized mutual information) and for
  modularity
- **Dataset Suggestions**:
  - [S&P 500 constituents with GICS sectors](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
  - Daily adjusted prices of about 150 constituents, downloaded with
    [yfinance](https://github.com/ranaroussi/yfinance)
- **Tasks**:
  - **Build the Correlation Graph**: Compute daily log returns for 2018-2021, connect
    each stock to its 10 most correlated stocks, and store the graph as a
    `torch_geometric.data.Data` with per-stock return statistics as node features
  - **Learn Node Embeddings**: Train a two-layer `GCNConv` encoder without labels
    using `DeepGraphInfomax`
  - **Cluster the Nodes**: Run k-means with one cluster per sector on the embeddings,
    and build two baselines: k-means on the raw correlation rows and Louvain
    communities from `networkx`
  - **Evaluate the Communities**: Compute NMI and the adjusted Rand index against the
    GICS sectors, and the modularity of each partition on the 2022-2023 correlation
    graph to check stability out of time
  - **Visualize the Communities**: Draw the graph with a spring layout colored by
    cluster, and show the cluster-versus-sector table
- **Bonus Ideas (Optional)**: Compare `GCNConv` with `GATConv` encoders; use the
  correlation values as edge weights; find the stocks that change community between
  2020 and 2022

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Build the Correlation Graph
  - Result: project dir created and container running, and a table with the number
    of stocks, edges, and node features of the correlation graph, and the number of
    stocks per GICS sector
- Milestone 2: API notebook
  - Project tasks: Learn Node Embeddings
  - Result: `pytorch_geometric.API.ipynb` covering `Data` and `edge_index`, `GCNConv`
    message passing, and `DeepGraphInfomax` on a small built-in graph such as
    `KarateClub`
- Milestone 3: Example notebook
  - Project tasks: Learn Node Embeddings, Cluster the Nodes, Evaluate the
    Communities, Visualize the Communities
  - Result: `pytorch_geometric.example.ipynb` running end to end

## Project 2: Fraud Detection in Financial Transactions

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

## Project 3: Trade Network Link Prediction

- **Project Objective**: Predict which country pairs start to trade in the future by
  learning country embeddings on the bilateral trade graph, optimizing for ROC-AUC
  and average precision, and measure how much the graph structure adds over a
  gravity-style baseline
- **Dataset Suggestions**:
  - [Atlas of Economic Complexity, International Trade Data (SITC)](https://doi.org/10.7910/DVN/H8SFD2):
    `sitc_country_country_year.csv` (bilateral exports and imports) and
    `sitc_country_year.csv` (economic complexity and diversity per country)
  - [World Bank GDP per capita](https://data.worldbank.org/indicator/NY.GDP.PCAP.CD)
    for the country features
- **Tasks**:
  - **Build the Trade Graphs**: Build one country graph per year, with an edge when
    exports exceed 1M USD, and node features from the complexity table and the GDP
    per capita
  - **Define the Problem**: Split by time with a rolling origin: message passing on
    the 2008 graph to predict new links in 2012 (train), 2012 to 2016 (validation),
    and 2016 to 2019 (test), with sampled non-links as negatives
  - **Train the Link Predictor**: Train a `SAGEConv` encoder with a dot-product
    decoder using `LinkNeighborLoader`, and a `GATConv` variant for comparison
  - **Evaluate the Models**: Compute ROC-AUC, average precision, and Hits@100 over 5
    seeds (mean and standard deviation) against Adamic-Adar and a logistic regression
    on log GDP per capita product and common partners, and repeat on 2018 to 2022
    to test robustness to the COVID shock
  - **Analyze the Errors**: Plot the precision-recall curves, list the 20 most
    confident predicted new links, and compare the recall by income group
- **Bonus Ideas (Optional)**: Regress the log trade value of existing edges; build a
  country-product graph from `sitc_country_product_year_2.csv` as `HeteroData` and
  convert the model with `to_hetero`
