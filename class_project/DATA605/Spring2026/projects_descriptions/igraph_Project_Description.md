# Igraph

## Description
- **igraph** is a powerful library for creating, manipulating, and visualizing
  complex networks and graphs in Python and R.
- It provides a rich set of functionalities for network analysis, including
  algorithms for community detection, shortest paths, and centrality measures.
- The library supports various input formats, enabling users to create graphs
  from adjacency matrices, edge lists, or directly from data frames.
- Igraph includes tools for visualizing networks with customizable layouts,
  colors, and sizes, making it easier to interpret complex relationships.
- It is highly efficient and can handle large graphs with millions of edges and
  nodes, making it suitable for big data applications.

## Project Objective
The goal of this project is to analyze social network data to detect communities
within a network and predict the likelihood of connections between users based
on their characteristics. Students will optimize the community detection process
and evaluate the accuracy of their predictions.

## Dataset Suggestions
1. **Facebook Social Network Dataset**
   - **Source**: SNAP (Stanford Network Analysis Project)
   - **URL**:
     [Facebook Social Network](http://snap.stanford.edu/data/ego-Facebook.html)
   - **Data Contains**: Information about user connections (edges) and user
     attributes (nodes).
   - **Access Requirements**: Free to download without authentication.

2. **Twitter Social Network Dataset**
   - **Source**: Kaggle
   - **URL**:
     [Twitter User Network](https://www.kaggle.com/datasets/benroshan/twitter-user-network)
   - **Data Contains**: User relationships, tweets, and user attributes.
   - **Access Requirements**: Free to download with a Kaggle account.

3. **Karate Club Dataset**
   - **Source**: NetworkX (available on GitHub)
   - **URL**:
     [Karate Club Dataset](https://github.com/networkx/networkx/blob/main/networkx/generators/social/karate_club.py)
   - **Data Contains**: A social network of friendships between members of a
     karate club.
   - **Access Requirements**: Directly available through the NetworkX library.

4. **Enron Email Dataset**
   - **Source**: Carnegie Mellon University
   - **URL**: [Enron Email Dataset](https://www.cs.cmu.edu/~enron/)
   - **Data Contains**: Email communications between Enron employees, which can
     be transformed into a social network.
   - **Access Requirements**: Free to download without authentication.

## Tasks
- **Data Preparation**: Load the selected dataset and preprocess it to create a
  graph representation using igraph.
- **Community Detection**: Implement community detection algorithms (e.g.,
  Louvain or Girvan-Newman) to identify clusters within the network.
- **Feature Engineering**: Extract relevant features from the graph (e.g., node
  degree, clustering coefficient) to use in the prediction model.
- **Connection Prediction**: Develop a machine learning model (e.g., logistic
  regression or random forest) to predict the likelihood of new connections
  based on user attributes and graph features.
- **Model Evaluation**: Assess the performance of the prediction model using
  appropriate metrics (e.g., accuracy, precision, recall) and visualize the
  results.
- **Visualization**: Create visualizations of the network and the detected
  communities to facilitate interpretation and presentation of findings.

## Bonus Ideas
- Implement additional community detection algorithms and compare their
  effectiveness.
- Explore the impact of different features on the prediction accuracy and
  perform feature selection.
- Extend the analysis to include temporal dynamics by examining how communities
  evolve over time.
- Create an interactive visualization using libraries like Plotly or Bokeh to
  allow users to explore the network dynamically.

## Useful Resources
- [igraph Official Documentation](https://igraph.org/python/)
- [SNAP Datasets](http://snap.stanford.edu/data/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [NetworkX Documentation](https://networkx.org/documentation/stable/)
- [Machine Learning with igraph](https://igraph.org/python/doc/tutorial/tutorial.html#machine-learning)
