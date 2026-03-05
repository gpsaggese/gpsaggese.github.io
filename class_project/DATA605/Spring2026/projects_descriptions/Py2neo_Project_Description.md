# Py2Neo

## Description
- **Graph Database Interface**: Py2neo is a client library for interacting with
  the Neo4j graph database, allowing users to work with graph data structures in
  Python effortlessly.
- **Cypher Query Language**: It supports Cypher, Neo4j's query language,
  enabling users to perform complex queries and manipulations on graph data.
- **Object-Graph Mapping**: Py2neo provides an object-graph mapping (OGM)
  feature that simplifies the process of converting between Python objects and
  graph nodes/relationships.
- **Data Visualization**: It includes utilities for visualizing graph data,
  making it easier to understand relationships and data structures.
- **Flexible and Extensible**: Py2neo is designed to be flexible and extensible,
  supporting various Neo4j features and allowing developers to build custom
  solutions.

## Project Objective
The goal of this project is to analyze a social network dataset to identify
influential nodes (users) within the network. Students will utilize machine
learning techniques to predict the likelihood of a user becoming an influencer
based on their connections and interactions.

## Dataset Suggestions
1. **Dataset Name**: Facebook Social Network Dataset
   - **Source**: SNAP (Stanford Network Analysis Project)
   - **URL**: [Facebook Dataset](http://snap.stanford.edu/data/facebook.html)
   - **Data Contains**: User relationships and interactions in a Facebook-like
     social network.
   - **Access Requirements**: Publicly available for download without
     authentication.

2. **Dataset Name**: Twitter Social Network Dataset
   - **Source**: Kaggle
   - **URL**:
     [Twitter Dataset](https://www.kaggle.com/datasets/kazemnejad/facebooksocialnetwork)
   - **Data Contains**: User connections, tweets, and retweet interactions.
   - **Access Requirements**: Free to use with a Kaggle account (no paid plans
     required).

3. **Dataset Name**: Reddit Comment Dataset
   - **Source**: Kaggle
   - **URL**:
     [Reddit Dataset](https://www.kaggle.com/datasets/benhamner/reddit-comments-may-2015)
   - **Data Contains**: Comments and user interactions from Reddit.
   - **Access Requirements**: Publicly accessible with a free Kaggle account.

4. **Dataset Name**: GitHub Social Network Dataset
   - **Source**: GitHub Archive
   - **URL**: [GitHub Archive](https://www.gharchive.org/)
   - **Data Contains**: User interactions, repositories, and contribution
     activities.
   - **Access Requirements**: Publicly available and can be accessed via direct
     download.

## Tasks
- **Data Ingestion**: Load the chosen dataset into Neo4j using Py2neo, creating
  nodes for users and relationships for interactions.
- **Data Exploration**: Use Py2neo to perform exploratory analysis on the graph
  structure, identifying key metrics such as node degree and clustering
  coefficients.
- **Feature Engineering**: Construct features from the graph data, such as
  centrality measures (e.g., degree, closeness, betweenness) to predict user
  influence.
- **Model Training**: Implement a machine learning model (e.g., logistic
  regression or a decision tree) to classify users as potential influencers
  based on engineered features.
- **Model Evaluation**: Evaluate the model's performance using metrics such as
  accuracy, precision, recall, and F1-score, and visualize the results.
- **Visualization**: Create visual representations of the graph to illustrate
  the relationships and highlight influential users.

## Bonus Ideas
- **Community Detection**: Implement algorithms to detect communities within the
  network and analyze how community structure affects influence.
- **Temporal Analysis**: Investigate how user influence changes over time by
  analyzing interactions across different time periods.
- **Comparative Analysis**: Compare the performance of different machine
  learning models (e.g., SVM, Random Forest) on the influencer prediction task.
- **Graph Algorithms**: Explore advanced graph algorithms (e.g., PageRank) to
  identify influential users and compare results with the machine learning
  approach.

## Useful Resources
- [Py2neo Documentation](https://py2neo.org/v4/)
- [Neo4j Graph Data Science](https://neo4j.com/docs/graph-data-science/current/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Stanford Network Analysis Project (SNAP)](http://snap.stanford.edu/)
- [GitHub Archive](https://www.gharchive.org/)
