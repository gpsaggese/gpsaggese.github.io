**Description**

In this project, students will utilize SNAP (Stanford Network Analysis Platform), a general-purpose network analysis and graph mining library. SNAP is designed for the analysis of large networks and provides an extensive set of algorithms for graph processing. Its features include:

- Efficient algorithms for graph processing, including clustering, centrality measures, and community detection.
- Support for large-scale networks with millions of nodes and edges.
- Various utilities for graph generation, manipulation, and visualization.

---

### Project 1: Social Network Analysis (Difficulty: 1)

**Project Objective**: Analyze a social network dataset to identify influential users and community structures within the network.

**Dataset Suggestions**: 
- Facebook Social Network (available on SNAP's website): A dataset containing user interactions and connections.
- Alternatively, use the "Twitter Social Graph" dataset available on Kaggle.

**Tasks**:
- Load and Preprocess Data: Import the dataset into SNAP and clean it for analysis.
- Identify Influential Users: Use centrality measures (e.g., degree centrality, betweenness centrality) to find key influencers in the network.
- Community Detection: Implement clustering algorithms (e.g., Louvain method) to discover communities within the network.
- Visualization: Create visual representations of the network using SNAP's visualization tools to illustrate the communities and influential users.

### Bonus Ideas (Optional):
- Compare different centrality measures and their effectiveness in identifying influencers.
- Explore the impact of removing certain influential nodes on the network's connectivity.

---

### Project 2: Fraud Detection in Financial Transactions (Difficulty: 2)

**Project Objective**: Build a model to detect fraudulent transactions in a financial transaction network using graph-based features.

**Dataset Suggestions**: 
- Credit Card Fraud Detection Dataset (available on Kaggle): Contains transactions labeled as fraudulent or legitimate.
- Alternatively, the "Enron Email Dataset" can be used to analyze communication patterns for fraud detection.

**Tasks**:
- Data Preparation: Transform the transaction data into a graph format where nodes represent accounts and edges represent transactions.
- Feature Engineering: Extract features such as transaction frequency, average transaction amount, and graph-based features like clustering coefficients.
- Model Training: Implement a machine learning model (e.g., Random Forest or SVM) to classify transactions as fraudulent or legitimate based on the engineered features.
- Evaluation: Assess the model's performance using metrics such as precision, recall, and F1-score.

### Bonus Ideas (Optional):
- Implement additional graph-based algorithms for anomaly detection.
- Compare the performance of traditional machine learning models with graph neural networks.

---

### Project 3: Recommendation System Using Graphs (Difficulty: 3)

**Project Objective**: Develop a recommendation system for movies using collaborative filtering techniques based on user-item interaction graphs.

**Dataset Suggestions**: 
- MovieLens 100K Dataset (available on Kaggle): Contains user ratings for a variety of movies.
- Alternatively, use the "Last.fm Dataset" for music recommendations.

**Tasks**:
- Graph Construction: Create a bipartite graph where users and movies are nodes, and edges represent user ratings.
- Similarity Measures: Implement algorithms to compute user and item similarity based on graph metrics (e.g., Jaccard index, cosine similarity).
- Recommendation Generation: Use collaborative filtering techniques to generate movie recommendations for users based on their neighbors in the graph.
- Evaluation: Validate the effectiveness of the recommendation system using metrics like Mean Absolute Error (MAE) or Root Mean Square Error (RMSE).

### Bonus Ideas (Optional):
- Integrate content-based filtering to enhance recommendations.
- Explore the impact of incorporating temporal dynamics in user preferences on the recommendation outcomes.

