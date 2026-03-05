# SNAP

## Description
- SNAP (Stanford Network Analysis Platform) is a comprehensive tool for
  analyzing and modeling large networks.
- It provides a rich set of functionalities for graph manipulation, network
  analysis, and visualization, making it suitable for various applications in
  social network analysis, bioinformatics, and more.
- SNAP supports both static and dynamic network data, allowing users to analyze
  time-varying relationships and structures.
- The platform is designed for scalability, capable of handling graphs with
  millions of nodes and edges efficiently.
- SNAP includes algorithms for community detection, centrality measures,
  clustering, and graph generation, which are essential for understanding
  complex networks.

## Project Objective
The goal of the project is to analyze a social network dataset to identify key
influencers and community structures within the network. Students will optimize
a community detection algorithm to classify nodes into different groups based on
their connections, and evaluate the effectiveness of their approach using
metrics like modularity and silhouette score.

## Dataset Suggestions
1. **Facebook Social Network Dataset**
   - **Source**: SNAP
   - **URL**: https://snap.stanford.edu/data/facebook_combined.txt.gz
   - **Data Contains**: Edges of a Facebook social network, representing
     friendships between users.
   - **Access Requirements**: Publicly accessible without authentication.

2. **Wiki-Vote Dataset**
   - **Source**: SNAP
   - **URL**: https://snap.stanford.edu/data/wiki-Vote.html
   - **Data Contains**: A directed network of votes among Wikipedia users,
     showing who voted for whom.
   - **Access Requirements**: Publicly accessible without authentication.

3. **Email Enron Dataset**
   - **Source**: SNAP
   - **URL**: https://snap.stanford.edu/data/email-Enron.html
   - **Data Contains**: An email communication network of Enron employees, with
     edges representing email exchanges.
   - **Access Requirements**: Publicly accessible without authentication.

4. **CA-GrQc Dataset**
   - **Source**: SNAP
   - **URL**: https://snap.stanford.edu/data/ca-GrQc.html
   - **Data Contains**: A collaboration network of scientific papers in the
     field of General Relativity and Quantum Cosmology.
   - **Access Requirements**: Publicly accessible without authentication.

## Tasks
- **Data Preprocessing**: Load and preprocess the selected dataset, ensuring it
  is in the correct format for analysis with SNAP.
- **Graph Construction**: Use SNAP to construct a graph from the dataset,
  visualizing the network to understand its structure.
- **Community Detection**: Implement and fine-tune community detection
  algorithms (e.g., Girvan-Newman, Louvain) to identify clusters within the
  network.
- **Influencer Identification**: Calculate centrality measures (e.g., degree,
  betweenness, closeness) to identify key influencers in the network.
- **Evaluation**: Evaluate the performance of the community detection algorithms
  using metrics like modularity and silhouette score, and analyze the results.

## Bonus Ideas
- Extend the project by comparing different community detection algorithms and
  their effectiveness on various datasets.
- Implement a visualization tool to dynamically display the network and its
  communities.
- Explore temporal aspects by analyzing how the network evolves over time if a
  dataset with timestamps is available.
- Conduct a sentiment analysis on the content of emails or messages exchanged in
  the network to see how it correlates with community structures.

## Useful Resources
1. [SNAP Official Documentation](http://snap.stanford.edu/snappy/index.html)
2. [SNAP GitHub Repository](https://github.com/snap-stanford/snap)
3. [Community Detection Algorithms in SNAP](http://snap.stanford.edu/snappy/doc/CommunityDetection.html)
4. [Stanford's Network Analysis Course](https://online.stanford.edu/courses/sohs-ystats1-statistics-network-analysis)
   (for foundational concepts)
5. [Visualization Techniques for Network Analysis](https://www.gephi.org/) (for
   additional visualization resources)
