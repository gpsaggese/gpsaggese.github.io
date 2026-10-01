# Description

NetworkX is a pure-Python library to create, manipulate, and analyze graphs and
complex networks. It solves the problem of computing structural properties of
relational data, e.g., paths, centrality, communities, and link scores, without
writing the graph algorithms by hand. It is worth a 60-minute tutorial because the
same small API covers social, transport, and communication networks, and it is the
base for the graph libraries used in machine learning.

## Technologies Used

NetworkX

- Directed, undirected, and multi graphs with node and edge attributes
- Algorithms for shortest paths, centrality, clustering, and community detection
- Link prediction scores such as `jaccard_coefficient`, `adamic_adar_index`, and
  `preferential_attachment`
- Graph generators and drawing with Matplotlib

# Tutorial

- Implement the tutorial "Learn NetworkX in 60 mins", following
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - Build it with `.claude/skills/tutorial_in_60_mins.create/SKILL.md`
  - Follow the workflow in `tutorials/README.gp.md` and the quality principles in
    `tutorials/tutorials_checklist.md`
- Check the previous tutorials and projects, listed in the section
  `Existing Tutorials and Projects` of `.claude/skills/tutorial_in_60_mins.rules.md`
  - Read the `Readme.md` of the Fall2025 NetworkX project, and reuse what is good
    - `class_project/msml610/Fall2025/projects/TutorTask26_Fall2025_NetworkX_Fraud_Detection_in_Financial_Transactions/`
  - Read the `README.md` of `tutorials/Neo4j/` for the graph database view
  - Read the Spring2025 Py2neo project, which models transaction networks
    - `class_project/data605/Spring2025/projects/TutorTask244_Spring2025_Modeling_Bitcoin_Transaction_Networks_with_Py2neo/`
- Create `tutorials/NetworkX/`, since it does not exist yet
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Compare briefly with igraph and SNAP from the performance point of view, e.g., time
  and memory for the same centrality algorithm
- Deliverables:
  - `networkx_utils.py`
  - `networkx.API.ipynb`
  - `networkx.example.ipynb`

# Project

## Project 1: Link Prediction in a Social Network

- **Difficulty**: 2 (Medium)
- **Project Objective**: Predict the missing friendships of a social network from its
  structure, and check if a learned model beats the single heuristic scores
- **Dataset Suggestions**:
  [SNAP - Facebook Ego Networks](https://snap.stanford.edu/data/egonets-Facebook.html)
- **Tasks**:
  - **Load the Graph**: Read the edge list into an `nx.Graph`, and report the nodes,
    edges, density, and degree distribution
  - **Define the Problem**: Hide 10% of the edges as positive test examples, and
    sample the same number of non-edges as negative examples
  - **Compute Link Features**: Compute the Jaccard coefficient, Adamic-Adar index,
    preferential attachment, and common neighbors on the training graph
  - **Train the Classifier**: Fit a logistic regression on the four features, and
    compare it with each single score
  - **Evaluate the Model**: Report ROC-AUC, precision, recall, and F1 on the held-out
    edges
  - **Visualize the Network**: Draw one ego network and highlight the correctly
    predicted links
- **Bonus Ideas (Optional)**: Add community membership as a feature; repeat the
  analysis on the timestamped email network to predict future links

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Load the Graph
  - Result: `tutorials/NetworkX/` container running, and the table of graph
    statistics with the degree histogram
- Milestone 2: API notebook
  - Project tasks: Compute Link Features
  - Result: `networkx.API.ipynb` covering `Graph` and `DiGraph`, shortest paths,
    centrality, community detection, the link prediction scores, and drawing
- Milestone 3: Example notebook
  - Project tasks: Define the Problem, Compute Link Features, Train the Classifier,
    Evaluate the Model, Visualize the Network
  - Result: `networkx.example.ipynb` running end to end

## Project 2: Influence and Communities in an Email Network

- **Difficulty**: 2 (Medium)
- **Project Objective**: Find the most influential people of an email network, and
  check if the detected communities match the departments
- **Dataset Suggestions**:
  [SNAP - email-Eu-core](https://snap.stanford.edu/data/email-Eu-core.html)
- **Tasks**:
  - **Build the Graph**: Load the edges as an `nx.DiGraph` and attach the department
    of each person as a node attribute
  - **Compute Centralities**: Compute degree, betweenness, closeness, and PageRank
    for every node
  - **Compare the Rankings**: Report the Spearman correlation between the rankings
    and the overlap of the top 20 people
  - **Detect Communities**: Run `greedy_modularity_communities` and
    `louvain_communities` on the undirected graph
  - **Evaluate the Communities**: Compare the communities with the departments using
    NMI and the adjusted Rand index
- **Bonus Ideas (Optional)**: Draw the graph colored by department and sized by
  PageRank

## Project 3: Robustness of the Airline Network

- **Difficulty**: 3 (Hard)
- **Project Objective**: Measure how fast the airline network breaks apart under
  random failures and under targeted attacks, and compare it with random graphs
- **Dataset Suggestions**: [OpenFlights Routes](https://openflights.org/data)
- **Tasks**:
  - **Build the Network**: Build a directed graph of airports from `routes.dat`, and
    keep the largest strongly connected component
  - **Characterize the Network**: Compute the degree distribution, the average
    shortest path length, and the clustering coefficient
  - **Simulate Random Failures**: Remove nodes at random, and record the size of the
    largest component after each removal, over 20 runs
  - **Simulate Targeted Attacks**: Remove nodes by degree and by betweenness, and
    record the same curve
  - **Compare with Null Models**: Repeat the attacks on `gnm_random_graph` and
    `barabasi_albert_graph` graphs with the same size
  - **Report the Robustness**: Compare the area under each curve, with the standard
    deviation over the runs
- **Bonus Ideas (Optional)**: Weight the edges by route frequency, and test if it
  changes the ranking of the attacks
