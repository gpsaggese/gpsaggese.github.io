# Description

NetworkX is a pure-Python library to create, manipulate, and analyze graphs and
complex networks. It solves the problem of computing structural properties of
relational data, e.g., paths, centrality, communities, and link scores, without
writing the graph algorithms by hand. It is worth a 60-minute tutorial because the
same small API covers social, financial, and trade networks, and it is the base for
the graph libraries used in machine learning.

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
- Create the project dir following the class instructions in
  `class_project/README.md`, section `Contribution to the Repo`
  - Start from `class_project/project_template`
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

## Project 1 (Fall2026): Link Prediction in a Stock Correlation Network

- **Project Objective**: Predict which pairs of stocks become strongly correlated
  next year from the structure of the network of the current year, and check if the
  structural scores add information beyond the sector and the past correlation of
  the pair
- **Dataset Suggestions**:
  - [S&P 500 Constituents](https://github.com/datasets/s-and-p-500-companies), for
    the tickers and the GICS sectors
  - Daily prices from [yfinance](https://github.com/ranaroussi/yfinance), for 100
    random constituents and two consecutive years, e.g., 2024 and 2025
- **Tasks**:
  - **Build the Graph**: Compute the daily log returns of the first year, link the
    10% of the pairs with the highest correlation in an `nx.Graph` with the GICS
    sector as node attribute, and report the nodes, edges, density, and degree
    distribution
  - **Define the Problem**: Build the same graph for the second year, and label each
    pair not linked in the first year as positive if it is linked in the second year
  - **Compute Link Scores**: Compute the Jaccard coefficient, Adamic-Adar index, and
    preferential attachment of every candidate pair on the first-year graph only
  - **Detect Communities**: Run `louvain_communities` on the first-year graph,
    compare the communities with the sectors using NMI, and use the same-community
    flag as an extra score
  - **Evaluate the Scores**: Report ROC-AUC and average precision of each score,
    next to a random score, the same-sector flag, and the first-year correlation of
    the pair as baselines
  - **Visualize the Network**: Draw the first-year graph colored by sector and sized
    by degree, and highlight the correctly predicted new links
- **Bonus Ideas (Optional)**: Fit a logistic regression on the scores and the
  sector flag, and check if it beats the best single score; build the minimum
  spanning tree of the distance `sqrt(2 * (1 - correlation))` and compare its hubs
  with the degree hubs

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Build the Graph
  - Result: project dir created and container running, and the table of graph
    statistics with the degree histogram of the first-year graph
- Milestone 2: API notebook
  - Project tasks: Compute Link Scores, Detect Communities
  - Result: `networkx.API.ipynb` covering `Graph` and `DiGraph`, shortest paths,
    centrality, community detection, the link prediction scores, and drawing
- Milestone 3: Example notebook
  - Project tasks: Define the Problem, Compute Link Scores, Detect Communities,
    Evaluate the Scores, Visualize the Network
  - Result: `networkx.example.ipynb` running end to end

## Project 2: Trade Hubs and Regional Blocs in the World Trade Network

- **Project Objective**: Find the hub countries of the world trade network, and
  check if the detected trade blocs match the World Bank regions
- **Dataset Suggestions**:
  - [UN Comtrade API](https://comtradedeveloper.un.org/), using the public preview
    endpoint for the 2022 total exports of each country
  - [World Bank - GDP in current US$](https://api.worldbank.org/v2/country/all/indicator/NY.GDP.MKTP.CD?format=json&date=2022&per_page=300)
  - [World Bank - Countries and Regions](https://api.worldbank.org/v2/country?format=json&per_page=300)
  - [COW Bilateral Trade](https://correlatesofwar.org/data-sets/bilateral-trade/),
    as an offline fallback with data until 2014
- **Tasks**:
  - **Build the Graph**: Take the 60 largest economies by GDP, keep the top 5
    export destinations of each among them, and load the flows in an `nx.DiGraph`
    with the export value as edge weight and the GDP and region as node attributes
  - **Compute Centralities**: Compute the in-strength, out-strength, betweenness
    with distance `1 / weight`, and weighted PageRank of every country
  - **Compare the Rankings**: Report the Spearman correlation of each centrality
    with the GDP, and the overlap of the top 10 countries of each ranking with the
    top 10 by GDP as baseline
  - **Detect Blocs**: Run `greedy_modularity_communities` and `louvain_communities`
    on the undirected weighted graph, and record the modularity
  - **Evaluate the Blocs**: Compare the communities with the World Bank regions
    using NMI and the adjusted Rand index, against the same scores for 100 random
    shuffles of the region labels
  - **Draw the Network**: Draw the graph with Matplotlib, colored by community and
    sized by PageRank
- **Bonus Ideas (Optional)**: Repeat the analysis for 2012 and measure how many
  countries change bloc; weight each edge by the share of the total exports of the
  exporter instead of the dollar value

## Project 3: Contagion in the Global Banking Network

- **Project Objective**: Measure how far the default of one country spreads through
  the cross-border bank claims, find which centrality predicts the systemic
  countries, and compare the network with random graphs
- **Dataset Suggestions**:
  - [BIS Locational Banking Statistics](https://data.bis.org/topics/LBS), for the
    cross-border claims of one quarter, e.g., 2024-Q4, through the
    [BIS data API](https://stats.bis.org/api-doc/v2/) with the dataflow
    `WS_LBS_D_PUB`
  - [World Bank - GDP in current US$](https://api.worldbank.org/v2/country/all/indicator/NY.GDP.MKTP.CD?format=json&date=2023&per_page=300)
- **Tasks**:
  - **Build the Network**: Download the claims of one quarter, drop the aggregate
    country codes, and build a weighted `nx.DiGraph` from the creditor to the debtor
    country with the GDP as node attribute
  - **Characterize the Network**: Compute the density, reciprocity, average
    clustering, and the top 10 creditors and debtors by weighted degree
  - **Simulate the Cascade**: Default one country, apply a loss of 50% on the claims
    on it, and fail each creditor whose losses exceed 2%, 5%, or 10% of its total
    claims, until no country fails, using every country as the seed once
  - **Test the Centralities**: Report the Spearman correlation between the cascade
    size of each seed and its in-strength, PageRank, betweenness, and GDP
  - **Compare with Null Models**: Repeat the cascades on `gnm_random_graph` and
    `barabasi_albert_graph` graphs with the same size and the observed weights
    shuffled over the edges, over 20 runs each
  - **Report the Fragility**: Plot the mean cascade size vs. the loss threshold for
    the observed and null networks, with the standard deviation over the runs
- **Bonus Ideas (Optional)**: Compare 2007-Q4 with 2024-Q4 and measure how the
  cascade sizes changed; add the trade flows of the world trade network as a second
  edge type in a `MultiDiGraph`
