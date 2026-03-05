# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Neo4j: Building a Movie-Director Graph
#
# This notebook demonstrates a complete end-to-end application that builds a
# graph database of movies and directors from the Netflix dataset, then
# visualizes the relationships.
#
# **Workflow:**
# 1. Load the Netflix CSV dataset with Pandas
# 2. Clean and preprocess the data
# 3. Populate a Neo4j graph database using the `Neo4jAPI` wrapper
# 4. Visualize the director-movie graph with NetworkX
#
# **Why graphs for this?**
# Graphs are an intuitive way to represent relationships. Each movie is
# connected to its director, forming a natural web. Using Neo4j, we can query
# and explore this data efficiently, and by visualizing it we reveal connections
# that might otherwise be hard to see in rows and columns.

# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

# %%
import logging

import pandas as pd

import helpers.hdbg as hdbg
import helpers.hprint as hprint
import tutorials.tutorial_neo4j.neo4j_utils as ttneouti

hdbg.init_logger(verbosity=logging.INFO)
_LOG = logging.getLogger(__name__)
hprint.config_notebook()

# %% [markdown]
# ## Part 1: Data Loading and Inspection
#
# Load the Netflix dataset from `artifacts/data/netflix.csv`. The dataset
# contains movie and TV show metadata including title, director, cast, country,
# and release year.
#
# Sample structure:
# ```
# | title        | release_year | director      | cast          | country |
# |--------------|--------------|---------------|---------------|---------|
# | Movie Title  | 2021         | John Doe      | Actor A, ...  | USA     |
# ```

# %%
# Load the dataset into a Pandas DataFrame.
csv_file = "artifacts/data/netflix.csv"
data = pd.read_csv(csv_file)
_LOG.info("Dataset shape: %s", data.shape)
data.head()

# %% [markdown]
# ## Part 2: Data Cleaning and Preprocessing
#
# Before loading into Neo4j we need clean, unambiguous data:
# - Fill empty `cast` fields with empty strings (cast is optional)
# - Drop rows with missing `director` or `country` (required for graph edges)
# - Strip whitespace from `title` to avoid duplicate nodes

# %%
# Fill missing cast values.
data["cast"] = data["cast"].fillna("")
# Remove rows with missing director or country.
data = data.dropna(subset=["director", "country"])
# Strip whitespace from title to avoid duplicates.
data["title"] = data["title"].str.strip()
_LOG.info("Cleaned dataset shape: %s", data.shape)
data.head()

# %% [markdown]
# ## Part 3: Graph Construction with Neo4j
#
# The `Neo4jAPI` wrapper class handles connection management and data loading.
# It uses `MERGE` instead of `CREATE` to prevent duplicate nodes:
# ```cypher
# MERGE (movie:Movie {title: $title, release_year: $release_year})
# MERGE (director:Director {name: $director})
# MERGE (director)-[:DIRECTED]->(movie)
# ```
#
# This creates:
# - **Movie nodes** with `title` and `release_year` properties
# - **Director nodes** with a `name` property
# - **[:DIRECTED]** relationships from director to movie

# %%
# Start the Neo4j server.
# !sudo neo4j start

# %%
# Initialize the Neo4j API wrapper.
neo4j_api = ttneouti.Neo4jAPI(
    uri="neo4j://localhost:7687",
    user="neo4j",
    password="new_password",
)

# %%
# Load the first 40 rows into Neo4j (lightweight demo).
neo4j_api.load_data(data[:40])
_LOG.info("Data loaded into Neo4j.")

# %% [markdown]
# ## Part 4: Graph Visualization
#
# Query the database and render the Director → Movie graph using NetworkX:
# ```cypher
# MATCH (d:Director)-[r:DIRECTED]->(m:Movie)
# WHERE d.name <> 'Unknown'
# RETURN d.name AS director, m.title AS movie, m.release_year AS year
# ```
#
# Visualization style:
# - **Blue nodes**: Directors
# - **Green nodes**: Movies (with release year)
# - **Arrows**: DIRECTED relationships
# - **Layout**: Spring layout for optimal spacing

# %%
# Generate the visualization.
neo4j_api.visualize_graph()

# %% [markdown]
# ## Clean Up

# %%
# Close the Neo4j connection.
neo4j_api.close()
_LOG.info("Connection closed.")
