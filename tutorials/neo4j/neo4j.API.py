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
# # Neo4j API Overview
#
# Neo4j is a graph database management system designed to handle large-scale,
# highly interconnected data. It enables users to model data as **nodes**
# (entities) and **relationships** (connections) with associated properties.
#
# This notebook covers:
# - Setting up and connecting to a Neo4j server
# - Creating nodes with labels and properties
# - Creating relationships between nodes
# - Write clauses: MERGE, SET, DELETE
# - Read clauses: MATCH, OPTIONAL MATCH, WHERE, COUNT
# - Visualizing a graph with NetworkX
#
# ![ER diagram example](artifacts/mermaid-diagram-2025-03-27-035437.png)

# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

# %%
import logging

import neo4j as nj
import py2neo as pyneo

import helpers.hdbg as hdbg
import helpers.hprint as hprint
import tutorials.tutorial_neo4j.neo4j_utils as ttneouti

hdbg.init_logger(verbosity=logging.INFO)
_LOG = logging.getLogger(__name__)
hprint.config_notebook()

# %% [markdown]
# ## 1. Starting the Neo4j Server
#
# Neo4j uses the following default ports:
# - `7474`: HTTP port for Neo4j Browser and REST API
# - `7687`: Bolt protocol port for database queries
# - `7473`: HTTPS port (optional)
#
# Start the server with:
# ```bash
# sudo neo4j start
# ```

# %%
# Start the Neo4j server inside the Docker container.
# !sudo neo4j start

# %% [markdown]
# ## 2. Connecting to the Server
#
# Use the Bolt protocol URI and authentication credentials to connect.
#
# Key functions:
# - `GraphDatabase.driver(URI, auth=(USER, PASSWORD))`: creates the driver
# - `driver.verify_connectivity()`: verifies the connection
#
# ![Connection diagram](artifacts/mermaid-diagram-2025-03-27-034817.png)

# %%
# URI and authentication details.
URI = "neo4j://localhost:7687"
USER = "neo4j"
PASSWORD = "neo4j"

# Create a driver instance.
driver = nj.GraphDatabase.driver(URI, auth=(USER, PASSWORD))
driver.verify_connectivity()
_LOG.info("Connection established.")

# %% [markdown]
# ## 3. Updating the Password
#
# The default credentials (`neo4j/neo4j`) must be changed on first use.
# Once updated, the change is permanent unless you do a clean reinstallation.
#
# Steps:
# 1. Run the ALTER CURRENT USER Cypher command via `execute_write`
# 2. Reconnect with the new password

# %%
# Change the default password.
ttneouti.change_password(driver, "neo4j", "new_password")

# Reconnect with the new password.
driver = nj.GraphDatabase.driver(URI, auth=("neo4j", "new_password"))
driver.verify_connectivity()
_LOG.info("Connection established with new password.")

# Connect to the graph using py2neo for inspection.
graph = pyneo.Graph(URI, auth=(USER, "new_password"))

# %% [markdown]
# ## 4. Creating Nodes
#
# A **node** is a fundamental unit of data in Neo4j. It can have:
# - **Labels**: categorize the node (e.g., `Person`, `Employee`)
# - **Properties**: key-value pairs (e.g., `name`, `age`, `city`)
#
# The `CREATE` statement adds new nodes; `tx.run()` executes queries within
# a transaction.

# %%
with driver.session() as session:
    # Create a simple Person node.
    session.execute_write(ttneouti.create_person, "Dave")
    # Create an Employee node.
    session.execute_write(ttneouti.create_node_with_label, "Employee", "Grace")
    # Create a node with both Person and Employee labels.
    session.execute_write(
        ttneouti.create_node_with_multiple_labels, ["Person", "Employee"], "Hank"
    )
    # Create a Person with multiple properties.
    session.execute_write(
        ttneouti.create_node_with_properties,
        "Person",
        {"name": "Ivy", "age": 28, "city": "New York"},
    )
    # Create a Person and get back the created node.
    created_node = session.execute_write(
        ttneouti.return_created_node, "Person", "Jack"
    )
_LOG.info("Created node: %s", created_node)

# View all nodes and relationships.
ttneouti.view_graph(graph)

# %%
# Clear the database before the next example.
with driver.session() as session:
    session.execute_write(ttneouti.clear_database)

# %% [markdown]
# ## 5. Creating Relationships Between Nodes
#
# Relationships connect two nodes with a directed edge and a type
# (e.g., `KNOWS`, `WORKS_WITH`). They can also carry properties.
#
# Pattern:
# ```cypher
# MATCH (a:Person {name: $node1_name}), (b:Person {name: $node2_name})
# CREATE (a)-[:KNOWS]->(b)
# ```
#
# **Example graph:**
# - `Jack` --[:KNOWS]--> `Dave`
# - `Grace` --[:WORKS_WITH {since: 2020}]--> `Hank`
#
# ![Relationship graph](artifacts/mermaid-diagram-2025-03-27-035900.png)

# %%
with driver.session() as session:
    # Recreate nodes for the relationships demo.
    session.execute_write(ttneouti.create_person, "Dave")
    session.execute_write(ttneouti.create_person, "Jack")
    session.execute_write(ttneouti.create_node_with_label, "Employee", "Grace")
    session.execute_write(
        ttneouti.create_node_with_multiple_labels, ["Person", "Employee"], "Hank"
    )
    # Create a simple KNOWS relationship.
    session.execute_write(
        ttneouti.create_relationship, "Person", "Jack", "KNOWS", "Person", "Dave"
    )
    # Create a WORKS_WITH relationship with a 'since' property.
    session.execute_write(
        ttneouti.create_relationship_with_properties,
        "Employee",
        "Grace",
        "WORKS_WITH",
        {"since": 2020},
        "Employee",
        "Hank",
    )
_LOG.info("Relationships created.")
ttneouti.view_graph(graph)

# %% [markdown]
# ## 6. Write Clauses
#
# ### MERGE
# Ensures the node or relationship exists:
# - If it exists, it is matched
# - If it doesn't exist, it is created
#
# ### SET
# Updates properties of a node or relationship.
#
# ### DELETE
# Removes nodes or relationships.

# %%
with driver.session() as session:
    # Create Alice and Bob.
    session.execute_write(
        ttneouti.create_node_with_properties,
        "Person",
        {"name": "Alice", "age": 30},
    )
    session.execute_write(
        ttneouti.create_node_with_properties,
        "Person",
        {"name": "Bob", "age": 25},
    )
    # Create a KNOWS relationship.
    session.execute_write(
        ttneouti.create_relationship, "Person", "Alice", "KNOWS", "Person", "Bob"
    )
    # MERGE: add Charlie if not present.
    session.execute_write(
        ttneouti.merge_node, "Person", {"name": "Charlie", "age": 25}
    )
    # MERGE: add Alice-Charlie relationship.
    session.execute_write(
        ttneouti.merge_relationship, "Person", "Alice", "KNOWS", "Person", "Charlie"
    )
    # SET: update Alice's age and add city.
    session.execute_write(
        ttneouti.set_properties,
        "Person",
        "Alice",
        {"age": 31, "city": "New York"},
    )
    _LOG.info("Graph before deletion:")
    ttneouti.view_graph(graph)

    # DELETE: remove Alice-Bob relationship.
    session.execute_write(
        ttneouti.delete_relationship,
        "Person",
        "Alice",
        "KNOWS",
        "Person",
        "Bob",
    )
    # DELETE: remove Bob node.
    session.execute_write(ttneouti.delete_node, "Person", "Bob")
    _LOG.info("Graph after deletion:")
    ttneouti.view_graph(graph)

# %% [markdown]
# ## 7. Read Clauses
#
# ### MATCH
# Retrieves nodes, relationships, or paths matching a pattern.
# ```cypher
# MATCH (n) RETURN n
# ```
#
# ### OPTIONAL MATCH
# Like MATCH but includes nodes with no matching relationships (returns null).
#
# ### WHERE
# Filters results by conditions.
# ```cypher
# MATCH (a:Person) WHERE a.age > 25 RETURN a.name, a.age
# ```
#
# ### COUNT
# Aggregates by counting nodes, relationships, or paths.

# %%
with driver.session() as session:
    # Find and print all nodes.
    session.execute_read(ttneouti.find_all_nodes)
    # Find who Grace works with.
    session.execute_read(ttneouti.find_relations, "Grace")
    # Optional match: persons and who they know.
    session.execute_read(ttneouti.optional_match)
    # Where clause: persons older than 25.
    session.execute_read(ttneouti.where_clause)
    # Count all Person nodes.
    session.execute_read(ttneouti.count_function)

# %% [markdown]
# ## 8. Clean Up

# %%
driver.close()
_LOG.info("Connection closed.")
