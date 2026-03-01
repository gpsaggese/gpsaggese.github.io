"""
Utility functions for Neo4j-based graph database workflows.

Import as:

import tutorials.tutorial_neo4j.neo4j_utils as ttneouti
"""

import logging
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import py2neo as pyneo

import helpers.hdbg as hdbg

_LOG = logging.getLogger(__name__)


# #############################################################################
# Connection utilities
# #############################################################################


def change_password(
    driver, current_password: str, new_password: str
) -> None:
    """
    Change the Neo4j database password.

    :param driver: active Neo4j driver instance
    :param current_password: the current password
    :param new_password: the desired new password
    """

    def _change_password(tx, current_password, new_password):
        tx.run(
            "ALTER CURRENT USER SET PASSWORD FROM $current_password TO $new_password",
            current_password=current_password,
            new_password=new_password,
        )

    with driver.session(database="system") as session:
        session.execute_write(_change_password, current_password, new_password)


def view_graph(graph: pyneo.Graph) -> None:
    """
    Print all nodes and relationships in the graph to the log.

    :param graph: py2neo Graph instance to inspect
    """
    nodes = graph.nodes.match()
    relationships = graph.relationships.match()
    _LOG.info("Nodes in the graph:")
    for node in nodes:
        _LOG.info(node)
    _LOG.info("Relationships in the graph:")
    for relationship in relationships:
        _LOG.info(relationship)


# #############################################################################
# Node creation
# #############################################################################


def create_person(tx, name: str) -> None:
    """
    Create a node with the label 'Person' and a name property.

    :param tx: active Neo4j transaction
    :param name: the name of the person
    """
    tx.run("CREATE (a:Person {name: $name})", name=name)


def create_node_with_label(tx, label: str, name: str) -> None:
    """
    Create a node with a specified label and a name property.

    :param tx: active Neo4j transaction
    :param label: label for the node (e.g., 'Employee')
    :param name: the name property value
    """
    tx.run(f"CREATE (a:{label} {{name: $name}})", name=name)


def create_node_with_multiple_labels(tx, labels: list, name: str) -> None:
    """
    Create a node with multiple labels and a name property.

    :param tx: active Neo4j transaction
    :param labels: list of label strings (e.g., ['Person', 'Employee'])
    :param name: the name property value
    """
    label_str = ":".join(labels)
    tx.run(f"CREATE (a:{label_str} {{name: $name}})", name=name)


def create_node_with_properties(tx, label: str, properties: dict) -> None:
    """
    Create a node with a specified label and multiple properties.

    :param tx: active Neo4j transaction
    :param label: label for the node
    :param properties: dictionary of property key-value pairs
    """
    props_str = ", ".join([f"{key}: ${key}" for key in properties.keys()])
    tx.run(f"CREATE (a:{label} {{{props_str}}})", **properties)


def return_created_node(tx, label: str, name: str):
    """
    Create a node and return the created node object.

    :param tx: active Neo4j transaction
    :param label: label for the node
    :param name: the name property value
    :return: the created Neo4j node object
    """
    result = tx.run(
        f"CREATE (a:{label} {{name: $name}}) RETURN a", name=name
    )
    return result.single()[0]


def clear_database(tx) -> None:
    """
    Delete all nodes and relationships from the database.

    :param tx: active Neo4j transaction
    """
    tx.run("MATCH (n) DETACH DELETE n")


# #############################################################################
# Relationship creation
# #############################################################################


def create_relationship(
    tx,
    node1_label: str,
    node1_name: str,
    relationship_type: str,
    node2_label: str,
    node2_name: str,
) -> None:
    """
    Create a directed relationship between two existing nodes.

    :param tx: active Neo4j transaction
    :param node1_label: label of the source node
    :param node1_name: name property of the source node
    :param relationship_type: type of relationship (e.g., 'KNOWS')
    :param node2_label: label of the target node
    :param node2_name: name property of the target node
    """
    tx.run(
        f"MATCH (a:{node1_label} {{name: $node1_name}}), "
        f"(b:{node2_label} {{name: $node2_name}}) "
        f"CREATE (a)-[:{relationship_type}]->(b)",
        node1_name=node1_name,
        node2_name=node2_name,
    )


def create_relationship_with_properties(
    tx,
    node1_label: str,
    node1_name: str,
    relationship_type: str,
    properties: dict,
    node2_label: str,
    node2_name: str,
) -> None:
    """
    Create a directed relationship with properties between two existing nodes.

    :param tx: active Neo4j transaction
    :param node1_label: label of the source node
    :param node1_name: name property of the source node
    :param relationship_type: type of relationship (e.g., 'WORKS_WITH')
    :param properties: dictionary of relationship property key-value pairs
    :param node2_label: label of the target node
    :param node2_name: name property of the target node
    """
    props_str = ", ".join([f"{key}: ${key}" for key in properties.keys()])
    tx.run(
        f"MATCH (a:{node1_label} {{name: $node1_name}}), "
        f"(b:{node2_label} {{name: $node2_name}}) "
        f"CREATE (a)-[:{relationship_type} {{{props_str}}}]->(b)",
        node1_name=node1_name,
        node2_name=node2_name,
        **properties,
    )


# #############################################################################
# Write clauses
# #############################################################################


def merge_node(tx, label: str, properties: dict) -> None:
    """
    Merge a node: create it if it doesn't exist, or match if it does.

    :param tx: active Neo4j transaction
    :param label: label for the node
    :param properties: dictionary of property key-value pairs
    """
    props_str = ", ".join([f"{key}: ${key}" for key in properties.keys()])
    tx.run(f"MERGE (a:{label} {{{props_str}}})", **properties)


def merge_relationship(
    tx,
    node1_label: str,
    node1_name: str,
    relationship_type: str,
    node2_label: str,
    node2_name: str,
) -> None:
    """
    Merge a relationship: create it if it doesn't exist, or match if it does.

    :param tx: active Neo4j transaction
    :param node1_label: label of the source node
    :param node1_name: name property of the source node
    :param relationship_type: type of relationship
    :param node2_label: label of the target node
    :param node2_name: name property of the target node
    """
    tx.run(
        f"MATCH (a:{node1_label} {{name: $node1_name}}), "
        f"(b:{node2_label} {{name: $node2_name}}) "
        f"MERGE (a)-[:{relationship_type}]->(b)",
        node1_name=node1_name,
        node2_name=node2_name,
    )


def set_properties(
    tx, label: str, name: str, properties: dict
) -> None:
    """
    Update properties of an existing node.

    :param tx: active Neo4j transaction
    :param label: label of the node to update
    :param name: name property of the node to update
    :param properties: dictionary of property key-value pairs to set
    """
    props_str = ", ".join([f"a.{key} = ${key}" for key in properties.keys()])
    tx.run(
        f"MATCH (a:{label} {{name: $name}}) SET {props_str}",
        name=name,
        **properties,
    )


def delete_node(tx, label: str, name: str) -> None:
    """
    Delete a node from the database.

    :param tx: active Neo4j transaction
    :param label: label of the node to delete
    :param name: name property of the node to delete
    """
    tx.run(f"MATCH (a:{label} {{name: $name}}) DELETE a", name=name)


def delete_relationship(
    tx,
    node1_label: str,
    node1_name: str,
    relationship_type: str,
    node2_label: str,
    node2_name: str,
) -> None:
    """
    Delete a relationship between two nodes.

    :param tx: active Neo4j transaction
    :param node1_label: label of the source node
    :param node1_name: name property of the source node
    :param relationship_type: type of relationship to delete
    :param node2_label: label of the target node
    :param node2_name: name property of the target node
    """
    tx.run(
        f"MATCH (a:{node1_label} {{name: $node1_name}})"
        f"-[r:{relationship_type}]->"
        f"(b:{node2_label} {{name: $node2_name}}) DELETE r",
        node1_name=node1_name,
        node2_name=node2_name,
    )


# #############################################################################
# Read clauses
# #############################################################################


def find_all_nodes(tx) -> None:
    """
    Find and log all nodes in the database.

    :param tx: active Neo4j transaction
    """
    result = tx.run("MATCH (n) RETURN n")
    for record in result:
        _LOG.info(record[0])


def find_relations(tx, name: str) -> None:
    """
    Find and log all employees that a given employee works with.

    :param tx: active Neo4j transaction
    :param name: name of the employee to query
    """
    result = tx.run(
        "MATCH (a:Employee {name: $name})-[:WORKS_WITH]->(Employee) "
        "RETURN Employee.name ORDER BY Employee.name",
        name=name,
    )
    record = result.single()
    _LOG.info(record)


def optional_match(tx) -> None:
    """
    Use OPTIONAL MATCH to find persons and who they know (including unconnected).

    :param tx: active Neo4j transaction
    """
    result = tx.run(
        "OPTIONAL MATCH (a:Person)-[r:KNOWS]->(b:Person) "
        "RETURN a.name, b.name"
    )
    for record in result:
        _LOG.info("%s knows %s", record["a.name"], record["b.name"])


def where_clause(tx) -> None:
    """
    Find all Person nodes where age is greater than 25.

    :param tx: active Neo4j transaction
    """
    result = tx.run(
        "MATCH (a:Person) WHERE a.age > 25 RETURN a.name, a.age"
    )
    for record in result:
        _LOG.info("%s is %s years old", record["a.name"], record["a.age"])


def count_function(tx) -> None:
    """
    Count and log the total number of Person nodes in the database.

    :param tx: active Neo4j transaction
    """
    result = tx.run("MATCH (a:Person) RETURN COUNT(a) as count")
    record = result.single()
    _LOG.info("Total number of Person nodes: %s", record["count"])


# #############################################################################
# Visualization
# #############################################################################


def plot_graph(results) -> None:
    """
    Visualize a Neo4j query result as a directed graph with NetworkX.

    :param results: iterable of Neo4j records with 'from', 'to', and 'rel' keys
    """
    G = nx.DiGraph()
    for record in results:
        G.add_node(record["from"])
        G.add_node(record["to"])
        G.add_edge(record["from"], record["to"], label=record["rel"])
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 8))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="skyblue",
        node_size=2000,
        edge_color="gray",
        font_size=15,
        font_weight="bold",
    )
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_color="red", font_size=12
    )
    plt.title("Neo4j Graph Visualization")
    plt.show()


# #############################################################################
# Neo4jAPI class
# #############################################################################


class Neo4jAPI:
    """
    A wrapper class for interacting with a Neo4j database.

    Encapsulates common Neo4j operations including data loading, querying,
    and visualization without requiring direct Cypher query writing.
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        """
        Initialize the Neo4jAPI instance.

        :param uri: URI of the Neo4j database (e.g., 'neo4j://localhost:7687')
        :param user: username for authentication
        :param password: password for authentication
        """
        import neo4j as nj

        hdbg.dassert_isinstance(uri, str)
        hdbg.dassert_isinstance(user, str)
        hdbg.dassert_isinstance(password, str)
        self.driver = nj.GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        """
        Close the connection to the Neo4j database.
        """
        self.driver.close()

    def run_query(self, query: str, *, parameters: Optional[dict] = None):
        """
        Run a Cypher query on the Neo4j database.

        :param query: Cypher query string to execute
        :param parameters: optional dictionary of query parameters
        :return: Neo4j Result object
        """
        with self.driver.session() as session:
            return session.run(query, parameters)

    def load_data(self, dataframe: pd.DataFrame) -> None:
        """
        Load movie-director data from a Pandas DataFrame into Neo4j.

        Uses MERGE to avoid creating duplicate nodes or relationships.

        :param dataframe: DataFrame with columns 'title', 'release_year', 'director'
        """
        query = """
        MERGE (movie:Movie {title: $title, release_year: $release_year})
        MERGE (director:Director {name: $director})
        MERGE (director)-[:DIRECTED]->(movie)
        """
        with self.driver.session() as session:
            for _, row in dataframe.iterrows():
                if pd.isna(row["title"]) or pd.isna(row["director"]):
                    continue
                params = {
                    "title": row["title"],
                    "release_year": row["release_year"],
                    "director": row["director"],
                }
                session.run(query, params)

    def visualize_graph(self) -> None:
        """
        Fetch the Director-Movie graph from Neo4j and visualize it.

        Directors are shown as blue nodes, movies as green nodes, with
        directed edges labeled 'DIRECTED'. Uses a spring layout for spacing.
        """
        query = """
        MATCH (d:Director)-[r:DIRECTED]->(m:Movie)
        WHERE d.name <> 'Unknown'
        RETURN d.name AS director, m.title AS movie, m.release_year AS year
        """
        with self.driver.session() as session:
            results = list(session.run(query))
        G = nx.DiGraph()
        for record in results:
            director = record["director"]
            movie = record["movie"]
            year = record["year"]
            G.add_node(director, label="Director", type="Director")
            G.add_node(movie, label=f"{movie} ({year})", type="Movie")
            G.add_edge(director, movie, relationship="DIRECTED")
        pos = nx.spring_layout(G, seed=42, k=0.5)
        plt.figure(figsize=(15, 10))
        director_nodes = [
            n for n, attr in G.nodes(data=True) if attr["type"] == "Director"
        ]
        movie_nodes = [
            n for n, attr in G.nodes(data=True) if attr["type"] == "Movie"
        ]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=director_nodes,
            node_color="skyblue",
            node_size=800,
            label="Director",
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=movie_nodes,
            node_color="lightgreen",
            node_size=500,
            label="Movie",
        )
        nx.draw_networkx_edges(
            G, pos, arrowstyle="->", arrowsize=15, edge_color="gray"
        )
        nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")
        edge_labels = nx.get_edge_attributes(G, "relationship")
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_size=8
        )
        plt.legend(scatterpoints=1, loc="upper right", fontsize=10)
        plt.title("Movie-Director Graph", fontsize=16)
        plt.axis("off")
        plt.show()
