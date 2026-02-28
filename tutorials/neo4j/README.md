# Neo4j Tutorial

- This folder contains the setup for running a Neo4j graph database tutorial
  within a containerized environment

## Quick Start
- From the root of the repository, change your directory to the Neo4j tutorial
  folder:
  ```bash
  > cd tutorials/tutorial_neo4j
  ```

- Build the Docker image:
  ```bash
  > ./docker_build.sh
  ```

- Launch Jupyter Lab in the container:
  ```bash
  > ./docker_jupyter.sh
  ```

- Once `./docker_jupyter.sh` is running, follow this sequence:
  1. **`neo4j.API.ipynb`**: Start here to learn the Neo4j native API —
     connecting to the server, creating nodes and relationships, and using
     Cypher write/read clauses.
  2. **`neo4j.example.ipynb`**: Proceed to this notebook for a complete
     application that builds a Movie-Director graph from the Netflix dataset
     and visualizes it with NetworkX.

- For more information on the Docker build system refer to [Project template
  readme](https://github.com/gpsaggese/umd_classes/blob/master/class_project/project_template/README.md)
