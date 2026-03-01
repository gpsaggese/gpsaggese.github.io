# Langchain and Neo4J

## Description
- **Langchain** is a framework designed for developing applications that
  leverage large language models (LLMs) by facilitating the integration of
  various components such as data retrieval, prompt management, and agent-based
  interactions.
- **Neo4j** is a graph database management system that enables the storage,
  retrieval, and manipulation of data in a graph format, making it ideal for
  applications that require complex relationships and connections.
- The combination of Langchain and Neo4j allows for building sophisticated
  applications that can query graph data using natural language, enabling users
  to interact with data intuitively.
- Key features include support for various data connectors, customizable prompt
  templates, and the ability to create agents that can perform tasks based on
  user queries and data relationships.

## Project Objective
The goal of this project is to develop a knowledge graph-based
question-answering system that can retrieve and present information from a
dataset using natural language queries. The project aims to optimize the
accuracy and relevance of the responses generated based on user input.

## Dataset Suggestions
1. **Kaggle: MovieLens Dataset**
   - **Source**: Kaggle
   - **URL**:
     [MovieLens](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)
   - **Data Contains**: Movie ratings, titles, genres, and user information.
   - **Access Requirements**: Free to use after creating a Kaggle account.

2. **Hugging Face Datasets: WikiMovies**
   - **Source**: Hugging Face
   - **URL**: [WikiMovies](https://huggingface.co/datasets/wikimovies)
   - **Data Contains**: Movie-related information including plots, genres, and
     cast.
   - **Access Requirements**: Free access without authentication.

3. **Open Government Data: European Union Open Data Portal**
   - **Source**: EU Open Data Portal
   - **URL**: [EU Open Data](https://data.europa.eu/en)
   - **Data Contains**: Various datasets, including cultural data and public
     services information.
   - **Access Requirements**: No authentication required, but some datasets may
     require browsing to find relevant data.

4. **GitHub: Open Movie Database**
   - **Source**: GitHub Repository
   - **URL**: [OMDb API](https://github.com/omdbapi/omdb)
   - **Data Contains**: Movie and TV show data, including descriptions, ratings,
     and release years.
   - **Access Requirements**: Free to use, with no authentication needed.

## Tasks
- **Data Ingestion**: Load the selected dataset into Neo4j, transforming it into
  a graph format that captures the relationships between entities (e.g., movies,
  actors, genres).
- **Graph Schema Design**: Design a schema that effectively represents the
  relationships in the dataset, including nodes for movies, users, and ratings.
- **Langchain Integration**: Set up Langchain to facilitate natural language
  queries to the Neo4j database, enabling users to ask questions about movies
  and receive relevant information.
- **Model Training/Fine-tuning**: If necessary, fine-tune a language model to
  improve the accuracy of responses based on the dataset context.
- **Evaluation and Optimization**: Test the system with various queries,
  evaluate the relevance and accuracy of the responses, and optimize the prompt
  design and graph structure as needed.

## Bonus Ideas
- Implement a recommendation engine using collaborative filtering based on user
  ratings to suggest movies to users.
- Explore the use of additional datasets to enrich the knowledge graph, such as
  actor biographies or box office performance.
- Challenge students to create a visual representation of the knowledge graph
  using Neo4j's built-in visualization tools.

## Useful Resources
- [Langchain Documentation](https://langchain.readthedocs.io/en/latest/)
- [Neo4j Documentation](https://neo4j.com/docs/)
- [Neo4j Python Driver](https://neo4j.com/docs/api/python-driver/current/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)
