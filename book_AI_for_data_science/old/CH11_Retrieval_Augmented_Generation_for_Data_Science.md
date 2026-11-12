# Chapter 11: Retrieval-Augmented Generation for Data Science

## 1. RAG Fundamentals

### 1.1 The Problem RAG Solves

- LLMs are trained on general web data; they lack:
  - Knowledge of private/internal datasets and documentation
  - Up-to-date information beyond their training cutoff
  - Domain-specific knowledge (e.g., proprietary feature definitions, internal
    APIs, company data dictionaries)
- RAG solution: at inference time, retrieve relevant context from a knowledge
  base and inject it into the LLM prompt
  - LLM uses the retrieved context to answer questions that it could not
    answer from training data alone

### 1.2 RAG Pipeline Architecture

- Indexing phase (offline):
  1. Chunk documents into passages (typically 256-512 tokens)
  2. Embed each chunk into a dense vector using an embedding model
  3. Store vectors in a vector database (FAISS, Weaviate, Chroma, Pinecone)
- Retrieval phase (online):
  1. Embed the user query
  2. Search the vector database for the K most similar chunks
  3. Inject retrieved chunks into the prompt context
  4. Generate the answer using the LLM

### 1.3 Retrieval Strategies

- **Dense retrieval**: embedding similarity search; strong for semantic matching
- **Sparse retrieval (BM25)**: keyword-based; strong for exact term matching
- **Hybrid retrieval**: combine dense and sparse scores; best of both worlds
- Reranking: use a cross-encoder to rerank the top-K candidates retrieved by
  a cheaper first-stage retriever

- TUTORIAL: LlamaIndex - Build a basic RAG system over a corpus of data science
  papers; index the documents, retrieve relevant chunks, and generate answers
  with an LLM; evaluate retrieval precision

- TUTORIAL: Haystack - Implement a document QA pipeline with Haystack using
  dense retrieval and a generative reader; compare BM25 vs. embedding-based
  retrieval on a domain-specific corpus

- TUTORIAL: sentence-transformers - Build a semantic similarity search engine
  for data science notebooks using sentence-transformers embeddings; compare
  retrieval quality with and without fine-tuning on domain data

---

## 2. Domain Knowledge Injection

### 2.1 Why Domain Knowledge Matters

- General LLMs struggle with:
  - Dataset-specific questions: "What does the column `clv_90d` mean in our
    customer table?"
  - Internal API documentation: "How do I use the internal feature store API?"
  - Company-specific business rules: "What is our definition of an active user?"
- RAG injects this knowledge at inference time without retraining the LLM
- Alternatives to RAG:
  - Fine-tuning: expensive and requires labeled examples
  - Long context: stuff all documentation into the context; limited by context
    window and attention degradation at long range

### 2.2 Building a Data Science Knowledge Base

- Candidate knowledge sources for data scientists:
  - Data dictionaries and column definitions
  - Jupyter notebooks with analysis results
  - Pipeline documentation and runbooks
  - Past EDA reports and findings
  - Model cards and evaluation reports
- Chunking strategy: preserve notebook cell boundaries; keep related code and
  markdown cells together

### 2.3 Evaluation of RAG Systems

- Retrieval metrics: recall@K (did the relevant document appear in the top K?)
- Generation metrics:
  - Faithfulness: does the generated answer contradict the retrieved context?
  - Answer relevance: does the answer address the question?
- RAG evaluation frameworks: RAGAS, TruLens, DeepEval

- TUTORIAL: DocsGPT - Build a private knowledge base from internal data science
  documentation; demonstrate how domain-specific RAG outperforms a general LLM
  on dataset-specific questions

- TUTORIAL: txtai - Build a semantic search index over a collection of data
  science notebooks; use AI-powered ranking to surface the most relevant
  notebook for a given analysis task

- TUTORIAL: FastText - Train domain-specific word embeddings on a corpus of
  data science documentation using FastText; compare retrieval quality to
  sentence-transformers on internal queries

---

## 3. RAG for Code and SQL Generation

### 3.1 Code Generation with Context

- LLMs generate generic code; retrieval injects project-specific context:
  - Retrieve: similar functions from the existing codebase
  - Retrieve: relevant unit tests
  - Retrieve: pipeline conventions and coding standards
- Result: generated code matches the project's style and uses existing utilities
  instead of reinventing them

### 3.2 Text-to-SQL with RAG

- Text-to-SQL: generate SQL from natural language questions about a database
  - Challenge: LLM must know the schema, table relationships, and column
    semantics
  - RAG solution: retrieve relevant table definitions and example queries
    before generating SQL
- Schema-linking: identify which tables and columns are relevant to the question
  before generating the SQL; RAG handles this automatically

### 3.3 Code Review with RAG

- Augmented code review:
  - Retrieve similar code patterns from the codebase to find inconsistencies
  - Retrieve relevant test cases to suggest missing test coverage
  - Retrieve past bug reports to check if the new code has similar patterns

- TUTORIAL: LlamaIndex - Build a code retrieval system that indexes a codebase;
  generate SQL or Python code that matches the style of existing project code
  by retrieving relevant examples before generation

- TUTORIAL: Gensim - Build a topic model over a SQL schema documentation corpus;
  use the topics to route natural language queries to the most relevant table
  definitions before SQL generation

- TUTORIAL: spaCy - Use spaCy's dependency parsing to extract schema entities
  from natural language questions; use extracted entities to filter the schema
  search space before SQL generation

---

## 4. Knowledge Graphs and Structured Retrieval

### 4.1 Limitations of Vector Search for Relational Questions

- Vector search is powerful for semantic similarity but weak for:
  - Multi-hop relational questions: "Who is the manager of the team that owns
    dataset X?"
  - Constraint queries: "Find all datasets updated in the last 7 days with
    more than 1M rows"
  - Exact relationship lookup: "Which models use feature Y?"
- Knowledge graphs encode relationships explicitly; Cypher/SPARQL queries are
  precise

### 4.2 Knowledge Graph Construction from Tabular Data

- Extract entities (tables, columns, models, pipelines) from metadata
- Extract relationships (uses, produced_by, updated_at, owned_by) from
  pipeline logs and documentation
- Store in a graph database (Neo4j, TigerGraph)
- AI can extract entities and relationships from unstructured documentation

### 4.3 Hybrid Retrieval: Vectors + Graphs

- Best-of-both approach:
  1. Vector search for semantic similarity (find related concepts)
  2. Graph traversal for relational queries (follow relationships)
- LangChain + Neo4j: supports hybrid retrieval; routes queries to vector
  search or Cypher based on question type

- TUTORIAL: Langchain and Neo4j - Build a knowledge graph from a structured
  dataset; use LangChain to query the graph with natural language; demonstrate
  how structured retrieval outperforms vector search for relational questions

- TUTORIAL: Py2neo - Construct and query a Neo4j knowledge graph from tabular
  data; use AI to generate Cypher queries from natural language; build a
  schema-aware question answering system

- TUTORIAL: igraph - Build an in-memory data lineage graph using igraph;
  traverse the graph to find all datasets and models affected by an upstream
  data quality issue; use AI to generate the graph traversal queries
