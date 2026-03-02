# LlamaIndex API Overview

LlamaIndex is a framework for building retrieval-augmented generation (RAG) systems on top of large language models.

This tutorial demonstrates the core abstractions:

- Model configuration via `Settings`
- Document ingestion with `SimpleDirectoryReader`
- Node parsing (chunking)
- Vector index construction
- Query engine orchestration
- Prompt customization

The API design separates:

- Embedding models (for semantic search)
- Language models (for generation)
- Index structures (for retrieval)
- Query engines (for orchestration)

The corresponding notebook `llamaindex.API.ipynb` provides runnable examples of each component.