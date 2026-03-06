# LlamaIndex Example: Real-World RAG Workflow

This example demonstrates a complete Retrieval-Augmented Generation (RAG) pipeline using real-world textual data.

## Workflow

1. Download public domain books from Project Gutenberg
2. Clean and preprocess the text
3. Split documents into semantic chunks
4. Build a vector index
5. Query across multiple documents
6. Inspect retrieved context
7. Compare retrieval depth (`top_k`)

## Purpose

The goal is to show how LlamaIndex can:

- Ground responses in external data
- Retrieve relevant passages from large corpora
- Enable cross-document semantic search
- Provide transparency through source inspection

See `llamaindex.example.ipynb` for the full runnable implementation.