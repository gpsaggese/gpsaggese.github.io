# Semantic Search API Documentation

## Overview

This API provides semantic search capabilities using Sentence Transformers. It allows you to index text documents and search them by meaning rather than keyword matching.

## Core Classes

### `SearchResult`

A dataclass representing a single search result.

**Attributes:**
- `index` (int): Position of the result in the original document collection
- `score` (float): Cosine similarity score (0-1, higher is better)
- `text` (str): The actual document text

### `SemanticSearchEngine`

The main API class for performing semantic search.

**Methods:**

#### `__init__(model_name: str = "all-MiniLM-L6-v2")`
Initialize the search engine with a specific model.

**Parameters:**
- `model_name`: Name of the SentenceTransformer model to use

#### `load_model() -> None`
Load the sentence transformer model into memory.

#### `index_documents(texts: List[str]) -> None`
Index a collection of documents for searching.

**Parameters:**
- `texts`: List of text strings to index

**Side effects:**
- Generates and stores embeddings for all documents
- This is a one-time operation per document collection

#### `search(query: str, top_k: int = 5) -> List[SearchResult]`
Search indexed documents for matches to the query.

**Parameters:**
- `query`: Search query string  
- `top_k`: Number of results to return (default: 5)

**Returns:**
- List of `SearchResult` objects, ordered by relevance (highest first)

**Raises:**
- `ValueError`: If no documents have been indexed

## Helper Functions

### `load_wikipedia_data(file_path: str, max_articles: int = None) -> List[str]`

Load Wikipedia articles from a parquet file.

**Parameters:**
- `file_path`: Path to the parquet file
- `max_articles`: Optional limit on number of articles to load

**Returns:**
- List of article texts

## Design Decisions

### Why Sentence Transformers?
- Pre-trained for semantic similarity tasks
- Fast inference (suitable for real-time search)
- No fine-tuning required

### Why Cosine Similarity?
- Scale-invariant (document length doesn't affect results)
- Interpretable (0-1 scale)
- Computationally efficient

### Why This API Structure?
- **Separation of concerns**: Model loading, indexing, and searching are separate operations
- **Flexibility**: Users can swap models without changing code
- **Performance**: One-time indexing, fast repeated searches

## Usage Example
```python
from semantic_search_utils import SemanticSearchEngine

# Initialize
engine = SemanticSearchEngine()

# Index documents
documents = ["Paris is the capital of France", 
             "The Eiffel Tower is in Paris"]
engine.index_documents(documents)

# Search
results = engine.search("famous tower in Paris", top_k=2)

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Text: {result.text}")
```