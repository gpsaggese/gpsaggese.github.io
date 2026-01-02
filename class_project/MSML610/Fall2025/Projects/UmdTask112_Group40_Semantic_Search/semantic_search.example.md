# Semantic Search Web Application

## Overview

This is a complete web application that demonstrates semantic search over Wikipedia articles using Flask.

## What It Does

The application provides a web interface where users can:
1. Enter search queries in natural language
2. Receive semantically relevant Wikipedia articles
3. See similarity scores for each result

## Architecture
```
User Browser
    ↓
Flask Web Server (app.py)
    ↓
SemanticSearchEngine (semantic_search_utils.py)
    ↓
Sentence Transformers Model
    ↓
Wikipedia Data (data_sample.parquet)
```

## Key Components

1. **Backend API** (`semantic_search_utils.py`)
   - Handles document indexing
   - Performs semantic search
   - Returns ranked results

2. **Web Server** (`app.py`)
   - Flask application
   - HTML templating
   - Query handling

3. **Data** (`data_sample.parquet`)
   - 5,000 Wikipedia articles
   - Pre-processed and cleaned

## How to Run

See README.md for detailed instructions.

## Example Queries

- "famous tower in Paris" → Finds Eiffel Tower articles
- "red planet" → Finds Mars articles  
- "artificial intelligence" → Finds AI/ML articles