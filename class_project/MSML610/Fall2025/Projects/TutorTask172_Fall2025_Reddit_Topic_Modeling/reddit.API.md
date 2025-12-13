# Reddit Topic Modeling API

## Overview

This document describes the internal **API layer** for the project  
**“Topic Modeling on Reddit Comments (r/worldnews)”** developed for **MSML610 – Fall 2025**.

The API is implemented in `reddit_utils.py` and provides reusable, modular building blocks for:

- Loading and preprocessing Reddit comment data
- Training **unsupervised fastText** embeddings
- Generating document-level embeddings
- Performing topic modeling using K-Means clustering
- Visualizing topics using t-SNE

This API is intentionally decoupled from notebooks.  
All notebooks act only as **consumers** of this API.

---

## Design Goals

- **Modularity**: Core logic is isolated from notebooks
- **Reproducibility**: Deterministic sampling and clustering
- **Scalability**: Efficient handling of thousands of comments
- **Clarity**: Simple interfaces aligned with fastText and sklearn APIs

---

## API Architecture

