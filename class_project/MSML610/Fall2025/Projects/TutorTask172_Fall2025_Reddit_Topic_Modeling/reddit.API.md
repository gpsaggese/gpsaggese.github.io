# Reddit Topic Modeling API

This document describes the internal API used for the Reddit Topic Modeling project.

## Core Functions (from reddit_utils.py)
1. **clean_text(text)** – Tokenizes and removes stopwords/punctuation.
2. **generate_embeddings(texts)** – Uses fastText to get text embeddings.
3. **run_kmeans(vectors, n_clusters)** – Clusters embeddings to discover topics.
4. **visualize_tsne(vectors, labels)** – Visualizes clusters using t-SNE.

These abstractions make it easier to reuse the same NLP pipeline for different subreddits.
