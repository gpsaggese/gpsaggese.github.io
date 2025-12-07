<!-- toc -->

* [DGL Example — GNN-Based Movie Recommendation System](#dgl-example--gnn-based-movie-recommendation-system)

  * [Overview](#overview)
  * [Problem Statement](#problem-statement)
  * [End-to-End Pipeline Summary](#end-to-end-pipeline-summary)
  * [Graph Representation](#graph-representation)

    * [Base Model: Bipartite User–Movie Graph](#base-model-bipartite-user–movie-graph)
    * [Bonus Model: Knowledge-Aware Heterograph (Users–Movies–Tags)](#bonus-model-knowledge-aware-heterograph-users–movies–tags)
  * [Core Components in `dgl_utils.py`](#core-components-in-dgl_utilspy)

    * [1. Loading and Sampling MovieLens](#1-loading-and-sampling-movielens)
    * [2. Preprocessing and ID Remapping](#2-preprocessing-and-id-remapping)
    * [3. Train/Val/Test Temporal Splitting](#3-trainvaltest-temporal-splitting)
    * [4. Graph Construction](#4-graph-construction)
    * [5. Dataloaders & Negative Sampling](#5-dataloaders--negative-sampling)
    * [6. Full-Ranking Evaluation](#6-full-ranking-evaluation)
  * [Model Architecture](#model-architecture)

    * [Bipartite GraphSAGE Encoder](#bipartite-graphsage-encoder)
    * [Heterogeneous GraphSAGE Encoder (Bonus)](#heterogeneous-graphsage-encoder-bonus)
    * [MLP Link Predictor](#mlp-link-predictor)
  * [How to Run](#how-to-run)
  * [Results & Observations](#results--observations)

    * [Bipartite Model Performance](#bipartite-model-performance)
    * [Heterograph Model Performance](#heterograph-model-performance)
  * [Design Choices](#design-choices)
  * [Limitations & Future Work](#limitations--future-work)
  * [References](#references)

<!-- tocstop -->

# DGL Example — GNN-Based Movie Recommendation System

## Overview

This project uses the **MovieLens 20M dataset** (Kaggle link: [https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset/data?select=rating.csv](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset/data?select=rating.csv)), a large real-world dataset widely used for benchmarking recommender systems.

From this dataset we use exactly **four files**:

* `rating.csv` — explicit user ratings with timestamps
* `movie.csv` — movie metadata including titles and pipe-separated genres
* `genome_scores.csv` — movie–tag relevance scores (0–1)
* `genome_tags.csv` — mapping of `tagId` → human-readable tag text

These files must be placed inside:

```
data/raw/
```

This notebook builds a complete **GNN-based recommendation system** using DGL on top of this dataset and includes an optional semantic extension using Genome Tags.

---

This example notebook (`DGL.example.ipynb`) demonstrates how to build a **graph-based movie recommendation system** using the **Deep Graph Library (DGL)** and a **GraphSAGE encoder**.

The recommender is treated as a **link prediction** problem on a graph:

* **Nodes**: users, movies, and optionally semantic tags
* **Edges**: interactions or semantic associations
* **Goal**: predict missing edges (i.e., whether a user would watch/like a movie)

The notebook is intentionally written as a **thin tutorial**: all heavy logic such as preprocessing, graph construction, dataloading, evaluation, and tag‑graph generation is delegated to `dgl_utils.py`.

The result is a clean, readable walkthrough of a modern GNN recommender system with an optional knowledge-aware extension.

---

## Problem Statement

Traditional collaborative filtering approaches (matrix factorization, heuristics, embeddings) fail to exploit the explicit **graph structure** in user–item interactions and struggle to integrate rich metadata.

Our goals:

1. Convert MovieLens data into a **heterogeneous graph**
2. Build a **GraphSAGE-based encoder** that learns from graph neighborhoods
3. Train a binary **link-prediction head** to score user–movie edges
4. Evaluate using realistic **full-ranking metrics**
5. Extend the model with **MovieLens Genome Tags** to add semantics

This creates an explainable, inductive, and extensible recommendation framework.

---

## End-to-End Pipeline Summary

The notebook implements the following pipeline:

1. Load raw MovieLens ratings and movies
2. Sample users for computational efficiency
3. Preprocess and map raw IDs → dense indices
4. Expand movie genres into multi-hot vectors
5. Create temporal train/val/test splits for each user
6. Generate negative samples
7. Build a DGL bipartite graph
8. Define a GraphSAGE encoder + MLP predictor
9. Train using BCEWithLogitsLoss
10. Evaluate using Precision@K, Recall@K, NDCG@K
11. Produce personalized recommendations
12. **Bonus:** load genome tags, build tag edges
13. **Bonus:** construct a user–movie–tag heterograph
14. **Bonus:** train a heterogeneous GNN recommender

---

## Graph Representation

### Base Model: Bipartite User–Movie Graph

The foundational graph is a simple bipartite structure:

```
(user) --rates--> (movie)
(movie) --rev_rates--> (user)
```

Movie nodes include:

* Multi-hot genre vectors

User nodes include:

* Learnable embeddings

This representation supports efficient message passing using GraphSAGE.

---

### Bonus Model: Knowledge-Aware Heterograph (Users–Movies–Tags)

Genome tags enrich the graph with semantic context.

Nodes:

* `user`
* `movie`
* `tag`

Edges:

* `user --rates--> movie`
* `movie --has_tag--> tag`
* Reverse edges for message passing

Relevance scores from MovieLens Genome Tags are filtered (≥ 0.8) to keep only meaningful associations.

This enables **semantic smoothing**, where preference information passes through tag nodes.

---

## Core Components in `dgl_utils.py`

Below is the complete pipeline **with the exact function used in each step**.

### 1. Loading and Sampling MovieLens

Functions used:

* **`load_movielens_data()`** → loads `rating.csv` and `movie.csv`
* **`sample_users()`** → downsamples user base and filters by minimum interactions

These functions ensure the dataset is correctly loaded, validated, and trimmed for faster experimentation.

---

### 2. Preprocessing and ID Remapping

Functions used:

* **`build_id_mappings()`** → creates `userId → user_idx` and `movieId → movie_idx`
* **`apply_id_mappings()`** → applies mapped IDs to ratings
* **`explode_genres()`** → expands pipe-separated genres into multi-hot vectors
* **`attach_movie_indices()`** → adds dense movie indices to movie table
* **`preprocess_and_save()`** → orchestrates all preprocessing and saves:

  * `users.csv`
  * `movies.csv`
  * `ratings.csv`

This prepares numerical IDs and feature matrices needed by the GNN.

---

### 3. Train/Val/Test Temporal Splitting

Functions used:

* **`make_splits()`** → main driver
* **`_split_user_interactions()`** → time-based splitting for each user
* **`_generate_negative_samples_for_split()`** → generates negative edges for each split

Outputs six files:

* `train_pos.csv`, `train_neg.csv`
* `val_pos.csv`, `val_neg.csv`
* `test_pos.csv`, `test_neg.csv`

---

### 4. Graph Construction

Functions used:

* **`build_graph()`** → constructs bipartite user–movie DGL graph
* **`build_hetero_graph()`** *(bonus)* → constructs heterograph with user, movie, tag nodes
* **`build_movie_tag_edges()`** *(bonus)* → prepares movie–tag edge pairs
* **`load_genome_tags()`** *(bonus)* → loads tag metadata files

Graph types:

* **Bipartite:** user ↔ movie edges
* **Heterograph:** user ↔ movie ↔ tag edges

---

### 5. Dataloaders & Negative Sampling

Functions used:

* **`build_dataloader()`** → creates PyTorch `DataLoader` for training

Produces batches of:

```
(user_idx, movie_idx, label)
```

---

### 6. Full-Ranking Evaluation

Functions used:

* **`evaluate_full_ranking()`** → computes Precision@K, Recall@K, NDCG@K
* **`precision_at_k()`**, `recall_at_k()`, `ndcg_at_k()` → metric utilities
* **`recommend_for_user()`**, `show_recommendations_with_scores()` → qualitative demo

This evaluates the recommender in a real-world ranking setup by scoring **all movies** for each user.

---

## Model Architecture

### Bipartite GraphSAGE Encoder

* Learnable user embeddings
* Movie genres projected into feature space
* Multi-layer GraphSAGE aggregation
* Residual connections
* BatchNorm + Dropout

This forms the backbone of the bipartite recommender.

### Heterogeneous GraphSAGE Encoder (Bonus)

* Adds learnable tag embeddings
* Adds relations:

  * `rates`
  * `rev_rates`
  * `has_tag`
  * `tag_of`
* Message passing incorporates semantic structure

### MLP Link Predictor

Concatenates:

```
[user_emb || movie_emb]
```

Through:

* Linear → BatchNorm → LeakyReLU → Dropout → Linear

Outputs a logit for link prediction.

---

## How to Run

### 1. Place Required Files in `data/raw/`

The following four MovieLens CSVs must be placed in:

```
data/raw/
```

Files:

* `rating.csv`
* `movie.csv`
* `genome_scores.csv`
* `genome_tags.csv`

The notebook will fail gracefully if any required file is missing.

---

### 2. Run the Notebook in Order

`DGL.example.ipynb` is written top–down. Execute each cell sequentially.

Pipeline summary:

1. Load raw data → `load_movielens_data()`
2. Sample users → `sample_users()`
3. Preprocess & save → `preprocess_and_save()`
4. Create temporal splits → `make_splits()`
5. Build bipartite graph → `build_graph()`
6. Train GNN model → training loop in notebook
7. Evaluate ranking metrics → `evaluate_full_ranking()`
8. **Bonus:** load tag files → `load_genome_tags()`
9. Build tag edges → `build_movie_tag_edges()`
10. Build heterograph → `build_hetero_graph()`
11. Train hetero GNN model
12. Final evaluation

---

### 3. Adjustable Parameters and What They Control

| Parameter                                                 | Defined In                | Meaning                                              |
| --------------------------------------------------------- | ------------------------- | ---------------------------------------------------- |
| `n_users`                                                 | `sample_users()`          | How many users to sample for the tutorial dataset    |
| `min_ratings_per_user`                                    | `sample_users()`          | Filters out users with too few ratings               |
| `train_ratio`, `val_ratio`, `test_ratio`                  | `make_splits()`           | Controls temporal splitting proportions              |
| `num_neg_per_pos`                                         | `make_splits()`           | Number of negative edges generated per positive edge |
| `batch_size`                                              | `build_dataloader()`      | Batch size during training                           |
| `encoder_hidden_dim`, `encoder_out_dim`, `encoder_layers` | model definition          | Controls GNN depth & embedding size                  |
| `lp_hidden_dim`, `lp_dropout`                             | model definition          | Defines MLP link predictor structure                 |
| `epochs`                                                  | training loop             | Total iterations for training                        |
| `relevance_threshold`                                     | `build_movie_tag_edges()` | Controls strength of movie–tag associations          |

These parameters allow the user to tune both the computational cost and the model expressiveness.

---

1. Place MovieLens data in `data/raw/`
2. Run notebook cells sequentially
3. Adjust parameters such as:

   * `n_users`
   * `hidden_dim`
   * `epochs`
4. Train bipartite model
5. Evaluate ranking metrics
6. Optionally, run the bonus section:

   * Load genome tags
   * Build heterograph
   * Train heterogeneous model
   * Evaluate again

All computational heavy lifting occurs via `dgl_utils.py`.

---

## Results & Observations

### Bipartite Model Performance

Typical metrics for Precision@10, Recall@10, and NDCG@10 fall in the range:

* P@10 ≈ 0.02
* R@10 ≈ 0.03–0.04
* NDCG@10 ≈ 0.02–0.03

These demonstrate meaningful learning from user–movie interactions.

### Heterograph Model Performance

Incorporating tags increases model complexity and requires more tuning. Initial results may be low without sufficient training or tag coverage.

However, the heterograph introduces:

* **Semantic propagation** via tag nodes
* **Richer movie embeddings**
* Potential for large improvements with optimization

This extension illustrates how knowledge-aware recommenders can be built.

---

## Design Choices

* Dense ID remapping for consistency
* Movie genre vectors as interpretable features
* GraphSAGE for inductive, neighborhood-based learning
* Full-ranking evaluation for realistic testing
* Clean separation between tutorial notebook and utility logic
* Optional bonus section showcasing heterogeneous GNNs

---

## Limitations & Future Work

* Training epochs kept small for tutorial speed
* Basic negative sampling
* No early stopping or hyperparameter search
* Heterograph model may need stronger optimization

Future directions:

* GAT or R-GCN for typed message passing
* Richer semantic graphs (actors, directors, keywords)
* Hard negative mining
* Learnable edge importance / attention

---

## References

* Deep Graph Library (DGL) Documentation

* MovieLens 20M Dataset — GroupLens Research

* Hamilton et al., "Inductive Representation Learning on Large Graphs" (GraphSAGE)

* Comprehensive GNN surveys and recommender literature

* [https://medium.com/@bscarleth.gtz/introduction-to-graph-neural-networks-an-illustrated-guide-c3f19da2ba39](https://medium.com/@bscarleth.gtz/introduction-to-graph-neural-networks-an-illustrated-guide-c3f19da2ba39)

* [https://medium.com/stanford-cs224w/online-link-prediction-with-graph-neural-networks-46c1054f2aa4](https://medium.com/stanford-cs224w/online-link-prediction-with-graph-neural-networks-46c1054f2aa4)
