# clip_embed.example.ipynb Documentation

## Overview

This comprehensive notebook demonstrates the complete workflow for building a multimodal visual search engine using CLIP embeddings and FAISS. It covers dataset exploration, embedding generation, indexing, similarity search, evaluation, and visualization.

## Contents

### 1. Dataset Introduction

The notebook starts with an introduction to the Flickr8K dataset:
- **Dataset Description**: 8,000 images, each paired with five descriptive captions
- **Dataset Statistics**: 
  - Total records: 40,455 image-caption pairs
  - Unique images: 8,091
  - Unique captions: 40,201
- **Sample Visualization**: Displays random image-caption pairs from the dataset

### 2. CLIP Model Setup

The notebook demonstrates how to:
- Load the CLIP model (`openai/clip-vit-base-patch32`) and processor
- Test the model by processing sample image-text pairs
- Understand how CLIP maps images and text into a shared 512-dimensional embedding space

### 3. FAISS Index Initialization

Introduction to FAISS (Facebook AI Similarity Search):
- Creates two separate FAISS indexes:
  - `image_index`: For storing image embeddings
  - `text_index`: For storing text embeddings
- Uses `IndexFlatIP` (Inner Product) for similarity search
- Embedding dimension: 512 (matching CLIP output)

### 4. Embedding Generation and Indexing

#### Image Embeddings
- Iterates through all 8,091 unique images in the dataset
- Generates CLIP embeddings for each image via API calls
- Normalizes embeddings for optimal similarity search
- Adds normalized embeddings to the image index
- Saves the index to `clip_embed.image_index.faiss`

#### Text Embeddings
- Iterates through all 40,201 unique captions in the dataset
- Generates CLIP embeddings for each caption via API calls
- Normalizes embeddings for optimal similarity search
- Adds normalized embeddings to the text index
- Saves the index to `clip_embed.text_index.faiss`

### 5. Similarity Search Examples

Demonstrates practical search functionality:
- **Text-to-Image Search**: Example query "lion" retrieves relevant images
- Shows how to normalize query embeddings and perform k-nearest neighbor search
- Displays retrieved images with their similarity scores

### 6. Evaluation Metrics

Implements comprehensive evaluation functions:

#### Precision@k and Recall@k Functions
- `precision_at_k`: Calculates precision for top-k retrieved items
- `recall_at_k`: Calculates recall for top-k retrieved items

#### Evaluation Functions
- `evaluate_text_to_image_retrieval`: Evaluates text-to-image retrieval performance
- `evaluate_image_to_text_retrieval`: Evaluates image-to-text retrieval performance

Both functions:
- Build ground truth mappings from the dataset
- Support configurable k values (e.g., [1, 5, 10, 20])
- Allow sampling for faster evaluation
- Return average precision@k and recall@k metrics

### 7. Performance Evaluation Results

The notebook includes evaluation results on 500 sample queries:

#### Text-to-Image Retrieval Results
- Precision@1: 0.2760, Recall@1: 0.2750
- Precision@5: 0.1012, Recall@5: 0.5050
- Precision@10: 0.0612, Recall@10: 0.6110
- Precision@20: 0.0358, Recall@20: 0.7140

#### Image-to-Text Retrieval Results
- Precision@1: 0.4300, Recall@1: 0.0860
- Precision@5: 0.2908, Recall@5: 0.2908
- Precision@10: 0.1932, Recall@10: 0.3864
- Precision@20: 0.1241, Recall@20: 0.4964

#### Visualization
- Creates side-by-side plots showing precision@k and recall@k trends
- Displays a summary table comparing both retrieval directions

### 8. Clustering and Visualization

#### Image Embeddings Clustering
- Reconstructs all 8,091 image embeddings from the FAISS index
- Applies PCA to reduce dimensions from 512 to 50
- Uses UMAP (or PCA fallback) to reduce to 2D for visualization
- Applies K-Means clustering with k=10
- Visualizes clusters in 2D space with color-coded cluster labels

#### Text Embeddings Clustering
- Reconstructs all 40,201 text embeddings from the FAISS index
- Applies the same dimensionality reduction pipeline
- Visualizes text embedding clusters in 2D space

## Key Features

- **Complete Workflow**: From dataset loading to evaluation and visualization
- **Performance Metrics**: Comprehensive evaluation with precision@k and recall@k
- **Visualization**: Both evaluation metrics plots and embedding cluster visualizations
- **Scalability**: Demonstrates handling of large-scale datasets (8K+ images, 40K+ captions)
- **Practical Examples**: Real-world search queries and results

## Dependencies

The notebook uses:
- `pandas` and `numpy` for data manipulation
- `PIL` for image processing
- `tqdm` for progress bars
- `requests` for API calls
- `torch` and `transformers` for CLIP model
- `faiss` for similarity search
- `sklearn` for PCA and K-Means clustering
- `matplotlib` for visualization
- `umap-learn` for dimensionality reduction (optional)

## Purpose

This notebook serves as:
- A comprehensive tutorial for building multimodal search systems
- A reference implementation for CLIP-based retrieval
- An evaluation framework for measuring retrieval performance
- A visualization tool for understanding embedding spaces
- A demonstration of best practices for large-scale embedding indexing

## Output Files

The notebook generates:
- `clip_embed.image_index.faiss`: FAISS index containing all image embeddings
- `clip_embed.text_index.faiss`: FAISS index containing all text embeddings
- Evaluation metrics and visualizations displayed inline

