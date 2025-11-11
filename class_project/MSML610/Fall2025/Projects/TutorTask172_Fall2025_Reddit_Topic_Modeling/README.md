# Reddit Topic Modeling (MSML610 Fall 2025)

## Overview
This project performs topic modeling on Reddit discussions from the **r/worldnews** subreddit using modern NLP techniques.

### Tools Used
- **fastText**: to generate word embeddings
- **K-Means**: for unsupervised clustering
- **t-SNE**: for visualizing clusters in 2D
- **Transformers (BART-large-MNLI)**: for zero-shot topic labeling

### Dataset
Since the Pushshift API was unavailable, a **Kaggle dataset** ("1 Million Reddit Comments from 40 Subreddits") was used instead.
A random sample of **5K–10K comments** from r/worldnews was selected for efficient runtime.

## Project Structure
- `reddit_utils.py`: helper functions
- `reddit.API.ipynb` and `reddit.API.md`: explain the API and structure
- `reddit.example.ipynb` and `reddit.example.md`: full example notebook and markdown
- `Dockerfile`: reproducible environment setup
- `README.md`: summary and instructions

## How to Run
```bash
docker build -t reddit_topic_modeling .
docker run -p 8888:8888 reddit_topic_modeling
```
Then open Jupyter Lab and run the notebooks sequentially.
