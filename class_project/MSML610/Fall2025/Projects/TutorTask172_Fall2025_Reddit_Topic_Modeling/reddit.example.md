# Reddit Topic Modeling Example: r/worldnews

This document presents a complete, end-to-end example of **topic modeling on Reddit comments** using the **r/worldnews** subreddit.  
The goal of this example is to demonstrate how unsupervised text embeddings and clustering techniques can be combined to discover dominant discussion themes in large-scale social media data.

All reusable logic is implemented in `reddit_utils.py`.  
This notebook (`reddit.example.ipynb`) focuses on execution, results, and interpretation.

---

## Problem Statement

Reddit generates massive volumes of unstructured text through user comments.  
Understanding the dominant topics within a subreddit such as **r/worldnews** is challenging due to the scale and diversity of discussions.

This example demonstrates how to:
- Clean raw Reddit comments
- Learn semantic word representations using **fastText**
- Cluster comments into topics using **K-Means**
- Visualize topic structure using **t-SNE**

---

## Dataset

Originally, the project intended to use the **Pushshift API** to collect Reddit comments.  
However, due to recent API access restrictions, a public Kaggle dataset was used instead:

**Dataset:**  
*1 Million Reddit Comments from 40 Subreddits* (Kaggle)

From this dataset:
- Only comments from **r/worldnews** were selected
- A random sample of **5,000–10,000 comments** was used to balance runtime and topic quality

The dataset contains at least the following required fields:
- `subreddit`
- `body`

---

## Pipeline Overview

The example follows these steps:

1. **Load and Filter Data**
   - Filter comments belonging to `r/worldnews`
   - Randomly sample comments
   - Remove empty or deleted entries

2. **Text Cleaning**
   - Remove URLs, punctuation, and stopwords
   - Convert text to lowercase
   - Retain meaningful tokens only

3. **fastText Embedding Training**
   - Train an **unsupervised fastText model** (`skipgram`)
   - Training is performed on the cleaned Reddit comments
   - This avoids downloading large pretrained models

4. **Document Embedding Generation**
   - Each comment is represented as the average of its word vectors

5. **Topic Modeling**
   - Apply **K-Means clustering** to document embeddings
   - Each cluster represents a topic

6. **Topic Interpretation**
   - Extract top keywords per cluster
   - Display example comments for interpretability

7. **Visualization**
   - Reduce embeddings to 2D using **t-SNE**
   - Plot clustered comments
   - Save visualization as `tsne_plot.png`

---

## Results

The discovered topics reveal clear thematic structure in **r/worldnews** discussions:

- Dominant clusters focus on **Politics**, **Government**, and **International Affairs**
- Secondary topics include **Economic Issues**, **Conflicts**, and **Regional Crises**
- Automated moderation and bot-generated comments form a distinct cluster

The **t-SNE visualization** shows strong separation between clusters, indicating that fastText embeddings capture meaningful semantic differences between discussion topics.

---

## Output Artifacts

Running the example produces the following outputs:

- Trained fastText model (`.bin`)
- Topic assignments for each comment
- A topic summary table with keywords and example comments
- A 2D visualization saved as:
  tsne_plot.png

