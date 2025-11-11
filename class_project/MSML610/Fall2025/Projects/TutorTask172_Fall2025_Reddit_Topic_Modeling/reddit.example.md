# Midterm Project Report: Topic Modeling on Reddit Comments (r/worldnews)

## Overview
This project applies topic modeling techniques to Reddit comments to uncover key discussion themes in the r/worldnews subreddit. The main objective was to identify, cluster, and visualize dominant global news topics using unsupervised learning.

## Data Source and Sampling
Originally, the plan was to collect Reddit comments via the Pushshift API as described in the project instructions. However, due to recent access restrictions on Pushshift (API now limited to moderators), I used a **similar public dataset** from **Kaggle — “1 Million Reddit Comments from 40 Subreddits”**.

From this dataset, I filtered comments belonging to **r/worldnews** and randomly sampled **5,000 comments** (for faster runtime and cleaner visualization).  
A larger 10,000-comment version was also tested later for better topic diversity.

## Methodology
1. **Text Cleaning** – removed URLs, punctuation, stopwords, and deleted/removed markers.  
2. **Embedding Generation** – generated dense text representations using **fastText (cc.en.300.bin)** embeddings.  
3. **Clustering** – applied **K-Means (k=8)** to group similar comments into topic clusters.  
4. **Topic Labeling** – used **Zero-Shot Classification (BART-large-MNLI)** to automatically assign human-readable topic labels (e.g., *Politics*, *Society*, *Economy*, *Climate Change*).  
5. **Visualization** – used **t-SNE** for dimensionality reduction and plotted clusters; created a **bar chart** to show topic frequency.  

## Results
- The most prominent topics in r/worldnews were **Politics**, **Society**, and **Economy**, showing the strong dominance of political discussions.  
- Smaller clusters represented themes like **War & Conflict** and **Climate Change**.  
- The t-SNE visualization clearly separated clusters by semantic similarity, while the bar chart quantified their frequency.

## Limitations
- Due to API access issues, the dataset was pre-collected rather than live-scraped.  
- The Kaggle dataset may not fully reflect current Reddit activity.  
- Only 5,000–10,000 comments were analyzed for runtime efficiency; a full million-comment run would require distributed computing.

## Conclusion
Even with a limited sample, the fastText + K-Means + t-SNE pipeline effectively revealed coherent topics and thematic boundaries within r/worldnews. This demonstrates how unsupervised text embeddings can uncover high-level structures in large-scale social media discussions.

---

**Author:** Saanvi Joginipally  
**Course:** MSML610 – Midterm Project  
