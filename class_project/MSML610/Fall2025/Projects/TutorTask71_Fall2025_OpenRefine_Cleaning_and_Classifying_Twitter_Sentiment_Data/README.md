# Cleaning and Classifying Twitter Sentiment Data (Sentiment140)

**Course:** MSML 610
**Difficulty:** 3  
**Dataset:** [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)

---

## Objective

The goal of this project is to **clean, prepare, and classify** a dataset of tweets for sentiment analysis.  
It demonstrates how to:
1. Clean and preprocess tweets using **OpenRefine**  
2. Engineer interpretable text features  
3. Load and inspect the cleaned data in **Python**  
4. Train and evaluate sentiment classification models

---

## Dataset Overview

**Raw File:** `training.1600000.processed.noemoticon.csv` (untracked)  
**Cleaned Output:** `Sentiment140_raw.csv` (not tracked)

### Raw Columns
| Column | Description |
|--------|--------------|
| `target` | 0 = negative, 2 = neutral, 4 = positive |
| `id` | Tweet ID |
| `date` | Timestamp |
| `flag` | Query / unused |
| `user` | Username |
| `text` | Tweet content |

### Cleaned Columns (After OpenRefine)
| Column | Description |
|---------|--------------|
| `text_clean` | Lowercased, cleaned text without URLs, mentions, or symbols |
| `word_count` | Number of words in the cleaned text |
| `char_count` | Character count excluding spaces |
| `avg_word_length` | Average word length |
| `label_str` | Sentiment label (negative, neutral, positive) |
| `pos_word_count` | Count of common positive words |
| `neg_word_count` | Count of common negative words |
| `sentiment_score_norm` | Normalized (pos−neg)/(pos+neg) |

---

## Data Cleaning and Feature Engineering

Cleaning was performed in **OpenRefine** using a reproducible JSON recipe (see [`API.md`](./API.md)).  

### Key Steps
1. **Rename columns** to match `target, id, date, flag, user, text`
2. **Create `text_clean`** – lowercase text, remove URLs, @mentions, hashtags, and non-alphabetic characters
3. **Remove blank rows** where `text_clean` is empty
4. **Generate text features:**
   - `word_count`, `char_count`, `avg_word_length`
5. **Map numeric labels** → string labels (`0→negative`, `2→neutral`, `4→positive`)
6. **Lexicon counts:**
   - `pos_word_count` (positive words)
   - `neg_word_count` (negative words)
7. **Compute sentiment score:**
   - `sentiment_score_norm = (pos − neg) / (pos + neg)` with zero handling
8. **Convert feature columns** to numeric for later modeling

---

## Repository Structure

| File | Purpose |
|------|----------|
| `API.md` | Contains the OpenRefine cleaning recipe (JSON) |
| `API.ipynb` | Feature engineering and exploration of dataset |
| `example.ipynb` | Demonstrates loading and sanity-checking the cleaned data |
| `utils_data_io.py` | Load and split cleaned data for modeling |
| `utils_post_processing.py` | Feature inspection and summary helpers |
| `Dockerfile` | Containerized setup for reproducible notebook runs |

---

## Local Setup using Docker 

1. **Build the Docker image**
    ```bash
    docker build -t twitter-cleaning .

2. **Run the container**
    ```bash
    docker run --rm -it -p 8888:8888 -v "$(pwd)":/app twitter-cleaning

Then open your browser at http://localhost:8888