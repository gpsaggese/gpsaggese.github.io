# SBERT API Documentation

This document describes how this project uses SBERT (Sentence-BERT) for
generating fixed-size sentence embeddings, along with the lightweight utility
functions implemented in `src/SBERT_utils.py`.

The goal is to provide a simple and repeatable API that standardizes:

1. Loading configuration
2. Loading cleaned text and labels
3. Generating or loading embeddings
4. Integrating SBERT inside downstream classifiers

---

## 1. Model: SentenceTransformer

We use the pretrained model: sentence-transformers/all-MiniLM-L6-v2

Key properties:
- Embedding dimension: **384**
- Fast CPU performance
- Good semantic similarity accuracy for financial text use cases

### Basic Usage

```python
from sentence_transformers import SentenceTransformer
```

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(list_of_sentences, batch_size=64, show_progress_bar=True)

Outputs a matrix of shape: (num_sentences, 384)

## 2. Utility Functions (SBERT_utils.py)

These helper functions keep your notebooks and scripts clean and consistent.

load_config(path)

Loads config.yaml.
```python
cfg = load_config("config.yaml")
```
load_clean_data(cfg)
```python
df, labels = load_clean_data(cfg)
```
load_embeddings(cfg)
Loads SBERT embeddings from: data/processed/sbert_embeddings.npy
```python
X = load_embeddings(cfg)
```
### Why this API?
	•	Avoids rewriting boilerplate in every notebook.
	•	Standardizes where data and embeddings live.
	•	Makes the project reproducible for TAs and future contributors.

## 3. End-to-End Example
```python
from SBERT_utils import load_config, load_clean_data, load_embeddings
from sentence_transformers import SentenceTransformer

cfg = load_config("config.yaml")

df, labels = load_clean_data(cfg)
X = load_embeddings(cfg)

model = SentenceTransformer(cfg["model"]["name"])
new_embedding = model.encode(["Market sentiment improved today."])
```

## 4. Notes & Limitations

	•	SBERT is used in inference mode only unless fine-tuning is explicitly run.
	•	Embeddings must be regenerated if the preprocessing logic changes.
	•	The API is intentionally minimal to support multiple downstream models:
	•	logistic regression
	•	linear SVM
	•	fine-tuned transformer classifier
