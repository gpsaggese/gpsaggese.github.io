# SBERT API Documentation (Sentence-Transformers)

**Tool documented:** `sentence-transformers` (Sentence-BERT embeddings)  
**Goal of this file:** Explain the *tool itself* (its native Python API + common usage patterns).  
Project-specific pipeline details (dataset, model training, results) belong in `SBERT_Example.*`.

---

## 1. What is Sentence-Transformers (SBERT)?

`sentence-transformers` provides easy-to-use, pretrained Transformer models that map text (sentences / short documents) into fixed-size dense vectors (“sentence embeddings”).  
These embeddings can then be used for:

- semantic similarity (cosine similarity)
- clustering / nearest-neighbor search
- retrieval (semantic search)
- as features for downstream ML models (e.g., Logistic Regression, SVM)

Core idea: similar meanings → embeddings close together in vector space.

---

## 2. Native API (Most Important Objects / Functions)

### 2.1 `SentenceTransformer`

**Import**
```python
from sentence_transformers import SentenceTransformer
```

**Construct a model**
```python
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
```

**Encode text into embeddings**
```python
sentences = ["I love this product.", "This is terrible."]
emb = model.encode(sentences, convert_to_numpy=True)
print(emb.shape)  # (N, D)
```

Key parameters you’ll commonly use:

- `batch_size`: speed / memory tradeoff
- `show_progress_bar`: helpful for long runs
- `device`: `"cpu"`, `"cuda"` (if available)
- `normalize_embeddings=True`: useful if you always use cosine similarity

---

### 2.2 `sentence_transformers.util` utilities

**Cosine similarity**
```python
from sentence_transformers import util
import torch

a = model.encode(["profit increased"], convert_to_tensor=True)
b = model.encode(["earnings improved"], convert_to_tensor=True)
score = util.cos_sim(a, b)
print(score)
```

**Semantic search (top-k most similar)**
```python
query = "great quarterly results"
corpus = ["losses increased", "profit rose", "market unchanged"]
query_emb = model.encode([query], convert_to_tensor=True)
corpus_emb = model.encode(corpus, convert_to_tensor=True)

hits = util.semantic_search(query_emb, corpus_emb, top_k=2)
print(hits[0])  # list of dicts: {'corpus_id': ..., 'score': ...}
```

---

## 3. Lightweight Wrapper Layer (What we add on top)

In this repository, we keep notebooks clean by calling *small wrapper functions* (in `src/SBERT_utils.py`) that:

- create a `SentenceTransformer` model by name
- encode a list of sentences with consistent defaults (batching, numpy output, optional normalization)
- (optionally) compute cosine similarity / semantic search using `util`

This wrapper layer is intentionally thin: it does **not** change SBERT behavior; it just standardizes how we call the library across scripts and notebooks.

Example wrapper style:
```python
from SBERT_utils import build_sbert_embeddings

sentences = ["text A", "text B"]
X = build_sbert_embeddings(sentences, model_name="sentence-transformers/all-MiniLM-L6-v2")
print(X.shape)
```

---

## 4. Design Notes (Why this API structure)

- **Separation of concerns:** SBERT tool usage (encode/similarity) is reusable across many tasks.
- **Notebook readability:** notebooks should demonstrate usage, not contain long utility code.
- **Reproducibility:** a single place to set defaults (model name, batch size, normalization).

---

## 5. Common Pitfalls

- **Comparing cosine similarity without normalization:** consider `normalize_embeddings=True` if your workflow is mostly cosine similarity.
- **Mixing CPU/GPU tensors:** if you use `convert_to_tensor=True`, keep tensors on the same device.
- **Large corpora:** semantic search scales with corpus size; consider ANN libraries (FAISS) for very large datasets.

---

## 6. Where to look next in this repository

- `SBERT_API.ipynb` → a minimal notebook demonstrating the tool API (generic examples).
- `SBERT_Example.ipynb` and `SBERT_Example.md` → the end-to-end project experiment and results (dataset + modeling).
