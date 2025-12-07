# Semantic Search Engine

**Author:** Ali Fehmi Yildiz  
**UID:** 121326737  
**Project:** MSML610 Final Project  
**Difficulty:** Medium

---

## What I Built

I built a search engine that understands what you mean, not just the words you type.

**Example:**
- I search: "famous tower in Paris"
- Regular search: No results (those exact words not found)
- My search: Finds the Eiffel Tower! 

My system understands that "famous tower in Paris" means the Eiffel Tower, even though I never typed those exact words.

---

## Why This Matters

Regular search engines only find exact word matches. If you don't know the right words, you don't find what you need.

My search engine understands meaning:
- I can search for "red planet" and find Mars
- I can search for "AI and neural networks" and find machine learning articles
- I can ask questions in my own words and still get good results

This is called **semantic search** - searching by meaning instead of matching words.

---

## Project Structure

I organized this project into two layers:

### **API Layer** (Reusable Library)
The core search functionality that anyone can use:
- `semantic_search_utils.py` - Main library with SemanticSearchEngine class
- `semantic_search.API.md` - Documentation explaining how to use the API
- `semantic_search.API.ipynb` - Quick examples showing the API in action

### **Example Application** (Complete Demo)
A working search application built with the API:
- `semantic_search.example.ipynb` - Full tutorial with explanations and visualizations
- `semantic_search.example.md` - Documentation of the application
- `app.py` - Flask web interface

### **Deployment**
Files for running the project:
- `Dockerfile` - Container definition
- `docker-compose.yml` - Easy deployment
- `requirements.txt` - Python dependencies
- `data_sample.parquet` - Sample Wikipedia data (5,000 articles)

---

## What I Used

**Data:** 50,000 Wikipedia articles  
**AI Model:** Sentence Transformers (all-MiniLM-L6-v2)  
**Programming:** Python with Flask  
**Tools:** Jupyter notebooks, Docker

---

## How to Use It

### Option 1: Docker (Easiest)
```bash
docker-compose up
```
Then open: http://localhost:5000

### Option 2: Run Locally
```bash
pip install -r requirements.txt
python app.py
```
Then open: http://localhost:5000

### Option 3: See the Tutorial
```bash
jupyter notebook semantic_search.example.ipynb
```

### Option 4: Use Just the API
```python
from semantic_search_utils import SemanticSearchEngine

engine = SemanticSearchEngine()
engine.index_documents(["Article 1...", "Article 2..."])
results = engine.search("your query", top_k=5)
```

---

## How It Works

**Step 1:** Load Wikipedia articles  
**Step 2:** Convert each article to 384 numbers (embeddings)  
**Step 3:** When you search, convert your query to 384 numbers  
**Step 4:** Find articles with similar numbers = similar meaning!

---

## My Results

"famous tower in Paris" → Found Eiffel Tower (0.54 similarity)  
"artificial intelligence" → Found AI/ML articles  
"red planet" → Found Mars articles  
Search speed: ~20 milliseconds

---

## What I Learned

- Sentence Transformers for semantic understanding
- Vector search with cosine similarity
- Docker deployment
- API design and software architecture

---

## References

- Sentence Transformers: https://www.sbert.net/
- "Sentence-BERT" by Reimers & Gurevych (2019)
- Wikipedia Dataset: https://www.kaggle.com/datasets/jjinho/wikipedia-20230701

---

**Ali Fehmi Yildiz** | UID: 121326737 | MSML610 - December 2025