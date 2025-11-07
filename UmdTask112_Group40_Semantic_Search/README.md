# 🧠 Project 2 – Semantic Search Engine (SBERT + Cosine Similarity)

This project implements a **semantic search engine** that retrieves the most relevant Wikipedia articles based on their **semantic meaning**, not just keyword overlap.  
It uses **Sentence-BERT (SBERT)** to generate embeddings and **cosine similarity** to rank search results.  
You can test the system both via the **command line** and a **Flask web interface**.

---

## 🚀 Features

- Loads Wikipedia articles from the Kaggle dataset [`jjinho/wikipedia-20230701`](https://www.kaggle.com/datasets/jjinho/wikipedia-20230701)
- Embeds text using **Sentence-BERT (all-MiniLM-L6-v2)**
- Supports:
  - Command-line queries (`python main.py`)
  - Web-based interaction (`python app.py`)
- Uses **cosine similarity** for semantic relevance
- Automatically loads from **KaggleHub** or local `.parquet` files
- Configurable dataset size (up to 20,000+ articles)

---
## 🗂️ Project Structure

UmdTask112_Group40_Semantic_Search/
│
├── main.py # Core logic (data loading, embeddings, CLI mode)
├── app.py # Flask web interface
├── requirements.txt # Dependencies list
├── data/ # Optional: local dataset (.parquet files)
└── README.md # Documentation.

## 🧩 Technologies Used

| Library | Purpose |
|----------|----------|
| `sentence-transformers` | Sentence-BERT model for embeddings |
| `scikit-learn` | Cosine similarity calculation |
| `Flask` | Web interface |
| `KaggleHub` | Dataset download and integration |
| `pandas` | Data handling |
| `tqdm` | Progress tracking |

---

## 🧮 How to Run

### ▶️ Command-Line Version
Run the project from your terminal:
```bash
python main.py
Query: Eiffel Tower
Query: quantum physics
Query: exit


### Web Version
python app.py