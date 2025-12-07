#!/usr/bin/env python3
"""
Semantic Search Flask App for Docker
This is the production runtime - matches semantic_search_complete.ipynb logic.
The notebook contains the detailed demonstration and explanation.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template_string

print("\n" + "="*70)
print("🔍 Semantic Search Engine - Docker Runtime")
print("="*70 + "\n")

# Configuration (same as notebook Step 2)
DATA_DIR = Path("data")
DOCKER_MODE = os.environ.get('DOCKER_ENV') == 'true'
MAX_ARTICLES = 5_000 if DOCKER_MODE else 40_000

if DOCKER_MODE:
    print("🐳 DOCKER MODE: Using 5,000 articles for quick demo")
    print("   Embedding time: ~2-3 minutes\n")
else:
    print("💻 FULL MODE: Using up to 40,000 articles\n")

# Data loading (same as notebook Step 3)
def load_wikipedia_articles(max_articles: int) -> List[str]:
    """Load Wikipedia articles - matches notebook logic."""
    if DOCKER_MODE:
        sample_file = Path("data_sample.parquet")
        if sample_file.exists():
            print(f"[Docker] Loading from {sample_file}...")
            df = pd.read_parquet(sample_file, columns=["text"]).dropna(subset=["text"])
            texts = df["text"].tolist()[:max_articles]
            print(f"[Docker] Loaded {len(texts):,} articles\n")
            return texts
        else:
            raise RuntimeError(f"❌ {sample_file} not found!")
    
    # Normal mode - load from data/ directory
    all_files = sorted(DATA_DIR.glob("*.parquet"))
    if not all_files:
        raise RuntimeError(f"❌ No parquet files in {DATA_DIR}")
    
    texts = []
    per_file_target = max_articles // len(all_files) + 1
    
    for file in all_files:
        df = pd.read_parquet(file, columns=["text"]).dropna(subset=["text"])
        if not df.empty:
            n = min(per_file_target, len(df))
            sample = df["text"].sample(n=n, random_state=42).tolist()
            texts.extend(sample)
            if len(texts) >= max_articles:
                break
    
    return texts[:max_articles]

# Step 1: Load data
print("Step 1: Loading Wikipedia articles...")
texts = load_wikipedia_articles(MAX_ARTICLES)
print(f"✓ Loaded {len(texts):,} articles\n")

# Step 2: Load model (same as notebook Step 5)
print("Step 2: Loading Sentence Transformer model...")
print("   Model: all-MiniLM-L6-v2")
model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"✓ Model loaded (384 dimensions)\n")

# Step 3: Generate embeddings (same as notebook Step 6)
print(f"Step 3: Generating embeddings for {len(texts):,} articles...")
print("   This may take 2-3 minutes...\n")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    batch_size=256,
    convert_to_numpy=True
)

print(f"\n✓ Embeddings generated: {embeddings.shape}")
print(f"   Memory: ~{embeddings.nbytes / 1024 / 1024:.1f} MB\n")

# HTML Template (same as notebook Step 11)
HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Semantic Search Engine</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 900px; margin: 40px auto; padding: 0 20px; background: #f5f5f5;
        }
        .container { background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; margin-bottom: 10px; }
        .subtitle { color: #7f8c8d; margin-bottom: 30px; }
        .search-box { margin: 30px 0; }
        input[type=text] { 
            width: 100%; padding: 15px; font-size: 16px;
            border: 2px solid #ddd; border-radius: 5px; box-sizing: border-box;
        }
        input[type=text]:focus { outline: none; border-color: #3498db; }
        button { 
            margin-top: 15px; padding: 12px 30px; font-size: 16px;
            background: #3498db; color: white; border: none;
            border-radius: 5px; cursor: pointer; transition: background 0.3s;
        }
        button:hover { background: #2980b9; }
        .result { 
            margin: 20px 0; padding: 20px; background: #f8f9fa;
            border-left: 4px solid #3498db; border-radius: 5px;
        }
        .score { color: #27ae60; font-weight: bold; font-size: 14px; margin-bottom: 10px; }
        .snippet { line-height: 1.6; color: #2c3e50; }
        .stats { 
            background: #ecf0f1; padding: 15px; border-radius: 5px;
            margin-bottom: 20px; font-size: 14px; color: #7f8c8d;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Semantic Search Engine</h1>
        <p class="subtitle">Powered by Sentence Transformers (all-MiniLM-L6-v2)</p>
        
        <div class="stats">📊 Searching across {{ num_articles }} Wikipedia articles</div>

        <div class="search-box">
            <form method="post">
                <input type="text" name="query" value="{{ query }}" 
                       placeholder="Try: 'famous tower in Paris' or 'quantum physics'" autofocus />
                <button type="submit">Search</button>
            </form>
        </div>

        {% if results %}
            <h2>Results ({{ results|length }})</h2>
            {% for r in results %}
                <div class="result">
                    <div class="score">Rank #{{ r.rank }} — Similarity: {{ "%.4f"|format(r.score) }}</div>
                    <div class="snippet">{{ r.snippet }}</div>
                </div>
            {% endfor %}
        {% elif query %}
            <p style="text-align:center; color:#7f8c8d;">No results found for "{{ query }}"</p>
        {% endif %}
    </div>
</body>
</html>
"""

# Flask app (same as notebook Step 11)
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    results = []
    
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query:
            # Perform semantic search
            query_emb = model.encode([query])
            sims = cosine_similarity(query_emb, embeddings)[0]
            top_idx = np.argsort(sims)[::-1][:5]
            
            for rank, idx in enumerate(top_idx, start=1):
                results.append({
                    "rank": rank,
                    "score": float(sims[idx]),
                    "snippet": texts[idx][:400].replace("\n", " ")
                })
    
    return render_template_string(
        HTML_TEMPLATE,
        query=query,
        results=results,
        num_articles=len(texts)
    )

# Start Flask server (same as notebook Step 12)
if __name__ == "__main__":
    print("="*70)
    print("🌐 Starting Flask Web Server")
    print("   Visit: http://localhost:5000")
    print("="*70)
    print()
    
    app.run(host="0.0.0.0", port=5000, debug=False)