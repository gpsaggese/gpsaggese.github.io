from __future__ import annotations

from pathlib import Path
from typing import List

from flask import Flask, request, render_template_string
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Import the existing logic from main.py
from main import load_wikipedia_articles, build_embeddings

app = Flask(__name__)


TEXTS: List[str] | None = None
MODEL = None
EMBEDDINGS = None

MAX_ARTICLES = 40_000  


def init_engine():
    """
    Lazily initialize the semantic search engine.
    Called on first request only.
    """
    global TEXTS, MODEL, EMBEDDINGS

    if TEXTS is not None and MODEL is not None and EMBEDDINGS is not None:
        return

    print(f"[Web] Initializing engine with {MAX_ARTICLES} articles...")
    TEXTS = load_wikipedia_articles(max_articles=MAX_ARTICLES)
    MODEL, EMBEDDINGS = build_embeddings(TEXTS)
    print("[Web] Engine ready.")


def run_search(query: str, top_k: int = 5):
    """
    Perform semantic search and return list of result dicts.
    """
    assert MODEL is not None and EMBEDDINGS is not None and TEXTS is not None

    query_emb = MODEL.encode([query])
    sims = cosine_similarity(query_emb, EMBEDDINGS)[0]

    top_idx = np.argsort(sims)[::-1][:top_k]
    results = []
    for rank, idx in enumerate(top_idx, start=1):
        results.append(
            {
                "rank": rank,
                "score": float(sims[idx]),
                "snippet": TEXTS[idx][:400].replace("\n", " "),
            }
        )
    return results



HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Semantic Search Demo</title>
    <style>
        body { font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
               max-width: 800px; margin: 40px auto; padding: 0 16px; }
        h1 { margin-bottom: 0.2rem; }
        form { margin: 1rem 0; }
        input[type=text] { width: 100%; padding: 8px 10px; font-size: 1rem; }
        button { margin-top: 0.5rem; padding: 8px 16px; font-size: 1rem; }
        .result { margin-bottom: 1.4rem; padding-bottom: 0.8rem; border-bottom: 1px solid #eee; }
        .score { color: #666; font-size: 0.9rem; }
        .snippet { margin-top: 0.3rem; }
        .loading { color: #888; font-style: italic; }
    </style>
</head>
<body>
    <h1>Semantic Search Engine</h1>
    <p>Type a query (e.g. <code>Eiffel Tower</code>, <code>quantum physics</code>, <code>machine learning</code>)</p>

    <form method="post">
        <input type="text" name="query" value="{{ query|default('') }}" autofocus />
        <button type="submit">Search</button>
    </form>

    {% if initializing %}
        <p class="loading">Initializing model and loading articles... this can take a bit on first run.</p>
    {% endif %}

    {% if results %}
        <h2>Results</h2>
        {% for r in results %}
            <div class="result">
                <div class="score">#{{ r.rank }} — score {{ "%.3f"|format(r.score) }}</div>
                <div class="snippet">{{ r.snippet }}</div>
            </div>
        {% endfor %}
    {% elif query %}
        <p>No results found.</p>
    {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    results = []
    initializing = False


    global TEXTS, MODEL, EMBEDDINGS
    if TEXTS is None or MODEL is None or EMBEDDINGS is None:
        initializing = True
        init_engine()

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query:
            results = run_search(query)

    return render_template_string(
        HTML_TEMPLATE,
        query=query,
        results=results,
        initializing=initializing,
    )


if __name__ == "__main__":

    app.run(host="127.0.0.1", port=5000, debug=True)
