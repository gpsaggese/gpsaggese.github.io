from __future__ import annotations

from pathlib import Path
import random
from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


DATA_DIR = Path(__file__).parent / "data"

MAX_ARTICLES = 40_000

try:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter

    HAS_KAGGLEHUB = True
except ImportError:
    kagglehub = None
    KaggleDatasetAdapter = None
    HAS_KAGGLEHUB = False


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_from_kagglehub(max_articles: int) -> List[str]:
    """
    Load article texts directly from Kaggle using KaggleHub.

    Requirements:
      - pip install "kagglehub[pandas-datasets]"
      - Kaggle API key at ~/.kaggle/kaggle.json (or Windows equivalent)
    """
    if not HAS_KAGGLEHUB:
        raise RuntimeError("kagglehub is not installed in this environment.")

    print("[KaggleHub] Fetching dataset jjinho/wikipedia-20230701 ...")

    # file_path="" lets kagglehub choose a sensible default for this dataset.
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "jjinho/wikipedia-20230701",
        ""
    )

    if "text" not in df.columns:
        raise RuntimeError(
            "Expected a 'text' column in the Kaggle dataset but did not find one."
        )

    df = df.dropna(subset=["text"])
    texts = df["text"].tolist()[:max_articles]
    print(f"[KaggleHub] Loaded {len(texts)} articles")
    return texts


def load_from_local_parquet(max_articles: int) -> List[str]:
    """
    Load article texts from local parquet shards (a.parquet, b.parquet, ...).

    This function:
      - Looks at ALL *.parquet files in data/
      - Samples roughly max_articles / num_files from each file
      - Returns up to max_articles texts total
    """
    if not DATA_DIR.exists():
        raise RuntimeError(f"Local data directory does not exist: {DATA_DIR}")

    all_files = sorted(DATA_DIR.glob("*.parquet"))
    if not all_files:
        raise RuntimeError(f"No parquet files found in {DATA_DIR}")

    print(f"[Local] Found {len(all_files)} parquet shards in {DATA_DIR}")

    texts: List[str] = []

    # Balanced sampling: try to take an equal share from each shard
    per_file_target = max_articles // len(all_files) + 1
    random.seed(42)

    for file in all_files:
        print(f"[Local] Sampling from {file.name}")
        df = pd.read_parquet(file, columns=["text"]).dropna(subset=["text"])

        if df.empty:
            continue

        n = min(per_file_target, len(df))
        sample = df["text"].sample(n=n, random_state=42).tolist()
        texts.extend(sample)

        if len(texts) >= max_articles:
            break

    texts = texts[:max_articles]
    if not texts:
        raise RuntimeError("No texts loaded from local parquet files.")

    print(f"[Local] Loaded {len(texts)} articles (sampled across all shards).")
    return texts


def load_wikipedia_articles(max_articles: int = MAX_ARTICLES) -> List[str]:
    """
    High-level loader used by the rest of the program.

    Strategy:
      1. Try KaggleHub (online loading from Kaggle).
      2. If that fails, fall back to local parquet files in data/.
    """
    # 1) Try KaggleHub
    if HAS_KAGGLEHUB:
        try:
            return load_from_kagglehub(max_articles)
        except Exception as e:
            print(f"[WARN] KaggleHub loading failed: {e}")

    # 2) Fallback: local parquet files
    try:
        return load_from_local_parquet(max_articles)
    except Exception as e:
        raise RuntimeError(
            "Could not load Wikipedia data from KaggleHub or local parquet files.\n"
            "Please either:\n"
            "  - Install kagglehub and configure your Kaggle API key, OR\n"
            "  - Download jjinho/wikipedia-20230701 into the 'data/' folder."
        ) from e


# -----------------------------------------------------------------------------
# Embeddings + semantic search
# -----------------------------------------------------------------------------

def build_embeddings(texts: List[str]):
    """
    Encode article texts with SBERT and return (model, embeddings).
    """
    print("[Embeddings] Loading SBERT model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"[Embeddings] Encoding {len(texts)} articles...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=256,  # adjust if you hit memory issues
    )
    print("[Embeddings] Done.")
    return model, embeddings


def semantic_search(query: str, model, embeddings, texts: List[str], top_k: int = 5):
    """
    Run a semantic search over the articles and print the top_k matches.
    """
    query_emb = model.encode([query])
    sims = cosine_similarity(query_emb, embeddings)[0]

    top_idx = np.argsort(sims)[::-1][:top_k]

    print(f"\nTop {top_k} results for: {query!r}\n")
    for rank, idx in enumerate(top_idx, start=1):
        score = sims[idx]
        snippet = texts[idx][:400].replace("\n", " ")
        print(f"{rank}. score={score:.3f}")
        print(snippet)
        print("-" * 80)



def main():
    print(f"[Config] MAX_ARTICLES = {MAX_ARTICLES}")
    texts = load_wikipedia_articles(max_articles=MAX_ARTICLES)
    model, embeddings = build_embeddings(texts)

    print("\nSemantic search ready. Type a query (or 'exit' to quit).")
    while True:
        query = input("\nQuery: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        semantic_search(query, model, embeddings, texts, top_k=5)


if __name__ == "__main__":
    main()
