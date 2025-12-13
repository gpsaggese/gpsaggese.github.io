"""
reddit_utils.py

Reusable utilities for:
- Cleaning Reddit comments
- Loading a subreddit subset from a CSV
- Training fastText unsupervised 
- Building document embeddings (average word vectors)
- KMeans clustering for topic discovery
- t-SNE visualization + saving a plot
- Producing a topic summary table
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# ----------------------------
# Text cleaning
# ----------------------------
_URL_RE = re.compile(r"http\S+|www\S+")
_NON_ALPHA_RE = re.compile(r"[^A-Za-z\s]+")


def get_english_stopwords() -> set:
    """
    Return NLTK English stopwords set.

    NOTE:
    - Do NOT call nltk.download() inside utils.
    - In notebooks do:
        import nltk
        nltk.download("stopwords")
    """
    try:
        from nltk.corpus import stopwords  # pylint: disable=import-error
    except Exception as e:
        raise RuntimeError(
            "NLTK is required. Install and download stopwords.\n"
            "Example:\n"
            "  pip install nltk\n"
            "  import nltk; nltk.download('stopwords')"
        ) from e

    return set(stopwords.words("english"))


def clean_text(text: str, stopwords_set: Optional[set] = None, min_token_len: int = 3) -> str:
    """
    Clean Reddit comment text:
    - remove URLs
    - remove non-alphabetic characters
    - lowercase + split
    - remove stopwords (optional)
    - remove short tokens
    """
    if not isinstance(text, str):
        return ""

    text = _URL_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)

    tokens = text.lower().split()

    if stopwords_set is not None:
        tokens = [t for t in tokens if t not in stopwords_set]

    tokens = [t for t in tokens if len(t) >= min_token_len]
    return " ".join(tokens)


# ----------------------------
# Data loading
# ----------------------------
def load_subreddit_csv(
    csv_path: str,
    subreddit: str = "worldnews",
    n_samples: int = 10_000,
    random_state: int = 42,
    stopwords_set: Optional[set] = None,
    body_col: str = "body",
    subreddit_col: str = "subreddit",
) -> pd.DataFrame:
    """
    Load a Reddit comments CSV, filter to a subreddit, sample, and add clean_comment column.
    Expected columns by default: 'subreddit', 'body'
    """
    df = pd.read_csv(csv_path)

    if subreddit_col not in df.columns or body_col not in df.columns:
        raise ValueError(
            f"CSV must contain '{subreddit_col}' and '{body_col}' columns. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[df[subreddit_col].astype(str).str.lower() == subreddit.lower()]
    df = df[df[body_col].notna()].copy()

    if len(df) == 0:
        raise ValueError(f"No rows found for subreddit == '{subreddit}'.")

    if len(df) > n_samples:
        df = df.sample(n_samples, random_state=random_state)

    df["clean_comment"] = df[body_col].astype(str).apply(lambda x: clean_text(x, stopwords_set))
    df = df[df["clean_comment"].str.strip() != ""].reset_index(drop=True)
    return df


def write_training_file(clean_texts: Iterable[str], out_path: str = "training_data.txt") -> str:
    """
    Write one cleaned comment per line for fastText train_unsupervised input.
    Returns the output file path.
    """
    with open(out_path, "w", encoding="utf-8") as f:
        for line in clean_texts:
            if isinstance(line, str) and line.strip():
                f.write(line.strip() + "\n")
    return out_path


# ----------------------------
# fastText training (GeeksforGeeks-style)
# ----------------------------
def train_fasttext_unsupervised(
    train_txt_path: str,
    model_type: str = "skipgram",
    dim: int = 100,
    epoch: int = 10,
    lr: float = 0.05,
    min_count: int = 2,
    minn: int = 3,
    maxn: int = 6,
    thread: int = 4,
    save_path: Optional[str] = None,
):
    """
    Train fastText unsupervised model via Python fasttext package.
    """
    try:
        import fasttext  # type: ignore
    except Exception as e:
        raise RuntimeError("fasttext package is required. Try: pip install fasttext") from e

    ft_model = fasttext.train_unsupervised(
        input=train_txt_path,
        model=model_type,
        dim=dim,
        epoch=epoch,
        lr=lr,
        minCount=min_count,
        minn=minn,
        maxn=maxn,
        thread=thread,
    )

    if save_path:
        ft_model.save_model(save_path)

    return ft_model


# ----------------------------
# Embeddings
# ----------------------------
def average_vector(ft_model, text: str) -> np.ndarray:
    """
    Compute document embedding by averaging fastText word vectors.
    Works for fastText models.
    """
    dim = ft_model.get_dimension()
    if not isinstance(text, str) or not text.strip():
        return np.zeros(dim, dtype=np.float32)

    words = text.split()
    if not words:
        return np.zeros(dim, dtype=np.float32)

    vecs = [ft_model.get_word_vector(w) for w in words]
    return np.mean(vecs, axis=0).astype(np.float32)


def compute_document_embeddings(ft_model, clean_texts: List[str]) -> np.ndarray:
    """
    Compute embedding matrix X of shape (n_docs, dim)
    """
    dim = ft_model.get_dimension()
    X = np.zeros((len(clean_texts), dim), dtype=np.float32)
    for i, txt in enumerate(clean_texts):
        X[i] = average_vector(ft_model, txt)
    return X


# ----------------------------
# Topic modeling with KMeans
# ----------------------------
def run_kmeans(X: np.ndarray, n_clusters: int = 5, random_state: int = 42, n_init: int = 10):
    """
    Fit KMeans and return (labels, model)
    """
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(X)
    return labels, km


def top_words_by_cluster(clean_texts: List[str], labels: np.ndarray, top_n: int = 12) -> Dict[int, List[str]]:
    """
    Return dict: cluster_id -> top words by frequency within that cluster.
    """
    buckets: Dict[int, List[str]] = defaultdict(list)
    for txt, lab in zip(clean_texts, labels):
        buckets[int(lab)].extend(txt.split())

    out: Dict[int, List[str]] = {}
    for lab, words in buckets.items():
        out[lab] = [w for w, _ in Counter(words).most_common(top_n)]
    return out


def build_topic_summary(
    df: pd.DataFrame,
    labels_col: str = "topic",
    body_col: str = "body",
    clean_col: str = "clean_comment",
    top_n_words: int = 10,
    n_examples: int = 3,
) -> pd.DataFrame:
    """
    Build a topic summary table:
    - topic_id
    - count
    - top_words
    - example_comments
    """
    if labels_col not in df.columns or clean_col not in df.columns or body_col not in df.columns:
        raise ValueError(f"DataFrame must include '{labels_col}', '{clean_col}', '{body_col}'.")

    words = top_words_by_cluster(df[clean_col].tolist(), df[labels_col].to_numpy(), top_n=top_n_words)

    rows = []
    for t in sorted(words.keys()):
        examples = df[df[labels_col] == t][body_col].head(n_examples).tolist()
        rows.append(
            {
                "topic_id": int(t),
                "count": int((df[labels_col] == t).sum()),
                "top_words": ", ".join(words[t]),
                "example_comments": " || ".join(examples),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------
# t-SNE + plotting
# ----------------------------
def run_tsne(
    X: np.ndarray,
    n_points: int = 2000,
    random_state: int = 42,
    perplexity: int = 30,
    learning_rate: int = 200,
    n_iter: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run t-SNE on at most n_points from X.
    Returns (idx, X_2d):
      - idx: indices selected from X
      - X_2d: t-SNE 2D coordinates for those indices
    """
    from sklearn.manifold import TSNE

    n = X.shape[0]
    n_points = min(n_points, n)

    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=n_points, replace=False)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=random_state,
        init="random",
    )
    X_2d = tsne.fit_transform(X[idx])
    return idx, X_2d


def save_tsne_plot(
    X_2d: np.ndarray,
    labels: List[str],
    out_path: str = "tsne_plot.png",
    title: str = "t-SNE of r/worldnews topics (fastText + KMeans)",
):
    """
    Save a t-SNE scatter plot colored by topic label.
    Returns the output image path.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_df = pd.DataFrame({"x": X_2d[:, 0], "y": X_2d[:, 1], "topic_label": labels})

    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=plot_df, x="x", y="y", hue="topic_label", s=15, linewidth=0)
    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend(title="Topic", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    return out_path
