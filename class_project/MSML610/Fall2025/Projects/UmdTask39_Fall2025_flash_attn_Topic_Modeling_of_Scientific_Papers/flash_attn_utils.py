# flash_attn_utils.py (Python 3.8-compatible typing)

import json, re, logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Iterable, Tuple, Union

import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModel

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import umap
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_whitespace_re = re.compile(r"\s+")

def stream_arxiv_jsonl(path: Union[str, Path], max_docs: Optional[int] = None):
    """
    Stream a large JSONL (one record per line). Yields a minimal dict per paper.
    """
    path = Path(path)
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield {
                "id": rec.get("id"),
                "title": rec.get("title"),
                "abstract": rec.get("abstract"),
                "categories": rec.get("categories"),
            }
            n += 1
            if max_docs is not None and n >= max_docs:
                break

def preprocess_record(rec: Dict[str, Any], min_abstract_len: int = 50) -> Optional[Dict[str, Any]]:
    title = (rec.get("title") or "").strip()
    abstract = (rec.get("abstract") or "").strip()
    if not title or not abstract or len(abstract) < min_abstract_len:
        return None
    text = f"{title}. {abstract}"
    text = _whitespace_re.sub(" ", text).lower()
    return {
        "id": rec.get("id"),
        "title": title,
        "abstract": abstract,
        "categories": rec.get("categories"),
        "text": text,
    }

def load_and_preprocess_arxiv(path: Union[str, Path], max_docs: Optional[int] = None, min_abstract_len: int = 50) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for raw in stream_arxiv_jsonl(path, max_docs=max_docs):
        cleaned = preprocess_record(raw, min_abstract_len=min_abstract_len)
        if cleaned is not None:
            rows.append(cleaned)
    return pd.DataFrame.from_records(rows)

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_encoder(model_name: str = "intfloat/e5-base-v2") -> Tuple[AutoTokenizer, AutoModel, torch.device]:
    device = pick_device()
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name)
    mdl.to(device).eval()
    logger.info("Loaded model %s on %s", model_name, device)
    return tok, mdl, device

@torch.no_grad()
def encode_texts(
    texts: Iterable[str],
    tok: AutoTokenizer,
    mdl: AutoModel,
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 256,
) -> np.ndarray:
    texts = list(texts)
    embs: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i+batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = mdl(**enc)
        last_hidden = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        summed = (last_hidden * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        pooled = summed / lengths
        embs.append(pooled.detach().cpu().numpy())
    return np.vstack(embs)

def cluster_kmeans(embeddings: np.ndarray, k: int = 20, random_state: int = 42) -> Tuple[np.ndarray, KMeans]:
    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    labels = km.fit_predict(embeddings)
    return labels, km

def describe_topics_tfidf(texts: List[str], labels: np.ndarray, top_k: int = 10) -> Dict[int, List[Tuple[str, float]]]:
    vec = TfidfVectorizer(max_features=50_000, ngram_range=(1,2), min_df=5)
    X = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())
    topics: Dict[int, List[Tuple[str, float]]] = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        centroid = X[idx].mean(axis=0).A1
        top = centroid.argsort()[::-1][:top_k]
        topics[int(c)] = list(zip(vocab[top], centroid[top].tolist()))
    return topics

def reduce_2d(embeddings: np.ndarray, method: str = "umap", random_state: int = 42) -> np.ndarray:
    if method == "tsne":
        proj = TSNE(n_components=2, init="pca", random_state=random_state, perplexity=30, learning_rate="auto")
        return proj.fit_transform(embeddings)
    reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=15, min_dist=0.1)
    return reducer.fit_transform(embeddings)
