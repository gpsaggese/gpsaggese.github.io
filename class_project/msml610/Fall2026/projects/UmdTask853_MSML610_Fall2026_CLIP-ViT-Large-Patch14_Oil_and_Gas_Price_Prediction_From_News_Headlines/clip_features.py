"""Frozen CLIP headline representations and daily news-event features.

Encoding requires Transformers and PyTorch; utilities for daily aggregation are
pure NumPy/Pandas so they can be unit-tested without model downloads.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_utils import EVENT_PROMPTS

MODEL_ID = "openai/clip-vit-large-patch14"


def _cache_key(headlines: list[str], model_id: str) -> str:
    digest = hashlib.sha256(model_id.encode())
    for item in headlines:
        digest.update(b"\0")
        digest.update(item.encode("utf-8"))
    return digest.hexdigest()[:20]


def encode_texts(texts: list[str], *, model, processor, device: str, batch_size: int = 16) -> np.ndarray:
    """Encode unique text with frozen CLIP; L2-normalize for cosine similarity."""
    import torch
    if batch_size < 1:
        raise ValueError("batch_size must be >=1")
    all_vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        tokens = processor(text=batch, padding=True, truncation=True, max_length=77, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.inference_mode():
            vectors = model.get_text_features(**tokens)
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=-1)
        all_vectors.append(vectors.float().cpu().numpy())
    return np.concatenate(all_vectors) if all_vectors else np.empty((0, 0), dtype=np.float32)


def load_or_encode_headlines(
    news: pd.DataFrame, *, cache_dir: str | Path, model_id: str = MODEL_ID, batch_size: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """Return vectors for news rows and prompt vectors, caching by exact text.

    Cache identity includes model ID, headline order/content, and prompts. It
    never reuses embeddings for a different set of headlines.
    """
    import torch
    from transformers import CLIPModel, CLIPProcessor

    headlines = news["headline_text"].astype(str).tolist()
    all_text = headlines + EVENT_PROMPTS
    key = _cache_key(all_text, model_id)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"clip_headlines_{key}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as arrays:
            return arrays["headlines"], arrays["prompts"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(model_id, cache_dir=str(cache_dir))
    model = CLIPModel.from_pretrained(model_id, cache_dir=str(cache_dir)).to(device)
    model.requires_grad_(False)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("CLIP must remain frozen")
    vectors = encode_texts(all_text, model=model, processor=processor, device=device, batch_size=batch_size)
    headline_vectors, prompt_vectors = vectors[: len(headlines)], vectors[len(headlines) :]
    if not np.isfinite(vectors).all():
        raise ValueError("Nonfinite CLIP text features")
    np.savez_compressed(cache, headlines=headline_vectors, prompts=prompt_vectors)
    return headline_vectors, prompt_vectors


def make_daily_clip_table(
    mapped_news: pd.DataFrame, headline_vectors: np.ndarray, prompt_vectors: np.ndarray
) -> pd.DataFrame:
    """Average embeddings by eligible forecast date, plus prompt cosines/counts."""
    if len(mapped_news) != headline_vectors.shape[0] or headline_vectors.ndim != 2:
        raise ValueError("One 2D embedding per input headline is required")
    if prompt_vectors.shape != (len(EVENT_PROMPTS), headline_vectors.shape[1]):
        raise ValueError("Prompt embedding dimensions do not match headlines")
    if mapped_news.empty:
        cols = ["date", "news_count", "has_news"] + [f"clip_{i:03d}" for i in range(headline_vectors.shape[1])] + [f"event_{i:02d}" for i in range(len(EVENT_PROMPTS))]
        return pd.DataFrame(columns=cols)
    vec_cols = [f"clip_{i:03d}" for i in range(headline_vectors.shape[1])]
    table = pd.DataFrame(headline_vectors, columns=vec_cols)
    table["date"] = pd.to_datetime(mapped_news["date"].to_numpy())
    daily = table.groupby("date", sort=True)[vec_cols].mean().reset_index()
    counts = table.groupby("date").size().rename("news_count").reset_index()
    daily = daily.merge(counts, on="date", validate="one_to_one")
    daily["has_news"] = 1.0
    aggregated = daily[vec_cols].to_numpy(dtype=float)
    norms = np.linalg.norm(aggregated, axis=1, keepdims=True)
    normalized = aggregated / np.maximum(norms, 1e-12)
    for i, col in enumerate(EVENT_PROMPTS):
        daily[f"event_{i:02d}"] = normalized @ prompt_vectors[i]
    return daily
