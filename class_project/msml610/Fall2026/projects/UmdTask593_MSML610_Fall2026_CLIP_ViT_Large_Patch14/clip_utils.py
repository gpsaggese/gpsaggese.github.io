"""
clip_utils.py

Utility functions supporting `clip.API.ipynb` and `clip.example.ipynb`.

The notebooks call these functions instead of writing raw logic inline.

Import as:

import clip_utils as cliputil
"""

import html
import logging
import math
import os
import re
import textwrap
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

# Store the Hugging Face model cache inside the project directory, so that
# the CLIP weights are downloaded once and survive container restarts.
# This must run before `transformers` is imported.
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(_PROJECT_DIR, "cache", "hf"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor

import helpers.hnotebook as hnotebo

_LOG = logging.getLogger(__name__)

# A few MVSA JPEGs are missing some trailing bytes: decode them anyway instead
# of failing, since the missing bytes barely affect the image.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Sentiment classes in a fixed order, used as integer class ids.
LABELS = ["negative", "neutral", "positive"]

# Pretrained CLIP checkpoint used throughout the project.
MODEL_ID = "openai/clip-vit-large-patch14"

# Maximum number of tokens the CLIP text encoder accepts.
MAX_TEXT_TOKENS = 77


def init_loggers(notebook_log: logging.Logger) -> None:
    global _LOG
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# Label processing
# #############################################################################


def _majority_vote(votes: List[str]) -> Optional[str]:
    """
    Return the label chosen by at least 2 of the 3 annotators.

    :param votes: labels from the 3 annotators
    :return: majority label, or `None` if all 3 annotators disagree
    """
    label, count = Counter(votes).most_common(1)[0]
    return label if count >= 2 else None


def _combine_text_image(
    text_label: Optional[str], image_label: Optional[str]
) -> Optional[str]:
    """
    Merge the text and image labels into one post label.

    Follow the rule of Xu & Mao (2017):
    - Same label: keep it.
    - One side neutral: take the non-neutral side.
    - Positive vs negative: drop the post.

    :return: post label, or `None` if the post is dropped
    """
    if text_label is None or image_label is None:
        return None
    if text_label == image_label:
        return text_label
    if text_label == "neutral":
        return image_label
    if image_label == "neutral":
        return text_label
    return None


def load_mvsa_labels(label_path: str) -> pd.DataFrame:
    """
    Parse `labelResultAll.txt` and aggregate the annotator votes.

    Each row of the file is `ID  t,i  t,i  t,i`, i.e., 3 annotators each
    giving a (text, image) sentiment pair.

    :param label_path: path to `labelResultAll.txt`
    :return: one row per post with columns `id`, `text_label`,
        `image_label`, `label` (`None` where the post is dropped)
    """
    rows = []
    with open(label_path, encoding="utf-8") as f:
        # Skip the header.
        next(f)
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            post_id = int(parts[0])
            votes = [p.split(",") for p in parts[1:4]]
            text_votes = [v[0] for v in votes]
            image_votes = [v[1] for v in votes]
            rows.append((post_id, text_votes, image_votes))
    df = pd.DataFrame(rows, columns=["id", "text_votes", "image_votes"])
    df["text_label"] = df["text_votes"].map(_majority_vote)
    df["image_label"] = df["image_votes"].map(_majority_vote)
    df["label"] = [
        _combine_text_image(t, i)
        for t, i in zip(df["text_label"], df["image_label"])
    ]
    df = df.drop(columns=["text_votes", "image_votes"])
    _LOG.info(
        "Loaded %d posts, %d kept after label aggregation",
        len(df),
        df["label"].notna().sum(),
    )
    return df


# #############################################################################
# Text and image files
# #############################################################################


def clean_tweet(text: str) -> str:
    """
    Normalize tweet text before feeding it to the CLIP text encoder.

    - Decode HTML entities stored by Twitter (e.g., `&amp;` -> `&`,
      `&lt;3` -> `<3`).
    - Remove URLs and the `RT` prefix.
    - Replace user mentions with `@user`.
    - Keep hashtag words but drop the `#` symbol.
    """
    text = html.unescape(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"^RT\s+", " ", text)
    text = re.sub(r"@\w+", "@user", text)
    text = text.replace("#", "")
    return re.sub(r"\s+", " ", text).strip()


def _read_text(path: str) -> Optional[str]:
    """
    Read a tweet text file, returning `None` if it is missing or empty.
    """
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()
    return text or None


def _is_valid_image(path: str) -> bool:
    """
    Check that an image file exists and can be decoded.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:  # pylint: disable=broad-except
        return False


def attach_files(df: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    """
    Attach the tweet text and image path of each post, dropping broken posts.

    :param df: output of `load_mvsa_labels()` with the dropped posts removed
    :param data_dir: directory with `<id>.txt` and `<id>.jpg` files
    :return: `df` with `text_raw`, `text`, `image_path` columns
    """
    df = df.copy()
    texts, images, reasons = [], [], []
    for post_id in tqdm(df["id"], desc="Checking files"):
        text = _read_text(os.path.join(data_dir, f"{post_id}.txt"))
        image_path = os.path.join(data_dir, f"{post_id}.jpg")
        reason = None
        if text is None:
            reason = "missing_or_empty_text"
        elif not _is_valid_image(image_path):
            reason = "missing_or_broken_image"
        texts.append(text)
        images.append(image_path)
        reasons.append(reason)
    df["text_raw"] = texts
    df["image_path"] = images
    df["drop_reason"] = reasons
    n_bad = df["drop_reason"].notna().sum()
    if n_bad > 0:
        _LOG.warning(
            "Dropping %d posts:\n%s",
            n_bad,
            df["drop_reason"].value_counts().to_string(),
        )
    df = df[df["drop_reason"].isna()].drop(columns=["drop_reason"])
    df["text"] = df["text_raw"].map(clean_tweet)
    return df.reset_index(drop=True)


# #############################################################################
# Train / validation / test split
# #############################################################################


def split_dataset(
    df: pd.DataFrame,
    *,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Add a `split` column with a stratified train / val / test split.

    Stratify on `label` so that each split keeps the class imbalance.
    """
    df = df.copy()
    train_val, test = train_test_split(
        df, test_size=test_frac, stratify=df["label"], random_state=seed
    )
    val_size = val_frac / (1 - test_frac)
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        stratify=train_val["label"],
        random_state=seed,
    )
    df.loc[train.index, "split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"
    return df


def build_mvsa_table(
    raw_dir: str,
    out_path: str,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run the full data preparation and save the result as a CSV file.

    :param raw_dir: directory containing `labelResultAll.txt` and `data/`
    :param out_path: output CSV path, e.g. `data/processed/labels.csv`
    :return: one row per usable post with columns `id`, `text`,
        `text_raw`, `image_path`, `text_label`, `image_label`, `label`,
        `label_id`, `split`
    """
    df = load_mvsa_labels(os.path.join(raw_dir, "labelResultAll.txt"))
    df = df[df["label"].notna()]
    df = attach_files(df, os.path.join(raw_dir, "data"))
    df["label_id"] = df["label"].map(LABELS.index)
    df = split_dataset(df, seed=seed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    _LOG.info("Saved %d posts to %s", len(df), out_path)
    return df


def summarize_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count posts per split and label, with the label share in each split.
    """
    counts = pd.crosstab(df["split"], df["label"])[LABELS]
    counts["total"] = counts.sum(axis=1)
    return counts.loc[["train", "val", "test"]]


# #############################################################################
# CLIP model
# #############################################################################


def get_device() -> str:
    """
    Return `cuda` if a GPU is available, otherwise `cpu`.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_clip(
    model_id: str = MODEL_ID, *, device: Optional[str] = None
) -> Tuple[CLIPModel, CLIPProcessor]:
    """
    Load the pretrained CLIP model in inference mode, with its processor.

    The model is frozen: it is only used to extract features.
    """
    device = device or get_device()
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_id)
    _LOG.info("Loaded %s on %s", model_id, device)
    return model, processor


def as_embedding(output: Any, expected_dim: int) -> torch.Tensor:
    """
    Extract the projected embedding from `get_*_features()`.

    Depending on the `transformers` version, `get_image_features()` and
    `get_text_features()` return either a tensor or a model output object.
    """
    if isinstance(output, torch.Tensor):
        emb = output
    else:
        emb = None
        for key in ("image_embeds", "text_embeds", "pooler_output"):
            emb = getattr(output, key, None)
            if emb is not None:
                break
        if emb is None:
            emb = output[0]
    if emb.shape[-1] != expected_dim:
        raise ValueError(
            f"Expected embeddings of size {expected_dim}, got {tuple(emb.shape)}"
        )
    return emb


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Scale each row to unit length, so that a dot product is a cosine similarity.
    """
    return x / x.norm(dim=-1, keepdim=True)


# #############################################################################
# Embedding extraction
# #############################################################################


@torch.inference_mode()
def embed_images(
    image_paths: List[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    *,
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encode images into L2-normalized CLIP embeddings.

    :return: array of shape `(len(image_paths), projection_dim)`
    """
    device = next(model.parameters()).device
    dim = model.config.projection_dim
    feats = []
    for start in tqdm(
        range(0, len(image_paths), batch_size),
        desc="Image embeddings",
        disable=not show_progress,
        leave=False,
    ):
        images = []
        for path in image_paths[start : start + batch_size]:
            with Image.open(path) as img:
                images.append(img.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt").to(device)
        emb = as_embedding(model.get_image_features(**inputs), dim)
        feats.append(_l2_normalize(emb).float().cpu().numpy())
    return np.concatenate(feats).astype(np.float32)


@torch.inference_mode()
def embed_texts(
    texts: List[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    *,
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encode texts into L2-normalized CLIP embeddings.

    Texts longer than 77 tokens are truncated.

    :return: array of shape `(len(texts), projection_dim)`
    """
    device = next(model.parameters()).device
    dim = model.config.projection_dim
    feats = []
    for start in tqdm(
        range(0, len(texts), batch_size),
        desc="Text embeddings",
        disable=not show_progress,
        leave=False,
    ):
        inputs = processor(
            text=list(texts[start : start + batch_size]),
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_TOKENS,
            return_tensors="pt",
        ).to(device)
        emb = as_embedding(model.get_text_features(**inputs), dim)
        feats.append(_l2_normalize(emb).float().cpu().numpy())
    return np.concatenate(feats).astype(np.float32)


def count_truncated_texts(texts: List[str], processor: CLIPProcessor) -> int:
    """
    Count how many texts exceed the 77-token limit of the CLIP text encoder.
    """
    lengths = [
        len(ids) for ids in processor.tokenizer(list(texts))["input_ids"]
    ]
    return int(sum(n > MAX_TEXT_TOKENS for n in lengths))


def load_embeddings(cache_path: str) -> Dict[str, np.ndarray]:
    """
    Load cached embeddings with keys `ids`, `image`, `text`.
    """
    with np.load(cache_path) as data:
        return {key: data[key] for key in ("ids", "image", "text")}


def update_text_embeddings(
    df: pd.DataFrame,
    cache_path: str,
    *,
    model: Optional[CLIPModel] = None,
    processor: Optional[CLIPProcessor] = None,
) -> Dict[str, np.ndarray]:
    """
    Recompute only the text embeddings in an existing cache.

    Use this after changing the text cleaning: the image embeddings, which
    are much slower to compute, are kept as they are.

    :param df: table with the same posts, in the same order, as the cache
    :param cache_path: existing cache from `extract_embeddings()`
    :return: updated cache
    """
    cache = load_embeddings(cache_path)
    if not np.array_equal(cache["ids"], df["id"].to_numpy()):
        raise ValueError(f"Cache {cache_path} does not match `df`")
    if model is None or processor is None:
        model, processor = load_clip()
    cache["text"] = embed_texts(df["text"].tolist(), model, processor)
    np.savez(cache_path, **cache)
    _LOG.info("Updated text embeddings in %s", cache_path)
    return cache


def extract_embeddings(
    df: pd.DataFrame,
    cache_path: str,
    *,
    model: Optional[CLIPModel] = None,
    processor: Optional[CLIPProcessor] = None,
    batch_size: int = 32,
    chunk_size: int = 1000,
    overwrite: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compute CLIP image and text embeddings for all posts, with caching.

    - If `cache_path` exists and matches the post ids of `df`, load it and
      skip the computation.
    - Otherwise, encode the posts in chunks of `chunk_size`, saving each
      chunk to disk. If the run is interrupted, calling the function again
      resumes from the last saved chunk.

    :param df: table from `build_mvsa_table()` with `id`, `image_path`,
        `text` columns
    :param cache_path: output file, e.g. `data/processed/clip_embeddings.npz`
    :return: dict with `ids` (N,), `image` (N, 768), `text` (N, 768), rows
        in the same order as `df`
    """
    ids = df["id"].to_numpy()
    if os.path.exists(cache_path) and not overwrite:
        cache = load_embeddings(cache_path)
        if np.array_equal(cache["ids"], ids):
            _LOG.info("Loaded cached embeddings from %s", cache_path)
            return cache
        _LOG.warning("Cache %s does not match `df`: recomputing", cache_path)
    if model is None or processor is None:
        model, processor = load_clip()
    chunk_dir = os.path.splitext(cache_path)[0] + "_chunks"
    os.makedirs(chunk_dir, exist_ok=True)
    image_paths = df["image_path"].tolist()
    texts = df["text"].tolist()
    n_chunks = math.ceil(len(df) / chunk_size)
    chunk_paths = []
    for k in tqdm(range(n_chunks), desc="Chunks"):
        chunk_path = os.path.join(chunk_dir, f"chunk_{k:04d}.npz")
        chunk_paths.append(chunk_path)
        if os.path.exists(chunk_path) and not overwrite:
            continue
        rows = slice(k * chunk_size, (k + 1) * chunk_size)
        image_emb = embed_images(
            image_paths[rows], model, processor, batch_size=batch_size
        )
        text_emb = embed_texts(texts[rows], model, processor)
        np.savez(chunk_path, ids=ids[rows], image=image_emb, text=text_emb)
    # Merge the chunks into a single cache file.
    parts = [load_embeddings(path) for path in chunk_paths]
    cache = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    if not np.array_equal(cache["ids"], ids):
        raise ValueError(
            f"Chunks in {chunk_dir} do not match `df`: delete the directory "
            "and run again"
        )
    np.savez(cache_path, **cache)
    _LOG.info(
        "Saved embeddings for %d posts to %s", len(cache["ids"]), cache_path
    )
    return cache


# #############################################################################
# Tutorial helpers (`clip.API.ipynb`)
# #############################################################################


def sample_posts(
    df: pd.DataFrame,
    *,
    n_per_label: int = 2,
    split: str = "test",
    seed: int = 0,
) -> pd.DataFrame:
    """
    Pick `n_per_label` random posts for each sentiment label.
    """
    subset = df[df["split"] == split]
    posts = subset.groupby("label").sample(n=n_per_label, random_state=seed)
    order = posts["label"].map(LABELS.index)
    return posts.iloc[order.argsort(kind="stable")].reset_index(drop=True)


def load_images(image_paths: List[str]) -> List[Image.Image]:
    """
    Load images as RGB, the format expected by the CLIP processor.
    """
    images = []
    for path in image_paths:
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    return images


def show_posts(
    posts: pd.DataFrame, *, ncols: int = 3, max_chars: int = 90
) -> None:
    """
    Show the image of each post with its label and (shortened) tweet text.
    """
    nrows = math.ceil(len(posts) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.8 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, row in zip(axes, posts.itertuples()):
        with Image.open(row.image_path) as img:
            ax.imshow(img.convert("RGB"))
        text = row.text
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        ax.set_title(
            f"id {row.id} [{row.label}]\n" + textwrap.fill(text, 40), fontsize=9
        )
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_similarity(
    sim: np.ndarray,
    *,
    row_labels: List[str],
    col_labels: List[str],
    title: str = "Cosine similarity (rows: images, columns: texts)",
) -> None:
    """
    Plot an image-text similarity matrix as an annotated heatmap.
    """
    fig, ax = plt.subplots(
        figsize=(1.3 * len(col_labels) + 2.5, 1.0 * len(row_labels) + 1.5)
    )
    im = ax.imshow(sim, cmap="Blues")
    ax.grid(False)
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            color = "white" if sim[i, j] > sim.mean() + sim.std() else "black"
            ax.text(
                j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                fontsize=9, color=color,
            )
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Text of post")
    ax.set_ylabel("Image of post")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def zero_shot_probs(
    emb: np.ndarray,
    class_prompts: List[str],
    model: CLIPModel,
    processor: CLIPProcessor,
) -> np.ndarray:
    """
    Classify embeddings by their similarity to one prompt per class.

    Follow CLIP: scale the cosine similarities by the learned temperature
    `logit_scale` and apply a softmax over the classes.

    :param emb: L2-normalized image or text embeddings, shape (N, 768)
    :param class_prompts: one sentence per class, in the order of `LABELS`
    :return: class probabilities, shape (N, len(class_prompts))
    """
    prompt_emb = embed_texts(class_prompts, model, processor, show_progress=False)
    scale = model.logit_scale.exp().item()
    logits = scale * emb @ prompt_emb.T
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    return probs / probs.sum(axis=1, keepdims=True)


def zero_shot_table(posts: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    """
    Show zero-shot class probabilities next to the true label of each post.
    """
    table = pd.DataFrame(probs.round(2), columns=LABELS, index=posts["id"])
    table["predicted"] = [LABELS[k] for k in probs.argmax(axis=1)]
    table["true"] = posts["label"].to_numpy()
    return table
