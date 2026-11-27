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
# Token-level features (H3)
# #############################################################################

# Image tokens kept per post: CLS + 8x8 pooled patches.
IMAGE_TOKENS = 65
# Image patch grid of ViT-L/14 at 224x224 (16x16 = 256 patches).
_PATCH_GRID = 16
# Pooled patch grid.
_POOLED_GRID = 8


def _pool_image_tokens(hidden: torch.Tensor) -> torch.Tensor:
    """
    Reduce ViT tokens from 257 (CLS + 16x16) to 65 (CLS + 8x8).

    :param hidden: `last_hidden_state` of shape (B, 257, 1024)
    :return: tensor of shape (B, 65, 1024)
    """
    cls = hidden[:, :1]
    patches = hidden[:, 1:]
    b, n, d = patches.shape
    if n != _PATCH_GRID * _PATCH_GRID:
        raise ValueError(f"Expected {_PATCH_GRID**2} patches, got {n}")
    # (B, 256, D) -> (B, D, 16, 16) -> average 2x2 blocks -> (B, D, 8, 8).
    grid = patches.transpose(1, 2).reshape(b, d, _PATCH_GRID, _PATCH_GRID)
    k = _PATCH_GRID // _POOLED_GRID
    pooled = torch.nn.functional.avg_pool2d(grid, kernel_size=k)
    pooled = pooled.flatten(2).transpose(1, 2)
    return torch.cat([cls, pooled], dim=1)


@torch.inference_mode()
def embed_image_tokens(
    image_paths: List[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    *,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Encode images into pooled ViT tokens (CLS + 8x8 patches).

    :return: float16 array of shape `(len(image_paths), 65, 1024)`
    """
    device = next(model.parameters()).device
    images = []
    for path in image_paths:
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    feats = []
    for start in range(0, len(images), batch_size):
        inputs = processor(
            images=images[start : start + batch_size], return_tensors="pt"
        ).to(device)
        hidden = model.vision_model(
            pixel_values=inputs["pixel_values"]
        ).last_hidden_state
        feats.append(_pool_image_tokens(hidden).half().cpu().numpy())
    return np.concatenate(feats)


@torch.inference_mode()
def embed_text_tokens(
    texts: List[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    *,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode texts into token features, padded to 77 tokens.

    :return: float16 array `(len(texts), 77, 768)` and uint8 mask
        `(len(texts), 77)` with 1 for real tokens and 0 for padding
    """
    device = next(model.parameters()).device
    feats, masks = [], []
    for start in range(0, len(texts), batch_size):
        inputs = processor(
            text=list(texts[start : start + batch_size]),
            padding="max_length",
            truncation=True,
            max_length=MAX_TEXT_TOKENS,
            return_tensors="pt",
        ).to(device)
        hidden = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        ).last_hidden_state
        feats.append(hidden.half().cpu().numpy())
        masks.append(inputs["attention_mask"].to(torch.uint8).cpu().numpy())
    return np.concatenate(feats), np.concatenate(masks)


def _open_array(
    path: str, shape: Tuple[int, ...], dtype: Any, overwrite: bool
) -> np.ndarray:
    """
    Open a `.npy` file on disk as a writable memory map, creating it if needed.
    """
    if os.path.exists(path) and not overwrite:
        arr = np.load(path, mmap_mode="r+")
        if arr.shape == shape and arr.dtype == dtype:
            return arr
        _LOG.warning("%s has the wrong shape: recreating it", path)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def extract_token_features(
    df: pd.DataFrame,
    out_dir: str,
    kind: str,
    *,
    model: Optional[CLIPModel] = None,
    processor: Optional[CLIPProcessor] = None,
    chunk_size: int = 256,
    overwrite: bool = False,
) -> None:
    """
    Extract token-level CLIP features for H3 and save them in `out_dir`.

    Files (rows in the same order as `df`):
    - `ids.npy` (N,)
    - `image_tokens.npy` (N, 65, 1024) float16, for `kind="image"`
    - `text_tokens.npy` (N, 77, 768) float16 and `text_mask.npy` (N, 77),
      for `kind="text"`

    The arrays are written chunk by chunk to memory-mapped files, so the
    full data never has to fit in RAM. If the run is interrupted, calling the
    function again resumes after the last finished chunk.

    :param kind: `image` (slow, about 2-3 h on CPU) or `text` (a few minutes)
    """
    if kind not in ("image", "text"):
        raise ValueError(f"Unknown kind: {kind}")
    os.makedirs(out_dir, exist_ok=True)
    ids = df["id"].to_numpy()
    ids_path = os.path.join(out_dir, "ids.npy")
    if os.path.exists(ids_path) and not np.array_equal(np.load(ids_path), ids):
        raise ValueError(f"{ids_path} does not match `df`: use a new out_dir")
    np.save(ids_path, ids)
    n = len(df)
    if model is None or processor is None:
        model, processor = load_clip()
    # One flag per chunk, saved after the chunk is written.
    n_chunks = math.ceil(n / chunk_size)
    done_path = os.path.join(out_dir, f"{kind}_done.npy")
    if os.path.exists(done_path) and not overwrite:
        done = np.load(done_path)
    else:
        done = np.zeros(n_chunks, dtype=bool)
    if kind == "image":
        feats = _open_array(
            os.path.join(out_dir, "image_tokens.npy"),
            (n, IMAGE_TOKENS, model.config.vision_config.hidden_size),
            np.float16,
            overwrite,
        )
        paths = df["image_path"].tolist()
    else:
        feats = _open_array(
            os.path.join(out_dir, "text_tokens.npy"),
            (n, MAX_TEXT_TOKENS, model.config.text_config.hidden_size),
            np.float16,
            overwrite,
        )
        mask = _open_array(
            os.path.join(out_dir, "text_mask.npy"),
            (n, MAX_TEXT_TOKENS),
            np.uint8,
            overwrite,
        )
        texts = df["text"].tolist()
    _LOG.info("%s tokens: %d / %d chunks already done", kind, done.sum(), n_chunks)
    for k in tqdm(range(n_chunks), desc=f"{kind} tokens"):
        if done[k]:
            continue
        rows = slice(k * chunk_size, min((k + 1) * chunk_size, n))
        if kind == "image":
            feats[rows] = embed_image_tokens(paths[rows], model, processor)
        else:
            feats[rows], mask[rows] = embed_text_tokens(
                texts[rows], model, processor
            )
            mask.flush()
        feats.flush()
        done[k] = True
        np.save(done_path, done)
    _LOG.info("Saved %s token features for %d posts to %s", kind, n, out_dir)


def load_token_features(out_dir: str) -> Dict[str, np.ndarray]:
    """
    Load the H3 token features as read-only memory maps (no RAM copy).

    :return: dict with `ids`, `image` (N, 65, 1024), `text` (N, 77, 768),
        `mask` (N, 77)
    """
    def _load(name: str) -> np.ndarray:
        return np.load(os.path.join(out_dir, name), mmap_mode="r")

    feats = {
        "ids": np.load(os.path.join(out_dir, "ids.npy")),
        "image": _load("image_tokens.npy"),
        "text": _load("text_tokens.npy"),
        "mask": _load("text_mask.npy"),
    }
    for kind in ("image", "text"):
        done = np.load(os.path.join(out_dir, f"{kind}_done.npy"))
        if not done.all():
            raise ValueError(f"{kind} token extraction is not finished")
    return feats


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


# #############################################################################
# Project data (`clip.example.ipynb`)
# #############################################################################


def load_project_data(
    labels_path: str, cache_path: str
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Load the post table and the cached CLIP embeddings, checking they match.

    :return: `(df, emb)` with rows of `emb["image"]` / `emb["text"]` aligned
        with the rows of `df`
    """
    df = pd.read_csv(labels_path)
    emb = load_embeddings(cache_path)
    if not np.array_equal(emb["ids"], df["id"].to_numpy()):
        raise ValueError(
            f"{cache_path} does not match {labels_path}: rerun the extraction"
        )
    return df, emb


# Input feature sets compared in the baselines.
FEATURE_SETS = ["image", "text", "concat"]


def build_features(emb: Dict[str, np.ndarray], kind: str) -> np.ndarray:
    """
    Build the input matrix for one feature set.

    - `image`: image embedding only (768-d)
    - `text`: text embedding only (768-d)
    - `concat`: image and text embeddings side by side (1536-d)
    """
    if kind == "image":
        return emb["image"]
    if kind == "text":
        return emb["text"]
    if kind == "concat":
        return np.concatenate([emb["image"], emb["text"]], axis=1)
    raise ValueError(f"Unknown feature set '{kind}'")


def split_xy(
    df: pd.DataFrame, X: np.ndarray, split: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select the rows of one split and return `(X, y)` with integer labels.
    """
    mask = (df["split"] == split).to_numpy()
    return X[mask], df.loc[mask, "label_id"].to_numpy()


# #############################################################################
# Evaluation
# #############################################################################


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute accuracy, macro F1, and per-class precision / recall.

    Macro F1 is the main metric: with ~66% positive posts, accuracy alone
    rewards always predicting "positive".
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_recall_fscore_support,
    )

    labels = list(range(len(LABELS)))
    prec, rec, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
    }
    for k, name in enumerate(LABELS):
        metrics[f"precision_{name}"] = prec[k]
        metrics[f"recall_{name}"] = rec[k]
    return metrics


def results_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Turn `{model_name: metrics}` into a table sorted by macro F1.
    """
    table = pd.DataFrame(results).T
    return table.sort_values("macro_f1", ascending=False).round(3)


def plot_confusion_matrices(
    preds: Dict[str, Tuple[np.ndarray, np.ndarray]], *, ncols: int = 3
) -> None:
    """
    Plot row-normalized confusion matrices, one per model.

    Each row sums to 1, so the diagonal is the recall of each class.

    :param preds: `{model_name: (y_true, y_pred)}`
    """
    from sklearn.metrics import confusion_matrix

    n = len(preds)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows))
    axes = np.atleast_1d(axes).ravel()
    labels = list(range(len(LABELS)))
    for ax, (name, (y_true, y_pred)) in zip(axes, preds.items()):
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        for i in labels:
            for j in labels:
                color = "white" if cm[i, j] > 0.6 else "black"
                ax.text(
                    j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color=color,
                )
        ax.set_xticks(labels)
        ax.set_xticklabels(LABELS)
        ax.set_yticks(labels)
        ax.set_yticklabels(LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(name, fontsize=10)
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# #############################################################################
# Baselines
# #############################################################################


def majority_baseline(y_train: np.ndarray, n: int) -> np.ndarray:
    """
    Predict the most frequent training class for every post.
    """
    return np.full(n, np.bincount(y_train).argmax())


def train_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    c_grid: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
    seed: int = 42,
) -> Tuple[Any, float, pd.DataFrame]:
    """
    Train a class-weighted logistic regression, picking `C` on validation.

    `class_weight="balanced"` up-weights the rare negative class.

    :return: `(best_model, best_C, search_table)` where `search_table` lists
        the validation macro F1 for each `C`
    """
    from sklearn.linear_model import LogisticRegression

    rows, best = [], (None, None, -1.0)
    for c in c_grid:
        model = LogisticRegression(
            C=c, class_weight="balanced", max_iter=3000, random_state=seed
        )
        model.fit(X_train, y_train)
        f1 = evaluate(y_val, model.predict(X_val))["macro_f1"]
        rows.append({"C": c, "val_macro_f1": round(f1, 3)})
        if f1 > best[2]:
            best = (model, c, f1)
    _LOG.info("Best C=%s (val macro F1=%.3f)", best[1], best[2])
    return best[0], best[1], pd.DataFrame(rows)


# #############################################################################
# Fusion heads (H1, H2, H3)
# #############################################################################

# All heads are small PyTorch modules trained on top of the frozen CLIP
# features. Each head takes a batch of inputs and returns class logits
# of shape (B, 3).


class LinearHead(torch.nn.Module):
    """
    Baseline: standardize the input features and apply one linear layer.

    Trained with softmax cross-entropy, this is a (multinomial) logistic
    regression, trained with the same loop and epochs as the fusion heads.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray, n_classes: int = 3):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.fc = torch.nn.Linear(len(mean), n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc((x - self.mean) / self.std)


def make_linear_head(X: np.ndarray, train_idx: np.ndarray) -> LinearHead:
    """
    Build the linear baseline with standardization statistics from training.
    """
    x = X[train_idx]
    return LinearHead(x.mean(axis=0), x.std(axis=0) + 1e-6)


def feature_batch(X: np.ndarray):
    """
    Return a function `idx -> (x,)` over a feature matrix (linear baseline).
    """

    def _batch(idx: np.ndarray) -> Tuple[torch.Tensor, ...]:
        return (torch.from_numpy(np.asarray(X[idx], dtype=np.float32)),)

    return _batch


class StackHead(torch.nn.Module):
    """
    H1: standardize each modality, average with fixed weights 1/2, classify.

    The mean and std of each modality are computed on the training split.
    """

    def __init__(
        self,
        img_mean: np.ndarray,
        img_std: np.ndarray,
        txt_mean: np.ndarray,
        txt_std: np.ndarray,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        for name, value in [
            ("img_mean", img_mean),
            ("img_std", img_std),
            ("txt_mean", txt_mean),
            ("txt_std", txt_std),
        ]:
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        self.fc = torch.nn.Linear(len(img_mean), n_classes)

    def forward(self, txt: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        img = (img - self.img_mean) / self.img_std
        txt = (txt - self.txt_mean) / self.txt_std
        return self.fc(0.5 * img + 0.5 * txt)


def make_stack_head(emb: Dict[str, np.ndarray], train_idx: np.ndarray) -> StackHead:
    """
    Build H1 with standardization statistics from the training rows.
    """
    stats = []
    for kind in ("image", "text"):
        x = emb[kind][train_idx]
        stats += [x.mean(axis=0), x.std(axis=0) + 1e-6]
    return StackHead(*stats)


class GatedFusionHead(torch.nn.Module):
    """
    H2: A-norm gated fusion.

    The anchor modality is the main input; the other modality (the source)
    is added through a bounded, zero-initialized gate:
    - `u = LayerNorm([z | a])`, `c = cos(txt, img)`
    - `g = g_min + (g_max - g_min) * sigmoid(Linear([u | c]))`, one per post
    - `z <- z + g * a`, then a residual FFN block and a small classifier

    After each forward pass, the gate values are kept in `self.last_gate`
    (shape (B, 1)) for analysis.
    """

    def __init__(
        self,
        *,
        anchor: str = "text",
        in_dim: int = 768,
        hidden: int = 256,
        ffn_dim: int = 512,
        cls_dim: int = 128,
        dropout: float = 0.2,
        g_min: float = 0.05,
        g_max: float = 0.95,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        if anchor not in ("text", "image"):
            raise ValueError(f"Unknown anchor: {anchor}")
        self.anchor = anchor
        self.g_min, self.g_max = g_min, g_max
        nn = torch.nn
        self.proj_anchor = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden)
        )
        self.proj_source = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden)
        )
        # Gate stabilizer 1: normalize the gate input.
        self.gate_norm = nn.LayerNorm(2 * hidden)
        self.gate = nn.Linear(2 * hidden + 1, 1)
        # Gate stabilizer 2: zero init, so g = (g_min + g_max) / 2 at start.
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.n1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden),
        )
        self.n2 = nn.LayerNorm(hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, cls_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cls_dim, n_classes),
        )
        self.last_gate: Optional[torch.Tensor] = None

    def forward(self, txt: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        anchor, source = (txt, img) if self.anchor == "text" else (img, txt)
        z = self.proj_anchor(anchor)
        a = self.proj_source(source)
        u = self.gate_norm(torch.cat([z, a], dim=-1))
        # Cached CLIP embeddings are L2-normalized: the dot product is the
        # cosine similarity.
        c = (txt * img).sum(dim=-1, keepdim=True)
        s = self.gate(torch.cat([u, c], dim=-1))
        # Gate stabilizer 3: g stays in [g_min, g_max].
        g = self.g_min + (self.g_max - self.g_min) * torch.sigmoid(s)
        self.last_gate = g.detach()
        z = z + g * a
        h = self.n2(z + self.ffn(self.n1(z)))
        return self.classifier(h)


class _CrossModalBlock(torch.nn.Module):
    """
    One cross-modal Transformer block: queries from one modality attend to
    keys / values from the other, followed by an FFN (post-norm).
    """

    def __init__(self, d: int, heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        nn = torch.nn
        self.attn = nn.MultiheadAttention(
            d, heads, dropout=dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d),
        )
        self.ln2 = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_padding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        :param kv_padding: (B, T) bool, True where `kv` is padding
        """
        out, _ = self.attn(
            q, kv, kv, key_padding_mask=kv_padding, need_weights=False
        )
        x = self.ln1(q + self.drop(out))
        return self.ln2(x + self.drop(self.ffn(x)))


class _AttnPool(torch.nn.Module):
    """
    Attention pooling: a learned score per token, softmax, weighted sum.
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.score = torch.nn.Linear(d, 1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        :param mask: (B, T) bool, True for real tokens
        """
        s = self.score(x).squeeze(-1)
        if mask is not None:
            s = s.masked_fill(~mask, float("-inf"))
        w = torch.softmax(s, dim=-1).unsqueeze(-1)
        return (w * x).sum(dim=1)


class TokenCrossAttentionHead(torch.nn.Module):
    """
    H3: token-level cross-modal attention.

    - Project image tokens (B, 65, 1024) and text tokens (B, T, 768) to 256.
    - CM t->i: image tokens query the text tokens.
    - CM i->t: text tokens query the image tokens.
    - Summarize each side with mean + attention pooling, concatenate, and
      classify with one linear layer.
    """

    def __init__(
        self,
        *,
        img_dim: int = 1024,
        txt_dim: int = 768,
        d: int = 256,
        heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        nn = torch.nn
        self.img_proj = nn.Sequential(nn.Linear(img_dim, d), nn.LayerNorm(d))
        self.txt_proj = nn.Sequential(nn.Linear(txt_dim, d), nn.LayerNorm(d))
        self.cm_t2i = _CrossModalBlock(d, heads, ffn_dim, dropout)
        self.cm_i2t = _CrossModalBlock(d, heads, ffn_dim, dropout)
        self.pool_img = _AttnPool(d)
        self.pool_txt = _AttnPool(d)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * d, n_classes)

    def forward(
        self,
        img_tok: torch.Tensor,
        txt_tok: torch.Tensor,
        txt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param txt_mask: (B, T), 1 for real tokens and 0 for padding
        """
        mask = txt_mask.bool()
        z_img = self.img_proj(img_tok)
        z_txt = self.txt_proj(txt_tok)
        # Q = image, K/V = text (padding masked); and the reverse.
        h_img = self.cm_t2i(z_img, z_txt, kv_padding=~mask)
        h_txt = self.cm_i2t(z_txt, z_img)
        m = mask.unsqueeze(-1).float()
        txt_mean = (h_txt * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        s_img = torch.cat([h_img.mean(dim=1), self.pool_img(h_img)], dim=-1)
        s_txt = torch.cat([txt_mean, self.pool_txt(h_txt, mask)], dim=-1)
        return self.fc(self.drop(torch.cat([s_img, s_txt], dim=-1)))


def count_params(model: torch.nn.Module) -> int:
    """
    Count the trainable parameters of a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# #############################################################################
# Batches
# #############################################################################


def embedding_batch(emb: Dict[str, np.ndarray]):
    """
    Return a function `idx -> (txt, img)` over the cached 768-d embeddings
    (input of H1 and H2).
    """

    def _batch(idx: np.ndarray) -> Tuple[torch.Tensor, ...]:
        return (
            torch.from_numpy(np.asarray(emb["text"][idx], dtype=np.float32)),
            torch.from_numpy(np.asarray(emb["image"][idx], dtype=np.float32)),
        )

    return _batch


def token_batch(tok: Dict[str, np.ndarray]):
    """
    Return a function `idx -> (img_tok, txt_tok, txt_mask)` over the token
    features from `load_token_features()` (input of H3).

    Text tokens are cut to the longest tweet in the batch, to save time.
    """

    def _batch(idx: np.ndarray) -> Tuple[torch.Tensor, ...]:
        # Memory maps read faster with sorted indices.
        idx = np.sort(idx)
        mask = np.asarray(tok["mask"][idx])
        t = max(int(mask.sum(axis=1).max()), 1)
        img = np.asarray(tok["image"][idx], dtype=np.float32)
        txt = np.asarray(tok["text"][idx, :t], dtype=np.float32)
        return (
            torch.from_numpy(img),
            torch.from_numpy(txt),
            torch.from_numpy(mask[:, :t].astype(np.int64)),
        )

    return _batch


def split_indices(df: pd.DataFrame, split: str) -> np.ndarray:
    """
    Return the row positions of one split.
    """
    return np.where((df["split"] == split).to_numpy())[0]


# #############################################################################
# Training
# #############################################################################


def _class_weights(y: np.ndarray, n_classes: int = 3) -> torch.Tensor:
    """
    Balanced class weights `n / (k * count)`, as `class_weight="balanced"`.
    """
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    return torch.tensor(len(y) / (n_classes * counts), dtype=torch.float32)


@torch.no_grad()
def predict_head(
    model: torch.nn.Module,
    make_batch,
    idx: np.ndarray,
    *,
    batch_size: int = 256,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Predict class probabilities for the rows `idx` (in sorted order).

    :return: `(probs, gates)`; `gates` is the H2 gate per post, or None for
        the other heads
    """
    model.eval()
    idx = np.sort(idx)
    probs, gates = [], []
    for start in range(0, len(idx), batch_size):
        logits = model(*make_batch(idx[start : start + batch_size]))
        probs.append(torch.softmax(logits, dim=-1).numpy())
        gate = getattr(model, "last_gate", None)
        if gate is not None:
            gates.append(gate.squeeze(-1).numpy())
    return np.concatenate(probs), (np.concatenate(gates) if gates else None)


def train_head(
    model: torch.nn.Module,
    make_batch,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    patience: Optional[int] = None,
    seed: int = 42,
) -> Tuple[torch.nn.Module, pd.DataFrame]:
    """
    Train a fusion head with class-weighted cross-entropy and AdamW.

    The model is trained for exactly `epochs` epochs (unless `patience` is
    set for early stopping). After each epoch the validation macro F1 is
    computed, and the weights of the best epoch are restored at the end.

    :param make_batch: function `idx -> tuple of input tensors`
    :param y: integer labels for all rows of `df`
    :return: `(model, history)` with one row per epoch
    """
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    loss_fn = torch.nn.CrossEntropyLoss(weight=_class_weights(y[train_idx]))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    y_val = y[np.sort(val_idx)]
    best_f1, best_state, bad, rows = -1.0, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train_idx)
        total = 0.0
        for start in range(0, len(order), batch_size):
            idx = np.sort(order[start : start + batch_size])
            logits = model(*make_batch(idx))
            loss = loss_fn(logits, torch.from_numpy(y[idx]).long())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        probs, _ = predict_head(model, make_batch, val_idx)
        f1 = evaluate(y_val, probs.argmax(axis=1))["macro_f1"]
        rows.append(
            {"epoch": epoch, "train_loss": total / len(order), "val_macro_f1": f1}
        )
        _LOG.info(
            "epoch %d  train loss %.4f  val macro F1 %.4f",
            epoch, rows[-1]["train_loss"], f1,
        )
        if f1 > best_f1:
            best_f1, bad = f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if patience is not None and bad >= patience:
                break
    model.load_state_dict(best_state)
    best_epoch = int(pd.DataFrame(rows)["val_macro_f1"].idxmax()) + 1
    _LOG.info("Best val macro F1 %.4f at epoch %d", best_f1, best_epoch)
    return model, pd.DataFrame(rows)


def plot_history(histories: Dict[str, pd.DataFrame]) -> None:
    """
    Plot training loss and validation macro F1 per epoch for each head.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for name, hist in histories.items():
        axes[0].plot(hist["epoch"], hist["train_loss"], label=name)
        axes[1].plot(hist["epoch"], hist["val_macro_f1"], label=name)
    axes[0].set_title("Training loss")
    axes[1].set_title("Validation macro F1")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.tight_layout()
    plt.show()


def gate_table(gates: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """
    Summarize the H2 gate value g per true class.
    """
    df = pd.DataFrame({"label": [LABELS[k] for k in y], "g": gates})
    return df.groupby("label")["g"].describe()[["mean", "std", "min", "max"]].round(3)
