# src/SBERT_utils.py

import os
from typing import Dict, Any

import yaml
import numpy as np
import pandas as pd


# ---------- Config helpers ----------

def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config and return as a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_paths(cfg: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract all important paths from the config.
    Keeps the rest of the code from hard-coding paths.
    """
    data_cfg = cfg["data"]
    embed_cfg = cfg.get("embeddings", {})

    return {
        "raw_csv": data_cfg["raw_csv"],
        "cleaned_csv": data_cfg["cleaned_csv"],
        "labels_npy": data_cfg["labels_npy"],
        "embeddings_npy": embed_cfg.get("embeddings_npy", "data/processed/sbert_embeddings.npy"),
    }


# ---------- Data loading helpers ----------

def load_clean_data(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Load the preprocessed Financial PhraseBank data.

    Expects preprocess.py to have created the cleaned CSV.
    """
    paths = get_paths(cfg)
    cleaned_csv = paths["cleaned_csv"]

    if not os.path.exists(cleaned_csv):
        raise FileNotFoundError(
            f"Cleaned CSV not found at '{cleaned_csv}'. "
            "Run `python src/preprocess.py --config config.yaml` first."
        )

    df = pd.read_csv(cleaned_csv)
    return df


def load_embeddings(cfg: Dict[str, Any]) -> np.ndarray:
    """
    Load SBERT sentence embeddings.

    Expects sbert_embed.py to have created the .npy file.
    """
    paths = get_paths(cfg)
    emb_path = paths["embeddings_npy"]

    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f"Embeddings file not found at '{emb_path}'. "
            "Run `python src/sbert_embed.py --config config.yaml` first."
        )

    embeddings = np.load(emb_path)
    return embeddings