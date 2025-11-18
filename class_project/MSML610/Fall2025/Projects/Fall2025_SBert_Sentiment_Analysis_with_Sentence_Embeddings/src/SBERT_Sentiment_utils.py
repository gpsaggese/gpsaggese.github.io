# src/SBERT_Sentiment_utils.py

from .preprocess import load_config as _load_config_pre, preprocess as _preprocess, main as preprocess_main
from .sbert_embed import main as embed_main

# Thin wrappers the notebooks can call

def run_preprocessing(config_path: str = "config.yaml"):
    """Run the preprocessing pipeline (CSV -> cleaned CSV + labels.npy)."""
    preprocess_main(config_path)

def run_embedding(config_path: str = "config.yaml"):
    """Run the SBERT embedding pipeline (cleaned CSV -> sbert_embeddings.npy)."""
    embed_main(config_path)