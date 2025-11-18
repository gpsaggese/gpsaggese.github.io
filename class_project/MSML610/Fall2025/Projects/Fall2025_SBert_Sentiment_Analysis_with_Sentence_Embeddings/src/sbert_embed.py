import argparse
import os
import yaml
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(config_path: str):
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    prep_cfg = cfg["preprocessing"]
    embed_cfg = cfg["embeddings"]

    cleaned_csv = data_cfg["cleaned_csv"]
    embeddings_npy = data_cfg["embeddings_npy"]
    text_col = prep_cfg["text_column"]

    model_name = embed_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    batch_size = embed_cfg.get("batch_size", 64)
    device = embed_cfg.get("device", "cpu")

    df = pd.read_csv(cleaned_csv)
    sentences = df[text_col].astype(str).tolist()

    model = SentenceTransformer(model_name, device=device)
    print(f"Encoding {len(sentences)} sentences with {model_name} on {device}...")

    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    os.makedirs(os.path.dirname(embeddings_npy), exist_ok=True)
    np.save(embeddings_npy, embeddings)
    print(f"Saved embeddings with shape {embeddings.shape} to {embeddings_npy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    args = parser.parse_args()
    main(args.config)