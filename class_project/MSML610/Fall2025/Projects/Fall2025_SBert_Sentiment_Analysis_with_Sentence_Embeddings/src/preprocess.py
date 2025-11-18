import argparse
import os
import yaml
import numpy as np
import pandas as pd


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def preprocess(df: pd.DataFrame, text_col: str, label_col: str):
    # keep only needed columns
    df = df[[text_col, label_col]].dropna()

    # simple cleaning
    df[text_col] = df[text_col].astype(str).str.strip()

    # map label strings -> ints (adjust to your actual labels)
    mapping = {"negative": 0, "neutral": 1, "positive": 2}
    df[label_col] = df[label_col].map(mapping)

    df = df[df[label_col].notna()]
    df[label_col] = df[label_col].astype(int)
    return df


def main(config_path: str):
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    prep_cfg = cfg["preprocessing"]

    raw_csv = data_cfg["raw_csv"]
    cleaned_csv = data_cfg["cleaned_csv"]
    labels_npy = data_cfg["labels_npy"]

    text_col = prep_cfg["text_column"]      # "sentence"
    label_col = prep_cfg["label_column"]    # "sentiment"

    df_raw = pd.read_csv(
        raw_csv,
        encoding="latin-1",
        header=None,
        names=[text_col, label_col],
    )

    df_clean = preprocess(df_raw, text_col, label_col)

    os.makedirs(os.path.dirname(cleaned_csv), exist_ok=True)
    df_clean.to_csv(cleaned_csv, index=False)
    np.save(labels_npy, df_clean[label_col].values)

    print(f"Saved cleaned CSV to {cleaned_csv}")
    print(f"Saved labels to {labels_npy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    args = parser.parse_args()
    main(args.config)