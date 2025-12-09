import argparse
import os
import numpy as np
import pandas as pd
import yaml


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def preprocess_financial_phrasebank(df: pd.DataFrame,
                                    text_col_name: str,
                                    label_col_name: str) -> pd.DataFrame:
    """
    Parse the original Financial PhraseBank CSV:

        "positive","Some sentence..."
        "neutral","Another sentence..."

    and return a DataFrame with:
        text_col_name   = sentence text
        label_col_name  = integer label {0,1,2}
    """
    # Raw file has 2 columns: label, sentence
    if df.shape[1] != 2:
        raise ValueError(
            f"Expected 2 columns in raw data (label, sentence). Got shape={df.shape}"
        )

    # Name them explicitly
    df.columns = ["label_raw", "sentence_raw"]

    # Clean up strings
    df["label_raw"] = (
        df["label_raw"]
        .astype(str)
        .str.strip()
        .str.strip('"')
        .str.lower()
    )
    df["sentence"] = (
        df["sentence_raw"]
        .astype(str)
        .str.strip()
        .str.strip('"')
    )

    # Map labels to integers
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    df["sentiment"] = df["label_raw"].map(label_map)

    # Drop anything with unknown labels or empty text
    before = len(df)
    df = df[df["sentiment"].notna()]
    df = df[df["sentence"].str.len() > 0]
    after = len(df)

    if after == 0:
        raise ValueError(
            "After preprocessing, no rows remain. "
            "Check that label strings are one of: "
            f"{list(label_map.keys())}"
        )

    # Rename columns to match config (text_col_name, label_col_name)
    df = df.rename(
        columns={
            "sentence": text_col_name,
            "sentiment": label_col_name,
        }
    )

    # Keep only the two useful columns
    df = df[[text_col_name, label_col_name]]

    return df


def main(config_path: str):
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    prep_cfg = cfg["preprocessing"]

    raw_csv = data_cfg["raw_csv"]
    cleaned_csv = data_cfg["cleaned_csv"]
    labels_npy = data_cfg["labels_npy"]

    text_col = prep_cfg["text_column"]     # "sentence"
    label_col = prep_cfg["label_column"]   # "sentiment"

    # Read the raw Financial PhraseBank CSV:
    #  - no header
    #  - two columns: label, sentence
    df_raw = pd.read_csv(
        raw_csv,
        header=None,
        encoding="latin1",
        sep=None,         # let pandas auto-detect comma/semicolon
        engine="python",
    )

    df_clean = preprocess_financial_phrasebank(df_raw, text_col, label_col)

    os.makedirs(os.path.dirname(cleaned_csv), exist_ok=True)
    df_clean.to_csv(cleaned_csv, index=False)
    np.save(labels_npy, df_clean[label_col].values)

    print(f"[preprocess] Saved cleaned CSV to {cleaned_csv}")
    print(f"[preprocess] Saved labels to {labels_npy}")
    print(df_clean[label_col].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    args = parser.parse_args()
    main(args.config)