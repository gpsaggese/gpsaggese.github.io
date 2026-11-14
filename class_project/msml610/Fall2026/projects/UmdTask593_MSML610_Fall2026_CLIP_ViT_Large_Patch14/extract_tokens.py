"""
Extract token-level CLIP features for the H3 fusion head.

Usage (from the project directory, inside the Docker container):
    python extract_tokens.py text     # a few minutes
    python extract_tokens.py image    # about 2-3 h on CPU

If a run is interrupted, run the same command again: it resumes.
"""

import logging
import os
import sys

import pandas as pd
import torch

import clip_utils as cliputil

LABELS_PATH = "data/processed/labels.csv"
OUT_DIR = "data/processed/clip_tokens"


def main() -> None:
    kind = sys.argv[1] if len(sys.argv) > 1 else "text"
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    torch.set_num_threads(os.cpu_count())
    df = pd.read_csv(LABELS_PATH)
    cliputil.extract_token_features(df, OUT_DIR, kind)


if __name__ == "__main__":
    main()
