#!/usr/bin/env python
# coding: utf-8

# ================================================================
# 0. Imports and Configuration
# ================================================================

import os
import sys
from pathlib import Path
import glob
import json
import itertools

import polars as pl
from datasets import Dataset, load_from_disk
from transformers import GPT2TokenizerFast

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Config ---
DATASET_NAME = "lucadiliello/bookcorpusopen"
MAX_SEQ_LENGTH = 512
VALIDATION_SPLIT = 0.05      # 5% for validation
PACK_TO_MAX_LENGTH = True
MAX_SAMPLES = None           # Set to an int to subsample for testing

# Paths (relative to notebooks/ when run via SLURM script)
DATA_DIR = Path("data/preprocessed/v1")
DATA_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
LOCAL_DATA_DIR = "/scratch/zt1/project/msml610/user/vikranth/bookcorpus"

print(f"Cache directory: {CACHE_DIR}")
print(f"Output directory: {DATA_DIR}")


def hf_dataset_exists(path: Path) -> bool:
    """Return True if a HF dataset saved_to_disk exists at this path."""
    return (path / "dataset_info.json").exists()


# ================================================================
# 1. Load dataset from local parquet using Polars
# ================================================================

print("Loading BookCorpusOpen from LOCAL PARQUET FILES using Polars...")
files = sorted(glob.glob(f"{LOCAL_DATA_DIR}/train-*.parquet"))
print(f"Found {len(files)} parquet shards.")

if len(files) == 0:
    raise FileNotFoundError("No parquet files found!")

df = pl.concat([pl.read_parquet(f) for f in files], how="vertical")
print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")

dataset = Dataset.from_pandas(df.to_pandas(), preserve_index=False)
print("✓ Dataset loaded successfully")

if MAX_SAMPLES is not None:
    dataset = dataset.select(range(MAX_SAMPLES))
    print(f"Using subset of dataset: {len(dataset)} samples")
else:
    print(f"Using full dataset: {len(dataset)} samples")


# ================================================================
# 2. Split dataset
# ================================================================

split_idx = int(len(dataset) * (1 - VALIDATION_SPLIT))
train_data = dataset.select(range(split_idx))
val_data = dataset.select(range(split_idx, len(dataset)))

print(f"Train samples: {len(train_data)}")
print(f"Val samples:   {len(val_data)}")
print(f"Validation split: {VALIDATION_SPLIT * 100:.1f}%")

# ================================================================
# 3. Load tokenizer
# ================================================================

print("Loading GPT-2 tokenizer...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir=CACHE_DIR)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
print(f"Max model length:     {tokenizer.model_max_length}")

# Quick sanity check
test_text = "Hello, this is a test sentence."
test_tokens = tokenizer(test_text, return_tensors="pt")
print("\nTest tokenization:")
print(f"  Text:     {test_text}")
print(f"  Token IDs:{test_tokens['input_ids'][0].tolist()}")
print(f"  Decoded:  {tokenizer.decode(test_tokens['input_ids'][0])}")


# ================================================================
# 4. Tokenization with checkpoint support
# ================================================================

tokenized_train_path = DATA_DIR / "train_tokenized"
tokenized_val_path = DATA_DIR / "val_tokenized"


def tokenize_function(examples):
    # No truncation here; we pack into fixed-length blocks later
    return tokenizer(
        examples["text"],
        truncation=False,
        padding=False,
        return_attention_mask=False,
        return_tensors=None,
    )


if hf_dataset_exists(tokenized_train_path) and hf_dataset_exists(tokenized_val_path):
    print("\n✓ Found cached tokenized data — skipping tokenization")
    train_tokenized = load_from_disk(str(tokenized_train_path))
    val_tokenized = load_from_disk(str(tokenized_val_path))
else:
    print("\nRunning tokenization...")
    print("----------------------------------------")

    train_tokenized = train_data.map(
        tokenize_function,
        batched=True,
        remove_columns=train_data.column_names,
        num_proc=1,  # keep 1 to be safe on memory
        desc="Tokenizing train",
    )

    val_tokenized = val_data.map(
        tokenize_function,
        batched=True,
        remove_columns=val_data.column_names,
        num_proc=1,
        desc="Tokenizing val",
    )

    train_tokenized.save_to_disk(str(tokenized_train_path))
    val_tokenized.save_to_disk(str(tokenized_val_path))
    print("✓ Tokenization saved for future runs")

print(f"\nTrain tokenized samples: {len(train_tokenized)}")
print(f"Val tokenized samples:   {len(val_tokenized)}")


# ================================================================
# 5. Grouping into fixed-length blocks with checkpoint
# ================================================================

grouped_train_path = DATA_DIR / "train_grouped"
grouped_val_path = DATA_DIR / "val_grouped"


def group_texts(examples):
    # Concatenate all token lists from this batch
    all_ids = list(itertools.chain.from_iterable(examples["input_ids"]))
    total_len = (len(all_ids) // MAX_SEQ_LENGTH) * MAX_SEQ_LENGTH
    all_ids = all_ids[:total_len]

    # Split into fixed-size blocks
    blocks = [
        all_ids[i : i + MAX_SEQ_LENGTH]
        for i in range(0, total_len, MAX_SEQ_LENGTH)
    ]

    attention = [[1] * MAX_SEQ_LENGTH for _ in range(len(blocks))]
    labels = [b.copy() for b in blocks]

    return {
        "input_ids": blocks,
        "attention_mask": attention,
        "labels": labels,
    }


if hf_dataset_exists(grouped_train_path) and hf_dataset_exists(grouped_val_path):
    print("\n✓ Found cached grouped blocks — skipping grouping")
    train_tokenized = load_from_disk(str(grouped_train_path))
    val_tokenized = load_from_disk(str(grouped_val_path))
else:
    print("\nGrouping tokenized data into fixed-length blocks...")
    print("----------------------------------------")

    train_tokenized = train_tokenized.map(
        group_texts,
        batched=True,
        num_proc=1,
        desc="Grouping train into blocks",
    )

    val_tokenized = val_tokenized.map(
        group_texts,
        batched=True,
        num_proc=1,
        desc="Grouping val into blocks",
    )

    train_tokenized.save_to_disk(str(grouped_train_path))
    val_tokenized.save_to_disk(str(grouped_val_path))
    print("✓ Grouped blocks saved to disk")

print(f"\nAfter grouping:")
print(f"  Train blocks: {len(train_tokenized)}")
print(f"  Val blocks:   {len(val_tokenized)}")


# ================================================================
# 6. Save final processed data
# ================================================================

final_train_path = DATA_DIR / "train"
final_val_path = DATA_DIR / "val"

if hf_dataset_exists(final_train_path) and hf_dataset_exists(final_val_path):
    print("\n✓ Final preprocessed data exists — skipping final save")
else:
    print("\nSaving final preprocessed datasets...")
    train_tokenized.save_to_disk(str(final_train_path))
    val_tokenized.save_to_disk(str(final_val_path))
    tokenizer.save_pretrained(str(DATA_DIR / "tokenizer"))

    metadata = {
        "dataset_name": DATASET_NAME,
        "max_seq_length": MAX_SEQ_LENGTH,
        "validation_split": VALIDATION_SPLIT,
        "vocab_size": tokenizer.vocab_size,
        "train_samples": len(train_tokenized),
        "val_samples": len(val_tokenized),
        "pack_to_max_length": PACK_TO_MAX_LENGTH,
        "tokenizer_type": "gpt2",
    }

    with open(DATA_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("✓ Final data and metadata saved")

print("\n============================================================")
print("Data preprocessing complete")
print("============================================================")
