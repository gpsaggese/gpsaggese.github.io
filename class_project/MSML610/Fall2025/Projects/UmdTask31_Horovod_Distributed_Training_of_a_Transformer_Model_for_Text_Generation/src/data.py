"""
Data loading utilities for preprocessed datasets.

Loads preprocessed tokenized data from disk (created by notebooks/00_data_preprocessing.ipynb).
"""

import os
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from datasets import load_from_disk

from .utils.distributed import get_rank, get_world_size, print_once


class TextDataset(Dataset):
    """
    Dataset for tokenized text sequences.
    
    Loads preprocessed data from disk.
    """
    
    def __init__(self, tokenized_data, max_length: int = 512):
        """
        Initialize dataset.
        
        Args:
            tokenized_data: HuggingFace Dataset with 'input_ids' and 'attention_mask'.
            max_length: Maximum sequence length (for validation).
        """
        self.dataset = tokenized_data
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a single item.
        
        Returns:
            Dictionary with 'input_ids', 'attention_mask', and 'labels'.
        """
        ex = self.dataset[idx]
        input_ids = torch.tensor(ex['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(ex['attention_mask'], dtype=torch.long)
        
        # For language modeling, labels are the same as input_ids
        # (shifted by 1 inside the model loss computation)
        labels = input_ids.clone()
        # Set padding positions to -100 (ignored by loss)
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def load_preprocessed_data(data_dir: str = "data/preprocessed") -> Tuple[TextDataset, TextDataset]:
    """
    Load preprocessed train and validation datasets from disk.
    
    Args:
        data_dir: Directory containing preprocessed data (should have 'train' and 'val' subdirectories).
        
    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data directory not found: {data_dir}\n"
            f"Please run notebooks/00_data_preprocessing.ipynb first to prepare the data."
        )
    
    train_path = data_path / "train"
    val_path = data_path / "val"
    
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found in {data_dir}\n"
            f"Expected subdirectories: train/ and val/\n"
            f"Please run notebooks/00_data_preprocessing.ipynb first."
        )
    
    print_once(f"[INFO] Loading preprocessed data from: {data_dir}")
    
    # Load datasets
    train_data = load_from_disk(str(train_path))
    val_data = load_from_disk(str(val_path))
    
    print_once(f"[INFO] Loaded {len(train_data):,} training samples")
    print_once(f"[INFO] Loaded {len(val_data):,} validation samples")
    
    # Get max_length from metadata if available
    metadata_path = data_path / "metadata.json"
    max_length = 512  # default
    if metadata_path.exists():
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            max_length = metadata.get('max_seq_length', 512)
    
    # Create datasets
    train_dataset = TextDataset(train_data, max_length)
    val_dataset = TextDataset(val_data, max_length)
    
    return train_dataset, val_dataset


def create_distributed_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True
) -> DataLoader:
    """
    Create a DataLoader with Horovod DistributedSampler.
    
    Args:
        dataset: PyTorch Dataset.
        batch_size: Batch size per GPU.
        shuffle: Whether to shuffle data.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for faster GPU transfer.
        drop_last: Whether to drop incomplete batches (True for train, False for val).
        
    Returns:
        PyTorch DataLoader with distributed sampling.
    """
    world_size = get_world_size()
    rank = get_rank()
    
    # Use DistributedSampler if distributed training
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )
        # Don't shuffle in DataLoader when using DistributedSampler
        dataloader_shuffle = False
    else:
        sampler = None
        dataloader_shuffle = shuffle
    
    # DataLoader configuration
    dataloader_kwargs = {
        'batch_size': batch_size,
        'sampler': sampler,
        'shuffle': dataloader_shuffle,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'drop_last': drop_last
    }
    
    # Add persistent workers and prefetch for better throughput
    if num_workers > 0:
        dataloader_kwargs['persistent_workers'] = True
        dataloader_kwargs['prefetch_factor'] = 2
    
    dataloader = DataLoader(dataset, **dataloader_kwargs)
    
    return dataloader


def get_dataloaders(config) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders from configuration.
    
    Args:
        config: Configuration object.
        
    Returns:
        Tuple of (train_dataloader, val_dataloader).
    """
    # Get data directory from config or use default
    data_dir = getattr(config.data, 'data_dir', 'data/preprocessed')
    
    # Load preprocessed datasets
    train_dataset, val_dataset = load_preprocessed_data(data_dir)
    
    # Create DataLoaders
    train_dataloader = create_distributed_dataloader(
        train_dataset,
        batch_size=config.training.per_gpu_batch_size,
        shuffle=True,
        num_workers=config.data.get('num_workers', 4),
        drop_last=True  # Drop incomplete batches for consistent training
    )
    
    val_dataloader = create_distributed_dataloader(
        val_dataset,
        batch_size=config.training.per_gpu_batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=config.data.get('num_workers', 4),
        drop_last=False  # Don't drop incomplete validation batches
    )
    
    print_once(f"[INFO] Created DataLoaders:")
    print_once(f"  - Train batches: {len(train_dataloader)}")
    print_once(f"  - Val batches: {len(val_dataloader)}")
    print_once(f"  - Global batch size: {config.training.per_gpu_batch_size * get_world_size()}")
    
    return train_dataloader, val_dataloader
