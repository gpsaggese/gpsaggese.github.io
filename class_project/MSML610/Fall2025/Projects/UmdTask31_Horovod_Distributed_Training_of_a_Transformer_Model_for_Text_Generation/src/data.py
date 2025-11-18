"""
Data loading utilities for BookCorpus Open dataset.

Handles:
- Loading dataset from HuggingFace
- Tokenization with GPT-2 tokenizer
- Creating distributed DataLoaders with Horovod
- Synthetic fallback data for testing
"""

import os
from typing import Optional, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import GPT2TokenizerFast

from .utils.distributed import get_rank, get_world_size, print_once


class TextDataset(Dataset):
    """
    Dataset for tokenized text sequences.
    
    Memory-efficient: Keeps reference to HuggingFace Dataset instead of
    materializing all data into Python lists.
    """
    
    def __init__(self, tokenized_data, max_length: int = 512):
        """
        Initialize dataset.
        
        Args:
            tokenized_data: HuggingFace Dataset with 'input_ids' and 'attention_mask',
                          or dict with lists (for backward compatibility).
            max_length: Maximum sequence length.
        """
        # Keep reference to dataset to avoid materializing all data
        self.dataset = tokenized_data
        self.max_length = max_length
        
        # Check if it's a HuggingFace Dataset or dict
        # Dict also has __getitem__/__len__, so check explicitly
        if isinstance(tokenized_data, dict):
            # Dict with lists (backward compatibility)
            self.is_hf_dataset = False
            self.input_ids = tokenized_data.get('input_ids', [])
            self.attention_mask = tokenized_data.get('attention_mask', [])
        elif hasattr(tokenized_data, 'column_names'):
            # HuggingFace Dataset (has column_names attribute)
            self.is_hf_dataset = True
        elif hasattr(tokenized_data, '__len__') and hasattr(tokenized_data, '__getitem__'):
            # Assume HuggingFace Dataset if it has these methods but isn't a dict
            self.is_hf_dataset = True
        else:
            # Fallback: treat as dict-like
            self.is_hf_dataset = False
            self.input_ids = tokenized_data.get('input_ids', []) if hasattr(tokenized_data, 'get') else []
            self.attention_mask = tokenized_data.get('attention_mask', []) if hasattr(tokenized_data, 'get') else []
        
    def __len__(self) -> int:
        if self.is_hf_dataset:
            return len(self.dataset)
        else:
            return len(self.input_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item.
        
        Returns:
            Dictionary with 'input_ids', 'attention_mask', and 'labels'.
        """
        if self.is_hf_dataset:
            # Fetch from HuggingFace Dataset (lazy, memory-efficient)
            ex = self.dataset[idx]
            input_ids = torch.tensor(ex['input_ids'], dtype=torch.long)
            attention_mask = torch.tensor(ex['attention_mask'], dtype=torch.long)
        else:
            # Backward compatibility: materialize from lists
            input_ids = torch.tensor(self.input_ids[idx], dtype=torch.long)
            attention_mask = torch.tensor(self.attention_mask[idx], dtype=torch.long)
        
        # For language modeling, labels are the same as input_ids
        # (shifted by 1 inside the model loss computation)
        labels = input_ids.clone()
        # Set padding positions to -100 (ignored by loss)
        # This distinguishes real EOS tokens from padding when EOS is used as PAD
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class SyntheticTextDataset(Dataset):
    """
    Synthetic dataset for testing when real data unavailable.
    """
    
    def __init__(self, num_samples: int = 1000, seq_length: int = 512, vocab_size: int = 50257):
        """
        Initialize synthetic dataset.
        
        Args:
            num_samples: Number of synthetic samples.
            seq_length: Sequence length.
            vocab_size: Vocabulary size.
        """
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Generate a random sample."""
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,), dtype=torch.long)
        attention_mask = torch.ones(self.seq_length, dtype=torch.long)
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def load_and_tokenize_dataset(
    dataset_name: str = "lucadiliello/bookcorpusopen",
    dataset_split: str = "train",
    max_samples: Optional[int] = None,
    validation_split: float = 0.05,
    max_length: int = 512,
    cache_dir: Optional[str] = None,
    pack_to_max_length: bool = False
) -> Tuple[TextDataset, TextDataset]:
    """
    Load and tokenize BookCorpus Open dataset.
    
    Args:
        dataset_name: HuggingFace dataset identifier.
        dataset_split: Dataset split to load.
        max_samples: Maximum number of samples to use (None for all).
        validation_split: Fraction of data for validation.
        max_length: Maximum sequence length.
        cache_dir: Cache directory for datasets and tokenizers.
        
    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    rank = get_rank()
    
    # Setup cache directory (prioritize environment variables for cluster environments)
    if cache_dir is None:
        # Try HF_HOME, TRANSFORMERS_CACHE, DATASET_CACHE, then fall back to default
        cache_dir = os.environ.get('HF_HOME') or \
                    os.environ.get('TRANSFORMERS_CACHE') or \
                    os.environ.get('DATASET_CACHE') or \
                    os.path.expanduser('~/.cache/huggingface')
    
    print_once(f"[INFO] Using cache directory: {cache_dir}")
    
    # Load tokenizer
    print_once(f"[INFO] Loading GPT-2 tokenizer...")
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', cache_dir=cache_dir)
        tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have pad token
    except Exception as e:
        print_once(f"[ERROR] Failed to load tokenizer: {e}")
        print_once("[INFO] Falling back to synthetic dataset.")
        return create_synthetic_datasets(max_samples or 1000, max_length)
    
    # Load dataset
    print_once(f"[INFO] Loading dataset: {dataset_name} (split: {dataset_split})...")
    try:
        from datasets import load_dataset
        from .utils.distributed import barrier
        
        # Load dataset (rank 0 downloads first, then barrier synchronization)
        if rank == 0:
            dataset = load_dataset(dataset_name, split=dataset_split, cache_dir=cache_dir)
        
        # Synchronize all processes before other ranks attempt to load
        barrier()
        
        if rank != 0:
            dataset = load_dataset(dataset_name, split=dataset_split, cache_dir=cache_dir)
        
        print_once(f"[INFO] Loaded {len(dataset)} samples.")
        
        # Limit samples if specified
        if max_samples is not None and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
            print_once(f"[INFO] Using {max_samples} samples.")
        
        # Split into train and validation
        split_idx = int(len(dataset) * (1 - validation_split))
        train_data = dataset.select(range(split_idx))
        val_data = dataset.select(range(split_idx, len(dataset)))
        
        print_once(f"[INFO] Train samples: {len(train_data)}, Val samples: {len(val_data)}")
        
        # Tokenize
        print_once(f"[INFO] Tokenizing dataset (max_length={max_length})...")
        
        def tokenize_function(examples):
            """Tokenize text samples."""
            if pack_to_max_length:
                # Pack later: do not pad, do not truncate here
                return tokenizer(
                    examples['text'],
                    truncation=False,
                    padding=False,
                    return_attention_mask=False,
                    return_tensors=None
                )
            else:
                return tokenizer(
                    examples['text'],
                    truncation=True,
                    padding='max_length',
                    max_length=max_length,
                    return_tensors=None
                )
        
        # Tokenize in batches (with parallelism for faster processing)
        import os
        num_proc = min(os.cpu_count() or 4, 8)  # Cap at 8 to avoid overhead
        
        train_tokenized = train_data.map(
            tokenize_function,
            batched=True,
            remove_columns=train_data.column_names,
            desc="Tokenizing train",
            num_proc=num_proc
        )

        val_tokenized = val_data.map(
            tokenize_function,
            batched=True,
            remove_columns=val_data.column_names,
            desc="Tokenizing validation",
            num_proc=num_proc
        )

        # Optional: group texts to fixed-length blocks to reduce padding
        if pack_to_max_length:
            block_size = max_length

            def group_texts(examples):
                # Concatenate all texts
                concatenated = {k: sum(examples[k], []) for k in examples.keys()}
                total_length = len(concatenated['input_ids'])
                total_length = (total_length // block_size) * block_size
                result = {}
                for k, t in concatenated.items():
                    result[k] = [t[i:i + block_size] for i in range(0, total_length, block_size)]
                # Create attention masks (all ones, no padding inside blocks)
                result['attention_mask'] = [[1] * block_size for _ in range(len(result['input_ids']))]
                return result

            train_tokenized = train_tokenized.map(
                group_texts,
                batched=True,
                desc="Grouping train into blocks",
                num_proc=max(1, num_proc // 2)
            )

            val_tokenized = val_tokenized.map(
                group_texts,
                batched=True,
                desc="Grouping val into blocks",
                num_proc=max(1, num_proc // 2)
            )
        
        # Convert to PyTorch datasets
        train_dataset = TextDataset(train_tokenized, max_length)
        val_dataset = TextDataset(val_tokenized, max_length)
        
        print_once("[INFO] Dataset loaded and tokenized successfully.")
        
        return train_dataset, val_dataset
        
    except Exception as e:
        print_once(f"[ERROR] Failed to load dataset: {e}")
        print_once("[INFO] Falling back to synthetic dataset.")
        return create_synthetic_datasets(max_samples or 1000, max_length)


def create_synthetic_datasets(
    num_samples: int = 1000,
    seq_length: int = 512,
    vocab_size: int = 50257,
    validation_split: float = 0.1
) -> Tuple[SyntheticTextDataset, SyntheticTextDataset]:
    """
    Create synthetic datasets for testing.
    
    Args:
        num_samples: Total number of samples.
        seq_length: Sequence length.
        vocab_size: Vocabulary size.
        validation_split: Fraction for validation.
        
    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    num_train = int(num_samples * (1 - validation_split))
    num_val = num_samples - num_train
    
    train_dataset = SyntheticTextDataset(num_train, seq_length, vocab_size)
    val_dataset = SyntheticTextDataset(num_val, seq_length, vocab_size)
    
    print_once(f"[INFO] Created synthetic datasets: {num_train} train, {num_val} val")
    
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
    
    # DataLoader improvements for throughput
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


def get_dataloaders(config: Any) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders from configuration.
    
    Args:
        config: Configuration object.
        
    Returns:
        Tuple of (train_dataloader, val_dataloader).
    """
    # Load and tokenize dataset
    train_dataset, val_dataset = load_and_tokenize_dataset(
        dataset_name=config.data.dataset_name,
        dataset_split=config.data.dataset_split,
        max_samples=config.data.get('max_samples', None),
        validation_split=config.data.validation_split,
        max_length=config.model.get('max_seq_len', 512),
        cache_dir=os.environ.get('HF_HOME', None),
        pack_to_max_length=config.data.get('pack_to_max_length', False)
    )
    
    # Create DataLoaders
    train_dataloader = create_distributed_dataloader(
        train_dataset,
        batch_size=config.training.per_gpu_batch_size,
        shuffle=config.data.get('shuffle', True),
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
