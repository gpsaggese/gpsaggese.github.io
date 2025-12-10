#!/usr/bin/env python3
"""
Standard PyTorch DistributedDataParallel training script for ViT-Large on ImageNet-1k.
Uses BF16 mixed precision and W&B logging.
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
import wandb


def setup_distributed():
    """Initialize distributed training."""
    # For SLURM or torchrun
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    
    # Initialize process group
    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://'
        )
    
    return device, rank, local_rank, world_size


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_model(num_labels=1000):
    """Load ViT-Large model."""
    print(f"Loading google/vit-large-patch16-224 model with {num_labels} classes...")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-large-patch16-224",
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.bfloat16
    )
    return model


def _filter_imagefolder_by_classes(dataset, allowed_class_names=None, max_per_class=0):
    """Optionally filter an ImageFolder dataset by class names and/or cap samples per class.

    Args:
        dataset: torchvision.datasets.ImageFolder instance
        allowed_class_names: Optional[Set[str]] of class directory names to keep
        max_per_class: int, if > 0 cap number of samples per class
    Returns:
        dataset (modified in place) and the number of kept samples
    """
    if allowed_class_names is None and max_per_class <= 0:
        return dataset, len(getattr(dataset, 'samples', getattr(dataset, 'imgs', [])))

    # Map class name -> class index present in this dataset
    class_to_idx = dataset.class_to_idx
    allowed_idx = None
    if allowed_class_names is not None:
        allowed_idx = {class_to_idx[c] for c in allowed_class_names if c in class_to_idx}

    samples = getattr(dataset, 'samples', getattr(dataset, 'imgs', []))
    targets = getattr(dataset, 'targets', None)

    kept = []
    kept_targets = []
    per_class_count = {}

    for i, (path, target) in enumerate(samples):
        if allowed_idx is not None and target not in allowed_idx:
            continue
        if max_per_class > 0:
            c = per_class_count.get(target, 0)
            if c >= max_per_class:
                continue
            per_class_count[target] = c + 1
        kept.append((path, target))
        if targets is not None:
            kept_targets.append(target)

    # Write back
    if hasattr(dataset, 'samples'):
        dataset.samples = kept
    if hasattr(dataset, 'imgs'):
        dataset.imgs = kept
    if targets is not None:
        dataset.targets = kept_targets

    return dataset, len(kept)


def get_data_loaders(data_path, batch_size, world_size, rank, num_workers=4,
                     allowed_class_names=None, max_per_class=0):
    """Create ImageNet/Food-101 data loaders. Returns loaders and number of classes."""
    # ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Load dataset from directory structure (ImageNet or Food-101 format)
    # Priority: 1) train/test (Food-101), 2) train/val (ImageNet), 3) food101_train/food101_val
    train_path = os.path.join(data_path, 'train')
    if not os.path.exists(train_path):
        train_path = os.path.join(data_path, 'food101_train')
    
    # Check for test directory first (Food-101 format), then val (ImageNet format)
    val_path = os.path.join(data_path, 'test')
    if not os.path.exists(val_path):
        val_path = os.path.join(data_path, 'val')
        if not os.path.exists(val_path):
            val_path = os.path.join(data_path, 'food101_val')
    
    if rank == 0:
        print(f"Loading training data from: {train_path}")
        print(f"Loading validation data from: {val_path}")
    
    train_dataset = datasets.ImageFolder(
        root=train_path,
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=val_path,
        transform=train_transform
    )

    # Optional filtering by class list / max samples per class
    if allowed_class_names is not None or max_per_class > 0:
        train_dataset, kept_train = _filter_imagefolder_by_classes(
            train_dataset, allowed_class_names, max_per_class
        )
        val_dataset, kept_val = _filter_imagefolder_by_classes(
            val_dataset, allowed_class_names, max_per_class=0  # don't cap val per class
        )
        if rank == 0:
            print(f"Filtered datasets: train kept {kept_train} samples; val kept {kept_val} samples")
    
    # Distributed samplers (only if multi-GPU)
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
    else:
        # Single GPU: use regular DataLoader with shuffle
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
    
    # Get number of classes from dataset
    num_classes = len(train_dataset.classes)
    if rank == 0:
        print(f"Dataset has {num_classes} classes")
    
    return train_loader, val_loader, num_classes


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, rank):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass with autocast for BF16
        with autocast(dtype=torch.bfloat16):
            outputs = model(images, labels=labels)
            loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if rank == 0 and batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def validate(model, val_loader, device):
    """Validate the model."""
    model.eval()
    total_correct = 0
    total_samples = 0
    
    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with autocast(dtype=torch.bfloat16):
            outputs = model(images)
            predictions = outputs.logits.argmax(dim=1)
        
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)
    
    accuracy = total_correct / total_samples * 100
    return accuracy


def get_synthetic_data_loaders(batch_size, world_size, rank, num_batches=10):
    """Create synthetic data loaders for testing."""
    from torch.utils.data import Dataset, DataLoader
    from torch.utils.data.distributed import DistributedSampler
    
    class SyntheticDataset(Dataset):
        def __init__(self, num_samples=1000):
            self.num_samples = num_samples
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Create synthetic ImageNet-like data (3x224x224)
            image = torch.randn(3, 224, 224)
            label = torch.randint(0, 1000, (1,)).item()
            return image, label
    
    train_dataset = SyntheticDataset(num_samples=num_batches * batch_size * world_size)
    val_dataset = SyntheticDataset(num_samples=100 * batch_size * world_size)
    
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,  # No workers for synthetic data
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Standard PyTorch DDP training for ViT-Large')
    parser.add_argument('--data-path', type=str, default='/home/siddpath/scratch.msml610/datasets/food-101', 
                        help='Path to dataset (default: Food-101 dataset)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--class-list', type=str, default=None,
                        help='Optional path to a text file with class folder names to include (one per line), e.g., ImageNet-100 list')
    parser.add_argument('--max-samples-per-class', type=int, default=0,
                        help='If > 0, cap number of training samples per class')
    parser.add_argument('--test-mode', action='store_true',
                        help='Test mode: use synthetic data instead of real ImageNet data')
    parser.add_argument('--test-batches', type=int, default=5,
                        help='Number of batches to run in test mode')
    parser.add_argument('--num-labels', type=int, default=None,
                        help='Number of classes/labels (auto-detected from dataset if not specified)')
    parser.add_argument('--single-gpu', action='store_true',
                        help='Single GPU mode: force world_size=1, useful for testing')
    args = parser.parse_args()
    
    # Setup distributed training
    device, rank, local_rank, world_size = setup_distributed()
    
    # Override for single GPU mode
    if args.single_gpu:
        world_size = 1
        rank = 0
        local_rank = 0
        if dist.is_initialized():
            dist.destroy_process_group()
        os.environ['WORLD_SIZE'] = '1'
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        print("=" * 60)
        print("SINGLE GPU MODE ENABLED")
        print("=" * 60)
    
    if rank == 0:
        print(f"World size: {world_size}")
        print(f"Starting training on rank {rank}")
        print(f"Device: {device}")
        if args.single_gpu:
            print("Running in SINGLE GPU mode (testing)")
        if args.max_samples_per_class > 0:
            print(f"Limiting to {args.max_samples_per_class} samples per class (testing mode)")
    
    # Initialize W&B (only on rank 0, optional in test mode)
    use_wandb = not args.test_mode
    if rank == 0 and use_wandb:
        try:
            wandb.init(
                project='vit-imagenet-training',
                name='standard-ddp',
                config=vars(args)
            )
        except Exception as e:
            if rank == 0:
                print(f"Warning: W&B initialization failed: {e}")
            use_wandb = False
    
    # Data loaders (will determine num_classes)
    if args.test_mode:
        if rank == 0:
            print("=" * 60)
            print("TEST MODE: Using synthetic data")
            print("=" * 60)
        train_loader, val_loader = get_synthetic_data_loaders(
            args.batch_size,
            world_size,
            rank,
            num_batches=args.test_batches
        )
        num_classes = 1000  # Synthetic data uses 1000 classes
    else:
        if args.data_path is None:
            if rank == 0:
                print("ERROR: --data-path is required when not in test mode")
            return
        
        # Read optional class list
        allowed_class_names = None
        if args.class_list is not None:
            with open(args.class_list, 'r') as f:
                allowed_class_names = set(line.strip() for line in f if line.strip())

        train_loader, val_loader, num_classes = get_data_loaders(
            args.data_path,
            args.batch_size,
            world_size,
            rank,
            args.num_workers,
            allowed_class_names=allowed_class_names,
            max_per_class=max(0, args.max_samples_per_class)
        )
    
    # Use specified num_labels or auto-detected from dataset
    num_labels = args.num_labels if args.num_labels is not None else num_classes
    
    if rank == 0:
        print(f"Training dataset size: {len(train_loader.dataset)}")
        print(f"Validation dataset size: {len(val_loader.dataset)}")
        print(f"Using {num_labels} classes for model")
    
    # Get model with correct number of classes
    model = get_model(num_labels=num_labels)
    model = model.to(device)
    
    # Wrap with DDP only if multi-GPU
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    else:
        if rank == 0:
            print("Single GPU mode: skipping DDP wrapper")
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    
    # BF16 mixed precision used without GradScaler
    
    # Training loop
    for epoch in range(args.epochs):
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"{'='*60}")
        
        # Set epoch for sampler (important for shuffling) - only if using distributed sampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        epoch_start_time = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, rank
        )
        
        # Validate
        val_accuracy = validate(model, val_loader, device)
        
        epoch_time = time.time() - epoch_start_time
        
        if rank == 0:
            print(f"\nEpoch {epoch+1} Results:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Accuracy: {val_accuracy:.2f}%")
            print(f"  Epoch Time: {epoch_time:.2f}s")
            
            # Log to W&B
            if use_wandb:
                current_device_index = device.index if hasattr(device, 'index') and device.index is not None else torch.cuda.current_device()
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_accuracy': val_accuracy,
                    'epoch_time': epoch_time,
                    'gpu_memory_allocated': torch.cuda.memory_allocated(current_device_index) / 1e9,
                    'gpu_memory_reserved': torch.cuda.memory_reserved(current_device_index) / 1e9,
                })
    
    if rank == 0:
        print("\nTraining completed!")
        if use_wandb:
            wandb.finish()
    
    cleanup_distributed()


if __name__ == '__main__':
    main()

