"""
Utility functions for DeepSpeed and FSDP training with Vision Transformers.

This module provides reusable functions for:
- Model loading and setup
- Data preparation and loading
- DeepSpeed configuration and initialization
- Training and evaluation loops
- Performance measurement
- HTA (performance) analysis
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
import deepspeed
from typing import Optional, Tuple, Dict, Set, Any
import pandas as pd


# ============================================================================
# WandB Connection Test
# ============================================================================

def load_wandb_api_key_from_kaggle_secrets() -> bool:
    """
    Load WandB API key from Kaggle Secrets and set as environment variable.
    
    Returns:
        True if key was loaded, False otherwise
    """
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        api_key = user_secrets.get_secret("WANDB_API_KEY")
        if api_key:
            import os
            os.environ["WANDB_API_KEY"] = api_key
            return True
        return False
    except:
        return False


def test_wandb_connection(timeout: int = 30) -> bool:
    """
    Test WandB connection and login status.
    
    Args:
        timeout: Timeout in seconds for WandB operations
        
    Returns:
        True if WandB is properly configured and connected, False otherwise
    """
    try:
        import wandb
        
        print(" Testing WandB connection...")
        
        # Finish any existing runs first
        try:
            wandb.finish()
        except:
            pass
        
        # Check if already logged in by trying to access API
        try:
            print("Checking WandB login status...")
            # Try to get API - this will fail if not logged in
            api = wandb.Api(timeout=timeout)
            # Try a simple API call to verify connection
            try:
                api.viewer()
                print("WandB API connection verified")
            except:
                # If viewer() fails, try login
                print("WandB API available but not authenticated. Attempting login...")
                wandb.login(timeout=timeout)
                print("[OK] WandB login successful")
        except Exception as e:
            print(f" WandB API check failed: {e}")
            print("Attempting to login...")
            try:
                wandb.login(timeout=timeout)
                print(" WandB login successful")
            except Exception as e2:
                print(f" WandB login failed: {e2}")
                print(" Please run: wandb login")
                print("Or set WANDB_API_KEY environment variable")
                return False
        
        # Test a simple init/finish cycle
        try:
            print("Testing WandB init/finish cycle...")
            run = wandb.init(
                project="wandb-connection-test",
                name="test-connection",
                mode="online",
                reinit=True
            )
            if run is not None:
                wandb.finish()
                print(" WandB connection test successful!")
                return True
            else:
                print("WandB init returned None")
                return False
        except Exception as e:
            print(f" WandB init/finish test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except ImportError:
        print("wandb package not installed. Install with: pip install wandb")
        return False
    except Exception as e:
        print(f" WandB connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def is_wandb_authenticated() -> bool:
    """
    Check if WandB is authenticated without prompting for API key.
    Also checks Kaggle Secrets for API key.
    
    Returns:
        True if WandB is authenticated, False otherwise
    """
    try:
        import wandb
        import os
        
        # Try to load from Kaggle Secrets first
        load_wandb_api_key_from_kaggle_secrets()
        
        # Check if API key is set in environment
        if os.getenv("WANDB_API_KEY"):
            try:
                api = wandb.Api()
                api.viewer()
                return True
            except:
                return False
        
        # Try to access API without prompting
        try:
            api = wandb.Api()
            api.viewer()
            return True
        except:
            # Check if wandb is logged in via settings
            try:
                settings = wandb.Settings()
                if hasattr(settings, 'api_key') and settings.api_key:
                    return True
            except:
                pass
            return False
    except:
        return False


# ============================================================================
# Model Loading Functions
# ============================================================================

def get_model_name_for_experiment(experiment_type: str = "multi_gpu") -> str:
    """
    Get appropriate ViT model name based on experiment type.
    
    Args:
        experiment_type: Type of experiment ('multi_gpu', 'single_gpu', 'single_gpu_offload')
        
    Returns:
        Hugging Face model name
    """
    model_map = {
        'multi_gpu': 'google/vit-huge-patch14-224-in21k',  # 630M params - for multi-GPU experiments
        'single_gpu': 'google/vit-base-patch16-224',  # 86M params - for single-GPU simple training
        'single_gpu_offload': 'google/vit-base-patch16-224',  # 86M params - same model to compare ZeRO-Offload memory savings
    }
    
    # Default to multi_gpu if not specified
    return model_map.get(experiment_type, model_map['multi_gpu'])


def get_model_display_name(model_name: str) -> str:
    """
    Get display name for model (e.g., 'ViT-Huge', 'ViT-Large', 'ViT-Base').
    
    Args:
        model_name: Hugging Face model name
        
    Returns:
        Display name for the model
    """
    if 'vit-huge' in model_name.lower():
        return 'ViT-Huge'
    elif 'vit-large' in model_name.lower():
        return 'ViT-Large'
    elif 'vit-base' in model_name.lower():
        return 'ViT-Base'
    elif 'vit-medium' in model_name.lower():
        return 'ViT-Medium'
    else:
        return 'ViT'


def generate_wandb_experiment_name(
    method: str,
    model_name: str,
    dataset_name: str = "Food101",
    world_size: int = 1,
    micro_batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    gpu_type: str = "T4"
) -> str:
    """
    Generate WandB experiment name in the format:
    DeepSpeed-ZeRO-PlusPlus-ViT-Huge-Food101-2xT4-MBS16-GAS1-Batch-size-16-per-gpu
    
    Args:
        method: Training method ('ddp', 'zero2', 'zero3', 'zeropp', 'single_gpu_simple', 'single_gpu_offload')
        model_name: Hugging Face model name
        dataset_name: Dataset name (default: "Food101")
        world_size: Number of GPUs
        micro_batch_size: Batch size per GPU
        gradient_accumulation_steps: Gradient accumulation steps
        gpu_type: GPU type (default: "T4")
        
    Returns:
        Formatted experiment name for WandB
    """
    # Map method to display name
    method_map = {
        'ddp': 'DDP',
        'zero2': 'ZeRO-Stage2',
        'zero3': 'ZeRO-Stage3',
        'zeropp': 'ZeRO-Stage3-PlusPlus',
        'single_gpu_simple': 'SingleGPU-Simple',
        'single_gpu_offload': 'SingleGPU-ZeRO-Offload'
    }
    method_display = method_map.get(method, method.upper())
    
    # Add "DeepSpeed-" prefix for DeepSpeed methods (not for DDP or single_gpu_simple)
    deepspeed_methods = ['zero2', 'zero3', 'zeropp', 'single_gpu_offload']
    if method in deepspeed_methods:
        method_display = f"DeepSpeed-{method_display}"
    
    # Get model display name
    model_display = get_model_display_name(model_name)
    
    # Format GPU info
    if world_size > 1:
        gpu_info = f"{world_size}x{gpu_type}"
    else:
        gpu_info = f"1x{gpu_type}"
    
    # Calculate effective batch size
    effective_batch_per_gpu = micro_batch_size * gradient_accumulation_steps
    
    # Build experiment name
    experiment_name = (
        f"{method_display}-{model_display}-{dataset_name}-"
        f"{gpu_info}-MBS{micro_batch_size}-GAS{gradient_accumulation_steps}-"
        f"Batch-size-{effective_batch_per_gpu}-per-gpu"
    )
    
    return experiment_name


def load_vit_model(
    model_name: str = "google/vit-base-patch16-224",
    num_labels: int = 1000,
    torch_dtype: torch.dtype = torch.bfloat16,
    enable_gradient_checkpointing: bool = False
) -> ViTForImageClassification:
    """
    Load a pre-trained Vision Transformer model from Hugging Face.
    
    Args:
        model_name: Hugging Face model identifier (e.g., 'google/vit-base-patch16-224')
        num_labels: Number of output classes
        torch_dtype: Data type for model weights (default: bfloat16)
        enable_gradient_checkpointing: Whether to enable gradient checkpointing
        
    Returns:
        ViTForImageClassification model instance
    """
    print(f"Loading {model_name} model with {num_labels} classes...")
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        torch_dtype=torch_dtype
    )
    
    if enable_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    return model


# ============================================================================
# Data Preparation Functions
# ============================================================================

def create_image_transforms(
    image_size: int = 224,
    use_augmentation: bool = True,
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Create data transformation pipelines for training and validation.
    
    Args:
        image_size: Target image size (default: 224 for ViT)
        use_augmentation: Whether to apply data augmentation for training
        normalize_mean: Mean values for normalization (ImageNet defaults)
        normalize_std: Std values for normalization (ImageNet defaults)
        
    Returns:
        Tuple of (train_transform, val_transform)
    """
    normalize = transforms.Normalize(mean=normalize_mean, std=normalize_std)
    
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=4),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    
    return train_transform, val_transform


def load_food101_dataset(
    data_root: str = "./data",
    train_transform: Optional[transforms.Compose] = None,
    test_transform: Optional[transforms.Compose] = None
) -> Tuple[datasets.Food101, datasets.Food101]:
    """
    Load Food-101 dataset using torchvision.
    
    Args:
        data_root: Root directory for dataset storage
        train_transform: Transform pipeline for training set
        test_transform: Transform pipeline for test set
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    import torchvision
    
    trainset = torchvision.datasets.Food101(
        root=data_root,
        split="train",
        download=True,
        transform=train_transform
    )
    
    testset = torchvision.datasets.Food101(
        root=data_root,
        split="test",
        download=True,
        transform=test_transform
    )
    
    print(f"Food-101 dataset loaded:")
    print(f"  Training set: {len(trainset)} samples")
    print(f"  Test set: {len(testset)} samples")
    print(f"  Number of classes: {len(trainset.classes)}")
    
    return trainset, testset


def create_data_loaders(
    train_dataset,
    val_dataset,
    batch_size: int,
    world_size: int = 1,
    rank: int = 0,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create distributed or single-GPU data loaders.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation/test dataset
        batch_size: Batch size per GPU
        world_size: Number of GPUs (1 for single-GPU)
        rank: Current GPU rank
        num_workers: Number of data loader workers
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    if world_size > 1:
        # Distributed training
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
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False
        )
    else:
        # Single GPU
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False
        )
    
    return train_loader, val_loader


# ============================================================================
# DeepSpeed Configuration Functions
# ============================================================================

def create_deepspeed_config(
    zero_stage: int = 3,
    micro_batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    offload_optimizer: bool = False,
    offload_param: bool = False,
    use_bf16: bool = True,
    use_fp16: bool = False,
    gradient_clipping: float = 1.0,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    use_zeropp: bool = False
) -> Dict:
    """
    Create a DeepSpeed configuration dictionary.
    
    Args:
        zero_stage: ZeRO optimization stage (0, 1, 2, 3)
        micro_batch_size: Batch size per GPU
        gradient_accumulation_steps: Number of gradient accumulation steps
        offload_optimizer: Whether to offload optimizer to CPU
        offload_param: Whether to offload parameters to CPU (ZeRO-3 only)
        use_bf16: Enable BF16 mixed precision
        use_fp16: Enable FP16 mixed precision (mutually exclusive with bf16)
        gradient_clipping: Gradient clipping value
        learning_rate: Learning rate
        weight_decay: Weight decay for optimizer
        use_zeropp: Enable ZeRO++ (quantized communication for ZeRO-3)
        
    Returns:
        DeepSpeed configuration dictionary
    """
    config = {
        # train_batch_size is auto-calculated by DeepSpeed from train_micro_batch_size_per_gpu and gradient_accumulation_steps
        "train_micro_batch_size_per_gpu": int(micro_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "gradient_clipping": gradient_clipping,
        "zero_optimization": {
            "stage": int(zero_stage),
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "betas": [0.9, 0.999],
                "eps": 1e-8
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": learning_rate,
                "warmup_num_steps": 1000
            }
        },
        "steps_per_print": 100,
        "wall_clock_breakdown": False
    }
    
    # Add ZeRO offloading if requested
    if zero_stage >= 2 and offload_optimizer:
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True
        }
    
    if zero_stage == 3 and offload_param:
        config["zero_optimization"]["offload_param"] = {
            "device": "cpu",
            "pin_memory": True
        }
    
    # Add ZeRO++ (quantized communication) - requires ZeRO Stage 3
    if use_zeropp and zero_stage == 3:
        config["zero_optimization"]["zero_quantized_weights"] = True
        config["zero_optimization"]["zero_hpz_partition_size"] = 1
    
    # Add mixed precision
    if use_bf16:
        config["bf16"] = {"enabled": True}
    elif use_fp16:
        config["fp16"] = {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        }
    
    return config


def save_deepspeed_config(config: Dict, filepath: str):
    """Save DeepSpeed configuration to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"DeepSpeed config saved to {filepath}")


def load_deepspeed_config(filepath: str) -> Dict:
    """Load DeepSpeed configuration from JSON file."""
    with open(filepath, 'r') as f:
        config = json.load(f)
    return config


def initialize_deepspeed_model(
    model: nn.Module,
    config_path: str,
    model_parameters: Optional[torch.nn.Parameter] = None
) -> Tuple:
    """
    Initialize DeepSpeed model engine.
    
    Args:
        model: PyTorch model to wrap
        config_path: Path to DeepSpeed config JSON file
        model_parameters: Model parameters (default: model.parameters())
        
    Returns:
        Tuple of (model_engine, optimizer, lr_scheduler, training_dataloader)
    """
    if model_parameters is None:
        model_parameters = model.parameters()
    
    model_engine, optimizer, lr_scheduler, _ = deepspeed.initialize(
        model=model,
        model_parameters=model_parameters,
        config=config_path
    )
    
    return model_engine, optimizer, lr_scheduler


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch_deepspeed(
    model_engine,
    train_loader: DataLoader,
    epoch: int,
    rank: int = 0,
    log_interval: int = 100,
    enable_profiling: bool = False,
    profiler: Optional[torch.profiler.profile] = None,
    experiment_name: Optional[str] = None,
    profiling_warmup_steps: int = 1,
    profiling_active_steps: int = 3,
    profiling_repeat: int = 1
) -> float:
    """
    Train for one epoch using DeepSpeed.
    
    Args:
        model_engine: DeepSpeed model engine
        train_loader: Training data loader
        epoch: Current epoch number
        rank: Process rank (for logging)
        log_interval: Logging interval in batches
        enable_profiling: Whether to enable PyTorch profiler
        profiler: PyTorch profiler instance (if None and enable_profiling=True, creates one)
        experiment_name: Name of the experiment (used for profiler folder naming)
        profiling_warmup_steps: Number of warmup steps before profiling starts recording
        profiling_active_steps: Number of steps to actively profile and record
        profiling_repeat: Number of times to repeat the profiling cycle
        
    Returns:
        Average training loss
    """
    model_engine.train()
    total_loss = 0.0
    num_batches = 0
    
    # Setup profiler if needed
    if enable_profiling and profiler is None:
        # Create experiment-specific folder for profiler traces
        if experiment_name:
            profiler_dir = f'./profiling_traces/{experiment_name}'
            os.makedirs(profiler_dir, exist_ok=True)
            profiler_path = f'{profiler_dir}/epoch_{epoch}'
        else:
            profiler_path = f'./profiling_traces/epoch_{epoch}'
        
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1, 
                warmup=profiling_warmup_steps, 
                active=profiling_active_steps, 
                repeat=profiling_repeat
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_path),
            record_shapes=True,
            with_stack=True
        )
        profiler.start()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(model_engine.device, non_blocking=True)
        labels = labels.to(model_engine.device, non_blocking=True)
        
        if enable_profiling and profiler is not None:
            profiler.step()
        
        # Forward pass
        loss = model_engine(images, labels=labels).loss
        
        # Backward pass (DeepSpeed handles gradient accumulation internally)
        model_engine.backward(loss)
        model_engine.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if rank == 0 and batch_idx % log_interval == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    if enable_profiling and profiler is not None:
        profiler.stop()
        if rank == 0:
            if experiment_name:
                print(f"Profiling data saved for {experiment_name} epoch {epoch} in ./profiling_traces/{experiment_name}/")
            else:
                print(f"Profiling data saved for epoch {epoch}")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def train_epoch_standard(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    use_amp: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    log_interval: int = 100,
    enable_profiling: bool = False,
    profiler: Optional[torch.profiler.profile] = None,
    experiment_name: Optional[str] = None,
    profiling_warmup_steps: int = 1,
    profiling_active_steps: int = 3,
    profiling_repeat: int = 1
) -> float:
    """
    Train for one epoch using standard PyTorch training.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Training device
        epoch: Current epoch number
        use_amp: Whether to use automatic mixed precision
        dtype: Data type for mixed precision
        log_interval: Logging interval in batches
        enable_profiling: Whether to enable PyTorch profiler
        profiler: PyTorch profiler instance (if None and enable_profiling=True, creates one)
        experiment_name: Name of the experiment (used for profiler folder naming)
        profiling_warmup_steps: Number of warmup steps before profiling starts recording
        profiling_active_steps: Number of steps to actively profile and record
        profiling_repeat: Number of times to repeat the profiling cycle
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Setup profiler if needed
    if enable_profiling and profiler is None:
        # Create experiment-specific folder for profiler traces
        if experiment_name:
            profiler_dir = f'./profiling_traces/{experiment_name}'
            os.makedirs(profiler_dir, exist_ok=True)
            profiler_path = f'{profiler_dir}/epoch_{epoch}'
        else:
            profiler_path = f'./profiling_traces/ddp_epoch_{epoch}'
        
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1, 
                warmup=profiling_warmup_steps, 
                active=profiling_active_steps, 
                repeat=profiling_repeat
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_path),
            record_shapes=True,
            with_stack=True
        )
        profiler.start()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        if enable_profiling and profiler is not None:
            profiler.step()
        
        optimizer.zero_grad()
        
        if use_amp:
            with autocast(dtype=dtype):
                outputs = model(images, labels=labels)
                loss = outputs.loss if hasattr(outputs, 'loss') else criterion(outputs.logits, labels)
        else:
            outputs = model(images, labels=labels)
            loss = outputs.loss if hasattr(outputs, 'loss') else criterion(outputs.logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % log_interval == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    if enable_profiling and profiler is not None:
        profiler.stop()
        if experiment_name:
            print(f" Profiling data saved for {experiment_name} epoch {epoch} in ./profiling_traces/{experiment_name}/")
        else:
            print(f" Profiling data saved for epoch {epoch}")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


# ============================================================================
# Evaluation Functions
# ============================================================================

@torch.no_grad()
def evaluate_deepspeed(
    model_engine,
    val_loader: DataLoader,
    rank: int = 0
) -> Tuple[float, Dict]:
    """
    Evaluate model using DeepSpeed engine.
    
    Args:
        model_engine: DeepSpeed model engine
        val_loader: Validation/test data loader
        rank: Process rank (for logging)
        
    Returns:
        Tuple of (accuracy, metrics_dict)
    """
    model_engine.eval()
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_labels = []
    
    for images, labels in val_loader:
        images = images.to(model_engine.device, non_blocking=True)
        labels = labels.to(model_engine.device, non_blocking=True)
        
        outputs = model_engine(images)
        predictions = outputs.logits.argmax(dim=1)
        
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0.0
    
    metrics = {
        'accuracy': accuracy,
        'total_samples': total_samples,
        'correct': total_correct,
        'predictions': all_predictions,
        'labels': all_labels
    }
    
    return accuracy, metrics


@torch.no_grad()
def evaluate_standard(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    use_amp: bool = True,
    dtype: torch.dtype = torch.bfloat16
) -> Tuple[float, Dict]:
    """
    Evaluate model using standard PyTorch.
    
    Args:
        model: PyTorch model
        val_loader: Validation/test data loader
        device: Evaluation device
        criterion: Optional loss function
        use_amp: Whether to use automatic mixed precision
        dtype: Data type for mixed precision
        
    Returns:
        Tuple of (accuracy, metrics_dict)
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        if use_amp:
            with autocast(dtype=dtype):
                outputs = model(images)
        else:
            outputs = model(images)
        
        predictions = outputs.logits.argmax(dim=1)
        
        if criterion is not None:
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()
        
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0.0
    avg_loss = total_loss / len(val_loader) if criterion is not None else None
    
    metrics = {
        'accuracy': accuracy,
        'total_samples': total_samples,
        'correct': total_correct,
        'predictions': all_predictions,
        'labels': all_labels
    }
    
    if avg_loss is not None:
        metrics['loss'] = avg_loss
    
    return accuracy, metrics


# ============================================================================
# Experiment Runner Function
# ============================================================================

def run_training_experiment(
    experiment_name: Optional[str] = None,
    method: str = "ddp",  # 'ddp', 'zero2', 'zero3', 'zeropp', 'single_gpu_simple', 'single_gpu_offload'
    model_name: str = "",
    train_loader: DataLoader = None,
    test_loader: DataLoader = None,
    trainset: Any = None,
    testset: Any = None,
    rank: int = 0,
    world_size: int = 1,
    num_epochs: int = 1,
    micro_batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    enable_profiling: bool = False,
    profiling_warmup_steps: int = 1,
    profiling_active_steps: int = 3,
    profiling_repeat: int = 1,
    wandb_project: str = "Distributed ViT training systems-Latest_run",
    dataset_name: str = "Food101",
    gpu_type: str = "T4"
) -> Dict:
    """
    Run a complete training experiment with the specified method.
    
    Args:
        experiment_name: Name of the experiment (for logging and profiler folders).
                         If None, will be auto-generated in format:
                         "Method-Model-Dataset-GPUs-MBS-GAS-Batch-size-per-gpu"
        method: Training method ('ddp', 'zero2', 'zero3', 'zeropp', 'single_gpu_simple', 'single_gpu_offload')
        model_name: Hugging Face model name
        train_loader: Training data loader
        test_loader: Test data loader
        trainset: Training dataset
        testset: Test dataset
        rank: Process rank
        world_size: Number of GPUs
        num_epochs: Number of training epochs
        micro_batch_size: Batch size per GPU
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        weight_decay: Weight decay
        enable_profiling: Whether to enable profiling
        profiling_warmup_steps: Profiling warmup steps
        profiling_active_steps: Profiling active steps
        profiling_repeat: Profiling repeat count
        wandb_project: WandB project name (default: "Distributed ViT training systems-Latest_run")
        dataset_name: Dataset name for experiment naming (default: "Food101")
        gpu_type: GPU type for experiment naming (default: "T4")
        
    Returns:
        Dictionary with experiment results
    """
    try:
        import wandb
    except ImportError:
        wandb = None
        if rank == 0:
            print(" wandb not available. Continuing without logging.")
    
    # Auto-generate experiment name if not provided
    if experiment_name is None:
        experiment_name = generate_wandb_experiment_name(
            method=method,
            model_name=model_name,
            dataset_name=dataset_name,
            world_size=world_size,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gpu_type=gpu_type
        )
    
    start_time = time.time()
    device, rank, local_rank, world_size = setup_distributed()
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_data_parallel = False
    effective_world_size = world_size
    if world_size == 1 and available_gpus > 1 and method == 'ddp':
        # Single-process notebook but multiple GPUs: fall back to DataParallel for DDP method
        use_data_parallel = True
        effective_world_size = available_gpus
        if rank == 0:
            print(f" Using DataParallel with {effective_world_size} GPUs (single-process environment).")
    elif world_size == 1 and available_gpus > 1 and method != 'ddp':
        if rank == 0:
            print(f" Multiple GPUs detected ({available_gpus}) but method '{method}' requires multi-process launch; continuing with single GPU.")
    
    # Align world_size for downstream logic
    world_size = effective_world_size
    
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"Starting Experiment: {experiment_name}")
        print(f"Method: {method.upper()}")
        print(f"Model: {model_name}")
        print(f"GPUs: {effective_world_size}")
        print(f"Epochs: {num_epochs}")
        print(f"{'='*80}\n")
    
    # Initialize WandB (only on rank 0)
    if rank == 0:
        print("\n Checking WandB availability and authentication...")
        if wandb is None:
            print(" WandB package not installed!")
            raise ImportError("WandB is required but not installed. Install with: pip install wandb")
        else:
            # Check if WandB is authenticated before trying to init
            print(" Verifying authentication status...")
            if not is_wandb_authenticated():
                print("\n" + "="*80)
                print(" WandB is NOT authenticated!")
                print("="*80)
                print("\n Please authenticate WandB before running experiments:")
                print("  1. Set WANDB_API_KEY environment variable:")
                print("     export WANDB_API_KEY=your_api_key_here")
                print("  2. OR run: wandb login")
                print("  3. OR in Kaggle: Add WANDB_API_KEY to Secrets (Add-ons → Secrets)")
                print("\n Get your API key from: https://wandb.ai/authorize")
                print("="*80 + "\n")
                raise RuntimeError("WandB authentication required. Please login to WandB before running experiments.")
            else:
                print(" WandB is authenticated")
                print(" Initializing WandB run...")
                try:
                    wandb.init(
                        project=wandb_project,
                        name=experiment_name,
                        config={
                            'method': method,
                            'model_name': model_name,
                            'num_epochs': num_epochs,
                            'micro_batch_size': micro_batch_size,
                            'gradient_accumulation_steps': gradient_accumulation_steps,
                            'learning_rate': learning_rate,
                            'weight_decay': weight_decay,
                            'world_size': world_size,
                            'enable_profiling': enable_profiling
                        },
                        reinit=True
                    )
                    print(f"[OK] WandB initialized successfully")
                    print(f"[WANDB] Project: {wandb_project}")
                    print(f"[WANDB] Run name: {experiment_name}")
                except Exception as e:
                    print(f"\n[ERROR] WandB initialization failed: {e}")
                    print("[ERROR] Cannot continue without WandB logging.")
                    raise RuntimeError(f"WandB initialization failed: {e}") from e
    
    try:
        # Load model
        if rank == 0:
            print(f" Loading model: {model_name}...")
        num_labels = len(trainset.classes)
        if rank == 0:
            print(f" Number of classes: {num_labels}")
        model = load_vit_model(
            model_name=model_name,
            num_labels=num_labels,
            torch_dtype=torch.bfloat16,
            enable_gradient_checkpointing=True
        )
        if rank == 0:
            print(" Model loaded successfully")
        
        # Get initial memory stats
        if rank == 0:
            print(" Getting initial GPU memory stats...")
        initial_memory = get_gpu_memory_stats(rank) if torch.cuda.is_available() else {}
        if rank == 0:
            print(" Initial memory stats collected")
        
        # Run training based on method
        if method == 'single_gpu_simple':
            # Single-GPU simple training (standard PyTorch)
            if rank == 0:
                print(" Setting up single-GPU simple training...")
            model = model.to(device)
            if rank == 0:
                print(" Model moved to device")
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
            criterion = nn.CrossEntropyLoss()
            
            train_losses = []
            for epoch in range(num_epochs):
                if hasattr(train_loader.sampler, 'set_epoch'):
                    train_loader.sampler.set_epoch(epoch)
                
                loss = train_epoch_standard(
                    model=model,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    epoch=epoch,
                    use_amp=True,
                    dtype=torch.bfloat16,
                    enable_profiling=enable_profiling,
                    experiment_name=experiment_name,
                    profiling_warmup_steps=profiling_warmup_steps,
                    profiling_active_steps=profiling_active_steps,
                    profiling_repeat=profiling_repeat
                )
                train_losses.append(loss)
                
                if rank == 0:
                    accuracy, metrics = evaluate_standard(model, test_loader, device)
                    if wandb is not None:
                        wandb.log({
                            'epoch': epoch + 1,
                            'train_loss': loss,
                            'test_accuracy': accuracy
                        })
                    print(f"Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={accuracy:.2f}%")
            
            final_accuracy, final_metrics = evaluate_standard(model, test_loader, device)
            model_for_throughput = model
            
        elif method == 'single_gpu_offload':
            # Single-GPU with ZeRO-Offload
            ds_config = create_deepspeed_config(
                zero_stage=2,
                micro_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                offload_optimizer=True,
                use_bf16=True,
                learning_rate=learning_rate,
                weight_decay=weight_decay
            )
            
            config_path = f"ds_config_{experiment_name}.json"
            save_deepspeed_config(ds_config, config_path)
            
            model_engine, optimizer, lr_scheduler = initialize_deepspeed_model(
                model=model,
                config_path=config_path,
                model_parameters=model.parameters()
            )
            
            train_losses = []
            for epoch in range(num_epochs):
                if hasattr(train_loader.sampler, 'set_epoch'):
                    train_loader.sampler.set_epoch(epoch)
                
                loss = train_epoch_deepspeed(
                    model_engine=model_engine,
                    train_loader=train_loader,
                    epoch=epoch,
                    rank=rank,
                    enable_profiling=enable_profiling,
                    experiment_name=experiment_name,
                    profiling_warmup_steps=profiling_warmup_steps,
                    profiling_active_steps=profiling_active_steps,
                    profiling_repeat=profiling_repeat
                )
                train_losses.append(loss)
                
                if rank == 0:
                    accuracy, metrics = evaluate_deepspeed(model_engine, test_loader, rank)
                    if wandb is not None:
                        wandb.log({
                            'epoch': epoch + 1,
                            'train_loss': loss,
                            'test_accuracy': accuracy
                        })
                    print(f"Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={accuracy:.2f}%")
            
            final_accuracy, final_metrics = evaluate_deepspeed(model_engine, test_loader, rank)
            model_for_throughput = model_engine
            
        elif method == 'ddp':
            # Standard DDP training (with DataParallel fallback for single-process multi-GPU)
            if use_data_parallel:
                model = nn.DataParallel(model.to(device))
            elif world_size > 1:
                model = nn.parallel.DistributedDataParallel(model.to(device), device_ids=[local_rank])
            else:
                model = model.to(device)
            
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
            criterion = nn.CrossEntropyLoss()
            
            train_losses = []
            for epoch in range(num_epochs):
                if hasattr(train_loader.sampler, 'set_epoch'):
                    train_loader.sampler.set_epoch(epoch)
                
                loss = train_epoch_standard(
                    model=model.module if hasattr(model, 'module') else model,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    epoch=epoch,
                    use_amp=True,
                    dtype=torch.bfloat16,
                    enable_profiling=enable_profiling,
                    experiment_name=experiment_name,
                    profiling_warmup_steps=profiling_warmup_steps,
                    profiling_active_steps=profiling_active_steps,
                    profiling_repeat=profiling_repeat
                )
                train_losses.append(loss)
                
                if rank == 0:
                    eval_model = model.module if hasattr(model, 'module') else model
                    accuracy, metrics = evaluate_standard(eval_model, test_loader, device)
                    if wandb is not None:
                        wandb.log({
                            'epoch': epoch + 1,
                            'train_loss': loss,
                            'test_accuracy': accuracy
                        })
                    print(f"Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={accuracy:.2f}%")
            
            if rank == 0:
                eval_model = model.module if hasattr(model, 'module') else model
                final_accuracy, final_metrics = evaluate_standard(eval_model, test_loader, device)
                model_for_throughput = eval_model
            else:
                final_accuracy, final_metrics = 0.0, {}
                model_for_throughput = model
                
        elif method in ['zero2', 'zero3', 'zeropp']:
            # DeepSpeed ZeRO training
            zero_stage = 2 if method == 'zero2' else 3
            use_zeropp = (method == 'zeropp')
            
            ds_config = create_deepspeed_config(
                zero_stage=zero_stage,
                micro_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                use_bf16=True,
                use_zeropp=use_zeropp,
                learning_rate=learning_rate,
                weight_decay=weight_decay
            )
            
            config_path = f"ds_config_{experiment_name}.json"
            save_deepspeed_config(ds_config, config_path)
            
            model_engine, optimizer, lr_scheduler = initialize_deepspeed_model(
                model=model,
                config_path=config_path,
                model_parameters=model.parameters()
            )
            
            train_losses = []
            for epoch in range(num_epochs):
                if hasattr(train_loader.sampler, 'set_epoch'):
                    train_loader.sampler.set_epoch(epoch)
                
                loss = train_epoch_deepspeed(
                    model_engine=model_engine,
                    train_loader=train_loader,
                    epoch=epoch,
                    rank=rank,
                    enable_profiling=enable_profiling,
                    experiment_name=experiment_name,
                    profiling_warmup_steps=profiling_warmup_steps,
                    profiling_active_steps=profiling_active_steps,
                    profiling_repeat=profiling_repeat
                )
                train_losses.append(loss)
                
                if rank == 0:
                    accuracy, metrics = evaluate_deepspeed(model_engine, test_loader, rank)
                    if wandb is not None:
                        wandb.log({
                            'epoch': epoch + 1,
                            'train_loss': loss,
                            'test_accuracy': accuracy
                        })
                    print(f"Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={accuracy:.2f}%")
            
            final_accuracy, final_metrics = evaluate_deepspeed(model_engine, test_loader, rank)
            model_for_throughput = model_engine
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Get final memory stats
        final_memory = get_gpu_memory_stats(rank) if torch.cuda.is_available() else {}
        
        # Measure throughput
        throughput_metrics = measure_throughput(
            model_for_throughput,
            train_loader,
            num_batches=50,
            is_deepspeed=(method not in ['single_gpu_simple', 'ddp'])
        )
        
        total_time = time.time() - start_time
        
        # Compile results
        results = {
            'experiment_name': experiment_name,
            'method': method,
            'model_name': model_name,
            'num_epochs': num_epochs,
            'final_accuracy': final_accuracy if rank == 0 else 0.0,
            'avg_train_loss': sum(train_losses) / len(train_losses) if train_losses else 0.0,
            'total_time': total_time,
            'throughput_samples_per_sec': throughput_metrics.get('throughput_samples_per_sec', 0),
            'gpu_memory_max_allocated_gb': final_memory.get('max_allocated_gb', 0),
            'gpu_memory_max_reserved_gb': final_memory.get('max_reserved_gb', 0),
            'world_size': world_size,
            'status': 'success'
        }
        
        if rank == 0:
            if wandb is not None:
                wandb.log({
                    'final_accuracy': final_accuracy,
                    'total_time': total_time,
                    'throughput': throughput_metrics.get('throughput_samples_per_sec', 0),
                    'samples_per_sec': throughput_metrics.get('throughput_samples_per_sec', 0),
                    'gpu_memory_gb': final_memory.get('max_allocated_gb', 0)
                })
                wandb.finish()
            
            print(f"\n{'='*80}")
            print(f"Experiment {experiment_name} completed!")
            print(f"  Final Accuracy: {final_accuracy:.2f}%")
            print(f"  Total Time: {total_time:.2f}s")
            print(f"  Throughput: {throughput_metrics.get('throughput_samples_per_sec', 0):.2f} samples/sec")
            print(f"{'='*80}\n")
        
        return results
        
    except Exception as e:
        if rank == 0:
            print(f" Experiment {experiment_name} failed: {e}")
            import traceback
            traceback.print_exc()
            if wandb is not None:
                wandb.finish()
        
        return {
            'experiment_name': experiment_name,
            'method': method,
            'status': 'failed',
            'error': str(e)
        }


# ============================================================================
# Performance Measurement Functions
# ============================================================================

def measure_throughput(
    model_engine_or_model,
    dataloader: DataLoader,
    num_batches: int = 100,
    is_deepspeed: bool = True
) -> Dict:
    """
    Measure training/inference throughput (samples per second).
    
    Args:
        model_engine_or_model: DeepSpeed engine or PyTorch model
        dataloader: Data loader for measurement
        num_batches: Number of batches to measure
        is_deepspeed: Whether using DeepSpeed engine
        
    Returns:
        Dictionary with throughput metrics
    """
    if is_deepspeed:
        model_engine_or_model.eval()
        device = model_engine_or_model.device
    else:
        model_engine_or_model.eval()
        device = next(model_engine_or_model.parameters()).device
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    samples_processed = 0
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if i >= num_batches:
                break
            
            images = images.to(device, non_blocking=True)
            
            if is_deepspeed:
                _ = model_engine_or_model(images)
            else:
                _ = model_engine_or_model(images)
            
            samples_processed += images.size(0)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed_time = time.time() - start_time
    
    throughput = samples_processed / elapsed_time if elapsed_time > 0 else 0.0
    
    return {
        'throughput_samples_per_sec': throughput,
        'samples_processed': samples_processed,
        'elapsed_time': elapsed_time,
        'num_batches': num_batches
    }


def get_gpu_memory_stats(device_id: int = 0) -> Dict:
    """
    Get GPU memory usage statistics.
    
    Args:
        device_id: GPU device ID
        
    Returns:
        Dictionary with memory statistics
    """
    if not torch.cuda.is_available():
        return {}
    
    torch.cuda.set_device(device_id)
    
    return {
        'allocated_gb': torch.cuda.memory_allocated(device_id) / 1e9,
        'reserved_gb': torch.cuda.memory_reserved(device_id) / 1e9,
        'max_allocated_gb': torch.cuda.max_memory_allocated(device_id) / 1e9,
        'max_reserved_gb': torch.cuda.max_memory_reserved(device_id) / 1e9
    }


# ============================================================================
# Distributed Setup Functions
# ============================================================================

def setup_distributed() -> Tuple[torch.device, int, int, int]:
    """
    Initialize distributed training environment.
    Auto-detects number of GPUs if WORLD_SIZE not set.
    
    Returns:
        Tuple of (device, rank, local_rank, world_size)
    """
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # Priority: 1) Environment variable, 2) Auto-detect from GPUs
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        if world_size > 1:
            print(f" Using WORLD_SIZE={world_size} from environment")
    elif torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        os.environ['WORLD_SIZE'] = str(world_size)
        if world_size > 1:
            print(f" Auto-detected {world_size} GPUs, set WORLD_SIZE={world_size}")
    else:
        world_size = 1
    
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')
    
    if world_size > 1:
        if not dist.is_initialized():
            # Set default distributed init if not already set
            if 'MASTER_ADDR' not in os.environ:
                os.environ['MASTER_ADDR'] = 'localhost'
            if 'MASTER_PORT' not in os.environ:
                os.environ['MASTER_PORT'] = '12355'
            
            dist.init_process_group(
                backend='nccl' if torch.cuda.is_available() else 'gloo',
                init_method='env://',
                rank=rank,
                world_size=world_size
            )
    
    return device, rank, local_rank, world_size


def cleanup_distributed():
    """Cleanup distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================================
# HTA (Performance) Analysis Functions
# ============================================================================

def calculate_speedup_metrics(
    results_df: pd.DataFrame,
    baseline_method: str = "single_gpu_simple"
) -> pd.DataFrame:
    """
    Calculate speedup metrics relative to baseline.
    
    Args:
        results_df: DataFrame with experiment results
        baseline_method: Method to use as baseline for comparison
        
    Returns:
        DataFrame with speedup metrics added
    """
    # Filter non-profiling results
    non_profiling = results_df[~results_df['method'].str.contains('profiling', case=False, na=False)].copy()
    
    # Find baseline
    baseline = non_profiling[non_profiling['method'] == baseline_method]
    if len(baseline) == 0:
        # Try alternative baseline
        if 'ddp' in non_profiling['method'].values:
            baseline = non_profiling[non_profiling['method'] == 'ddp'].iloc[0:1]
            baseline_method = 'ddp'
        else:
            print(f" Baseline method '{baseline_method}' not found. Using first method.")
            baseline = non_profiling.iloc[0:1]
            baseline_method = non_profiling.iloc[0]['method'] if len(non_profiling) > 0 else None
    else:
        baseline = baseline.iloc[0:1]
    
    if len(baseline) == 0 or baseline_method is None:
        print(" No baseline found. Cannot calculate speedup metrics.")
        return results_df
    
    baseline_time = baseline.iloc[0].get('total_time', 0)
    baseline_throughput = baseline.iloc[0].get('throughput_samples_per_sec', 0)
    baseline_memory = baseline.iloc[0].get('gpu_memory_max_allocated_gb', 0)
    
    # Calculate speedup metrics for all methods
    speedup_metrics = []
    for _, row in non_profiling.iterrows():
        method = row['method']
        if method == baseline_method:
            speedup_metrics.append({
                'method': method,
                'speedup_time': 1.0,
                'speedup_throughput': 1.0,
                'memory_reduction_pct': 0.0,
                'efficiency_score': 1.0
            })
        else:
            method_time = row.get('total_time', 0) if pd.notna(row.get('total_time')) else 0
            method_throughput = row.get('throughput_samples_per_sec', 0) if pd.notna(row.get('throughput_samples_per_sec')) else 0
            method_memory = row.get('gpu_memory_max_allocated_gb', 0) if pd.notna(row.get('gpu_memory_max_allocated_gb')) else 0
            
            speedup_time = baseline_time / method_time if method_time > 0 else 0
            speedup_throughput = method_throughput / baseline_throughput if baseline_throughput > 0 else 0
            memory_reduction = ((baseline_memory - method_memory) / baseline_memory * 100) if baseline_memory > 0 else 0
            
            # Efficiency score: weighted combination of speedup and memory efficiency
            efficiency_score = (speedup_throughput * 0.6) + (abs(memory_reduction) / 100 * 0.4) if memory_reduction > 0 else speedup_throughput * 0.6
            
            speedup_metrics.append({
                'method': method,
                'speedup_time': speedup_time,
                'speedup_throughput': speedup_throughput,
                'memory_reduction_pct': memory_reduction,
                'efficiency_score': efficiency_score
            })
    
    speedup_df = pd.DataFrame(speedup_metrics)
    
    # Merge back into results
    results_with_speedup = results_df.merge(speedup_df, on='method', how='left')
    
    return results_with_speedup


def analyze_scaling_efficiency(results_df: pd.DataFrame) -> Dict:
    """
    Analyze scaling efficiency for multi-GPU experiments.
    
    Args:
        results_df: DataFrame with experiment results
        
    Returns:
        Dictionary with scaling efficiency metrics
    """
    non_profiling = results_df[~results_df['method'].str.contains('profiling', case=False, na=False)].copy()
    
    # Find single-GPU and multi-GPU results
    multi_gpu_methods = ['ddp', 'zero2', 'zero3', 'zeropp']
    
    scaling_analysis = {}
    
    for multi_method in multi_gpu_methods:
        multi_result = non_profiling[non_profiling['method'] == multi_method]
        if len(multi_result) == 0:
            continue
        
        # For DeepSpeed methods, compare with single-GPU simple
        single_result = non_profiling[non_profiling['method'] == 'single_gpu_simple']
        
        if len(single_result) > 0 and len(multi_result) > 0:
            single_time = single_result.iloc[0].get('total_time', 0)
            multi_time = multi_result.iloc[0].get('total_time', 0)
            single_throughput = single_result.iloc[0].get('throughput_samples_per_sec', 0)
            multi_throughput = multi_result.iloc[0].get('throughput_samples_per_sec', 0)
            
            if single_time > 0 and multi_time > 0:
                # Assume 2 GPUs for multi-GPU (can be made configurable)
                num_gpus = 2
                ideal_speedup = num_gpus
                actual_speedup = single_time / multi_time
                scaling_efficiency = (actual_speedup / ideal_speedup) * 100 if ideal_speedup > 0 else 0
                
                scaling_analysis[multi_method] = {
                    'num_gpus': num_gpus,
                    'ideal_speedup': ideal_speedup,
                    'actual_speedup': actual_speedup,
                    'scaling_efficiency_pct': scaling_efficiency,
                    'throughput_improvement': ((multi_throughput / single_throughput - 1) * 100) if single_throughput > 0 else 0
                }
    
    return scaling_analysis


def generate_hta_report(
    results_df: pd.DataFrame,
    output_path: str = "HTA_analysis_report.txt"
) -> str:
    """
    Generate comprehensive HTA performance analysis report.
    
    Args:
        results_df: DataFrame with experiment results
        output_path: Path to save the report
        
    Returns:
        Report text as string
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("HTA (PERFORMANCE) ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Filter non-profiling results
    non_profiling = results_df[~results_df['method'].str.contains('profiling', case=False, na=False)].copy()
    
    if len(non_profiling) == 0:
        report_lines.append("[ERROR] No non-profiling results found for analysis.")
        report = "\n".join(report_lines)
        with open(output_path, 'w') as f:
            f.write(report)
        return report
    
    # 1. Summary Statistics
    report_lines.append("1. SUMMARY STATISTICS")
    report_lines.append("-" * 80)
    report_lines.append(f"Total experiments analyzed: {len(non_profiling)}")
    report_lines.append(f"Methods tested: {', '.join(non_profiling['method'].unique())}")
    report_lines.append("")
    
    # 2. Throughput Analysis
    report_lines.append("2. THROUGHPUT ANALYSIS")
    report_lines.append("-" * 80)
    throughput_sorted = non_profiling.sort_values('throughput_samples_per_sec', ascending=False, na_last=True)
    report_lines.append("\nThroughput Ranking (samples/second):")
    for idx, (_, row) in enumerate(throughput_sorted.iterrows(), 1):
        throughput = row.get('throughput_samples_per_sec', 'N/A')
        method = row['method'].upper()
        if pd.notna(throughput):
            report_lines.append(f"  {idx}. {method}: {throughput:.2f} samples/sec")
        else:
            report_lines.append(f"  {idx}. {method}: N/A")
    
    # Calculate throughput statistics
    valid_throughput = non_profiling['throughput_samples_per_sec'].dropna()
    if len(valid_throughput) > 0:
        report_lines.append(f"\nThroughput Statistics:")
        report_lines.append(f"  Mean: {valid_throughput.mean():.2f} samples/sec")
        report_lines.append(f"  Median: {valid_throughput.median():.2f} samples/sec")
        report_lines.append(f"  Min: {valid_throughput.min():.2f} samples/sec")
        report_lines.append(f"  Max: {valid_throughput.max():.2f} samples/sec")
        report_lines.append(f"  Std Dev: {valid_throughput.std():.2f} samples/sec")
    report_lines.append("")
    
    # 3. Memory Efficiency Analysis
    report_lines.append("3. MEMORY EFFICIENCY ANALYSIS")
    report_lines.append("-" * 80)
    memory_sorted = non_profiling.sort_values('gpu_memory_max_allocated_gb', na_last=True)
    report_lines.append("\nMemory Usage Ranking (GB):")
    for idx, (_, row) in enumerate(memory_sorted.iterrows(), 1):
        memory = row.get('gpu_memory_max_allocated_gb', 'N/A')
        method = row['method'].upper()
        if pd.notna(memory):
            report_lines.append(f"  {idx}. {method}: {memory:.2f} GB")
        else:
            report_lines.append(f"  {idx}. {method}: N/A")
    
    # Calculate memory statistics
    valid_memory = non_profiling['gpu_memory_max_allocated_gb'].dropna()
    if len(valid_memory) > 0:
        report_lines.append(f"\nMemory Statistics:")
        report_lines.append(f"  Mean: {valid_memory.mean():.2f} GB")
        report_lines.append(f"  Median: {valid_memory.median():.2f} GB")
        report_lines.append(f"  Min: {valid_memory.min():.2f} GB")
        report_lines.append(f"  Max: {valid_memory.max():.2f} GB")
        report_lines.append(f"  Range: {valid_memory.max() - valid_memory.min():.2f} GB")
    report_lines.append("")
    
    # 4. Speedup Analysis
    report_lines.append("4. SPEEDUP ANALYSIS (vs Baseline)")
    report_lines.append("-" * 80)
    
    # Calculate speedup metrics
    results_with_speedup = calculate_speedup_metrics(non_profiling)
    
    baseline_method = 'single_gpu_simple' if 'single_gpu_simple' in non_profiling['method'].values else 'ddp'
    report_lines.append(f"Baseline: {baseline_method.upper()}")
    report_lines.append("")
    
    speedup_data = results_with_speedup[['method', 'speedup_time', 'speedup_throughput', 'memory_reduction_pct', 'efficiency_score']].dropna()
    
    if len(speedup_data) > 0:
        report_lines.append("Speedup Metrics:")
        for _, row in speedup_data.iterrows():
            method = row['method'].upper()
            speedup_t = row.get('speedup_time', 0)
            speedup_th = row.get('speedup_throughput', 0)
            mem_red = row.get('memory_reduction_pct', 0)
            eff_score = row.get('efficiency_score', 0)
            
            report_lines.append(f"\n  {method}:")
            report_lines.append(f"    Time Speedup: {speedup_t:.2f}x")
            report_lines.append(f"    Throughput Speedup: {speedup_th:.2f}x")
            report_lines.append(f"    Memory Reduction: {mem_red:+.1f}%")
            report_lines.append(f"    Efficiency Score: {eff_score:.3f}")
    report_lines.append("")
    
    # 5. Scaling Efficiency (Multi-GPU)
    report_lines.append("5. SCALING EFFICIENCY ANALYSIS")
    report_lines.append("-" * 80)
    scaling_analysis = analyze_scaling_efficiency(non_profiling)
    
    if scaling_analysis:
        report_lines.append("\nMulti-GPU Scaling Efficiency:")
        for method, metrics in scaling_analysis.items():
            report_lines.append(f"\n  {method.upper()}:")
            report_lines.append(f"    GPUs: {metrics['num_gpus']}")
            report_lines.append(f"    Ideal Speedup: {metrics['ideal_speedup']:.2f}x")
            report_lines.append(f"    Actual Speedup: {metrics['actual_speedup']:.2f}x")
            report_lines.append(f"    Scaling Efficiency: {metrics['scaling_efficiency_pct']:.1f}%")
            report_lines.append(f"    Throughput Improvement: {metrics['throughput_improvement']:+.1f}%")
    else:
        report_lines.append("\nNo multi-GPU scaling data available.")
    report_lines.append("")
    
    # 6. Recommendations
    report_lines.append("6. RECOMMENDATIONS")
    report_lines.append("-" * 80)
    
    # Find best method for different criteria
    best_throughput = throughput_sorted.iloc[0] if len(throughput_sorted) > 0 else None
    best_memory = memory_sorted.iloc[0] if len(memory_sorted) > 0 else None
    
    if best_throughput is not None:
        report_lines.append(f"\nBest Throughput: {best_throughput['method'].upper()}")
        report_lines.append(f"  Throughput: {best_throughput.get('throughput_samples_per_sec', 'N/A'):.2f} samples/sec")
    
    if best_memory is not None:
        report_lines.append(f"\nBest Memory Efficiency: {best_memory['method'].upper()}")
        report_lines.append(f"  Memory Usage: {best_memory.get('gpu_memory_max_allocated_gb', 'N/A'):.2f} GB")
    
    # Find best overall efficiency
    if 'efficiency_score' in results_with_speedup.columns:
        best_efficiency = results_with_speedup.nlargest(1, 'efficiency_score')
        if len(best_efficiency) > 0:
            report_lines.append(f"\nBest Overall Efficiency: {best_efficiency.iloc[0]['method'].upper()}")
            report_lines.append(f"  Efficiency Score: {best_efficiency.iloc[0]['efficiency_score']:.3f}")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("End of HTA Analysis Report")
    report_lines.append("=" * 80)
    
    report = "\n".join(report_lines)
    
    # Save report
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"[OK] HTA analysis report saved to {output_path}")
    return report


def create_performance_visualizations(
    results_df: pd.DataFrame,
    output_dir: str = "./HTA_visualizations"
) -> None:
    """
    Create performance visualization plots.
    
    Args:
        results_df: DataFrame with experiment results
        output_dir: Directory to save visualization files
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print(" matplotlib/seaborn not available. Skipping visualizations.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter non-profiling results
    non_profiling = results_df[~results_df['method'].str.contains('profiling', case=False, na=False)].copy()
    
    if len(non_profiling) == 0:
        print(" No data available for visualization")
        return
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Throughput Comparison
    plt.figure()
    throughput_data = non_profiling[['method', 'throughput_samples_per_sec']].dropna()
    if len(throughput_data) > 0:
        throughput_data = throughput_data.sort_values('throughput_samples_per_sec', ascending=True)
        plt.barh(throughput_data['method'], throughput_data['throughput_samples_per_sec'])
        plt.xlabel('Throughput (samples/second)')
        plt.ylabel('Method')
        plt.title('Throughput Comparison Across Methods')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/throughput_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Memory Usage Comparison
    plt.figure()
    memory_data = non_profiling[['method', 'gpu_memory_max_allocated_gb']].dropna()
    if len(memory_data) > 0:
        memory_data = memory_data.sort_values('gpu_memory_max_allocated_gb', ascending=True)
        plt.barh(memory_data['method'], memory_data['gpu_memory_max_allocated_gb'])
        plt.xlabel('GPU Memory Usage (GB)')
        plt.ylabel('Method')
        plt.title('Memory Usage Comparison Across Methods')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/memory_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Throughput vs Memory Trade-off
    plt.figure()
    tradeoff_data = non_profiling[['method', 'throughput_samples_per_sec', 'gpu_memory_max_allocated_gb']].dropna()
    if len(tradeoff_data) > 0:
        plt.scatter(
            tradeoff_data['gpu_memory_max_allocated_gb'],
            tradeoff_data['throughput_samples_per_sec'],
            s=100,
            alpha=0.6
        )
        for _, row in tradeoff_data.iterrows():
            plt.annotate(
                row['method'],
                (row['gpu_memory_max_allocated_gb'], row['throughput_samples_per_sec']),
                fontsize=8
            )
        plt.xlabel('GPU Memory Usage (GB)')
        plt.ylabel('Throughput (samples/second)')
        plt.title('Throughput vs Memory Trade-off')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/throughput_memory_tradeoff.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"[OK] Visualizations saved to {output_dir}/")


def run_hta_analysis(
    csv_path: str = "experiment_results.csv",
    output_report: str = "HTA_analysis_report.txt",
    create_plots: bool = True
) -> Dict:
    """
    Run complete HTA analysis on experiment results.
    
    Args:
        csv_path: Path to experiment results CSV
        output_report: Path to save analysis report
        create_plots: Whether to create visualization plots
        
    Returns:
        Dictionary with analysis results
    """
    print("=" * 80)
    print("HTA (PERFORMANCE) ANALYSIS")
    print("=" * 80)
    print("")
    
    # Load results
    if not os.path.exists(csv_path):
        print(f" Results file not found: {csv_path}")
        return {}
    
    results_df = pd.read_csv(csv_path)
    print(f" Loaded {len(results_df)} experiment results from {csv_path}")
    
    # Generate report
    report = generate_hta_report(results_df, output_report)
    
    # Create visualizations
    if create_plots:
        create_performance_visualizations(results_df)
    
    # Calculate metrics
    non_profiling = results_df[~results_df['method'].str.contains('profiling', case=False, na=False)].copy()
    results_with_speedup = calculate_speedup_metrics(non_profiling)
    scaling_analysis = analyze_scaling_efficiency(non_profiling)
    
    analysis_results = {
        'report_path': output_report,
        'total_experiments': len(results_df),
        'non_profiling_experiments': len(non_profiling),
        'scaling_analysis': scaling_analysis,
        'speedup_metrics': results_with_speedup[['method', 'speedup_time', 'speedup_throughput', 'memory_reduction_pct', 'efficiency_score']].to_dict('records') if 'speedup_time' in results_with_speedup.columns else []
    }
    
    print("\n[OK] HTA analysis complete!")
    print(f"  Report: {output_report}")
    if create_plots:
        print(f"  Visualizations: ./HTA_visualizations/")
    
    return analysis_results

