"""
Main training script for Horovod distributed training.

Implements:
- Adaptive GPU logic (smoke test if world_size < 2)
- Horovod distributed training
- Checkpointing and logging
- Training and validation loops
"""

import argparse
import math
import os
import sys
import time
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .utils.distributed import setup_horovod, metric_average, barrier as dist_barrier
from .utils.config import load_config
from .utils.logging import setup_logging, TensorBoardLogger
from .utils.recorder import RunRecorder
from .data import get_dataloaders
from .models.transformer_lm import TransformerLM
from .metrics import compute_perplexity, MetricsTracker, compute_accuracy_for_allreduce


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    """
    Create a learning rate scheduler with linear warmup and linear decay.
    
    Args:
        optimizer: Optimizer.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        
    Returns:
        Learning rate scheduler.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return LambdaLR(optimizer, lr_lambda)


def run_smoke_test():
    """
    Run a quick smoke test when only 1 GPU is available.
    Demonstrates that the code works without full distributed training.
    """
    print("[INFO] ===== Running Smoke Test =====")
    print("[INFO] This is a demonstration run only.")
    print("[INFO] For actual distributed training, allocate at least 2 GPUs.")
    
    # Create a tiny dummy model
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Run a few dummy forward/backward passes
    for i in range(5):
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"[INFO] Smoke test step {i+1}/5: loss = {loss.item():.4f}")
    
    print("[INFO] ===== Smoke Test Complete =====")
    print("[INFO] The code is working correctly.")
    print("[INFO] To run full training, launch with 2+ GPUs using horovodrun.")


def save_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    step: int,
    loss: float,
    checkpoint_dir: str,
    filename: Optional[str] = None,
    scheduler=None,
    scaler=None,
    global_step: Optional[int] = None
):
    """
    Save model checkpoint (rank 0 only).
    
    Args:
        model: Model to save.
        optimizer: Optimizer to save.
        epoch: Current epoch.
        step: Current step.
        loss: Current loss.
        checkpoint_dir: Directory to save checkpoints.
        filename: Checkpoint filename (auto-generated if None).
        scheduler: Learning rate scheduler (optional).
        scaler: AMP GradScaler (optional, for mixed precision).
        global_step: Global training step (optional).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if filename is None:
        filename = f"checkpoint_epoch{epoch}_step{step}.pt"
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    # Save scheduler state if provided
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save AMP scaler state if provided
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    # Save global step if provided
    if global_step is not None:
        checkpoint['global_step'] = global_step
    
    torch.save(checkpoint, checkpoint_path)
    print(f"[INFO] Checkpoint saved: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer = None,
    scheduler = None,
    scaler = None
):
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        model: Model to load weights into.
        optimizer: Optimizer to load state into (optional).
        scheduler: Learning rate scheduler to load state into (optional).
        scaler: AMP GradScaler to load state into (optional).
        
    Returns:
        Dictionary with checkpoint information.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    print(f"[INFO] Loaded checkpoint from: {checkpoint_path}")
    print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - Step: {checkpoint.get('step', 'N/A')}")
    print(f"  - Global Step: {checkpoint.get('global_step', 'N/A')}")
    print(f"  - Loss: {checkpoint.get('loss', 'N/A'):.4f}")
    
    return checkpoint


def train_epoch(
    model: nn.Module,
    train_dataloader,
    optimizer,
    scheduler,
    epoch: int,
    config,
    logger,
    tb_logger: TensorBoardLogger,
    rank: int,
    world_size: int,
    device: torch.device,
    use_amp: bool = False,
    scaler = None,
    pad_token_id: int = 50256,
    recorder: Optional[RunRecorder] = None,
    amp_dtype = None
):
    """
    Train for one epoch.
    
    Args:
        model: Model to train.
        train_dataloader: Training dataloader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        epoch: Current epoch number.
        config: Configuration object.
        logger: Logger instance.
        tb_logger: TensorBoard logger.
        rank: Process rank.
        world_size: Number of processes.
        device: Device to use for training.
        
    Returns:
        Average training loss for the epoch.
    """
    model.train()
    metrics_tracker = MetricsTracker()
    
    total_steps = len(train_dataloader)
    start_time = time.time()
    
    # Gradient accumulation
    accumulation_steps = getattr(config.training, 'gradient_accumulation_steps', 1)
    accumulation_counter = 0
    
    # Track metrics across accumulation steps
    accumulated_loss_sum = 0.0
    accumulated_correct_sum = 0.0
    accumulated_total_sum = 0.0
    
    for step, batch in enumerate(train_dataloader):
        # Compute update-based global_step for TensorBoard alignment
        updates_per_epoch = math.ceil(total_steps / accumulation_steps)
        current_update = (step // accumulation_steps)
        global_step = epoch * updates_per_epoch + current_update
        
        # Move batch to device (non-blocking for faster transfer)
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        
        # Forward pass (with mixed precision if enabled)
        if use_amp:
            from torch.cuda.amp import autocast
            with autocast(dtype=amp_dtype):
                logits, loss = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            # Scale loss by accumulation steps
            loss = loss / accumulation_steps
            
            # Backward pass with scaler
            scaler.scale(loss).backward()
        else:
            logits, loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            # Scale loss by accumulation steps
            loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()
        
        accumulation_counter += 1
        
        # Accumulate metrics across micro-batches (unscaled for averaging)
        unscaled_loss = loss.item() * accumulation_steps
        accumulated_loss_sum += unscaled_loss
        
        # Compute accuracy for this micro-batch (shifted for next-token prediction)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        correct_count, total_count = compute_accuracy_for_allreduce(
            shift_logits, shift_labels, ignore_index=-100
        )
        accumulated_correct_sum += correct_count
        accumulated_total_sum += total_count
        
        # Update weights only after accumulating gradients
        is_update_step = (accumulation_counter % accumulation_steps == 0)
        
        if is_update_step:
            # Skip synchronization for micro-steps in Horovod (only sync on update step)
            if world_size > 1:
                try:
                    import horovod.torch as hvd
                    # Only synchronize on the actual update step
                    if use_amp and scaler is not None:
                        # Unscale gradients before clipping
                        scaler.unscale_(optimizer)
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                        # Optimizer step with scaler (synchronizes here)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                        # Optimizer step (synchronizes here)
                        optimizer.step()
                except ImportError:
                    # Fallback if Horovod not available
                    if use_amp and scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                        optimizer.step()
            else:
                # Single process
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                    optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
            accumulation_counter = 0
            
            # Compute average metrics over all accumulation steps (not just last micro-batch)
            avg_loss = accumulated_loss_sum / accumulation_steps
            avg_accuracy = accumulated_correct_sum / accumulated_total_sum if accumulated_total_sum > 0 else 0.0
            
            # Average across all processes
            avg_loss = metric_average(avg_loss, name=f'train_loss_epoch{epoch}_update{current_update}')
            avg_correct = metric_average(accumulated_correct_sum, name=f'train_correct_epoch{epoch}_update{current_update}')
            avg_total = metric_average(accumulated_total_sum, name=f'train_total_epoch{epoch}_update{current_update}')
            avg_accuracy = avg_correct / avg_total if avg_total > 0 else 0.0
            
            # Update metrics tracker
            metrics_tracker.update(avg_loss, logits, labels, accuracy=avg_accuracy)
            
            # Reset accumulation counters
            accumulated_loss_sum = 0.0
            accumulated_correct_sum = 0.0
            accumulated_total_sum = 0.0
        
        # Logging (only on update steps and at log interval)
        if is_update_step:
            # Use update-based step counting for log interval
            update_steps = (step // accumulation_steps) + 1
            log_interval_updates = config.training.log_interval // accumulation_steps if accumulation_steps > 1 else config.training.log_interval
            
            if update_steps % max(1, log_interval_updates) == 0:
                current_lr = scheduler.get_last_lr()[0]
                perplexity = compute_perplexity(avg_loss)
                elapsed = time.time() - start_time
                updates_per_sec = update_steps / elapsed if elapsed > 0 else 0
                
                if rank == 0:
                    logger.info(
                        f"Epoch {epoch} | Micro-Step {step}/{total_steps} | Update {update_steps} | "
                        f"Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | Acc: {avg_accuracy:.4f} | "
                        f"LR: {current_lr:.6f} | Updates/s: {updates_per_sec:.2f}"
                    )
                    
                    tb_logger.add_scalar('train/loss', avg_loss, global_step)
                    tb_logger.add_scalar('train/perplexity', perplexity, global_step)
                    tb_logger.add_scalar('train/learning_rate', current_lr, global_step)
                    tb_logger.add_scalar('train/accuracy', avg_accuracy, global_step)

                    # Structured recorder (rank 0 only)
                    if recorder is not None:
                        recorder.log_train_step(
                            epoch=epoch,
                            update=update_steps,
                            global_step=global_step,
                            lr=current_lr,
                            loss=avg_loss,
                            perplexity=perplexity,
                            accuracy=avg_accuracy,
                            throughput=updates_per_sec,
                        )
        
        # Save checkpoint (only on update steps)
        if is_update_step and rank == 0:
            update_steps = (step // accumulation_steps) + 1
            if update_steps % config.training.save_interval == 0:
                updates_per_epoch = math.ceil(len(train_dataloader) / accumulation_steps)
                current_global_step = epoch * updates_per_epoch + (step // accumulation_steps)
                save_checkpoint(
                    model, optimizer, epoch, step, avg_loss,
                    config.paths.checkpoint_dir, scheduler=scheduler,
                    scaler=scaler, global_step=current_global_step
                )
    
    # Epoch summary
    avg_metrics = metrics_tracker.get_average_metrics()
    
    if rank == 0:
        logger.info(f"Epoch {epoch} complete:")
        logger.info(f"  - Avg Loss: {avg_metrics['loss']:.4f}")
        logger.info(f"  - Avg Perplexity: {avg_metrics['perplexity']:.2f}")
        if 'accuracy' in avg_metrics:
            logger.info(f"  - Avg Accuracy: {avg_metrics['accuracy']:.4f}")
    
    return avg_metrics['loss']


def validate(
    model: nn.Module,
    val_dataloader,
    epoch: int,
    config,
    logger,
    tb_logger: TensorBoardLogger,
    rank: int,
    device: torch.device,
    use_amp: bool = False,
    pad_token_id: int = 50256,
    updates_per_epoch: int = 0,
    amp_dtype = None
):
    """
    Run validation.
    
    Args:
        model: Model to validate.
        val_dataloader: Validation dataloader.
        epoch: Current epoch number.
        config: Configuration object.
        logger: Logger instance.
        tb_logger: TensorBoard logger.
        rank: Process rank.
        device: Device to use for validation.
        use_amp: Whether to use mixed precision.
        pad_token_id: Padding token ID.
        
    Returns:
        Tuple of (average validation loss, metrics dictionary).
    """
    model.eval()
    metrics_tracker = MetricsTracker()
    
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            # Move batch to device (non-blocking for faster transfer)
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            # Forward pass (with mixed precision if enabled)
            if use_amp:
                from torch.cuda.amp import autocast
                with autocast(dtype=amp_dtype):
                    logits, loss = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
            else:
                logits, loss = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            
            # Compute metrics (average across all processes)
            loss_value = loss.item()
            avg_loss = metric_average(loss_value, name=f'val_loss_epoch{epoch}_step{step}')
            
            # Compute accuracy with proper distributed averaging (shifted for next-token prediction)
            # Shift logits and labels to align with next-token prediction (same as loss computation)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            correct_count, total_count = compute_accuracy_for_allreduce(shift_logits, shift_labels, ignore_index=-100)
            avg_correct = metric_average(correct_count, name=f'val_correct_epoch{epoch}_step{step}')
            avg_total = metric_average(total_count, name=f'val_total_epoch{epoch}_step{step}')
            accuracy = avg_correct / avg_total if avg_total > 0 else 0.0
            
            metrics_tracker.update(avg_loss, logits, labels, accuracy=accuracy)
    
    # Validation summary
    avg_metrics = metrics_tracker.get_average_metrics()
    
    if rank == 0:
        logger.info(f"Validation (Epoch {epoch}):")
        logger.info(f"  - Val Loss: {avg_metrics['loss']:.4f}")
        logger.info(f"  - Val Perplexity: {avg_metrics['perplexity']:.2f}")
        if 'accuracy' in avg_metrics:
            logger.info(f"  - Val Accuracy: {avg_metrics['accuracy']:.4f}")
        
        # Log to TensorBoard (use update-based global_step from training)
        # Align validation step to end-of-epoch update count
        global_step = (epoch + 1) * (updates_per_epoch if updates_per_epoch > 0 else 1)
        tb_logger.add_scalar('val/loss', avg_metrics['loss'], global_step)
        tb_logger.add_scalar('val/perplexity', avg_metrics['perplexity'], global_step)
        if 'accuracy' in avg_metrics:
            tb_logger.add_scalar('val/accuracy', avg_metrics['accuracy'], global_step)
    
    return avg_metrics['loss'], avg_metrics


def run_distributed_training(
    config,
    rank: int,
    world_size: int,
    local_rank: int,
    resume_checkpoint: Optional[str] = None,
    run_name: Optional[str] = None,
    output_dir: Optional[str] = None,
):
    """
    Run distributed training with Horovod.
    
    Args:
        config: Configuration object.
        rank: Process rank.
        world_size: Number of processes.
        local_rank: Local rank (GPU index) for this process.
        resume_checkpoint: Path to checkpoint to resume from (optional).
    """
    # Setup logging
    logger = setup_logging(
        log_dir=config.paths.log_dir,
        rank=rank
    )
    
    # Determine device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(local_rank)
    else:
        device = torch.device('cpu')
    
    logger.info(f"Starting distributed training:")
    logger.info(f"  - World size: {world_size}")
    logger.info(f"  - Rank: {rank}")
    logger.info(f"  - Local rank: {local_rank}")
    logger.info(f"  - Device: {device}")
    logger.info(f"  - Model: Custom Transformer (d_model={config.model.d_model}, layers={config.model.n_layers})")
    logger.info(f"  - Epochs: {config.training.epochs}")
    logger.info(f"  - Per-GPU batch size: {config.training.per_gpu_batch_size}")
    logger.info(f"  - Global batch size: {config.training.per_gpu_batch_size * world_size}")
    
    # Setup TensorBoard (rank 0 only)
    tb_logger = TensorBoardLogger(config.paths.tensorboard_dir, rank)

    # Setup structured run recorder (rank 0 only)
    recorder = None
    try:
        out_dir = output_dir or os.path.join(config.paths.tensorboard_dir, "structured")
        recorder = RunRecorder(base_dir=out_dir, rank=rank, config=config, run_name=run_name)
        if rank == 0 and recorder.directory:
            logger.info(f"RunRecorder active. Run directory: {recorder.directory}")
    except Exception as e:
        if rank == 0:
            logger.warning(f"Failed to initialize RunRecorder: {e}")
    
    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Also seed Python random and numpy for complete reproducibility
    import random
    import numpy as np
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    # Load data
    logger.info("Loading data...")
    train_dataloader, val_dataloader = get_dataloaders(config)
    logger.info("Data prepared on rank %s. Waiting for all ranks before training...", rank)
    dist_barrier()
    logger.info("All ranks synchronized after data loading.")
    
    # Create model
    logger.info("Creating custom transformer model...")
    model = TransformerLM(
        vocab_size=config.model.vocab_size,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout,
        gradient_checkpointing=getattr(config.training, 'gradient_checkpointing', False)
    )
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Move model to device
    model = model.to(device)
    
    # Enable TF32 on Ampere/Hopper GPUs for faster training (if available)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 enabled for faster training (Ampere/Hopper GPUs)")
    
    # Optional: torch.compile for PyTorch 2.x (test first on H100)
    # Uncomment if you want to try it:
    # if hasattr(torch, 'compile'):
    #     model = torch.compile(model)
    #     logger.info("Model compiled with torch.compile")
    
    # Setup mixed precision training
    use_amp = False
    amp_dtype = None
    scaler = None
    if torch.cuda.is_available():
        # Prefer new config: training.amp_dtype: bf16|fp16|fp32
        amp_key = getattr(config.training, 'amp_dtype', None)
        if isinstance(amp_key, str):
            key = amp_key.strip().lower()
            if key == 'bf16' or key == 'bfloat16':
                use_amp = True
                amp_dtype = torch.bfloat16
                logger.info("Mixed precision training (BF16) enabled")
            elif key == 'fp16' or key == 'float16':
                use_amp = True
                amp_dtype = torch.float16
                from torch.cuda.amp import GradScaler
                scaler = GradScaler()
                logger.info("Mixed precision training (FP16) enabled")
            else:
                logger.info("AMP disabled (amp_dtype set to FP32 or unrecognized)")
        else:
            # Backward compatibility: fp16 boolean at root of config
            if getattr(config, 'fp16', False):
                use_amp = True
                amp_dtype = torch.float16
                from torch.cuda.amp import GradScaler
                scaler = GradScaler()
                logger.info("Mixed precision training (FP16) enabled [legacy flag]")
    
    # Get pad_token_id from model or config
    if hasattr(model, 'pad_token_id'):
        pad_token_id = model.pad_token_id
    elif hasattr(model, 'config') and hasattr(model.config, 'pad_token_id'):
        pad_token_id = model.config.pad_token_id
    else:
        pad_token_id = getattr(config.model, 'pad_token_id', 50256)
    
    logger.info(f"Using pad_token_id: {pad_token_id}")
    
    # Create optimizer (use fused=True on CUDA if available for speedup)
    # Optionally scale LR by world size for strong scaling
    base_lr = config.training.learning_rate
    if getattr(config.training, 'scale_lr_by_world_size', False) and world_size > 1:
        scaled_lr = base_lr * world_size
        logger.info(f"Scaling learning rate by world size: {base_lr} -> {scaled_lr}")
        base_lr = scaled_lr

    optimizer_kwargs = {
        'lr': base_lr,
        'weight_decay': config.training.weight_decay
    }
    # Enable fused AdamW on CUDA if supported (PyTorch 2.0+)
    if torch.cuda.is_available() and hasattr(torch.optim.AdamW, '__init__'):
        try:
            # Test if fused parameter is supported
            import inspect
            sig = inspect.signature(torch.optim.AdamW.__init__)
            if 'fused' in sig.parameters:
                optimizer_kwargs['fused'] = True
                logger.info("Using fused AdamW optimizer for faster training")
        except Exception:
            pass  # Fall back to regular AdamW
    
    optimizer = AdamW(model.parameters(), **optimizer_kwargs)
    
    # Wrap optimizer with Horovod DistributedOptimizer
    try:
        import horovod.torch as hvd
        
        if world_size > 1:
            # Optional Horovod compression
            try:
                dist_cfg = getattr(config, 'distributed', None)
                comp_val = None
                if dist_cfg is not None and hasattr(dist_cfg, 'get'):
                    comp_val = dist_cfg.get('horovod_compression', 'none')
                elif dist_cfg is not None:
                    comp_val = getattr(dist_cfg, 'horovod_compression', 'none')
                comp_val = (comp_val or 'none').lower()
                compression = hvd.Compression.fp16 if comp_val == 'fp16' else hvd.Compression.none
                if comp_val == 'fp16':
                    logger.info("Using Horovod compression: fp16")
            except Exception:
                compression = hvd.Compression.none

            optimizer = hvd.DistributedOptimizer(
                optimizer,
                named_parameters=model.named_parameters(),
                compression=compression
            )
            
            # Broadcast initial parameters from rank 0 to all other processes
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            
            # Skip optimizer state broadcast to avoid NumPy compatibility issues
            # The optimizer will sync naturally during the first backward pass
            # hvd.broadcast_optimizer_state(optimizer, root_rank=0)
            
            logger.info("Horovod optimizer initialized successfully (skipped optimizer state broadcast).")
        else:
            logger.info("Single-process mode: Horovod optimizer not needed.")
        
    except Exception as e:
        if world_size > 1:
            logger.error(f"Failed to initialize Horovod optimizer: {e}")
            raise RuntimeError("Horovod distributed setup failed — aborting.") from e
        else:
            logger.warning(f"Horovod unavailable; running single-process mode: {e}")
    
    # Create learning rate scheduler
    # Account for gradient accumulation: scheduler steps only on optimizer updates
    accumulation_steps = getattr(config.training, 'gradient_accumulation_steps', 1)
    updates_per_epoch = math.ceil(len(train_dataloader) / accumulation_steps)
    num_training_steps = updates_per_epoch * config.training.epochs
    
    # Scale warmup steps to match optimizer update steps
    warmup_steps = math.ceil(config.training.warmup_steps / accumulation_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
    logger.info(f"Training configuration:")
    logger.info(f"  - Batches per epoch: {len(train_dataloader)}")
    logger.info(f"  - Gradient accumulation steps: {accumulation_steps}")
    logger.info(f"  - Optimizer updates per epoch: {updates_per_epoch}")
    logger.info(f"  - Total optimizer updates: {num_training_steps} ({config.training.epochs} epochs)")
    logger.info(f"  - Warmup steps (scaled): {warmup_steps}")
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
        if rank == 0:
            logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = load_checkpoint(resume_checkpoint, model, optimizer, scheduler, scaler)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('loss', float('inf'))
        global_step_start = checkpoint.get('global_step', 0)
        if rank == 0:
            logger.info(f"Resuming from epoch {start_epoch}, step {global_step_start}, best_val_loss: {best_val_loss:.4f}")
    
    # Training loop
    train_loss = None
    val_loss = None
    
    for epoch in range(start_epoch, config.training.epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{config.training.epochs}")
        logger.info(f"{'='*60}")
        
        # Set epoch for distributed sampler (ensures proper shuffling)
        train_sampler = getattr(train_dataloader, "sampler", None)
        if train_sampler is not None:
            from torch.utils.data import DistributedSampler
            if isinstance(train_sampler, DistributedSampler):
                train_sampler.set_epoch(epoch)
                logger.info(f"Set sampler epoch to {epoch}")
        
        # Train
        updates_per_epoch = math.ceil(len(train_dataloader) / accumulation_steps)
        global_step_offset = epoch * updates_per_epoch
        
        train_loss = train_epoch(
            model, train_dataloader, optimizer, scheduler,
            epoch, config, logger, tb_logger, rank, world_size, device,
            use_amp=use_amp, scaler=scaler, pad_token_id=pad_token_id, recorder=recorder, amp_dtype=amp_dtype
        )
        
        # Validate every epoch
        val_loss, val_metrics = validate(
            model, val_dataloader, epoch, config,
            logger, tb_logger, rank, device,
            use_amp=use_amp, pad_token_id=pad_token_id, updates_per_epoch=updates_per_epoch, amp_dtype=amp_dtype
        )

        # Record validation metrics to structured log
        if rank == 0 and recorder is not None:
            global_step_val = (epoch + 1) * (updates_per_epoch if updates_per_epoch > 0 else 1)
            recorder.log_val_epoch(
                epoch=epoch,
                global_step=global_step_val,
                loss=val_loss,
                perplexity=val_metrics['perplexity'],
                accuracy=val_metrics.get('accuracy')
            )
        
        # Save best model (rank 0 only)
        if rank == 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            global_step = global_step_offset + updates_per_epoch
            save_checkpoint(
                model, optimizer, epoch, len(train_dataloader),
                val_loss, config.paths.checkpoint_dir,
                filename="best_model.pt", scheduler=scheduler,
                scaler=scaler, global_step=global_step
            )
            logger.info(f"New best model saved (val_loss: {val_loss:.4f})")
        
        # Save epoch checkpoint (rank 0 only)
        if rank == 0:
            global_step = global_step_offset + updates_per_epoch
            save_checkpoint(
                model, optimizer, epoch, len(train_dataloader),
                train_loss, config.paths.checkpoint_dir,
                filename=f"checkpoint_epoch{epoch}.pt", scheduler=scheduler,
                scaler=scaler, global_step=global_step
            )
    
    # Final save
    if rank == 0:
        # Use the last validation loss if available, otherwise use last train loss
        final_loss = val_loss if val_loss is not None else (train_loss if train_loss is not None else 0.0)
        updates_per_epoch = math.ceil(len(train_dataloader) / accumulation_steps)
        final_global_step = config.training.epochs * updates_per_epoch
        save_checkpoint(
            model, optimizer, config.training.epochs - 1,
            len(train_dataloader), final_loss,
            config.paths.checkpoint_dir,
            filename="final_model.pt", scheduler=scheduler,
            scaler=scaler, global_step=final_global_step
        )
        logger.info("Training complete!")
    
    # Cleanup
    try:
        tb_logger.close()
    except Exception:
        pass
    try:
        if recorder is not None:
            recorder.close()
    except Exception:
        pass


def main():
    """
    Main entry point for training script.
    """
    parser = argparse.ArgumentParser(description="Horovod Distributed Training for Transformer LM")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional name to include in structured run directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Base directory for structured run logs (defaults to runs/structured)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"[INFO] Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Initialize Horovod
    world_size, rank, local_rank, use_cuda = setup_horovod(require_distributed=True)
    
    # Check if we have enough GPUs for distributed training
    if world_size < 2:
        if rank == 0:
            print("[INFO] ===================================================")
            print("[INFO] Insufficient GPUs for distributed training.")
            print("[INFO] This project requires at least 2 GPUs.")
            print("[INFO] Running smoke test instead...")
            print("[INFO] ===================================================\n")
            run_smoke_test()
        sys.exit(0)
    
    # Run distributed training
    try:
        run_distributed_training(
            config, rank, world_size, local_rank,
            resume_checkpoint=args.resume,
            run_name=args.run_name,
            output_dir=args.output_dir,
        )
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
