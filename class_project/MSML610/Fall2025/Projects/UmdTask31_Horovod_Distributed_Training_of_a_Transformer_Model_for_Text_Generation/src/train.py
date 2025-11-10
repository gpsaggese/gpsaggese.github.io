"""
Main training script for Horovod distributed training.

Implements:
- Adaptive GPU logic (smoke test if world_size < 2)
- Horovod distributed training
- Checkpointing and logging
- Training and validation loops
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .utils.distributed import setup_horovod, is_main_process, print_once, metric_average
from .utils.config import load_config
from .utils.logging import setup_logging, TensorBoardLogger
from .data import get_dataloaders
from .models.hf_wrapper import get_model_from_config
from .metrics import compute_perplexity, MetricsTracker


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
    filename: Optional[str] = None
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
    
    torch.save(checkpoint, checkpoint_path)
    print(f"[INFO] Checkpoint saved: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer = None
):
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        model: Model to load weights into.
        optimizer: Optimizer to load state into (optional).
        
    Returns:
        Dictionary with checkpoint information.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"[INFO] Loaded checkpoint from: {checkpoint_path}")
    print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - Step: {checkpoint.get('step', 'N/A')}")
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
    world_size: int
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
        
    Returns:
        Average training loss for the epoch.
    """
    model.train()
    metrics_tracker = MetricsTracker()
    
    total_steps = len(train_dataloader)
    start_time = time.time()
    
    for step, batch in enumerate(train_dataloader):
        global_step = epoch * total_steps + step
        
        # Move batch to device
        input_ids = batch['input_ids'].cuda() if torch.cuda.is_available() else batch['input_ids']
        attention_mask = batch['attention_mask'].cuda() if torch.cuda.is_available() else batch['attention_mask']
        labels = batch['labels'].cuda() if torch.cuda.is_available() else batch['labels']
        
        # Forward pass
        logits, loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Compute metrics (average across all processes)
        loss_value = loss.item()
        avg_loss = metric_average(loss_value, name=f'train_loss_epoch{epoch}_step{step}')
        
        metrics_tracker.update(avg_loss, logits, labels)
        
        # Logging
        if step % config.training.log_interval == 0:
            current_lr = scheduler.get_last_lr()[0]
            perplexity = compute_perplexity(avg_loss)
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
            
            if rank == 0:
                logger.info(
                    f"Epoch {epoch} | Step {step}/{total_steps} | "
                    f"Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | "
                    f"LR: {current_lr:.6f} | Steps/s: {steps_per_sec:.2f}"
                )
                
                tb_logger.add_scalar('train/loss', avg_loss, global_step)
                tb_logger.add_scalar('train/perplexity', perplexity, global_step)
                tb_logger.add_scalar('train/learning_rate', current_lr, global_step)
        
        # Save checkpoint
        if rank == 0 and step > 0 and step % config.training.save_interval == 0:
            save_checkpoint(
                model, optimizer, epoch, step, avg_loss,
                config.paths.checkpoint_dir
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
    rank: int
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
        
    Returns:
        Average validation loss.
    """
    model.eval()
    metrics_tracker = MetricsTracker()
    
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            # Move batch to device
            input_ids = batch['input_ids'].cuda() if torch.cuda.is_available() else batch['input_ids']
            attention_mask = batch['attention_mask'].cuda() if torch.cuda.is_available() else batch['attention_mask']
            labels = batch['labels'].cuda() if torch.cuda.is_available() else batch['labels']
            
            # Forward pass
            logits, loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Compute metrics (average across all processes)
            loss_value = loss.item()
            avg_loss = metric_average(loss_value, name=f'val_loss_epoch{epoch}_step{step}')
            
            metrics_tracker.update(avg_loss, logits, labels)
    
    # Validation summary
    avg_metrics = metrics_tracker.get_average_metrics()
    
    if rank == 0:
        logger.info(f"Validation (Epoch {epoch}):")
        logger.info(f"  - Val Loss: {avg_metrics['loss']:.4f}")
        logger.info(f"  - Val Perplexity: {avg_metrics['perplexity']:.2f}")
        if 'accuracy' in avg_metrics:
            logger.info(f"  - Val Accuracy: {avg_metrics['accuracy']:.4f}")
        
        # Log to TensorBoard
        global_step = epoch * len(val_dataloader)
        tb_logger.add_scalar('val/loss', avg_metrics['loss'], global_step)
        tb_logger.add_scalar('val/perplexity', avg_metrics['perplexity'], global_step)
    
    return avg_metrics['loss']


def run_distributed_training(config, rank: int, world_size: int):
    """
    Run distributed training with Horovod.
    
    Args:
        config: Configuration object.
        rank: Process rank.
        world_size: Number of processes.
    """
    # Setup logging
    logger = setup_logging(
        log_dir=config.paths.log_dir,
        rank=rank
    )
    
    logger.info(f"Starting distributed training:")
    logger.info(f"  - World size: {world_size}")
    logger.info(f"  - Rank: {rank}")
    logger.info(f"  - Model type: {config.model.type}")
    logger.info(f"  - Epochs: {config.training.epochs}")
    logger.info(f"  - Per-GPU batch size: {config.training.per_gpu_batch_size}")
    logger.info(f"  - Global batch size: {config.training.per_gpu_batch_size * world_size}")
    
    # Setup TensorBoard (rank 0 only)
    tb_logger = TensorBoardLogger(config.paths.tensorboard_dir, rank)
    
    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Load data
    logger.info("Loading data...")
    train_dataloader, val_dataloader = get_dataloaders(config)
    
    # Create model
    logger.info("Creating model...")
    model = get_model_from_config(config)
    
    # Move model to GPU
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Wrap optimizer with Horovod DistributedOptimizer
    try:
        import horovod.torch as hvd
        
        if world_size > 1:
            optimizer = hvd.DistributedOptimizer(
                optimizer,
                named_parameters=model.named_parameters()
            )
            
            # Broadcast initial parameters from rank 0 to all other processes
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)
            
            logger.info("Horovod optimizer initialized successfully.")
        else:
            logger.info("Single-process mode: Horovod optimizer not needed.")
        
    except Exception as e:
        if world_size > 1:
            logger.error(f"Failed to initialize Horovod optimizer: {e}")
            raise RuntimeError("Horovod distributed setup failed — aborting.") from e
        else:
            logger.warning(f"Horovod unavailable; running single-process mode: {e}")
    
    # Create learning rate scheduler
    num_training_steps = len(train_dataloader) * config.training.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    logger.info(f"Training for {num_training_steps} steps ({config.training.epochs} epochs)")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config.training.epochs):
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
        train_loss = train_epoch(
            model, train_dataloader, optimizer, scheduler,
            epoch, config, logger, tb_logger, rank, world_size
        )
        
        # Validate
        if (epoch + 1) % 1 == 0:  # Validate every epoch
            val_loss = validate(
                model, val_dataloader, epoch, config,
                logger, tb_logger, rank
            )
            
            # Save best model (rank 0 only)
            if rank == 0 and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, epoch, len(train_dataloader),
                    val_loss, config.paths.checkpoint_dir,
                    filename="best_model.pt"
                )
                logger.info(f"New best model saved (val_loss: {val_loss:.4f})")
        
        # Save epoch checkpoint (rank 0 only)
        if rank == 0:
            save_checkpoint(
                model, optimizer, epoch, len(train_dataloader),
                train_loss, config.paths.checkpoint_dir,
                filename=f"checkpoint_epoch{epoch}.pt"
            )
    
    # Final save
    if rank == 0:
        save_checkpoint(
            model, optimizer, config.training.epochs - 1,
            len(train_dataloader), train_loss,
            config.paths.checkpoint_dir,
            filename="final_model.pt"
        )
        logger.info("Training complete!")
    
    # Cleanup
    tb_logger.close()


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
        run_distributed_training(config, rank, world_size)
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

