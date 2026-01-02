"""
Logging utilities for distributed training.

Provides rank-aware logging to console and file.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logging(
    log_dir: str = "logs",
    log_name: Optional[str] = None,
    rank: int = 0,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logging configuration for distributed training.
    
    Only rank 0 writes to file; all ranks can write to console.
    
    Args:
        log_dir: Directory to save log files.
        log_name: Name for the log file. If None, auto-generate with timestamp.
        rank: Process rank.
        level: Logging level.
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(f"horovod_training_rank{rank}")
    logger.setLevel(level)
    logger.handlers.clear()
    
    # Console handler for all ranks (but with rank prefix)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler only for rank 0
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        
        if log_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_name = f"train_{timestamp}.log"
        
        log_path = os.path.join(log_dir, log_name)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_path}")
    
    return logger


def get_logger(name: str = "horovod_training") -> logging.Logger:
    """
    Get or create a logger instance.
    
    Args:
        name: Logger name.
        
    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


class TensorBoardLogger:
    """
    Wrapper for TensorBoard SummaryWriter with rank awareness.
    Only rank 0 writes to TensorBoard.
    """
    
    def __init__(self, log_dir: str, rank: int = 0):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs.
            rank: Process rank.
        """
        self.rank = rank
        self.writer = None
        
        if rank == 0:
            try:
                from torch.utils.tensorboard import SummaryWriter
                os.makedirs(log_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir)
            except ImportError:
                print("[WARN] TensorBoard not available. Skipping TensorBoard logging.")
    
    def add_scalar(self, tag: str, scalar_value: float, global_step: int):
        """Add scalar value to TensorBoard."""
        if self.writer is not None and self.rank == 0:
            self.writer.add_scalar(tag, scalar_value, global_step)
    
    def add_scalars(self, main_tag: str, tag_scalar_dict: dict, global_step: int):
        """Add multiple scalar values to TensorBoard."""
        if self.writer is not None and self.rank == 0:
            self.writer.add_scalars(main_tag, tag_scalar_dict, global_step)
    
    def add_text(self, tag: str, text_string: str, global_step: int):
        """Add text to TensorBoard."""
        if self.writer is not None and self.rank == 0:
            self.writer.add_text(tag, text_string, global_step)
    
    def add_histogram(self, tag: str, values, global_step: int):
        """Add histogram to TensorBoard."""
        if self.writer is not None and self.rank == 0:
            self.writer.add_histogram(tag, values, global_step)
    
    def flush(self):
        """Flush pending events to disk."""
        if self.writer is not None and self.rank == 0:
            self.writer.flush()
    
    def close(self):
        """Close the TensorBoard writer."""
        if self.writer is not None and self.rank == 0:
            self.writer.close()

