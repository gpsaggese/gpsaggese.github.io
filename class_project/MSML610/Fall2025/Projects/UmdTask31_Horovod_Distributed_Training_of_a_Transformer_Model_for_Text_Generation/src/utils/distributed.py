"""
Distributed training utilities using Horovod.

Provides helpers for:
- Horovod initialization with graceful fallback
- Rank-aware operations (logging, saving, etc.)
- GPU device setup
"""

import os
import sys
from typing import Optional, Tuple

import torch


def setup_horovod(require_distributed: bool = True) -> Tuple[int, int, int, bool]:
    """
    Initialize Horovod and return distributed training information.
    
    Args:
        require_distributed: If True and world_size < 2, exit after smoke test.
        
    Returns:
        Tuple of (world_size, rank, local_rank, use_cuda)
        
    Raises:
        SystemExit: If Horovod not available or world_size < 2 (when required)
    """
    try:
        import horovod.torch as hvd
    except ImportError:
        print("[ERROR] Horovod not installed. Cannot perform distributed training.")
        print("[INFO] Running CPU-only smoke test and exiting.")
        return 1, 0, 0, False
    
    try:
        hvd.init()
    except Exception as e:
        print(f"[ERROR] Failed to initialize Horovod: {e}")
        print("[INFO] Running single-process mode.")
        return 1, 0, 0, torch.cuda.is_available()
    
    world_size = hvd.size()
    rank = hvd.rank()
    local_rank = hvd.local_rank()
    
    # Setup CUDA device
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device_name = torch.cuda.get_device_name(local_rank)
        if rank == 0:
            print(f"[INFO] Using CUDA device: {device_name}")
    else:
        if rank == 0:
            print("[WARN] CUDA not available. Running on CPU.")
    
    # Check if we have enough GPUs for distributed training
    if world_size < 2:
        if rank == 0:
            print(f"[WARN] Only {world_size} process(es) detected.")
            if require_distributed:
                print("[WARN] Distributed training requires at least 2 GPUs.")
                print("[INFO] This project requires distributed training. Exiting.")
        return world_size, rank, local_rank, use_cuda
    
    if rank == 0:
        print(f"[INFO] Horovod initialized successfully:")
        print(f"  - World size: {world_size}")
        print(f"  - Using CUDA: {use_cuda}")
        if use_cuda:
            print(f"  - GPU(s) per node: {torch.cuda.device_count()}")
    
    return world_size, rank, local_rank, use_cuda


def is_main_process(rank: Optional[int] = None) -> bool:
    """
    Check if current process is the main process (rank 0).
    
    Args:
        rank: Process rank. If None, will try to get from Horovod.
        
    Returns:
        True if main process, False otherwise.
    """
    if rank is not None:
        return rank == 0
    
    try:
        import horovod.torch as hvd
        return hvd.rank() == 0
    except (ImportError, RuntimeError):
        # If Horovod not available, assume single process
        return True


def print_once(message: str, rank: Optional[int] = None):
    """
    Print a message only from the main process.
    
    Args:
        message: Message to print.
        rank: Process rank. If None, will determine automatically.
    """
    if is_main_process(rank):
        print(message, flush=True)


def barrier():
    """
    Synchronization barrier for all processes.
    Only effective if Horovod is initialized.
    """
    try:
        import horovod.torch as hvd
        hvd.allreduce(torch.tensor(0), name='barrier')
    except (ImportError, RuntimeError):
        pass


def get_world_size() -> int:
    """
    Get the total number of processes in the distributed job.
    
    Returns:
        Number of processes (1 if not distributed).
    """
    try:
        import horovod.torch as hvd
        return hvd.size()
    except (ImportError, RuntimeError):
        return 1


def get_rank() -> int:
    """
    Get the rank of current process.
    
    Returns:
        Process rank (0 if not distributed).
    """
    try:
        import horovod.torch as hvd
        return hvd.rank()
    except (ImportError, RuntimeError):
        return 0


def get_local_rank() -> int:
    """
    Get the local rank of current process on this node.
    
    Returns:
        Local process rank (0 if not distributed).
    """
    try:
        import horovod.torch as hvd
        return hvd.local_rank()
    except (ImportError, RuntimeError):
        return 0


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """
    Reduce a tensor across all processes.
    
    Args:
        tensor: Tensor to reduce.
        average: If True, compute average; otherwise sum.
        
    Returns:
        Reduced tensor.
    """
    try:
        import horovod.torch as hvd
        tensor = tensor.clone()
        hvd.allreduce_(tensor, average=average)
        return tensor
    except (ImportError, RuntimeError):
        return tensor


def metric_average(value: float, name: str = 'metric') -> float:
    """
    Average a scalar metric across all processes.
    
    Args:
        value: Metric value to average.
        name: Name for the metric (for Horovod operation naming).
        
    Returns:
        Averaged metric value.
    """
    try:
        import horovod.torch as hvd
        tensor = torch.tensor(value)
        avg_tensor = hvd.allreduce(tensor, name=name, average=True)
        return avg_tensor.item()
    except (ImportError, RuntimeError):
        return value

