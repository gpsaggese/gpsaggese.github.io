#!/usr/bin/env python3
"""
Test script to verify PyTorch distributed environment setup.
This script helps debug the Slurm/PyTorch distributed setup quickly.
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def print_environment_info():
    """Print basic environment information."""
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")


def test_distributed_setup():
    """Test PyTorch distributed initialization."""
    # Get environment variables
    rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', 0)))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '29500')
    
    print(f"\nEnvironment variables:")
    print(f"  LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'not set')}")
    print(f"  RANK: {os.environ.get('RANK', 'not set')}")
    print(f"  WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'not set')}")
    print(f"  MASTER_ADDR: {master_addr}")
    print(f"  MASTER_PORT: {master_port}")
    
    # Initialize process group
    if world_size > 1:
        try:
            dist.init_process_group(
                backend='nccl',
                init_method=f'tcp://{master_addr}:{master_port}',
                rank=rank,
                world_size=world_size
            )
            print(f"\n✓ Successfully initialized distributed process group")
            print(f"  Rank: {rank}")
            print(f"  World size: {world_size}")
            
            # Test all_reduce
            tensor = torch.ones(1).cuda() * rank
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            print(f"  All-reduce test passed. Sum: {tensor.item()}")
            
            dist.destroy_process_group()
            print(f"✓ Process group destroyed successfully")
            
        except Exception as e:
            print(f"✗ Failed to initialize distributed process group: {e}")
            return False
    else:
        print(f"\nRunning in single-process mode (rank={rank}, world_size={world_size})")
    
    return True


def test_gpu_info():
    """Print detailed GPU information."""
    if not torch.cuda.is_available():
        print("\nNo CUDA devices available")
        return
    
    print(f"\nGPU Information:")
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"  Multiprocessors: {props.multi_processor_count}")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        
        # Current memory usage
        torch.cuda.set_device(i)
        memory_allocated = torch.cuda.memory_allocated(i) / 1e9
        memory_reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"  Memory Allocated: {memory_allocated:.2f} GB")
        print(f"  Memory Reserved: {memory_reserved:.2f} GB")


def main():
    """Main test function."""
    print("=" * 60)
    print("PyTorch Distributed Environment Test")
    print("=" * 60)
    
    # Print environment info
    print_environment_info()
    
    # Print GPU info
    test_gpu_info()
    
    # Test distributed setup
    test_distributed_setup()
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()

