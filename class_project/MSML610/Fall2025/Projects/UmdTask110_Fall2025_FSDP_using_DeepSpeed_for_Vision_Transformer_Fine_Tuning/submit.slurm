#!/bin/bash
#SBATCH --job-name=vit_imagenet_training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules (adjust these for your HPCC environment)
# module load python/3.11
# module load cuda/12.1
# module load cudnn/8.9

# Activate your Python environment
# Replace 'distributed_vision_training_env' with your actual environment name
source distributed_vision_training_env/bin/activate

# Set environment variables for PyTorch distributed
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export NCCL_DEBUG=INFO

# Print environment info
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $GPU_COUNT"

# ============================================================================
# OPTION 1: Test Environment (Uncomment to run)
# ============================================================================
# srun python test_environment.py

# ============================================================================
# OPTION 2: Standard DDP Training (Uncomment to run)
# ============================================================================
# srun python train_standard_ddp.py \
#     --data-path /path/to/imagenet \
#     --batch-size 64 \
#     --epochs 2 \
#     --learning-rate 1e-4 \
#     --num-workers 8

# Or using torchrun:
# torchrun --nproc_per_node=4 train_standard_ddp.py \
#     --data-path /path/to/imagenet \
#     --batch-size 64 \
#     --epochs 2 \
#     --learning-rate 1e-4 \
#     --num-workers 8

# ============================================================================
# OPTION 3: DeepSpeed Training (Uncomment to run)
# ============================================================================
# srun deepspeed --num_gpus=4 train_deepspeed_stage3.py \
#     --data-path /path/to/imagenet \
#     --deepspeed-config deepspeed_config_stage3.json \
#     --epochs 2 \
#     --num-workers 8

# Or using DeepSpeed with explicit backend:
# srun --gres=gpu:4 deepspeed train_deepspeed_stage3.py \
#     --data-path /path/to/imagenet \
#     --deepspeed-config deepspeed_config_stage3.json \
#     --epochs 2 \
#     --num-workers 8

# ============================================================================
# Example: Run both training scripts sequentially for comparison
# ============================================================================
# echo "Starting Standard DDP Training..."
# srun python train_standard_ddp.py \
#     --data-path /path/to/imagenet \
#     --batch-size 64 \
#     --epochs 1 \
#     --learning-rate 1e-4 \
#     --num-workers 8

# echo "Starting DeepSpeed Training..."
# srun deepspeed --num_gpus=4 train_deepspeed_stage3.py \
#     --data-path /path/to/imagenet \
#     --deepspeed-config deepspeed_config_stage3.json \
#     --epochs 1 \
#     --num-workers 8

# Print job completion
echo "Job completed at: $(date)"

