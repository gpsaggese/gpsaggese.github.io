#!/bin/bash
###############################################################################
# Slurm Job Script for Zaratan HPC Cluster (UMD)
#
# Account: msml610
# Partition: gpu
# GPUs: H100 (80GB VRAM, preferred) or A100 (40GB VRAM, fallback)
#
# Usage:
#   sbatch scripts/train_zaratan.sh [config_file]
#   sbatch scripts/train_zaratan.sh configs/h100.yaml
#
# Monitor:
#   squeue -u vikranth
#
# View output:
#   tail -f logs/horovod_transformer-<JOB_ID>.out
###############################################################################

#SBATCH --job-name=horovod_transformer
#SBATCH --account=msml610          # UMD MSML610 allocation
#SBATCH --partition=gpu            # GPU partition
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=4        # Number of tasks (processes) per node
#SBATCH --gres=gpu:H100:4          # Preferred: 4x H100 (80GB each)
# #SBATCH --gres=gpu:A100:4        # Fallback: 4x A100 (40GB each) - uncomment if H100s unavailable
#SBATCH --cpus-per-task=4          # CPUs per task
#SBATCH --mem=32G                  # Memory per node
#SBATCH --time=12:00:00            # Maximum runtime (12 hours)
#SBATCH --output=logs/%x-%j.out    # Output file (%x=job-name, %j=job-id)

# Email notifications (optional)
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=vikranth@umd.edu

###############################################################################
# IMPORTANT NOTES:
#
# 1. GPU Allocation Syntax:
#    Zaratan may use either:
#      --gpus-per-node=4  (recommended)
#    OR
#      --gres=gpu:4
#    Check your cluster documentation and uncomment the appropriate line.
#
# 2. Module System:
#    Zaratan uses environment modules. You need to load appropriate modules.
#    Common modules: cuda, anaconda, gcc, etc.
#    Check available modules with: module avail
#
# 3. Python Environment:
#    You should have a conda/virtual environment with Horovod installed.
#    Activate it in the "Setup Python Environment" section below.
#
# 4. Horovod Installation:
#    Horovod must be compiled with CUDA and NCCL support for GPU training.
#    If not available, follow installation instructions:
#    https://horovod.readthedocs.io/en/stable/install_include.html
###############################################################################

# Print job information
echo "=========================================="
echo "Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Working directory: $(pwd)"
echo "=========================================="
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

###############################################################################
# Module Loading (Zaratan-specific)
###############################################################################
echo "Loading modules..."

# Purge existing modules to avoid conflicts
module purge

# Load CUDA 12.3 for H100/A100 GPUs
module load cuda/12.3

echo "Loaded modules:"
module list
echo ""

###############################################################################
# Setup Python Environment
###############################################################################
echo "Setting up Python environment..."

# Activate Python environment if needed
# Uncomment ONE of the following based on your setup:
# source ~/envs/msml610/bin/activate                              # virtualenv
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate msml610  # conda
# (Or rely on system/container Python if already configured)

echo "Python version:"
python --version
echo ""

echo "PyTorch version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" || echo "PyTorch not found"
echo ""

echo "Horovod version:"
python -c "import horovod.torch as hvd; print(f'Horovod: {hvd.__version__}')" || echo "Horovod not found"
echo ""

###############################################################################
# Environment Variables
###############################################################################
echo "Setting environment variables..."

# Zaratan-specific paths (msml610 allocation)
export PROJECT_HOME=/afs/shell.umd.edu/project/msml610/user/vikranth
export SCRATCH_DIR=/scratch/zt1/project/msml610/vikranth

# HuggingFace cache directory (use scratch for large datasets/models)
export HF_HOME=$SCRATCH_DIR/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export DATASET_CACHE=$HF_HOME
mkdir -p $HF_HOME

# PyTorch environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

# NCCL settings (for multi-GPU training with H100/A100)
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0

# Horovod settings
export HOROVOD_GPU_OPERATIONS=NCCL
export HOROVOD_TIMELINE=$SCRATCH_DIR/timeline.json  # Optional: performance profiling

echo "Environment configured:"
echo "  PROJECT_HOME: $PROJECT_HOME"
echo "  SCRATCH_DIR: $SCRATCH_DIR"
echo "  HF_HOME: $HF_HOME"
echo ""

###############################################################################
# Print GPU Information
###############################################################################
echo "GPU Information:"
nvidia-smi
echo ""

###############################################################################
# Configuration
###############################################################################
# Accept config file as first argument, default to base.yaml
CONFIG_FILE=${1:-configs/base.yaml}

echo "Using configuration: $CONFIG_FILE"
echo ""

###############################################################################
# Run Distributed Training
###############################################################################
echo "=========================================="
echo "Starting Horovod Distributed Training"
echo "=========================================="
echo ""

# Dynamically determine number of processes
NP=${SLURM_NTASKS:-0}

# Fallback to SLURM_GPUS_PER_NODE if SLURM_NTASKS is not set
if [ "$NP" -le 0 ] && [ -n "$SLURM_GPUS_PER_NODE" ]; then
    NP=$SLURM_GPUS_PER_NODE
fi

# Fallback to SLURM_GPUS_ON_NODE if still not set
if [ "$NP" -le 0 ] && [ -n "$SLURM_GPUS_ON_NODE" ]; then
    NP=$SLURM_GPUS_ON_NODE
fi

# Final check
if [ "$NP" -le 0 ]; then
    echo "[ERROR] Could not infer number of processes from Slurm environment."
    echo "        Set SLURM_NTASKS, SLURM_GPUS_PER_NODE, or SLURM_GPUS_ON_NODE."
    exit 1
fi

echo "Launching Horovod training:"
echo "  Processes: $NP"
echo "  Config: $CONFIG_FILE"
echo "  HF Cache: $HF_HOME"
echo ""

# Run with horovodrun
# -np: number of processes (dynamically determined)
horovodrun -np $NP \
    python -m src.train --config "$CONFIG_FILE"

# Capture exit code
EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Training finished with exit code: $EXIT_CODE"
echo "=========================================="
echo ""

# Print training summary
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo ""
    echo "Checkpoints saved in: checkpoints/"
    echo "Logs saved in: logs/"
    echo "TensorBoard logs in: runs/"
    echo ""
    echo "To view TensorBoard:"
    echo "  tensorboard --logdir runs/"
else
    echo "Training failed! Check error logs."
fi

exit $EXIT_CODE

