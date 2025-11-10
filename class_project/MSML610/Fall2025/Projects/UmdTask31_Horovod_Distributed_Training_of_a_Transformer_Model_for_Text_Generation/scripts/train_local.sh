#!/bin/bash
###############################################################################
# Local Training Script for Horovod Distributed Training
#
# This script runs distributed training on a local machine with multiple GPUs.
# Adjust the number of processes (-np) based on available GPUs.
###############################################################################

set -e  # Exit on error

# Configuration
NUM_GPUS=2  # Change this to match your local GPU count
CONFIG_FILE="configs/base.yaml"  # Change to use different config

# Check if horovodrun is available
if ! command -v horovodrun &> /dev/null; then
    echo "[ERROR] horovodrun not found. Please install Horovod first."
    echo "Install with: pip install horovod[pytorch]"
    exit 1
fi

# Print GPU information
echo "=========================================="
echo "Local Distributed Training"
echo "=========================================="
echo "Number of GPUs: $NUM_GPUS"
echo "Config file: $CONFIG_FILE"
echo "=========================================="
echo ""

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "Available GPUs:"
    nvidia-smi --list-gpus
    echo ""
else
    echo "[WARN] nvidia-smi not found. Running on CPU."
    echo ""
fi

# Setup environment (if needed)
export HF_HOME=${HF_HOME:-"$HOME/.cache/huggingface"}
export PYTHONUNBUFFERED=1

echo "Starting training with $NUM_GPUS GPUs..."
echo ""

# Run distributed training
# -np: number of processes (should match number of GPUs)
# -H localhost:N: run N processes on localhost
horovodrun -np $NUM_GPUS -H localhost:$NUM_GPUS \
    python -m src.train --config $CONFIG_FILE

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="

