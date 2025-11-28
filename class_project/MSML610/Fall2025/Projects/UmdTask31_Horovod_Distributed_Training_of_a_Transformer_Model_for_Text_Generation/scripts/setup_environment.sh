#!/bin/bash
###############################################################################
# Environment Setup Script for Zaratan HPC
#
# This script helps set up a Python virtual environment with required packages
# for Horovod distributed training.
#
# Usage:
#   bash scripts/setup_environment.sh
#
# This creates a virtual environment in the project directory with:
# - PyTorch (from module)
# - Horovod
# - Transformers
# - Other dependencies
###############################################################################

set -e  # Exit on error

echo "=========================================="
echo "Setting up Python Environment for Zaratan"
echo "=========================================="
echo ""

# Load required modules
echo "Loading cluster modules..."
module purge
module load gcc/11.3.0
module load cuda/12.3.0
module load python/3.10.10/gcc/11.3.0
module load openmpi/4.1.5/gcc/11.3.0/zen2
module load nccl/2.18.1-1/gcc/11.3.0/zen2
module load pytorch/2.0.1/gcc/11.3.0/openmpi/4.1.5/cuda/12.3.0/zen2

echo "Python version:"
python --version
which python
echo ""

# Create virtual environment in project directory
VENV_DIR="venv"
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    read -p "Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
    else
        echo "Using existing virtual environment."
        source "$VENV_DIR/bin/activate"
        exit 0
    fi
fi

echo "Creating virtual environment..."
python -m venv "$VENV_DIR"

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing required packages..."
echo "Note: PyTorch should already be available from the module system"
echo ""

# Install packages that aren't in modules
echo "Installing build helpers (pybind11) for Horovod..."
pip install --no-cache-dir pybind11

echo "Installing Horovod (this may take a few minutes)..."
# Ensure compiler and include paths are set for the build
export CC=$(which gcc)
export CXX=$(which g++)
export CMAKE_C_COMPILER=$CC
export CMAKE_CXX_COMPILER=$CXX
export HOROVOD_NCCL_HOME=/cvmfs/hpcsw.umd.edu/spack-software/2023.11.20/linux-rhel8-zen2/gcc-11.3.0/nccl-2.18.1-1-iz663sfeinfvn4trcoxsrb2sch5lyev4
export HOROVOD_NCCL_LINK=SHARED
export HOROVOD_WITHOUT_GLOO=1
export HOROVOD_WITH_MPI=1
export MPICC=mpicc
export MPICXX=mpic++
export CPATH=$HOROVOD_NCCL_HOME/include:$(python -c "import pybind11, pathlib; print(pathlib.Path(pybind11.__file__).resolve().parent/'include')"):$CPATH
export LD_LIBRARY_PATH=$HOROVOD_NCCL_HOME/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=$(python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).resolve().parents[2] / 'share' / 'cmake')")

PIP_NO_BUILD_ISOLATION=1 \
HOROVOD_GPU_OPERATIONS=NCCL \
HOROVOD_WITH_PYTORCH=1 \
CC=$CC CXX=$CXX CMAKE_C_COMPILER=$CC CMAKE_CXX_COMPILER=$CXX \
pip install --no-cache-dir --no-build-isolation horovod[pytorch]

echo "Installing other dependencies..."
pip install transformers datasets pyyaml tensorboard numpy tqdm pandas matplotlib

# Optional: Install metrics packages
pip install nltk rouge-score
pip install psutil cloudpickle

echo ""
echo "=========================================="
echo "Environment setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To use in training script, add to train_zaratan.sh:"
echo "  source venv/bin/activate"
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import horovod.torch as hvd; print(f'Horovod: {hvd.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
echo ""
echo "All packages installed successfully!"
