#!/bin/bash
###############################################################################
# Slurm Job Script for Zaratan HPC Cluster (UMD) - H100 Optimized
#
# Account: hpcintro-c+
# Partition: gpu-h100 (H100 nodes)
# GPUs: H100 (80GB VRAM) - 4 GPUs per node, single node only
#
# This script is optimized for:
# - H100 GPU nodes with 4 GPUs per node
# - Single-node distributed training (no cross-node communication)
# - Automatic H100 node detection and allocation
# - Full utilization of H100 features (Tensor Cores, high bandwidth)
#
# Usage:
#   sbatch scripts/train_zaratan.sh [config_file]
#   sbatch scripts/train_zaratan.sh configs/base.yaml
#
# Monitor:
#   squeue -u vikranth
#   scontrol show job <JOB_ID>
#
# View output:
#   tail -f logs/horovod_transformer_h100-<JOB_ID>.out
#
# Check GPU utilization:
#   ssh <NODE_NAME> nvidia-smi
###############################################################################

#SBATCH --job-name=horovod_transformer_h100
#SBATCH --account=hpcintro-+
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1                       # Single node only - all 4 GPUs from same node
#SBATCH --ntasks-per-node=4             # 4 processes (one per GPU)
#SBATCH --gres=gpu:h100:4               # Request 4x H100 (80GB each)
#SBATCH --cpus-per-task=8               # 8 CPUs per task (32 total for 4 GPUs)
#SBATCH --mem=64G                       # Memory per node
#SBATCH --time=12:00:00                 # Maximum runtime
#SBATCH --output=logs/%x-%j.out         # Output file (%x=job-name, %j=job-id)
# #SBATCH --exclusive                   # Optional: request exclusive node access

# Optional: Target specific H100 node (uncomment and modify as needed)
# Check available H100 nodes with: sinfo -p gpu-h100 -o "%N %G %t"
# #SBATCH --nodelist=gpu-a6-3      # Example: target gpu-a6-3 (modify based on availability)
# #SBATCH --nodelist=gpu-a6-7      # Alternative: gpu-a6-7

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

###############################################################################
# Job Information and Node/GPU Allocation
###############################################################################
echo "=========================================="
echo "Job Information and GPU Allocation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node(s): $SLURM_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Total CPUs: $SLURM_CPUS_ON_NODE"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "Working directory: $(pwd)"
echo ""

# Verify single-node allocation
if [ "$SLURM_JOB_NUM_NODES" -ne 1 ]; then
    echo "[ERROR] This script requires exactly 1 node. Got $SLURM_JOB_NUM_NODES nodes."
    echo "[ERROR] Aborting to prevent cross-node communication issues."
    exit 1
fi

# Print GPU allocation details
echo "GPU Allocation:"
if command -v nvidia-smi &> /dev/null; then
    echo "Available GPUs on node:"
    nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "Total GPUs detected: $GPU_COUNT"
    
    if [ "$GPU_COUNT" -ne 4 ]; then
        echo "[WARNING] Expected 4 GPUs but found $GPU_COUNT"
        echo "[WARNING] Continuing anyway, but performance may be suboptimal"
    fi
    
    # Check GPU model
    GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "GPU Model: $GPU_MODEL"
    
    if [[ ! "$GPU_MODEL" =~ "H100" ]]; then
        echo "[WARNING] Expected H100 GPU but detected: $GPU_MODEL"
        echo "[WARNING] Script will continue, but optimizations are for H100"
    else
        echo "[OK] H100 GPU detected - optimizations enabled"
    fi
else
    echo "[WARNING] nvidia-smi not available - cannot verify GPU allocation"
fi

echo "=========================================="
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

###############################################################################
# Module Loading (Zaratan-specific)
# Based on successful Horovod installation setup
###############################################################################
echo "Loading modules..."

# Purge existing modules to avoid conflicts
module purge

# Load modules in the correct order (matching successful Horovod build)
echo "Loading GCC 11.3.0..."
module load gcc/11.3.0

echo "Loading CUDA 12.3.0..."
module load cuda/12.3.0

# Verify CUDA version matches H100 requirements
CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
echo "CUDA Version: $CUDA_VERSION"

# Check if CUDA version is >= 11.8 (required for H100)
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)

if [ "$CUDA_MAJOR" -lt 11 ]; then
    echo "[ERROR] CUDA 11.8+ required for H100. Found: $CUDA_VERSION"
    exit 1
elif [ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -lt 8 ]; then
    echo "[ERROR] CUDA 11.8+ required for H100. Found: $CUDA_VERSION"
    exit 1
fi

echo "[OK] CUDA version compatible with H100 (requires 11.8+, found $CUDA_VERSION)"
echo ""

# Load Python 3.10.10 (base, not CUDA-specific - PyTorch module handles CUDA)
echo "Loading Python 3.10.10..."
module load python/3.10.10/gcc/11.3.0

# Load OpenMPI with CUDA support (required for Horovod)
# Use zen2 for GPU compute nodes (icelake is for CPU-only nodes)
echo "Loading OpenMPI 4.1.5 with CUDA support..."
module load openmpi/4.1.5/gcc/11.3.0/cuda/12.3.0/zen2

# Load NCCL for multi-GPU communication
echo "Loading NCCL 2.18.1-1..."
module load nccl/2.18.1-1/gcc/11.3.0/zen2

# Load PyTorch 2.0.1 with CUDA 12.3 and OpenMPI (for Horovod)
echo "Loading PyTorch 2.0.1 with CUDA 12.3..."
module load pytorch/2.0.1/gcc/11.3.0/openmpi/4.1.5/cuda/12.3.0/zen2

# Load CMake (may be needed for some operations)
echo "Loading CMake 3.26.3..."
module load cmake/3.26.3/gcc/11.3.0/zen2 2>/dev/null || echo "[INFO] CMake module not found, continuing..."

echo "Loaded modules:"
module list
echo ""

###############################################################################
# Setup Python Environment
###############################################################################
echo "Setting up Python environment..."

# Activate virtual environment with Horovod installed
# Based on successful setup: Horovod is installed in ~/envs/horovod-py310
VENV_PATH="$HOME/envs/horovod-py310"

if [ -f "$VENV_PATH/bin/activate" ]; then
    echo "Activating virtual environment: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
    echo "[OK] Virtual environment activated"
elif [ -f "venv/bin/activate" ]; then
    echo "Activating project virtual environment: venv"
    source venv/bin/activate
    echo "[OK] Project virtual environment activated"
else
    echo "[WARNING] No virtual environment found at $VENV_PATH or ./venv"
    echo "[WARNING] Continuing with system Python (Horovod may not be available)"
fi

echo "Python version:"
python --version
which python
echo ""

echo "PyTorch and CUDA Information:"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version (PyTorch): {torch.version.cuda}')
    print(f'cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else \"N/A\"}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'    Compute Capability: {torch.cuda.get_device_capability(i)}')
        print(f'    Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB')
    # Check for H100-specific features
    if torch.cuda.device_count() > 0:
        props = torch.cuda.get_device_properties(0)
        if props.major >= 9:  # H100 is compute capability 9.0
            print('[OK] GPU supports H100 features (Compute Capability 9.0+)')
        else:
            print(f'[WARNING] GPU compute capability: {props.major}.{props.minor} (H100 is 9.0)')
" || echo "[ERROR] PyTorch not found or CUDA not available"
echo ""

echo "Horovod version:"
python -c "import horovod.torch as hvd; print(f'Horovod: {hvd.__version__}'); print(f'NCCL built: {hvd.nccl_built()}'); print(f'MPI built: {hvd.mpi_built()}')" 2>&1 || {
    echo "[ERROR] Horovod not found!"
    echo "[ERROR] Please install Horovod following the setup guide:"
    echo "[ERROR] 1. Allocate interactive node: srun --pty --partition=gpu --gres=gpu:h100:4 --ntasks=4 --cpus-per-task=4 --time=00:30:00 /bin/bash"
    echo "[ERROR] 2. Load modules and install Horovod from source (see setup documentation)"
    exit 1
}
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
mkdir -p $HF_HOME 2>/dev/null || true

# Ensure we're in the project directory
cd "$PROJECT_HOME/UmdTask31_Horovod_Distributed_Training_of_a_Transformer_Model_for_Text_Generation" || {
    echo "[ERROR] Cannot find project directory. Please check PROJECT_HOME path."
    exit 1
}

# PyTorch environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

# NCCL settings (optimized for H100 single-node, 4-GPU configuration)
export NCCL_DEBUG=WARN                    # Set to INFO for debugging, WARN for production
export NCCL_IB_DISABLE=0                  # Enable InfiniBand if available (for future multi-node)
export NCCL_SOCKET_IFNAME=^docker0,lo     # Exclude docker and loopback interfaces
export NCCL_P2P_DISABLE=0                 # Enable P2P for same-node GPU communication
export NCCL_SHM_DISABLE=0                 # Enable shared memory (faster for same-node)
export NCCL_NET_GDR_LEVEL=2               # GPU Direct RDMA level (for H100)

# Horovod settings (optimized for H100)
export HOROVOD_GPU_OPERATIONS=NCCL
export HOROVOD_GPU_ALLREDUCE=NCCL
export HOROVOD_GPU_ALLGATHER=NCCL
export HOROVOD_GPU_BROADCAST=NCCL
export HOROVOD_TIMELINE=$SCRATCH_DIR/timeline.json  # Performance profiling (optional)

# PyTorch optimizations for H100 (Hopper architecture)
export TORCH_CUDNN_V8_API_ENABLED=1       # Enable cuDNN v8 API (H100 optimized)
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 # Allow TF32 for cuBLAS (H100)

# CUDA optimizations for H100
export CUDA_LAUNCH_BLOCKING=0             # Set to 1 for debugging, 0 for performance
export CUDA_DEVICE_ORDER=PCI_BUS_ID       # Consistent GPU ordering

echo "Environment configured:"
echo "  PROJECT_HOME: $PROJECT_HOME"
echo "  SCRATCH_DIR: $SCRATCH_DIR"
echo "  HF_HOME: $HF_HOME"
echo ""

###############################################################################
# Print Detailed GPU Information
###############################################################################
echo "=========================================="
echo "Detailed GPU Information"
echo "=========================================="
nvidia-smi
echo ""

# Print GPU topology (important for multi-GPU performance)
echo "GPU Topology (NCCL):"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi topo -m 2>/dev/null || echo "[INFO] nvidia-smi topo not available"
fi
echo ""

# Verify all 4 GPUs are on the same node
echo "Verifying GPU allocation..."
if [ -n "$SLURM_STEP_GPUS" ] || [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
    echo "SLURM_STEP_GPUS: ${SLURM_STEP_GPUS:-not set}"
fi
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

# Determine number of processes
NP=${SLURM_NTASKS:-0}

# Fallback to SLURM_GPUS_PER_NODE if SLURM_NTASKS is not set
if [ "$NP" -le 0 ] && [ -n "$SLURM_GPUS_PER_NODE" ]; then
    NP=$SLURM_GPUS_PER_NODE
fi

# Fallback to SLURM_GPUS_ON_NODE if still not set
if [ "$NP" -le 0 ] && [ -n "$SLURM_GPUS_ON_NODE" ]; then
    NP=$SLURM_GPUS_ON_NODE
fi

# Fallback: count GPUs directly
if [ "$NP" -le 0 ]; then
    if command -v nvidia-smi &> /dev/null; then
        NP=$(nvidia-smi --list-gpus | wc -l)
        echo "[INFO] Inferred $NP processes from nvidia-smi"
    fi
fi

# Option to require exactly 4 GPUs (default), or allow any NP>=2 if ALLOW_ANY_NP=1
REQUIRE_FOUR=${ALLOW_ANY_NP:-0}
if [ "$REQUIRE_FOUR" -eq 0 ]; then
    if [ "$NP" -ne 4 ]; then
        echo "[ERROR] This script requires exactly 4 processes (one per GPU)."
        echo "[ERROR] Detected: $NP processes"
        echo "[ERROR] Please ensure --ntasks-per-node=4 and --gres=gpu:h100:4 or set ALLOW_ANY_NP=1"
        exit 1
    fi
    echo "[OK] Confirmed 4 processes for 4-GPU single-node training"
else
    if [ "$NP" -lt 2 ]; then
        echo "[ERROR] Need at least 2 processes for distributed training (ALLOW_ANY_NP=1)"
        exit 1
    fi
    echo "[OK] Allowing $NP processes (ALLOW_ANY_NP=1)"
fi

echo "Launching Horovod training:"
echo "  Processes: $NP"
echo "  Config: $CONFIG_FILE"
echo "  HF Cache: $HF_HOME"
echo ""

# Check if horovodrun is available
if ! command -v horovodrun &> /dev/null; then
    echo "[ERROR] horovodrun not found. Please install Horovod first."
    echo "Install with: HOROVOD_GPU_OPERATIONS=NCCL pip install horovod[pytorch]"
    exit 1
fi

# Run with horovodrun (single-node, 4-GPU configuration)
# -np: number of processes (must be 4)
# -H: host specification (localhost:4 means 4 processes on localhost)
# This ensures all processes run on the same node
echo "Launching Horovod with single-node configuration:"
echo "  Host: localhost:4 (all 4 GPUs on same node)"
echo "  Processes: $NP"
echo "  Config: $CONFIG_FILE"
echo ""

# Use localhost binding to ensure single-node execution
RUN_NAME_ARG=""
if [ -n "$SLURM_JOB_ID" ]; then
  RUN_NAME_ARG="--run_name job$SLURM_JOB_ID"
fi

horovodrun \
    -np $NP \
    -H localhost:$NP \
    python -m src.train --config "$CONFIG_FILE" $RUN_NAME_ARG

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

###############################################################################
# Post-Training Summary and Monitoring Tips
###############################################################################
echo ""
echo "=========================================="
echo "Training Summary and Monitoring Tips"
echo "=========================================="
echo ""
echo "MONITORING GPU UTILIZATION:"
echo "  To monitor GPU usage during training, SSH to the compute node:"
echo "    ssh $SLURM_NODELIST"
echo "    watch -n 1 nvidia-smi"
echo ""
echo "  Or use:"
echo "    nvidia-smi dmon -s u -c 100  # Monitor utilization for 100 seconds"
echo ""
echo "TROUBLESHOOTING COMMON H100 ISSUES:"
echo ""
echo "  1. Low GPU Utilization (< 80%):"
echo "     - Check if data loading is the bottleneck (increase num_workers)"
echo "     - Verify batch size is large enough (H100 can handle large batches)"
echo "     - Check for CPU-GPU transfer bottlenecks (use pin_memory=True)"
echo "     - Enable mixed precision (fp16: true in config) for H100"
echo ""
echo "  2. Out of Memory (OOM) Errors:"
echo "     - Reduce per_gpu_batch_size in config"
echo "     - Increase gradient_accumulation_steps"
echo "     - Enable gradient checkpointing (if implemented)"
echo ""
echo "  3. NCCL Communication Errors:"
echo "     - Set NCCL_DEBUG=INFO to see detailed logs"
echo "     - Verify all GPUs are on same node (should be with this script)"
echo "     - Check NCCL version compatibility"
echo ""
echo "  4. Slow Training:"
echo "     - Verify TF32 is enabled (automatic on H100 with PyTorch 2.0+)"
echo "     - Check if using fused optimizers (AdamW with fused=True)"
echo "     - Ensure data is preprocessed and cached"
echo "     - Use token packing (pack_to_max_length: true) to reduce padding"
echo ""
echo "  5. Job Pending Too Long:"
echo "     - Check queue: squeue -u vikranth"
echo "     - Try removing --nodelist to allow any H100 node"
echo "     - Check partition limits: sinfo -p gpu"
echo ""
echo "PERFORMANCE OPTIMIZATION FOR H100:"
echo "  - Enable fp16 in config for 2x speedup (fp16: true)"
echo "  - Use larger batch sizes (H100 has 80GB VRAM)"
echo "  - Enable token packing to reduce padding overhead"
echo "  - Use fused AdamW optimizer (automatic if available)"
echo "  - TF32 is automatically enabled for H100"
echo ""
echo "CHECKPOINTS AND LOGS:"
echo "  - Checkpoints: checkpoints/"
echo "  - Training logs: logs/train_rank0.log"
echo "  - TensorBoard: runs/"
echo "  - Job output: logs/horovod_transformer_h100-${SLURM_JOB_ID}.out"
echo ""
echo "To view TensorBoard after training:"
echo "  tensorboard --logdir runs/ --port 6006"
echo ""

exit $EXIT_CODE
