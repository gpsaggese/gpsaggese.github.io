# Zaratan HPC Cluster Setup Guide

This guide provides specific instructions for setting up and running training on the Zaratan HPC cluster at UMD.

## Cluster Information

- **Cluster**: Zaratan (University of Maryland HPC)
- **Scheduler**: Slurm
- **Target Node**: `gpu-a6-7` (H100 GPUs)
- **GPU Type**: NVIDIA H100 (4 GPUs per node, 80GB VRAM each)
- **Partitions**: `gpu`, `gpu-h100`, `gpu-h100-hi`

## Prerequisites

1. **SSH Access**: `ssh vikranth@login.zaratan.umd.edu`
2. **Account**: `msml610` allocation
3. **Project Directory**: `/afs/shell.umd.edu/project/msml610/user/vikranth/`

## Step 1: Set Up Python Environment

The cluster uses a module system. You need to set up a Python environment with Horovod.

### Option A: Use Setup Script (Recommended)

```bash
# On Zaratan login node
cd /afs/shell.umd.edu/project/msml610/user/vikranth/UmdTask31_Horovod_Distributed_Training_of_a_Transformer_Model_for_Text_Generation

# Run setup script
bash scripts/setup_environment.sh
```

This will:
- Load required modules (CUDA, Python, PyTorch)
- Create a virtual environment
- Install Horovod and other dependencies

### Option B: Manual Setup

```bash
# Load modules
module purge
module load cuda/12.3.0
module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/zen2
module load pytorch/2.0.1/gcc/11.3.0/openmpi/4.1.5/cuda/12.3.0/zen2

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install Horovod
HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_PYTORCH=1 pip install horovod[pytorch]

# Install other dependencies
pip install transformers datasets pyyaml tensorboard numpy tqdm nltk rouge-score
```

## Step 2: Update Training Script

The training script has been updated to use the correct cluster modules. Key changes:

- **GPU specification**: `--gres=gpu:h100:4` (lowercase)
- **Optional node targeting**: `--nodelist=gpu-a6-7` (commented out by default)
- **Module loading**: Exact module versions as per cluster
- **Memory**: Increased to 64G for H100 workloads
- **Time limit**: 7 days for long training runs

## Step 3: Activate Environment in Training Script

If you created a virtual environment, uncomment the activation line in `scripts/train_zaratan.sh`:

```bash
# Find this section in train_zaratan.sh (around line 114-116)
# Uncomment and adjust path:
source venv/bin/activate
# Or if in project directory:
source $PROJECT_HOME/UmdTask31_Horovod_Distributed_Training_of_a_Transformer_Model_for_Text_Generation/venv/bin/activate
```

## Step 4: Verify Setup

```bash
# Load modules
module purge
module load cuda/12.3.0
module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/zen2
module load pytorch/2.0.1/gcc/11.3.0/openmpi/4.1.5/cuda/12.3.0/zen2

# Activate environment (if created)
source venv/bin/activate

# Verify
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import horovod.torch as hvd; print(f'Horovod: {hvd.__version__}')"
```

## Step 5: Submit Training Job

```bash
# Navigate to project directory
cd /afs/shell.umd.edu/project/msml610/user/vikranth/UmdTask31_Horovod_Distributed_Training_of_a_Transformer_Model_for_Text_Generation

# Submit job
sbatch scripts/train_zaratan.sh configs/base.yaml

# Or use H100 optimized config
sbatch scripts/train_zaratan.sh configs/h100.yaml
```

## Step 6: Monitor Training

```bash
# Check job status
squeue -u vikranth

# Watch output (replace <JOB_ID>)
tail -f logs/horovod_transformer_h100-<JOB_ID>.out

# Check node assignment
scontrol show job <JOB_ID>
```

## Available Modules

Check available modules:
```bash
module avail
module avail cuda
module avail python
module avail pytorch
```

## Node Information

Check H100 node status:
```bash
# Check partition status
sinfo -p gpu

# Check specific node
scontrol show node gpu-a6-7

# Check GPU availability
sinfo -p gpu -o "%N %G %t"
```

## Interactive Job (for Testing)

If you need to test interactively:
```bash
srun --pty --gres=gpu:h100:4 --nodelist=gpu-a6-7 --time=1:00:00 bash

# Then load modules and test
module load cuda/12.3.0
module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/zen2
module load pytorch/2.0.1/gcc/11.3.0/openmpi/4.1.5/cuda/12.3.0/zen2
source venv/bin/activate
python -c "import torch; print(torch.cuda.is_available())"
```

## Troubleshooting

### Issue: "Horovod not found"
- Solution: Install Horovod in your virtual environment (see Step 1)

### Issue: "Module not found"
- Solution: Check exact module names with `module avail <name>`

### Issue: "CUDA not available"
- Solution: Make sure you loaded `cuda/12.3.0` module before PyTorch

### Issue: Job pending too long
- Solution: Try removing `--nodelist` to allow any available H100 node
- Or try different partitions: `gpu-h100` or `gpu-h100-hi`

### Issue: Out of memory
- Solution: Reduce batch size in config file
- Or increase memory: `#SBATCH --mem=128G`

## Quick Reference

```bash
# Submit job
sbatch scripts/train_zaratan.sh configs/base.yaml

# Check status
squeue -u vikranth

# Monitor output
tail -f logs/horovod_transformer_h100-<JOB_ID>.out

# Cancel job
scancel <JOB_ID>

# Check modules
module list
module avail

# Interactive session
srun --pty --gres=gpu:h100:4 --time=1:00:00 bash
```

## Notes

- **No global conda/pip**: Use cluster modules or your own venv
- **Module system required**: All main software (CUDA, PyTorch, Python) must come from modules
- **Horovod**: Usually needs to be installed in your environment (not available as module)
- **Node targeting**: `--nodelist=gpu-a6-7` is optional - comment out if node is busy
