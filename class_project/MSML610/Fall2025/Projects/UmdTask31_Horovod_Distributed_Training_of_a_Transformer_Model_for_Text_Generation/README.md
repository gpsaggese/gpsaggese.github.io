# Horovod Distributed Training of a Transformer Model for Text Generation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**UMD MSML610 Course Project** — Distributed training of transformer-based language models using Horovod for efficient multi-GPU training on the Zaratan HPC cluster.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Local Training](#local-training)
  - [Cluster Training (Zaratan)](#cluster-training-zaratan)
  - [Text Generation](#text-generation)
- [Configuration](#configuration)
- [Model Architectures](#model-architectures)
- [Results and Evaluation](#results-and-evaluation)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Overview

This project implements **distributed training of transformer-based language models** for text generation using:

- **Horovod**: Efficient distributed deep learning framework
- **PyTorch**: Deep learning framework
- **BookCorpus Open**: Large-scale text dataset from HuggingFace
- **Custom Transformer**: Decoder-only architecture (GPT-like)
- **Pretrained Models**: DistilGPT-2 and GPT-2 for comparison

The project is designed to run on UMD's **Zaratan HPC cluster** with NVIDIA A100 GPUs, but can also run locally on multi-GPU systems.

### Key Objectives

1. Implement a custom transformer-based language model
2. Enable distributed training across multiple GPUs using Horovod
3. Train on BookCorpus Open dataset
4. Compare custom model with pretrained models (DistilGPT-2, GPT-2)
5. Generate coherent text from trained models
6. Evaluate with perplexity, BLEU, and ROUGE metrics

---

## Features

- **Adaptive GPU Logic**: Automatically detects GPU count and runs smoke test if < 2 GPUs
- **Horovod Integration**: Efficient data-parallel training with gradient synchronization
- **Flexible Architecture**: Custom transformer or pretrained HuggingFace models
- **Comprehensive Logging**: TensorBoard, file logs, and console output (rank-aware)
- **Checkpointing**: Save/load model checkpoints with optimizer state
- **Text Generation**: Interactive and batch generation with temperature/top-k/top-p sampling
- **Metrics Tracking**: Loss, perplexity, accuracy, BLEU, ROUGE
- **Slurm Integration**: Ready-to-use job scripts for HPC clusters
- **Jupyter Notebooks**: Interactive exploration of data and models

---

## Project Structure

```
.
├── configs/                    # Configuration files
│   ├── base.yaml              # Base configuration
│   ├── custom_transformer.yaml # Custom model config
│   └── gpt2_small.yaml        # GPT-2 fine-tuning config
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── data.py                # Data loading and preprocessing
│   ├── train.py               # Main training script
│   ├── generate.py            # Text generation
│   ├── metrics.py             # Evaluation metrics
│   │
│   ├── models/                # Model implementations
│   │   ├── __init__.py
│   │   ├── transformer_lm.py  # Custom transformer (GPT-like)
│   │   └── hf_wrapper.py      # HuggingFace model wrapper
│   │
│   └── utils/                 # Utility modules
│       ├── __init__.py
│       ├── distributed.py     # Horovod helpers
│       ├── config.py          # Configuration management
│       └── logging.py         # Logging utilities
│
├── scripts/                    # Training scripts
│   ├── train_local.sh         # Local multi-GPU training
│   └── train_zaratan.sh       # Slurm job script for Zaratan
│
├── notebooks/                  # Jupyter notebooks
│   ├── 00_env_and_dataset.ipynb
│   └── 01_model_smoke_test.ipynb
│
├── checkpoints/               # Model checkpoints (created during training)
├── logs/                      # Training logs
├── runs/                      # TensorBoard logs
│
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU training)
- OpenMPI or another MPI implementation (for Horovod)
- NCCL (for multi-GPU training)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd UmdTask31_Horovod_Distributed_Training_of_a_Transformer_Model_for_Text_Generation
```

### Step 2: Install Dependencies

#### Option A: Using pip

```bash
pip install -r requirements.txt
```

#### Option B: Using conda (recommended)

```bash
conda create -n horovod_training python=3.9
conda activate horovod_training

# Install PyTorch with CUDA support
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install transformers datasets pyyaml tensorboard

# Install Horovod with GPU support
HOROVOD_GPU_OPERATIONS=NCCL pip install horovod[pytorch]
```

### Step 3: Verify Installation

```bash
# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Check Horovod
python -c "import horovod.torch as hvd; print(f'Horovod: {hvd.__version__}')"

# Check Transformers
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

---

## Quick Start

### 1. Explore the Data and Models (Jupyter Notebooks)

```bash
jupyter notebook notebooks/00_env_and_dataset.ipynb
```

### 2. Run Local Training (2+ GPUs required)

```bash
bash scripts/train_local.sh
```

### 3. Submit to Zaratan Cluster

```bash
sbatch scripts/train_zaratan.sh
```

### 4. Run Recording and Reports

The training script logs to TensorBoard and also writes a structured CSV + metadata package (rank 0) for easy grading and review.

- Structured runs live under `runs/structured/<timestamp>_run/`
- Files include: `metrics.csv`, `run_metadata.json`, and `config_used.yaml`
- Generate a static HTML report with plots:

```bash
python scripts/generate_report.py --run_dir runs/structured/<timestamp>_run
```

Open the generated `report.html` in a browser.

### 4. Generate Text

```bash
python -m src.generate \
  --checkpoint checkpoints/best_model.pt \
  --config configs/base.yaml \
  --model_type custom \
  --prompt "Once upon a time" \
  --max_new_tokens 100 \
  --interactive
```

---

## Usage

### Local Training

For local training with multiple GPUs:

```bash
# Edit scripts/train_local.sh to set NUM_GPUS
# Then run:
bash scripts/train_local.sh
```

Or run directly with horovodrun:

```bash
horovodrun -np 2 -H localhost:2 \
  python -m src.train --config configs/base.yaml
```

### Cluster Training (Zaratan)

1. **Edit the Slurm script** (`scripts/train_zaratan.sh`):
   - Set `--ntasks-per-node=4` and `--gres=gpu:h100:4` (or desired GPU count/type)
   - Set `--partition=gpu-h100` (or your cluster's H100 partition)
   - Uncomment and configure module loads
   - Set environment activation commands

2. **Submit the job**:
   ```bash
   sbatch scripts/train_zaratan.sh
   ```

3. **Monitor the job**:
   ```bash
   # Check job status
   squeue -u $USER
   
   # View output
   tail -f logs/horovod_transformer_h100-<JOB_ID>.out
   ```

### Text Generation

#### Single Generation

```bash
python -m src.generate \
  --checkpoint checkpoints/best_model.pt \
  --config configs/base.yaml \
  --model_type custom \
  --prompt "In a galaxy far, far away" \
  --max_new_tokens 150 \
  --temperature 0.8 \
  --top_k 50 \
  --top_p 0.9 \
  --num_samples 3
```

#### Interactive Mode

```bash
python -m src.generate \
  --checkpoint checkpoints/best_model.pt \
  --config configs/base.yaml \
  --model_type custom \
  --interactive
```

#### Using Pretrained Models (no checkpoint needed)

```bash
python -m src.generate \
  --model_type distilgpt2 \
  --prompt "Once upon a time" \
  --interactive
```

---

## Configuration

Configuration files are in YAML format. Key sections:

### Model Configuration

```yaml
model:
  type: "custom"  # or "gpt2", "distilgpt2"
  d_model: 512
  n_heads: 8
  n_layers: 6
  d_ff: 2048
  dropout: 0.1
  max_seq_len: 512
  vocab_size: 50257
```

### Training Configuration

```yaml
training:
  epochs: 3
  per_gpu_batch_size: 8
  learning_rate: 5.0e-4
  warmup_steps: 500
  weight_decay: 0.01
  max_grad_norm: 1.0
  log_interval: 100
  save_interval: 1000
```

### Data Configuration

```yaml
data:
  dataset_name: "lucadiliello/bookcorpusopen"
  max_samples: null  # null for all data
  validation_split: 0.05
  num_workers: 4
```

See example configs in `configs/` directory.

---

## Model Architectures

### 1. Custom TransformerLM (Decoder-only)

A GPT-like transformer with:
- Causal self-attention (autoregressive)
- Positional encoding (sinusoidal)
- Multi-head attention
- Feed-forward networks with GELU activation
- Layer normalization (pre-norm)
- Weight tying (embedding & output projection)

**Parameters**: ~15-150M (configurable)

### 2. DistilGPT-2

- 82M parameters
- 6 layers, 768 hidden size, 12 attention heads
- Distilled from GPT-2
- Fast and efficient

### 3. GPT-2 Small

- 117M parameters
- 12 layers, 768 hidden size, 12 attention heads
- Pretrained on WebText
- Strong baseline

---

## Results and Evaluation

### Metrics

- **Loss**: Cross-entropy loss
- **Perplexity**: exp(loss) — lower is better
- **Accuracy**: Token-level accuracy
- **BLEU**: N-gram overlap with reference text
- **ROUGE**: Recall-oriented metric for text generation

### Monitoring

**TensorBoard**:
```bash
tensorboard --logdir runs/
```

**Training Logs**:
```bash
tail -f logs/train_<timestamp>.log
```

---

## Troubleshooting

### Horovod Installation Issues

```bash
# Check MPI installation
which mpirun

# Check NCCL
ldconfig -p | grep nccl

# Reinstall Horovod with verbose output
HOROVOD_WITH_PYTORCH=1 HOROVOD_GPU_OPERATIONS=NCCL pip install --no-cache-dir horovod[pytorch]
```

### Single GPU Warning

If you see:
```
[WARN] Only 1 GPU detected — running smoke test (no full training).
```

This is expected behavior. The project requires **at least 2 GPUs** for distributed training. Allocate more GPUs with:
```bash
horovodrun -np 4 -H localhost:4 python -m src.train --config configs/base.yaml
```

### CUDA Out of Memory

Reduce batch size in config:
```yaml
training:
  per_gpu_batch_size: 4  # Reduce from 8
```

### Dataset Download Fails

Set HuggingFace cache directory:
```bash
export HF_HOME=/path/to/cache
```

Or use synthetic data (automatic fallback).

---

## References

- [Horovod Documentation](https://horovod.readthedocs.io/)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Transformer paper)
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2 paper)
- [BookCorpus Open Dataset](https://huggingface.co/datasets/lucadiliello/bookcorpusopen)

---

## Course Information

**Course**: UMD MSML610 - Advanced Machine Learning Systems  
**Project**: Distributed Training of Transformer Models  
**Cluster**: Zaratan HPC (UMD)  

---

## License

This project is for educational purposes as part of UMD MSML610 coursework.

---

## Contributing

This is a course project. For questions or issues, please contact the course instructor or TAs.

---

## Acknowledgments

- UMD Division of IT for providing access to Zaratan HPC cluster
- HuggingFace for datasets and transformers library
- Horovod team for distributed training framework
- Course instructors and TAs for guidance

---

## Contact

For questions about this project, please contact via the course discussion forum or email the course staff.

---

## Running on Zaratan (msml610)

This project is configured to run on the University of Maryland Zaratan HPC cluster under the **msml610** allocation.

### Cluster Configuration

- **Account**: `msml610`
- **Partition**: `gpu`
- **Max GPUs per job**: 4 (1 node)
- **Preferred GPU**: `H100` (80 GB VRAM)
- **Fallback GPU**: `A100` (40 GB VRAM)
- **CUDA Module**: `cuda/12.3`
- **User**: `vikranth`

### Directory Structure

- **Project Home** (scripts, configs):
  ```
  /afs/shell.umd.edu/project/msml610/user/vikranth/
  ```

- **Scratch** (datasets, checkpoints, cache):
  ```
  /scratch/zt1/project/msml610/vikranth/
  ```

### Environment Variables

The training script automatically sets:

```bash
export PROJECT_HOME=/afs/shell.umd.edu/project/msml610/user/vikranth
export SCRATCH_DIR=/scratch/zt1/project/msml610/vikranth
export HF_HOME=$SCRATCH_DIR/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export DATASET_CACHE=$HF_HOME
```

### Submitting Jobs

**Basic submission** (uses `configs/base.yaml`):
```bash
sbatch scripts/train_zaratan.sh
```

**With specific config**:
```bash
sbatch scripts/train_zaratan.sh configs/h100.yaml
sbatch scripts/train_zaratan.sh configs/custom_transformer.yaml
```

### Monitoring Jobs

**Check job status**:
```bash
squeue -u vikranth
```

**View output** (while running):
```bash
tail -f logs/horovod_transformer_h100-<JOB_ID>.out
```

**Cancel job**:
```bash
scancel <JOB_ID>
```

### GPU Selection

The script prefers H100 GPUs but can fall back to A100s:

```bash
# Default: H100 (edit scripts/train_zaratan.sh)
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:4
```

### Adaptive Behavior

Per project requirements, if Zaratan allocates only **1 GPU**, `src/train.py` automatically:
- Detects single-GPU mode
- Runs a smoke test
- Exits cleanly with status message

This ensures distributed training logic is only executed with 2+ GPUs.

### Module Setup

The script loads:
```bash
module purge
module load cuda/12.3
```

If using a Python virtual environment, uncomment the appropriate line in `scripts/train_zaratan.sh`:
```bash
# source ~/envs/msml610/bin/activate
```

---

**Happy Training!**
