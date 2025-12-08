# Training a Custom Transformer Language Model from Scratch

**UMD MSML610 Course Project**

This project implements distributed training of a custom transformer-based language model from scratch on the BookCorpus dataset using Horovod for multi-GPU training on the Zaratan HPC cluster.

## Project Overview

This project trains a custom decoder-only transformer architecture (GPT-like) language model from scratch. The implementation includes data preprocessing, distributed training using Horovod, and text generation capabilities. The model is designed to be trained on multiple GPUs for efficient training of large language models.

## Quick Start

### 1. Data Preprocessing

Run the preprocessing notebook to download and prepare the BookCorpus dataset:

```bash
jupyter notebook notebooks/00_data_preprocessing.ipynb
```

This notebook downloads the BookCorpus dataset, tokenizes it using GPT-2 tokenizer, creates train/validation splits, and saves preprocessed data to `notebooks/data/preprocessed/v1/`.

### 2. Model Training

Submit the training job to the Zaratan HPC cluster:

```bash
sbatch scripts/train_zaratan.sh
```

### 3. Text Generation

Generate text using a trained model:

```bash
python -m src.generate \
  --checkpoint checkpoints/best_model.pt \
  --config configs/config.yaml \
  --prompt "Once upon a time" \
  --max_new_tokens 100
```

## Project Structure

```
.
├── notebooks/
│   ├── 00_data_preprocessing.ipynb    # Data preprocessing notebook
│   ├── 00_data_preprocessing.py       # Preprocessing script
│   └── data/
│       └── preprocessed/              # Preprocessed data (generated)
│
├── src/
│   ├── models/
│   │   └── transformer_lm.py          # Custom transformer model implementation
│   ├── data.py                        # Data loading utilities
│   ├── train.py                       # Distributed training script
│   ├── generate.py                    # Text generation script
│   ├── metrics.py                     # Evaluation metrics
│   └── utils/                         # Utility modules
│       ├── config.py                  # Configuration management
│       ├── distributed.py             # Distributed training utilities
│       ├── logging.py                 # Logging configuration
│       └── recorder.py                # Metrics recording
│
├── configs/
│   ├── config.yaml                    # Main configuration file
│   └── preprocess_zaratan.sh          # Preprocessing job script
│
├── scripts/
│   └── train_zaratan.sh               # Training job script for Zaratan HPC
│
├── checkpoints/                       # Model checkpoints (generated during training)
├── logs/                              # Training logs (generated during training)
└── runs/                              # TensorBoard logs (generated during training)
```

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA 11.8 or higher (for GPU training)
- OpenMPI (for Horovod)
- NCCL (for multi-GPU communication)

### Dependencies

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install Horovod with GPU support:

```bash
HOROVOD_GPU_OPERATIONS=NCCL pip install horovod[pytorch]
```

## Workflow

### Step 1: Data Preprocessing

The preprocessing step downloads and prepares the BookCorpus dataset:

1. Downloads BookCorpus dataset (approximately 4GB)
2. Tokenizes text using GPT-2 tokenizer
3. Creates train/validation splits (95%/5%)
4. Packs tokens into fixed-length blocks to reduce padding
5. Saves preprocessed data to `notebooks/data/preprocessed/v1/`

Note: This step needs to be run only once. The preprocessed data can be reused for multiple training runs.

### Step 2: Model Training

The training process:

- Loads preprocessed data from `notebooks/data/preprocessed/v1/`
- Initializes the custom transformer model
- Trains using Horovod for distributed multi-GPU training
- Saves model checkpoints to `checkpoints/`
- Logs training metrics to TensorBoard in `runs/`

Training can be performed locally or on the Zaratan HPC cluster using the provided scripts.

### Step 3: Text Generation

After training, use the generation script to produce text:

```bash
python -m src.generate \
  --checkpoint checkpoints/best_model.pt \
  --config configs/config.yaml \
  --prompt "Your prompt text here" \
  --max_new_tokens 150 \
  --temperature 0.8 \
  --interactive
```

## Configuration

All training parameters are configured in `configs/config.yaml`. Key configuration sections include:

- **Model Architecture**: Layer dimensions, number of layers, attention heads, etc.
- **Training Parameters**: Number of epochs, batch size, learning rate, optimizer settings
- **Data Settings**: Path to preprocessed data directory
- **Hardware Settings**: Mixed precision training, gradient accumulation
- 

## Model Architecture

The model implements a decoder-only transformer architecture similar to GPT:

**Architecture Components:**
- Causal self-attention mechanism (autoregressive)
- Sinusoidal positional encoding
- Multi-head attention with 12 heads
- Feed-forward networks with GELU activation
- Pre-norm layer normalization
- Weight tying between embedding and output projection layers

**Default Model Configuration:**
- Total parameters: approximately 150 million
- Model dimension (d_model): 768
- Number of transformer layers: 12
- Number of attention heads: 12
- Feed-forward dimension (d_ff): 3072
- Maximum sequence length: 512 tokens
- Vocabulary size: 50,257 (GPT-2 tokenizer)

## Monitoring and Logging

### TensorBoard

View training metrics in real-time:

```bash
tensorboard --logdir runs/
```

### Training Logs

Monitor training progress:

```bash
tail -f logs/train_*.log
```

### Structured Metrics

Training metrics are automatically saved to `runs/structured/<timestamp>_run/`:

- `metrics.csv`: Training and validation metrics per epoch
- `run_metadata.json`: Metadata about the training run
- `config_used.yaml`: Configuration file used for the run

## Zaratan HPC Cluster Setup

### Cluster Configuration

- Account: `msml610-class`
- Partition: `gpu` (H100 GPUs)
- GPU Configuration: 4x H100 (80GB VRAM each)

### Environment Setup

The training script (`scripts/train_zaratan.sh`) handles:

- Loading required modules (CUDA, Python, OpenMPI, NCCL)
- Activating the virtual environment
- Setting up environment variables
- Configuring Horovod for single-node multi-GPU training

### Job Submission

Submit a training job:

```bash
sbatch scripts/train_zaratan.sh
```

### Job Monitoring

Check job status:

```bash
squeue -u <username>
```

View job output:

```bash
tail -f logs/horovod_transformer_h100-<JOB_ID>.out
```

## Troubleshooting

### Preprocessed Data Not Found

**Issue**: Training script cannot find preprocessed data.

**Solution**: Run the data preprocessing notebook first:
```bash
jupyter notebook notebooks/00_data_preprocessing.ipynb
```

### Insufficient GPUs

**Issue**: Only 1 GPU detected or insufficient GPUs for distributed training.

**Solution**: This project requires 2 or more GPUs for distributed training. For local testing:
```bash
horovodrun -np 4 -H localhost:4 python -m src.train --config configs/config.yaml
```

### CUDA Out of Memory

**Issue**: GPU runs out of memory during training.

**Solution**: Reduce batch size in `configs/config.yaml`:
```yaml
training:
  per_gpu_batch_size: 8  # Reduce from default value
```

Alternatively, enable gradient accumulation to maintain effective batch size.

### Horovod Installation Issues

**Issue**: Horovod not found or installation fails.

**Solution**: Install Horovod with proper GPU support:
```bash
HOROVOD_GPU_OPERATIONS=NCCL pip install horovod[pytorch]
```

Ensure CUDA and NCCL are properly installed before installing Horovod.

## Key Features

- **Custom Model Implementation**: Complete transformer architecture implemented from scratch
- **Data Preprocessing**: Automated preprocessing pipeline for BookCorpus dataset
- **Distributed Training**: Multi-GPU training using Horovod framework
- **H100 Optimization**: BF16 mixed precision and optimized batch sizes for H100 GPUs
- **Comprehensive Logging**: TensorBoard integration and structured CSV metrics
- **Text Generation**: Interactive and batch text generation capabilities
- **Modular Design**: Clean separation of data, models, training, and utilities

## License

This project is developed for educational purposes as part of the UMD MSML610 coursework.

## Acknowledgments

- UMD Division of IT for providing access to the Zaratan HPC cluster
- HuggingFace for datasets and tokenizer libraries
- Uber Horovod team for the distributed training framework
