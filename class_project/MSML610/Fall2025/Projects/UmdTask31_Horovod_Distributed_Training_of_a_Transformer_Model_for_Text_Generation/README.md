# Distributed Training of a Transformer Model for Text Generation on BookCorpus Dataset using Horovod

UMD MSML610 Course Project

## Project Overview

This project implements a **distributed training pipeline for a custom transformer-based language model** trained from scratch on the BookCorpus dataset using **Horovod** for multi-GPU training. The goal of the project is to design, implement, and validate a full end-to-end workflow for large-scale language model training in a high-performance computing environment.

Due to **GPU allocation limitations on the Zaratan HPC cluster**, full multi-GPU training runs could not be executed to completion during the project timeline. However, the **entire distributed training pipeline**, including data preprocessing, model architecture, Horovod integration, checkpointing, and logging, was fully implemented, validated, and tested via smoke tests and controlled runs.

The project demonstrates practical understanding of distributed systems, large-scale data pipelines, and transformer-based language modeling.

---

## Quick Start

### 1. Data Preprocessing

Run the preprocessing notebook to download and prepare the BookCorpus dataset:

```bash
jupyter notebook notebooks/00_data_preprocessing.ipynb
```

This notebook:

* Downloads the BookCorpus dataset
* Tokenizes text using the GPT-2 tokenizer
* Creates train and validation splits
* Packs tokens into fixed-length blocks
* Saves final preprocessed datasets and tokenizer to disk

Preprocessed data is saved to:

```
notebooks/data/preprocessed/v1/
```

This step needs to be run only once.

---

### 2. Model Training

Submit the training job to the Zaratan HPC cluster:

```bash
sbatch scripts/train_zaratan.sh configs/config.yaml
```

Note:
While the training script, Horovod setup, and distributed logic are fully implemented and validated, **GPU allocation queue delays on Zaratan prevented full multi-GPU training runs from completing** during the project period.

---

### 3. Text Generation

Generate text using a trained or partially trained model checkpoint:

```bash
python -m src.generate \
  --checkpoint checkpoints/best_model.pt \
  --config configs/config.yaml \
  --prompt "Once upon a time" \
  --max_new_tokens 100
```

---

## Project Structure

```
.
├── notebooks/
│   ├── 00_data_preprocessing.ipynb    # Data preprocessing notebook
│   ├── 00_data_preprocessing.py       # Preprocessing script
│   └── data/
│       └── preprocessed/              # Final preprocessed datasets
│
├── src/
│   ├── models/
│   │   └── transformer_lm.py          # Custom transformer model
│   ├── data.py                        # Data loading utilities
│   ├── train.py                       # Distributed training script
│   ├── generate.py                    # Text generation
│   ├── metrics.py                    # Evaluation metrics
│   └── utils/
│       ├── config.py                  # Configuration handling
│       ├── distributed.py             # Horovod utilities
│       ├── logging.py                 # Logging utilities
│       └── recorder.py                # Structured metrics recording
│
├── configs/
│   ├── config.yaml                    # Training configuration
│
├── scripts/
│   └── train_zaratan.sh               # HPC training script
│
├── checkpoints/                       # Model checkpoints
├── logs/                              # Training logs
└── runs/                              # TensorBoard logs
```

---

## Installation

### Prerequisites

* Python 3.9+
* CUDA 11.8+
* OpenMPI
* NCCL

### Dependencies

```bash
pip install -r requirements.txt
```

Install Horovod with GPU support:

```bash
HOROVOD_GPU_OPERATIONS=NCCL pip install horovod[pytorch]
```

---

## Workflow

### Step 1: Data Preprocessing

* BookCorpus dataset downloaded
* GPT-2 tokenizer applied
* 95 percent train, 5 percent validation split
* Fixed-length token blocks created
* Final datasets saved to disk

### Step 2: Model Training

* Loads preprocessed tokenized data
* Initializes custom GPT-style transformer
* Uses Horovod for multi-GPU training
* Implements checkpointing, mixed precision, and logging

Due to GPU unavailability, **full training epochs could not be completed**, but the training loop, Horovod synchronization, and fault tolerance mechanisms were fully implemented and validated.

### Step 3: Text Generation

* Generates text using trained checkpoints
* Supports temperature, top-k, and nucleus sampling
* Interactive and batch generation modes

---

## Model Architecture

The model is a **decoder-only transformer**, inspired by GPT-style architectures.

Key components:

* Causal self-attention
* Multi-head attention
* Pre-norm layer normalization
* GELU activations
* Weight tying between embeddings and output projection

Default configuration:

* Vocabulary size: 50,257
* Model dimension: 768
* Transformer layers: 12
* Attention heads: 12
* Feed-forward dimension: 3072
* Maximum sequence length: 512

---

## Monitoring and Logging

* TensorBoard logging to `runs/`
* Structured CSV metrics
* Periodic checkpoint saving
* Resume-from-checkpoint support

Logging and checkpoint mechanisms were validated during smoke tests and partial runs.

---

## Zaratan HPC Cluster Setup

* Account: msml610-class
* Partition: gpu
* GPU type: H100
* Configuration: multi-GPU single-node training

Despite correct configuration, **GPU allocation delays prevented multi-GPU execution**, which was communicated to and acknowledged by the course instructor.

---

## Troubleshooting

### Preprocessed Data Not Found

Run preprocessing first:

```bash
jupyter notebook notebooks/00_data_preprocessing.ipynb
```

### Insufficient GPUs

At least 2 GPUs are required for distributed training.

### CUDA Out of Memory

Reduce batch size or increase gradient accumulation.

---

## Key Features

* Custom transformer model from scratch
* End-to-end preprocessing pipeline
* Distributed training with Horovod
* Fault-tolerant checkpointing
* Structured logging and metrics
* Text generation utilities
* Modular, extensible design

All components required for large-scale distributed training were fully implemented and validated, even though full training execution was limited by hardware availability.

---

## License

This project was developed for educational purposes as part of the UMD MSML610 coursework.

---

## Acknowledgments

* UMD Division of IT for Zaratan HPC access
* HuggingFace for datasets and tokenizers
* Horovod development team
