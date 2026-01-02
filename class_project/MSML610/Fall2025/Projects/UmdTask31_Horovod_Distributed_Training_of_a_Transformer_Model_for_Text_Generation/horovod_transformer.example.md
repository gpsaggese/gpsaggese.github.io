<!-- toc -->
- [Project description](#project-description)
  * [Table of Contents](#table-of-contents)
  + [Hierarchy](#hierarchy)
  * [General Guidelines](#general-guidelines)
  * [Project Overview](#project-overview)
  * [Architecture](#architecture)
  * [Data Pipeline](#data-pipeline)
  * [Training Workflow](#training-workflow)
  * [Usage Examples](#usage-examples)
  * [Results and Evaluation](#results-and-evaluation)
<!-- tocstop -->

# Project description

Distributed training of a custom transformer-based language model on the BookCorpus dataset using Horovod for multi-GPU training. This project implements a complete end-to-end pipeline from data preprocessing to model training and text generation, demonstrating practical distributed deep learning on HPC infrastructure.

## Table of Contents

The markdown code can have a TOC. This can be generated automatically with the linter or other tools.

### Hierarchy

Hierarchy of the markdown file should be followed.

# Level 1 (Used as title)
All the subheadings should follow the below structure:
## Level 2
### Level 3

**Note** Level 1 Heading (Title) should be `Project description`.

## General Guidelines

- Follow the instructions in [README](/DATA605/DATA605_Spring2025/README.md) on what to write in the project example.
- Include descriptions of how you used the API in your projects with descriptions on architecture, data, etc used in `horovod_transformer.example.py`.
- The file should be named with the following conventions:
  - Since this project uses the Horovod distributed transformer API, it is named `horovod_transformer.example.md`

## Project Overview

This project implements a distributed training system for transformer-based language models using Horovod. The system is designed to train custom GPT-style transformer models from scratch on the BookCorpus dataset, leveraging multi-GPU parallelism for efficient large-scale training.

### Key Features

- **Custom Transformer Architecture**: Decoder-only transformer with configurable depth and width
- **Distributed Training**: Horovod-based multi-GPU training with efficient gradient synchronization
- **End-to-End Pipeline**: Complete workflow from data preprocessing to text generation
- **HPC Optimized**: Designed for H100 GPUs on Zaratan HPC cluster
- **Production Ready**: Checkpointing, logging, metrics tracking, and fault tolerance

### Project Goals

1. Implement a custom transformer language model from scratch
2. Set up distributed training infrastructure using Horovod
3. Train the model on BookCorpus dataset
4. Generate coherent text from trained models
5. Demonstrate practical distributed deep learning workflows

## Architecture

### Model Architecture

The project uses a custom decoder-only transformer architecture (`TransformerLM`) inspired by GPT models:

**Key Components:**

- **Token Embeddings**: Learnable embeddings for vocabulary tokens
- **Positional Encoding**: Sinusoidal positional encodings for sequence position information
- **Transformer Blocks**: Stack of decoder blocks, each containing:
  - Multi-head causal self-attention
  - Pre-norm layer normalization
  - Position-wise feed-forward network (GELU activation)
  - Residual connections
- **Output Projection**: Linear layer mapping to vocabulary with weight tying

**Default Configuration:**
- Vocabulary size: 50,257 (GPT-2 tokenizer)
- Model dimension: 768
- Number of layers: 12
- Attention heads: 12
- Feed-forward dimension: 3,072
- Maximum sequence length: 512
- Total parameters: ~117M

### Distributed Training Architecture

The training system uses Horovod for distributed training:

**Components:**

1. **Horovod DistributedOptimizer**: Wraps PyTorch optimizer to synchronize gradients
2. **DistributedSampler**: Ensures each GPU processes different data partitions
3. **Gradient Accumulation**: Allows effective larger batch sizes
4. **Mixed Precision Training**: BF16/FP16 support for H100 GPUs
5. **Checkpointing System**: Saves model, optimizer, and scheduler states

**Training Flow:**

```
Data Loading → Model Forward → Loss Computation → Backward Pass →
Gradient Accumulation → Horovod AllReduce → Optimizer Step →
Learning Rate Schedule → Checkpointing (periodic)
```

## Data Pipeline

### Dataset

**BookCorpus Dataset**: Large collection of book texts used for unsupervised language model training.

- **Source**: BookCorpusOpen (HuggingFace)
- **Preprocessing**: Tokenized with GPT-2 tokenizer
- **Splits**: 95% training, 5% validation
- **Sequence Length**: Fixed-length blocks of 512 tokens

### Data Preprocessing Workflow

1. **Load Raw Data**: Load BookCorpus from parquet files
2. **Tokenization**: Apply GPT-2 tokenizer to convert text to token IDs
3. **Sequence Packing**: Pack tokens into fixed-length sequences (512 tokens)
4. **Train/Val Split**: Create 95/5 split
5. **Save to Disk**: Save preprocessed datasets in HuggingFace Dataset format

### Data Loading

- **DistributedSampler**: Ensures each GPU gets non-overlapping data
- **Efficient Loading**: Pin memory, persistent workers, prefetching
- **Batch Handling**: Supports gradient accumulation with proper batch management

## Training Workflow

### Step 1: Data Preprocessing

Run the preprocessing notebook to prepare the dataset:

```bash
jupyter notebook notebooks/00_data_preprocessing.ipynb
```

This creates preprocessed datasets at `notebooks/data/preprocessed/v1/`.

### Step 2: Configuration

Configure training parameters in `configs/config.yaml`:

- Model architecture parameters
- Training hyperparameters (learning rate, batch size, epochs)
- Data paths
- Distributed settings

### Step 3: Launch Distributed Training

Submit training job to HPC cluster:

```bash
sbatch scripts/train_zaratan.sh configs/config.yaml
```

Or run locally with horovodrun:

```bash
horovodrun -np 4 python -m src.train --config configs/config.yaml
```

### Step 4: Monitor Training

- **TensorBoard**: `tensorboard --logdir runs/`
- **Log Files**: `logs/train_rank0.log`
- **Structured Metrics**: `runs/structured/`

### Step 5: Text Generation

Generate text from trained checkpoint:

```bash
python -m src.generate \
  --checkpoint checkpoints/best_model.pt \
  --config configs/config.yaml \
  --prompt "Once upon a time" \
  --max_new_tokens 100
```

## Usage Examples

### Complete Workflow Example

See `horovod_transformer.example.py` for a complete example demonstrating:

```python
from horovod_transformer.example import TransformerTrainingExample

example = TransformerTrainingExample()
example.run_complete_workflow(
    train=True,
    generate=True,
    checkpoint_path="checkpoints/best_model.pt"
)
```

### Minimal Training Example

```python
from src.train import run_distributed_training
from src.utils.config import load_config
from src.utils.distributed import setup_horovod

config = load_config("configs/config.yaml")
world_size, rank, local_rank, _ = setup_horovod(require_distributed=True)

run_distributed_training(
    config=config,
    rank=rank,
    world_size=world_size,
    local_rank=local_rank
)
```

### Text Generation Example

```python
from src.generate import load_model_for_generation, generate_text

model, tokenizer, config = load_model_for_generation(
    checkpoint_path="checkpoints/best_model.pt",
    config_path="configs/config.yaml",
    device="cuda"
)

generated = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt="The future of AI",
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9
)
```

## Results and Evaluation

### Training Metrics

The system tracks several metrics during training:

- **Loss**: Cross-entropy loss for next-token prediction
- **Perplexity**: exp(loss), measures model uncertainty
- **Accuracy**: Token-level prediction accuracy
- **Learning Rate**: Current learning rate (with warmup and decay)
- **Throughput**: Training updates per second

### Monitoring

- **TensorBoard Logging**: Real-time visualization of training metrics
- **Structured CSV Logs**: Detailed metrics for analysis
- **Checkpointing**: Periodic saves for fault tolerance and model selection

### Text Generation Quality

Generated text is evaluated qualitatively by examining:

- **Coherence**: Logical flow and consistency
- **Relevance**: Adherence to the input prompt
- **Diversity**: Variety in generated outputs
- **Length**: Appropriate generation length

### Performance Characteristics

- **Training Speed**: Scales linearly with number of GPUs (up to network limits)
- **Memory Efficiency**: Gradient checkpointing and mixed precision reduce VRAM usage
- **Fault Tolerance**: Checkpointing allows resuming from interruptions

## API Usage in This Project

This project extensively uses the Horovod Transformer API as documented in `horovod_transformer.API.md`:

1. **Model API**: Creating and configuring transformer models
2. **Training API**: Distributed training with checkpointing
3. **Data API**: Loading and preprocessing datasets
4. **Distributed API**: Horovod setup and synchronization
5. **Generation API**: Text generation from trained models
6. **Configuration API**: YAML-based configuration management

See `horovod_transformer.example.py` for detailed code examples of API usage throughout the project workflow.

