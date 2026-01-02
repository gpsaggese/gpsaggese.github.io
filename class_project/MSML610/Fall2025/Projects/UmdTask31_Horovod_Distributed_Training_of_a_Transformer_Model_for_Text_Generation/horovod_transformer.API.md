<!-- toc -->
- [Horovod Transformer Tutorial](#horovod-transformer-tutorial)
  * [Table of Contents](#table-of-contents)
  + [Hierarchy](#hierarchy)
  * [General Guidelines](#general-guidelines)
  * [API Overview](#api-overview)
  * [Model API](#model-api)
  * [Training API](#training-api)
  * [Data Loading API](#data-loading-api)
  * [Distributed Training API](#distributed-training-api)
  * [Text Generation API](#text-generation-api)
  * [Configuration API](#configuration-api)
<!-- tocstop -->

# Horovod Transformer Tutorial

Tutorial for the native API of the Horovod distributed transformer language model training system.

## Table of Contents

The markdown code can have a TOC. This can be generated automatically with the linter or other tools.

### Hierarchy

Hierarchy of the markdown file should be followed.

# Level 1 (Used as title)
All the subheadings should follow the below structure:
## Level 2
### Level 3

**Note** Level 1 Heading (Title) should be `Horovod Transformer Tutorial`.

## General Guidelines

- Follow the instructions in [README](/DATA605/DATA605_Spring2025/README.md) on what to write in the API tutorial.
- Include descriptions of API works based on what is explored in `horovod_transformer.API.py`.
- The file should be named with the following conventions:
  - Since this project is based on Horovod distributed transformer training, it is named `horovod_transformer.API.md`

## API Overview

The Horovod Transformer API provides a comprehensive interface for distributed training of transformer-based language models. The system is built on top of:

- **Horovod**: For multi-GPU distributed training with efficient gradient synchronization
- **PyTorch**: Deep learning framework
- **Custom Transformer Architecture**: GPT-style decoder-only transformer for autoregressive language modeling

The API is organized into several key modules:

1. **Model API**: TransformerLM class for model creation and forward passes
2. **Training API**: Distributed training loop with checkpointing and logging
3. **Data Loading API**: Efficient data loading with distributed sampling
4. **Distributed Training API**: Horovod integration for multi-GPU training
5. **Text Generation API**: Autoregressive text generation from trained models
6. **Configuration API**: YAML-based configuration management

## Model API

### TransformerLM Class

The `TransformerLM` class implements a decoder-only transformer architecture for language modeling.

**Initialization:**

```python
from src.models.transformer_lm import TransformerLM

model = TransformerLM(
    vocab_size=50257,
    d_model=768,
    n_heads=12,
    n_layers=12,
    d_ff=3072,
    max_seq_len=512,
    dropout=0.1,
    pad_token_id=50256,
    gradient_checkpointing=False
)
```

**Key Parameters:**
- `vocab_size`: Size of the vocabulary (default: 50257 for GPT-2 tokenizer)
- `d_model`: Model dimension (default: 768)
- `n_heads`: Number of attention heads (default: 12)
- `n_layers`: Number of transformer layers (default: 12)
- `d_ff`: Feed-forward hidden dimension (default: 3072)
- `max_seq_len`: Maximum sequence length (default: 512)
- `dropout`: Dropout probability (default: 0.1)

**Forward Pass:**

```python
logits, loss = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels
)
```

**Text Generation:**

```python
generated_ids = model.generate(
    input_ids=input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    eos_token_id=50256
)
```

## Training API

### Main Training Function

The training API is accessed through the `src.train` module:

```python
from src.train import run_distributed_training
from src.utils.config import load_config

config = load_config("configs/config.yaml")
world_size, rank, local_rank, _ = setup_horovod()

run_distributed_training(
    config=config,
    rank=rank,
    world_size=world_size,
    local_rank=local_rank,
    resume_checkpoint=None,
    run_name="my_training_run"
)
```

**Key Features:**
- Automatic Horovod initialization and optimizer wrapping
- Mixed precision training (bf16/fp16) for H100 GPUs
- Gradient accumulation for effective larger batch sizes
- Learning rate scheduling with warmup
- Checkpointing and resume support
- TensorBoard logging
- Structured metrics recording

### Checkpoint Management

```python
from src.train import save_checkpoint, load_checkpoint

# Save checkpoint
save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    step=step,
    loss=loss,
    checkpoint_dir="checkpoints",
    filename="checkpoint_epoch5.pt",
    scheduler=scheduler,
    scaler=scaler,
    global_step=global_step
)

# Load checkpoint
checkpoint = load_checkpoint(
    checkpoint_path="checkpoints/best_model.pt",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler
)
```

## Data Loading API

### Loading Preprocessed Data

```python
from src.data import get_dataloaders, load_preprocessed_data

# Load datasets
train_dataset, val_dataset = load_preprocessed_data(
    data_dir="notebooks/data/preprocessed/v1"
)

# Create distributed data loaders
train_dataloader, val_dataloader = get_dataloaders(config)
```

**Features:**
- Automatic distributed sampling with `DistributedSampler`
- Efficient data loading with pin_memory and persistent workers
- Support for gradient accumulation batch handling

## Distributed Training API

### Horovod Setup

```python
from src.utils.distributed import (
    setup_horovod,
    get_rank,
    get_world_size,
    get_local_rank,
    metric_average,
    is_main_process
)

# Initialize Horovod
world_size, rank, local_rank, use_cuda = setup_horovod(require_distributed=True)

# Get distributed information
current_rank = get_rank()
total_processes = get_world_size()
local_gpu_id = get_local_rank()

# Average metrics across processes
avg_loss = metric_average(local_loss, name="train_loss")

# Check if main process
if is_main_process():
    print("This only prints on rank 0")
```

### Distributed Optimizer

```python
import horovod.torch as hvd
from torch.optim import AdamW

# Create optimizer
optimizer = AdamW(model.parameters(), lr=3e-4)

# Wrap with Horovod DistributedOptimizer
optimizer = hvd.DistributedOptimizer(
    optimizer,
    named_parameters=model.named_parameters(),
    compression=hvd.Compression.none
)

# Broadcast initial parameters
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
```

## Text Generation API

### Loading Model for Generation

```python
from src.generate import load_model_for_generation, generate_text

# Load trained model
model, tokenizer, config = load_model_for_generation(
    checkpoint_path="checkpoints/best_model.pt",
    config_path="configs/config.yaml",
    device="cuda"
)

# Generate text
generated_texts = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt="Once upon a time",
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    num_samples=1,
    device="cuda"
)
```

### Interactive Generation

```python
from src.generate import interactive_generation

interactive_generation(
    model=model,
    tokenizer=tokenizer,
    config=config,
    device="cuda"
)
```

## Configuration API

### Loading Configuration

```python
from src.utils.config import load_config, Config

# Load from YAML file
config = load_config("configs/config.yaml")

# Access nested attributes
model_dim = config.model.d_model
learning_rate = config.training.learning_rate
batch_size = config.training.per_gpu_batch_size

# Convert to dictionary
config_dict = config.to_dict()
```

### Configuration Structure

The configuration file (`configs/config.yaml`) supports:

- **Model configuration**: Architecture parameters
- **Training configuration**: Hyperparameters, batch sizes, learning rates
- **Data configuration**: Data paths, number of workers
- **Generation configuration**: Sampling parameters
- **Paths configuration**: Checkpoint, log, and TensorBoard directories
- **Distributed configuration**: Horovod compression settings

## Usage Examples

See `horovod_transformer.example.py` for complete usage examples demonstrating how to use the API in practice.

