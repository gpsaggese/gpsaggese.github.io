# DeepSpeed API Documentation

Complete documentation of the DeepSpeed API and utility functions in `DeepSpeed_utils.py`.

## Core DeepSpeed Concepts

DeepSpeed provides memory-efficient distributed training through:
- **ZeRO (Zero Redundancy Optimizer)**: Partitions optimizer states, gradients, and parameters across GPUs
- **Mixed Precision**: BF16/FP16 support for faster training and lower memory
- **CPU Offloading**: Offload optimizer states and parameters to CPU for extreme memory savings
- **Gradient Accumulation**: Simulate larger batch sizes with limited GPU memory

## Key DeepSpeed APIs

### Initialization

```python
import deepspeed

model_engine, optimizer, lr_scheduler = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=config_path  # Path to JSON config or dict
)
```

### Model Engine

The DeepSpeed model engine extends PyTorch models:

```python
# Forward pass (same as PyTorch)
outputs = model_engine(inputs)

# Backward pass (handles gradient accumulation)
model_engine.backward(loss)

# Optimizer step (handles gradient clipping, learning rate scheduling)
model_engine.step()
```

## Utility Functions in `DeepSpeed_utils.py`

### Model Loading

#### `load_vit_model()`
Loads a pre-trained Vision Transformer from Hugging Face.

```python
model = load_vit_model(
    model_name="google/vit-huge-patch14-224-in21k",
    num_labels=101,
    torch_dtype=torch.bfloat16,
    enable_gradient_checkpointing=True
)
```

**Parameters:**
- `model_name`: Hugging Face model identifier
- `num_labels`: Number of output classes
- `torch_dtype`: Data type (default: `torch.bfloat16`)
- `enable_gradient_checkpointing`: Enable memory-efficient training

**Returns:** `ViTForImageClassification` model instance

---

### Data Preparation

#### `create_image_transforms()`
Creates data transformation pipelines for training and validation.

```python
train_transform, val_transform = create_image_transforms(
    image_size=224,
    use_augmentation=True,
    normalize_mean=(0.485, 0.456, 0.406),
    normalize_std=(0.229, 0.224, 0.225)
)
```

**Returns:** Tuple of `(train_transform, val_transform)`

#### `load_food101_dataset()`
Loads Food-101 dataset with proper transforms.

```python
trainset, testset = load_food101_dataset(
    root="./data",
    train_transform=train_transform,
    test_transform=test_transform
)
```

#### `create_data_loaders()`
Creates distributed data loaders with proper samplers.

```python
train_loader, test_loader = create_data_loaders(
    trainset=trainset,
    testset=testset,
    batch_size=16,
    world_size=2,
    rank=0,
    num_workers=4,
    pin_memory=True
)
```

**Key Features:**
- Automatically uses `DistributedSampler` for multi-GPU
- Handles single-GPU case gracefully
- Supports pin_memory for faster data transfer

---

### DeepSpeed Configuration

#### `create_deepspeed_config()`
Creates a DeepSpeed configuration dictionary.

```python
config = create_deepspeed_config(
    zero_stage=3,
    micro_batch_size=8,
    gradient_accumulation_steps=4,
    offload_optimizer=False,
    offload_param=False,
    use_bf16=True,
    learning_rate=5e-5,
    weight_decay=0.01
)
```

**Parameters:**
- `zero_stage`: ZeRO stage (0, 1, 2, 3)
- `micro_batch_size`: Batch size per GPU
- `gradient_accumulation_steps`: Steps to accumulate gradients
- `offload_optimizer`: Offload optimizer states to CPU
- `offload_param`: Offload parameters to CPU (ZeRO Stage 3 only)
- `use_bf16`: Enable BF16 mixed precision
- `learning_rate`: Learning rate
- `weight_decay`: Weight decay

**Returns:** Dictionary compatible with DeepSpeed config format

#### `save_deepspeed_config()` / `load_deepspeed_config()`
Save/load DeepSpeed configurations to/from JSON files.

```python
save_deepspeed_config(config, "deepspeed_config_stage3.json")
config = load_deepspeed_config("deepspeed_config_stage3.json")
```

---

### DeepSpeed Initialization

#### `initialize_deepspeed_model()`
Initializes a DeepSpeed model engine.

```python
model_engine, optimizer, lr_scheduler = initialize_deepspeed_model(
    model=model,
    config_path="deepspeed_config_stage3.json",
    model_parameters=model.parameters()
)
```

**Returns:** Tuple of `(model_engine, optimizer, lr_scheduler)`

---

### Training Functions

#### `train_epoch_deepspeed()`
Training loop for DeepSpeed models.

```python
avg_loss = train_epoch_deepspeed(
    model_engine=model_engine,
    train_loader=train_loader,
    device=device,
    epoch=0,
    enable_profiling=False,
    experiment_name="my_experiment"
)
```

**Features:**
- Handles gradient accumulation automatically
- Supports PyTorch Profiler integration
- Logs metrics to WandB
- Returns average training loss

#### `train_epoch_standard()`
Training loop for standard PyTorch models (DDP or single-GPU).

```python
avg_loss = train_epoch_standard(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    epoch=0,
    use_amp=True,
    dtype=torch.bfloat16
)
```

---

### Evaluation Functions

#### `evaluate_deepspeed()`
Evaluation with DeepSpeed models.

```python
accuracy, metrics = evaluate_deepspeed(
    model_engine=model_engine,
    val_loader=test_loader,
    device=device
)
```

**Returns:** Tuple of `(accuracy, metrics_dict)`

#### `evaluate_standard()`
Evaluation with standard PyTorch models.

```python
accuracy, metrics = evaluate_standard(
    model=model,
    val_loader=test_loader,
    device=device,
    use_amp=True
)
```

---

### Experiment Runner

#### `run_training_experiment()`
Complete experiment runner that handles all training methods.

```python
results = run_training_experiment(
    method="zero3",  # 'ddp', 'zero2', 'zero3', 'zeropp', 'single_gpu_simple'
    model_name="google/vit-huge-patch14-224-in21k",
    train_loader=train_loader,
    test_loader=test_loader,
    rank=0,
    world_size=2,
    num_epochs=1,
    micro_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    enable_profiling=True,
    wandb_project="My Project"
)
```

**Supported Methods:**
- `'ddp'`: PyTorch DistributedDataParallel
- `'zero2'`: DeepSpeed ZeRO Stage 2
- `'zero3'`: DeepSpeed ZeRO Stage 3
- `'zeropp'`: DeepSpeed ZeRO++ (quantized gradients)
- `'single_gpu_simple'`: Standard single-GPU training
- `'single_gpu_offload'`: Single-GPU with CPU offloading

**Returns:** Dictionary with experiment results:
- `'final_accuracy'`: Final validation accuracy
- `'train_losses'`: List of training losses per epoch
- `'epoch_duration'`: Time per epoch
- `'peak_memory_gb'`: Peak GPU memory usage
- `'throughput'`: Training throughput (samples/second)

---

### Distributed Setup

#### `setup_distributed()`
Sets up distributed training environment.

```python
device, rank, local_rank, world_size = setup_distributed()
```

**Returns:** Tuple of `(device, rank, local_rank, world_size)`

**Features:**
- Auto-detects environment variables (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`)
- Falls back to single-GPU if not in distributed mode
- Configures CUDA device

#### `cleanup_distributed()`
Cleans up distributed training resources.

```python
cleanup_distributed()
```

---

### Performance Measurement

#### `measure_throughput()`
Measures training throughput.

```python
throughput = measure_throughput(
    model=model,
    train_loader=train_loader,
    device=device,
    num_batches=10
)
```

**Returns:** Throughput in samples/second

#### `get_gpu_memory_stats()`
Gets current GPU memory statistics.

```python
stats = get_gpu_memory_stats(device_id=0)
# Returns: {
#     'allocated_gb': ...,
#     'reserved_gb': ...,
#     'max_allocated_gb': ...,
#     'max_reserved_gb': ...
# }
```

---

### WandB Integration

#### `load_wandb_api_key_from_kaggle_secrets()`
Loads WandB API key from Kaggle Secrets.

```python
success = load_wandb_api_key_from_kaggle_secrets()
```

**Returns:** `True` if key was loaded, `False` otherwise

#### `is_wandb_authenticated()`
Checks if WandB is properly authenticated.

```python
if is_wandb_authenticated():
    wandb.init(project="My Project")
```

#### `test_wandb_connection()`
Tests WandB connection and attempts login if needed.

```python
if test_wandb_connection():
    print("WandB ready!")
```

---

### Profiling Support

#### `run_hta_analysis()`
Runs HTA (Holistic Trace Analysis) on profiling traces.

```python
report = run_hta_analysis(
    trace_dir="./profiling_traces/my_experiment",
    output_dir="./hta_reports"
)
```

**Returns:** Dictionary with performance analysis results

---

### Visualization

#### `create_performance_visualizations()`
Creates performance comparison visualizations.

```python
create_performance_visualizations(
    results_df=results_dataframe,
    output_dir="./plots"
)
```

Generates charts for:
- Memory usage comparison
- Throughput comparison
- Training time comparison
- Scaling efficiency

---

## ZeRO Stages Explained

### ZeRO Stage 0 (Baseline DDP)
- **What**: Standard PyTorch DDP
- **Memory**: Full model + optimizer states + gradients replicated on each GPU
- **Use Case**: Small models that fit on single GPU

### ZeRO Stage 1
- **What**: Partitions optimizer states across GPUs
- **Memory Savings**: ~4x reduction in optimizer state memory
- **Use Case**: Models with large optimizer states (AdamW with large models)

### ZeRO Stage 2
- **What**: Partitions optimizer states AND gradients
- **Memory Savings**: ~8x reduction (optimizer + gradients)
- **Use Case**: Models with limited GPU memory, moderate communication overhead

### ZeRO Stage 3
- **What**: Partitions optimizer states, gradients, AND parameters
- **Memory Savings**: Linear scaling with number of GPUs
- **Use Case**: Very large models that don't fit on single GPU
- **Trade-off**: Increased communication overhead

### ZeRO++ (Enhanced ZeRO Stage 3)
- **What**: ZeRO Stage 3 + quantized gradient communication
- **Memory Savings**: Same as ZeRO Stage 3
- **Speed Improvement**: Reduced communication time
- **Use Case**: Communication-bound scenarios with many GPUs

---

## Example Usage

### Complete Training Pipeline

```python
from DeepSpeed_utils import (
    load_vit_model,
    create_data_loaders,
    run_training_experiment
)

# 1. Load model
model = load_vit_model(
    model_name="google/vit-huge-patch14-224-in21k",
    num_labels=101,
    enable_gradient_checkpointing=True
)

# 2. Prepare data
trainset, testset = load_food101_dataset(...)
train_loader, test_loader = create_data_loaders(
    trainset, testset,
    batch_size=16,
    world_size=2,
    rank=0
)

# 3. Run experiment
results = run_training_experiment(
    method="zero3",
    model_name="google/vit-huge-patch14-224-in21k",
    train_loader=train_loader,
    test_loader=test_loader,
    world_size=2,
    num_epochs=1,
    enable_profiling=True
)

print(f"Final Accuracy: {results['final_accuracy']:.2f}%")
print(f"Peak Memory: {results['peak_memory_gb']:.2f} GB")
```

---

## See Also

- **`DeepSpeed.API.ipynb`**: Interactive examples and demonstrations
- **`DeepSpeed.example.md`**: Guide to example notebooks
- **`README.md`**: Overall project documentation

## References

- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [ZeRO Paper](https://arxiv.org/abs/1910.02054)
- [ZeRO++ Paper](https://arxiv.org/abs/2306.02209)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
