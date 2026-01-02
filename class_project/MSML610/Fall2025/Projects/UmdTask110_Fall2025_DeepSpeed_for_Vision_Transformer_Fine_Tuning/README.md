# DeepSpeed for Vision Transformer Fine-Tuning

A comprehensive project for fine-tuning large Vision Transformers (ViTs) using DeepSpeed ZeRO optimization and PyTorch DistributedDataParallel (DDP). This project demonstrates memory-efficient distributed training strategies for large-scale vision models on multiple GPUs.

## Project Overview

This project provides a complete framework for:
- **Single-GPU Training**: LoRA-based fine-tuning with gradient checkpointing
- **Multi-GPU Training**: Distributed training using DDP, DeepSpeed ZeRO Stage 2, ZeRO Stage 3, and ZeRO++
- **Performance Analysis**: Profiling and trace analysis using PyTorch Profiler and HTA (Holistic Trace Analysis)
- **Experiment Tracking**: WandB integration for metrics, memory usage, and performance monitoring

## Project Structure

```
.
├── DeepSpeed_utils.py                    # Core utility functions for training
├── DeepSpeed.API.ipynb                   # Interactive API demonstrations
├── DeepSpeed.API.md                      # API documentation
├── DeepSpeed.example.md                  # Example notebooks guide
├── Deepspeed_single_gpu.example.ipynb    # Single-GPU LoRA experiments
├── DeepSpeed_multi_gpu.example.ipynb     # Multi-GPU distributed experiments
├── DeepSpeed_trace_analysis.example.ipynb # HTA trace analysis
├── Dockerfile                            # Docker container configuration
├── requirements.txt                      # Python dependencies
└── README.md                            # This file
```

## Key Features

### 1. **Distributed Training Methods**

- **DDP (DistributedDataParallel)**: Standard PyTorch multi-GPU training with full model replication
- **ZeRO Stage 2**: Partitions optimizer states and gradients across GPUs
- **ZeRO Stage 3**: Partitions optimizer states, gradients, and model parameters
- **ZeRO++**: Enhanced ZeRO Stage 3 with quantized gradient communication

### 2. **Memory Optimization Techniques**

- **Gradient Checkpointing**: Reduces memory during backward pass
- **Mixed Precision Training**: BF16/FP16 support for faster training and lower memory
- **CPU Offloading**: Optional offloading of optimizer states and parameters to CPU
- **Gradient Accumulation**: Supports large effective batch sizes with limited GPU memory

### 3. **Models and Datasets**

- **Models**: 
  - ViT-Base (~86M parameters) for single-GPU experiments
  - ViT-Huge (~630M parameters) for multi-GPU experiments
  - All models are trained from scratch (not fine-tuned from pre-trained weights)
- **Datasets**:
  - CIFAR-100 (single-GPU experiments)
  - Food-101 (multi-GPU experiments)

### 4. **Performance Analysis**

- **PyTorch Profiler**: Detailed CPU/GPU activity traces
- **HTA (Holistic Trace Analysis)**: Advanced performance bottleneck identification
- **WandB Logging**: Real-time metrics, memory usage, and throughput tracking

## Prerequisites

1. **Hardware**: 
   - Multi-GPU system (tested with 2x Tesla T4, 1x A100)
   - CUDA-capable GPUs with sufficient memory

2. **Software**: 
   - Python 3.8+
   - PyTorch 2.0+ with CUDA support
   - DeepSpeed 0.10+
   - NCCL for multi-GPU communication

3. **Dependencies**: Install all requirements:
   ```bash
   pip install -r requirements.txt
   ```

4. **WandB Setup**: 
   - Create account at https://wandb.ai
   - Run `wandb login` or set `WANDB_API_KEY` environment variable
   - For Kaggle: Add `WANDB_API_KEY` to Secrets (Add-ons → Secrets)

## Note on Experiment Runtime

**Important**: Both Single-GPU and Multi-GPU experiments take several hours to complete due to the large scale of Vision Transformer models used (ranging from ~86 million parameters to ~630 million parameters) and the fact that all models are trained from scratch (not fine-tuned from pre-trained weights). During video presentations and demonstrations, pre-run experimental results will be shown instead of executing the notebooks step-by-step in real-time.

## Quick Start

### Single-GPU Training

See `Deepspeed_single_gpu.example.ipynb` for:
- LoRA-based fine-tuning on CIFAR-100
- Gradient checkpointing
- Standard PyTorch optimizer (no DeepSpeed)

### Multi-GPU Training

See `DeepSpeed_multi_gpu.example.ipynb` for:
- DDP training with ViT-Huge on Food-101
- DeepSpeed ZeRO Stage 2, 3, and ZeRO++ experiments
- Profiling and trace collection

### API Usage

See `DeepSpeed.API.ipynb` for minimal examples of:
- Creating DeepSpeed configurations
- Initializing DeepSpeed models
- Basic training patterns

### Trace Analysis

See `DeepSpeed_trace_analysis.example.ipynb` for:
- Extracting and merging profiling traces
- HTA analysis for performance bottlenecks
- Visualizing GPU utilization and communication patterns

## Usage Examples

### Using the Utility Functions

```python
from DeepSpeed_utils import (
    load_vit_model,
    create_data_loaders,
    run_training_experiment
)

# Load model
model = load_vit_model(
    model_name="google/vit-huge-patch14-224-in21k",
    num_labels=101,
    enable_gradient_checkpointing=True
)

# Run experiment
results = run_training_experiment(
    method="zero3",
    model_name="google/vit-huge-patch14-224-in21k",
    train_loader=train_loader,
    test_loader=test_loader,
    world_size=2,
    num_epochs=1,
    enable_profiling=True
)
```

### Running Experiments

All training experiments are contained within the Jupyter notebooks. The notebooks use the utility functions from `DeepSpeed_utils.py` to run distributed training experiments. For command-line usage, refer to the examples within the notebooks and the API documentation in `DeepSpeed.API.md`.

## Experiment Tracking

All experiments automatically log to WandB with:
- Training/validation loss and accuracy
- GPU memory usage (allocated, reserved, peak)
- Training throughput (samples/second)
- Epoch duration
- System metrics (GPU utilization, temperature)

## Performance Comparison

The project enables comparison of:
- **Memory Efficiency**: Peak GPU memory usage across methods
- **Training Speed**: Throughput (samples/second) and epoch duration
- **Scalability**: Performance scaling with number of GPUs
- **Communication Overhead**: Communication time vs computation time

## Configuration Files

### DeepSpeed Configuration

DeepSpeed configurations are created programmatically within the notebooks using the utility functions. Configuration options include:
- ZeRO Stage 2, 3, and ZeRO++ optimization
- BF16/FP16 mixed precision
- Gradient accumulation
- Optimizer and parameter offloading options

See `DeepSpeed.API.md` for detailed configuration options and examples.

## Troubleshooting

### CUDA Out of Memory

1. Reduce `micro_batch_size` or `train_micro_batch_size_per_gpu`
2. Increase `gradient_accumulation_steps`
3. Enable gradient checkpointing
4. Use ZeRO Stage 3 with CPU offloading

### NCCL Errors

1. Ensure NCCL is properly installed
2. Set `export NCCL_DEBUG=INFO` for detailed logs
3. Check network connectivity between GPUs

### WandB Authentication

1. Run `wandb login` or set `WANDB_API_KEY`
2. In Kaggle: Add API key to Secrets
3. Check network connectivity to WandB servers

## References

- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [ZeRO Paper](https://arxiv.org/abs/1910.02054)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [HTA Documentation](https://github.com/facebookresearch/hta)

## License

See LICENSE file in the repository.

## Contributing

This is a course project for MSML 610. For questions or issues, please refer to the course materials or contact the instructor.
