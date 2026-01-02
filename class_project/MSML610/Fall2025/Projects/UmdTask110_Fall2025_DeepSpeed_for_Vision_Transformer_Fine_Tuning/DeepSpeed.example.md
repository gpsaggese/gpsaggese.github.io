# DeepSpeed Example Notebooks Guide

This document explains what each example notebook does and how to use them.

## Overview

The project includes three main example notebooks:

1. **`Deepspeed_single_gpu.example.ipynb`**: Single-GPU LoRA fine-tuning experiments
2. **`DeepSpeed_multi_gpu.example.ipynb`**: Multi-GPU distributed training experiments
3. **`DeepSpeed_trace_analysis.example.ipynb`**: Performance trace analysis using HTA

---

## 1. Deepspeed_single_gpu.example.ipynb

### Purpose
Demonstrates efficient single-GPU fine-tuning of Vision Transformers using LoRA (Low-Rank Adaptation) and gradient checkpointing.

### What It Does

#### Setup and Environment
- Tests PyTorch and CUDA availability
- Verifies GPU configuration
- Sets up WandB authentication (supports Kaggle Secrets)

#### Dataset Preparation
- Downloads CIFAR-100 dataset (50,000 training, 10,000 test images)
- Applies ViT-compatible transforms:
  - Resizes images to 224x224 (ViT input size)
  - Applies data augmentation (random horizontal flip, random crop)
  - Normalizes with ImageNet statistics

#### Model Loading
- Loads pre-trained ViT-Base (`google/vit-base-patch16-224`)
- Configures for 100 classes (CIFAR-100)
- Enables gradient checkpointing to reduce memory usage

#### LoRA Configuration
- Uses PEFT (Parameter-Efficient Fine-Tuning) library
- Applies LoRA adapters to attention layers
- Configures LoRA rank (r=16) and alpha parameters
- Freezes base model parameters, only trains LoRA weights

#### Training Process
- Standard PyTorch training loop (no DeepSpeed)
- AdamW optimizer with learning rate 5e-5
- BF16 mixed precision training
- WandB logging for loss tracking
- 3 epochs of training

### Key Features
- **Memory Efficient**: LoRA reduces trainable parameters from ~86M to ~1M
- **Fast Training**: Gradient checkpointing enables larger batch sizes
- **Easy to Run**: Works on single GPU without distributed setup

### When to Use
- Learning LoRA fine-tuning techniques
- Testing on smaller models/datasets
- Single-GPU environments
- Quick experiments before scaling up

---

## 2. DeepSpeed_multi_gpu.example.ipynb

### Purpose
Demonstrates multi-GPU distributed training with ViT-Huge using different optimization strategies: DDP, ZeRO Stage 2, ZeRO Stage 3, and ZeRO++.

### What It Does

#### Environment Setup
- Detects available GPUs (tested with 2x Tesla T4)
- Sets up distributed training environment
- Configures WandB for multi-GPU logging

#### Dataset Preparation
- Downloads Food-101 dataset (75,750 training, 25,250 test images)
- Applies ViT-compatible transforms
- Creates distributed data loaders with `DistributedSampler`

#### Model Configuration
- Loads ViT-Huge (`google/vit-huge-patch14-224-in21k`) - ~630M parameters
- Configures for 101 classes (Food-101)
- Enables gradient checkpointing

#### Experiment Types

**1. Standard DDP (DistributedDataParallel)**
- Full model replication on each GPU
- Standard PyTorch DDP wrapper
- BF16 mixed precision with GradScaler
- Profiling support for trace collection

**2. DeepSpeed ZeRO Stage 2**
- Partitions optimizer states and gradients
- Reduces memory compared to DDP
- Configuration: `deepspeed_config_stage2_vit_huge_bs16.json`
- Merges profiling traces from all GPUs

**3. DeepSpeed ZeRO Stage 3**
- Partitions optimizer states, gradients, AND parameters
- Maximum memory efficiency
- Configuration: `deepspeed_config_stage3_vit_huge_bs16.json`
- Requires parameter gathering for forward pass

**4. DeepSpeed ZeRO++**
- Enhanced ZeRO Stage 3 with quantized gradients
- Reduces communication overhead
- Configuration: `deepspeed_config_zeropp.json`
- Best for communication-bound scenarios

#### Profiling Integration
- PyTorch Profiler with custom labels:
  - `## Training Step {i} ##`: Top-level step marker
  - `## Forward Pass ##`: Forward computation
  - `## Backward Pass ##`: Backward computation
  - `## Optimizer Step ##`: Optimizer update
- Multi-GPU trace merging for unified analysis
- Saves traces as JSON for Perfetto/HTA analysis

#### Training Workflow
Each experiment:
1. Initializes distributed process group
2. Loads model and data
3. Sets up DeepSpeed engine (if applicable)
4. Runs training loop with profiling
5. Merges traces from all GPUs (rank 0)
6. Logs metrics to WandB
7. Cleans up distributed resources

### Key Features
- **Multi-GPU Support**: Runs on 2+ GPUs using `torchrun` or `deepspeed` launcher
- **Memory Comparison**: Demonstrates memory savings across ZeRO stages
- **Profiling Ready**: Generates traces for performance analysis
- **Production Ready**: Handles distributed setup, error recovery, and cleanup

### When to Use
- Comparing distributed training strategies
- Training large models that don't fit on single GPU
- Understanding memory vs speed trade-offs
- Generating profiling data for analysis

### Running the Experiments

**DDP:**
```bash
torchrun --nproc_per_node=2 train_standard_ddp_vit_huge_bs16.py
```

**DeepSpeed ZeRO Stage 2:**
```bash
deepspeed --num_gpus=2 train_deepspeed_stage2_vit_huge_bs16.py
```

**DeepSpeed ZeRO Stage 3:**
```bash
deepspeed --num_gpus=2 train_deepspeed_stage3_vit_huge_bs16.py
```

**DeepSpeed ZeRO++:**
```bash
deepspeed --num_gpus=2 train_deepspeed_zeropp.py
```

---

## 3. DeepSpeed_trace_analysis.example.ipynb

### Purpose
Analyzes PyTorch profiling traces using HTA (Holistic Trace Analysis) to identify performance bottlenecks.

### What It Does

#### Trace Extraction
- Locates profiling trace ZIP files from previous experiments
- Extracts traces to local analysis directory
- Organizes traces by experiment type (Stage 2, Stage 3, ZeRO++)

#### Trace Structure
Each trace contains:
- CPU and CUDA kernel events
- Memory allocation/deallocation events
- Communication events (NCCL all-reduce, all-gather)
- Custom labeled regions (forward, backward, optimizer)

#### HTA Analysis
- Loads traces into HTA framework
- Generates performance reports:
  - **GPU Utilization**: Percentage of time GPU is active
  - **Kernel Breakdown**: Time spent in different CUDA kernels
  - **Communication Overhead**: Time spent in NCCL operations
  - **Memory Bandwidth**: Memory transfer rates
  - **Idle Time**: GPU idle periods

#### Visualization
- Generates charts for:
  - Timeline view of GPU activities
  - Kernel duration distribution
  - Communication vs computation ratio
  - Memory usage over time

#### Performance Insights
Identifies:
- **Bottlenecks**: Long-running operations
- **Inefficiencies**: GPU idle time, poor kernel utilization
- **Communication Overhead**: Time spent in all-reduce/all-gather
- **Memory Issues**: Frequent allocations, high memory pressure

### Key Features
- **Multi-GPU Analysis**: Handles merged traces from multiple GPUs
- **Automated Reports**: Generates comprehensive performance summaries
- **Visualization**: Charts and graphs for easy interpretation
- **Comparative Analysis**: Compare different optimization strategies

### When to Use
- After running multi-GPU experiments
- Debugging performance issues
- Optimizing training speed
- Understanding communication patterns
- Comparing ZeRO stages

### Workflow

1. **Extract Traces**: Unzip profiling trace files
2. **Load into HTA**: Initialize HTA with trace directory
3. **Generate Report**: Run analysis and get insights
4. **Visualize**: Create charts and timelines
5. **Compare**: Analyze differences between methods

### Output
- Performance summary reports
- Visualization charts (GPU utilization, kernel breakdown)
- Recommendations for optimization
- Comparison tables between methods

---

## Common Patterns Across Notebooks

### 1. WandB Integration
All notebooks use WandB for experiment tracking:
- Automatic experiment naming
- Loss and accuracy logging
- Memory usage tracking
- System metrics (GPU utilization, temperature)

### 2. Error Handling
- Graceful degradation if WandB unavailable
- Distributed process cleanup
- Resource cleanup on errors

### 3. Reproducibility
- Fixed random seeds
- Deterministic data loading
- Consistent model initialization

### 4. Memory Management
- Gradient checkpointing enabled
- Mixed precision training (BF16)
- Efficient data loading with pin_memory

---

## Getting Started

### Prerequisites
1. Install dependencies: `pip install -r requirements.txt`
2. Set up WandB: `wandb login` or set `WANDB_API_KEY`
3. Ensure multi-GPU access (for multi-GPU notebook)

### Recommended Order
1. Start with **single-GPU notebook** to understand basics
2. Move to **multi-GPU notebook** for distributed training
3. Use **trace analysis notebook** to optimize performance

### Tips
- Run single-GPU experiments first to verify setup
- Start with small number of steps for profiling
- Check GPU memory usage before scaling up
- Use trace analysis to identify bottlenecks
- Compare WandB metrics across experiments

---

## Troubleshooting

### Single-GPU Notebook
- **OOM Errors**: Reduce batch size or enable more aggressive gradient checkpointing
- **WandB Issues**: Check API key and network connectivity

### Multi-GPU Notebook
- **NCCL Errors**: Ensure GPUs are visible and NCCL is installed
- **Hanging**: Check that all processes are running (notebook limitations)
- **Trace Merging**: Ensure all ranks complete before merging

### Trace Analysis
- **Missing Traces**: Verify profiling was enabled and traces were saved
- **HTA Errors**: Check trace file format and HTA installation

---

For detailed API documentation, see `DeepSpeed.API.md`.
For overall project information, see `README.md`.

